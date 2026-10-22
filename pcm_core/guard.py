"""Independent windows + actual output: fail closed instead of merely logging a regression."""
import hashlib
import struct
import numpy as np
from .micro import grid, measure_window, refined_map


def same_map(a, b, tolerance=1e-10):
    if hasattr(a,'layers') or hasattr(b,'layers'):
        return a.to_dict()==b.to_dict()
    if len(a.pieces) != len(b.pieces):
        return False
    for pa, pb in zip(a.pieces, b.pieces):
        knots = np.unique(np.r_[pa.target, pb.target])
        spans = np.diff(knots)
        points = np.unique(np.r_[knots, knots[:-1]+spans*.25,
                                 knots[:-1]+spans*.5, knots[:-1]+spans*.75])
        if np.max(np.abs(a.forward(points) - b.forward(points))) > tolerance:
            return False
    return True


def holdout_times(mapping, settings):
    values = grid(mapping, settings, phase=.5)
    # These edge probes use a single short window, never the training subwindows.
    for p in mapping.pieces:
        left = max(p.target[0] + .20,
                   float(np.interp(max(p.reference[0], 0.) + .20, p.reference, p.target)))
        values += [left, p.target[-1] - .20]
    return sorted(set(float(t) for t in values))


def _signature(mapping, t, settings):
    """Exact local polynomial signature for reusing unchanged PCM probes."""
    span = (min(.45, settings.anchor_step / 6) + settings.micro_window / 2
            + settings.micro_radius * 2 + .05) / (1 - settings.max_rate_change)
    parts = []
    for p in mapping.pieces:
        if t + span < p.target[0] or t - span > p.target[-1]:
            continue
        lo = max(0, np.searchsorted(p.target, t - span) - 1)
        hi = min(len(p.target), np.searchsorted(p.target, t + span) + 1)
        parts += [p.target[lo:hi].tobytes(), p._base_reference[lo:hi].tobytes()]
        for x, spline in p._curves:
            a = max(0, np.searchsorted(x, t - span) - 1)
            b = min(len(x) - 1, np.searchsorted(x, t + span))
            parts += [x[a:b + 1].tobytes(), spline.c[:, a:b].tobytes()]
    # Keep a fixed-size exact-content fingerprint instead of retaining copies
    # of all polynomial bytes for tens of thousands of cached PCM windows.
    digest = hashlib.sha256()
    if hasattr(mapping,'layers'):
        import json
        center=float(mapping.forward([t])[0])
        local=mapping.local_layers([center-span,center+span])
        digest.update(json.dumps(local,sort_keys=True).encode())
        if hasattr(mapping.base,'layers'):
            digest.update(_signature(mapping.base,t,settings))
    for part in parts:
        digest.update(struct.pack('<Q', len(part)))
        digest.update(part)
    return digest.digest()


def _change_extent(base, candidate, t, settings, edge=False):
    """A zero center correction can still modify its PCM comparison window."""
    length = min(settings.micro_window, settings.anchor_step / 8)
    span = length / 2 + (0. if edge else min(.45, settings.anchor_step / 6))
    span += settings.sinc_radius / settings.analysis_rate
    probes = np.linspace(t - span, t + span, 9)
    if hasattr(candidate,'extent'):
        return candidate.extent(base,candidate.base.forward(probes))
    a, b = base.forward(probes), candidate.forward(probes)
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.


def gate_micro(reference, target, base, measurements, settings, log=print, progress=None, probe_times=None, strong_fast=False,
               prior_schedules=(), map_builder=None, classic_rollback=False):
    """Gate training and independent centers with the final renderer's PCM rules.

    Measurements still require neighbour/reciprocal consensus. Independent
    validation uses the exact central PCM before/after, including uniqueness
    and renderer fidelity; it does not repeatedly fit the source clock again.
    """
    enabled = {m.target_time for m in measurements if m.status == "MEASURED"}
    times = sorted(set(holdout_times(base, settings) + grid(base, settings))) if probe_times is None else sorted(set(probe_times))
    training = {round(m.target_time, 9) for m in measurements if m.method != 'analysis-boundary'}
    boundaries = {round(m.target_time, 9) for m in measurements if m.method == 'analysis-boundary'}
    normal_holdouts = ({round(t, 9) for t in grid(base, settings, phase=.5)} if probe_times is None
                       else {round(t, 9) for t in times} - training - boundaries)
    audit = []
    pcm_cache, before_cache = {}, {}
    prior_schedules = [(schedule, sorted(set(prior_times)) if prior_times is not None else
                        sorted(set(holdout_times(base, schedule)+grid(base, schedule))))
                       for schedule, prior_times in prior_schedules]
    total_checks = len(times)+sum(len(values) for _, values in prior_schedules)
    if progress is not None and hasattr(progress, 'target_times'):
        progress.target_times = times+[t for _, values in prior_schedules for t in values]

    def check(candidate, attempt):
        if progress is not None and hasattr(progress, 'begin_pass'):
            progress.begin_pass(attempt, 11)
        pcm = verify_candidate_pcm(reference, target, base, candidate, settings,
                                   progress=(lambda done, total: progress(done, total_checks)) if progress else None,
                                   target_times=times,
                                   row_cache=pcm_cache, baseline_cache=before_cache)
        rows = []
        for r in pcm:
            t = r['target_time']
            extent = r['change_extent_sec']
            old, new = r['baseline_residual_sec'], r['residual_sec']
            status = 'UNVERIFIABLE' if r['status'] == 'UNVERIFIABLE' else 'REGRESSION' if r['status'] == 'REGRESSION' else 'UNCHANGED'
            if r['status'] == 'VERIFIED' and old is not None and new is not None:
                if extent >= settings.min_improvement and abs(new) + settings.min_improvement < abs(old):
                    status = 'IMPROVED'
            if strong_fast and extent >= settings.min_improvement:
                p = next((p for p in candidate.pieces if p.target[0] <= t <= p.target[-1]), None)
                fast = p is not None and abs(float(p.rate_at(np.array([t]))[0]) - 1) > .02
                if fast and ((r['expected_score'] or 0.) < .85 or (r['expected_margin'] or 0.) < .12):
                    status = 'UNVERIFIABLE'
                if (r['baseline_score'] is not None and r['expected_score'] is not None
                        and r['expected_score'] < r['baseline_score'] - .05):
                    status = 'REGRESSION'
            if extent < settings.min_improvement:
                status = 'UNCHANGED'
            rows.append({'target_time': t, 'change_sec': float(candidate.forward([t])[0] - base.forward([t])[0]),
                         'before_residual_sec': old, 'after_residual_sec': new,
                         'before_score': r['baseline_score'], 'after_score': r['expected_score'],
                         'status': status, 'change_extent_sec': extent,
                         'probe_kind': 'anchor' if round(t, 9) in training else 'holdout' if round(t, 9) in normal_holdouts else 'edge'})
        # A zoom must also preserve the previous candidate at its ORIGINAL
        # centers and window sizes, not merely at a different dense grid.
        completed = len(times)
        for schedule, prior_times in prior_schedules:
            previous = verify_candidate_pcm(reference, target, base, candidate, schedule,
                target_times=prior_times, row_cache=pcm_cache, baseline_cache=before_cache,
                progress=(lambda done, total, offset=completed: progress(offset+done, total_checks)) if progress else None)
            completed += len(prior_times)
            for r in previous:
                if r['micro_changed'] and r['status'] in {'REGRESSION', 'UNVERIFIABLE'}:
                    rows.append({'target_time': r['target_time'], 'change_sec': 0.,
                                 'before_residual_sec': r['baseline_residual_sec'],
                                 'after_residual_sec': r['residual_sec'],
                                 'before_score': r['baseline_score'], 'after_score': r['expected_score'],
                                 'status': r['status'], 'change_extent_sec': r['change_extent_sec'],
                                 'probe_kind': 'prior_grid'})
        bad = [r for r in rows if r['change_extent_sec'] >= settings.min_improvement
               and r['status'] in {'REGRESSION', 'UNVERIFIABLE'}]
        audit.append({'status': 'GATE_ATTEMPT', 'attempt': attempt,
                      'improved': sum(r['status'] == 'IMPROVED' for r in rows),
                      'unchanged': sum(r['status'] == 'UNCHANGED' for r in rows), 'failed': len(bad)})
        audit.extend([{**r, 'attempt': attempt} for r in bad])
        log(f"  Проход {attempt}/11: улучшено {sum(r['status'] == 'IMPROVED' for r in rows)}, спорных окон {len(bad)}.")
        return rows, bad

    rows = []
    for attempt in range(1, 9):
        try:
            candidate = (map_builder or refined_map)(base, measurements, settings, enabled, audit)
        except ValueError as exc:
            audit.append({'attempt': attempt, 'status': 'INVALID_MAP', 'reason': str(exc)})
            return base, rows, audit
        if same_map(base, candidate):
            return base, rows, audit
        rows, bad = check(candidate, attempt)
        if not bad:
            return (candidate if any(r['status'] == 'IMPROVED' for r in rows) else base), rows, audit
        if attempt == 8:
            break
        removed = set()
        ordered = np.array(sorted(enabled))
        for r in bad:
            i = int(np.searchsorted(ordered, r['target_time']))
            neighbours = ordered[max(0, i-2):i+2]
            closest = sorted((abs(t-r['target_time']), t) for t in neighbours)
            removed.update(t for _, t in closest[:2])
        if not removed:
            break
        enabled -= removed

    # Keep already verified independent curves after the bounded gate. Each
    # alternative starts from the same candidate; a failed mask cannot spread
    # its defects by repeatedly masking neighbouring good regions.
    original = candidate
    failed_times = [r['target_time'] for r in bad]
    for index, (rate, whole) in enumerate(((settings.max_rate_change, False), (.001, False), (.001, True)), 9):
        try:
            reduced, removed = cancel_unsafe_components(base, original, failed_times,
                settings.max_rate_change, whole_components=whole, rollback_rate=rate,
                taper_scale=2. if classic_rollback else 1.,combine_masks=not classic_rollback)
        except ValueError as exc:
            audit.append({'status': 'GATE_LOCAL_CANCEL_INVALID', 'attempt': index, 'reason': str(exc)})
            continue
        audit.append({'status': 'GATE_LOCAL_CANCEL', 'attempt': index, 'removed_components': removed})
        if same_map(base, reduced):
            continue
        rows, remaining = check(reduced, index)
        if not remaining and any(r['status'] == 'IMPROVED' for r in rows):
            return reduced, rows, audit
    return base, rows, audit


def verify_render(reference, original_target, rendered, base, final, settings, progress=None, target_times=None,
                  row_cache=None, baseline_cache=None):
    """Compare decoded output against reference at the SAME reference timeline.

    Include holdouts and anchors. Virtual base uses the identical interpolation
    kernel. Final decoded residuals, not predicted map errors, drive acceptance.
    """
    from .model import TimeMap
    virtual = rendered is None
    output_map = None if virtual else TimeMap.affine(len(rendered) / settings.analysis_rate)
    row_cache = {} if row_cache is None else row_cache
    baseline_cache = {} if baseline_cache is None else baseline_cache
    target_times = sorted(set(holdout_times(base, settings) + grid(base, settings))) if target_times is None else sorted(set(target_times))
    rows = []
    for index, t in enumerate(target_times):
        cache_key = (settings, t, _signature(final, t, settings)) if virtual else None
        if virtual and cache_key in row_cache:
            rows.append(row_cache[cache_key])
            if progress:
                progress(index + 1, len(target_times))
            continue
        center = float(final.forward([t])[0])
        if not np.isfinite(center):
            continue
        key = (settings, t)
        if key not in baseline_cache:
            baseline_cache[key] = measure_window(reference, original_target, base, t, settings,
                                                 quantize_pcm16=settings.verification_pcm16)
        baseline_center = baseline_cache[key]
        expected = measure_window(reference, original_target, final, t, settings,
                                  quantize_pcm16=settings.verification_pcm16)
        actual = expected if virtual else measure_window(reference, rendered, output_map, center, settings)
        def valid(m):
            return bool(m and not m["near_edge"] and m["score"] >= settings.min_raw_score
                        and m["margin"] >= settings.min_raw_margin)
        context = settings
        verification_length = min(settings.micro_window, settings.anchor_step / 8)
        if (not valid(actual) or not valid(baseline_center)) and settings.anchor_step < .5:
            from dataclasses import replace
            wide = replace(settings, anchor_step=.5)
            wide_before = measure_window(reference, original_target, base, t, wide,
                                         quantize_pcm16=settings.verification_pcm16)
            wide_expected = measure_window(reference, original_target, final, t, wide,
                                           quantize_pcm16=settings.verification_pcm16)
            wide_actual = wide_expected if virtual else measure_window(reference, rendered, output_map, center, wide)
            if valid(wide_actual) and valid(wide_before) and valid(wide_expected):
                actual, baseline_center, expected = wide_actual, wide_before, wide_expected
                context = wide
                verification_length = min(settings.micro_window, wide.anchor_step / 8)
        extent = _change_extent(base, final, t, context, edge=True)
        changed = extent >= settings.min_improvement
        # Compare the same central PCM measurement before/after. Neighbour
        # consensus can fail at a splice even when its center is measurable;
        # it must not masquerade as loss of that center in the decoded file.
        before_error = baseline_center['correction_sec'] if valid(baseline_center) else None
        row = {"target_time": t, "reference_time": center, "micro_changed": changed,
               "change_extent_sec": extent,
               "residual_sec": actual["correction_sec"] if valid(actual) else None,
               "score": actual["score"] if actual else None, "baseline_residual_sec": before_error,
               "expected_score": expected['score'] if expected else None,
               "expected_margin": expected['margin'] if expected else None,
               "baseline_score": baseline_center['score'] if baseline_center else None,
               "verification_window_sec": verification_length,
               "baseline_center_measurable": valid(baseline_center),
               "status": "UNVERIFIABLE"}
        if valid(actual):
            row["status"] = "VERIFIED"
            previously_measured = before_error is not None
            worsened = previously_measured and (
                abs(actual['correction_sec']) > abs(before_error) + 1 / settings.analysis_rate
            )
            # Compare renderer fidelity to the predicted corrected PCM, not
            # to the waveform of the intentionally uncorrected source.
            fidelity_lost = (not valid(expected) or (actual['method'] == expected['method']
                             and actual['score'] < expected['score'] - .025))
            if (changed and not previously_measured) or worsened or fidelity_lost:
                row["status"] = "REGRESSION"
        elif valid(baseline_center):
            # Losing a previously measurable common waveform is a regression,
            # including a lag outside the ±25 ms search radius or corrupted PCM.
            row["status"] = "REGRESSION"
        rows.append(row)
        if virtual:
            row_cache[cache_key] = row
        if progress:
            progress(index + 1, len(target_times))
    return rows


def verify_candidate_pcm(reference, original_target, base, final, settings, **kwargs):
    """Same final acceptance rules, same sinc PCM, before multichannel rendering."""
    return verify_render(reference, original_target, None, base, final, settings, **kwargs)


def render_verdict(rows, require_micro=True):
    changed = [r for r in rows if r["micro_changed"]]
    verified = [r for r in changed if r["status"] == "VERIFIED"]
    bad = [r for r in rows if r["status"] in {"REGRESSION", "RESIDUAL_TOO_LARGE"}
           or (r["micro_changed"] and r["status"] == "UNVERIFIABLE")]
    # For different language mixes, coarse held-out validation remains useful,
    # but unknown raw residuals are explicitly not sample-level verification.
    sufficient = len(verified) >= 3 and len(verified) / max(1, len(changed)) >= .60
    improved = [r for r in verified if r['baseline_residual_sec'] is not None
                and abs(r['residual_sec']) + .25 / 48000 < abs(r['baseline_residual_sec'])]
    sufficient = sufficient and len(improved) >= 3
    if bad:
        return "REJECTED"
    if sufficient:
        return "VERIFIED_PCM"
    return "UNVERIFIED_PCM" if require_micro else "VERIFIED_ENVELOPE_ONLY"


def cancel_unsafe_components(base, candidate, bad_times, max_rate_change=.02, whole_components=False,
                             rollback_rate=.001, taper_scale=1., combine_masks=True, protected_half_window=.25):
    """Return unsafe PCM neighbourhoods to the baseline, retaining distant curves.

    Add knots only near a failed window, with a smooth return to zero. Removing
    a whole clock curve would throw away a movie's improvements for one splice.
    All modified boundaries must subsequently pass decoded-output verification.
    """
    from .model import Piece, TimeMap, curve_spline
    from scipy.interpolate import PchipInterpolator

    if hasattr(candidate,'reduced'):
        reference_times=candidate.base.forward(np.asarray(bad_times))
        return candidate.reduced(reference_times,protected_half_window,protected_base=base)

    def active_intervals(y, derivative=None):
        active = (np.abs(y[:-1]) > 1e-12) | (np.abs(y[1:]) > 1e-12)
        if derivative is not None:
            derivative = np.asarray(derivative)
            active |= (np.abs(derivative[:-1]) > 1e-12) | (np.abs(derivative[1:]) > 1e-12)
        return active

    def rebuild(bp, p, curves):
        # A zoom is added to an already verified map. Cancel ONLY its extra
        # curves; the verified base must survive even inside a rejected zoom.
        knots = np.unique(np.r_[bp.target, p.target, *[c['target'] for c in curves]])
        extra = np.zeros(knots.shape)
        for curve in curves:
            x, y = np.asarray(curve['target']), np.asarray(curve['delta'])
            valid = (knots >= x[0]) & (knots <= x[-1])
            extra[valid] += curve_spline(curve)(knots[valid])
        return Piece(knots, bp.reference_at(knots) + extra, [*bp.corrections, *curves])

    output = []
    removed = []
    for bp, p in zip(base.pieces, candidate.pieces):
        retained = []
        for curve in p.corrections[len(bp.corrections):]:
            x = np.asarray(curve['target'], float)
            y = np.asarray(curve['delta'], float).copy()
            dy = np.asarray(curve['derivative'], float).copy() if 'derivative' in curve else None
            active = active_intervals(y,dy)
            i = 0
            while i < len(active):
                if not active[i]:
                    i += 1
                    continue
                a = i
                while i + 1 < len(active) and active[i + 1]:
                    i += 1
                b = i + 1
                times = [t for t in bad_times if x[a] - protected_half_window <= t <= x[b] + protected_half_window]
                if times:
                    # The caller protects its complete PCM window and interpolation
                    # halo. The taper must also respect the permitted rate.
                    # A rate that is legal for an edit can still destroy PCM
                    # coherence over a 125/220 ms window. Undo a local offset
                    # at clock-drift speed, not at the maximum edit speed.
                    taper = max(.15, taper_scale * float(np.max(np.abs(y[a:b + 1]))) /
                                min(max_rate_change, rollback_rate))
                    additions = []
                    for t in times:
                        lo, hi = max(x[a], t - protected_half_window - taper), min(x[b], t + protected_half_window + taper)
                        additions.extend(np.linspace(lo, hi, max(3, int((hi-lo)/.04)+2)))
                        additions += [max(x[a], t-protected_half_window), min(x[b], t+protected_half_window)]
                        removed.append({'start': float(lo), 'end': float(hi), 'failed_time': float(t)})
                    old_x, old_y = x, y
                    old_curve = {'target': x, 'delta': y}
                    if dy is not None:
                        old_curve['derivative'] = dy
                    old_spline = curve_spline(old_curve)
                    x = np.unique(np.r_[x, additions])
                    y = old_spline(x)
                    if dy is not None:
                        dy = old_spline(x, nu=1)
                    combined = np.ones(len(x))
                    combined_derivative = np.zeros(len(x))
                    local = (x >= old_x[a]) & (x <= old_x[b])
                    for t in times:
                        distance = np.maximum(0., np.abs(x-t) - protected_half_window)
                        weight = .5 - .5 * np.cos(np.pi * np.minimum(1., distance / taper))
                        # Overlapping failures describe ONE unsafe region.
                        # Multiplying their masks attenuates good audio farther
                        # away merely because the same defect had more probes.
                        angle = np.pi*np.minimum(1., distance/taper)
                        dw = .5*np.sin(angle)*np.pi/taper*np.sign(x-t)
                        dw[distance >= taper] = 0.
                        if combine_masks:
                            selected = weight < combined
                            combined[selected] = weight[selected]
                            combined_derivative[selected] = dw[selected]
                        else:
                            combined_derivative = combined_derivative*weight+combined*dw
                            combined *= weight
                    if dy is not None:
                        dy[local] = dy[local]*combined[local]+y[local]*combined_derivative[local]
                    y[local] *= combined[local]
                    # Recompute components against the enlarged knot array.
                    active = active_intervals(y,dy)
                    i = int(np.searchsorted(x, old_x[b])) - 1
                i += 1
            if active.any():
                retained.append({'target': x.tolist(), 'delta': y.tolist(),
                                 **({'derivative': dy.tolist()} if dy is not None else {})})
        output.append(rebuild(bp, p, retained))
    result = TimeMap(output, list(base.notes))
    try:
        if whole_components:
            raise ValueError('Отмена неподтверждённого участка целиком.')
        return result.validate(max_rate_change), removed
    except ValueError:
        # Multiplying an existing near-limit clock curve by a smooth mask can
        # still exceed the rate limit. Remove complete connected components
        # instead. Zero plateaus separate their PCHIP derivatives exactly.
        output = []
        for bp, p in zip(base.pieces, candidate.pieces):
            components = []
            residuals = p.corrections[len(bp.corrections):]
            for curve_index, curve in enumerate(residuals):
                x, y = np.asarray(curve['target']), np.asarray(curve['delta'])
                active = active_intervals(y,curve.get('derivative'))
                starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
                ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
                components += [(curve_index, int(a), int(b), float(x[a]), float(x[b]))
                               for a, b in zip(starts, ends)]
            blocked = {i for i, c in enumerate(components)
                       if any(c[3]-protected_half_window <= t <= c[4]+protected_half_window for t in bad_times)}
            # A later residual curve depends on earlier corrections: cancel
            # all overlapping components together, rather than leave it alone.
            while True:
                extra = {i for i, c in enumerate(components) if i not in blocked
                         and any(c[3] < components[j][4] and c[4] > components[j][3] for j in blocked)}
                if not extra:
                    break
                blocked |= extra
            curves = [{key: list(value) for key, value in c.items()} for c in residuals]
            for i in blocked:
                ci, a, b, lo, hi = components[i]
                curves[ci]['delta'][a:b+1] = [0.]*(b-a+1)
                if 'derivative' in curves[ci]:
                    curves[ci]['derivative'][a:b+1] = [0.]*(b-a+1)
                removed.append({'start': lo, 'end': hi, 'mode': 'whole_component_rate_fallback'})
            curves = [c for c in curves if active_intervals(np.asarray(c['delta']),c.get('derivative')).any()]
            output.append(rebuild(bp, p, curves))
        return TimeMap(output, list(base.notes)).validate(max_rate_change), removed
