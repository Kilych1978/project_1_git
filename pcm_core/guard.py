"""Independent windows + actual output: fail closed instead of merely logging a regression."""
import numpy as np
from .micro import grid, measure, measure_window, refined_map


def same_map(a, b, tolerance=1e-10):
    if len(a.pieces) != len(b.pieces):
        return False
    for pa, pb in zip(a.pieces, b.pieces):
        knots = np.unique(np.r_[pa.target, pb.target])
        points = np.unique(np.r_[knots, (knots[1:] + knots[:-1]) / 2])
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


def compare_at(reference, target, base, candidate, t, settings, edge=False, old=None):
    old = old if old is not None else (measure_window(reference, target, base, t, settings) if edge else measure(reference, target, base, t, settings))
    new = measure_window(reference, target, candidate, t, settings) if edge else measure(reference, target, candidate, t, settings)
    def extract(value):
        if edge:
            if (value and not value["near_edge"] and value["score"] >= settings.min_raw_score
                    and value["margin"] >= settings.min_raw_margin):
                return value["correction_sec"], value["score"]
        elif value.status == "MEASURED":
            return value.correction_sec, value.score
        return None, None
    be, bs = extract(old)
    ae, ass = extract(new)
    change = float(candidate.forward([t])[0] - base.forward([t])[0])
    row = {"target_time": t, "change_sec": change, "before_residual_sec": be,
           "after_residual_sec": ae, "before_score": bs, "after_score": ass, "status": "UNVERIFIABLE"}
    tolerance = 1. / settings.analysis_rate
    if abs(change) < settings.min_improvement:
        row["status"] = "UNCHANGED"
    elif be is not None and ae is not None:
        # Retiming a changing delay necessarily changes the local waveform
        # rate. A smaller peak alone is not a timing regression when both
        # independent estimates remain unambiguous and timing improves.
        regression = abs(ae) > abs(be) + tolerance
        improvement = abs(ae) + settings.min_improvement < abs(be)
        row["status"] = "REGRESSION" if regression else "IMPROVED" if improvement else "UNCHANGED"
    return row


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
    return tuple(parts)


def _change_extent(base, candidate, t, settings, edge=False):
    """A zero center correction can still modify its PCM comparison window."""
    length = min(settings.micro_window, settings.anchor_step / 8)
    span = length / 2 + (0. if edge else min(.45, settings.anchor_step / 6))
    span += settings.sinc_radius / settings.analysis_rate
    probes = np.linspace(t - span, t + span, 9)
    a, b = base.forward(probes), candidate.forward(probes)
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.


def gate_micro(reference, target, base, measurements, settings, log=print, progress=None, probe_times=None, strong_fast=False):
    enabled = {m.target_time for m in measurements if m.status == "MEASURED"}
    # Include training centers as well: the actual renderer checks them too.
    times = sorted(set(holdout_times(base, settings) + grid(base, settings))) if probe_times is None else sorted(set(probe_times))
    if probe_times is None:
        training_times = {round(t, 9) for t in grid(base, settings)}
        normal_holdouts = {round(t, 9) for t in grid(base, settings, phase=.5)}
    else:
        training_times = {round(m.target_time, 9) for m in measurements if m.method != 'analysis-boundary'}
        boundaries = {round(m.target_time, 9) for m in measurements if m.method == 'analysis-boundary'}
        normal_holdouts = {round(t, 9) for t in times} - training_times - boundaries
    final_rows = []
    audit = []
    old_cache, row_cache = {}, {}
    # Quarantine only affected anchors, rebuilding from the unchanged coarse map.
    for attempt in range(8):
        if progress is not None and hasattr(progress, 'begin_pass'):
            progress.begin_pass(attempt+1, 8)
        try:
            candidate = refined_map(base, measurements, settings, enabled, audit)
        except ValueError as exc:
            audit.append({"attempt": attempt + 1, "status": "INVALID_MAP", "reason": str(exc)})
            return base, final_rows, audit
        if same_map(base, candidate):
            return base, final_rows, audit
        rows = []
        for index, t in enumerate(times):
            key_time = round(t, 9)
            edge = key_time not in normal_holdouts and key_time not in training_times
            signature = (t, _signature(candidate, t, settings))
            if signature in row_cache:
                row = row_cache[signature]
            else:
                change = float(candidate.forward([t])[0] - base.forward([t])[0])
                extent = _change_extent(base, candidate, t, settings, edge)
                if extent < settings.min_improvement:
                    row = {'target_time': t, 'change_sec': change,
                           'before_residual_sec': None, 'after_residual_sec': None,
                           'before_score': None, 'after_score': None, 'status': 'UNCHANGED'}
                else:
                    if t not in old_cache:
                        old_cache[t] = measure_window(reference, target, base, t, settings) if edge else measure(reference, target, base, t, settings)
                    row = compare_at(reference, target, base, candidate, t, settings, edge, old_cache[t])
                    # The recorded output must have a measurable center, even
                    # if its neighbours supplied a virtual consensus.
                    central = measure_window(reference, target, candidate, t, settings)
                    if not (central and not central['near_edge'] and central['score'] >= settings.min_raw_score
                            and central['margin'] >= settings.min_raw_margin):
                        row = {**row, 'status': 'UNVERIFIABLE'}
                if strong_fast and row['status'] != 'UNVERIFIABLE' and extent >= settings.min_improvement:
                    central = measure_window(reference, target, candidate, t, settings)
                    p = next((p for p in candidate.pieces if p.target[0] <= t <= p.target[-1]), None)
                    fast = p is not None and abs(float(p.rate_at(np.array([t]))[0]) - 1) > .02
                    if fast and (not central or central['score'] < .85 or central['margin'] < .12):
                        row = {**row, 'status': 'UNVERIFIABLE'}
                    if (row['before_score'] is not None and row['after_score'] is not None
                            and row['after_score'] < row['before_score'] - .05):
                        row = {**row, 'status': 'REGRESSION'}
                row = {**row, 'change_extent_sec': extent}
                row['probe_kind'] = 'edge' if edge else 'anchor' if key_time in training_times else 'holdout'
                row_cache[signature] = row
            rows.append(row)
            if progress:
                progress(index + 1, len(times))
        bad = [r for r in rows if r.get('change_extent_sec', abs(r['change_sec'])) >= settings.min_improvement
               and r["status"] in {"REGRESSION", "UNVERIFIABLE"}]
        audit.append({'status': 'GATE_ATTEMPT', 'attempt': attempt + 1,
                      'improved': sum(r['status'] == 'IMPROVED' for r in rows),
                      'unchanged': sum(r['status'] == 'UNCHANGED' for r in rows),
                      'failed': len(bad)})
        audit.extend([{**r, 'attempt': attempt + 1} for r in bad])
        if not bad:
            improvements = sum(r["status"] == "IMPROVED" for r in rows)
            log(f"  Независимая проверка: улучшено {improvements}, ухудшений 0.")
            if improvements:
                return candidate, rows, audit
            return base, rows, audit
        removed = set()
        ordered_enabled = np.array(sorted(enabled))
        for r in bad:
            i = int(np.searchsorted(ordered_enabled, r['target_time']))
            neighbours = ordered_enabled[max(0, i-2):i+2]
            distances = sorted((abs(t - r["target_time"]), t) for t in neighbours)
            removed.update(t for _, t in distances[:2])
        if not removed:
            break
        enabled -= removed
        final_rows = rows
        log(f"  Защита: отменены {len(removed)} соседних микропоправок; повторная проверка.")
    # A bounded unresolved correction never escapes as a synced file.
    # Clear only unresolved changed regions after the bounded iterations.
    # Retain unrelated curves, then verify the rebuilt map once more.
    active = sorted(enabled)
    runs = []
    for t in active:
        if not runs or t - runs[-1][-1] > settings.anchor_step * 1.01:
            runs.append([t])
        else:
            runs[-1].append(t)
    blocked_runs = set()
    for r in final_rows:
        if r['status'] not in {'REGRESSION', 'UNVERIFIABLE'}:
            continue
        for run in runs:
            if run[0] - settings.anchor_step <= r['target_time'] <= run[-1] + settings.anchor_step:
                blocked_runs.update(run)
    if blocked_runs:
        enabled -= blocked_runs
        candidate = refined_map(base, measurements, settings, enabled, audit)
        rows = []
        for t in times:
            edge = round(t, 9) not in normal_holdouts and round(t, 9) not in training_times
            if _change_extent(base, candidate, t, settings, edge) < settings.min_improvement:
                continue
            row = compare_at(reference, target, base, candidate, t, settings, edge, old_cache.get(t))
            central = measure_window(reference, target, candidate, t, settings)
            if not (central and not central['near_edge'] and central['score'] >= settings.min_raw_score
                    and central['margin'] >= settings.min_raw_margin):
                row = {**row, 'status': 'UNVERIFIABLE'}
            if strong_fast and central:
                p = next((p for p in candidate.pieces if p.target[0] <= t <= p.target[-1]), None)
                fast = p is not None and abs(float(p.rate_at(np.array([t]))[0]) - 1) > .02
                if fast and (central['score'] < .85 or central['margin'] < .12):
                    row = {**row, 'status': 'UNVERIFIABLE'}
                if (row['before_score'] is not None and row['after_score'] is not None
                        and row['after_score'] < row['before_score'] - .05):
                    row = {**row, 'status': 'REGRESSION'}
            rows.append(row)
        if rows and all(r['status'] in {'IMPROVED', 'UNCHANGED'} for r in rows) and any(r['status'] == 'IMPROVED' for r in rows):
            return candidate, rows, audit
    return base, final_rows, audit


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
            baseline_cache[key] = measure_window(reference, original_target, base, t, settings)
        baseline_center = baseline_cache[key]
        expected = measure_window(reference, original_target, final, t, settings)
        actual = expected if virtual else measure_window(reference, rendered, output_map, center, settings)
        def valid(m):
            return bool(m and not m["near_edge"] and m["score"] >= settings.min_raw_score
                        and m["margin"] >= settings.min_raw_margin)
        verification_length = min(settings.micro_window, settings.anchor_step / 8)
        if (not valid(actual) or not valid(baseline_center)) and settings.anchor_step < .5:
            # A 15 ms window can contain an ambiguous fragment of a musical
            # note even after a correct alignment. Require a unique match in
            # a longer waveform, with the SAME context before, predicted and
            # decoded after. Never waive the alternative-peak margin gate.
            from dataclasses import replace
            context = replace(settings, anchor_step=.5)
            wide_before = measure_window(reference, original_target, base, t, context)
            wide_expected = measure_window(reference, original_target, final, t, context)
            wide_actual = wide_expected if virtual else measure_window(reference, rendered, output_map, center, context)
            if valid(wide_actual) and valid(wide_before) and valid(wide_expected):
                actual, baseline_center, expected = wide_actual, wide_before, wide_expected
                verification_length = min(settings.micro_window, context.anchor_step / 8)
        changed = abs(float(final.forward([t])[0] - base.forward([t])[0])) >= settings.min_improvement
        # Compare the same central PCM measurement before/after. Neighbour
        # consensus can fail at a splice even when its center is measurable;
        # it must not masquerade as loss of that center in the decoded file.
        before_error = baseline_center['correction_sec'] if valid(baseline_center) else None
        row = {"target_time": t, "reference_time": center, "micro_changed": changed,
               "residual_sec": actual["correction_sec"] if valid(actual) else None,
               "score": actual["score"] if actual else None, "baseline_residual_sec": before_error,
               "expected_score": expected['score'] if expected else None,
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
                             rollback_rate=.001):
    """Return unsafe PCM neighbourhoods to the baseline, retaining distant curves.

    Add knots only near a failed window, with a smooth return to zero. Removing
    a whole clock curve would throw away a movie's improvements for one splice.
    All modified boundaries must subsequently pass decoded-output verification.
    """
    from .model import Piece, TimeMap
    output = []
    removed = []
    for bp, p in zip(base.pieces, candidate.pieces):
        retained = []
        for curve in p.corrections:
            x = np.asarray(curve['target'], float)
            y = np.asarray(curve['delta'], float).copy()
            active_intervals = (np.abs(y[:-1]) > 1e-12) | (np.abs(y[1:]) > 1e-12)
            i = 0
            while i < len(active_intervals):
                if not active_intervals[i]:
                    i += 1
                    continue
                a = i
                while i + 1 < len(active_intervals) and active_intervals[i + 1]:
                    i += 1
                b = i + 1
                times = [t for t in bad_times if x[a] - .25 <= t <= x[b] + .25]
                if times:
                    # A 250 ms protected half-window covers all PCM contexts and sinc
                    # halo. The taper is long enough for the permitted rate.
                    # A rate that is legal for an edit can still destroy PCM
                    # coherence over a 125/220 ms window. Undo a local offset
                    # at clock-drift speed, not at the maximum edit speed.
                    taper = max(.15, 2 * float(np.max(np.abs(y[a:b + 1]))) /
                                min(max_rate_change, rollback_rate))
                    additions = []
                    for t in times:
                        lo, hi = max(x[a], t - .25 - taper), min(x[b], t + .25 + taper)
                        additions.extend(np.linspace(lo, hi, max(3, int((hi-lo)/.04)+2)))
                        additions += [max(x[a], t-.25), min(x[b], t+.25)]
                        removed.append({'start': float(lo), 'end': float(hi), 'failed_time': float(t)})
                    old_x, old_y = x, y
                    x = np.unique(np.r_[x, additions])
                    from scipy.interpolate import PchipInterpolator
                    y = PchipInterpolator(old_x, old_y)(x)
                    for t in times:
                        distance = np.maximum(0., np.abs(x-t) - .25)
                        weight = .5 - .5 * np.cos(np.pi * np.minimum(1., distance / taper))
                        # A long taper must stop at this component's existing
                        # zero barriers, preserving independent regions exactly.
                        local = (x >= old_x[a]) & (x <= old_x[b])
                        y[local] *= weight[local]
                    # Recompute components against the enlarged knot array.
                    active_intervals = (np.abs(y[:-1]) > 1e-12) | (np.abs(y[1:]) > 1e-12)
                    i = int(np.searchsorted(x, old_x[b])) - 1
                i += 1
            if np.any(np.abs(y) > 1e-12):
                retained.append({'target': x.tolist(), 'delta': y.tolist()})
        knots = np.unique(np.r_[bp.target, p.target])
        # Baseline is the identity of the actual coarse render. This path is
        # used by the post-render refiner, not by the legacy coarse pipeline.
        trial = Piece(knots, bp.reference_at(knots), retained)
        delta = trial.correction_at(knots)
        output.append(Piece(knots, bp.reference_at(knots) + delta, retained))
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
            for curve_index, curve in enumerate(p.corrections):
                x, y = np.asarray(curve['target']), np.asarray(curve['delta'])
                active = (np.abs(y[:-1]) > 1e-12) | (np.abs(y[1:]) > 1e-12)
                starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
                ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
                components += [(curve_index, int(a), int(b), float(x[a]), float(x[b]))
                               for a, b in zip(starts, ends)]
            blocked = {i for i, c in enumerate(components)
                       if any(c[3]-.25 <= t <= c[4]+.25 for t in bad_times)}
            # A later residual curve depends on earlier corrections: cancel
            # all overlapping components together, rather than leave it alone.
            while True:
                extra = {i for i, c in enumerate(components) if i not in blocked
                         and any(c[3] < components[j][4] and c[4] > components[j][3] for j in blocked)}
                if not extra:
                    break
                blocked |= extra
            curves = [{'target': list(c['target']), 'delta': list(c['delta'])} for c in p.corrections]
            for i in blocked:
                ci, a, b, lo, hi = components[i]
                curves[ci]['delta'][a:b+1] = [0.]*(b-a+1)
                removed.append({'start': lo, 'end': hi, 'mode': 'whole_component_rate_fallback'})
            curves = [c for c in curves if np.any(np.abs(c['delta']) > 1e-12)]
            knots = np.unique(np.r_[bp.target, p.target])
            trial = Piece(knots, bp.reference_at(knots), curves)
            output.append(Piece(knots, bp.reference_at(knots) + trial.correction_at(knots), curves))
        return TimeMap(output, list(base.notes)).validate(max_rate_change), removed
