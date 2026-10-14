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


def compare_at(reference, target, base, candidate, t, settings, edge=False):
    old = measure_window(reference, target, base, t, settings) if edge else measure(reference, target, base, t, settings)
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
        same_method = old["method"] == new["method"] if edge else old.method == new.method
        regression = abs(ae) > abs(be) + tolerance or (same_method and ass < bs - .025)
        improvement = abs(ae) + settings.min_improvement < abs(be)
        row["status"] = "REGRESSION" if regression else "IMPROVED" if improvement else "UNCHANGED"
    return row


def gate_micro(reference, target, base, measurements, settings, log=print):
    enabled = {m.target_time for m in measurements if m.status == "MEASURED"}
    times = holdout_times(base, settings)
    normal_holdouts = set(grid(base, settings, phase=.5))
    final_rows = []
    audit = []
    # Quarantine only affected anchors, rebuilding from the unchanged coarse map.
    for attempt in range(4):
        try:
            candidate = refined_map(base, measurements, settings, enabled)
        except ValueError as exc:
            audit.append({"attempt": attempt + 1, "status": "INVALID_MAP", "reason": str(exc)})
            return base, final_rows, audit
        if same_map(base, candidate):
            return base, final_rows, audit
        rows = []
        for t in times:
            edge = t not in normal_holdouts
            row = compare_at(reference, target, base, candidate, t, settings, edge)
            rows.append(row)
        bad = [r for r in rows if abs(r["change_sec"]) >= settings.min_improvement
               and r["status"] in {"REGRESSION", "UNVERIFIABLE"}]
        audit.extend([{**r, "attempt": attempt + 1} for r in rows])
        if not bad:
            improvements = sum(r["status"] == "IMPROVED" for r in rows)
            log(f"  Независимая проверка: улучшено {improvements}, ухудшений 0.")
            if improvements:
                return candidate, rows, audit
            return base, rows, audit
        removed = set()
        for r in bad:
            distances = sorted((abs(t - r["target_time"]), t) for t in enabled)
            removed.update(t for _, t in distances[:2])
        if not removed:
            break
        enabled -= removed
        final_rows = rows
        log(f"  Защита: отменены {len(removed)} соседних микропоправок; повторная проверка.")
    # A bounded unresolved correction never escapes as a synced file.
    return base, final_rows, audit


def verify_render(reference, original_target, rendered, base, final, settings):
    """Compare decoded output against reference at the SAME reference timeline.

    Include holdouts and anchors. Virtual base uses the identical interpolation
    kernel. Final decoded residuals, not predicted map errors, drive acceptance.
    """
    from .model import TimeMap
    output_map = TimeMap.affine(len(rendered) / settings.analysis_rate)
    target_times = sorted(set(holdout_times(final, settings) + grid(final, settings)))
    rows = []
    for t in target_times:
        center = float(final.forward([t])[0])
        if not np.isfinite(center):
            continue
        actual = measure_window(reference, rendered, output_map, center, settings)
        baseline = measure(reference, original_target, base, t, settings)
        def valid(m):
            return bool(m and not m["near_edge"] and m["score"] >= settings.min_raw_score
                        and m["margin"] >= settings.min_raw_margin)
        changed = abs(float(final.forward([t])[0] - base.forward([t])[0])) >= settings.min_improvement
        row = {"target_time": t, "reference_time": center, "micro_changed": changed,
               "residual_sec": actual["correction_sec"] if valid(actual) else None,
               "score": actual["score"] if actual else None, "baseline_residual_sec": baseline.correction_sec if baseline.status == "MEASURED" else None,
               "status": "UNVERIFIABLE"}
        if valid(actual):
            row["status"] = "VERIFIED"
            if (changed and (baseline.status != "MEASURED" or abs(actual["correction_sec"]) > abs(baseline.correction_sec) + 1 / settings.analysis_rate
                             or (actual["method"] == baseline.method and actual["score"] < baseline.score - .025))):
                row["status"] = "REGRESSION"
            # A correct continuous map should have no remaining gross PCM lag.
            elif abs(actual["correction_sec"]) > .002:
                row["status"] = "RESIDUAL_TOO_LARGE"
        elif baseline.status == "MEASURED":
            # Losing a previously measurable common waveform is a regression,
            # including a lag outside the ±25 ms search radius or corrupted PCM.
            row["status"] = "REGRESSION"
        rows.append(row)
    return rows


def render_verdict(rows, require_micro=True):
    verified = [r for r in rows if r["status"] == "VERIFIED"]
    bad = [r for r in rows if r["status"] in {"REGRESSION", "RESIDUAL_TOO_LARGE"}
           or (r["micro_changed"] and r["status"] == "UNVERIFIABLE")]
    # For different language mixes, coarse held-out validation remains useful,
    # but unknown raw residuals are explicitly not sample-level verification.
    sufficient = len(verified) >= 3 and len(verified) / max(1, len(rows)) >= .60
    if bad:
        return "REJECTED"
    if sufficient:
        return "VERIFIED_PCM"
    return "UNVERIFIED_PCM" if require_micro else "VERIFIED_ENVELOPE_ONLY"
