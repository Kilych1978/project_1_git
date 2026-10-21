"""Established residual-PCHIP proposal, computed afresh from current PCM.

This is a numerical map builder, not a loader of previous runs or folders.
It provides the comparison candidate for the source-clock proposal.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from .model import Piece,TimeMap

def refined_map(base, measurements, settings, enabled_times=None, audit=None):
    """Only consecutive supported anchors form a region; rejected points are barriers.

    Anchor times are inserted exactly (v7's 1-second map grid could miss an anchor).
    No convolution/averaging erases an accepted residual estimate.
    """
    output = []
    for piece in base.pieces:
        rows = [m for m in measurements if piece.target[0] <= m.target_time <= piece.target[-1]]
        # An ambiguous short window is not proof of a clock discontinuity.
        # Bridge one missing center only when TWO measured centers on each
        # side independently predict the same clock. All bridged PCM centers
        # still undergo the ordinary independent/final-output gates.
        bridged = set()
        if len(rows) >= 5:
            from dataclasses import replace
            original_rows = list(rows)
            for i in range(2, len(rows)-2):
                m = original_rows[i]
                neighbours = [original_rows[j] for j in (i-2, i-1, i+1, i+2)]
                if (m.status != 'UNVERIFIABLE' or m.method == 'analysis-boundary'
                        or any(n.status != 'MEASURED' or n.correction_sec is None
                               or n.score < .75 or (enabled_times is not None and n.target_time not in enabled_times)
                               for n in neighbours)):
                    continue
                tx = np.array([n.target_time for n in neighbours])
                vy = np.array([n.correction_sec for n in neighbours])
                if np.max(np.diff(np.r_[tx[:2], m.target_time, tx[2:]])) > settings.anchor_step * 1.01:
                    continue
                left_slope = (vy[1]-vy[0])/(tx[1]-tx[0])
                right_slope = (vy[3]-vy[2])/(tx[3]-tx[2])
                left = vy[1]+left_slope*(m.target_time-tx[1])
                right = vy[2]+right_slope*(m.target_time-tx[2])
                if (abs(left-right) <= .00005 and abs(left_slope-right_slope) <= .0001
                        and max(abs(left_slope), abs(right_slope)) <= .003):
                    rows[i] = replace(m, correction_sec=float((left+right)/2),
                                      status='MEASURED', method='clock-gap-prediction')
                    bridged.add(m.target_time)
                    if audit is not None:
                        audit.append({'status': 'CLOCK_GAP_PREDICTION', 'target_time': m.target_time})
        nodes = []
        run = []
        def commit():
            if len(run) >= 2:
                # Unit/near-unit rate corrections keep a constant offset through
                # small edge zones, then taper to the unproved base map.
                times = np.array([m.target_time for m in run])
                vals = np.array([m.correction_sec for m in run])
                if np.max(np.abs(vals)) <= settings.min_improvement:
                    return
                # A stable residual clock rate is estimated from all training
                # anchors. Do not bend the time map to sample/codec noise.
                fitted = False
                if len(run) >= 3:
                    weights = np.array([m.score for m in run]) ** 2
                    slope, intercept = np.polyfit(times - times[0], vals, 1, w=weights)
                    fitted_values = slope * (times - times[0]) + intercept
                    if np.max(np.abs(vals - fitted_values)) <= 1. / settings.analysis_rate:
                        vals = fitted_values
                        fitted = True
                left = max(piece.target[0], times[0] - settings.anchor_step / 2)
                right = min(piece.target[-1], times[-1] + settings.anchor_step / 2)
                left_value = right_value = 0.
                # Uniform residual/rate drift may be extended to the edges, but
                # only when every anchor supports it; edge holdouts still gate it.
                if len(run) == len(rows) and len(run) >= 3:
                    slope, intercept = np.polyfit(times, vals, 1)
                    if np.max(np.abs(vals - (slope * times + intercept))) < settings.raw_consensus:
                        left, right = piece.target[0], piece.target[-1]
                        left_value, right_value = slope * left + intercept, slope * right + intercept
                # Validate each supported run independently. A bad boundary
                # must never invalidate unrelated improvements in a movie.
                # A smaller, independently checked correction is preferable
                # to returning the entire fine map to zero.
                accepted_gain = None
                cap = np.full(vals.shape, np.inf)
                if left_value == 0.:
                    cap = np.minimum(cap, (times - left) * settings.max_rate_change * .6)
                if right_value == 0.:
                    cap = np.minimum(cap, (right - times) * settings.max_rate_change * .6)
                shaped_values = np.clip(vals, -cap, cap)
                options = [(1., vals, False), (1., shaped_values, True)]
                options += [(gain, vals * gain, False) for gain in (.5, .25, .125, .0625)]
                edge_limited = False
                for gain, trial_values, shaped in options:
                    curve = {'target': [left, *times.tolist(), right],
                             'delta': [left_value * gain, *trial_values.tolist(), right_value * gain]}
                    local_times = np.unique(np.r_[piece.target, curve['target']])
                    interpolation = PchipInterpolator(curve['target'], curve['delta'], extrapolate=False)
                    local_delta = np.nan_to_num(interpolation(local_times), nan=0.)
                    try:
                        local_piece = Piece(local_times, piece.reference_at(local_times) + local_delta,
                                            [*piece.corrections, curve])
                        TimeMap([local_piece]).validate(settings.max_rate_change)
                    except ValueError:
                        continue
                    accepted_gain = gain
                    edge_limited = shaped
                    nodes.extend(zip(curve['target'], curve['delta']))
                    break
                if audit is not None and (accepted_gain != 1. or fitted or edge_limited):
                    audit.append({'status': 'LOCAL_CURVE', 'start': float(times[0]),
                                  'end': float(times[-1]), 'gain': accepted_gain,
                                  'affine_fit': fitted, 'edge_limited': edge_limited, 'anchors': len(run)})
        for m in rows:
            active = (m.status == "MEASURED" and m.correction_sec is not None
                      and (enabled_times is None or m.target_time in enabled_times or m.target_time in bridged))
            gap_ok = not run or m.target_time - run[-1].target_time <= settings.anchor_step * 1.01
            # Separate abrupt changes rather than pretending they are a
            # continuous clock-rate correction. Each side keeps its evidence.
            slope_ok = not active or not run or abs(m.correction_sec - run[-1].correction_sec) <= (
                .75 * settings.max_rate_change * (m.target_time - run[-1].target_time))
            if active and gap_ok and slope_ok:
                run.append(m)
            else:
                commit()
                run = [m] if active else []
        commit()
        if not nodes:
            output.append(Piece(piece.target.copy(), piece.reference.copy(), list(piece.corrections)))
            continue
        unique = {float(t): float(v) for t, v in nodes}
        # Preserve the base's existing nonlinear knots and each fine anchor exactly.
        nt = np.unique(np.r_[piece.target, list(unique)])
        xs = np.array(sorted(unique))
        ys = np.array([unique[x] for x in xs])
        interpolator = PchipInterpolator(xs, ys, extrapolate=False)
        corrections = np.nan_to_num(interpolator(nt), nan=0.)
        curves = [*piece.corrections, {"target": xs.tolist(), "delta": ys.tolist()}]
        new_piece = Piece(nt, piece.reference_at(nt) + corrections, curves)
        # The combined interpolator can change derivatives at neighbouring
        # zero boundaries. If necessary, damp just those knots, not the map.
        for _ in range(12):
            rates = new_piece.rate_extrema()
            if np.max(np.abs(rates - 1)) <= settings.max_rate_change + 1e-10:
                break
            lo = np.nextafter(new_piece.target[:-1], new_piece.target[1:])
            hi = np.nextafter(new_piece.target[1:], new_piece.target[:-1])
            d0 = new_piece.correction_at(lo, derivative=2)
            d1 = new_piece.correction_at(hi, derivative=2)
            crossing = (d0 * d1 < 0) & (np.abs(d1 - d0) > 1e-15)
            roots = lo[crossing] - d0[crossing] * (hi[crossing] - lo[crossing]) / (d1[crossing] - d0[crossing])
            probes = np.r_[lo, hi, roots]
            bad = probes[np.abs(new_piece.rate_at(probes) - 1) > settings.max_rate_change]
            changed = set()
            for t in bad:
                i = np.clip(np.searchsorted(xs, t), 1, len(xs) - 1)
                changed.update([i - 1, i])
            if not changed:
                break
            ys[list(changed)] *= .5
            interpolation = PchipInterpolator(xs, ys, extrapolate=False)
            corrections = np.nan_to_num(interpolation(nt), nan=0.)
            curves = [*piece.corrections, {'target': xs.tolist(), 'delta': ys.tolist()}]
            new_piece = Piece(nt, piece.reference_at(nt) + corrections, curves)
            if audit is not None:
                audit.append({'status': 'LOCAL_RATE_LIMIT', 'knots': [float(xs[i]) for i in sorted(changed)]})
        output.append(new_piece)
    return TimeMap(output, list(base.notes)).validate(settings.max_rate_change)
