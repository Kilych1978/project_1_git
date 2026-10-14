"""PCM estimates on warped audio, disjoint subwindows, explicit polarity/ambiguity gates."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt
from .dsp import mapped_window, ncc_valid, peak
from .model import Piece, TimeMap


@dataclass
class Measurement:
    target_time: float
    reference_time: float
    correction_sec: float | None
    score: float = 0.
    margin: float = 0.
    spread_sec: float | None = None
    support: int = 0
    polarity: int = 0
    status: str = "UNVERIFIABLE"
    method: str = "none"

    def to_dict(self):
        return asdict(self)


class _InverseMap:
    """Use the exact inverse of a candidate for reciprocal PCM confirmation."""
    def __init__(self, mapping):
        self.mapping = mapping

    def forward(self, times):
        return self.mapping.inverse(times)

    def inverse(self, times):
        return self.mapping.forward(times)

    def inverse_rate(self, times):
        values = np.asarray(times, dtype=float)
        rates = np.full(values.shape, np.nan)
        for p in self.mapping.pieces:
            valid = (values >= p.target[0]) & (values <= p.target[-1])
            rates[valid] = p.rate_at(values[valid])
        return rates


def measure_window(reference, target, mapping, target_time, settings, radius=None, window_sec=None):
    sr = settings.analysis_rate
    center = float(mapping.forward([target_time])[0])
    if not np.isfinite(center):
        return None
    # Keep training/holdout PCM regions disjoint even in --step 0.5/1 modes.
    length = round(min(window_sec or settings.micro_window, settings.anchor_step / 8) * sr)
    warped, start = mapped_window(target, mapping, center, length, sr, settings.sinc_radius)
    if warped is None or np.sqrt(np.mean(warped.astype(np.float64) ** 2)) < 1e-6:
        return None
    search_samples = round((radius or settings.micro_radius) * sr)
    low = start - search_samples
    high = start + length + search_samples
    if low < 0 or high > len(reference):
        return None
    search = np.asarray(reference[low:high], dtype=np.float64)
    curve = ncc_valid(search, warped)
    # A distant peak of a pure tone is not independent evidence of timing.
    found = peak(curve, exclusion=max(2, round(.0006 * sr)), absolute=True)
    if found is None:
        return None
    method = "raw"
    # Different mixes can mask the common waveform in the full-band correlation.
    # Rescue it only when >=2 frequency bands independently agree on delay.
    # Narrow bands or a periodic tone fail the alternative-peak margin gate.
    if found["score"] < settings.min_raw_score or found["margin"] < settings.min_raw_margin:
        bands = []
        for low_hz, high_hz in ((200, 1500), (1500, 4500), (4500, 12000)):
            filt = butter(3, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
            filtered_search = sosfiltfilt(filt, search)
            filtered_template = sosfiltfilt(filt, warped)
            bc = ncc_valid(filtered_search, filtered_template)
            bp = peak(bc, exclusion=max(2, round(.0006 * sr)), absolute=True)
            if bp and bp["score"] >= settings.min_raw_score and bp["margin"] >= settings.min_raw_margin:
                bands.append((bp, (bp["index"] + bp["fraction"] - search_samples) / sr))
        agreement = []
        for bp, lag in bands:
            group = [(b, l) for b, l in bands if b["polarity"] == bp["polarity"] and abs(l - lag) <= .00012]
            if len(group) > len(agreement):
                agreement = group
        if len(agreement) >= 2:
            lag = float(np.median([v for _, v in agreement]))
            found = {"index": search_samples + lag * sr, "fraction": 0.,
                     "score": float(np.median([b["score"] for b, _ in agreement])),
                     "margin": float(min(b["margin"] for b, _ in agreement)),
                     "polarity": agreement[0][0]["polarity"]}
            method = "multiband"
    lag = (found["index"] + found["fraction"] - search_samples) / sr
    near_edge = min(found["index"], len(curve) - 1 - found["index"]) < 2
    if ((found["score"] < settings.min_raw_score or found["margin"] < settings.min_raw_margin)
            and window_sec is None and length > round(.040 * sr)):
        # An imperfect BASE slope can smear a 220 ms window even when its
        # center delay is real. A shorter waveform window measures that center
        # without pretending that the residual rate is already corrected.
        shorter = measure_window(reference, target, mapping, target_time, settings, radius, window_sec=.040)
        if shorter and shorter["score"] >= settings.min_raw_score and shorter["margin"] >= settings.min_raw_margin:
            shorter["method"] += "-short"
            return shorter
    return {**found, "correction_sec": float(lag), "near_edge": near_edge, "method": method,
            "zero_score": float(abs(curve[search_samples]))}


def measure(reference, target, mapping, target_time, settings):
    center = float(mapping.forward([target_time])[0])
    result = Measurement(float(target_time), center, None)
    if not np.isfinite(center):
        return result
    observations = []
    spacing = min(.45, settings.anchor_step / 6)
    for dt in (-spacing, 0., spacing):
        found = measure_window(reference, target, mapping, target_time + dt, settings)
        if (found and not found["near_edge"] and found["score"] >= settings.min_raw_score
                and found["margin"] >= settings.min_raw_margin):
            observations.append((dt, found))
    result.support = len(observations)
    if len(observations) < 2:
        return result
    # Fit residual slope, rather than falsely treating rate drift as disagreement.
    xs = np.array([x for x, _ in observations])
    # Consensus is about the source clock, not about the residual of a
    # deliberately curved candidate. Subtracting a curved time map makes a
    # perfectly consistent source residual look curved/inconsistent. Undo that
    # known map before checking the three independent observations.
    values = np.array([m['correction_sec'] + float(mapping.forward([target_time + dt])[0])
                       - (target_time + dt) for dt, m in observations])
    slope, intercept = np.polyfit(xs, values, 1)
    scatter = np.max(np.abs(values - (intercept + slope * xs)))
    # Neighbor agreement is a gate, not an averaging filter. Averaging a
    # curved residual erases the exact center measurement by tens of us.
    central = next((m for dt, m in observations if dt == 0.), None)
    correction = central['correction_sec'] if central is not None else intercept + target_time - center
    polarities = {m["polarity"] for _, m in observations}
    result.score = float(np.median([m["score"] for _, m in observations]))
    result.margin = float(np.min([m["margin"] for _, m in observations]))
    result.spread_sec = float(scatter)
    result.polarity = observations[0][1]["polarity"]
    result.method = "multiband" if any(m["method"].startswith("multiband") for _, m in observations) else "raw"
    if any(m["method"].endswith("-short") for _, m in observations):
        result.method += "-short"
    if (scatter > settings.raw_consensus or abs(slope) > .003 or len(polarities) != 1):
        result.status = "INCONSISTENT"
        # At an existing splice, neighbouring windows can legitimately have
        # different delays. Accept its center only with strong unique PCM and
        # an independent reverse search, not by relaxing the consensus gate.
        if (central is not None and central['score'] >= .85 and central['margin'] >= .12
                and len(polarities) == 1):
            event = center + central['correction_sec']
            inverse = _InverseMap(mapping)
            predicted = float(inverse.forward([event])[0])
            reverse = measure_window(target, reference, inverse, event, settings)
            if (np.isfinite(predicted) and reverse and not reverse['near_edge']
                    and reverse['score'] >= .85 and reverse['margin'] >= .12
                    and reverse['polarity'] == central['polarity']
                    and abs(predicted + reverse['correction_sec'] - target_time) <= 2 / settings.analysis_rate):
                result.status = 'MEASURED'
                result.correction_sec = float(central['correction_sec'])
                result.method += '-boundary'
        return result
    result.correction_sec = float(correction)
    result.status = "MEASURED"
    return result


def grid(mapping, settings, phase=0.):
    centers = []
    # Keep edge probes and training windows disjoint, without discarding a
    # whole second of usable PCM at each edge of a short diagnostic fragment.
    margin = (.25 + min(.45, settings.anchor_step / 6)
              + min(settings.micro_window, settings.anchor_step / 8))
    for p in mapping.pieces:
        # A large negative offset can place initial target anchors before the
        # start of reference audio. Start in the COMMON timeline, not at t=0.
        lo = max(p.target[0] + margin, margin,
                 float(np.interp(max(p.reference[0], 0.) + margin, p.reference, p.target)))
        hi = p.target[-1] - margin
        if hi <= lo:
            continue
        # Each piece has a stable grid. Holdout is halfway between training anchors.
        values = np.arange(lo + phase * settings.anchor_step, hi, settings.anchor_step)
        centers.extend(values.tolist())
    return centers


def collect(reference, target, mapping, settings, log=print, progress=None, centers=None):
    records = []
    centers = grid(mapping, settings) if centers is None else centers
    for i, t in enumerate(centers):
        records.append(measure(reference, target, mapping, t, settings))
        if progress:
            progress(i + 1, len(centers))
        if i % 25 == 0 or i == len(centers) - 1:
            log(f"  PCM-микропоиск: {i + 1}/{len(centers)} точек, подтверждено {sum(r.status == 'MEASURED' for r in records)}.")
    return records


def refined_map(base, measurements, settings, enabled_times=None, audit=None):
    """Only consecutive supported anchors form a region; rejected points are barriers.

    Anchor times are inserted exactly (v7's 1-second map grid could miss an anchor).
    No convolution/averaging erases an accepted residual estimate.
    """
    output = []
    for piece in base.pieces:
        rows = [m for m in measurements if piece.target[0] <= m.target_time <= piece.target[-1]]
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
                    if np.max(np.abs(vals - fitted_values)) <= .25 / settings.analysis_rate:
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
                      and (enabled_times is None or m.target_time in enabled_times))
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
