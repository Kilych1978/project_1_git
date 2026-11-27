"""Conservative, channel-aware coarse time measurements from spectral envelopes.

This module proposes measurements; it never modifies an audio map or audio.  A
measurement is confirmed only after channel/band selection on training windows
and verification on separate windows.  Scores are envelope correlations, not a
claim that PCM from different mixes is identical.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate

_MIN_SCORE = 0.62
_MIN_MARGIN = 0.10
_MIN_ENERGY = 1.0e-5
_RELATIVE_ENERGY = 0.01
_MAX_RATE_CHANGE = 0.08


@dataclass
class _Prepared:
    values: np.ndarray
    energy: np.ndarray
    times: np.ndarray
    views: list[str]
    hop: float
    active: list[int]
    duplicate_views: dict[str, str]


def _prepare(bank: Any, begin: float, end: float) -> _Prepared:
    raw = np.asarray(bank.values)
    energy = np.asarray(bank.energy)
    hop = float(bank.hop_sec)
    origin = float(bank.origin_sec)
    if raw.ndim != 3 or energy.shape != raw.shape[:2] or hop <= 0:
        raise ValueError('Invalid feature bank dimensions or hop.')
    if len(bank.views) != raw.shape[1] or raw.shape[2] < 2:
        raise ValueError('Invalid feature bank views or bands.')
    # The only material copied from a possibly memory-mapped bank is this bounded
    # query region.  A margin supports the one-second local trend subtraction.
    first = max(0, int(math.floor((begin - 1.0 - origin) / hop)))
    last = min(len(raw), int(math.ceil((end + 1.0 - origin) / hop)) + 1)
    if last <= first:
        return _Prepared(np.empty((0, raw.shape[1], raw.shape[2])),
                         np.empty((0, raw.shape[1])), np.empty(0),
                         list(bank.views), hop, [], {})
    values = np.array(raw[first:last], dtype=np.float64)
    physical = np.array(energy[first:last], dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(physical)):
        raise ValueError('Non-finite feature data.')
    np.maximum(values, 0.0, out=values)
    times = origin + np.arange(first, last, dtype=np.float64) * hop
    region = (times >= begin) & (times <= end)
    if not region.any():
        region[:] = True
    level = np.sqrt(np.mean(physical[region] ** 2, axis=0))
    reference_level = max(float(np.max(level)), _MIN_ENERGY)
    active = [i for i, x in enumerate(level)
              if x >= _MIN_ENERGY and x >= reference_level * _RELATIVE_ENERGY]
    duplicates: dict[str, str] = {}
    kept: list[int] = []
    # Duplicates are removed for work reduction, never counted as independent
    # evidence.  Every actual vote below comes from a different time interval.
    for index in active:
        x = values[region, index].ravel()
        norm = float(np.linalg.norm(x))
        duplicate = None
        for old in kept:
            y = values[region, old].ravel()
            ynorm = float(np.linalg.norm(y))
            if norm and ynorm and np.linalg.norm(x / norm - y / ynorm) < 1e-3:
                duplicate = old
                break
        if duplicate is None:
            kept.append(index)
        else:
            duplicates[str(bank.views[index])] = str(bank.views[duplicate])
    # Log RMS prevents a single loud transient from owning a whole window.  The
    # floor is physical, and cannot lift a nearly silent Side projection past
    # the energy gate above.
    floor = max(_MIN_ENERGY * 0.1, float(np.max(values)) * 1e-5)
    values = np.log(np.maximum(values, floor))
    trend_frames = max(3, int(round(1.0 / hop)) | 1)
    values -= uniform_filter1d(values, size=trend_frames, axis=0, mode='nearest')
    scale = np.std(values[region], axis=0)
    # Numerical residue in otherwise constant bands must not become evidence.
    scale_floor = np.maximum(0.03, np.median(scale, axis=1, keepdims=True) * 0.2)
    values /= np.maximum(scale, scale_floor)[None, :, :]
    return _Prepared(values, physical, times, list(bank.views), hop, kept, duplicates)


def _bands(bank: Any, count: int) -> list[tuple[str, np.ndarray]]:
    parameters = getattr(bank, 'parameters', {}) or {}
    centers = np.asarray(parameters.get('band_centers_hz', []), dtype=float)
    output = [('all', np.arange(count))]
    if len(centers) == count:
        for name, lo, hi in [('low_mid', 120.0, 1500.0), ('high', 1500.0, 6000.0)]:
            indices = np.flatnonzero((centers >= lo) & (centers < hi))
            if len(indices) >= 3 and len(indices) < count:
                output.append((name, indices))
    return output


def _interpolate(values: np.ndarray, times: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query, times, values[:, i])
                            for i in range(values.shape[1])])


def _peak(ref: _Prepared, tgt: _Prepared, rv: int, tv: int,
          bands: np.ndarray, target_center: float, width: float,
          offset_center: float, radius: float, rate: float = 1.0) -> dict | None:
    hop = max(ref.hop, tgt.hop)
    n = max(8, int(round(width / hop)))
    relative = (np.arange(n) - (n - 1) / 2.0) * hop
    target_times = target_center + relative
    if target_times[0] < tgt.times[0] or target_times[-1] > tgt.times[-1]:
        return None
    template = _interpolate(tgt.values[:, tv][:, bands], tgt.times, target_times)
    # Center each band in the template; reference rolling means are accounted
    # for analytically below, including changing gain in different mixes.
    template -= np.mean(template, axis=0)
    template_power = float(np.sum(template * template))
    template_energy = float(np.sqrt(np.mean(np.interp(target_times, tgt.times,
                                                      tgt.energy[:, tv]) ** 2)))
    if template_power < n * len(bands) * 0.015 or template_energy < _MIN_ENERGY:
        return None
    desired_lo = target_center + offset_center - radius
    desired_hi = target_center + offset_center + radius
    half = rate * (n - 1) * hop / 2.0
    lo = max(desired_lo, float(ref.times[0]) + half)
    hi = min(desired_hi, float(ref.times[-1]) - half)
    if hi <= lo:
        return None
    # This virtual reference grid advances at the fitted rate.  Thus each
    # compared holdout waveform sees the same local speed, not just a shift.
    step = rate * hop
    candidate_count = int(math.floor((hi - lo) / step)) + 1
    sequence_times = lo - half + np.arange(candidate_count + n - 1) * step
    sequence = _interpolate(ref.values[:, rv][:, bands], ref.times, sequence_times)
    numerator = np.zeros(candidate_count, dtype=np.float64)
    for band in range(len(bands)):
        numerator += correlate(sequence[:, band], template[:, band],
                               mode='valid', method='fft')
    sums = np.vstack((np.zeros((1, len(bands))), np.cumsum(sequence, axis=0)))
    squares = np.r_[0.0, np.cumsum(np.sum(sequence * sequence, axis=1))]
    rolling_sum = sums[n:] - sums[:-n]
    power = squares[n:] - squares[:-n] - np.sum(rolling_sum * rolling_sum, axis=1) / n
    # FFT roundoff can leave a tiny nonzero numerator over exact padding
    # silence.  Dividing it by an almost-zero feature variance produces a
    # spurious NCC=1 after clipping.  Reject such *candidate windows before*
    # selecting either the best peak or its competing peak.  Rejecting only
    # the chosen winner afterwards would hide a perfectly valid next peak.
    physical_sequence = np.interp(sequence_times, ref.times, ref.energy[:, rv])
    physical_squares = np.r_[0.0, np.cumsum(physical_sequence ** 2)]
    physical_power = (physical_squares[n:] - physical_squares[:-n]) / n
    variance_floor = max(1e-12, float(np.max(power)) * 1e-12)
    valid = (power > variance_floor) & (physical_power >= _MIN_ENERGY ** 2)
    if not np.any(valid):
        return None
    curve = numerator / np.sqrt(np.maximum(template_power * power, 1e-30))
    curve = np.clip(curve, -1.0, 1.0)
    curve[~valid] = -np.inf
    # Band-RMS features are nonnegative before whitening: a negative spectral
    # envelope correlation is not polarity reversal of PCM and is not a match.
    index = int(np.argmax(curve))
    best = float(curve[index])
    centers = lo + np.arange(candidate_count) * step
    exclusion = max(0.12, 3.0 * hop)
    outside = valid & (np.abs(centers - centers[index]) > exclusion)
    second = float(np.max(curve[outside])) if outside.any() else best
    margin = best - second
    local = np.flatnonzero((curve >= best - 0.05) &
                          (np.abs(centers - centers[index]) <= exclusion))
    peak_width = float(np.ptp(centers[local])) if len(local) > 1 else hop
    chosen = float(centers[index])
    reference_times = chosen + rate * relative
    physical = float(np.sqrt(np.mean(np.interp(reference_times, ref.times,
                                               ref.energy[:, rv]) ** 2)))
    if physical < _MIN_ENERGY:
        return None
    lag = chosen - target_center
    boundary = abs(lag - offset_center) >= max(0.0, radius - 2.0 * hop)
    return {'target_time': float(target_center), 'reference_time': chosen,
            'target_start': float(target_times[0]), 'target_end': float(target_times[-1]),
            'window_sec': float(width),
            'offset': lag, 'score': best, 'margin': float(margin),
            'peak_width_sec': peak_width, 'hit_boundary': bool(boundary),
            'reference_rms': physical, 'target_rms': template_energy,
            'passes_peak_gate': bool(best >= _MIN_SCORE and margin >= _MIN_MARGIN
                                     and not boundary)}


def _fit(probes: list[dict], query: float, hop: float,
         support: float) -> dict | None:
    usable = [p for p in probes if p['passes_peak_gate']]
    if len(usable) < 3:
        return None
    best = None
    tolerance = max(0.025, 2.0 * hop)
    for a, b in itertools.combinations(usable, 2):
        dx = b['target_time'] - a['target_time']
        if abs(dx) < support * 0.25:
            continue
        slope = (b['offset'] - a['offset']) / dx
        if abs(slope) > _MAX_RATE_CHANGE:
            continue
        intercept = a['offset'] - slope * (a['target_time'] - query)
        selected = [p for p in usable
                    if abs(p['offset'] - intercept - slope * (p['target_time'] - query)) <= tolerance]
        if len(selected) < 3:
            continue
        span = max(p['target_time'] for p in selected) - min(p['target_time'] for p in selected)
        if span < support * 0.5:
            continue
        metric = (len(selected), sum(p['score'] for p in selected), span)
        if best is None or metric > best[0]:
            best = metric, selected
    if best is None:
        return None
    selected = best[1]
    xx = np.asarray([p['target_time'] - query for p in selected])
    yy = np.asarray([p['offset'] for p in selected])
    slope, offset = np.polyfit(xx, yy, 1, w=[p['score'] for p in selected])
    if abs(slope) > _MAX_RATE_CHANGE:
        return None
    errors = yy - (offset + slope * xx)
    return {'offset': float(offset), 'rate': float(1.0 + slope),
            'train_count': len(selected), 'train_span_sec': float(np.ptp(xx)),
            'train_max_error_sec': float(np.max(np.abs(errors))),
            'train_score': float(np.median([p['score'] for p in selected])),
            'train_margin': float(np.median([p['margin'] for p in selected])),
            'train_times': [p['target_time'] for p in selected]}


def _frozen(frozen: Any) -> tuple[str | None, str | None, str | None]:
    if frozen is None:
        return None, None, None
    if isinstance(frozen, dict):
        return frozen.get('reference_view'), frozen.get('target_view'), frozen.get('band')
    if isinstance(frozen, (tuple, list)) and 2 <= len(frozen) <= 3:
        return str(frozen[0]), str(frozen[1]), str(frozen[2]) if len(frozen) == 3 else None
    raise ValueError('frozen_views must be a view pair or a dictionary.')


def match_window(refbank: Any, tgtbank: Any, target_time: float,
                 search_center: float = 0.0, search_radius: float = 15.0,
                 window_sec: float = 6.0, frozen_views: Any = None) -> dict:
    """Measure a local affine relation without consulting an existing time map.

    ``search_center`` is an offset in seconds (reference minus target), not an
    absolute reference timestamp.  The returned ``offset`` is that difference
    at ``target_time``; ``rate`` is d(reference time)/d(target time).
    ``window_sec`` is total training/holdout support, not one correlation window.
    ``frozen_views`` accepts (reference_view, target_view[, band]) or a dict with
    those keys.  It never bypasses energy, uniqueness, or heldout checks.
    """
    answer = {'status': 'UNVERIFIABLE', 'target_time': float(target_time),
              'method': 'band_rms_log_local_whiten_v1',
              'support_sec': float(window_sec),
              'offset': None, 'rate': None, 'score': 0.0, 'margin': 0.0,
              'uncertainty_sec': None, 'reference_view': None,
              'target_view': None, 'band': None, 'probes': [],
              'disjoint_holdouts': [], 'hit_boundary': False,
              'reason': 'No reliable common spectral events.'}
    if not all(math.isfinite(float(x)) for x in
               (target_time, search_center, search_radius, window_sec)):
        raise ValueError('Time parameters must be finite.')
    if search_radius <= 0 or window_sec < 3.0:
        raise ValueError('A positive search radius and at least 3 s support are required.')
    if np.shape(refbank.values)[-1] != np.shape(tgtbank.values)[-1]:
        raise ValueError('Reference and target feature bands must match.')
    frozen_ref, frozen_tgt, frozen_band = _frozen(frozen_views)
    half = window_sec / 2.0
    ref = _prepare(refbank, target_time + search_center - search_radius - half,
                   target_time + search_center + search_radius + half)
    tgt = _prepare(tgtbank, target_time - half, target_time + half)
    # A frozen name may be a harmless duplicate of an active view.  Preserve
    # that exact requested identity; do not silently switch channels at render.
    for prepared, requested in ((ref, frozen_ref), (tgt, frozen_tgt)):
        if requested in prepared.duplicate_views:
            prepared.active = [prepared.views.index(requested)]
    answer['duplicate_views'] = {'reference': ref.duplicate_views, 'target': tgt.duplicate_views}
    answer['active_views'] = {'reference': [ref.views[i] for i in ref.active],
                              'target': [tgt.views[i] for i in tgt.active]}
    if not len(ref.times) or not len(tgt.times) or not ref.active or not tgt.active:
        answer['reason'] = 'Insufficient non-silent, non-duplicate feature data.'
        return answer
    hop = max(ref.hop, tgt.hop)
    width = min(0.65, window_sec / 10.0)
    # Interleaved, explicitly non-overlapping intervals.  No holdout chooses a
    # view, frequency band, rate, or training candidate.
    train_times = target_time + window_sec * np.asarray([-0.375, -0.125, 0.125, 0.375])
    holdout_times = target_time + window_sec * np.asarray([-0.25, 0.0, 0.25])
    candidates = []
    diagnostic = None
    for rv in ref.active:
        if frozen_ref is not None and ref.views[rv] != frozen_ref:
            continue
        for tv in tgt.active:
            if frozen_tgt is not None and tgt.views[tv] != frozen_tgt:
                continue
            for name, bands in _bands(refbank, ref.values.shape[2]):
                if frozen_band is not None and name != frozen_band:
                    continue
                probes = [_peak(ref, tgt, rv, tv, bands, float(t), width,
                                search_center, search_radius) for t in train_times]
                probes = [p for p in probes if p is not None]
                fit = _fit(probes, target_time, hop, window_sec)
                peak_quality = (sum(p['passes_peak_gate'] for p in probes),
                                float(np.median([p['score'] for p in probes])) if probes else -1.0)
                item = {'rv': rv, 'tv': tv, 'band': name, 'bands': bands,
                        'probes': probes, 'fit': fit, 'quality': peak_quality}
                if diagnostic is None or peak_quality > diagnostic['quality']:
                    diagnostic = item
                if fit is not None:
                    candidates.append(item)
    answer['training_candidate_count'] = len(candidates)
    if not candidates:
        if diagnostic is not None:
            answer['probes'] = diagnostic['probes']
            answer['reference_view'] = ref.views[diagnostic['rv']]
            answer['target_view'] = tgt.views[diagnostic['tv']]
            answer['band'] = diagnostic['band']
            peaks = diagnostic['probes']
            if peaks:
                answer['score'] = float(np.median([p['score'] for p in peaks]))
                answer['margin'] = float(np.median([p['margin'] for p in peaks]))
                answer['hit_boundary'] = any(p['hit_boundary'] for p in peaks)
                if sum(p['score'] >= _MIN_SCORE and p['margin'] < _MIN_MARGIN for p in peaks) >= 3:
                    answer['status'] = 'AMBIGUOUS'
                    answer['reason'] = 'Competing or repeating spectral events.'
                elif answer['hit_boundary'] and sum(p['score'] >= _MIN_SCORE for p in peaks) >= 3:
                    answer['status'] = 'BOUNDARY'
                    answer['reason'] = 'Prominent events touch the search boundary.'
        return answer
    candidates.sort(key=lambda c: (c['fit']['train_count'], c['fit']['train_score'],
                                   c['fit']['train_margin']), reverse=True)
    chosen = candidates[0]
    fit = chosen['fit']
    answer.update({k: fit[k] for k in ('offset', 'rate')})
    answer.update(reference_view=ref.views[chosen['rv']], target_view=tgt.views[chosen['tv']],
                  band=chosen['band'], band_indices=chosen['bands'].tolist(),
                  probes=chosen['probes'], training=fit,
                  score=fit['train_score'], margin=fit['train_margin'])
    answer['frozen_selection'] = {'reference_view': answer['reference_view'],
                                  'target_view': answer['target_view'],
                                  'band': answer['band']}
    answer['frozen_views'] = [answer['reference_view'], answer['target_view']]
    contradictory_training = [p for p in chosen['probes']
                              if p['passes_peak_gate'] and
                              abs(p['offset'] - fit['offset'] -
                                  (fit['rate'] - 1.0) * (p['target_time'] - target_time))
                              > max(0.04, 3.0 * hop)]
    if contradictory_training:
        answer['status'] = 'AMBIGUOUS'
        answer['reason'] = 'Strong training events contradict a single affine relation.'
        return answer
    # A similarly strong but incompatible training solution is evidence of
    # ambiguity, not an extra vote for whichever channel happened to rank first.
    for rival in candidates[1:]:
        other = rival['fit']
        difference = abs(other['offset'] - fit['offset']) + half * abs(other['rate'] - fit['rate'])
        if (other['train_count'] >= fit['train_count'] and
                other['train_score'] >= fit['train_score'] - 0.05 and difference > 0.10):
            answer['status'] = 'AMBIGUOUS'
            answer['reason'] = 'Independent channel/band proposals disagree.'
            answer['conflicting_training'] = {**other, 'reference_view': ref.views[rival['rv']],
                                               'target_view': tgt.views[rival['tv']], 'band': rival['band']}
            return answer
    held = []
    for time in holdout_times:
        predicted = fit['offset'] + (fit['rate'] - 1.0) * (float(time) - target_time)
        point = _peak(ref, tgt, chosen['rv'], chosen['tv'], chosen['bands'],
                      float(time), width, search_center, search_radius, fit['rate'])
        if point is None:
            continue
        point['predicted_offset'] = float(predicted)
        point['residual_sec'] = float(point['offset'] - predicted)
        point['confirms_frozen_model'] = bool(point['passes_peak_gate'] and
                                             abs(point['residual_sec']) <= max(0.04, 3.0 * hop))
        held.append(point)
    answer['disjoint_holdouts'] = held
    answer['hit_boundary'] = any(p['hit_boundary'] for p in chosen['probes'] + held)
    if any(p['hit_boundary'] and p['score'] >= _MIN_SCORE
           for p in chosen['probes'] + held):
        answer['status'] = 'BOUNDARY'
        answer['reason'] = 'Prominent training or holdout events touch the search boundary.'
        return answer
    contradictions = [p for p in held if p['passes_peak_gate']
                      and not p['confirms_frozen_model']]
    if contradictions:
        answer['status'] = 'AMBIGUOUS'
        answer['reason'] = 'Independent holdout events contradict the frozen affine proposal.'
        return answer
    passed = [p for p in held if p['confirms_frozen_model']]
    spread = max((p['target_time'] for p in passed), default=0.0) - min((p['target_time'] for p in passed), default=0.0)
    if len(passed) < 2 or spread < window_sec * 0.24:
        answer['reason'] = 'The frozen training proposal failed independent holdout windows.'
        return answer
    error = max(fit['train_max_error_sec'], max(abs(p['residual_sec']) for p in passed))
    peak_width = max(p['peak_width_sec'] for p in passed)
    answer['uncertainty_sec'] = float(max(2.0 * hop, 2.0 * error, peak_width / 2.0))
    answer['score'] = float(min(fit['train_score'], np.median([p['score'] for p in passed])))
    answer['margin'] = float(min(fit['train_margin'], np.median([p['margin'] for p in passed])))
    if abs(fit['offset'] - search_center) >= search_radius - max(0.05, answer['uncertainty_sec']):
        answer['status'] = 'BOUNDARY'
        answer['hit_boundary'] = True
        answer['reason'] = 'The affine proposal touches the search boundary.'
    else:
        answer['status'] = 'CONFIRMED'
        answer['reason'] = 'Training channel/band proposal confirmed by disjoint holdout events.'
    return answer
