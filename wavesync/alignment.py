"""Coarse envelope search, PAL/film rate hypotheses, robust timeline fitting.

The v3.3 normalized FFT matcher is retained as a compact function, without its
duplicate definitions. Offset/rate are estimated jointly, not confused with lag.
"""
import numpy as np
from scipy.stats import theilslopes
from .dsp import ncc_valid, peak
from .model import Piece, TimeMap


class AlignmentError(RuntimeError):
    def __init__(self, message, records=None):
        super().__init__(message)
        self.records = records or []


def envelope_match(reference, target, target_time, rate, prediction, window, radius, settings):
    hop = settings.envelope_hop
    half_feature = settings.envelope_window / 2
    n = max(32, round(window / hop))
    u = (np.arange(n) - (n - 1) / 2) * hop
    indices = (target_time + u / rate - half_feature) / hop
    if indices[0] < 0 or indices[-1] > len(target) - 1:
        return None
    template = np.interp(indices, np.arange(len(target)), target)
    low = max(0, int(np.floor((prediction + u[0] - half_feature - radius) / hop)))
    high = min(len(reference), int(np.ceil((prediction + u[-1] - half_feature + radius) / hop)) + 1)
    curve = ncc_valid(reference[low:high], template)
    found = peak(curve, exclusion=max(1, round(.25 / hop)))
    if found is None:
        return None
    center = (low + found["index"] + found["fraction"]) * hop + half_feature - u[0]
    return {"target_time": float(target_time), "reference_time": float(center),
            "offset": float(center - target_time), "score": found["score"], "margin": found["margin"]}


def robust_line(times, ref_times):
    # Bound cost for long films; Theil-Sen has quadratic pair storage.
    t = np.asarray(times)
    r = np.asarray(ref_times)
    if len(t) > 100:
        indices = np.linspace(0, len(t) - 1, 100).astype(int)
        t, r = t[indices], r[indices]
    slope = float(theilslopes(r, t)[0])
    offset = float(np.median(r - slope * t))
    return slope, offset


def coarse_alignment(ref_env, tar_env, ref_duration, tar_duration, settings, log=print):
    window = min(settings.coarse_window, min(ref_duration, tar_duration) / 4)
    if window < .4:
        raise AlignmentError("Дорожка слишком короткая: требуется минимум 1,6 секунды.")
    margin = window / (2 * (1 - settings.max_rate_change)) + .05
    centers = np.linspace(margin, tar_duration - margin, 7)
    rates = [1., 25 / 24, 24 / 25, 25 / (24000 / 1001), (24000 / 1001) / 25, 1000 / 1001, 1001 / 1000]
    duration_ratio = ref_duration / tar_duration
    if abs(duration_ratio - 1) <= settings.max_rate_change:
        rates.append(duration_ratio)
    hypotheses = []
    for rate in sorted(set(rates)):
        records = []
        for t in centers:
            result = envelope_match(ref_env, tar_env, t, rate, rate * t, window,
                                    settings.search_radius, settings)
            if result:
                records.append(result)
        if records:
            # Several windows must support the same rate; one musical cue is insufficient.
            merit = float(np.median([r["score"] for r in records]))
            hypotheses.append((merit, rate, records))
    if not hypotheses:
        raise AlignmentError("Не найден общий звук для начального сопоставления.")
    merit, rate, initial = max(hypotheses, key=lambda row: row[0])
    strong = [r for r in initial if r["score"] >= settings.min_coarse_score
              and r["margin"] >= settings.min_coarse_margin]
    if len(strong) < 3:
        raise AlignmentError("Недостаточно общих звуковых событий. Автоматическая коррекция отменена.", initial)
    fitted_rate, offset = robust_line([r["target_time"] for r in strong], [r["reference_time"] for r in strong])
    residuals = np.array([r["reference_time"] - (fitted_rate * r["target_time"] + offset) for r in strong])
    if abs(fitted_rate - 1) <= settings.max_rate_change and np.median(np.abs(residuals)) < .020:
        rate = fitted_rate
    else:
        offset = float(np.median([r["reference_time"] - rate * r["target_time"] for r in strong]))
    log(f"  Начальная оценка: скорость {rate:.9f}, смещение {offset:+.6f} с.")
    times = np.unique(np.r_[margin, np.arange(margin, tar_duration - margin, settings.anchor_step), tar_duration - margin])
    records = []
    for i, t in enumerate(times):
        r = envelope_match(ref_env, tar_env, t, rate, rate * t + offset, window, settings.search_radius, settings)
        if r:
            r["accepted"] = r["score"] >= settings.min_coarse_score and r["margin"] >= settings.min_coarse_margin
            records.append(r)
        else:
            records.append({"target_time": float(t), "accepted": False, "score": 0., "margin": 0.})
        if i % 25 == 0 or i == len(times) - 1:
            log(f"  Крупное выравнивание: {i + 1}/{len(times)} точек.")
    accepted = [r for r in records if r["accepted"]]
    if len(accepted) < 3 or len(accepted) / len(records) < .60:
        raise AlignmentError("Карта не подтверждена на достаточной части дорожки. Рендер отменён.", records)
    t = np.array([r["target_time"] for r in accepted])
    r = np.array([r["reference_time"] for r in accepted])
    if np.max(np.diff(t)) > settings.max_untrusted_gap:
        raise AlignmentError("Обнаружен длинный участок без надёжного общего звука. Нужна ручная проверка.", records)
    rate, offset = robust_line(t, r)
    if abs(rate - 1) > settings.max_rate_change:
        raise AlignmentError("Скорость выходит за пределы безопасного диапазона ±8%.", records)
    errors = r - (rate * t + offset)
    changes = np.diff(errors)
    cuts = np.flatnonzero(np.abs(changes) > np.maximum(.080, .003 * np.diff(t)))
    if len(cuts):
        ranges = "; ".join(f"{t[i]:.2f}–{t[i+1]:.2f} с" for i in cuts[:8])
        raise AlignmentError("Обнаружен возможный монтажный скачок: " + ranges
                             + ". Он не будет размазан растяжением. См. docs/EDIT_MAP_RU.md.", records)
    # Remove envelope quantization when a constant-rate model explains the data.
    # Otherwise retain the nonlinear trend and let PCM anchors resolve the residual.
    if np.max(np.abs(errors)) < .008:
        mapping = TimeMap.affine(tar_duration, rate, offset)
    else:
        node_t = np.r_[0., t, tar_duration]
        left_rate = float((r[1] - r[0]) / (t[1] - t[0]))
        right_rate = float((r[-1] - r[-2]) / (t[-1] - t[-2]))
        node_r = np.r_[r[0] - left_rate * t[0], r, r[-1] + right_rate * (tar_duration - t[-1])]
        mapping = TimeMap([Piece(node_t, node_r)])
    try:
        mapping.validate(settings.max_rate_change)
    except ValueError as exc:
        raise AlignmentError(str(exc), records) from exc
    mapping.notes.append("Начало и конец карты экстраполированы по ближайшему подтверждённому тренду; проверяются отдельно.")
    return mapping, records


def validate_coarse(mapping, ref_env, tar_env, tar_duration, settings):
    """Holdout windows never used in the coarse fit; compare absolute estimates equally."""
    window = min(settings.coarse_window / 2, tar_duration / 8)
    margin = max(window / 2 / .92 + .05, .5)
    centers = np.arange(margin + settings.anchor_step / 2, tar_duration - margin, settings.anchor_step)
    if len(centers) < 3:
        centers = np.linspace(margin, tar_duration - margin, 5)
    records = []
    for t in centers:
        pred = float(mapping.forward([t])[0])
        if not np.isfinite(pred):
            continue
        rate = mapping.local_rate(t)
        if not any(p.target[0] <= t - window / (2 * rate)
                   and t + window / (2 * rate) <= p.target[-1] for p in mapping.pieces):
            records.append({"target_time": float(t), "status": "CROSSES_EDIT"})
            continue
        if pred - window / 2 < .02 or pred + window / 2 > len(ref_env) * settings.envelope_hop - .02:
            records.append({"target_time": float(t), "status": "OUTSIDE_COMMON_AUDIO"})
            continue
        found = envelope_match(ref_env, tar_env, t, rate, pred, window, settings.search_radius, settings)
        if not found or found["score"] < settings.min_coarse_score or found["margin"] < settings.min_coarse_margin:
            records.append({"target_time": float(t), "status": "UNVERIFIABLE"})
            continue
        residual = found["reference_time"] - pred
        original_error = abs(found["reference_time"] - t)
        # Envelope validation cannot certify microseconds. Only coarse gross errors.
        regression = abs(residual) > max(.020, original_error + .010)
        records.append({**found, "predicted_reference": pred, "residual_sec": float(residual),
                        "original_error_sec": float(original_error), "status": "REGRESSION" if regression else "VERIFIED"})
    verified = [r for r in records if r["status"] == "VERIFIED"]
    if len(verified) < 3 or len(verified) / max(1, len(records)) < .60 or any(r["status"] == "REGRESSION" for r in records):
        raise AlignmentError("Крупная карта не прошла независимую проверку. Результат не будет выдан как синхронизированный.", records)
    return records
