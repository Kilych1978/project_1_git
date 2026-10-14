# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MICRO WAVEFORM REFINEMENT
# MODULE: micro_refiner.py
# ============================================================
# v6.9.1
#
# Fine local timing refinement at approximately millisecond resolution.
#
# Idea:
#   The ordinary 10 ms envelope is excellent for finding the correct piece,
#   but it can hide a 3–30 ms residual that becomes obvious when a waveform
#   is zoomed deeply in Audacity.
#
# This module therefore works directly from the already decoded first-channel
# PCM and builds a fine 2 ms RMS contour sampled every 1 ms. It compares the
# TARGET contour with the REFERENCE contour around the offset predicted by the
# current synchronization map.
#
# It does NOT require the two raw waveforms to be sample-identical. The main
# score is based on normalized local shape, with a second edge/variation score
# so the algorithm follows the same local acoustic events even when amplitude,
# language, or encoding differ.
#
# The resulting correction is only a CANDIDATE. Existing dense non-regression
# and real-render safety guards still have final authority.
# ============================================================

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from config import *
from common import *
from validator import map_offset_at_time


# ============================================================
# MICRO ANALYSIS SETTINGS
# ============================================================
# All tunables live in config.py. Keeping one source of truth is deliberate:
# the next test should require changing config.py only.


def _next_power_of_two(value: int) -> int:
    value = max(2, int(value))
    return 1 << (value - 1).bit_length()


def _read_pcm_samples(
    pcm_path: Path,
    start_sample: int,
    end_sample: int,
) -> np.ndarray:
    """Reads a bounded int16 PCM slice without loading the complete film."""
    if end_sample <= start_sample:
        return np.empty(0, dtype=np.float32)

    start_sample = max(0, int(start_sample))
    end_sample = max(start_sample, int(end_sample))

    count = end_sample - start_sample
    with pcm_path.open("rb") as file:
        file.seek(start_sample * 2)
        raw = file.read(count * 2)

    if not raw:
        return np.empty(0, dtype=np.float32)

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    samples /= 32768.0
    return samples


def _rms_feature(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """2 ms RMS frames with 1 ms hop, computed from cumulative energy."""
    signal = np.asarray(signal, dtype=np.float32)
    frame = max(1, int(round(sample_rate * MICRO_FEATURE_FRAME_MS / 1000.0)))
    hop = max(1, int(round(sample_rate * MICRO_FEATURE_HOP_MS / 1000.0)))

    if signal.size < frame:
        return np.empty(0, dtype=np.float32)

    power = np.asarray(signal, dtype=np.float64) ** 2
    cumulative = np.empty(power.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(power, out=cumulative[1:])

    starts = np.arange(
        0,
        signal.size - frame + 1,
        hop,
        dtype=np.int64,
    )
    sums = cumulative[starts + frame] - cumulative[starts]
    rms = np.sqrt(np.maximum(sums / float(frame), 0.0))

    return rms.astype(np.float32)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    values = values - np.mean(values)
    scale = np.std(values)
    if not math.isfinite(float(scale)) or scale < 1e-12:
        return np.zeros_like(values)
    return values / scale


def _valid_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 8:
        return 0.0
    a = _normalize(values_a)
    b = _normalize(values_b)
    if not np.any(a) or not np.any(b):
        return 0.0
    return float(np.dot(a, b) / float(len(a)))


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.empty(len(values) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(values, out=cumulative[1:])
    return cumulative[window:] - cumulative[:-window]


def _normalized_valid_xcorr(
    reference: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Normalized correlation for every valid placement of target in reference."""
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    n = len(target)
    m = len(reference)
    if n < 8 or m < n:
        return np.empty(0, dtype=np.float64)

    sum_t = float(np.sum(target))
    sum_t2 = float(np.dot(target, target))

    nfft = _next_power_of_two(m + n - 1)
    fft_reference = np.fft.rfft(reference, nfft)
    fft_target = np.fft.rfft(target[::-1], nfft)
    convolution = np.fft.irfft(
        fft_reference * fft_target,
        nfft,
    )

    # valid correlation for reference[start:start+n] against target
    dot = convolution[n - 1 : m]

    sum_r = _rolling_sum(reference, n)
    sum_r2 = _rolling_sum(reference * reference, n)

    numerator = dot - (sum_r * sum_t / float(n))
    variance_r = np.maximum(
        sum_r2 - (sum_r * sum_r / float(n)),
        0.0,
    )
    variance_t = max(
        sum_t2 - (sum_t * sum_t / float(n)),
        0.0,
    )

    denominator = np.sqrt(
        variance_r * variance_t
    )

    result = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _micro_score_from_features(
    reference_feature: np.ndarray,
    target_feature: np.ndarray,
):
    """Searches all lags and combines fine contour + local edge correlation."""
    if len(reference_feature) < len(target_feature) or len(target_feature) < 16:
        return {
            "best_index": None,
            "best_score": 0.0,
            "rms_corr": 0.0,
            "edge_corr": 0.0,
            "scores": np.empty(0, dtype=np.float64),
        }

    rms_scores = _normalized_valid_xcorr(
        reference_feature,
        target_feature,
    )

    reference_edge = np.abs(np.diff(reference_feature, prepend=reference_feature[:1]))
    target_edge = np.abs(np.diff(target_feature, prepend=target_feature[:1]))
    edge_scores = _normalized_valid_xcorr(
        reference_edge,
        target_edge,
    )

    count = min(len(rms_scores), len(edge_scores))
    if count == 0:
        return {
            "best_index": None,
            "best_score": 0.0,
            "rms_corr": 0.0,
            "edge_corr": 0.0,
            "scores": np.empty(0, dtype=np.float64),
        }

    rms_scores = rms_scores[:count]
    edge_scores = edge_scores[:count]
    scores = (
        MICRO_RMS_WEIGHT * rms_scores
        + MICRO_EDGE_WEIGHT * edge_scores
    )

    # Ignore NaN/Inf but do not hide a genuinely poor score.
    scores = np.nan_to_num(
        scores,
        nan=-1.0,
        posinf=-1.0,
        neginf=-1.0,
    )

    best_index = int(np.argmax(scores))
    return {
        "best_index": best_index,
        "best_score": float(scores[best_index]),
        "rms_corr": float(rms_scores[best_index]),
        "edge_corr": float(edge_scores[best_index]),
        "scores": scores,
        "rms_scores": rms_scores,
        "edge_scores": edge_scores,
    }


def _quadratic_peak_offset(scores: np.ndarray, index: int) -> float:
    """Sub-step refinement from three neighboring score samples."""
    if index is None or index <= 0 or index >= len(scores) - 1:
        return 0.0

    y0 = float(scores[index - 1])
    y1 = float(scores[index])
    y2 = float(scores[index + 1])
    denominator = y0 - 2.0 * y1 + y2
    if abs(denominator) < 1e-12:
        return 0.0

    delta = 0.5 * (y0 - y2) / denominator
    return float(np.clip(delta, -0.5, 0.5))


def micro_search_at_time(
    reference_pcm: Path,
    target_pcm: Path,
    target_center_sec: float,
    predicted_offset_sec: float,
):
    """
    Performs fine local search around the current map prediction.

    Search resolution is 1 ms. The compared descriptor is a 5 ms RMS contour
    moving every 1 ms, so it behaves like a "microscope" view of the waveform
    without attempting to match every individual sample.
    """
    sample_rate = ANALYSIS_SAMPLE_RATE
    half = MICRO_LOCAL_WINDOW_SEC
    radius = MICRO_SEARCH_RADIUS_SEC

    target_start = max(0.0, target_center_sec - half)
    target_end = target_center_sec + half

    reference_center = target_center_sec + predicted_offset_sec
    reference_start = max(0.0, reference_center - half - radius)
    reference_end = reference_center + half + radius

    target_samples = _read_pcm_samples(
        target_pcm,
        int(round(target_start * sample_rate)),
        int(round(target_end * sample_rate)),
    )

    reference_samples = _read_pcm_samples(
        reference_pcm,
        int(round(reference_start * sample_rate)),
        int(round(reference_end * sample_rate)),
    )

    target_feature = _rms_feature(target_samples, sample_rate)
    reference_feature = _rms_feature(reference_samples, sample_rate)

    if len(target_feature) < 16 or len(reference_feature) < len(target_feature):
        return None

    # Zero-map score at the CURRENT MAP prediction. It is the correct
    # baseline for a refinement pass: we ask whether a tiny additional shift
    # makes the local shape fit better than the map we already have.
    base_center_in_reference_samples = (
        int(round((reference_center - reference_start) * sample_rate))
    )
    hop_samples = max(1, int(round(sample_rate * MICRO_FEATURE_HOP_MS / 1000.0)))
    base_index = max(
        0,
        int(round(base_center_in_reference_samples / hop_samples))
        - int(round(half / (MICRO_FEATURE_HOP_MS / 1000.0))),
    )
    base_index = min(base_index, len(reference_feature) - len(target_feature))

    base_reference_feature = reference_feature[
        base_index : base_index + len(target_feature)
    ]
    baseline_score = _valid_correlation(
        base_reference_feature,
        target_feature,
    )

    search = _micro_score_from_features(
        reference_feature,
        target_feature,
    )

    if search["best_index"] is None:
        return None

    best_index = int(search["best_index"])

    # The feature starts are separated by MICRO_FEATURE_HOP_MS, so the offset
    # of the best valid placement can be computed directly from the expanded
    # reference segment start.
    feature_start_delta_sec = (
        best_index * MICRO_FEATURE_HOP_MS / 1000.0
    )
    target_feature_start_delta_sec = (
        (target_start - target_center_sec)
        + MICRO_FEATURE_FRAME_MS / 2000.0
    )
    reference_feature_start_delta_sec = (
        (reference_start - reference_center)
        + MICRO_FEATURE_FRAME_MS / 2000.0
    )

    best_reference_delta = (
        reference_feature_start_delta_sec
        + feature_start_delta_sec
        - target_feature_start_delta_sec
    )

    # Quadratic interpolation gives a sub-ms diagnostic estimate. The actual
    # map is still rendered through the existing time-domain renderer.
    sub_step = _quadratic_peak_offset(
        search["scores"],
        best_index,
    )
    refined_offset = float(
        predicted_offset_sec
        + best_reference_delta
        + sub_step * MICRO_SEARCH_STEP_SEC
    )

    scores = search["scores"]
    sorted_indices = np.argsort(scores)[::-1]
    second_score = -1.0
    for candidate_index in sorted_indices:
        if abs(int(candidate_index) - best_index) >= 2:
            second_score = float(scores[candidate_index])
            break
    if second_score < -0.5:
        second_score = float(scores[best_index])

    margin = float(scores[best_index] - second_score)
    score_gain = float(scores[best_index] - baseline_score)
    correction = float(refined_offset - predicted_offset_sec)

    return {
        "predicted_offset_sec": float(predicted_offset_sec),
        "baseline_score": float(baseline_score),
        "refined_offset_sec": refined_offset,
        "correction_sec": correction,
        "best_score": float(search["best_score"]),
        "rms_correlation": float(search["rms_corr"]),
        "edge_correlation": float(search["edge_corr"]),
        "score_gain": score_gain,
        "peak_margin": margin,
        "target_feature": target_feature,
        "reference_feature": reference_feature,
        "best_index": best_index,
        "reference_start_sec": reference_start,
        "target_start_sec": target_start,
        "scores": scores,
    }


def _is_strong_micro_candidate(
    result: dict,
    baseline_residual_sec: float | None = None,
) -> bool:
    if result is None:
        return False

    correction_sec = abs(float(result["correction_sec"]))

    # The dense validator measures the RESIDUAL left by the current map.
    # The micro result contains an absolute map offset, so comparing
    # abs(refined_offset) directly with the residual is mathematically wrong.
    # The residual after applying this micro correction is: 
    #     baseline_residual - correction
    # and that is what must genuinely improve.
    if baseline_residual_sec is not None:
        baseline_residual_sec = float(baseline_residual_sec)
        candidate_residual_sec = baseline_residual_sec - float(result["correction_sec"])
        if abs(candidate_residual_sec) > (
            abs(baseline_residual_sec) - MICRO_MIN_ERROR_IMPROVEMENT_SEC
        ):
            return False

    if correction_sec > MICRO_MAX_CORRECTION_SEC:
        return False

    return bool(
        result["best_score"] >= MICRO_MIN_SCORE
        and result["rms_correlation"] >= MICRO_MIN_RMS_CORRELATION
        and result["edge_correlation"] >= MICRO_MIN_EDGE_CORRELATION
        and result["score_gain"] >= MICRO_MIN_SCORE_GAIN
        and correction_sec >= MICRO_MIN_CORRECTION_SEC
        and result["peak_margin"] >= MICRO_PEAK_MARGIN
        and result.get("reverse_score", 0.0) >= MICRO_MIN_REVERSE_SCORE
        and result.get("reverse_rms_correlation", 0.0) >= MICRO_MIN_REVERSE_RMS_CORRELATION
        and result.get("reverse_disagreement_sec", float("inf")) <= MICRO_MAX_REVERSE_DISAGREEMENT_SEC
    )


def refine_sync_map_with_micro_waveform(
    analysis,
    base_sync_map,
    dense_validation_results,
    output_dir: Path,
    progress: Progress,
):
    """
    Builds a cautious micro-refined map from dense validation points.

    A correction becomes eligible only when several consecutive dense points
    independently support a stable local shift. Existing safety/render guards
    remain responsible for the final decision.
    """
    progress.stage(
        "МИКРО-УТОЧНЕНИЕ ФОРМЫ ВОЛНЫ",
        0.0,
    )

    reference_pcm = Path(analysis["reference_pcm"])
    target_pcm = Path(analysis["target_pcm"])

    records = []
    total = len(dense_validation_results)
    for index, dense in enumerate(dense_validation_results, start=1):
        target_time = float(dense["target_time"])
        predicted_offset = float(
            map_offset_at_time(
                base_sync_map,
                target_time,
            )
        )

        # Micro refinement is about the RESIDUAL LEFT BY THE CURRENT MAP,
        # not the original TARGET offset before the map.
        baseline_error = dense.get("after_error")
        if baseline_error is None:
            baseline_error = dense.get("before_error")
        if baseline_error is not None:
            baseline_error = abs(float(baseline_error))

        baseline_residual = dense.get("after_offset")
        if baseline_residual is None:
            baseline_residual = dense.get("before_offset")
        if baseline_residual is not None:
            # validator.after_offset is the absolute match offset. The
            # residual is relative to the currently proposed map.
            baseline_residual = float(baseline_residual) - predicted_offset

        record = {
            "target_time_sec": target_time,
            "predicted_offset_sec": predicted_offset,
            "baseline_error_sec": baseline_error,
            "baseline_residual_sec": baseline_residual,
            "status": "SKIPPED",
        }

        if (
            baseline_error is not None
            and baseline_error <= MICRO_MIN_BASELINE_ERROR_SEC
        ):
            try:
                result = micro_search_at_time(
                    reference_pcm,
                    target_pcm,
                    target_time,
                    predicted_offset,
                )
            except Exception as exc:
                result = None
                record["error"] = str(exc)

            if result is not None:
                # Дополнительная обратная проверка: повторяем поиск в обратном
                # направлении вокруг уже найденного смещения. Настоящий локальный
                # сдвиг должен воспроизводиться и так, и обратно; ложный пик часто
                # даёт другое смещение.
                reverse_center = target_time + float(result["refined_offset_sec"])
                reverse_result = micro_search_at_time(
                    target_pcm,
                    reference_pcm,
                    reverse_center,
                    -float(result["refined_offset_sec"]),
                )
                if reverse_result is not None:
                    reverse_implied_forward = -float(reverse_result["refined_offset_sec"])
                    reverse_disagreement = abs(
                        float(result["refined_offset_sec"])
                        - reverse_implied_forward
                    )
                    result["reverse_score"] = float(reverse_result["best_score"])
                    result["reverse_rms_correlation"] = float(reverse_result["rms_correlation"])
                    result["reverse_edge_correlation"] = float(reverse_result["edge_correlation"])
                    result["reverse_disagreement_sec"] = float(reverse_disagreement)
                    result["reverse_implied_offset_sec"] = float(reverse_implied_forward)
                else:
                    result["reverse_score"] = 0.0
                    result["reverse_rms_correlation"] = 0.0
                    result["reverse_edge_correlation"] = 0.0
                    result["reverse_disagreement_sec"] = float("inf")
                    result["reverse_implied_offset_sec"] = None

                record.update({
                    "baseline_score": result["baseline_score"],
                    "refined_offset_sec": result["refined_offset_sec"],
                    "correction_sec": result["correction_sec"],
                    "best_score": result["best_score"],
                    "rms_correlation": result["rms_correlation"],
                    "edge_correlation": result["edge_correlation"],
                    "score_gain": result["score_gain"],
                    "peak_margin": result["peak_margin"],
                    "reverse_score": result.get("reverse_score", 0.0),
                    "reverse_rms_correlation": result.get("reverse_rms_correlation", 0.0),
                    "reverse_edge_correlation": result.get("reverse_edge_correlation", 0.0),
                    "reverse_disagreement_sec": result.get("reverse_disagreement_sec"),
                    "reverse_implied_offset_sec": result.get("reverse_implied_offset_sec"),
                })
                record["status"] = (
                    "ACCEPTED_MICRO"
                    if _is_strong_micro_candidate(result, baseline_residual)
                    else "REJECTED_MICRO"
                )
            else:
                record["status"] = "NO_RESULT"

        records.append(record)

        progress.update(
            index / max(1, total) * 100.0,
            f"Проверка микросовмещения | t={format_seconds(target_time)}",
        )

    print()

    # --------------------------------------------------------
    # Только устойчивые непрерывные участки могут влиять на карту.
    # --------------------------------------------------------
    accepted = [
        item for item in records
        if item.get("status") == "ACCEPTED_MICRO"
    ]

    accepted_sorted = sorted(accepted, key=lambda item: item["target_time_sec"])
    runs = []
    current = []
    for item in accepted_sorted:
        if not current:
            current = [item]
            continue
        gap = (
            float(item["target_time_sec"])
            - float(current[-1]["target_time_sec"])
        )
        if gap <= MICRO_MAX_RUN_GAP_SEC:
            current.append(item)
        else:
            if len(current) >= MICRO_MIN_SUPPORT_POINTS:
                runs.append(current)
            current = [item]
    if len(current) >= MICRO_MIN_SUPPORT_POINTS:
        runs.append(current)

    new_map = {
        key: np.asarray(value).copy()
        if isinstance(value, np.ndarray)
        else value
        for key, value in base_sync_map.items()
    }

    map_time = np.asarray(new_map["target_time"], dtype=np.float64)
    base_offset = np.asarray(new_map["offset"], dtype=np.float64)
    micro_delta = np.zeros_like(base_offset, dtype=np.float64)

    run_summaries = []
    for run in runs:
        times = np.asarray(
            [float(item["target_time_sec"]) for item in run],
            dtype=np.float64,
        )
        deltas = np.asarray(
            [float(item["correction_sec"]) for item in run],
            dtype=np.float64,
        )

        first = float(times[0])
        last = float(times[-1])
        left = max(0.0, first - MICRO_TAPER_SEC)
        right = min(
            float(map_time[-1]),
            last + MICRO_TAPER_SEC,
        )

        key_times = [left, *times.tolist(), right]
        key_delta = [0.0, *deltas.tolist(), 0.0]

        inside = (
            (map_time >= left)
            & (map_time <= right)
        )
        if np.any(inside):
            micro_delta[inside] = np.interp(
                map_time[inside],
                np.asarray(key_times, dtype=np.float64),
                np.asarray(key_delta, dtype=np.float64),
            )

        run_summaries.append({
            "start_sec": first,
            "end_sec": last,
            "points": len(run),
            "median_correction_ms": float(np.median(deltas) * 1000.0),
            "max_correction_ms": float(np.max(np.abs(deltas)) * 1000.0),
        })

    new_map["offset"] = base_offset + micro_delta
    new_map["reference_time"] = map_time + new_map["offset"]

    # Сохраняем диагностику отдельных точек.
    csv_path = output_dir / MICRO_CSV
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([
            "target_time_sec",
            "predicted_offset_sec",
            "baseline_error_sec",
            "baseline_residual_sec",
            "baseline_score",
            "refined_offset_sec",
            "correction_sec",
            "best_score",
            "rms_correlation",
            "edge_correlation",
            "score_gain",
            "peak_margin",
            "reverse_score",
            "reverse_rms_correlation",
            "reverse_edge_correlation",
            "reverse_disagreement_sec",
            "reverse_implied_offset_sec",
            "status",
        ])
        for item in records:
            writer.writerow([
                f"{item['target_time_sec']:.6f}",
                f"{item['predicted_offset_sec']:.6f}",
                "" if item.get("baseline_error_sec") is None else f"{item['baseline_error_sec']:.6f}",
                "" if item.get("baseline_residual_sec") is None else f"{item['baseline_residual_sec']:.6f}",
                "" if item.get("baseline_score") is None else f"{item['baseline_score']:.6f}",
                "" if item.get("refined_offset_sec") is None else f"{item['refined_offset_sec']:.6f}",
                "" if item.get("correction_sec") is None else f"{item['correction_sec']:.6f}",
                "" if item.get("best_score") is None else f"{item['best_score']:.6f}",
                "" if item.get("rms_correlation") is None else f"{item['rms_correlation']:.6f}",
                "" if item.get("edge_correlation") is None else f"{item['edge_correlation']:.6f}",
                "" if item.get("score_gain") is None else f"{item['score_gain']:.6f}",
                "" if item.get("peak_margin") is None else f"{item['peak_margin']:.6f}",
                "" if item.get("reverse_score") is None else f"{item['reverse_score']:.6f}",
                "" if item.get("reverse_rms_correlation") is None else f"{item['reverse_rms_correlation']:.6f}",
                "" if item.get("reverse_edge_correlation") is None else f"{item['reverse_edge_correlation']:.6f}",
                "" if item.get("reverse_disagreement_sec") is None else f"{item['reverse_disagreement_sec']:.6f}",
                "" if item.get("reverse_implied_offset_sec") is None else f"{item['reverse_implied_offset_sec']:.6f}",
                item.get("status", ""),
            ])

    map_csv = output_dir / MICRO_MAP_CSV
    with map_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["target_time_sec", "base_offset_sec", "micro_correction_sec", "final_offset_sec"])
        for t, base, delta, final in zip(
            map_time,
            base_offset,
            micro_delta,
            new_map["offset"],
        ):
            writer.writerow([
                f"{float(t):.6f}",
                f"{float(base):.6f}",
                f"{float(delta):.6f}",
                f"{float(final):.6f}",
            ])

    # Save Russian report directly so the micro stage itself never creates an
    # English-only report waiting for later translation.
    strong_count = sum(item.get("status") == "ACCEPTED_MICRO" for item in records)
    rejected_count = sum(item.get("status") == "REJECTED_MICRO" for item in records)
    skipped_count = sum(item.get("status") == "SKIPPED" for item in records)
    no_result_count = sum(item.get("status") == "NO_RESULT" for item in records)
    max_correction_ms = (
        max(
            (abs(float(item.get("correction_sec", 0.0))) for item in accepted),
            default=0.0,
        ) * 1000.0
    )

    report_path = output_dir / MICRO_REPORT
    with report_path.open("w", encoding="utf-8") as file:
        file.write("МИКРО-УТОЧНЕНИЕ СИНХРОНИЗАЦИИ ПО ФОРМЕ ВОЛНЫ\n")
        file.write("=" * 78 + "\n\n")
        file.write("Идея: локально сравнивается форма сигнала с окном 5 мс при шаге 1 мс.\n")
        file.write("Это не буквальное сравнение каждого сэмпла: используется устойчивый локальный\n")
        file.write("контур громкости и его изменения, чтобы распознавать тот же фрагмент даже\n")
        file.write("при различиях микса, уровня и содержимого речи.\n\n")
        file.write(f"Точек плотной проверки: {len(records)}\n")
        file.write(f"Сильных микросовпадений: {strong_count}\n")
        file.write(f"Отклонённых микросовпадений: {rejected_count}\n")
        file.write(f"Пропущенных точек: {skipped_count}\n")
        file.write(f"Точек без результата: {no_result_count}\n")
        file.write(f"Участков, допущенных для построения карты: {len(runs)}\n")
        file.write(f"Максимальная принятая микрокоррекция: {max_correction_ms:.3f} мс\n")
        file.write(f"Максимально допустимая микрокоррекция: {MICRO_MAX_CORRECTION_SEC * 1000.0:.1f} мс\n")
        file.write(f"Минимальное реальное улучшение ошибки для принятия точки: {MICRO_MIN_ERROR_IMPROVEMENT_SEC * 1000.0:.1f} мс\n")
        file.write(f"Максимально допустимое расхождение прямой и обратной микропроверки: {MICRO_MAX_REVERSE_DISAGREEMENT_SEC * 1000.0:.1f} мс\n\n")
        file.write("ПРИНЦИП ЗАЩИТЫ:\n")
        file.write("Микрокоррекция не применяется по одной точке. Каждая сильная точка дополнительно\n")
        file.write("проверяется в обратном направлении: найденное смещение должно воспроизводиться\n")
        file.write(f"с расхождением не более {MICRO_MAX_REVERSE_DISAGREEMENT_SEC * 1000.0:.1f} мс. Для изменения карты требуется\n")
        file.write(f"не менее {MICRO_MIN_SUPPORT_POINTS} соседних контрольных точек с согласованным результатом.\n")
        file.write("После этого карта всё равно проходит обычную плотную проверку без ухудшения\n")
        file.write("и проверку уже реально отрендерированного файла.\n\n")
        if run_summaries:
            file.write("ПРИНЯТЫЕ УЧАСТКИ:\n")
            for run in run_summaries:
                file.write(
                    f"  {format_seconds(run['start_sec'])} - {format_seconds(run['end_sec'])}: "
                    f"{run['points']} точек, "
                    f"медианная коррекция {run['median_correction_ms']:+.3f} мс, "
                    f"максимальная {run['max_correction_ms']:.3f} мс\n"
                )
        else:
            file.write("ПРИНЯТЫХ УЧАСТКОВ НЕТ. Основная карта не изменена микропроходом.\n")
        file.write("\nВАЖНО: итоговое решение принимает защита от ухудшения после реального рендера.\n")
        file.write("Исходная целевая дорожка не изменяется.\n")

    # --------------------------------------------------------
    # Строим увеличенные наложения для наиболее сильных точек.
    # --------------------------------------------------------
    try:
        _save_micro_overlays(
            reference_pcm,
            target_pcm,
            accepted,
            output_dir / MICRO_OVERLAY_DIR,
        )
    except Exception as exc:
        with report_path.open("a", encoding="utf-8") as file:
            file.write(f"\nНе удалось построить графические микросовмещения: {exc}\n")

    return {
        "sync_map": new_map,
        "records": records,
        "accepted": accepted,
        "runs": runs,
        "run_summaries": run_summaries,
        "report_path": report_path,
        "csv_path": csv_path,
        "map_csv": map_csv,
    }



def _save_micro_raw_overlay(
    reference_pcm: Path,
    target_pcm: Path,
    target_time_sec: float,
    predicted_offset_sec: float,
    refined_offset_sec: float,
    output_path: Path,
):
    """Сохраняет настоящий увеличенный наложенный фрагмент PCM, только для визуальной диагностики."""
    import matplotlib.pyplot as plt

    sample_rate = ANALYSIS_SAMPLE_RATE
    half_window_sec = 0.060

    target_start = max(0.0, target_time_sec - half_window_sec)
    target_end = target_time_sec + half_window_sec
    reference_center_before = target_time_sec + predicted_offset_sec
    reference_center_after = target_time_sec + refined_offset_sec

    target = _read_pcm_samples(
        target_pcm,
        int(round(target_start * sample_rate)),
        int(round(target_end * sample_rate)),
    )
    if len(target) < 100:
        return

    ref_before_start = max(0.0, reference_center_before - half_window_sec)
    ref_before_end = reference_center_before + half_window_sec
    ref_after_start = max(0.0, reference_center_after - half_window_sec)
    ref_after_end = reference_center_after + half_window_sec

    ref_before = _read_pcm_samples(
        reference_pcm,
        int(round(ref_before_start * sample_rate)),
        int(round(ref_before_end * sample_rate)),
    )
    ref_after = _read_pcm_samples(
        reference_pcm,
        int(round(ref_after_start * sample_rate)),
        int(round(ref_after_end * sample_rate)),
    )

    count = min(len(target), len(ref_before), len(ref_after))
    if count < 100:
        return

    target = target[:count]
    ref_before = ref_before[:count]
    ref_after = ref_after[:count]

    def normalize_wave(values):
        values = np.asarray(values, dtype=np.float64)
        values = values - np.mean(values)
        scale = np.percentile(np.abs(values), 99.0)
        if scale < 1e-9:
            scale = np.max(np.abs(values))
        if scale < 1e-9:
            return values
        return values / scale

    target_n = normalize_wave(target)
    before_n = normalize_wave(ref_before)
    after_n = normalize_wave(ref_after)

    # Около 0.1 мс между точками отображения: так график остаётся
    # читаемым, но мелкий сдвиг формы всё ещё хорошо виден.
    display_step_samples = max(1, int(round(sample_rate * 0.0001)))
    display_indices = np.arange(0, count, display_step_samples, dtype=np.int64)
    target_n = target_n[display_indices]
    before_n = before_n[display_indices]
    after_n = after_n[display_indices]

    x = (display_indices.astype(np.float64) / sample_rate - half_window_sec) * 1000.0
    # Здесь reference переводится в общую шкалу TARGET:
    # t_target = t_reference - offset.
    x_before = x - predicted_offset_sec * 1000.0
    x_after = x - refined_offset_sec * 1000.0

    figure = plt.figure(figsize=(15, 5.5))
    ax = figure.add_subplot(111)
    ax.plot(x, target_n, label="Целевая дорожка")
    ax.plot(
        x_before,
        before_n,
        label=f"Эталон до уточнения ({predicted_offset_sec * 1000.0:+.2f} мс)",
        alpha=0.55,
    )
    ax.plot(
        x_after,
        after_n,
        label=f"Эталон после уточнения ({refined_offset_sec * 1000.0:+.2f} мс)",
        alpha=0.85,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlim(-60.0, 60.0)
    ax.set_xlabel("Относительное время, мс")
    ax.set_ylabel("Нормированная амплитуда")
    ax.set_title(
        f"Сырой сигнал, увеличенный фрагмент: {format_seconds(target_time_sec)} | "
        f"дополнительная коррекция {((refined_offset_sec - predicted_offset_sec) * 1000.0):+.2f} мс"
    )
    ax.grid(True, alpha=0.20)
    ax.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, format="jpg")
    plt.close(figure)


def _save_micro_overlays(
    reference_pcm: Path,
    target_pcm: Path,
    accepted: list[dict],
    output_dir: Path,
):
    """Creates zoomed visual overlays for the most informative micro points."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    if not accepted:
        return

    selected = sorted(
        accepted,
        key=lambda item: (
            float(item.get("score_gain", 0.0)),
            abs(float(item.get("correction_sec", 0.0))),
        ),
        reverse=True,
    )[:MICRO_MAX_OVERLAYS]

    import math as _math
    cols = 2
    rows = int(_math.ceil(len(selected) / cols))
    figure = plt.figure(figsize=(15, max(5, rows * 4.0)))

    for plot_index, item in enumerate(selected, start=1):
        target_time = float(item["target_time_sec"])
        predicted_offset = float(item["predicted_offset_sec"])
        refined_offset = float(item["refined_offset_sec"])

        result = micro_search_at_time(
            reference_pcm,
            target_pcm,
            target_time,
            predicted_offset,
        )
        if result is None:
            continue

        target_feature = _normalize(result["target_feature"])
        reference_feature = result["reference_feature"]

        start = float(result["reference_start_sec"])
        target_start = float(result["target_start_sec"])
        hop = MICRO_FEATURE_HOP_MS / 1000.0
        half = MICRO_LOCAL_WINDOW_SEC
        x = np.arange(len(target_feature), dtype=np.float64) * hop + (target_start - target_time)

        best_start = int(result["best_index"])
        best_ref = reference_feature[best_start : best_start + len(target_feature)]
        best_ref = _normalize(best_ref)

        base_index = max(0, int(round((target_time + predicted_offset - start - half) / hop)))
        base_ref = reference_feature[base_index : base_index + len(target_feature)]
        if len(base_ref) != len(target_feature):
            base_ref = best_ref
        else:
            base_ref = _normalize(base_ref)

        ax = figure.add_subplot(rows, cols, plot_index)
        ax.plot(x * 1000.0, target_feature, label="Целевая дорожка")
        ax.plot(x * 1000.0, base_ref, label="Эталон до уточнения", alpha=0.65)
        ax.plot(x * 1000.0, best_ref, label="Эталон после уточнения", alpha=0.85)
        ax.axvline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlim(-250.0, 250.0)
        ax.set_title(
            f"{format_seconds(target_time)} | "
            f"коррекция {item['correction_sec'] * 1000.0:+.2f} мс | "
            f"выигрыш {item['score_gain']:+.3f}"
        )
        ax.set_xlabel("Относительное время, мс")
        ax.set_ylabel("Нормированный контур")
        ax.grid(True, alpha=0.20)
        ax.legend(loc="best", fontsize=8)

    overview_path = Path(output_dir).parent / MICRO_OVERVIEW_JPEG
    figure.tight_layout()
    figure.savefig(overview_path, dpi=160, format="jpg")
    plt.close(figure)

    for index, item in enumerate(selected, start=1):
        single = plt.figure(figsize=(14, 5))
        result = micro_search_at_time(
            reference_pcm,
            target_pcm,
            float(item["target_time_sec"]),
            float(item["predicted_offset_sec"]),
        )
        if result is None:
            plt.close(single)
            continue

        target_feature = _normalize(result["target_feature"])
        best_start = int(result["best_index"])
        best_ref = _normalize(
            result["reference_feature"][best_start : best_start + len(target_feature)]
        )
        x = np.arange(len(target_feature), dtype=np.float64) * (MICRO_FEATURE_HOP_MS / 1000.0)
        x *= 1000.0
        x -= MICRO_LOCAL_WINDOW_SEC * 1000.0

        ax = single.add_subplot(111)
        ax.plot(x, target_feature, label="Целевая дорожка")
        ax.plot(x, best_ref, label="Эталон после микросовмещения")
        ax.axvline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlim(-250, 250)
        ax.set_xlabel("Относительное время, мс")
        ax.set_ylabel("Нормированный контур")
        ax.set_title(
            f"Микроскоп формы волны: {format_seconds(float(item['target_time_sec']))} | "
            f"смещение {float(item['correction_sec']) * 1000.0:+.2f} мс"
        )
        ax.grid(True, alpha=0.20)
        ax.legend(loc="best")
        single.tight_layout()
        single_path = Path(output_dir) / f"micro_overlay_{index:02d}_{float(item['target_time_sec']):09.3f}s.jpg"
        single.savefig(
            single_path,
            dpi=160,
            format="jpg",
        )
        plt.close(single)

        raw_dir = Path(output_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        _save_micro_raw_overlay(
            reference_pcm,
            target_pcm,
            float(item["target_time_sec"]),
            float(item["predicted_offset_sec"]),
            float(item["refined_offset_sec"]),
            raw_dir / f"micro_raw_overlay_{index:02d}_{float(item['target_time_sec']):09.3f}s.jpg",
        )
