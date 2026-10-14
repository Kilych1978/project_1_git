# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: validator.py
# ============================================================
# Independent validation, bidirectional checks, robust scoring, outliers and application-readiness checks.
#
# This module is part of the modular version of the project.
# Do not run it directly; start the pipeline with start_sync.py.
# ============================================================

from __future__ import annotations

import csv
import json
import math
import time
import numpy as np

from config import *
from common import *
from analyzer import *
from mapper import *
from aligner import _refine_offset_interpolated

VALIDATION_WINDOW_SEC = 20.0

VALIDATION_INTERVAL_SEC = 60.0

VALIDATION_MAX_SHIFT_SEC = 3.0

VALIDATION_SEARCH_STEP_SEC = 0.01

# Тонкое уточнение offset после грубого 10 ms поиска.
# Это позволяет различать небольшие реальные сдвиги, например 18 ms.
FINE_OFFSET_REFINE_RADIUS_SEC = 0.020
FINE_OFFSET_REFINE_STEP_SEC = 0.001


VALIDATION_BEFORE_AFTER_JPEG = (
    "08_BEFORE_AFTER_ALIGNMENT.jpg"
)

VALIDATION_ERROR_JPEG = (
    "09_ALIGNMENT_ERROR.jpg"
)

VALIDATION_JSON = (
    "alignment_validation.json"
)

VALIDATION_REPORT = (
    "ALIGNMENT_VALIDATION_REPORT.txt"
)


# ============================================================
# ИНТЕРПОЛЯЦИЯ OFFSET ИЗ КАРТЫ
# ============================================================

def map_offset_at_time(
    sync_map,
    target_time_sec,
):
    """
    Возвращает offset для конкретного Target time.
    """

    map_time = np.asarray(
        sync_map[
            "target_time"
        ],
        dtype=np.float64,
    )

    map_offset = np.asarray(
        sync_map[
            "offset"
        ],
        dtype=np.float64,
    )

    if len(map_time) == 0:
        return 0.0

    return float(
        np.interp(
            target_time_sec,
            map_time,
            map_offset,
        )
    )


# ============================================================
# ПОИСК ЛУЧШЕГО OFFSET ДЛЯ ВАЛИДАЦИИ
# ============================================================

def validation_search_offset(
    reference_envelope,
    target_envelope,
    target_center_sec,
    expected_offset_sec,
    window_sec,
    search_radius_sec,
    search_step_sec,
    hop_sec,
    search_center_offset_sec=None,
):
    """
    Независимый повторный поиск offset.

    Важное свойство v6.7: expected_offset_sec НЕ задаёт центр поиска.
    По умолчанию поиск выполняется вокруг нулевой коррекции, а candidate
    используется только для последующего сравнения. Это предотвращает
    самоподтверждение ложной карты.

    Грубый поиск остаётся на исходном шаге validation, после чего
    лучший кандидат уточняется с шагом 1 ms. Это уменьшает квантование
    результатов вокруг малых реальных расхождений.
    """
    target_segment = envelope_slice(
        target_envelope,
        target_center_sec,
        window_sec,
        hop_sec,
    )

    if len(target_segment) < 10:
        return {
            "offset": None,
            "correlation": 0.0,
        }

    target_norm = normalize_segment(target_segment)
    target_norm_value = np.linalg.norm(target_norm)
    if target_norm_value < 1e-12:
        return {
            "offset": None,
            "correlation": 0.0,
        }

    # КРИТИЧЕСКИ ВАЖНО: expected_offset_sec — только ожидаемое
    # значение, с которым позже сравнивается независимый результат.
    # Сам поиск по умолчанию всегда выполняется вокруг НУЛЯ.
    # Это не позволяет candidate map "подсказывать" валидатору,
    # где искать подтверждение собственного offset.
    if search_center_offset_sec is None:
        search_center_offset_sec = 0.0

    offsets = np.arange(
        float(search_center_offset_sec) - search_radius_sec,
        float(search_center_offset_sec) + search_radius_sec + search_step_sec * 0.5,
        search_step_sec,
        dtype=np.float64,
    )

    best_offset = None
    best_corr = -2.0
    target_length = len(target_segment)

    for offset in offsets:
        reference_center = target_center_sec + offset
        reference_segment = envelope_slice(
            reference_envelope,
            reference_center,
            window_sec,
            hop_sec,
        )

        if len(reference_segment) != target_length:
            continue

        reference_norm = normalize_segment(reference_segment)
        if np.linalg.norm(reference_norm) < 1e-12:
            continue

        correlation = float(np.dot(reference_norm, target_norm))
        if correlation > best_corr:
            best_corr = correlation
            best_offset = float(offset)

    if best_offset is None:
        return {
            "offset": None,
            "correlation": 0.0,
        }

    refined_offset, refined_corr = _refine_offset_interpolated(
        reference_envelope,
        target_envelope,
        target_center_sec,
        best_offset,
        window_sec,
        hop_sec,
    )

    if refined_corr >= best_corr:
        best_offset = refined_offset
        best_corr = refined_corr

    return {
        "offset": float(best_offset),
        "correlation": float(best_corr),
    }


# ============================================================
# VALIDATION POINTS
# ============================================================

def build_validation_points(
    target_duration,
    interval_sec=None,
):
    """
    Формируем независимые точки проверки.

    Используем сетку, отличную от локальных
    30-секундных якорей.

    Это важно:
    иначе мы будем проверять карту почти
    на тех же точках, из которых она была построена.
    """

    half_window = (
        VALIDATION_WINDOW_SEC
        / 2.0
    )

    start = (
        half_window
        + VALIDATION_MAX_SHIFT_SEC
        + 5.0
    )

    end = (
        target_duration
        - half_window
        - VALIDATION_MAX_SHIFT_SEC
        - 5.0
    )

    if end <= start:

        raise RuntimeError(
            "Недостаточно времени "
            "для независимой validation-сетки."
        )

    if interval_sec is None:
        interval_sec = VALIDATION_INTERVAL_SEC
    interval_sec = max(1.0, float(interval_sec))

    points = np.arange(
        start,
        end + 0.001,
        interval_sec,
        dtype=np.float64,
    )

    return points


# ============================================================
# НЕЗАВИСИМАЯ ВАЛИДАЦИЯ
# ============================================================

def bidirectional_validation_check(
    reference_envelope,
    target_envelope,
    target_center_sec,
    expected_offset_sec,
    hop_sec,
):
    """
    Проверяет candidate offset независимо в двух соседних половинах
    исходного validation-окна:

        LEFT  = [center-10, center]
        RIGHT = [center, center+10]

    Это НЕ зеркальный повтор того же самого расчёта: поиск выполняется
    по двум разным временным подокнам. Ложный пик, который возникает
    только из-за случайного совпадения речи/шума, часто не сохраняет
    один и тот же offset в обеих половинах.
    """

    half = BIDIR_SIDE_WINDOW_SEC
    left_center = target_center_sec - half / 2.0
    right_center = target_center_sec + half / 2.0

    left = validation_search_offset(
        reference_envelope,
        target_envelope,
        left_center,
        expected_offset_sec,
        half,
        VALIDATION_MAX_SHIFT_SEC,
        VALIDATION_SEARCH_STEP_SEC,
        hop_sec,
        search_center_offset_sec=0.0,
    )

    right = validation_search_offset(
        reference_envelope,
        target_envelope,
        right_center,
        expected_offset_sec,
        half,
        VALIDATION_MAX_SHIFT_SEC,
        VALIDATION_SEARCH_STEP_SEC,
        hop_sec,
        search_center_offset_sec=0.0,
    )

    left_offset = left["offset"]
    right_offset = right["offset"]
    left_corr = float(left["correlation"])
    right_corr = float(right["correlation"])

    result = {
        "left_offset": left_offset,
        "left_correlation": left_corr,
        "right_offset": right_offset,
        "right_correlation": right_corr,
        "consensus_offset": None,
        "consensus_correlation": 0.0,
        "disagreement": None,
        "prediction_error": None,
        "status": "UNRESOLVED",
    }

    if left_offset is None or right_offset is None:
        result["status"] = "ONE_SIDE_MISSING"
        return result

    disagreement = abs(float(left_offset) - float(right_offset))
    consensus_offset = (float(left_offset) + float(right_offset)) / 2.0
    consensus_corr = (left_corr + right_corr) / 2.0
    prediction_error = abs(consensus_offset - expected_offset_sec)

    result["consensus_offset"] = consensus_offset
    result["consensus_correlation"] = consensus_corr
    result["disagreement"] = disagreement
    result["prediction_error"] = prediction_error

    if (
        disagreement <= BIDIR_MAX_DISAGREEMENT_SEC
        and consensus_corr >= BIDIR_MIN_CORRELATION
    ):
        if prediction_error <= BIDIR_MAX_PREDICTION_ERROR_SEC:
            result["status"] = "CONFIRMED_MAP"
        elif consensus_corr >= BIDIR_STRONG_CORRELATION:
            result["status"] = "CONFIRMED_DISCREPANCY"
        else:
            result["status"] = "CONSISTENT_BUT_WEAK"
    else:
        result["status"] = "AMBIGUOUS"

    return result


# ============================================================
# НЕЗАВИСИМАЯ ВАЛИДАЦИЯ
# ============================================================

def validate_sync_map(
    analysis,
    sync_map,
    target_duration,
    progress,
    validation_interval_sec=None,
):
    """
    Независимая validation карты.

    Сначала выполняется старый 20-секундный полный поиск. Затем
    кандидат проверяется отдельно на левой и правой половине окна.
    Для итогового after_offset при согласованной двусторонней проверке
    используется их консенсус. Это защищает карту от ложных одиночных
    совпадений, которые раньше превращались в HARD outliers.
    """

    progress.stage(
        "НЕЗАВИСИМАЯ ПРОВЕРКА КАРТЫ",
        0.0,
    )

    reference_envelope = analysis["reference_envelope"]["values"]
    target_envelope = analysis["target_envelope"]["values"]
    hop_sec = analysis["target_envelope"]["hop_sec"]

    validation_times = build_validation_points(
        target_duration,
        interval_sec=validation_interval_sec,
    )
    results = []
    total = len(validation_times)
    started = time.time()

    for index, target_time in enumerate(validation_times, start=1):
        predicted_offset = map_offset_at_time(sync_map, target_time)

        before = validation_search_offset(
            reference_envelope,
            target_envelope,
            target_time,
            0.0,
            VALIDATION_WINDOW_SEC,
            VALIDATION_MAX_SHIFT_SEC,
            VALIDATION_SEARCH_STEP_SEC,
            hop_sec,
        )

        # ----------------------------------------------------
        # SYMMETRIC BASELINE CHECK
        # ----------------------------------------------------
        #
        # До v6.5 BEFORE и AFTER оценивались разными процедурами:
        # BEFORE — одним полным 20-секундным поиском,
        # AFTER   — с дополнительной двусторонней проверкой.
        #
        # Из-за этого даже НУЛЕВАЯ карта могла формально
        # выглядеть как ухудшение: один и тот же TARGET
        # проверялся разными измерителями.
        #
        # Теперь baseline получает ТОЧНО такую же bidirectional
        # проверку, как и candidate. Поэтому:
        #
        #     zero-map BEFORE == zero-map AFTER
        #
        # с точностью до численной погрешности поиска.
        # Это принципиально для non-regression guard.
        before_bidir = bidirectional_validation_check(
            reference_envelope,
            target_envelope,
            target_time,
            0.0,
            hop_sec,
        )

        raw_after = validation_search_offset(
            reference_envelope,
            target_envelope,
            target_time,
            predicted_offset,
            VALIDATION_WINDOW_SEC,
            VALIDATION_MAX_SHIFT_SEC,
            VALIDATION_SEARCH_STEP_SEC,
            hop_sec,
            search_center_offset_sec=0.0,
        )

        bidir = bidirectional_validation_check(
            reference_envelope,
            target_envelope,
            target_time,
            predicted_offset,
            hop_sec,
        )

        # При согласованной двусторонней проверке именно консенсус
        # считается независимой оценкой фактического offset.
        if bidir["consensus_offset"] is not None and bidir["status"] in {
            "CONFIRMED_MAP",
            "CONFIRMED_DISCREPANCY",
            "CONSISTENT_BUT_WEAK",
        }:
            final_after_offset = float(bidir["consensus_offset"])
            final_after_corr = float(bidir["consensus_correlation"])
        else:
            final_after_offset = raw_after["offset"]
            final_after_corr = float(raw_after["correlation"])

        # Для non-regression и статистики BEFORE используем
        # bidirectional baseline. Старый full-window результат
        # сохраняем отдельно только как диагностический legacy offset.
        if before_bidir["consensus_offset"] is not None:
            baseline_offset_for_metrics = float(
                before_bidir["consensus_offset"]
            )
        else:
            baseline_offset_for_metrics = before["offset"]

        if baseline_offset_for_metrics is not None and final_after_offset is not None:
            before_error = abs(float(baseline_offset_for_metrics))
            after_error = abs(float(final_after_offset) - predicted_offset)
        else:
            before_error = None
            after_error = None

        results.append({
            "target_time": float(target_time),
            "predicted_offset": float(predicted_offset),
            "before_offset": before["offset"],
            "before_correlation": float(before["correlation"]),
            "before_bidirectional_offset": before_bidir["consensus_offset"],
            "before_bidirectional_correlation": before_bidir["consensus_correlation"],
            "before_bidirectional_status": before_bidir["status"],
            "before_error_raw": (
                None
                if before["offset"] is None
                else abs(float(before["offset"]))
            ),
            "after_offset": final_after_offset,
            "after_correlation": final_after_corr,
            "before_error": before_error,
            "after_error": after_error,
            "raw_after_offset": raw_after["offset"],
            "raw_after_correlation": float(raw_after["correlation"]),
            "bidir_left_offset": bidir["left_offset"],
            "bidir_left_correlation": bidir["left_correlation"],
            "bidir_right_offset": bidir["right_offset"],
            "bidir_right_correlation": bidir["right_correlation"],
            "bidir_consensus_offset": bidir["consensus_offset"],
            "bidir_consensus_correlation": bidir["consensus_correlation"],
            "bidir_disagreement": bidir["disagreement"],
            "bidir_prediction_error": bidir["prediction_error"],
            "bidir_status": bidir["status"],
        })

        percent = index / total * 100.0
        elapsed = time.time() - started
        speed = index / max(0.001, elapsed)
        remaining = (total - index) / speed if speed > 0 else 0.0

        progress.update(
            percent,
            (
                "Validation | "
                f"t={target_time:.0f}s | "
                f"pred={predicted_offset:+.3f}s | "
                f"ETA={format_seconds(remaining)}"
            ),
        )

    print()
    progress.done(100.0, "Независимая проверка завершена")
    return results


# ============================================================
# СТАТИСТИКА VALIDATION
# ============================================================

def calculate_validation_metrics(
    results,
):
    """
    Сводит independent validation в понятные цифры.
    """

    before_errors = np.asarray(
        [
            r["before_error"]
            for r in results
            if r["before_error"]
            is not None
        ],
        dtype=np.float64,
    )

    after_errors = np.asarray(
        [
            r["after_error"]
            for r in results
            if r["after_error"]
            is not None
        ],
        dtype=np.float64,
    )

    before_corr = np.asarray(
        [
            r["before_correlation"]
            for r in results
            if r[
                "before_correlation"
            ]
            is not None
        ],
        dtype=np.float64,
    )

    after_corr = np.asarray(
        [
            r["after_correlation"]
            for r in results
            if r[
                "after_correlation"
            ]
            is not None
        ],
        dtype=np.float64,
    )

    bidir_statuses = [
        result.get("bidir_status", "UNRESOLVED")
        for result in results
    ]

    metrics = {
        "points":
            len(results),

        "valid_before":
            int(
                len(before_errors)
            ),

        "valid_after":
            int(
                len(after_errors)
            ),

        "bidir_confirmed_map":
            int(
                sum(
                    status == "CONFIRMED_MAP"
                    for status in bidir_statuses
                )
            ),

        "bidir_confirmed_discrepancy":
            int(
                sum(
                    status == "CONFIRMED_DISCREPANCY"
                    for status in bidir_statuses
                )
            ),

        "bidir_ambiguous":
            int(
                sum(
                    status not in {
                        "CONFIRMED_MAP",
                        "CONFIRMED_DISCREPANCY",
                        "CONSISTENT_BUT_WEAK",
                    }
                    for status in bidir_statuses
                )
            ),
    }

    if len(before_errors):

        metrics.update(
            {
                "before_mean_error":
                    float(
                        np.mean(
                            before_errors
                        )
                    ),

                "before_median_error":
                    float(
                        np.median(
                            before_errors
                        )
                    ),

                "before_max_error":
                    float(
                        np.max(
                            before_errors
                        ),
                    ),

                "before_p95_error":
                    float(
                        np.percentile(
                            before_errors,
                            95,
                        )
                    ),

                "before_mean_correlation":
                    float(
                        np.mean(
                            before_corr
                        )
                    ),

                "before_median_correlation":
                    float(
                        np.median(
                            before_corr
                        )
                    ),
            }
        )

    else:

        metrics.update(
            {
                "before_mean_error":
                    None,

                "before_median_error":
                    None,

                "before_max_error":
                    None,

                "before_mean_correlation":
                    None,

                "before_median_correlation":
                    None,
            }
        )

    if len(after_errors):

        metrics.update(
            {
                "after_mean_error":
                    float(
                        np.mean(
                            after_errors
                        )
                    ),

                "after_median_error":
                    float(
                        np.median(
                            after_errors
                        )
                    ),

                "after_max_error":
                    float(
                        np.max(
                            after_errors
                        )
                    ),

                "after_p95_error":
                    float(
                        np.percentile(
                            after_errors,
                            95,
                        )
                    ),

                "after_mean_correlation":
                    float(
                        np.mean(
                            after_corr
                        )
                    ),

                "after_median_correlation":
                    float(
                        np.median(
                            after_corr
                        )
                    ),
            }
        )

    else:

        metrics.update(
            {
                "after_mean_error":
                    None,

                "after_median_error":
                    None,

                "after_max_error":
                    None,

                "after_mean_correlation":
                    None,

                "after_median_correlation":
                    None,
            }
        )

    # --------------------------------------------------------
    # Насколько улучшилось среднее положение.
    # --------------------------------------------------------

    if (
        metrics[
            "before_mean_error"
        ] is not None
        and
        metrics[
            "after_mean_error"
        ] is not None
        and
        metrics[
            "before_mean_error"
        ] > 1e-12
    ):

        improvement = (
            1.0
            -
            (
                metrics[
                    "after_mean_error"
                ]
                /
                metrics[
                    "before_mean_error"
                ]
            )
        )

    else:

        improvement = None

    metrics[
        "error_improvement_ratio"
    ] = (
        None
        if improvement is None
        else float(
            improvement
        )
    )

    # --------------------------------------------------------
    # Улучшение корреляции.
    # --------------------------------------------------------

    if (
        metrics[
            "before_mean_correlation"
        ] is not None
        and
        metrics[
            "after_mean_correlation"
        ] is not None
    ):

        metrics[
            "correlation_gain"
        ] = float(
            metrics[
                "after_mean_correlation"
            ]
            -
            metrics[
                "before_mean_correlation"
            ]
        )

    else:

        metrics[
            "correlation_gain"
        ] = None

    # Доля точек в разных диапазонах ошибки — более наглядная метрика,
    # чем один общий score.
    thresholds = (
        0.005,
        0.010,
        0.020,
        0.050,
        0.100,
        0.250,
    )

    for prefix, errors in ((
        "before", before_errors,
    ), (
        "after", after_errors,
    )):
        values = np.asarray(errors, dtype=np.float64) if len(errors) else np.asarray([], dtype=np.float64)
        total = len(values)
        for threshold in thresholds:
            key = f"{prefix}_within_{int(round(threshold * 1000))}ms_percent"
            metrics[key] = (
                float(np.mean(values <= threshold) * 100.0)
                if total
                else None
            )

    return metrics


# ============================================================
# СВОДКА BIDIRECTIONAL VALIDATION
# ============================================================

def summarize_bidirectional_validation(results):
    """Сводка подтверждений bidirectional validation и HARD-кандидатов."""
    statuses=[str(item.get("bidir_status","UNRESOLVED")) for item in results]
    confirmed_map=sum(status=="CONFIRMED_MAP" for status in statuses)
    confirmed_discrepancy=sum(status=="CONFIRMED_DISCREPANCY" for status in statuses)
    consistent_but_weak=sum(status=="CONSISTENT_BUT_WEAK" for status in statuses)
    ambiguous=sum(status in {"AMBIGUOUS","ONE_SIDE_MISSING","UNRESOLVED"} for status in statuses)
    candidates=[item for item in find_validation_outliers(results) if item.get("severity")=="HARD"]
    hard_confirmed=sum(item.get("bidir_status")=="CONFIRMED_DISCREPANCY" for item in candidates)
    hard_ambiguous=sum(item.get("bidir_status") in {"AMBIGUOUS","ONE_SIDE_MISSING","UNRESOLVED","CONSISTENT_BUT_WEAK"} for item in candidates)
    return {
        "confirmed_map": int(confirmed_map),
        "confirmed_discrepancy": int(confirmed_discrepancy),
        "consistent_but_weak": int(consistent_but_weak),
        "ambiguous": int(ambiguous),
        "hard_candidates": int(len(candidates)),
        "hard_confirmed_discrepancy": int(hard_confirmed),
        "hard_ambiguous": int(hard_ambiguous),
    }


# ============================================================
# ГРАФИК BEFORE / AFTER
# ============================================================

def save_before_after_jpeg(
    output_dir,
    results,
    progress,
):
    """
    Самый наглядный график validation.

    Показываем:

        predicted map
        actual BEFORE offset
        actual AFTER offset

    Если карта хорошая,
    AFTER должен идти близко к нулю
    относительно prediction.
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ГРАФИК BEFORE / AFTER",
        0.0,
    )

    path = (
        output_dir
        / VALIDATION_BEFORE_AFTER_JPEG
    )

    times = np.asarray(
        [
            r["target_time"]
            for r in results
        ],
        dtype=np.float64,
    )

    predicted = np.asarray(
        [
            r["predicted_offset"]
            for r in results
        ],
        dtype=np.float64,
    )

    before = np.asarray(
        [
            (
                r["before_offset"]
                if r["before_offset"]
                is not None
                else np.nan
            )
            for r in results
        ],
        dtype=np.float64,
    )

    after_residual = np.asarray(
        [
            (
                (
                    r["after_offset"]
                    - r["predicted_offset"]
                )
                if (
                    r["after_offset"]
                    is not None
                )
                else np.nan
            )
            for r in results
        ],
        dtype=np.float64,
    )

    fig = plt.figure(
        figsize=(18, 10),
        dpi=160,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    ax.plot(
        times,
        predicted,
        linewidth=2.0,
        label="Predicted map",
    )

    ax.scatter(
        times,
        before,
        s=12,
        alpha=0.55,
        label="Actual offset BEFORE",
    )

    ax.scatter(
        times,
        after_residual,
        s=14,
        alpha=0.70,
        label="Residual AFTER",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title(
        "Independent Alignment Validation"
    )

    ax.set_xlabel(
        "Target time, seconds"
    )

    ax.set_ylabel(
        "Offset / residual, seconds"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        format="jpg",
        dpi=160,
    )

    plt.close(
        fig
    )

    progress.done(
        100.0,
        f"Validation graph: {path.name}",
    )

    return path


# ============================================================
# ГРАФИК ОШИБКИ
# ============================================================

def save_alignment_error_jpeg(
    output_dir,
    results,
    progress,
):
    """
    Показывает абсолютную ошибку:

        BEFORE
        AFTER
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ГРАФИК ОШИБКИ",
        0.0,
    )

    path = (
        output_dir
        / VALIDATION_ERROR_JPEG
    )

    times = np.asarray(
        [
            r["target_time"]
            for r in results
        ],
        dtype=np.float64,
    )

    before_error = np.asarray(
        [
            (
                r["before_error"]
                if r["before_error"]
                is not None
                else np.nan
            )
            for r in results
        ],
        dtype=np.float64,
    )

    after_error = np.asarray(
        [
            (
                r["after_error"]
                if r["after_error"]
                is not None
                else np.nan
            )
            for r in results
        ],
        dtype=np.float64,
    )

    fig = plt.figure(
        figsize=(18, 9),
        dpi=160,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    ax.plot(
        times,
        before_error,
        linewidth=1.2,
        label="Before absolute error",
    )

    ax.plot(
        times,
        after_error,
        linewidth=1.5,
        label="After residual error",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title(
        "Alignment Error Before / After"
    )

    ax.set_xlabel(
        "Target time, seconds"
    )

    ax.set_ylabel(
        "Absolute error, seconds"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        format="jpg",
        dpi=160,
    )

    plt.close(
        fig
    )

    progress.done(
        100.0,
        f"Error graph: {path.name}",
    )

    return path


# ============================================================
# ОТЧЁТ VALIDATION
# ============================================================

def save_validation_report(
    output_dir,
    metrics,
):
    """
    Читаемый отчёт.
    """

    path = (
        output_dir
        / VALIDATION_REPORT
    )

    lines = []

    lines.append(
        "INDEPENDENT AUDIO SYNC VALIDATION"
    )

    lines.append(
        "=" * 78
    )

    lines.append(
        ""
    )

    lines.append(
        f"Validation points: "
        f"{metrics['points']}"
    )

    lines.append(
        f"Valid BEFORE: "
        f"{metrics['valid_before']}"
    )

    lines.append(
        f"Valid AFTER: "
        f"{metrics['valid_after']}"
    )

    lines.append(
        ""
    )

    lines.append(
        "BEFORE:"
    )

    lines.append(
        "  Baseline measurement: symmetric bidirectional validation"
    )

    lines.append(
        f"  Mean error: "
        f"{metrics['before_mean_error']}"
    )

    lines.append(
        f"  Median error: "
        f"{metrics['before_median_error']}"
    )

    lines.append(
        f"  Max error: "
        f"{metrics['before_max_error']}"
    )

    lines.append(
        f"  Mean correlation: "
        f"{metrics['before_mean_correlation']}"
    )

    lines.append(
        f"  Median correlation: "
        f"{metrics['before_median_correlation']}"
    )

    lines.append(
        ""
    )

    lines.append(
        "AFTER:"
    )

    lines.append(
        f"  Mean residual: "
        f"{metrics['after_mean_error']}"
    )

    lines.append(
        f"  Median residual: "
        f"{metrics['after_median_error']}"
    )

    lines.append(
        f"  Max residual: "
        f"{metrics['after_max_error']}"
    )

    lines.append(
        f"  Mean correlation: "
        f"{metrics['after_mean_correlation']}"
    )

    lines.append(
        f"  Median correlation: "
        f"{metrics['after_median_correlation']}"
    )

    lines.append(
        ""
    )

    lines.append(
        f"Error improvement: "
        f"{metrics['error_improvement_ratio']}"
    )

    lines.append(
        f"Correlation gain: "
        f"{metrics['correlation_gain']}"
    )

    lines.append(
        ""
    )

    lines.append(
        "BIDIRECTIONAL CHECK:"
    )

    lines.append(
        f"  Confirmed map: {metrics.get('bidir_confirmed_map', 0)}"
    )

    lines.append(
        f"  Confirmed discrepancy: {metrics.get('bidir_confirmed_discrepancy', 0)}"
    )

    lines.append(
        f"  Ambiguous: {metrics.get('bidir_ambiguous', 0)}"
    )

    lines.append(
        ""
    )

    lines.append(
        "=" * 78
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )

    return path


# ============================================================
# СОХРАНЕНИЕ JSON
# ============================================================

def save_validation_json(
    output_dir,
    results,
    metrics,
):
    path = (
        output_dir
        / VALIDATION_JSON
    )

    data = {
        "metrics":
            metrics,

        "points":
            results,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path


# ============================================================
# ФИНАЛЬНЫЙ ВЕРДИКТ VALIDATION
# ============================================================

def evaluate_validation(
    metrics,
):
    """
    Решаем, действительно ли карта
    улучшила независимое сопоставление.

    Здесь сознательно используем независимую проверку,
    а не исходный score карты.
    """

    before_error = (
        metrics[
            "before_mean_error"
        ]
    )

    after_error = (
        metrics[
            "after_mean_error"
        ]
    )

    correlation_gain = (
        metrics[
            "correlation_gain"
        ]
    )

    improvement = (
        metrics[
            "error_improvement_ratio"
        ]
    )

    if (
        before_error is None
        or after_error is None
    ):

        return {
            "verdict": "FAILED",
            "score": 0.0,
            "reason":
                "Не удалось провести "
                "независимую проверку.",
        }

    score = 0.0

    # --------------------------------------------------------
    # Улучшение ошибки
    # --------------------------------------------------------

    if improvement is not None:

        if improvement >= 0.90:
            score += 55

        elif improvement >= 0.75:
            score += 45

        elif improvement >= 0.50:
            score += 35

        elif improvement >= 0.25:
            score += 20

    # --------------------------------------------------------
    # Абсолютная остаточная ошибка
    # --------------------------------------------------------

    if after_error <= 0.010:
        score += 30

    elif after_error <= 0.025:
        score += 25

    elif after_error <= 0.050:
        score += 18

    elif after_error <= 0.100:
        score += 10

    # --------------------------------------------------------
    # Корреляция
    # --------------------------------------------------------

    if correlation_gain is not None:

        if correlation_gain >= 0.20:
            score += 15

        elif correlation_gain >= 0.10:
            score += 12

        elif correlation_gain >= 0.05:
            score += 8

        elif correlation_gain >= 0.02:
            score += 4

    if score >= 85:

        verdict = "EXCELLENT"

    elif score >= 70:

        verdict = "GOOD"

    elif score >= 50:

        verdict = "ACCEPTABLE"

    elif score >= 30:

        verdict = "SUSPICIOUS"

    else:

        verdict = "BAD"

    return {
        "verdict":
            verdict,

        "score":
            float(score),

        "reason":
            (
                "Карта прошла "
                "независимую проверку."
            ),
    }


# ============================================================
# ПЕЧАТЬ VALIDATION В КОНСОЛЬ
# ============================================================

def print_validation_result(
    metrics,
    verdict,
    output_dir,
):
    print()
    print("=" * 78)
    print("НЕЗАВИСИМАЯ ПРОВЕРКА КАРТЫ")
    print("=" * 78)

    print()

    print(
        "Validation verdict:"
    )

    print(
        f"  {verdict['verdict']}"
    )

    print(
        f"  Score: "
        f"{verdict['score']:.1f}/100"
    )

    print()

    print(
        "BEFORE:"
    )

    print(
        "  Mean error:",
        f"{metrics['before_mean_error']:.6f} s",
    )

    print(
        "  Median error:",
        f"{metrics['before_median_error']:.6f} s",
    )

    print(
        "  Max error:",
        f"{metrics['before_max_error']:.6f} s",
    )

    print(
        "  Mean correlation:",
        f"{metrics['before_mean_correlation']:.6f}",
    )

    print()

    print(
        "AFTER:"
    )

    print(
        "  Mean residual:",
        f"{metrics['after_mean_error']:.6f} s",
    )

    print(
        "  Median residual:",
        f"{metrics['after_median_error']:.6f} s",
    )

    print(
        "  Max residual:",
        f"{metrics['after_max_error']:.6f} s",
    )

    print(
        "  Mean correlation:",
        f"{metrics['after_mean_correlation']:.6f}",
    )

    print()

    if (
        metrics[
            "error_improvement_ratio"
        ]
        is not None
    ):

        print(
            "Error improvement:",
            f"{metrics['error_improvement_ratio'] * 100:.2f}%",
        )

    if (
        metrics[
            "correlation_gain"
        ]
        is not None
    ):

        print(
            "Correlation gain:",
            f"{metrics['correlation_gain']:+.6f}",
        )

    print()

    print(
        "Результаты сохранены в:"
    )

    print(
        output_dir
    )

    print()

# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC
#
# ЧАСТЬ 8
#
# VALIDATION OUTLIERS
# + ROBUST FINAL MAP
# + FINAL PRE-APPLY CHECK
#
# ============================================================


# ============================================================
# НАСТРОЙКИ ЗАЩИТЫ КАРТЫ
# ============================================================

# Если независимая проверка отличается от карты
# больше этого значения — точка считается подозрительной.
FINAL_VALIDATION_OUTLIER_SEC = 0.25


# Очень сильный выброс.
FINAL_VALIDATION_HARD_OUTLIER_SEC = 0.75


# Минимальное количество соседних точек,
# необходимых для локальной проверки.
FINAL_NEIGHBOR_COUNT = 3


# Насколько далеко от соседнего тренда может отклониться
# точка, прежде чем мы перестанем ей доверять.
FINAL_NEIGHBOR_TOLERANCE_SEC = 0.20


# Названия файлов.
FINAL_VALIDATION_CSV = (
    "final_validation_points.csv"
)

FINAL_VALIDATION_OUTLIERS_CSV = (
    "validation_outliers.csv"
)

FINAL_ROBUST_MAP_CSV = (
    "final_robust_sync_map.csv"
)

FINAL_ROBUST_MAP_JPEG = (
    "10_FINAL_ROBUST_MAP.jpg"
)

FINAL_ROBUST_REPORT = (
    "FINAL_ROBUST_MAP_REPORT.txt"
)


# ============================================================
# СОХРАНЕНИЕ ВСЕХ VALIDATION ТОЧЕК
# ============================================================

def save_validation_points_csv(
    output_dir,
    results,
    progress,
):
    """Сохраняет все validation точки и двустороннюю диагностику."""

    progress.stage(
        "СОХРАНЕНИЕ VALIDATION ТОЧЕК",
        0.0,
    )

    path = output_dir / FINAL_VALIDATION_CSV

    columns = [
        "target_time_sec",
        "predicted_offset_sec",
        "before_offset_sec",
        "before_correlation",
        "after_offset_sec",
        "after_correlation",
        "before_error_sec",
        "after_residual_sec",
        "raw_after_offset_sec",
        "raw_after_correlation",
        "bidir_left_offset_sec",
        "bidir_left_correlation",
        "bidir_right_offset_sec",
        "bidir_right_correlation",
        "bidir_consensus_offset_sec",
        "bidir_consensus_correlation",
        "bidir_disagreement_sec",
        "bidir_prediction_error_sec",
        "bidir_status",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(columns)

        total = len(results)

        for index, result in enumerate(results, start=1):
            def fmt(name, digits=6):
                value = result.get(name)
                if value is None:
                    return ""
                return f"{float(value):.{digits}f}"

            writer.writerow([
                fmt("target_time"),
                fmt("predicted_offset"),
                fmt("before_offset"),
                fmt("before_correlation"),
                fmt("after_offset"),
                fmt("after_correlation"),
                fmt("before_error"),
                fmt("after_error"),
                fmt("raw_after_offset"),
                fmt("raw_after_correlation"),
                fmt("bidir_left_offset"),
                fmt("bidir_left_correlation"),
                fmt("bidir_right_offset"),
                fmt("bidir_right_correlation"),
                fmt("bidir_consensus_offset"),
                fmt("bidir_consensus_correlation"),
                fmt("bidir_disagreement"),
                fmt("bidir_prediction_error"),
                result.get("bidir_status", ""),
            ])

            if index % 20 == 0 or index == total:
                progress.update(
                    index / max(1, total) * 100.0,
                    "Запись validation CSV",
                )

    progress.done(100.0, "Validation точки сохранены")
    return path


# ============================================================
# ПОИСК VALIDATION OUTLIERS
# ============================================================

def find_validation_outliers(
    results,
):
    """
    Находим участки, где независимая проверка
    резко расходится с предсказанием карты.
    """

    outliers = []

    for index, result in enumerate(
        results
    ):

        residual = result.get(
            "after_error"
        )

        if residual is None:
            continue

        residual = abs(
            float(residual)
        )

        severity = "NORMAL"
        bidir_status = str(
            result.get("bidir_status", "UNRESOLVED")
        )

        # Confirmed discrepancies are useful diagnostics even when the
        # absolute residual is below the generic WARNING threshold.
        # This prevents a real independent disagreement from disappearing
        # into an empty validation_outliers.csv.
        if bidir_status == "CONFIRMED_DISCREPANCY":
            severity = "CONFIRMED_DISCREPANCY"
        elif (
            residual
            >= FINAL_VALIDATION_HARD_OUTLIER_SEC
        ):
            severity = "HARD"
        elif (
            residual
            >= FINAL_VALIDATION_OUTLIER_SEC
        ):
            severity = "WARNING"
        elif (
            bidir_status == "AMBIGUOUS"
            and residual >= 0.050
        ):
            severity = "AMBIGUOUS"

        if severity != "NORMAL":

            outliers.append(
                {
                    "index":
                        index,

                    "target_time":
                        float(
                            result[
                                "target_time"
                            ]
                        ),

                    "predicted_offset":
                        float(
                            result[
                                "predicted_offset"
                            ]
                        ),

                    "actual_offset":
                        (
                            None
                            if result[
                                "after_offset"
                            ] is None
                            else float(
                                result[
                                    "after_offset"
                                ]
                            )
                        ),

                    "residual":
                        residual,

                    "severity":
                        severity,

                    "correlation":
                        float(
                            result[
                                "after_correlation"
                            ]
                        ),

                    "bidir_status":
                        result.get(
                            "bidir_status",
                            "UNRESOLVED",
                        ),

                    "bidir_left_offset":
                        result.get(
                            "bidir_left_offset"
                        ),

                    "bidir_right_offset":
                        result.get(
                            "bidir_right_offset"
                        ),

                    "bidir_disagreement":
                        result.get(
                            "bidir_disagreement"
                        ),
                }
            )

    return outliers


# ============================================================
# СОХРАНЕНИЕ OUTLIERS
# ============================================================

def save_validation_outliers_csv(
    output_dir,
    outliers,
    progress,
):
    """
    Сохраняем только подозрительные validation точки.
    """

    progress.stage(
        "АНАЛИЗ VALIDATION OUTLIERS",
        0.0,
    )

    path = (
        output_dir
        / FINAL_VALIDATION_OUTLIERS_CSV
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(
            file,
            delimiter=";",
        )

        writer.writerow(
            [
                "target_time_sec",
                "predicted_offset_sec",
                "actual_offset_sec",
                "residual_sec",
                "severity",
                "correlation",
            ]
        )

        total = max(
            1,
            len(outliers),
        )

        for index, item in enumerate(
            outliers,
            start=1,
        ):

            writer.writerow(
                [
                    (
                        f"{item['target_time']:.6f}"
                    ),

                    (
                        f"{item['predicted_offset']:.6f}"
                    ),

                    (
                        ""
                        if item[
                            "actual_offset"
                        ] is None
                        else
                        f"{item['actual_offset']:.6f}"
                    ),

                    (
                        f"{item['residual']:.6f}"
                    ),

                    item[
                        "severity"
                    ],

                    (
                        f"{item['correlation']:.6f}"
                    ),
                ]
            )

            progress.update(
                index
                / total
                * 100.0,
                "Запись outliers",
            )

    progress.done(
        100.0,
        (
            f"Найдено выбросов: "
            f"{len(outliers)}"
        ),
    )

    return path


# ============================================================
# ЛОКАЛЬНАЯ РОБАСТНАЯ ПРОВЕРКА
# ============================================================

def build_robust_validation_mask(
    results,
):
    """
    Выделяем validation точки,
    которым можно доверять.

    Здесь не меняем саму карту.

    Только определяем:
        normal
        suspicious
        hard outlier
    """

    times = np.asarray(
        [
            r["target_time"]
            for r in results
        ],
        dtype=np.float64,
    )

    residuals = np.asarray(
        [
            (
                abs(
                    r["after_error"]
                )
                if r["after_error"]
                is not None
                else np.nan
            )
            for r in results
        ],
        dtype=np.float64,
    )

    mask = np.isfinite(
        residuals
    )

    if len(
        residuals
    ) < 5:

        return mask

    # --------------------------------------------------------
    # Первый проход:
    #
    # убираем сильные выбросы.
    # --------------------------------------------------------

    mask &= (
        residuals
        < FINAL_VALIDATION_HARD_OUTLIER_SEC
    )

    # --------------------------------------------------------
    # Второй проход:
    #
    # сравниваем точку с соседями.
    # --------------------------------------------------------

    for index in range(
        len(residuals)
    ):

        if not mask[index]:
            continue

        left = max(
            0,
            index
            - FINAL_NEIGHBOR_COUNT,
        )

        right = min(
            len(residuals),
            index
            + FINAL_NEIGHBOR_COUNT
            + 1,
        )

        neighborhood = residuals[
            left:right
        ]

        neighborhood = (
            neighborhood[
                np.isfinite(
                    neighborhood
                )
            ]
        )

        if len(
            neighborhood
        ) < 3:
            continue

        local_median = float(
            np.median(
                neighborhood
            )
        )

        # Если точка намного хуже местного
        # поведения — считаем её подозрительной.
        if (
            residuals[index]
            >
            local_median
            +
            FINAL_NEIGHBOR_TOLERANCE_SEC
        ):

            mask[index] = False

    return mask


# ============================================================
# ОЦЕНКА СОСТОЯНИЯ КАРТЫ ПО VALIDATION
# ============================================================

def calculate_final_validation_quality(
    results,
):
    """
    Даёт дополнительную оценку качества.

    В отличие от старого score,
    здесь основное внимание уделяется
    остаточной ошибке.
    """

    residuals = np.asarray(
        [
            abs(
                r["after_error"]
            )
            for r in results
            if r["after_error"]
            is not None
        ],
        dtype=np.float64,
    )

    if len(
        residuals
    ) == 0:

        return {
            "valid": False,
            "median": None,
            "mean": None,
            "p90": None,
            "p95": None,
            "max": None,
            "good_ratio": 0.0,
        }

    median = float(
        np.median(
            residuals
        )
    )

    mean = float(
        np.mean(
            residuals
        )
    )

    p90 = float(
        np.percentile(
            residuals,
            90,
        )
    )

    p95 = float(
        np.percentile(
            residuals,
            95,
        )
    )

    maximum = float(
        np.max(
            residuals
        )
    )

    good_ratio = float(
        np.mean(
            residuals
            <= 0.05
        )
    )

    return {
        "valid": True,
        "median": median,
        "mean": mean,
        "p90": p90,
        "p95": p95,
        "max": maximum,
        "good_ratio": good_ratio,
    }


# ============================================================
# ФИНАЛЬНЫЙ VERDICT
# ============================================================

def evaluate_final_map_for_application(
    map_result,
    validation_results,
):
    """
    Более строгий verdict перед Apply.

    """

    metrics = (
        calculate_final_validation_quality(
            validation_results
        )
    )

    if not metrics[
        "valid"
    ]:

        return {
            "verdict": "BAD",
            "score": 0.0,
            "metrics": metrics,
            "reason":
                "Нет валидных validation точек.",
        }

    score = 0.0

    # --------------------------------------------------------
    # Медианная ошибка
    # --------------------------------------------------------

    if metrics[
        "median"
    ] <= 0.010:

        score += 30

    elif metrics[
        "median"
    ] <= 0.020:

        score += 27

    elif metrics[
        "median"
    ] <= 0.050:

        score += 22

    elif metrics[
        "median"
    ] <= 0.100:

        score += 12

    # --------------------------------------------------------
    # 90-й процентиль
    # --------------------------------------------------------

    if metrics[
        "p90"
    ] <= 0.050:

        score += 25

    elif metrics[
        "p90"
    ] <= 0.100:

        score += 20

    elif metrics[
        "p90"
    ] <= 0.200:

        score += 12

    # --------------------------------------------------------
    # 95-й процентиль
    # --------------------------------------------------------

    if metrics[
        "p95"
    ] <= 0.100:

        score += 20

    elif metrics[
        "p95"
    ] <= 0.250:

        score += 14

    elif metrics[
        "p95"
    ] <= 0.500:

        score += 8

    # --------------------------------------------------------
    # Доля точек <= 50 ms
    # --------------------------------------------------------

    if metrics[
        "good_ratio"
    ] >= 0.95:

        score += 25

    elif metrics[
        "good_ratio"
    ] >= 0.90:

        score += 20

    elif metrics[
        "good_ratio"
    ] >= 0.80:

        score += 12

    elif metrics[
        "good_ratio"
    ] >= 0.70:

        score += 6

    # --------------------------------------------------------
    # Не учитываем единичный максимум слишком сильно.
    #
    # Нам важнее p90 / p95.
    # --------------------------------------------------------

    score = max(
        0.0,
        min(
            100.0,
            score,
        )
    )

    if score >= 90:

        verdict = "EXCELLENT"

    elif score >= 75:

        verdict = "GOOD"

    elif score >= 60:

        verdict = "ACCEPTABLE"

    elif score >= 40:

        verdict = "SUSPICIOUS"

    else:

        verdict = "BAD"

    return {
        "verdict":
            verdict,

        "score":
            float(score),

        "metrics":
            metrics,

        "reason":
            (
                "Оценка основана "
                "на распределении независимых "
                "остаточных ошибок."
            ),
    }


# ============================================================
# ГРАФИК РАСПРЕДЕЛЕНИЯ RESIDUAL
# ============================================================

def save_validation_distribution_jpeg(
    output_dir,
    validation_results,
    progress,
):
    """
    Показывает распределение остаточных ошибок.

    Это полезнее одного MAX.
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ГРАФИК РАСПРЕДЕЛЕНИЯ ОШИБКИ",
        0.0,
    )

    path = (
        output_dir
        / "11_VALIDATION_DISTRIBUTION.jpg"
    )

    residuals = np.asarray(
        [
            abs(
                r["after_error"]
            )
            for r in validation_results
            if r["after_error"]
            is not None
        ],
        dtype=np.float64,
    )

    fig = plt.figure(
        figsize=(16, 9),
        dpi=150,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    if len(
        residuals
    ):

        ax.hist(
            residuals,
            bins=40,
        )

        ax.axvline(
            0.05,
            linestyle="--",
            linewidth=1.0,
            label="50 ms",
        )

        ax.axvline(
            0.10,
            linestyle="--",
            linewidth=1.0,
            label="100 ms",
        )

        ax.axvline(
            0.25,
            linestyle="--",
            linewidth=1.0,
            label="250 ms",
        )

        ax.set_xlim(
            0,
            max(
                0.3,
                np.percentile(
                    residuals,
                    99,
                )
            )
        )

    ax.set_title(
        "Distribution of Independent Validation Residuals"
    )

    ax.set_xlabel(
        "Residual error, seconds"
    )

    ax.set_ylabel(
        "Number of validation points"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        format="jpg",
        dpi=150,
    )

    plt.close(
        fig
    )

    progress.done(
        100.0,
        "График распределения готов",
    )

    return path


# ============================================================
# СОХРАНЕНИЕ FINAL ROBUST MAP
# ============================================================

def save_robust_map_csv(
    output_dir,
    sync_map,
    progress,
):
    """
    Сохраняем карту в отдельный файл.

    Пока это та же карта,
    но отдельный файл нужен для безопасного
    этапа Apply.
    """

    progress.stage(
        "СОХРАНЕНИЕ ROBUST MAP",
        0.0,
    )

    path = (
        output_dir
        / FINAL_ROBUST_MAP_CSV
    )

    target_time = (
        sync_map[
            "target_time"
        ]
    )

    offset = (
        sync_map[
            "offset"
        ]
    )

    reference_time = (
        sync_map[
            "reference_time"
        ]
    )

    confidence = (
        sync_map[
            "confidence"
        ]
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(
            file,
            delimiter=";",
        )

        writer.writerow(
            [
                "target_time_sec",
                "offset_sec",
                "reference_time_sec",
                "confidence",
            ]
        )

        total = len(
            target_time
        )

        for index in range(
            total
        ):

            writer.writerow(
                [
                    (
                        f"{target_time[index]:.6f}"
                    ),
                    (
                        f"{offset[index]:.6f}"
                    ),
                    (
                        f"{reference_time[index]:.6f}"
                    ),
                    (
                        f"{confidence[index]:.6f}"
                    ),
                ]
            )

            if (
                index % 5000 == 0
                or index == total - 1
            ):

                progress.update(
                    index
                    / max(
                        1,
                        total - 1,
                    )
                    * 100.0,
                    "Запись robust map",
                )

    progress.done(
        100.0,
        "Robust map сохранена",
    )

    return path


# ============================================================
# ГРАФИК FINAL ROBUST MAP
# ============================================================

def save_robust_map_jpeg(
    output_dir,
    map_result,
    validation_results,
    progress,
):
    """
    Итоговый график:

        карта
        +
        независимые validation residuals
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ПОСТРОЕНИЕ ROBUST MAP ГРАФИКА",
        0.0,
    )

    path = (
        output_dir
        / FINAL_ROBUST_MAP_JPEG
    )

    sync_map = map_result[
        "map"
    ]

    times = np.asarray(
        [
            r["target_time"]
            for r in validation_results
        ],
        dtype=np.float64,
    )

    predicted = np.asarray(
        [
            r["predicted_offset"]
            for r in validation_results
        ],
        dtype=np.float64,
    )

    residuals = np.asarray(
        [
            (
                (
                    r["after_offset"]
                    - r["predicted_offset"]
                )
                if (
                    r["after_offset"]
                    is not None
                )
                else np.nan
            )
            for r in validation_results
        ],
        dtype=np.float64,
    )

    fig = plt.figure(
        figsize=(18, 10),
        dpi=160,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    ax.plot(
        sync_map[
            "target_time"
        ],
        sync_map[
            "offset"
        ],
        linewidth=2.4,
        label="Sync map",
    )

    ax.scatter(
        times,
        predicted,
        s=20,
        alpha=0.55,
        label="Validation predicted",
    )

    ax.scatter(
        times,
        predicted
        + residuals,
        s=16,
        alpha=0.45,
        label="Independent actual",
    )

    ax.set_title(
        "Final Robust Synchronization Map"
    )

    ax.set_xlabel(
        "Target time, seconds"
    )

    ax.set_ylabel(
        "Offset, seconds"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        format="jpg",
        dpi=160,
    )

    plt.close(
        fig
    )

    progress.done(
        100.0,
        "Robust map график готов",
    )

    return path


# ============================================================
# ROBUST MAP REPORT
# ============================================================

def save_robust_map_report(
    output_dir,
    final_quality,
    outliers,
    validation_results=None,
):
    path=output_dir/FINAL_ROBUST_REPORT
    metrics=final_quality["metrics"]
    lines=[
        "FINAL ROBUST SYNCHRONIZATION MAP",
        "="*78,
        "",
        f"VERDICT: {final_quality['verdict']}",
        f"SCORE: {final_quality['score']:.2f}/100",
        "",
        f"Median residual: {metrics['median']:.6f} s",
        f"Mean residual: {metrics['mean']:.6f} s",
        f"P90 residual: {metrics['p90']:.6f} s",
        f"P95 residual: {metrics['p95']:.6f} s",
        f"Max residual: {metrics['max']:.6f} s",
        f"Residual <= 50 ms: {metrics['good_ratio']*100:.2f}%",
        "",
        f"Validation outlier candidates: {len(outliers)}",
    ]
    hard=sum(1 for item in outliers if item.get("severity")=="HARD")
    warning=sum(1 for item in outliers if item.get("severity")=="WARNING")
    lines.append(f"Candidate HARD outliers: {hard}")
    lines.append(f"Warning outliers: {warning}")
    if validation_results is not None:
        b=summarize_bidirectional_validation(validation_results)
        lines += [
            "",
            "BIDIRECTIONAL CHECK:",
            f"  Confirmed map: {b['confirmed_map']}",
            f"  Confirmed discrepancy: {b['confirmed_discrepancy']}",
            f"  Ambiguous: {b['ambiguous']}",
            f"  Consistent but weak: {b['consistent_but_weak']}",
            f"  HARD candidates requiring manual review: {b['hard_confirmed_discrepancy']}",
            f"  Ambiguous HARD candidates: {b['hard_ambiguous']}",
        ]
    lines += ["", "The original TARGET has NOT been modified."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return path


# ============================================================
# ПОЛНЫЙ FINAL ROBUST ANALYSIS
# ============================================================

def finalize_robust_validation(
    output_dir,
    map_result,
    validation_results,
    progress,
):
    """
    Финальная стадия перед Apply.
    """

    save_validation_points_csv(
        output_dir,
        validation_results,
        progress,
    )

    outliers = (
        find_validation_outliers(
            validation_results
        )
    )

    save_validation_outliers_csv(
        output_dir,
        outliers,
        progress,
    )

    final_quality = (
        evaluate_final_map_for_application(
            map_result,
            validation_results,
        )
    )

    save_validation_distribution_jpeg(
        output_dir,
        validation_results,
        progress,
    )

    save_robust_map_csv(
        output_dir,
        map_result["map"],
        progress,
    )

    save_robust_map_jpeg(
        output_dir,
        map_result,
        validation_results,
        progress,
    )

    bidir_summary=summarize_bidirectional_validation(validation_results)
    final_quality["metrics"].update({
        "bidir_confirmed_map": bidir_summary["confirmed_map"],
        "bidir_confirmed_discrepancy": bidir_summary["confirmed_discrepancy"],
        "bidir_ambiguous": bidir_summary["ambiguous"],
        "bidir_consistent_but_weak": bidir_summary["consistent_but_weak"],
        "bidir_hard_candidates": bidir_summary["hard_candidates"],
        "bidir_hard_confirmed_discrepancy": bidir_summary["hard_confirmed_discrepancy"],
        "bidir_hard_ambiguous": bidir_summary["hard_ambiguous"],
    })

    save_robust_map_report(
        output_dir,
        final_quality,
        outliers,
        validation_results,
    )

    return {
        "outliers":
            outliers,

        "quality":
            final_quality,
    }


# ============================================================
# ПЕЧАТЬ FINAL ROBUST VERDICT
# ============================================================

def print_final_robust_verdict(
    robust_result,
    output_dir,
):
    quality = robust_result[
        "quality"
    ]

    metrics = quality[
        "metrics"
    ]

    outliers = robust_result[
        "outliers"
    ]

    print()
    print("=" * 78)
    print("FINAL ROBUST MAP VERDICT")
    print("=" * 78)

    print()

    print(
        "Verdict:",
        quality[
            "verdict"
        ],
    )

    print(
        "Score:",
        f"{quality['score']:.1f}/100",
    )

    print()

    print(
        "Median residual:",
        f"{metrics['median']:.6f} s",
    )

    print(
        "Mean residual:",
        f"{metrics['mean']:.6f} s",
    )

    print(
        "P90 residual:",
        f"{metrics['p90']:.6f} s",
    )

    print(
        "P95 residual:",
        f"{metrics['p95']:.6f} s",
    )

    print(
        "Max residual:",
        f"{metrics['max']:.6f} s",
    )

    print(
        "Residual <= 50ms:",
        f"{metrics['good_ratio'] * 100:.2f}%",
    )

    print()

    print("Validation outlier candidates:", len(outliers))
    hard=sum(1 for item in outliers if item.get("severity")=="HARD")
    print("Candidate HARD outliers:", hard)
    print("Confirmed discrepancies:", metrics.get("bidir_confirmed_discrepancy",0))
    print("Ambiguous validation points:", metrics.get("bidir_ambiguous",0))
    print("HARD candidates requiring manual review:", metrics.get("bidir_hard_confirmed_discrepancy",0))
    print("Ambiguous HARD candidates:", metrics.get("bidir_hard_ambiguous",0))
    print()

    if quality[
        "verdict"
    ] in {
        "EXCELLENT",
        "GOOD",
    }:

        print(
            "Карта прошла строгую "
            "финальную проверку."
        )

        print(
            "Следующий этап может "
            "переходить к APPLY."
        )

    elif quality[
        "verdict"
    ] == "ACCEPTABLE":

        print(
            "Карта в целом рабочая, "
            "но имеет заметные локальные проблемы."
        )

        print(
            "Перед APPLY желательно изучить "
            "validation_outliers.csv."
        )

    else:

        print(
            "Карта пока НЕ готова "
            "к безопасному APPLY."
        )

    print()

    print(
        "Рабочая папка:"
    )

    print(
        output_dir
    )

    print()

# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC
#
# ЧАСТЬ 9
#
# PRE-APPLY SAFETY CHECK
# + HARD OUTLIER REPORT
# + ROBUST APPLY MAP PREPARATION
#
# НИКАКОГО РЕНДЕРА АУДИО ЗДЕСЬ НЕТ.
#
# ============================================================


# ============================================================
# НАСТРОЙКИ PRE-APPLY
# ============================================================

# Интервал вокруг подозрительной точки,
# который удобно открыть в Audacity.
#
# Саму точку мы показываем отдельно.
