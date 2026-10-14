# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: mapper.py
# ============================================================
# Anchor cleaning, nonlinear map construction, map quality, drift analysis and final map reports.
#
# This module is part of the modular version of the project.
# Do not run it directly; start the pipeline with start_sync.py.
# ============================================================

from __future__ import annotations

import csv
import json
import math
import numpy as np

from config import *
from common import *
from aligner import *

# ============================================================
# НАСТРОЙКИ КАЧЕСТВА КАРТЫ
# ============================================================

# Минимальная корреляция для достаточно уверенной точки.
MAP_MIN_CORRELATION = 0.30


# Минимальный разрыв между первым и вторым пиком.
MAP_MIN_MARGIN = 0.015


# Максимально допустимое изменение offset
# между соседними измерениями.
#
# ВАЖНО:
#
# Это НЕ означает, что offset вообще не может меняться.
#
# Мы допускаем drift.
#
# Но если соседние точки вдруг отличаются, например:
#
#     +0.20
#     +0.21
#     +7.83
#     +0.22
#
# +7.83 будет считаться подозрительным выбросом.
#
# Значение 1.50 s достаточно мягкое для реальной
# нелинейности, но отсекает явные ложные совпадения.
MAP_MAX_LOCAL_JUMP_SEC = 1.50


# Более мягкая граница для сравнения с локальным
# линейным прогнозом.
MAP_MAX_PREDICTION_ERROR_SEC = 1.00


# Минимальный размер локального окна
# для построения устойчивого регрессионного прогноза.
MAP_LOCAL_REGRESSION_POINTS = 7


# Радиус медианного фильтра.
MAP_MEDIAN_RADIUS = 3


# Радиус дополнительного сглаживания.
MAP_SMOOTH_RADIUS = 3


# Минимальный процент валидных точек,
# при котором карта вообще рассматривается
# как потенциально пригодная.
MAP_MIN_VALID_RATIO = 0.60


# Минимальная доля точек с хорошей корреляцией.
MAP_MIN_GOOD_CORRELATION_RATIO = 0.55


# ============================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# ============================================================

def finite_float(
    value,
):
    """
    Безопасно превращает значение в float.

    None / NaN / Inf -> None
    """

    try:
        number = float(value)
    except Exception:
        return None

    if not math.isfinite(number):
        return None

    return number


# ============================================================
# ПОДГОТОВКА RAW POINTS
# ============================================================

def extract_raw_points(
    local_results,
):
    """
    Превращает результаты локального поиска
    в компактные массивы NumPy.

    Возвращаем:

        time
        offset
        correlation
        margin
        accepted
    """

    times = []
    offsets = []
    correlations = []
    margins = []
    accepted = []

    for result in local_results:

        target_time = finite_float(
            result.get(
                "target_time"
            )
        )

        offset = finite_float(
            result.get(
                "offset"
            )
        )

        correlation = finite_float(
            result.get(
                "correlation",
                0.0,
            )
        )

        margin = finite_float(
            result.get(
                "margin",
                0.0,
            )
        )

        tracking = bool(
            result.get(
                "accepted_for_tracking",
                False,
            )
        )

        if (
            target_time is None
            or offset is None
        ):
            continue

        if correlation is None:
            correlation = 0.0

        if margin is None:
            margin = 0.0

        times.append(
            target_time
        )

        offsets.append(
            offset
        )

        correlations.append(
            correlation
        )

        margins.append(
            margin
        )

        accepted.append(
            tracking
        )

    if not times:

        raise RuntimeError(
            "В локальных измерениях "
            "нет ни одной пригодной точки."
        )

    order = np.argsort(
        np.asarray(
            times,
            dtype=np.float64,
        )
    )

    return {
        "time":
            np.asarray(
                times,
                dtype=np.float64,
            )[order],

        "offset":
            np.asarray(
                offsets,
                dtype=np.float64,
            )[order],

        "correlation":
            np.asarray(
                correlations,
                dtype=np.float64,
            )[order],

        "margin":
            np.asarray(
                margins,
                dtype=np.float64,
            )[order],

        "accepted":
            np.asarray(
                accepted,
                dtype=bool,
            )[order],
    }


# ============================================================
# НАЧАЛЬНЫЙ ФИЛЬТР КАЧЕСТВА
# ============================================================

def initial_quality_mask(
    points,
):
    """
    Первичная фильтрация.

    Точка считается потенциально хорошей,
    если выполнены:

        correlation >= порога

    И

        margin >= порога

    Либо она была принята механизмом tracking.
    """

    correlation = points[
        "correlation"
    ]

    margin = points[
        "margin"
    ]

    tracking = points[
        "accepted"
    ]

    good = (
        (
            correlation
            >= MAP_MIN_CORRELATION
        )
        &
        (
            margin
            >= MAP_MIN_MARGIN
        )
    )

    # Tracking не должен полностью обходить
    # требования качества.
    #
    # Он лишь позволяет оставить точку,
    # если её корреляция чуть ниже порога,
    # но всё ещё разумна.
    tracking_good = (
        tracking
        &
        (
            correlation
            >= MAP_MIN_CORRELATION
            * 0.85
        )
    )

    return (
        good
        |
        tracking_good
    )


# ============================================================
# ВЗВЕШИВАНИЕ ТОЧЕК
# ============================================================

def calculate_point_weights(
    points,
):
    """
    Строит вес каждой точки.

    Вес зависит от:

        1. correlation
        2. margin

    Чем лучше совпадение и чем оно однозначнее,
    тем сильнее точка влияет на карту.
    """

    correlation = np.clip(
        points[
            "correlation"
        ],
        0.0,
        1.0,
    )

    margin = np.maximum(
        points[
            "margin"
        ],
        0.0,
    )

    # Нормируем margin.
    margin_score = np.clip(
        margin
        / max(
            MAP_MIN_MARGIN,
            1e-9,
        ),
        0.0,
        4.0,
    )

    margin_score = np.minimum(
        margin_score,
        1.0,
    )

    # correlation уже находится примерно [0;1].
    #
    # Делаем её немного нелинейной,
    # чтобы очень хорошие совпадения получили
    # заметно больший вес.
    corr_score = (
        correlation
        ** 2
    )

    weights = (
        corr_score
        * (
            0.25
            + 0.75
            * margin_score
        )
    )

    return np.asarray(
        weights,
        dtype=np.float64,
    )


# ============================================================
# ЛОКАЛЬНАЯ МЕДИАНА
# ============================================================

def local_median_prediction(
    values,
    index,
    radius,
):
    """
    Медианное значение вокруг точки.

    Используем для поиска одиночных выбросов.
    """

    start = max(
        0,
        index - radius,
    )

    end = min(
        len(values),
        index + radius + 1,
    )

    window = values[
        start:end
    ]

    if len(window) == 0:
        return float(
            values[index]
        )

    return float(
        np.median(
            window
        )
    )


# ============================================================
# ПЕРВАЯ ОЧИСТКА ПО МЕДИАНЕ
# ============================================================

def detect_median_outliers(
    points,
):
    """
    Находим очевидные одиночные скачки.

    Пример:

        0.20
        0.21
        0.22
        8.37   <-- выброс
        0.24
        0.25

    Такая точка будет отброшена.
    """

    offsets = points[
        "offset"
    ]

    good_quality = (
        initial_quality_mask(
            points
        )
    )

    outlier = np.zeros(
        len(offsets),
        dtype=bool,
    )

    for index in range(
        len(offsets)
    ):

        if not good_quality[
            index
        ]:
            outlier[index] = True
            continue

        # На краях используем доступный
        # локальный диапазон.
        prediction = (
            local_median_prediction(
                offsets,
                index,
                MAP_MEDIAN_RADIUS,
            )
        )

        error = abs(
            offsets[index]
            - prediction
        )

        if (
            error
            > MAP_MAX_LOCAL_JUMP_SEC
        ):

            outlier[index] = True

    return outlier


# ============================================================
# ЛОКАЛЬНАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ
# ============================================================

def weighted_linear_regression(
    x,
    y,
    weights,
):
    """
    Простая взвешенная линейная регрессия:

        y = a*x + b

    Возвращает:

        a
        b

    Используется только как локальный прогноз,
    а не как финальная карта.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
        &
        np.isfinite(weights)
        &
        (
            weights > 0
        )
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    weights = weights[
        valid
    ]

    if len(x) < 2:
        return (
            0.0,
            float(
                np.median(y)
            )
            if len(y)
            else 0.0,
        )

    weight_sum = np.sum(
        weights
    )

    if weight_sum <= 1e-12:

        return (
            0.0,
            float(
                np.median(y)
            ),
        )

    x_mean = (
        np.sum(
            weights
            * x
        )
        / weight_sum
    )

    y_mean = (
        np.sum(
            weights
            * y
        )
        / weight_sum
    )

    dx = (
        x
        - x_mean
    )

    dy = (
        y
        - y_mean
    )

    denominator = np.sum(
        weights
        * dx
        * dx
    )

    if denominator <= 1e-12:

        return (
            0.0,
            float(
                y_mean
            ),
        )

    slope = (
        np.sum(
            weights
            * dx
            * dy
        )
        / denominator
    )

    intercept = (
        y_mean
        - slope * x_mean
    )

    return (
        float(slope),
        float(intercept),
    )


# ============================================================
# ПРОВЕРКА ПРЕДСКАЗАНИЯ
# ============================================================

def detect_prediction_outliers(
    points,
    initial_mask,
):
    """
    Второй уровень проверки.

    Здесь мы спрашиваем:

    "Насколько найденная точка согласуется
     с поведением соседних точек?"

    Например:

        +0.10
        +0.12
        +0.15
        +0.18
        +3.91
        +0.24

    Даже если корреляция 3.91 случайно высокая,
    она будет противоречить локальной траектории.
    """

    times = points[
        "time"
    ]

    offsets = points[
        "offset"
    ]

    weights = calculate_point_weights(
        points
    )

    outlier = np.zeros(
        len(times),
        dtype=bool,
    )

    for index in range(
        len(times)
    ):

        if not initial_mask[
            index
        ]:
            outlier[index] = True
            continue

        # ----------------------------------------------------
        # Получаем ближайшие хорошие точки.
        # ----------------------------------------------------

        left = max(
            0,
            index
            - MAP_LOCAL_REGRESSION_POINTS,
        )

        right = min(
            len(times),
            index
            + MAP_LOCAL_REGRESSION_POINTS
            + 1,
        )

        local_indices = np.arange(
            left,
            right,
        )

        local_indices = (
            local_indices[
                initial_mask[
                    local_indices
                ]
            ]
        )

        # Саму точку из обучения
        # исключаем, если вокруг есть другие.
        if len(
            local_indices
        ) > 2:

            local_indices = (
                local_indices[
                    local_indices
                    != index
                ]
            )

        if len(
            local_indices
        ) < 3:
            continue

        local_x = times[
            local_indices
        ]

        local_y = offsets[
            local_indices
        ]

        local_w = weights[
            local_indices
        ]

        slope, intercept = (
            weighted_linear_regression(
                local_x,
                local_y,
                local_w,
            )
        )

        predicted = (
            slope
            * times[index]
            + intercept
        )

        error = abs(
            offsets[index]
            - predicted
        )

        if (
            error
            > MAP_MAX_PREDICTION_ERROR_SEC
        ):

            outlier[index] = True

    return outlier


# ============================================================
# ПОСТРОЕНИЕ ФИНАЛЬНОГО НАБОРА ЯКОРЕЙ
# ============================================================

def clean_anchor_points(
    local_results,
    progress: Progress,
):
    """
    Полный цикл очистки:

        raw
         ↓
        quality filter
         ↓
        median outliers
         ↓
        regression outliers
         ↓
        final anchors
    """

    progress.stage(
        "ОЧИСТКА RAW ЯКОРЕЙ",
        0.0,
    )

    points = extract_raw_points(
        local_results
    )

    progress.update(
        10.0,
        "Raw points loaded",
    )

    quality_mask = (
        initial_quality_mask(
            points
        )
    )

    progress.update(
        25.0,
        (
            "Первичный фильтр | "
            f"{np.sum(quality_mask)} / "
            f"{len(quality_mask)}"
        ),
    )

    median_outliers = (
        detect_median_outliers(
            points
        )
    )

    median_mask = (
        quality_mask
        &
        ~median_outliers
    )

    progress.update(
        50.0,
        (
            "Median filter | "
            f"{np.sum(median_mask)} / "
            f"{len(median_mask)}"
        ),
    )

    prediction_outliers = (
        detect_prediction_outliers(
            points,
            median_mask,
        )
    )

    final_mask = (
        median_mask
        &
        ~prediction_outliers
    )

    progress.update(
        75.0,
        (
            "Prediction filter | "
            f"{np.sum(final_mask)} / "
            f"{len(final_mask)}"
        ),
    )

    # --------------------------------------------------------
    # Создаём итоговые массивы.
    # --------------------------------------------------------

    clean = {
        "time":
            points[
                "time"
            ][final_mask],

        "offset":
            points[
                "offset"
            ][final_mask],

        "correlation":
            points[
                "correlation"
            ][final_mask],

        "margin":
            points[
                "margin"
            ][final_mask],

        "weight":
            calculate_point_weights(
                points
            )[final_mask],

        "original_index":
            np.nonzero(
                final_mask
            )[0],
    }

    if len(
        clean["time"]
    ) < 4:

        raise RuntimeError(
            "После очистки осталось "
            "слишком мало надёжных якорей "
            "для построения карты."
        )

    progress.update(
        95.0,
        (
            "Итоговые якоря: "
            f"{len(clean['time'])}"
        ),
    )

    progress.done(
        100.0,
        "Очистка якорей завершена",
    )

    return {
        "raw": points,
        "quality_mask": quality_mask,
        "median_outliers": median_outliers,
        "prediction_outliers":
            prediction_outliers,
        "final_mask": final_mask,
        "clean": clean,
    }


# ============================================================
# ИНТЕРПОЛЯЦИЯ OFFSET
# ============================================================

def build_interpolated_map(
    clean,
    target_duration,
    step_sec,
    progress: Progress,
):
    """
    Строит непрерывную карту:

        Target time
            ->
        Offset

    используя очищенные якоря.

    Между якорями применяется линейная интерполяция.

    Это пока базовая карта.
    Затем поверх неё будет сделано
    мягкое сглаживание.
    """

    progress.stage(
        "ПОСТРОЕНИЕ НЕЛИНЕЙНОЙ КАРТЫ",
        0.0,
    )

    anchor_time = np.asarray(
        clean["time"],
        dtype=np.float64,
    )

    anchor_offset = np.asarray(
        clean["offset"],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Убираем возможные дубликаты времени.
    # --------------------------------------------------------

    unique_time, unique_indices = (
        np.unique(
            anchor_time,
            return_index=True,
        )
    )

    unique_offset = (
        anchor_offset[
            unique_indices
        ]
    )

    # --------------------------------------------------------
    # Сортировка.
    # --------------------------------------------------------

    order = np.argsort(
        unique_time
    )

    unique_time = (
        unique_time[
            order
        ]
    )

    unique_offset = (
        unique_offset[
            order
        ]
    )

    # --------------------------------------------------------
    # Формируем сетку карты.
    # --------------------------------------------------------

    start = max(
        0.0,
        float(
            unique_time[0]
        ),
    )

    end = min(
        target_duration,
        float(
            unique_time[-1]
        ),
    )

    if end <= start:

        raise RuntimeError(
            "Невозможно построить временную карту."
        )

    map_time = build_exact_time_grid(
        start,
        end,
        step_sec,
    )

    map_offset = np.interp(
        map_time,
        unique_time,
        unique_offset,
    )

    # --------------------------------------------------------
    # Веса интерполяции.
    #
    # Не являются финальной уверенностью,
    # это только информация о плотности якорей.
    # --------------------------------------------------------

    map_confidence = np.interp(
        map_time,
        unique_time,
        clean[
            "weight"
        ][
            unique_indices
            if len(
                unique_indices
            )
            ==
            len(
                clean[
                    "weight"
                ]
            )
            else
            np.arange(
                len(
                    unique_time
                )
            )
        ],
    )

    progress.update(
        70.0,
        (
            f"Map points: "
            f"{len(map_time):,}"
        ),
    )

    progress.update(
        90.0,
        "Проверка интерполированной карты",
    )

    # --------------------------------------------------------
    # Проверяем монотонность времени.
    #
    # Пока карта хранит offset отдельно,
    # но соответствующее Reference time:
    #
    #     reference = target + offset
    #
    # должно возрастать.
    # --------------------------------------------------------

    reference_time = (
        map_time
        + map_offset
    )

    diffs = np.diff(
        reference_time
    )

    negative_count = int(
        np.sum(
            diffs <= 0
        )
    )

    if negative_count > 0:

        raise RuntimeError(
            "Построенная карта нарушает "
            "монотонность временной шкалы."
        )

    progress.done(
        100.0,
        "Нелинейная карта построена",
    )

    return {
        "target_time":
            map_time,

        "offset":
            map_offset,

        "reference_time":
            reference_time,

        "confidence":
            map_confidence,

        "anchor_time":
            unique_time,

        "anchor_offset":
            unique_offset,
    }


# ============================================================
# СГЛАЖИВАНИЕ КАРТЫ
# ============================================================

def smooth_offset_curve(
    offset,
    radius,
):
    """
    Мягкое сглаживание.

    ВАЖНО:
    не используем агрессивный фильтр.

    Нам нельзя "стереть" настоящий nonlinear drift.

    Поэтому используем несколько проходов
    короткого moving-average.
    """

    values = np.asarray(
        offset,
        dtype=np.float64,
    )

    if (
        radius <= 0
        or len(values) < 3
    ):

        return values.copy()

    width = (
        radius * 2
        + 1
    )

    kernel = (
        np.ones(
            width,
            dtype=np.float64,
        )
        / width
    )

    padded = np.pad(
        values,
        (
            radius,
            radius,
        ),
        mode="edge",
    )

    smoothed = np.convolve(
        padded,
        kernel,
        mode="valid",
    )

    return smoothed


def smooth_sync_map(
    sync_map,
    progress: Progress,
):
    """
    Применяет мягкое сглаживание к offset curve
    и повторно проверяет монотонность reference time.
    """

    progress.stage(
        "СГЛАЖИВАНИЕ КАРТЫ",
        0.0,
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

    progress.update(
        25.0,
        "Мягкое сглаживание",
    )

    smoothed = (
        smooth_offset_curve(
            offset,
            MAP_SMOOTH_RADIUS,
        )
    )

    progress.update(
        60.0,
        "Проверка reference timeline",
    )

    reference_time = (
        target_time
        + smoothed
    )

    differences = np.diff(
        reference_time
    )

    negative_count = int(
        np.sum(
            differences <= 0
        )
    )

    if negative_count > 0:

        raise RuntimeError(
            "После сглаживания карта "
            "перестала быть монотонной."
        )

    result = dict(
        sync_map
    )

    result[
        "offset_raw"
    ] = offset.copy()

    result[
        "offset"
    ] = smoothed

    result[
        "reference_time"
    ] = reference_time

    progress.done(
        100.0,
        "Карта сглажена и проверена",
    )

    return result


# ============================================================
# РАСЧЁТ МЕТРИК КАРТЫ
# ============================================================

def calculate_map_metrics(
    raw_points,
    sync_map,
):
    """
    Количественная характеристика карты.
    """

    raw_time = (
        raw_points[
            "time"
        ]
    )

    raw_offset = (
        raw_points[
            "offset"
        ]
    )

    final_offset = (
        sync_map[
            "offset"
        ]
    )

    target_time = (
        sync_map[
            "target_time"
        ]
    )

    reference_time = (
        sync_map[
            "reference_time"
        ]
    )

    metrics = {}

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    metrics[
        "raw_point_count"
    ] = int(
        len(raw_time)
    )

    if len(raw_offset):

        metrics[
            "raw_offset_min"
        ] = float(
            np.min(
                raw_offset
            )
        )

        metrics[
            "raw_offset_max"
        ] = float(
            np.max(
                raw_offset
            )
        )

        metrics[
            "raw_offset_median"
        ] = float(
            np.median(
                raw_offset
            )
        )

    else:

        metrics[
            "raw_offset_min"
        ] = 0.0

        metrics[
            "raw_offset_max"
        ] = 0.0

        metrics[
            "raw_offset_median"
        ] = 0.0

    # --------------------------------------------------------
    # FINAL CURVE
    # --------------------------------------------------------

    metrics[
        "map_point_count"
    ] = int(
        len(target_time)
    )

    metrics[
        "map_start_offset"
    ] = float(
        final_offset[0]
    )

    metrics[
        "map_end_offset"
    ] = float(
        final_offset[-1]
    )

    metrics[
        "map_min_offset"
    ] = float(
        np.min(
            final_offset
        )
    )

    metrics[
        "map_max_offset"
    ] = float(
        np.max(
            final_offset
        )
    )

    metrics[
        "map_offset_range"
    ] = float(
        np.max(
            final_offset
        )
        -
        np.min(
            final_offset
        )
    )

    # --------------------------------------------------------
    # Drift.
    #
    # Offset(end) - Offset(start)
    # --------------------------------------------------------

    metrics[
        "total_offset_change"
    ] = float(
        final_offset[-1]
        -
        final_offset[0]
    )

    duration = (
        target_time[-1]
        -
        target_time[0]
    )

    if duration > 0:

        metrics[
            "average_offset_rate"
        ] = float(
            (
                final_offset[-1]
                -
                final_offset[0]
            )
            / duration
        )

    else:

        metrics[
            "average_offset_rate"
        ] = 0.0

    # --------------------------------------------------------
    # Скорость изменения drift.
    # --------------------------------------------------------

    if len(final_offset) >= 3:

        offset_d1 = np.gradient(
            final_offset,
            target_time,
        )

        offset_d2 = np.gradient(
            offset_d1,
            target_time,
        )

        metrics[
            "max_abs_offset_slope"
        ] = float(
            np.max(
                np.abs(
                    offset_d1
                )
            )
        )

        metrics[
            "max_abs_offset_curvature"
        ] = float(
            np.max(
                np.abs(
                    offset_d2
                )
            )
        )

    else:

        metrics[
            "max_abs_offset_slope"
        ] = 0.0

        metrics[
            "max_abs_offset_curvature"
        ] = 0.0

    # --------------------------------------------------------
    # Reference timeline.
    # --------------------------------------------------------

    metrics[
        "reference_start"
    ] = float(
        reference_time[0]
    )

    metrics[
        "reference_end"
    ] = float(
        reference_time[-1]
    )

    metrics[
        "reference_duration"
    ] = float(
        reference_time[-1]
        -
        reference_time[0]
    )

    return metrics


# ============================================================
# СОХРАНЕНИЕ ФИНАЛЬНОЙ КАРТЫ
# ============================================================

def save_sync_map_csv(
    output_dir: Path,
    sync_map,
    progress: Progress,
):
    """
    Сохраняет непрерывную карту.

    Формат:

        target_time;
        offset;
        reference_time;
        confidence
    """

    progress.stage(
        "СОХРАНЕНИЕ ФИНАЛЬНОЙ КАРТЫ",
        0.0,
    )

    path = (
        output_dir
        / "final_sync_map.csv"
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
                    f"{target_time[index]:.6f}",
                    f"{offset[index]:.6f}",
                    f"{reference_time[index]:.6f}",
                    f"{confidence[index]:.6f}",
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
                    "Запись final_sync_map.csv",
                )

    progress.done(
        100.0,
        "Финальная карта сохранена",
    )

    return path


# ============================================================
# СОХРАНЕНИЕ МЕТРИК
# ============================================================

def save_map_metrics(
    output_dir: Path,
    metrics,
):

    path = (
        output_dir
        / "sync_map_metrics.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path


# ============================================================
# СРАВНЕНИЕ RAW И FINAL
# ============================================================

def calculate_raw_vs_map_error(
    raw_points,
    sync_map,
):
    """
    Показывает, насколько финальная интерполяционная
    карта проходит через очищенные измерения.
    """

    clean_time = (
        raw_points[
            "time"
        ]
    )

    clean_offset = (
        raw_points[
            "offset"
        ]
    )

    map_time = (
        sync_map[
            "target_time"
        ]
    )

    map_offset = (
        sync_map[
            "offset"
        ]
    )

    if len(clean_time) == 0:
        return {
            "mean_error": 0.0,
            "median_error": 0.0,
            "max_error": 0.0,
        }

    predicted = np.interp(
        clean_time,
        map_time,
        map_offset,
    )

    errors = np.abs(
        clean_offset
        - predicted
    )

    return {
        "mean_error": float(
            np.mean(
                errors
            )
        ),

        "median_error": float(
            np.median(
                errors
            )
        ),

        "max_error": float(
            np.max(
                errors
            )
        ),
    }


# ============================================================
# КОМПЛЕКСНОЕ ПОСТРОЕНИЕ КАРТЫ
# ============================================================

def build_final_sync_map(
    alignment,
    target_duration,
    progress: Progress,
):
    """
    Полный путь:

        local measurements
              ↓
        quality filtering
              ↓
        outlier removal
              ↓
        interpolation
              ↓
        smoothing
              ↓
        metrics
    """

    cleaned = clean_anchor_points(
        alignment[
            "local_results"
        ],
        progress,
    )

    # --------------------------------------------------------
    # Интерполируем.
    # --------------------------------------------------------

    base_map = (
        build_interpolated_map(
            cleaned[
                "clean"
            ],
            target_duration,
            MAP_STEP_SEC,
            progress,
        )
    )

    # --------------------------------------------------------
    # Сглаживаем.
    # --------------------------------------------------------

    final_map = (
        smooth_sync_map(
            base_map,
            progress,
        )
    )

    # --------------------------------------------------------
    # Метрики.
    # --------------------------------------------------------

    metrics = (
        calculate_map_metrics(
            cleaned[
                "clean"
            ],
            final_map,
        )
    )

    raw_error = (
        calculate_raw_vs_map_error(
            cleaned[
                "clean"
            ],
            final_map,
        )
    )

    metrics.update(
        {
            "map_fit_mean_error":
                raw_error[
                    "mean_error"
                ],

            "map_fit_median_error":
                raw_error[
                    "median_error"
                ],

            "map_fit_max_error":
                raw_error[
                    "max_error"
                ],
        }
    )

    return {
        "cleaned":
            cleaned,

        "map":
            final_map,

        "metrics":
            metrics,
    }


# ============================================================
# ВИЗУАЛИЗАЦИЯ RAW -> CLEAN -> FINAL
# ============================================================

def save_map_quality_jpeg(
    output_dir: Path,
    map_result,
    progress: Progress,
):
    """
    Один из самых важных графиков проекта.

    Показываем:

        RAW measurements
        CLEAN anchors
        FINAL nonlinear map
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ГРАФИК КАЧЕСТВА КАРТЫ",
        0.0,
    )

    path = (
        output_dir
        / "05_sync_map_quality.jpg"
    )

    raw = map_result[
        "cleaned"
    ]

    final_map = map_result[
        "map"
    ]

    raw_data = raw[
        "raw"
    ]

    clean = raw[
        "clean"
    ]

    fig = plt.figure(
        figsize=(18, 10),
        dpi=150,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    # --------------------------------------------------------
    # Raw
    # --------------------------------------------------------

    ax.scatter(
        raw_data["time"],
        raw_data["offset"],
        s=10,
        alpha=0.20,
        label="RAW measurements",
    )

    # --------------------------------------------------------
    # Clean anchors
    # --------------------------------------------------------

    ax.scatter(
        clean["time"],
        clean["offset"],
        s=18,
        alpha=0.75,
        label="Accepted anchors",
    )

    # --------------------------------------------------------
    # Final map
    # --------------------------------------------------------

    ax.plot(
        final_map["target_time"],
        final_map["offset"],
        linewidth=2.0,
        label="FINAL nonlinear map",
    )

    ax.set_title(
        "Waveform Synchronization Map"
    )

    ax.set_xlabel(
        "Target time, seconds"
    )

    ax.set_ylabel(
        "Reference - Target offset, seconds"
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
        f"Карта визуализирована: {path.name}",
    )

    return path


# ============================================================
# ПЕЧАТЬ МЕТРИК
# ============================================================

def print_map_metrics(
    metrics,
):
    print()
    print("=" * 78)
    print("МЕТРИКИ ФИНАЛЬНОЙ КАРТЫ")
    print("=" * 78)

    print()

    print(
        "Количество raw точек          :",
        metrics[
            "raw_point_count"
        ],
    )

    print(
        "Количество точек карты        :",
        metrics[
            "map_point_count"
        ],
    )

    print()

    print(
        "Начальный offset              :",
        f"{metrics['map_start_offset']:+.6f} s",
    )

    print(
        "Конечный offset               :",
        f"{metrics['map_end_offset']:+.6f} s",
    )

    print(
        "Общее изменение offset        :",
        f"{metrics['total_offset_change']:+.6f} s",
    )

    print()

    print(
        "Минимальный offset            :",
        f"{metrics['map_min_offset']:+.6f} s",
    )

    print(
        "Максимальный offset            :",
        f"{metrics['map_max_offset']:+.6f} s",
    )

    print(
        "Размах offset                 :",
        f"{metrics['map_offset_range']:.6f} s",
    )

    print()

    print(
        "Средняя скорость drift        :",
        f"{metrics['average_offset_rate']:+.9f}",
        "s/s",
    )

    print(
        "Макс. |offset slope|          :",
        f"{metrics['max_abs_offset_slope']:.9f}",
        "s/s",
    )

    print(
        "Макс. |offset curvature|      :",
        f"{metrics['max_abs_offset_curvature']:.12f}",
    )

    print()

    print(
        "Средняя ошибка карты          :",
        f"{metrics['map_fit_mean_error']:.6f} s",
    )

    print(
        "Медианная ошибка карты        :",
        f"{metrics['map_fit_median_error']:.6f} s",
    )

    print(
        "Максимальная ошибка карты     :",
        f"{metrics['map_fit_max_error']:.6f} s",
    )

    print()

# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC
#
# ЧАСТЬ 6
#
# FULL ANALYSIS PIPELINE
# + FINAL MAP VALIDATION
# + QUALITY VERDICT
# + ALIGNMENT REPORT
#
# ============================================================


# ============================================================
# ДОПОЛНИТЕЛЬНЫЕ НАСТРОЙКИ
# ============================================================

# Минимальная доля карты, которая должна быть покрыта
# реальными очищенными якорями.
MIN_MAP_COVERAGE_RATIO = 0.75


# Если нелинейная карта имеет слишком резкие изменения
# скорости drift, считаем это подозрительным.
MAX_REASONABLE_OFFSET_SLOPE = 0.02


# Максимально допустимая локальная ошибка
# между очищенным якорем и финальной картой.
MAX_REASONABLE_MAP_FIT_ERROR = 0.50


# ------------------------------------------------------------
# Названия файлов
# ------------------------------------------------------------

FINAL_REPORT_NAME = (
    "FINAL_SYNC_REPORT.txt"
)

FINAL_METRICS_NAME = (
    "final_sync_metrics.json"
)

FINAL_MAP_JPEG_NAME = (
    "06_FINAL_SYNC_MAP.jpg"
)

FINAL_DRIFT_JPEG_NAME = (
    "07_DRIFT_ANALYSIS.jpg"
)


# ============================================================
# ПОКРЫТИЕ ЯКОРЯМИ
# ============================================================

def calculate_anchor_coverage(
    clean,
    target_duration,
):
    """
    Сколько временной шкалы реально покрыто
    надёжными измерениями.

    Мы не хотим ситуацию:

        хорошие точки только первые 10 минут
        а остальные 2.5 часа получены простой
        интерполяцией.

    Поэтому отдельно оцениваем покрытие.
    """

    times = clean[
        "time"
    ]

    if len(times) < 2:
        return 0.0

    first = float(
        np.min(times)
    )

    last = float(
        np.max(times)
    )

    covered = max(
        0.0,
        last - first,
    )

    if target_duration <= 0:
        return 0.0

    return (
        covered
        / target_duration
    )


# ============================================================
# ОЦЕНКА РАСПРЕДЕЛЕНИЯ ЯКОРЕЙ
# ============================================================

def calculate_anchor_distribution(
    clean,
    target_duration,
):
    """
    Проверяет, равномерно ли распределены якоря
    по фильму.

    Это важно.

    Допустим:

        350 хороших точек

    звучит красиво.

    Но если 330 из них находятся вокруг
    одного участка фильма, карта плохая.
    """

    times = clean[
        "time"
    ]

    if len(times) < 2:
        return {
            "median_spacing": 0.0,
            "max_spacing": 0.0,
            "mean_spacing": 0.0,
            "coverage": 0.0,
        }

    sorted_times = np.sort(
        times
    )

    spacing = np.diff(
        sorted_times
    )

    coverage = (
        calculate_anchor_coverage(
            clean,
            target_duration,
        )
    )

    return {
        "median_spacing": float(
            np.median(
                spacing
            )
        ),

        "mean_spacing": float(
            np.mean(
                spacing
            )
        ),

        "max_spacing": float(
            np.max(
                spacing
            )
        ),

        "coverage": float(
            coverage
        ),
    }


# ============================================================
# ПРОВЕРКА ДРИФТА
# ============================================================

def analyze_drift_shape(
    sync_map,
):
    """
    Изучаем саму форму offset curve.

    Нам не обязательно нужен линейный drift.

    Наоборот:

        nonlinear drift
        +
        локальные изменения

    может быть реальной особенностью исходников.

    Поэтому здесь не пытаемся насильно
    аппроксимировать всё одной прямой.
    """

    time_axis = sync_map[
        "target_time"
    ]

    offset = sync_map[
        "offset"
    ]

    if len(offset) < 3:
        return {
            "slope_min": 0.0,
            "slope_max": 0.0,
            "slope_mean": 0.0,
            "slope_abs_max": 0.0,
            "curvature_abs_max": 0.0,
        }

    slope = np.gradient(
        offset,
        time_axis,
    )

    curvature = np.gradient(
        slope,
        time_axis,
    )

    return {
        "slope_min": float(
            np.min(
                slope
            )
        ),

        "slope_max": float(
            np.max(
                slope
            )
        ),

        "slope_mean": float(
            np.mean(
                slope
            )
        ),

        "slope_abs_max": float(
            np.max(
                np.abs(
                    slope
                )
            )
        ),

        "curvature_abs_max": float(
            np.max(
                np.abs(
                    curvature
                )
            )
        ),
    }


# ============================================================
# ПРОВЕРКА НА ПЛОХУЮ КАРТУ
# ============================================================

def evaluate_sync_map(
    map_result,
    target_duration,
):
    """
    Итоговая оценка карты.

    Возвращает:

        GOOD
        ACCEPTABLE
        SUSPICIOUS
        BAD

    ВАЖНО:

    Эта оценка НЕ утверждает математическую истину.

    Она оценивает внутреннюю согласованность карты.
    """

    clean = map_result[
        "cleaned"
    ][
        "clean"
    ]

    metrics = map_result[
        "metrics"
    ]

    distribution = (
        calculate_anchor_distribution(
            clean,
            target_duration,
        )
    )

    drift = (
        analyze_drift_shape(
            map_result[
                "map"
            ]
        )
    )

    valid_points = (
        metrics[
            "raw_point_count"
        ]
    )

    clean_points = len(
        clean[
            "time"
        ]
    )

    if valid_points > 0:

        valid_ratio = (
            clean_points
            / valid_points
        )

    else:

        valid_ratio = 0.0

    coverage = (
        distribution[
            "coverage"
        ]
    )

    fit_error = (
        metrics[
            "map_fit_median_error"
        ]
    )

    max_fit_error = (
        metrics[
            "map_fit_max_error"
        ]
    )

    slope_abs_max = (
        drift[
            "slope_abs_max"
        ]
    )

    # --------------------------------------------------------
    # Система оценки
    # --------------------------------------------------------

    problems = []
    warnings = []

    score = 100.0

    # --------------------------------------------------------
    # 1. Доля оставшихся точек
    # --------------------------------------------------------

    if valid_ratio < 0.40:

        score -= 35

        problems.append(
            "После очистки осталось "
            "слишком мало надёжных точек."
        )

    elif valid_ratio < 0.60:

        score -= 20

        warnings.append(
            "После очистки осталось "
            "меньше 60% исходных точек."
        )

    elif valid_ratio < 0.75:

        score -= 10

    # --------------------------------------------------------
    # 2. Coverage
    # --------------------------------------------------------

    if coverage < 0.50:

        score -= 30

        problems.append(
            "Надёжные якоря покрывают "
            "меньше половины фильма."
        )

    elif coverage < MIN_MAP_COVERAGE_RATIO:

        score -= 15

        warnings.append(
            "В некоторых больших областях "
            "карта опирается на интерполяцию."
        )

    # --------------------------------------------------------
    # 3. Ошибка подгонки
    # --------------------------------------------------------

    if fit_error > MAX_REASONABLE_MAP_FIT_ERROR:

        score -= 25

        problems.append(
            "Финальная карта плохо проходит "
            "через очищенные якоря."
        )

    elif fit_error > 0.20:

        score -= 10

        warnings.append(
            "Ошибка подгонки карты заметна."
        )

    # --------------------------------------------------------
    # 4. Максимальная ошибка
    # --------------------------------------------------------

    if max_fit_error > 1.50:

        score -= 15

        warnings.append(
            "Есть отдельные точки с "
            "большим отклонением от карты."
        )

    # --------------------------------------------------------
    # 5. Скорость изменения offset
    # --------------------------------------------------------

    if slope_abs_max > (
        MAX_REASONABLE_OFFSET_SLOPE
        * 3
    ):

        score -= 25

        problems.append(
            "Offset меняется слишком резко."
        )

    elif slope_abs_max > (
        MAX_REASONABLE_OFFSET_SLOPE
    ):

        score -= 10

        warnings.append(
            "В карте есть относительно "
            "резкие локальные изменения drift."
        )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )

    if (
        len(problems) > 0
        or score < 45
    ):

        verdict = "BAD"

    elif score < 65:

        verdict = "SUSPICIOUS"

    elif score < 82:

        verdict = "ACCEPTABLE"

    else:

        verdict = "GOOD"

    return {
        "verdict":
            verdict,

        "score":
            float(score),

        "valid_ratio":
            float(valid_ratio),

        "coverage":
            float(coverage),

        "fit_error":
            float(fit_error),

        "max_fit_error":
            float(max_fit_error),

        "slope_abs_max":
            float(slope_abs_max),

        "distribution":
            distribution,

        "drift":
            drift,

        "problems":
            problems,

        "warnings":
            warnings,
    }


# ============================================================
# ФИНАЛЬНЫЙ ГРАФИК КАРТЫ
# ============================================================

def save_final_map_jpeg(
    output_dir: Path,
    map_result,
    evaluation,
    progress: Progress,
):
    """
    Основной график карты.

    Показываем:

        очищенные якоря
        финальную карту
        стартовую точку
        конечную точку
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ПОСТРОЕНИЕ ФИНАЛЬНОЙ КАРТЫ",
        0.0,
    )

    path = (
        output_dir
        / FINAL_MAP_JPEG_NAME
    )

    clean = map_result[
        "cleaned"
    ][
        "clean"
    ]

    sync_map = map_result[
        "map"
    ]

    fig = plt.figure(
        figsize=(18, 10),
        dpi=160,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    # --------------------------------------------------------
    # Очищенные якоря
    # --------------------------------------------------------

    ax.scatter(
        clean["time"],
        clean["offset"],
        s=16,
        alpha=0.55,
        label="Clean anchors",
    )

    # --------------------------------------------------------
    # Финальная карта
    # --------------------------------------------------------

    ax.plot(
        sync_map[
            "target_time"
        ],
        sync_map[
            "offset"
        ],
        linewidth=2.4,
        label="Final sync map",
    )

    # --------------------------------------------------------
    # Начало
    # --------------------------------------------------------

    ax.scatter(
        [
            sync_map[
                "target_time"
            ][0]
        ],
        [
            sync_map[
                "offset"
            ][0]
        ],
        s=80,
        marker="o",
        label="Map start",
    )

    # --------------------------------------------------------
    # Конец
    # --------------------------------------------------------

    ax.scatter(
        [
            sync_map[
                "target_time"
            ][-1]
        ],
        [
            sync_map[
                "offset"
            ][-1]
        ],
        s=80,
        marker="x",
        label="Map end",
    )

    ax.set_title(
        (
            "FINAL AUDIO SYNC MAP — "
            f"{evaluation['verdict']} "
            f"({evaluation['score']:.1f}/100)"
        )
    )

    ax.set_xlabel(
        "Target time, seconds"
    )

    ax.set_ylabel(
        "Reference - Target offset, seconds"
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
        f"Финальная карта: {path.name}",
    )

    return path


# ============================================================
# ГРАФИК DRIFT
# ============================================================

def save_drift_analysis_jpeg(
    output_dir: Path,
    map_result,
    progress: Progress,
):
    """
    График первой производной offset.

    Здесь хорошо видно:

        линейный drift
        нелинейность
        резкие выбросы
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "АНАЛИЗ DRIFT",
        0.0,
    )

    path = (
        output_dir
        / FINAL_DRIFT_JPEG_NAME
    )

    sync_map = map_result[
        "map"
    ]

    time_axis = (
        sync_map[
            "target_time"
        ]
    )

    offset = (
        sync_map[
            "offset"
        ]
    )

    if len(offset) >= 3:

        slope = np.gradient(
            offset,
            time_axis,
        )

    else:

        slope = np.zeros_like(
            offset
        )

    fig = plt.figure(
        figsize=(18, 8),
        dpi=150,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    ax.plot(
        time_axis,
        slope,
        linewidth=1.5,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title(
        "Local Offset Drift Rate"
    )

    ax.set_xlabel(
        "Target time, seconds"
    )

    ax.set_ylabel(
        "d(offset) / dt"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

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
        f"Drift graph: {path.name}",
    )

    return path


# ============================================================
# ТЕКСТОВЫЙ ФИНАЛЬНЫЙ ОТЧЁТ
# ============================================================

def save_final_report(
    output_dir: Path,
    selected: SelectedTracks,
    map_result,
    evaluation,
):
    """
    Читабельный отчёт для пользователя.
    """

    path = (
        output_dir
        / FINAL_REPORT_NAME
    )

    metrics = map_result[
        "metrics"
    ]

    distribution = evaluation[
        "distribution"
    ]

    drift = evaluation[
        "drift"
    ]

    lines = []

    lines.append(
        "UNIVERSAL AUDIO WAVEFORM SYNC"
    )

    lines.append(
        "=" * 78
    )

    lines.append(
        ""
    )

    lines.append(
        "REFERENCE:"
    )

    lines.append(
        str(
            selected.reference.path
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "TARGET:"
    )

    lines.append(
        str(
            selected.target.path
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "MAP VERDICT:"
    )

    lines.append(
        f"  {evaluation['verdict']}"
    )

    lines.append(
        f"  Score: {evaluation['score']:.2f}/100"
    )

    lines.append(
        ""
    )

    lines.append(
        "MAP:"
    )

    lines.append(
        f"  Start offset : "
        f"{metrics['map_start_offset']:+.6f} s"
    )

    lines.append(
        f"  End offset   : "
        f"{metrics['map_end_offset']:+.6f} s"
    )

    lines.append(
        f"  Total change : "
        f"{metrics['total_offset_change']:+.6f} s"
    )

    lines.append(
        f"  Min offset   : "
        f"{metrics['map_min_offset']:+.6f} s"
    )

    lines.append(
        f"  Max offset   : "
        f"{metrics['map_max_offset']:+.6f} s"
    )

    lines.append(
        ""
    )

    lines.append(
        "ANCHOR QUALITY:"
    )

    lines.append(
        f"  Raw points       : "
        f"{metrics['raw_point_count']}"
    )

    lines.append(
        f"  Map points       : "
        f"{metrics['map_point_count']}"
    )

    lines.append(
        f"  Coverage         : "
        f"{distribution['coverage'] * 100:.2f}%"
    )

    lines.append(
        f"  Median spacing   : "
        f"{distribution['median_spacing']:.3f} s"
    )

    lines.append(
        f"  Maximum gap      : "
        f"{distribution['max_spacing']:.3f} s"
    )

    lines.append(
        ""
    )

    lines.append(
        "MAP FIT:"
    )

    lines.append(
        f"  Mean error       : "
        f"{metrics['map_fit_mean_error']:.6f} s"
    )

    lines.append(
        f"  Median error     : "
        f"{metrics['map_fit_median_error']:.6f} s"
    )

    lines.append(
        f"  Maximum error    : "
        f"{metrics['map_fit_max_error']:.6f} s"
    )

    lines.append(
        ""
    )

    lines.append(
        "DRIFT:"
    )

    lines.append(
        f"  Mean slope       : "
        f"{drift['slope_mean']:+.12f}"
    )

    lines.append(
        f"  Min slope        : "
        f"{drift['slope_min']:+.12f}"
    )

    lines.append(
        f"  Max slope        : "
        f"{drift['slope_max']:+.12f}"
    )

    lines.append(
        f"  Max abs slope    : "
        f"{drift['slope_abs_max']:.12f}"
    )

    lines.append(
        f"  Max curvature    : "
        f"{drift['curvature_abs_max']:.12f}"
    )

    lines.append(
        ""
    )

    lines.append(
        "WARNINGS:"
    )

    if evaluation[
        "warnings"
    ]:

        for warning in evaluation[
            "warnings"
        ]:

            lines.append(
                f"  - {warning}"
            )

    else:

        lines.append(
            "  none"
        )

    lines.append(
        ""
    )

    lines.append(
        "PROBLEMS:"
    )

    if evaluation[
        "problems"
    ]:

        for problem in evaluation[
            "problems"
        ]:

            lines.append(
                f"  - {problem}"
            )

    else:

        lines.append(
            "  none"
        )

    lines.append(
        ""
    )

    lines.append(
        "=" * 78
    )

    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "Audio correction has NOT been applied yet."
    )

    lines.append(
        "The source TARGET file remains untouched."
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
# ПОЛНАЯ ОЦЕНКА КАРТЫ
# ============================================================

def finalize_map_analysis(
    selected: SelectedTracks,
    output_dir: Path,
    alignment,
    map_result,
    progress: Progress,
):
    """
    Полностью завершает математическую часть.
    """

    target_duration = (
        selected.target.duration
    )

    progress.stage(
        "ОЦЕНКА КАЧЕСТВА ФИНАЛЬНОЙ КАРТЫ",
        0.0,
    )

    evaluation = (
        evaluate_sync_map(
            map_result,
            target_duration,
        )
    )

    progress.update(
        35.0,
        (
            f"Verdict: "
            f"{evaluation['verdict']}"
        ),
    )

    # --------------------------------------------------------
    # Сохраняем метрики с оценкой.
    # --------------------------------------------------------

    final_metrics = {
        "map_metrics":
            map_result[
                "metrics"
            ],

        "evaluation":
            evaluation,
    }

    metrics_path = (
        output_dir
        / FINAL_METRICS_NAME
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    progress.update(
        55.0,
        "Metrics saved",
    )

    # --------------------------------------------------------
    # Графики.
    # --------------------------------------------------------

    save_final_map_jpeg(
        output_dir,
        map_result,
        evaluation,
        progress,
    )

    save_drift_analysis_jpeg(
        output_dir,
        map_result,
        progress,
    )

    # --------------------------------------------------------
    # Текстовый отчёт.
    # --------------------------------------------------------

    report_path = (
        save_final_report(
            output_dir,
            selected,
            map_result,
            evaluation,
        )
    )

    progress.done(
        100.0,
        "Финальный анализ карты завершён",
    )

    return {
        "evaluation":
            evaluation,

        "metrics":
            final_metrics,

        "report_path":
            report_path,
    }


# ============================================================
# ПЕЧАТЬ ФИНАЛЬНОГО ВЕРДИКТА
# ============================================================

def print_final_verdict(
    map_result,
    final_analysis,
    output_dir,
):
    """
    Красиво выводит итоговую информацию.
    """

    evaluation = (
        final_analysis[
            "evaluation"
        ]
    )

    metrics = map_result[
        "metrics"
    ]

    print()
    print("=" * 78)
    print("ФИНАЛЬНЫЙ ВЕРДИКТ ПО КАРТЕ")
    print("=" * 78)

    print()

    print(
        "ОЦЕНКА:"
    )

    print(
        f"  {evaluation['verdict']}"
    )

    print(
        f"  Score: "
        f"{evaluation['score']:.1f}/100"
    )

    print()

    print(
        "Карта:"
    )

    print(
        "  начальный offset:",
        f"{metrics['map_start_offset']:+.6f} s",
    )

    print(
        "  конечный offset :",
        f"{metrics['map_end_offset']:+.6f} s",
    )

    print(
        "  общий drift     :",
        f"{metrics['total_offset_change']:+.6f} s",
    )

    print()

    print(
        "Покрытие якорями:",
        f"{evaluation['coverage'] * 100:.2f}%",
    )

    print(
        "Median map error:",
        f"{evaluation['fit_error']:.6f} s",
    )

    print(
        "Max map error:",
        f"{evaluation['max_fit_error']:.6f} s",
    )

    print()

    if evaluation[
        "problems"
    ]:

        print(
            "ПРОБЛЕМЫ:"
        )

        for item in evaluation[
            "problems"
        ]:

            print(
                "  !",
                item,
            )

        print()

    if evaluation[
        "warnings"
    ]:

        print(
            "ПРЕДУПРЕЖДЕНИЯ:"
        )

        for item in evaluation[
            "warnings"
        ]:

            print(
                "  -",
                item,
            )

        print()

    print(
        "Рабочая папка:"
    )

    print(
        output_dir
    )

    print()

    print(
        "Финальный отчёт:"
    )

    print(
        final_analysis[
            "report_path"
        ]
    )

    print()

    # --------------------------------------------------------
    # Пока аудио автоматически не создаём.
    #
    # Даже GOOD не означает автоматически применять карту.
    # Пользователь должен увидеть карту и дать команду.
    # --------------------------------------------------------

    if evaluation[
        "verdict"
    ] == "GOOD":

        print(
            "Карта выглядит хорошо."
        )

        print(
            "На следующем этапе можно "
            "переходить к применению."
        )

    elif evaluation[
        "verdict"
    ] == "ACCEPTABLE":

        print(
            "Карта выглядит приемлемо,"
            " но содержит некоторые предупреждения."
        )

        print(
            "Следующий этап сможет применить её "
            "только после дополнительной проверки."
        )

    elif evaluation[
        "verdict"
    ] == "SUSPICIOUS":

        print(
            "Карта вызывает сомнения."
        )

        print(
            "Автоматическое применение "
            "пока не рекомендуется."
        )

    else:

        print(
            "Карта выглядит плохой."
        )

        print(
            "Применять её к аудио нельзя."
        )

# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC
#
# ЧАСТЬ 7
#
# BEFORE / AFTER VALIDATION
#
# Проверяем:
#
#     Reference
#         vs
#     исходный Target
#
# и
#
#     Reference
#         vs
#     Target после применения временной карты
#
# Никакой перекодировки аудио здесь нет.
#
# ============================================================


# ============================================================
# НАСТРОЙКИ ВАЛИДАЦИИ
# ============================================================

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
