# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: confidence.py
# ============================================================
# Confidence classification and confidence-aware map construction/reporting.
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
from mapper import *

CONF_STRONG_CORR = 0.55


# Очень сильная корреляция.
CONF_EXCELLENT_CORR = 0.70


# Минимальный margin для уверенного пика.
CONF_STRONG_MARGIN = 0.025


# Если корреляция ниже этого уровня,
# локальное совпадение считаем практически ненадёжным.
CONF_WEAK_CORR = 0.40


# Даже если correlation чуть ниже сильного порога,
# точка может быть оставлена, если она очень хорошо
# согласуется с соседней картой.
CONF_TRACKING_TOLERANCE_SEC = 0.20


# Максимальный скачок offset,
# который может реально считаться частью основной карты
# между соседними 30-секундными точками.
CONF_MAX_ANCHOR_JUMP_SEC = 0.35


# Если точка отличается от двухстороннего тренда
# больше этого значения — не доверяем ей.
CONF_MAX_TREND_ERROR_SEC = 0.30


# Для реконструкции сложного участка используем
# соседние хорошие точки.
CONF_CONTEXT_RADIUS_POINTS = 4


# Названия выходных файлов.
CONFIDENCE_MAP_CSV = (
    "confidence_aware_map.csv"
)

CONFIDENCE_MAP_JPEG = (
    "13_CONFIDENCE_AWARE_MAP.jpg"
)

CONFIDENCE_MAP_REPORT = (
    "CONFIDENCE_AWARE_REPORT.txt"
)

CONFIDENCE_POINTS_CSV = (
    "confidence_classification.csv"
)


# ============================================================
# КЛАССИФИКАЦИЯ ЛОКАЛЬНЫХ ТОЧЕК
# ============================================================

def classify_local_measurements(
    local_results,
):
    """
    Каждая локальная точка получает класс:

        STRONG
        ACCEPTABLE
        WEAK
        INVALID

    Главное отличие от старого алгоритма:

    слабая точка НЕ становится якорем только потому,
    что tracking до неё дошёл.
    """

    points = extract_raw_points(
        local_results
    )

    times = points[
        "time"
    ]

    offsets = points[
        "offset"
    ]

    correlations = points[
        "correlation"
    ]

    margins = points[
        "margin"
    ]

    classes = np.empty(
        len(times),
        dtype=object,
    )

    confidence = np.zeros(
        len(times),
        dtype=np.float64,
    )

    for index in range(
        len(times)
    ):

        corr = correlations[
            index
        ]

        margin = margins[
            index
        ]

        # ----------------------------------------------------
        # INVALID
        # ----------------------------------------------------

        if (
            not math.isfinite(corr)
            or
            not math.isfinite(margin)
        ):

            classes[index] = (
                "INVALID"
            )

            confidence[index] = 0.0

            continue

        # ----------------------------------------------------
        # STRONG
        # ----------------------------------------------------

        if (
            corr
            >= CONF_STRONG_CORR
            and
            margin
            >= CONF_STRONG_MARGIN
        ):

            classes[index] = (
                "STRONG"
            )

            corr_score = np.clip(
                (
                    corr
                    - CONF_STRONG_CORR
                )
                /
                (
                    1.0
                    - CONF_STRONG_CORR
                ),
                0.0,
                1.0,
            )

            margin_score = np.clip(
                (
                    margin
                    - CONF_STRONG_MARGIN
                )
                /
                (
                    0.10
                    - CONF_STRONG_MARGIN
                ),
                0.0,
                1.0,
            )

            confidence[index] = (
                0.65
                * corr_score
                +
                0.35
                * margin_score
                + 1.0
            )

            continue

        # ----------------------------------------------------
        # ACCEPTABLE
        # ----------------------------------------------------

        if (
            corr
            >= 0.45
            and
            margin
            >= 0.010
        ):

            classes[index] = (
                "ACCEPTABLE"
            )

            corr_score = np.clip(
                (
                    corr
                    - 0.45
                )
                /
                0.55,
                0.0,
                1.0,
            )

            margin_score = np.clip(
                margin
                /
                0.05,
                0.0,
                1.0,
            )

            confidence[index] = (
                0.50
                * corr_score
                +
                0.50
                * margin_score
            )

            continue

        # ----------------------------------------------------
        # WEAK
        # ----------------------------------------------------

        if (
            corr
            >= CONF_WEAK_CORR
        ):

            classes[index] = (
                "WEAK"
            )

            confidence[index] = (
                0.25
                * np.clip(
                    corr,
                    0.0,
                    1.0,
                )
            )

            continue

        # ----------------------------------------------------
        # INVALID / BAD
        # ----------------------------------------------------

        classes[index] = (
            "INVALID"
        )

        confidence[index] = 0.0

    return {
        "points":
            points,

        "class":
            classes,

        "confidence":
            confidence,
    }


# ============================================================
# ПОИСК СИЛЬНОГО СОСЕДА СЛЕВА
# ============================================================

def find_previous_strong_index(
    classes,
    index,
):
    """
    Ближайший STRONG слева.
    """

    for current in range(
        index - 1,
        -1,
        -1,
    ):

        if classes[
            current
        ] == "STRONG":

            return current

    return None


# ============================================================
# ПОИСК СИЛЬНОГО СОСЕДА СПРАВА
# ============================================================

def find_next_strong_index(
    classes,
    index,
):
    """
    Ближайший STRONG справа.
    """

    for current in range(
        index + 1,
        len(classes),
    ):

        if classes[
            current
        ] == "STRONG":

            return current

    return None


# ============================================================
# ДВУСТОРОННИЙ ПРОГНОЗ
# ============================================================

def calculate_bidirectional_prediction(
    times,
    offsets,
    classes,
    index,
):
    """
    Строим независимый прогноз offset
    двумя способами:

        LEFT
        RIGHT

    Если оба прогноза близки —
    участок очень вероятно надёжен.

    Если расходятся —
    текущая точка подозрительна.
    """

    left_index = (
        find_previous_strong_index(
            classes,
            index,
        )
    )

    right_index = (
        find_next_strong_index(
            classes,
            index,
        )
    )

    left_prediction = None
    right_prediction = None

    if left_index is not None:

        left_prediction = float(
            offsets[
                left_index
            ]
        )

    if right_index is not None:

        right_prediction = float(
            offsets[
                right_index
            ]
        )

    # --------------------------------------------------------
    # Если имеются обе стороны,
    # интерполируем между ними.
    # --------------------------------------------------------

    if (
        left_index is not None
        and
        right_index is not None
    ):

        left_time = (
            times[
                left_index
            ]
        )

        right_time = (
            times[
                right_index
            ]
        )

        current_time = (
            times[
                index
            ]
        )

        if (
            right_time
            >
            left_time
        ):

            fraction = (
                current_time
                - left_time
            ) / (
                right_time
                - left_time
            )

            bilateral = (
                offsets[
                    left_index
                ]
                +
                fraction
                * (
                    offsets[
                        right_index
                    ]
                    -
                    offsets[
                        left_index
                    ]
                )
            )

            return {
                "prediction":
                    float(
                        bilateral
                    ),

                "left":
                    left_prediction,

                "right":
                    right_prediction,

                "left_index":
                    left_index,

                "right_index":
                    right_index,
            }

    # --------------------------------------------------------
    # Если только левая.
    # --------------------------------------------------------

    if left_prediction is not None:

        return {
            "prediction":
                left_prediction,

            "left":
                left_prediction,

            "right":
                None,

            "left_index":
                left_index,

            "right_index":
                None,
        }

    # --------------------------------------------------------
    # Если только правая.
    # --------------------------------------------------------

    if right_prediction is not None:

        return {
            "prediction":
                right_prediction,

            "left":
                None,

            "right":
                right_prediction,

            "left_index":
                None,

            "right_index":
                right_index,
        }

    return {
        "prediction":
            None,

        "left":
            None,

        "right":
            None,

        "left_index":
            None,

        "right_index":
            None,
    }


def _apply_strong_anchor_continuity_guard(
    times,
    offsets,
    classes,
    keep,
):
    """
    Убирает одиночные STRONG-скачки, которые математически выглядят
    как уверенная корреляция, но не поддерживаются соседними якорями.

    v6.6 доверял любой STRONG-точке напрямую. В The Beyond это позволило
    одиночному измерению +0.766 s около 1195 s стать полноценным якорем,
    после чего интерполяция построила локальную полусекундную "иглу".

    v6.7 сохраняет STRONG как исходный класс, но требует геометрической
    согласованности с ближайшими STRONG-соседями. Одиночный скачок больше
    допустимой кривизны или скачка не становится якорем. Само исходное
    измерение при этом сохраняется в отчёте как OUTLIER_IGNORED.
    """

    strong_indices = np.where(
        np.asarray(classes) == "STRONG"
    )[0]

    if len(strong_indices) < 3:
        return keep, np.zeros(len(times), dtype=bool)

    rejected = np.zeros(len(times), dtype=bool)

    for position in range(1, len(strong_indices) - 1):
        index = int(strong_indices[position])
        left_index = int(strong_indices[position - 1])
        right_index = int(strong_indices[position + 1])

        left_gap = float(times[index] - times[left_index])
        right_gap = float(times[right_index] - times[index])
        if left_gap <= 0.0 or right_gap <= 0.0:
            continue

        # Ожидаемый локальный offset по двум соседним STRONG-якорям.
        fraction = left_gap / (left_gap + right_gap)
        local_prediction = (
            float(offsets[left_index])
            + fraction
            * (float(offsets[right_index]) - float(offsets[left_index]))
        )
        trend_error = abs(
            float(offsets[index]) - local_prediction
        )

        left_jump = abs(
            float(offsets[index]) - float(offsets[left_index])
        )
        right_jump = abs(
            float(offsets[index]) - float(offsets[right_index])
        )

        singleton_spike = (
            trend_error > CONF_MAX_TREND_ERROR_SEC
            and left_jump > CONF_MAX_ANCHOR_JUMP_SEC
            and right_jump > CONF_MAX_ANCHOR_JUMP_SEC
        )

        if singleton_spike:
            keep[index] = False
            rejected[index] = True

    return keep, rejected


# ============================================================
# ПОСТРОЕНИЕ CONFIDENCE-AWARE ЯКОРЕЙ
# ============================================================

def build_confidence_aware_anchors(
    classified,
    progress,
):
    """
    Главная функция нового метода.

    Алгоритм:

        STRONG точки
            ↓
        строят основной коридор
            ↓
        ACCEPTABLE проверяем относительно
        этого коридора
            ↓
        WEAK / INVALID не влияют
            ↓
        сложные участки восстанавливаются
        по двум сторонам
    """

    points = classified[
        "points"
    ]

    classes = classified[
        "class"
    ]

    confidence = classified[
        "confidence"
    ]

    times = points[
        "time"
    ]

    offsets = points[
        "offset"
    ]

    correlations = points[
        "correlation"
    ]

    margins = points[
        "margin"
    ]

    # --------------------------------------------------------
    # Новый набор якорей.
    # --------------------------------------------------------

    keep = np.zeros(
        len(times),
        dtype=bool,
    )

    repaired = np.zeros(
        len(times),
        dtype=bool,
    )

    predicted_offset = np.full(
        len(times),
        np.nan,
        dtype=np.float64,
    )

    prediction_error = np.full(
        len(times),
        np.nan,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Сначала все STRONG, после чего проходит отдельная проверка
    # геометрической непрерывности.
    # --------------------------------------------------------

    keep[
        classes == "STRONG"
    ] = True

    keep, strong_continuity_rejected = (
        _apply_strong_anchor_continuity_guard(
            times,
            offsets,
            classes,
            keep,
        )
    )

    # --------------------------------------------------------
    # Теперь проходим по ACCEPTABLE.
    # --------------------------------------------------------

    for index in range(
        len(times)
    ):

        if (
            classes[index]
            != "ACCEPTABLE"
        ):
            continue

        prediction = (
            calculate_bidirectional_prediction(
                times,
                offsets,
                classes,
                index,
            )
        )

        predicted = (
            prediction[
                "prediction"
            ]
        )

        if predicted is None:
            continue

        predicted_offset[
            index
        ] = predicted

        error = abs(
            offsets[index]
            - predicted
        )

        prediction_error[
            index
        ] = error

        # ----------------------------------------------------
        # Если ACCEPTABLE точка хорошо ложится
        # между двумя STRONG соседями —
        # сохраняем.
        # ----------------------------------------------------

        if (
            error
            <= CONF_MAX_TREND_ERROR_SEC
        ):

            keep[
                index
            ] = True

        else:

            # Не удаляем окончательно:
            # она будет использоваться только
            # как диагностическая информация.
            repaired[
                index
            ] = True

    # --------------------------------------------------------
    # WEAK точки никогда не становятся
    # автономными якорями.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Теперь восстанавливаем сложные участки.
    #
    # Между STRONG / ACCEPTABLE якорями
    # используем геометрически устойчивую интерполяцию.
    # --------------------------------------------------------

    anchor_indices = (
        np.where(
            keep
        )[0]
    )

    if len(
        anchor_indices
    ) < 2:

        raise RuntimeError(
            "После confidence-фильтра "
            "осталось меньше двух надёжных якорей."
        )

    # --------------------------------------------------------
    # Формируем очищенную карту якорей.
    # --------------------------------------------------------

    anchor_times = (
        times[
            anchor_indices
        ]
    )

    anchor_offsets = (
        offsets[
            anchor_indices
        ]
    )

    anchor_correlations = (
        correlations[
            anchor_indices
        ]
    )

    anchor_margins = (
        margins[
            anchor_indices
        ]
    )

    anchor_confidence = (
        confidence[
            anchor_indices
        ]
    )

    progress.update(
        60.0,
        (
            f"Надёжные якоря: {len(anchor_indices)} | "
            f"отброшены одиночные STRONG-скачки: "
            f"{int(np.sum(strong_continuity_rejected))}"
        ),
    )

    # --------------------------------------------------------
    # Для каждого исходного измерения
    # рассчитываем, насколько оно согласуется
    # с новой картой.
    # --------------------------------------------------------

    interpolated_offset = np.interp(
        times,
        anchor_times,
        anchor_offsets,
    )

    residual = np.abs(
        offsets
        - interpolated_offset
    )

    progress.update(
        85.0,
        "Проверка сложных участков",
    )

    # --------------------------------------------------------
    # Классификация финальной точки.
    # --------------------------------------------------------

    final_status = np.empty(
        len(times),
        dtype=object,
    )

    for index in range(
        len(times)
    ):

        if keep[index]:

            final_status[index] = (
                "ANCHOR"
            )

        elif (
            residual[index]
            <= CONF_TRACKING_TOLERANCE_SEC
        ):

            final_status[index] = (
                "FOLLOWER"
            )

        elif (
            classes[index]
            == "WEAK"
        ):

            final_status[index] = (
                "WEAK_IGNORED"
            )

        else:

            final_status[index] = (
                "OUTLIER_IGNORED"
            )

    result = {
        "times":
            times,

        "offsets":
            offsets,

        "correlations":
            correlations,

        "margins":
            margins,

        "classes":
            classes,

        "confidence":
            confidence,

        "keep":
            keep,

        "strong_continuity_rejected":
            strong_continuity_rejected,

        "repaired":
            repaired,

        "predicted_offset":
            predicted_offset,

        "prediction_error":
            prediction_error,

        "interpolated_offset":
            interpolated_offset,

        "residual":
            residual,

        "final_status":
            final_status,

        "anchor_times":
            anchor_times,

        "anchor_offsets":
            anchor_offsets,

        "anchor_correlations":
            anchor_correlations,

        "anchor_margins":
            anchor_margins,

        "anchor_confidence":
            anchor_confidence,
    }

    progress.done(
        100.0,
        "Confidence-aware analysis завершён",
    )

    return result


# ============================================================
# ПОСТРОЕНИЕ ПОЛНОЙ CONFIDENCE-АWARE MAP
# ============================================================

def build_confidence_aware_map(
    confidence_result,
    target_duration,
    progress,
):
    """
    Создаём непрерывную карту только из надёжных точек.

    Слабые waveform участки:
        НЕ ДВИГАЮТ КАРТУ.

    Они получают offset из устойчивой траектории.
    """

    progress.stage(
        "ПОСТРОЕНИЕ CONFIDENCE-AWARE MAP",
        0.0,
    )

    anchor_times = (
        confidence_result[
            "anchor_times"
        ]
    )

    anchor_offsets = (
        confidence_result[
            "anchor_offsets"
        ]
    )

    anchor_confidence = (
        confidence_result[
            "anchor_confidence"
        ]
    )

    if len(
        anchor_times
    ) < 2:

        raise RuntimeError(
            "Недостаточно надёжных "
            "якорей для карты."
        )

    # --------------------------------------------------------
    # Полная временная сетка.
    # --------------------------------------------------------

    map_time = build_exact_time_grid(
        0.0,
        target_duration,
        MAP_STEP_SEC,
    )

    # --------------------------------------------------------
    # Интерполяция только по доверенным точкам.
    # --------------------------------------------------------

    map_offset = np.interp(
        map_time,
        anchor_times,
        anchor_offsets,
    )

    map_confidence = np.interp(
        map_time,
        anchor_times,
        anchor_confidence,
    )

    # --------------------------------------------------------
    # Небольшое сглаживание.
    #
    # Здесь уже не должны находиться
    # слабые ложные пики.
    # --------------------------------------------------------

    map_offset = (
        smooth_offset_curve(
            map_offset,
            2,
        )
    )

    reference_time = (
        map_time
        + map_offset
    )

    differences = np.diff(
        reference_time
    )

    if np.any(
        differences <= 0
    ):

        raise RuntimeError(
            "Confidence-aware map "
            "нарушает монотонность."
        )

    progress.update(
        80.0,
        (
            f"Map points: "
            f"{len(map_time):,}"
        ),
    )

    result = {
        "target_time":
            map_time,

        "offset":
            map_offset,

        "reference_time":
            reference_time,

        "confidence":
            map_confidence,

        "anchor_time":
            anchor_times,

        "anchor_offset":
            anchor_offsets,
    }

    progress.done(
        100.0,
        "Confidence-aware map готова",
    )

    return result


# ============================================================
# СРАВНЕНИЕ СТАРОЙ И НОВОЙ КАРТЫ
# ============================================================

def compare_old_new_maps(
    old_map,
    new_map,
):
    """
    Сравниваем старую карту и новую.
    """

    common_time = np.arange(
        0.0,
        min(
            old_map[
                "target_time"
            ][-1],
            new_map[
                "target_time"
            ][-1],
        ),
        1.0,
        dtype=np.float64,
    )

    old_offset = np.interp(
        common_time,
        old_map[
            "target_time"
        ],
        old_map[
            "offset"
        ],
    )

    new_offset = np.interp(
        common_time,
        new_map[
            "target_time"
        ],
        new_map[
            "offset"
        ],
    )

    difference = (
        new_offset
        - old_offset
    )

    return {
        "time":
            common_time,

        "old":
            old_offset,

        "new":
            new_offset,

        "difference":
            difference,
    }


# ============================================================
# ГРАФИК CONFIDENCE MAP
# ============================================================

def save_confidence_map_jpeg(
    output_dir,
    old_map,
    new_map,
    confidence_result,
    progress,
):
    """
    Один из ключевых графиков.

    Показываем:

        старая карта
        новая карта
        raw points
        weak points
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ГРАФИК CONFIDENCE-AWARE MAP",
        0.0,
    )

    path = (
        output_dir
        / CONFIDENCE_MAP_JPEG
    )

    times = (
        confidence_result[
            "times"
        ]
    )

    offsets = (
        confidence_result[
            "offsets"
        ]
    )

    classes = (
        confidence_result[
            "classes"
        ]
    )

    strong = (
        classes
        == "STRONG"
    )

    acceptable = (
        classes
        == "ACCEPTABLE"
    )

    weak = (
        classes
        == "WEAK"
    )

    invalid = (
        classes
        == "INVALID"
    )

    fig = plt.figure(
        figsize=(18, 11),
        dpi=160,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    # --------------------------------------------------------
    # Старая карта
    # --------------------------------------------------------

    ax.plot(
        old_map[
            "target_time"
        ],
        old_map[
            "offset"
        ],
        linewidth=1.4,
        alpha=0.65,
        label="Previous map",
    )

    # --------------------------------------------------------
    # Новая карта
    # --------------------------------------------------------

    ax.plot(
        new_map[
            "target_time"
        ],
        new_map[
            "offset"
        ],
        linewidth=2.4,
        label="Confidence-aware map",
    )

    # --------------------------------------------------------
    # Raw points
    # --------------------------------------------------------

    if np.any(strong):

        ax.scatter(
            times[
                strong
            ],
            offsets[
                strong
            ],
            s=20,
            alpha=0.75,
            label="STRONG",
        )

    if np.any(acceptable):

        ax.scatter(
            times[
                acceptable
            ],
            offsets[
                acceptable
            ],
            s=16,
            alpha=0.55,
            label="ACCEPTABLE",
        )

    if np.any(weak):

        ax.scatter(
            times[
                weak
            ],
            offsets[
                weak
            ],
            s=18,
            alpha=0.35,
            label="WEAK",
        )

    if np.any(invalid):

        ax.scatter(
            times[
                invalid
            ],
            offsets[
                invalid
            ],
            s=22,
            alpha=0.35,
            label="INVALID",
        )

    ax.set_title(
        "Confidence-Aware Synchronization Map"
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
        "Confidence-aware график готов",
    )

    return path


# ============================================================
# СОХРАНЕНИЕ CLASSIFICATION CSV
# ============================================================

def save_confidence_points_csv(
    output_dir,
    confidence_result,
    progress,
):
    """
    Подробная классификация каждого локального измерения.
    """

    progress.stage(
        "СОХРАНЕНИЕ CONFIDENCE CLASSIFICATION",
        0.0,
    )

    path = (
        output_dir
        / CONFIDENCE_POINTS_CSV
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
                "raw_offset_sec",
                "correlation",
                "margin",
                "class",
                "confidence",
                "kept",
                "predicted_offset",
                "prediction_error",
                "interpolated_offset",
                "residual",
                "final_status",
            ]
        )

        total = len(
            confidence_result[
                "times"
            ]
        )

        for index in range(
            total
        ):

            writer.writerow(
                [
                    (
                        f"{confidence_result['times'][index]:.6f}"
                    ),

                    (
                        f"{confidence_result['offsets'][index]:.6f}"
                    ),

                    (
                        f"{confidence_result['correlations'][index]:.6f}"
                    ),

                    (
                        f"{confidence_result['margins'][index]:.6f}"
                    ),

                    confidence_result[
                        "classes"
                    ][index],

                    (
                        f"{confidence_result['confidence'][index]:.6f}"
                    ),

                    (
                        "YES"
                        if confidence_result[
                            "keep"
                        ][index]
                        else "NO"
                    ),

                    (
                        ""
                        if not np.isfinite(
                            confidence_result[
                                "predicted_offset"
                            ][index]
                        )
                        else
                        f"{confidence_result['predicted_offset'][index]:.6f}"
                    ),

                    (
                        ""
                        if not np.isfinite(
                            confidence_result[
                                "prediction_error"
                            ][index]
                        )
                        else
                        f"{confidence_result['prediction_error'][index]:.6f}"
                    ),

                    (
                        f"{confidence_result['interpolated_offset'][index]:.6f}"
                    ),

                    (
                        f"{confidence_result['residual'][index]:.6f}"
                    ),

                    confidence_result[
                        "final_status"
                    ][index],
                ]
            )

            if (
                index % 25 == 0
                or index == total - 1
            ):

                progress.update(
                    index
                    / max(
                        1,
                        total - 1,
                    )
                    * 100.0,
                    "Запись confidence classification",
                )

    progress.done(
        100.0,
        "Confidence classification сохранена",
    )

    return path


# ============================================================
# СОХРАНЕНИЕ CONFIDENCE MAP
# ============================================================

def save_confidence_map_csv(
    output_dir,
    confidence_map,
    progress,
):
    """
    Финальная карта нового confidence-aware алгоритма.
    """

    progress.stage(
        "СОХРАНЕНИЕ CONFIDENCE-AWARE MAP",
        0.0,
    )

    path = (
        output_dir
        / CONFIDENCE_MAP_CSV
    )

    target_time = (
        confidence_map[
            "target_time"
        ]
    )

    offset = (
        confidence_map[
            "offset"
        ]
    )

    reference_time = (
        confidence_map[
            "reference_time"
        ]
    )

    confidence = (
        confidence_map[
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
                    "Запись confidence-aware map",
                )

    progress.done(
        100.0,
        "Confidence-aware map сохранена",
    )

    return path


# ============================================================
# ОТЧЁТ CONFIDENCE-AWARE
# ============================================================

def save_confidence_report(
    output_dir,
    confidence_result,
    old_map,
    new_map,
):
    """
    Отчёт показывает,
    сколько точек было реально использовано.
    """

    classes = (
        confidence_result[
            "classes"
        ]
    )

    strong_count = int(
        np.sum(
            classes
            == "STRONG"
        )
    )

    acceptable_count = int(
        np.sum(
            classes
            == "ACCEPTABLE"
        )
    )

    weak_count = int(
        np.sum(
            classes
            == "WEAK"
        )
    )

    invalid_count = int(
        np.sum(
            classes
            == "INVALID"
        )
    )

    old_offset = (
        old_map[
            "offset"
        ]
    )

    new_offset = (
        new_map[
            "offset"
        ]
    )

    time_axis = (
        new_map[
            "target_time"
        ]
    )

    common_time = np.arange(
        0.0,
        min(
            old_map[
                "target_time"
            ][-1],
            new_map[
                "target_time"
            ][-1],
        ),
        1.0,
        dtype=np.float64,
    )

    old_at_time = np.interp(
        common_time,
        old_map[
            "target_time"
        ],
        old_offset,
    )

    new_at_time = np.interp(
        common_time,
        time_axis,
        new_offset,
    )

    difference = (
        new_at_time
        - old_at_time
    )

    path = (
        output_dir
        / CONFIDENCE_MAP_REPORT
    )

    lines = []

    lines.append(
        "CONFIDENCE-AWARE SYNCHRONIZATION"
    )

    lines.append(
        "=" * 78
    )

    lines.append(
        ""
    )

    lines.append(
        f"STRONG points: "
        f"{strong_count}"
    )

    lines.append(
        f"ACCEPTABLE points: "
        f"{acceptable_count}"
    )

    lines.append(
        f"WEAK points: "
        f"{weak_count}"
    )

    lines.append(
        f"INVALID points: "
        f"{invalid_count}"
    )

    continuity_rejected = int(
        np.sum(
            confidence_result.get(
                "strong_continuity_rejected",
                np.zeros_like(confidence_result["times"], dtype=bool),
            )
        )
    )

    lines.append(
        f"STRONG singleton jumps rejected: {continuity_rejected}"
    )

    lines.append(
        ""
    )

    lines.append(
        "MAP DIFFERENCE FROM PREVIOUS VERSION:"
    )

    lines.append(
        f"  Mean absolute difference: "
        f"{np.mean(np.abs(difference)):.6f} s"
    )

    lines.append(
        f"  Median absolute difference: "
        f"{np.median(np.abs(difference)):.6f} s"
    )

    lines.append(
        f"  Max absolute difference: "
        f"{np.max(np.abs(difference)):.6f} s"
    )

    lines.append(
        ""
    )

    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "Weak and invalid waveform measurements "
        "are NOT allowed to move the main map."
    )

    lines.append(
        ""
    )

    lines.append(
        "The original TARGET remains untouched."
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
# PART 11 — ФИНАЛЬНЫЙ MAIN
#
# Подключает реальные функции, существующие в текущем файле.
#
# ВАЖНО:
#   1. Reference / Target выбираются как AudioInfo.
#   2. Слабые локальные совпадения не используются как якоря.
#   3. Confidence-aware map строится ДО финальной validation.
#   4. repair_map_using_validation() НЕ используется.
#   5. Apply/Render выполняется только после автоматической проверки.
# ============================================================


# ============================================================
# RENDER / APPLY
# ============================================================
