# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: applier.py
# ============================================================
# Manual diagnostics, APPLY map construction and pre-apply reporting.
#
# This module is part of the modular version of the project.
# Do not run it directly; start the pipeline with start_sync.py.
# ============================================================

from __future__ import annotations

import csv
import json
import math

from config import *
from common import *
from mapper import *
from validator import *

MANUAL_CHECK_RADIUS_SEC = 5.0


# Насколько можно корректировать карту
# при подавлении одиночного validation выброса.
#
# 0.0 = полностью доверяем исходной карте
# 1.0 = полностью заменяем локальным прогнозом
#
# Для безопасности используем середину.
OUTLIER_REPAIR_STRENGTH = 0.75


# Шаг итоговой карты, которая будет использоваться
# будущим APPLY.
#
# 0.25 секунды — достаточно подробно для формирования
# локальной скорости, но не создаёт десятки тысяч
# лишних atempo-сегментов.
APPLY_MAP_STEP_SEC = 0.25


# Минимальная длительность одного сегмента.
#
# Слишком короткие atempo-сегменты нам не нужны.
MIN_APPLY_SEGMENT_SEC = 2.0


# Максимально допустимое изменение скорости
# на одном сегменте.
#
# Если карта потребует чего-то экстремального,
# APPLY будет остановлен.
MAX_APPLY_TEMPO = 1.20
MIN_APPLY_TEMPO = 0.80

# ============================================================
# НАСТРОЙКИ BIDIRECTIONAL VALIDATION
# ============================================================

# Независимая проверка делит validation-окно на две половины:
# левую и правую. Истинное совпадение должно вести себя
# устойчиво сразу в обеих половинах, а одиночный ложный пик
# обычно не выдерживает такую проверку.
BIDIR_SIDE_WINDOW_SEC = 10.0
BIDIR_MAX_DISAGREEMENT_SEC = 0.080
BIDIR_MAX_PREDICTION_ERROR_SEC = 0.150
BIDIR_MIN_CORRELATION = 0.30
BIDIR_STRONG_CORRELATION = 0.45

# ============================================================
# НАСТРОЙКИ RENDER / APPLY
# ============================================================

RENDER_MAX_MAPPING_ERROR_SEC = 0.002
RENDER_MIN_SEGMENT_SEC = 1.0
RENDER_MAX_SEGMENT_SEC = 30.0
RENDER_OUTPUT_SUFFIX = "_SYNCED"
RENDER_PLAN_JSON = "RENDER_PLAN.json"
RENDER_REPORT = "RENDER_REPORT.txt"
FINAL_RENDER_VALIDATION_REPORT = "FINAL_RENDER_VALIDATION_REPORT.txt"
FINAL_RENDER_VALIDATION_JSON = "FINAL_RENDER_VALIDATION.json"


# Выходные файлы.

PRE_APPLY_MAP_CSV = (
    "pre_apply_sync_map.csv"
)

PRE_APPLY_REPORT = (
    "PRE_APPLY_REPORT.txt"
)

MANUAL_OUTLIER_REPORT = (
    "MANUAL_AUDACITY_CHECK.txt"
)

APPLY_PLAN_JSON = (
    "apply_plan.json"
)

APPLY_PLAN_JPEG = (
    "12_APPLY_PLAN.jpg"
)


# ============================================================
# ФОРМАТ ВРЕМЕНИ С МИЛЛИСЕКУНДАМИ
# ============================================================

def format_time_millis(
    seconds,
):
    """
    Например:

        317.42
        ->
        00:05:17.420
    """

    try:
        seconds = float(
            seconds
        )
    except Exception:
        return "--:--:--.---"

    if not math.isfinite(seconds):
        return "--:--:--.---"

    seconds = max(
        0.0,
        seconds,
    )

    total_ms = int(
        round(
            seconds * 1000.0
        )
    )

    hours = (
        total_ms
        // 3_600_000
    )

    minutes = (
        total_ms
        % 3_600_000
    ) // 60_000

    secs = (
        total_ms
        % 60_000
    ) // 1000

    millis = (
        total_ms
        % 1000
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{millis:03d}"
    )


# ============================================================
# ПОЛУЧЕНИЕ HARD OUTLIERS
# ============================================================

def extract_manual_review_candidates(
    validation_results,
    map_result,
):
    """Все HARD-кандидаты для диагностики."""
    outliers=find_validation_outliers(validation_results)
    hard=[item for item in outliers if item.get("severity")=="HARD"]
    hard.sort(key=lambda item:item["target_time"])
    return hard


def extract_manual_review_points(
    validation_results,
    map_result,
):
    """Только HARD-точки, подтверждённые bidirectional validation."""
    return [
        item for item in extract_manual_review_candidates(validation_results,map_result)
        if item.get("bidir_status")=="CONFIRMED_DISCREPANCY"
    ]


# ============================================================
# ОТЧЁТ ДЛЯ MANUAL AUDACITY CHECK
# ============================================================

def save_manual_audacity_report(
    output_dir,
    validation_results,
    map_result,
):
    """Подтверждённые HARD и неоднозначные HARD-кандидаты разделены."""
    candidates=extract_manual_review_candidates(validation_results,map_result)
    confirmed=[item for item in candidates if item.get("bidir_status")=="CONFIRMED_DISCREPANCY"]
    ambiguous=[item for item in candidates if item.get("bidir_status")!="CONFIRMED_DISCREPANCY"]
    path=output_dir/MANUAL_OUTLIER_REPORT
    lines=["MANUAL AUDACITY CHECK","="*78,"",f"CONFIRMED HARD DISCREPANCIES: {len(confirmed)}",f"AMBIGUOUS HARD CANDIDATES: {len(ambiguous)}",""]
    if not confirmed:
        lines += ["No HARD validation point was confirmed as a real map discrepancy by bidirectional validation.",""]
    def add_item(title,index,item):
        actual="N/A" if item.get("actual_offset") is None else f"{float(item['actual_offset']):+.6f} sec"
        lines.extend(["-"*78,f"{title} #{index}","",f"Target time: {format_time_millis(item['target_time'])}",f"  {float(item['target_time']):.3f} sec","","Audacity inspection window:",f"  {format_time_millis(max(0.0,float(item['target_time'])-MANUAL_CHECK_RADIUS_SEC))} -> {format_time_millis(float(item['target_time'])+MANUAL_CHECK_RADIUS_SEC)}","",f"Predicted offset: {float(item['predicted_offset']):+.6f} sec",f"Independent actual offset: {actual}",f"Residual: {float(item['residual']):.6f} sec",f"Correlation: {float(item['correlation']):.6f}",f"Bidirectional verification: {item.get('bidir_status','UNRESOLVED')}",""])
    for i,item in enumerate(confirmed,1): add_item("CONFIRMED DISCREPANCY",i,item)
    for i,item in enumerate(ambiguous,1): add_item("AMBIGUOUS CANDIDATE",i,item)
    lines += ["="*78,"","INTERPRETATION:","CONFIRMED DISCREPANCY means both validation sides independently support a deviation from the map.","AMBIGUOUS CANDIDATE means the waveform search did not provide enough independent evidence to confirm a map failure.","Ambiguous candidates do not modify the synchronization map and are diagnostic only."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return path, confirmed


# ============================================================
# ВОССТАНОВЛЕНИЕ ЛОКАЛЬНОГО OFFSET
# ============================================================

def repair_map_using_validation(
    sync_map,
    validation_results,
):
    """
    Очень осторожная защита карты.

    Мы НЕ заменяем карту validation-точками.

    Для каждого HARD/подозрительного участка:

        validation actual
              +
        соседний тренд карты
              ↓
        мягкая коррекция

    Только если validation действительно
    показывает систематическое отклонение.
    """

    target_time = np.asarray(
        sync_map[
            "target_time"
        ],
        dtype=np.float64,
    )

    original_offset = np.asarray(
        sync_map[
            "offset"
        ],
        dtype=np.float64,
    )

    repaired = (
        original_offset.copy()
    )

    hard_points = [
        result
        for result in validation_results
        if (
            result[
                "after_error"
            ] is not None
            and
            abs(
                result[
                    "after_error"
                ]
            )
            >= FINAL_VALIDATION_HARD_OUTLIER_SEC
        )
    ]

    if not hard_points:
        return {
            **sync_map,
            "offset":
                repaired,
            "original_offset":
                original_offset,
            "repaired_points":
                0,
        }

    for item in hard_points:

        time_sec = float(
            item[
                "target_time"
            ]
        )

        actual = item[
            "after_offset"
        ]

        if actual is None:
            continue

        actual = float(
            actual
        )

        index = int(
            np.argmin(
                np.abs(
                    target_time
                    - time_sec
                )
            )
        )

        # ----------------------------------------------------
        # Коррекция применяется не к одной точке,
        # а к мягкому локальному диапазону.
        # ----------------------------------------------------

        sigma = 45.0

        distance = (
            target_time
            - time_sec
        )

        influence = np.exp(
            -0.5
            * (
                distance
                / sigma
            )
            ** 2
        )

        current = (
            repaired[index]
        )

        correction = (
            actual
            - current
        )

        correction *= (
            OUTLIER_REPAIR_STRENGTH
        )

        repaired += (
            influence
            * correction
        )

    # --------------------------------------------------------
    # После коррекции ещё раз проверяем Reference timeline.
    # --------------------------------------------------------

    reference_time = (
        target_time
        + repaired
    )

    differences = np.diff(
        reference_time
    )

    if np.any(
        differences <= 0
    ):

        raise RuntimeError(
            "Защищённая карта после repair "
            "нарушает монотонность времени."
        )

    result = dict(
        sync_map
    )

    result[
        "original_offset"
    ] = original_offset

    result[
        "offset"
    ] = repaired

    result[
        "reference_time"
    ] = reference_time

    result[
        "repaired_points"
    ] = len(
        hard_points
    )

    return result


# ============================================================
# СОЗДАНИЕ РАВНОМЕРНОЙ APPLY MAP
# ============================================================

def build_apply_map(
    sync_map,
    target_duration,
    progress,
):
    """
    Приводим карту к равномерной сетке.

    На выходе:

        target_time
        reference_time
        offset
    """

    progress.stage(
        "ПОДГОТОВКА APPLY MAP",
        0.0,
    )

    original_time = np.asarray(
        sync_map[
            "target_time"
        ],
        dtype=np.float64,
    )

    original_offset = np.asarray(
        sync_map[
            "offset"
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Делаем полную карту от 0 до Target duration.
    #
    # Вне диапазона измерений используем ближайший
    # достоверный offset.
    #
    # Таким образом начало и конец аудио тоже получают
    # определённое положение.
    # --------------------------------------------------------

    if len(
        original_time
    ) < 2:

        raise RuntimeError(
            "Недостаточно точек "
            "для APPLY MAP."
        )

    full_time = build_exact_time_grid(
        0.0,
        target_duration,
        APPLY_MAP_STEP_SEC,
    )

    full_offset = np.interp(
        full_time,
        original_time,
        original_offset,
    )

    full_reference = (
        full_time
        + full_offset
    )

    # --------------------------------------------------------
    # Проверяем монотонность.
    # --------------------------------------------------------

    differences = np.diff(
        full_reference
    )

    if np.any(
        differences <= 0
    ):

        raise RuntimeError(
            "Apply map нарушает "
            "монотонность reference timeline."
        )

    progress.update(
        60.0,
        (
            f"Apply points: "
            f"{len(full_time):,}"
        ),
    )

    progress.update(
        90.0,
        "Проверка локальной скорости",
    )

    # --------------------------------------------------------
    # Локальный tempo.
    #
    # Входной Target interval:
    #
    #     dt_target
    #
    # должен стать:
    #
    #     dt_reference
    #
    # поэтому:
    #
    #     tempo =
    #     dt_target / dt_reference
    #
    # --------------------------------------------------------

    target_delta = np.diff(
        full_time
    )

    reference_delta = np.diff(
        full_reference
    )

    tempo = (
        target_delta
        / reference_delta
    )

    min_tempo = float(
        np.min(
            tempo
        )
    )

    max_tempo = float(
        np.max(
            tempo
        )
    )

    progress.done(
        100.0,
        "Apply map подготовлена",
    )

    return {
        "target_time":
            full_time,

        "offset":
            full_offset,

        "reference_time":
            full_reference,

        "tempo":
            tempo,

        "min_tempo":
            min_tempo,

        "max_tempo":
            max_tempo,
    }


# ============================================================
# ПРОВЕРКА APPLY MAP
# ============================================================

def validate_apply_map_limits(
    apply_map,
):
    """
    Проверяем, не требует ли карта
    экстремального изменения скорости.
    """

    tempo = apply_map[
        "tempo"
    ]

    if len(tempo) == 0:
        raise RuntimeError(
            "Apply map не содержит tempo."
        )

    min_tempo = float(
        np.min(
            tempo
        )
    )

    max_tempo = float(
        np.max(
            tempo
        )
    )

    if (
        min_tempo
        < MIN_APPLY_TEMPO
    ):

        raise RuntimeError(
            (
                "Карта требует слишком "
                f"медленного участка: "
                f"tempo={min_tempo:.4f}"
            )
        )

    if (
        max_tempo
        > MAX_APPLY_TEMPO
    ):

        raise RuntimeError(
            (
                "Карта требует слишком "
                f"быстрого участка: "
                f"tempo={max_tempo:.4f}"
            )
        )


# ============================================================
# СОХРАНЕНИЕ APPLY MAP
# ============================================================

def save_pre_apply_map_csv(
    output_dir,
    apply_map,
    progress,
):
    """
    Сохраняет карту именно в том виде,
    в котором она пойдёт в будущий APPLY.
    """

    progress.stage(
        "СОХРАНЕНИЕ PRE-APPLY MAP",
        0.0,
    )

    path = (
        output_dir
        / PRE_APPLY_MAP_CSV
    )

    target_time = (
        apply_map[
            "target_time"
        ]
    )

    offset = (
        apply_map[
            "offset"
        ]
    )

    reference_time = (
        apply_map[
            "reference_time"
        ]
    )

    tempo = (
        apply_map[
            "tempo"
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
                "tempo_to_reference",
            ]
        )

        for index in range(
            len(target_time)
        ):

            if (
                index
                < len(tempo)
            ):
                tempo_value = (
                    tempo[index]
                )
            elif len(tempo):
                tempo_value = (
                    tempo[-1]
                )
            else:
                tempo_value = 1.0

            writer.writerow(
                [
                    f"{target_time[index]:.6f}",
                    f"{offset[index]:.6f}",
                    f"{reference_time[index]:.6f}",
                    f"{tempo_value:.9f}",
                ]
            )

            if (
                index % 5000 == 0
                or
                index
                == len(target_time) - 1
            ):

                progress.update(
                    index
                    / max(
                        1,
                        len(target_time) - 1,
                    )
                    * 100.0,
                    "Запись pre_apply_sync_map.csv",
                )

    progress.done(
        100.0,
        "Pre-apply map сохранена",
    )

    return path


# ============================================================
# СОХРАНЕНИЕ APPLY PLAN
# ============================================================

def save_apply_plan_json(
    output_dir,
    apply_map,
    robust_result,
    progress,
):
    """
    Сохраняет параметры будущего применения.
    """

    path = (
        output_dir
        / APPLY_PLAN_JSON
    )

    data = {
        "status":
            "PREPARED_NOT_APPLIED",

        "points":
            int(
                len(
                    apply_map[
                        "target_time"
                    ]
                )
            ),

        "min_tempo":
            float(
                apply_map[
                    "min_tempo"
                ]
            ),

        "max_tempo":
            float(
                apply_map[
                    "max_tempo"
                ]
            ),

        "repaired_validation_points":
            int(
                robust_result[
                    "quality"
                ].get(
                    "repaired_points",
                    0,
                )
            ),

        "robust_verdict":
            robust_result[
                "quality"
            ][
                "verdict"
            ],

        "robust_score":
            float(
                robust_result[
                    "quality"
                ][
                    "score"
                ]
            ),
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

    progress.update(
        100.0,
        "Apply plan сохранён",
    )

    return path


# ============================================================
# ГРАФИК APPLY MAP
# ============================================================

def save_apply_plan_jpeg(
    output_dir,
    apply_map,
    progress,
):
    """
    Показывает:

        offset
        reference timeline
        tempo

    в одной визуальной отчётности.
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ГРАФИК APPLY PLAN",
        0.0,
    )

    path = (
        output_dir
        / APPLY_PLAN_JPEG
    )

    time_axis = (
        apply_map[
            "target_time"
        ]
    )

    offset = (
        apply_map[
            "offset"
        ]
    )

    tempo = (
        apply_map[
            "tempo"
        ]
    )

    fig = plt.figure(
        figsize=(18, 10),
        dpi=150,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    ax.plot(
        time_axis,
        offset,
        linewidth=1.8,
        label="Offset",
    )

    ax.plot(
        time_axis[:-1],
        tempo,
        linewidth=1.0,
        alpha=0.65,
        label="Local tempo",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title(
        "Pre-Apply Synchronization Map"
    )

    ax.set_xlabel(
        "Target time, seconds"
    )

    ax.set_ylabel(
        "Offset / tempo"
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
        "Apply plan график готов",
    )

    return path


# ============================================================
# PRE-APPLY ОТЧЁТ
# ============================================================

def save_pre_apply_report(
    output_dir,
    selected,
    robust_result,
    apply_map,
    hard_points,
):
    """
    Общий текстовый отчёт перед APPLY.
    """

    path = (
        output_dir
        / PRE_APPLY_REPORT
    )

    quality = (
        robust_result[
            "quality"
        ]
    )

    lines = []

    lines.append(
        "PRE-APPLY SYNCHRONIZATION REPORT"
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
        "ROBUST MAP VERDICT:"
    )

    lines.append(
        f"  {quality['verdict']}"
    )

    lines.append(
        f"  Score: "
        f"{quality['score']:.2f}/100"
    )

    lines.append(
        ""
    )

    metrics = quality[
        "metrics"
    ]

    lines.append(
        f"Median residual: "
        f"{metrics['median']:.6f} s"
    )

    lines.append(
        f"P90 residual: "
        f"{metrics['p90']:.6f} s"
    )

    lines.append(
        f"P95 residual: "
        f"{metrics['p95']:.6f} s"
    )

    lines.append(
        f"Max residual: "
        f"{metrics['max']:.6f} s"
    )

    lines.append(
        f"<=50ms: "
        f"{metrics['good_ratio'] * 100:.2f}%"
    )

    lines.append(
        ""
    )

    lines.append(
        "APPLY MAP:"
    )

    lines.append(
        f"  points: "
        f"{len(apply_map['target_time'])}"
    )

    lines.append(
        f"  min tempo: "
        f"{apply_map['min_tempo']:.9f}"
    )

    lines.append(
        f"  max tempo: "
        f"{apply_map['max_tempo']:.9f}"
    )

    lines.append(
        ""
    )

    lines.append(
        "HARD OUTLIERS:"
    )

    if hard_points:

        for index, item in enumerate(
            hard_points,
            start=1,
        ):

            lines.append(
                (
                    f"  #{index}: "
                    f"{item['target_time_human']} "
                    f"residual="
                    f"{item['residual']:.3f}s"
                )
            )

    else:

        lines.append(
            "  none"
        )

    lines.append(
        ""
    )

    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "  Source TARGET has NOT been modified."
    )

    lines.append(
        "  Audio rendering has NOT been performed."
    )

    lines.append(
        "  The next stage will create a new file."
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
# UNIVERSAL AUDIO WAVEFORM SYNC
#
# ЧАСТЬ 10
#
# CONFIDENCE-AWARE NONLINEAR TRACKING
#
# Задача:
#
#   НЕ ПОЗВОЛЯТЬ слабым waveform-совпадениям
#   ломать правильную траекторию.
#
# ============================================================


# ============================================================
# НАСТРОЙКИ НОВОЙ СИСТЕМЫ ДОВЕРИЯ
# ============================================================

# Сильная корреляция.
