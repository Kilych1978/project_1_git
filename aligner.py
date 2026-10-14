# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: aligner.py
# ============================================================
# Global and local waveform matching, initial graphs, and alignment state/report.
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

# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC
#
# ЧАСТЬ 3
#
# WAVEFORM VISUALIZATION
# + GLOBAL ALIGNMENT
# + LOCAL WAVEFORM MATCHING
#
# ============================================================

import numpy as np


# ============================================================
# НАСТРОЙКИ СОПОСТАВЛЕНИЯ
# ============================================================

# ------------------------------------------------------------
# ГРУБЫЙ ГЛОБАЛЬНЫЙ ПОИСК
# ------------------------------------------------------------

# Сколько секунд с каждой стороны разрешено искать
# относительно нулевого смещения.
#
# Например:
#
#   -15 секунд
#   ...
#   +15 секунд
#
# Это не финальная карта.
# Это только первичное определение общего положения.
GLOBAL_SEARCH_RADIUS_SEC = 15.0


# Длительность окна для глобального поиска.
GLOBAL_MATCH_WINDOW_SEC = 120.0


# Шаг для грубого глобального поиска.
GLOBAL_SEARCH_STEP_SEC = 0.25


# ------------------------------------------------------------
# ЛОКАЛЬНЫЕ ПОИСКИ
# ------------------------------------------------------------

# Размер локального окна.
#
# 30 секунд — достаточно длинный участок,
# чтобы форма волны обычно была уникальной.
LOCAL_MATCH_WINDOW_SEC = 30.0


# Через сколько секунд ставим очередную точку.
#
# Это даст примерно 370 локальных измерений
# для трёхчасового фильма.
LOCAL_ANCHOR_INTERVAL_SEC = 30.0


# Радиус локального поиска вокруг ожидаемого положения.
LOCAL_SEARCH_RADIUS_SEC = 8.0


# Точность поиска.
#
# 10 ms соответствует нашему envelope.
LOCAL_SEARCH_STEP_SEC = 0.01


# ------------------------------------------------------------
# КАЧЕСТВО
# ------------------------------------------------------------

# Минимальная корреляция, ниже которой совпадение
# считаем слабым.
MIN_LOCAL_CORRELATION = 0.15


# Насколько второй лучший кандидат должен быть хуже
# лучшего, чтобы совпадение считалось достаточно
# однозначным.
MIN_LOCAL_PEAK_MARGIN = 0.005


# ------------------------------------------------------------
# JPEG
# ------------------------------------------------------------

WAVEFORM_JPEG_NAME = (
    "01_waveform_overview.jpg"
)

ENVELOPE_JPEG_NAME = (
    "02_envelope_overview.jpg"
)

GLOBAL_MATCH_JPEG_NAME = (
    "03_global_alignment.jpg"
)

LOCAL_ANCHORS_JPEG_NAME = (
    "04_local_anchor_map.jpg"
)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def safe_corrcoef(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Безопасный коэффициент корреляции.

    В отличие от np.corrcoef() здесь явно обрабатываем
    случай почти постоянного сигнала.
    """

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    if len(a) != len(b):
        return 0.0

    if len(a) < 2:
        return 0.0

    a = (
        a
        - np.mean(a)
    )

    b = (
        b
        - np.mean(b)
    )

    norm_a = np.linalg.norm(
        a
    )

    norm_b = np.linalg.norm(
        b
    )

    if (
        norm_a < 1e-12
        or norm_b < 1e-12
    ):
        return 0.0

    value = (
        np.dot(a, b)
        / (
            norm_a
            * norm_b
        )
    )

    return float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )


# ============================================================
# ЛОКАЛЬНАЯ НОРМАЛИЗАЦИЯ
# ============================================================

def normalize_segment(
    segment: np.ndarray,
) -> np.ndarray:
    """
    Нормализация отдельного участка.

    Это очень важный элемент алгоритма.

    Нас НЕ интересует:

        Reference громче Target
        Reference тише Target

    Нас интересует только форма.

    Поэтому каждый участок отдельно:

        1. убираем среднее;
        2. делим на RMS / std.
    """

    segment = np.asarray(
        segment,
        dtype=np.float64,
    )

    if segment.size == 0:
        return segment

    segment = (
        segment
        - np.mean(segment)
    )

    norm = np.linalg.norm(
        segment
    )

    if norm < 1e-12:
        return np.zeros_like(
            segment
        )

    return (
        segment
        / norm
    )


# ============================================================
# ПОЛУЧЕНИЕ ОТРЕЗКА ENVELOPE
# ============================================================

def envelope_slice(
    envelope: np.ndarray,
    center_sec: float,
    window_sec: float,
    hop_sec: float,
):
    """
    Возвращает участок envelope вокруг указанного времени.
    """

    half = (
        window_sec
        / 2.0
    )

    start_sec = (
        center_sec
        - half
    )

    end_sec = (
        center_sec
        + half
    )

    start_index = int(
        round(
            start_sec
            / hop_sec
        )
    )

    end_index = int(
        round(
            end_sec
            / hop_sec
        )
    )

    start_index = max(
        0,
        start_index,
    )

    end_index = min(
        len(envelope),
        end_index,
    )

    if (
        end_index
        <= start_index
    ):
        return np.empty(
            0,
            dtype=np.float32,
        )

    return envelope[
        start_index:end_index
    ]


# ============================================================
# ОЦЕНКА СООТВЕТСТВИЯ ДВУХ ЛОКАЛЬНЫХ ОТРЕЗКОВ
# ============================================================

def compare_envelope_segments(
    reference_segment,
    target_segment,
):
    """
    Возвращает корреляцию двух участков.

    Длины должны быть одинаковыми.
    """

    if (
        len(reference_segment)
        != len(target_segment)
    ):
        return 0.0

    a = normalize_segment(
        reference_segment
    )

    b = normalize_segment(
        target_segment
    )

    if (
        a.size == 0
        or b.size == 0
    ):
        return 0.0

    return float(
        np.dot(
            a,
            b,
        )
    )


# ============================================================
# ПОИСК ЛУЧШЕГО OFFSET
# ============================================================

def search_best_offset_bruteforce(
    reference_envelope,
    target_envelope,
    target_center_sec,
    expected_offset_sec,
    window_sec,
    search_radius_sec,
    search_step_sec,
    hop_sec,
):
    """
    Ищет лучшее соответствие локального окна.

    Определение offset:

        reference_time
        =
        target_time + offset

    То есть:

        offset > 0

    означает, что соответствующий момент Reference
    находится позже момента Target.

    ------------------------------------------------------------
    ВАЖНО
    ------------------------------------------------------------

    Здесь мы НЕ доверяем FFT-лагу "на глаз".

    Мы явно сравниваем:

        Reference(center + offset)

    с

        Target(center)

    Благодаря этому знак offset однозначно определён
    нашей собственной системой координат.
    """

    target_segment = envelope_slice(
        target_envelope,
        target_center_sec,
        window_sec,
        hop_sec,
    )

    if (
        target_segment.size == 0
    ):
        return {
            "offset": None,
            "correlation": 0.0,
            "margin": 0.0,
            "second_correlation": 0.0,
        }

    target_norm = normalize_segment(
        target_segment
    )

    if np.linalg.norm(
        target_norm
    ) < 1e-12:
        return {
            "offset": None,
            "correlation": 0.0,
            "margin": 0.0,
            "second_correlation": 0.0,
        }

    # --------------------------------------------------------
    # Сетка кандидатов.
    # --------------------------------------------------------

    first_offset = (
        expected_offset_sec
        - search_radius_sec
    )

    last_offset = (
        expected_offset_sec
        + search_radius_sec
    )

    number_of_steps = int(
        round(
            (
                last_offset
                - first_offset
            )
            / search_step_sec
        )
    )

    if number_of_steps < 1:
        number_of_steps = 1

    candidates = []

    for step in range(
        number_of_steps + 1
    ):

        offset = (
            first_offset
            + step
            * search_step_sec
        )

        reference_center = (
            target_center_sec
            + offset
        )

        reference_segment = (
            envelope_slice(
                reference_envelope,
                reference_center,
                window_sec,
                hop_sec,
            )
        )

        if (
            reference_segment.size
            != target_segment.size
        ):
            continue

        reference_norm = (
            normalize_segment(
                reference_segment
            )
        )

        if (
            np.linalg.norm(
                reference_norm
            )
            < 1e-12
        ):
            continue

        correlation = float(
            np.dot(
                reference_norm,
                target_norm,
            )
        )

        candidates.append(
            (
                correlation,
                offset,
            )
        )

    if not candidates:
        return {
            "offset": None,
            "correlation": 0.0,
            "margin": 0.0,
            "second_correlation": 0.0,
        }

    # Сортируем по качеству.
    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    best_corr, best_offset = (
        candidates[0]
    )

    if len(candidates) >= 2:
        second_corr = (
            candidates[1][0]
        )
    else:
        second_corr = 0.0

    margin = (
        best_corr
        - second_corr
    )

    return {
        "offset": float(
            best_offset
        ),
        "correlation": float(
            best_corr
        ),
        "margin": float(
            margin
        ),
        "second_correlation": float(
            second_corr
        ),
    }


# ============================================================
# ГРУБЫЙ ГЛОБАЛЬНЫЙ ПОИСК
# ============================================================

def global_alignment_search(
    reference_envelope: np.ndarray,
    target_envelope: np.ndarray,
    reference_info,
    target_info,
    progress: Progress,
):
    """
    Ищет первоначальное общее смещение.

    Это НЕ финальная карта.

    Задача этапа:
        понять общий порядок времён.

    Например, если Target отличается примерно на:

        +3.4 s

    следующий локальный анализ уже будет искать
    в районе +3.4 s вместо полного диапазона.
    """

    print()
    print("=" * 78)
    print("GLOBAL WAVEFORM ALIGNMENT")
    print("=" * 78)

    hop_sec = (
        ENVELOPE_HOP_MS
        / 1000.0
    )

    reference_duration = (
        len(reference_envelope)
        * hop_sec
    )

    target_duration = (
        len(target_envelope)
        * hop_sec
    )

    # --------------------------------------------------------
    # Чтобы глобальный поиск был надёжным, используем
    # НЕ один маленький фрагмент.
    #
    # Берём несколько довольно длинных контрольных окон:
    #
    #   10%
    #   25%
    #   40%
    #   55%
    #   70%
    #   85%
    #
    # каждого фильма.
    #
    # Это сильно лучше одного 120-секундного окна.
    # --------------------------------------------------------

    checkpoints = []

    for fraction in (
        0.10,
        0.25,
        0.40,
        0.55,
        0.70,
        0.85,
    ):

        center = (
            target_duration
            * fraction
        )

        margin = (
            GLOBAL_MATCH_WINDOW_SEC
            / 2.0
            + GLOBAL_SEARCH_RADIUS_SEC
            + 1.0
        )

        if (
            center < margin
            or
            center
            > target_duration
            - margin
        ):
            continue

        checkpoints.append(
            center
        )

    if not checkpoints:

        raise RuntimeError(
            "Не удалось сформировать "
            "глобальные контрольные окна."
        )

    all_results = []

    # --------------------------------------------------------
    # Для глобального поиска используем более грубый шаг,
    # чем для финального локального поиска.
    #
    # Иначе 15 секунд / 0.01 = 1500 сравнений
    # на одно окно.
    #
    # Пока достаточно 0.25 s.
    # --------------------------------------------------------

    total = len(checkpoints)

    started = time.time()

    for index, center in enumerate(
        checkpoints,
        start=1,
    ):

        result = search_best_offset_bruteforce(
            reference_envelope,
            target_envelope,
            center,
            0.0,
            GLOBAL_MATCH_WINDOW_SEC,
            GLOBAL_SEARCH_RADIUS_SEC,
            GLOBAL_SEARCH_STEP_SEC,
            hop_sec,
        )

        result["target_time"] = (
            center
        )

        all_results.append(
            result
        )

        percent = (
            index
            / total
            * 100.0
        )

        speed = (
            index
            / max(
                0.001,
                time.time() - started,
            )
        )

        remaining = (
            (
                total
                - index
            )
            / speed
            if speed > 0
            else 0.0
        )

        progress.update(
            percent,
            (
                "Global alignment | "
                f"ETA "
                f"{format_seconds(remaining)}"
            ),
        )

    print()

    # --------------------------------------------------------
    # Вывод результатов.
    # --------------------------------------------------------

    print()
    print(
        "GLOBAL RESULTS"
    )
    print("-" * 78)

    for result in all_results:

        offset = result["offset"]

        if offset is None:
            print(
                f"{result['target_time']:8.1f}s "
                f"-> NO MATCH"
            )
            continue

        print(
            f"{result['target_time']:8.1f}s "
            f"-> "
            f"{offset:+8.3f}s "
            f"corr="
            f"{result['correlation']:.4f}"
        )

    # --------------------------------------------------------
    # Считаем робастный offset.
    # --------------------------------------------------------

    valid_offsets = [
        r["offset"]
        for r in all_results
        if (
            r["offset"] is not None
            and
            r["correlation"]
            >= MIN_LOCAL_CORRELATION
        )
    ]

    if not valid_offsets:

        print()
        print(
            "WARNING:"
        )
        print(
            "Глобально устойчивое совпадение "
            "не найдено."
        )

        coarse_offset = 0.0

    else:

        coarse_offset = float(
            np.median(
                valid_offsets
            )
        )

    print()
    print(
        "Robust global offset:"
    )
    print(
        f"{coarse_offset:+.4f} sec"
    )

    print()

    progress.done(
        100.0,
        "Global alignment complete",
    )

    return {
        "coarse_offset":
            coarse_offset,

        "results":
            all_results,

        "reference_duration":
            reference_duration,

        "target_duration":
            target_duration,
    }


# ============================================================
# СОЗДАНИЕ ЛОКАЛЬНЫХ ЯКОРЕЙ
# ============================================================

def build_local_anchor_table(
    reference_envelope,
    target_envelope,
    target_duration,
    initial_offset,
    progress: Progress,
):
    """
    Основной массив измерений.

    Для большого количества временных точек:

        Target time
             |
             v
        поиск Reference
             |
             v
        лучший offset
             |
             v
        correlation / margin

    На этом этапе ещё НЕ строим финальную карту.

    Мы сначала собираем "сырые измерения".
    """

    print()
    print("=" * 78)
    print("LOCAL WAVEFORM ANCHOR SEARCH")
    print("=" * 78)

    hop_sec = (
        ENVELOPE_HOP_MS
        / 1000.0
    )

    half_window = (
        LOCAL_MATCH_WINDOW_SEC
        / 2.0
    )

    # --------------------------------------------------------
    # Безопасные границы.
    # --------------------------------------------------------

    start_time = (
        half_window
        + LOCAL_SEARCH_RADIUS_SEC
        + abs(initial_offset)
        + 2.0
    )

    end_time = (
        target_duration
        - half_window
        - LOCAL_SEARCH_RADIUS_SEC
        - abs(initial_offset)
        - 2.0
    )

    if end_time <= start_time:

        raise RuntimeError(
            "Фильм слишком короткий "
            "для выбранных параметров "
            "локального поиска."
        )

    # --------------------------------------------------------
    # Формируем точки.
    # --------------------------------------------------------

    centers = np.arange(
        start_time,
        end_time + 0.001,
        LOCAL_ANCHOR_INTERVAL_SEC,
        dtype=np.float64,
    )

    total = len(
        centers
    )

    results = []

    started = time.time()

    # --------------------------------------------------------
    # Очень важный момент:
    #
    # мы НЕ ищем каждый участок вокруг одного и того же
    # глобального offset.
    #
    # Ожидаемый offset постепенно наследуется от предыдущей
    # надёжной точки.
    #
    # Это позволяет карте двигаться нелинейно.
    #
    # Например:
    #
    # +0.15
    # +0.18
    # +0.21
    # +0.24
    # ...
    #
    # а не:
    #
    # всё время искать вокруг +0.15
    #
    # Именно это должно помочь нам с drift.
    # --------------------------------------------------------

    expected_offset = (
        initial_offset
    )

    for index, center in enumerate(
        centers,
        start=1,
    ):

        result = search_best_offset_bruteforce(
            reference_envelope,
            target_envelope,
            center,
            expected_offset,
            LOCAL_MATCH_WINDOW_SEC,
            LOCAL_SEARCH_RADIUS_SEC,
            LOCAL_SEARCH_STEP_SEC,
            hop_sec,
        )

        result["target_time"] = (
            float(center)
        )

        # ----------------------------------------------------
        # Если найдено хорошее совпадение,
        # используем его как прогноз следующей точки.
        # ----------------------------------------------------

        if (
            result["offset"] is not None
            and
            result["correlation"]
            >= MIN_LOCAL_CORRELATION
            and
            result["margin"]
            >= MIN_LOCAL_PEAK_MARGIN
        ):

            expected_offset = (
                result["offset"]
            )

            result["accepted_for_tracking"] = (
                True
            )

        else:

            result[
                "accepted_for_tracking"
            ] = False

        results.append(
            result
        )

        percent = (
            index
            / total
            * 100.0
        )

        elapsed = (
            time.time()
            - started
        )

        speed = (
            index
            / max(
                0.001,
                elapsed,
            )
        )

        remaining = (
            (
                total
                - index
            )
            / speed
            if speed > 0
            else 0.0
        )

        current_offset = (
            result["offset"]
            if result["offset"] is not None
            else expected_offset
        )

        progress.update(
            percent,
            (
                "Local matching | "
                f"t={center:8.1f}s | "
                f"offset={current_offset:+7.3f}s | "
                f"corr={result['correlation']:.3f} | "
                f"ETA={format_seconds(remaining)}"
            ),
        )

    print()

    accepted = [
        r
        for r in results
        if r[
            "accepted_for_tracking"
        ]
    ]

    print()
    print(
        "Local anchor search complete."
    )

    print(
        "Total points   :",
        len(results),
    )

    print(
        "Accepted track :",
        len(accepted),
    )

    if accepted:

        offsets = np.asarray(
            [
                r["offset"]
                for r in accepted
            ],
            dtype=np.float64,
        )

        print(
            "Offset min     :",
            f"{np.min(offsets):+.4f}s",
        )

        print(
            "Offset median  :",
            f"{np.median(offsets):+.4f}s",
        )

        print(
            "Offset max     :",
            f"{np.max(offsets):+.4f}s",
        )

    print()

    progress.done(
        100.0,
        "Local anchor search complete",
    )

    return results


# ============================================================
# СОХРАНЕНИЕ СЫРЫХ ЯКОРЕЙ
# ============================================================

def save_local_anchor_csv(
    output_dir: Path,
    results,
):

    import csv

    path = (
        output_dir
        / "local_waveform_measurements.csv"
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
                "correlation",
                "second_correlation",
                "margin",
                "accepted_for_tracking",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    (
                        f"{result['target_time']:.6f}"
                    ),

                    (
                        ""
                        if result["offset"] is None
                        else
                        f"{result['offset']:.6f}"
                    ),

                    (
                        f"{result['correlation']:.6f}"
                    ),

                    (
                        f"{result['second_correlation']:.6f}"
                    ),

                    (
                        f"{result['margin']:.6f}"
                    ),

                    (
                        "YES"
                        if result[
                            "accepted_for_tracking"
                        ]
                        else "NO"
                    ),
                ]
            )

    return path


# ============================================================
# СОХРАНЕНИЕ GLOBAL РЕЗУЛЬТАТОВ
# ============================================================

def save_global_results_csv(
    output_dir: Path,
    global_result,
):

    import csv

    path = (
        output_dir
        / "global_alignment.csv"
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
                "correlation",
                "second_correlation",
                "margin",
            ]
        )

        for result in global_result[
            "results"
        ]:

            writer.writerow(
                [
                    (
                        f"{result['target_time']:.6f}"
                    ),

                    (
                        ""
                        if result["offset"] is None
                        else
                        f"{result['offset']:.6f}"
                    ),

                    (
                        f"{result['correlation']:.6f}"
                    ),

                    (
                        f"{result['second_correlation']:.6f}"
                    ),

                    (
                        f"{result['margin']:.6f}"
                    ),
                ]
            )

    return path


# ============================================================
# ВИЗУАЛИЗАЦИЯ WAVEFORM
# ============================================================

def save_waveform_overview_jpeg(
    output_dir: Path,
    reference_times,
    reference_waveform,
    target_times,
    target_waveform,
    progress: Progress,
):
    """
    Строит первоначальный график двух дорожек.

    ВАЖНО:
    Здесь они ещё НЕ синхронизируются.

    График показывает исходное временное положение.
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ПОСТРОЕНИЕ WAVEFORM ГРАФИКА",
        0.0,
    )

    path = (
        output_dir
        / WAVEFORM_JPEG_NAME
    )

    fig = plt.figure(
        figsize=(18, 10),
        dpi=140,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    # --------------------------------------------------------
    # Чтобы трёхчасовой сигнал не превращался в чёрную кашу,
    # используем визуальные данные как есть, но разрешаем
    # matplotlib самому отрисовать их с downsampling.
    # --------------------------------------------------------

    ax.plot(
        reference_times,
        reference_waveform,
        linewidth=0.35,
        label="REFERENCE",
        alpha=0.75,
    )

    ax.plot(
        target_times,
        target_waveform,
        linewidth=0.35,
        label="TARGET",
        alpha=0.55,
    )

    ax.set_title(
        "Original Waveforms — Reference vs Target"
    )

    ax.set_xlabel(
        "Time, seconds"
    )

    ax.set_ylabel(
        "Amplitude"
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
        dpi=140,
    )

    plt.close(
        fig
    )

    progress.done(
        100.0,
        f"График сохранён: {path.name}",
    )

    return path


# ============================================================
# ВИЗУАЛИЗАЦИЯ ENVELOPE
# ============================================================

def save_envelope_overview_jpeg(
    output_dir: Path,
    reference_envelope,
    target_envelope,
    progress: Progress,
):
    """
    График нормализованных огибающих.

    Это уже значительно ближе к тому,
    что будет использоваться математическим алгоритмом.
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ПОСТРОЕНИЕ ENVELOPE ГРАФИКА",
        0.0,
    )

    path = (
        output_dir
        / ENVELOPE_JPEG_NAME
    )

    ref_values = (
        reference_envelope[
            "values"
        ]
    )

    tgt_values = (
        target_envelope[
            "values"
        ]
    )

    ref_hop = (
        reference_envelope[
            "hop_sec"
        ]
    )

    tgt_hop = (
        target_envelope[
            "hop_sec"
        ]
    )

    reference_time = (
        np.arange(
            len(ref_values)
        )
        * ref_hop
    )

    target_time = (
        np.arange(
            len(tgt_values)
        )
        * tgt_hop
    )

    fig = plt.figure(
        figsize=(18, 10),
        dpi=140,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    ax.plot(
        reference_time,
        ref_values,
        linewidth=0.5,
        label="REFERENCE ENVELOPE",
        alpha=0.80,
    )

    ax.plot(
        target_time,
        tgt_values,
        linewidth=0.5,
        label="TARGET ENVELOPE",
        alpha=0.65,
    )

    ax.set_title(
        "Normalized Amplitude Envelopes"
    )

    ax.set_xlabel(
        "Time, seconds"
    )

    ax.set_ylabel(
        "Normalized energy"
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
        dpi=140,
    )

    plt.close(
        fig
    )

    progress.done(
        100.0,
        f"График сохранён: {path.name}",
    )

    return path


# ============================================================
# ГРАФИК GLOBAL ALIGNMENT
# ============================================================

def save_global_alignment_jpeg(
    output_dir: Path,
    global_result,
    progress: Progress,
):
    """
    График первых глобальных измерений.

    По X:
        Target time

    По Y:
        найденный offset

    Чем ровнее точки,
    тем ближе дорожки к простому линейному сдвигу.

    Если точки начинают менять положение,
    появляется nonlinear drift.
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ПОСТРОЕНИЕ GLOBAL ALIGNMENT ГРАФИКА",
        0.0,
    )

    path = (
        output_dir
        / GLOBAL_MATCH_JPEG_NAME
    )

    valid = [
        result
        for result in global_result[
            "results"
        ]
        if result[
            "offset"
        ] is not None
    ]

    fig = plt.figure(
        figsize=(16, 9),
        dpi=140,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    if valid:

        x = np.asarray(
            [
                result[
                    "target_time"
                ]
                for result in valid
            ]
        )

        y = np.asarray(
            [
                result[
                    "offset"
                ]
                for result in valid
            ]
        )

        c = np.asarray(
            [
                result[
                    "correlation"
                ]
                for result in valid
            ]
        )

        scatter = ax.scatter(
            x,
            y,
            c=c,
            s=70,
        )

        fig.colorbar(
            scatter,
            ax=ax,
            label="Correlation",
        )

        ax.axhline(
            global_result[
                "coarse_offset"
            ],
            linestyle="--",
            linewidth=1.0,
            label=(
                "Robust global offset"
            ),
        )

    ax.set_title(
        "Global Waveform Alignment"
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
        dpi=140,
    )

    plt.close(
        fig
    )

    progress.done(
        100.0,
        f"График сохранён: {path.name}",
    )

    return path


# ============================================================
# ГРАФИК ЛОКАЛЬНЫХ ЯКОРЕЙ
# ============================================================

def save_local_anchor_jpeg(
    output_dir: Path,
    local_results,
    progress: Progress,
):
    """
    Показывает всю первую локальную карту.

    Это пока ещё RAW MAP.

    Финальную карту будем делать после удаления
    выбросов и проверки монотонности.
    """

    import matplotlib.pyplot as plt

    progress.stage(
        "ПОСТРОЕНИЕ LOCAL ANCHOR ГРАФИКА",
        0.0,
    )

    path = (
        output_dir
        / LOCAL_ANCHORS_JPEG_NAME
    )

    accepted = [
        result
        for result in local_results
        if (
            result[
                "offset"
            ] is not None
        )
    ]

    fig = plt.figure(
        figsize=(18, 10),
        dpi=140,
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
    )

    if accepted:

        x = np.asarray(
            [
                result[
                    "target_time"
                ]
                for result in accepted
            ]
        )

        y = np.asarray(
            [
                result[
                    "offset"
                ]
                for result in accepted
            ]
        )

        corr = np.asarray(
            [
                result[
                    "correlation"
                ]
                for result in accepted
            ]
        )

        scatter = ax.scatter(
            x,
            y,
            c=corr,
            s=12,
            alpha=0.75,
        )

        fig.colorbar(
            scatter,
            ax=ax,
            label="Correlation",
        )

    ax.set_title(
        "Raw Local Waveform Measurements"
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

    fig.tight_layout()

    fig.savefig(
        path,
        format="jpg",
        dpi=140,
    )

    plt.close(
        fig
    )

    progress.done(
        100.0,
        f"График сохранён: {path.name}",
    )

    return path


# ============================================================
# СВОДКА ПЕРВОГО АНАЛИЗА
# ============================================================

def calculate_initial_sync_summary(
    global_result,
    local_results,
):
    """
    Даёт первую количественную оценку.

    Пока НЕ говорит:
        "карта хорошая"

    Потому что настоящая карта будет строиться позже.

    Здесь только собираем статистику.
    """

    valid_local = [
        result
        for result in local_results
        if (
            result[
                "offset"
            ] is not None
        )
    ]

    accepted_local = [
        result
        for result in local_results
        if (
            result[
                "accepted_for_tracking"
            ]
        )
    ]

    correlations = [
        result[
            "correlation"
        ]
        for result in valid_local
    ]

    if correlations:

        mean_corr = float(
            np.mean(
                correlations
            )
        )

        median_corr = float(
            np.median(
                correlations
            )
        )

        min_corr = float(
            np.min(
                correlations
            )
        )

        max_corr = float(
            np.max(
                correlations
            )
        )

    else:

        mean_corr = 0.0
        median_corr = 0.0
        min_corr = 0.0
        max_corr = 0.0

    return {
        "global_offset":
            global_result[
                "coarse_offset"
            ],

        "global_points":
            len(
                global_result[
                    "results"
                ]
            ),

        "local_points":
            len(
                local_results
            ),

        "local_valid":
            len(
                valid_local
            ),

        "local_accepted":
            len(
                accepted_local
            ),

        "mean_correlation":
            mean_corr,

        "median_correlation":
            median_corr,

        "min_correlation":
            min_corr,

        "max_correlation":
            max_corr,
    }


# ============================================================
# СОХРАНЕНИЕ ПЕРВОГО SUMMARY
# ============================================================

def save_initial_sync_summary(
    output_dir: Path,
    summary,
):

    path = (
        output_dir
        / "initial_sync_summary.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path

# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC
#
# ЧАСТЬ 4
#
# FAST FFT MATCHING
# + ACTUAL ANALYSIS PIPELINE
#
# ============================================================


# ============================================================
# БЫСТРАЯ FFT-КОРРЕЛЯЦИЯ
# ============================================================

def _next_power_of_two(value: int) -> int:
    """
    Возвращает ближайшую степень двойки >= value.
    """

    if value <= 1:
        return 1

    result = 1

    while result < value:
        result *= 2

    return result


# ============================================================
# СУБ-HOP УТОЧНЕНИЕ OFFSET
# ============================================================

def _interpolated_envelope_slice(
    envelope,
    center_sec,
    window_sec,
    hop_sec,
):
    """
    Возвращает envelope-отрезок с дробным центром времени.

    Основной поиск остаётся быстрым и работает с 10 ms сеткой.
    Эта функция используется только на локальном максимуме и
    линейно интерполирует соседние точки envelope, чтобы можно
    было уточнить offset с шагом 1 ms.
    """
    envelope = np.asarray(envelope, dtype=np.float64)
    if envelope.size < 2:
        return np.empty(0, dtype=np.float64)

    half = window_sec / 2.0
    start_sec = max(0.0, center_sec - half)
    end_sec = min((len(envelope) - 1) * hop_sec, center_sec + half)
    if end_sec <= start_sec:
        return np.empty(0, dtype=np.float64)

    count = int(round((end_sec - start_sec) / hop_sec)) + 1
    positions = np.linspace(start_sec, end_sec, count, dtype=np.float64) / hop_sec
    left = np.floor(positions).astype(np.int64)
    frac = positions - left
    right = np.minimum(left + 1, len(envelope) - 1)

    values = (
        envelope[left] * (1.0 - frac)
        + envelope[right] * frac
    )
    return values


def _refine_offset_interpolated(
    reference_envelope,
    target_envelope,
    target_center_sec,
    coarse_offset_sec,
    window_sec,
    hop_sec,
):
    """
    Уточняет уже найденный offset в узком +/-20 ms диапазоне с
    шагом 1 ms. Не используется как самостоятельный глобальный поиск.
    """
    target_segment = _interpolated_envelope_slice(
        target_envelope, target_center_sec, window_sec, hop_sec
    )
    if len(target_segment) < 10:
        return float(coarse_offset_sec), 0.0

    target_norm = normalize_segment(target_segment)
    if np.linalg.norm(target_norm) < 1e-12:
        return float(coarse_offset_sec), 0.0

    offsets = np.arange(
        coarse_offset_sec - FINE_OFFSET_REFINE_RADIUS_SEC,
        coarse_offset_sec + FINE_OFFSET_REFINE_RADIUS_SEC + FINE_OFFSET_REFINE_STEP_SEC * 0.5,
        FINE_OFFSET_REFINE_STEP_SEC,
        dtype=np.float64,
    )

    best_offset = float(coarse_offset_sec)
    best_corr = -2.0

    for offset in offsets:
        reference_segment = _interpolated_envelope_slice(
            reference_envelope,
            target_center_sec + float(offset),
            window_sec,
            hop_sec,
        )
        if len(reference_segment) != len(target_segment):
            continue
        reference_norm = normalize_segment(reference_segment)
        if np.linalg.norm(reference_norm) < 1e-12:
            continue
        correlation = float(np.dot(reference_norm, target_norm))
        if correlation > best_corr:
            best_corr = correlation
            best_offset = float(offset)

    return best_offset, max(0.0, best_corr)

# ============================================================
# БЫСТРЫЙ ЛОКАЛЬНЫЙ ПОИСК
# ============================================================

def search_best_offset_bruteforce(
    reference_envelope,
    target_envelope,
    target_center_sec,
    expected_offset_sec,
    window_sec,
    search_radius_sec,
    search_step_sec,
    hop_sec,
):
    """
    ВНИМАНИЕ:

    Название функции оставлено прежним специально.

    Части 3 использует именно это имя.

    Но внутри теперь НЕ brute force.

    Здесь используется FFT cross-correlation,
    поэтому программа не делает тысячи отдельных
    dot-product для каждого возможного offset.

    Система координат:

        Reference time =
        Target time + offset

    Поэтому:

        offset > 0

    означает:

        соответствующий момент Reference находится
        позже момента Target.
    """

    # ========================================================
    # 1. TARGET SEGMENT
    # ========================================================

    target_segment = envelope_slice(
        target_envelope,
        target_center_sec,
        window_sec,
        hop_sec,
    )

    if target_segment.size < 10:
        return {
            "offset": None,
            "correlation": 0.0,
            "margin": 0.0,
            "second_correlation": 0.0,
        }

    target_segment = np.asarray(
        target_segment,
        dtype=np.float64,
    )

    target_mean = np.mean(
        target_segment
    )

    target_zero = (
        target_segment
        - target_mean
    )

    target_energy = np.sum(
        target_zero
        * target_zero
    )

    if target_energy <= 1e-14:
        return {
            "offset": None,
            "correlation": 0.0,
            "margin": 0.0,
            "second_correlation": 0.0,
        }

    # ========================================================
    # 2. REFERENCE SEARCH REGION
    # ========================================================

    reference_center = (
        target_center_sec
        + expected_offset_sec
    )

    # Нам нужен полный участок:
    #
    # window
    #
    # плюс возможность сдвинуть его
    # на +/- search_radius.
    #
    reference_search_sec = (
        window_sec
        + 2.0
        * search_radius_sec
    )

    reference_search = envelope_slice(
        reference_envelope,
        reference_center,
        reference_search_sec,
        hop_sec,
    )

    target_points = len(
        target_segment
    )

    if len(reference_search) < (
        target_points + 10
    ):
        return {
            "offset": None,
            "correlation": 0.0,
            "margin": 0.0,
            "second_correlation": 0.0,
        }

    reference_search = np.asarray(
        reference_search,
        dtype=np.float64,
    )

    # ========================================================
    # 3. СКОЛЬЗЯЩИЕ SUM / SUM OF SQUARES
    #
    # Это нужно для нормированной корреляции.
    # ========================================================

    cumulative_sum = np.concatenate(
        (
            np.array(
                [0.0],
                dtype=np.float64,
            ),
            np.cumsum(
                reference_search
            ),
        )
    )

    cumulative_sq = np.concatenate(
        (
            np.array(
                [0.0],
                dtype=np.float64,
            ),
            np.cumsum(
                reference_search
                * reference_search
            ),
        )
    )

    candidate_count = (
        len(reference_search)
        - target_points
        + 1
    )

    if candidate_count <= 0:
        return {
            "offset": None,
            "correlation": 0.0,
            "margin": 0.0,
            "second_correlation": 0.0,
        }

    # ========================================================
    # 4. FFT CROSS-CORRELATION
    #
    # Для каждого возможного положения target window
    # внутри reference search region получаем numerator
    # корреляции.
    # ========================================================

    fft_length = _next_power_of_two(
        len(reference_search)
        + len(target_zero)
        - 1
    )

    reference_fft = np.fft.rfft(
        reference_search,
        n=fft_length,
    )

    target_fft = np.fft.rfft(
        target_zero,
        n=fft_length,
    )

    correlation_circular = np.fft.irfft(
        reference_fft
        * np.conjugate(
            target_fft
        ),
        n=fft_length,
    )

    # Первые candidate_count элементов —
    # именно нужные положения:
    #
    # reference[start:start+N]
    #
    # относительно target[0:N].
    cross = correlation_circular[
        :candidate_count
    ]

    # ========================================================
    # 5. ДЕЛАЕМ КОРРЕЛЯЦИЮ НОРМИРОВАННОЙ
    #
    # numerator:
    #
    #      sum((R-meanR) * (T-meanT))
    #
    # denominator:
    #
    #      sqrt(sum(R0²) * sum(T0²))
    #
    # ========================================================

    starts = np.arange(
        candidate_count,
        dtype=np.int64,
    )

    ends = (
        starts
        + target_points
    )

    ref_sum = (
        cumulative_sum[ends]
        - cumulative_sum[starts]
    )

    ref_sq_sum = (
        cumulative_sq[ends]
        - cumulative_sq[starts]
    )

    n = float(
        target_points
    )

    ref_variance_energy = (
        ref_sq_sum
        - (
            ref_sum
            * ref_sum
            / n
        )
    )

    # Численные ошибки FFT / cumsum
    # иногда дают крошечные отрицательные числа.
    ref_variance_energy = np.maximum(
        ref_variance_energy,
        0.0,
    )

    denominator = np.sqrt(
        ref_variance_energy
        * target_energy
    )

    # Защита от деления на ноль.
    normalized_corr = np.zeros(
        candidate_count,
        dtype=np.float64,
    )

    valid = (
        denominator > 1e-14
    )

    normalized_corr[valid] = (
        cross[valid]
        / denominator[valid]
    )

    normalized_corr = np.clip(
        normalized_corr,
        -1.0,
        1.0,
    )

    # ========================================================
    # 6. ОСТАВЛЯЕМ ТОЛЬКО РАЗРЕШЁННЫЕ OFFSETS
    # ========================================================

    # У reference_search центр находится
    # в expected_offset.
    #
    # Поэтому индекс search_radius соответствует
    # expected_offset.
    #

    radius_points = int(
        round(
            search_radius_sec
            / hop_sec
        )
    )

    center_index = int(
        round(
            (
                len(reference_search)
                - target_points
            )
            / 2.0
        )
    )

    minimum_index = max(
        0,
        center_index
        - radius_points,
    )

    maximum_index = min(
        candidate_count - 1,
        center_index
        + radius_points,
    )

    if maximum_index < minimum_index:
        return {
            "offset": None,
            "correlation": 0.0,
            "margin": 0.0,
            "second_correlation": 0.0,
        }

    search_indices = np.arange(
        minimum_index,
        maximum_index + 1,
        dtype=np.int64,
    )

    search_scores = (
        normalized_corr[
            search_indices
        ]
    )

    # ========================================================
    # 7. НАХОДИМ ЛУЧШИЙ КАНДИДАТ
    # ========================================================

    if search_scores.size == 0:
        return {
            "offset": None,
            "correlation": 0.0,
            "margin": 0.0,
            "second_correlation": 0.0,
        }

    best_local_position = int(
        np.argmax(
            search_scores
        )
    )

    best_index = int(
        search_indices[
            best_local_position
        ]
    )

    best_corr = float(
        normalized_corr[
            best_index
        ]
    )

    # --------------------------------------------------------
    # Второй лучший кандидат.
    #
    # Исключаем несколько ближайших точек к главному пику,
    # иначе при широкой корреляционной вершине второй пик
    # будет просто соседним sample.
    # --------------------------------------------------------

    exclusion_points = max(
        1,
        int(
            round(
                0.05
                / hop_sec
            )
        ),
    )

    secondary_mask = (
        np.abs(
            search_indices
            - best_index
        )
        > exclusion_points
    )

    secondary_indices = (
        search_indices[
            secondary_mask
        ]
    )

    if secondary_indices.size:

        secondary_scores = (
            normalized_corr[
                secondary_indices
            ]
        )

        second_corr = float(
            np.max(
                secondary_scores
            )
        )

    else:

        second_corr = 0.0

    margin = (
        best_corr
        - second_corr
    )

    # ========================================================
    # 8. INDEX -> OFFSET
    # ========================================================

    # center_index:
    #
    # положение, соответствующее expected_offset.
    #
    # Поэтому:
    #
    # offset =
    # expected_offset
    # +
    # (best_index - center_index) * hop
    #

    best_offset = (
        expected_offset_sec
        + (
            best_index
            - center_index
        )
        * hop_sec
    )

    refined_offset, refined_corr = _refine_offset_interpolated(
        reference_envelope,
        target_envelope,
        target_center_sec,
        best_offset,
        window_sec,
        hop_sec,
    )

    # Тонкое уточнение может слегка изменить peak correlation.
    if refined_corr >= best_corr:
        best_offset = refined_offset
        best_corr = refined_corr

    return {
        "offset": float(best_offset),
        "correlation": float(best_corr),
        "margin": float(margin),
        "second_correlation": float(second_corr),
    }


# ============================================================
# СОХРАНЕНИЕ ПЕРВОЙ КАРТЫ В JSON
# ============================================================

def save_analysis_state(
    output_dir: Path,
    global_result,
    local_results,
):
    """
    Сохраняет всё, что нашли на текущем этапе.

    Это позволит нам анализировать результаты даже
    если программа будет остановлена перед следующим этапом.
    """

    path = (
        output_dir
        / "raw_alignment_state.json"
    )

    data = {
        "global_alignment":
            global_result,

        "local_results":
            local_results,
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
# ОСНОВНОЙ PIPELINE ЧАСТИ 4
# ============================================================

def run_alignment_stage(
    selected: SelectedTracks,
    output_dir: Path,
    analysis,
    progress: Progress,
):
    """
    Запускает математическое сопоставление.
    """

    reference_envelope = (
        analysis[
            "reference_envelope"
        ]["values"]
    )

    target_envelope = (
        analysis[
            "target_envelope"
        ]["values"]
    )

    # --------------------------------------------------------
    # 1. GLOBAL ALIGNMENT
    # --------------------------------------------------------

    global_result = (
        global_alignment_search(
            reference_envelope,
            target_envelope,
            selected.reference,
            selected.target,
            progress,
        )
    )

    save_global_results_csv(
        output_dir,
        global_result,
    )

    # --------------------------------------------------------
    # 2. LOCAL ALIGNMENT
    # --------------------------------------------------------

    local_results = (
        build_local_anchor_table(
            reference_envelope,
            target_envelope,
            global_result[
                "target_duration"
            ],
            global_result[
                "coarse_offset"
            ],
            progress,
        )
    )

    save_local_anchor_csv(
        output_dir,
        local_results,
    )

    # --------------------------------------------------------
    # 3. СЫРОЕ СОСТОЯНИЕ
    # --------------------------------------------------------

    save_analysis_state(
        output_dir,
        global_result,
        local_results,
    )

    # --------------------------------------------------------
    # 4. GRAPHICS
    # --------------------------------------------------------

    progress.stage(
        "ГРАФИЧЕСКИЕ ОТЧЁТЫ",
        0.0,
    )

    save_waveform_overview_jpeg(
        output_dir,
        analysis[
            "reference_waveform_times"
        ],
        analysis[
            "reference_waveform"
        ],
        analysis[
            "target_waveform_times"
        ],
        analysis[
            "target_waveform"
        ],
        progress,
    )

    progress.stage(
        "ENVELOPE GRAPH",
        25.0,
    )

    save_envelope_overview_jpeg(
        output_dir,
        analysis[
            "reference_envelope"
        ],
        analysis[
            "target_envelope"
        ],
        progress,
    )

    progress.stage(
        "GLOBAL ALIGNMENT GRAPH",
        50.0,
    )

    save_global_alignment_jpeg(
        output_dir,
        global_result,
        progress,
    )

    progress.stage(
        "LOCAL ANCHOR GRAPH",
        75.0,
    )

    save_local_anchor_jpeg(
        output_dir,
        local_results,
        progress,
    )

    progress.done(
        100.0,
        "Графические отчёты готовы",
    )

    # --------------------------------------------------------
    # 5. SUMMARY
    # --------------------------------------------------------

    summary = (
        calculate_initial_sync_summary(
            global_result,
            local_results,
        )
    )

    save_initial_sync_summary(
        output_dir,
        summary,
    )

    return {
        "global_result":
            global_result,

        "local_results":
            local_results,

        "summary":
            summary,
    }


# ============================================================
# ПЕЧАТЬ ПЕРВОЙ ОЦЕНКИ
# ============================================================

def print_initial_alignment_report(
    alignment,
    output_dir: Path,
):
    """
    Пока это НЕ финальная оценка качества карты.

    Это предварительный отчёт.
    """

    summary = alignment[
        "summary"
    ]

    print()
    print("=" * 78)
    print("ПРЕДВАРИТЕЛЬНЫЙ ОТЧЁТ О СОВМЕЩЕНИИ")
    print("=" * 78)

    print()

    print(
        "Global offset :",
        f"{summary['global_offset']:+.4f} sec",
    )

    print(
        "Local points  :",
        summary["local_points"],
    )

    print(
        "Valid points  :",
        summary["local_valid"],
    )

    print(
        "Tracked points:",
        summary["local_accepted"],
    )

    print()

    print(
        "Correlation:"
    )

    print(
        "  mean   :",
        f"{summary['mean_correlation']:.4f}",
    )

    print(
        "  median :",
        f"{summary['median_correlation']:.4f}",
    )

    print(
        "  minimum:",
        f"{summary['min_correlation']:.4f}",
    )

    print(
        "  maximum:",
        f"{summary['max_correlation']:.4f}",
    )

    print()

    # --------------------------------------------------------
    # НЕ называем карту хорошей или плохой окончательно.
    #
    # Здесь только предварительный сигнал.
    # --------------------------------------------------------

    if (
        summary[
            "local_accepted"
        ]
        >=
        summary[
            "local_points"
        ]
        * 0.80
        and
        summary[
            "median_correlation"
        ]
        >= 0.45
    ):

        print(
            "ПРЕДВАРИТЕЛЬНАЯ ОЦЕНКА:"
        )

        print(
            "Большинство локальных участков "
            "имеет устойчивое совпадение."
        )

    elif (
        summary[
            "local_accepted"
        ]
        >=
        summary[
            "local_points"
        ]
        * 0.50
        and
        summary[
            "median_correlation"
        ]
        >= 0.25
    ):

        print(
            "ПРЕДВАРИТЕЛЬНАЯ ОЦЕНКА:"
        )

        print(
            "Совпадение есть, но карта требует "
            "дальнейшей очистки."
        )

    else:

        print(
            "ПРЕДВАРИТЕЛЬНАЯ ОЦЕНКА:"
        )

        print(
            "Сейчас слишком много слабых "
            "или неоднозначных совпадений."
        )

    print()

    print(
        "Полная информация сохранена в:"
    )

    print(
        output_dir
    )

    print()

# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC
#
# ЧАСТЬ 5
#
# RAW MEASUREMENTS
#       ↓
# QUALITY ANALYSIS
#       ↓
# OUTLIER REJECTION
#       ↓
# NONLINEAR MAP
#       ↓
# SMOOTH / VALIDATE
#
# ============================================================


# ============================================================
# НАСТРОЙКИ КАЧЕСТВА КАРТЫ
