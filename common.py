# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: common.py
# ============================================================
# Shared dataclasses, progress display, and general utility functions.
#
# This module is part of the modular version of the project.
# Do not run it directly; start the pipeline with start_sync.py.
# ============================================================

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import BASE_DIR


@dataclass
class AudioInfo:
    """
    Информация об аудиофайле.
    """

    path: Path

    codec_name: str
    codec_long_name: str

    sample_format: str
    sample_rate: int

    channels: int
    channel_layout: str

    bit_rate: int

    duration: float

    stream_index: int

    format_name: str
    format_long_name: str

    size_bytes: int

@dataclass
class SelectedTracks:
    """
    Выбранные пользователем Reference и Target.
    """

    reference: AudioInfo
    target: AudioInfo

class Progress:
    """
    Универсальный вывод прогресса.

    Используется всеми серьёзными этапами программы.

    Пример:

        [ 37.4%] Декодирование Target | 1:07:21 / 3:00:00

    При наличии скорости и ETA:

        [ 37.4%] Декодирование Target |
        speed 8.31x | ETA 00:22:14

    """

    def __init__(self, context=""):
        self.started = time.time()
        self.stage_started = time.time()
        self.context = context

    # --------------------------------------------------------
    # Начало крупного этапа
    # --------------------------------------------------------

    def stage(self, name: str, overall_percent: float = 0.0):
        self.stage_started = time.time()

        print()
        print("=" * 78)

        # Заголовок этапа — это только заголовок.
        # Процент относится к реальной операции и показывается в update()/done().
        print(f"{self.context} | {name}" if self.context else name)
        print("=" * 78)

    # --------------------------------------------------------
    # Обычный процент
    # --------------------------------------------------------

    def update(
        self,
        percent: float,
        message: str = "",
        processed_sec: Optional[float] = None,
        total_sec: Optional[float] = None,
        speed: Optional[float] = None,
    ):
        percent = max(
            0.0,
            min(
                100.0,
                float(percent),
            ),
        )

        line = f"\r[{percent:6.2f}%]"

        if self.context:
            line += f" {self.context} |"

        if message:
            line += f" {message}"

        if processed_sec is not None:
            line += (
                " | "
                + format_seconds(
                    processed_sec
                )
            )

        if total_sec is not None:
            line += (
                " / "
                + format_seconds(
                    total_sec
                )
            )

        if speed is not None and speed > 0:
            line += (
                f" | {speed:.2f}x"
            )

        print(
            line,
            end="",
            flush=True,
        )

    # --------------------------------------------------------
    # Завершение строки прогресса
    # --------------------------------------------------------

    def done(
        self,
        percent: float,
        message: str = "Готово",
    ):
        self.update(
            percent,
            message,
        )

        print()

    # --------------------------------------------------------
    # ETA
    # --------------------------------------------------------

    def eta(
        self,
        processed_sec: float,
        total_sec: float,
    ) -> float:

        elapsed = (
            time.time()
            - self.stage_started
        )

        if (
            elapsed <= 0
            or processed_sec <= 0
            or total_sec <= processed_sec
        ):
            return 0.0

        speed = (
            processed_sec
            / elapsed
        )

        if speed <= 0:
            return 0.0

        remaining = (
            total_sec
            - processed_sec
        )

        return (
            remaining
            / speed
        )

    # --------------------------------------------------------
    # Скорость
    # --------------------------------------------------------

    def speed(
        self,
        processed_sec: float,
    ) -> float:

        elapsed = (
            time.time()
            - self.stage_started
        )

        if elapsed <= 0:
            return 0.0

        return (
            processed_sec
            / elapsed
        )

def format_seconds(seconds: float) -> str:
    """
    Перевод секунд в HH:MM:SS.
    """

    if not math.isfinite(seconds):
        return "--:--:--"

    seconds = max(
        0.0,
        float(seconds),
    )

    total = int(seconds)

    hours = total // 3600
    minutes = (
        total % 3600
    ) // 60
    secs = total % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )

def build_exact_time_grid(
    start: float,
    end: float,
    step: float,
):
    """
    Равномерная сетка с ОБЯЗАТЕЛЬНЫМ точным конечным значением.

    np.arange() может закончиться чуть раньше end из-за остатка,
    что для APPLY критично: последний кусок Target нельзя терять.
    """

    import numpy as np

    start = float(start)
    end = float(end)
    step = float(step)

    if step <= 0:
        raise ValueError("step должен быть > 0")

    if end < start:
        raise ValueError("end должен быть >= start")

    if end == start:
        return np.array([start], dtype=np.float64)

    values = np.arange(
        start,
        end,
        step,
        dtype=np.float64,
    )

    if len(values) == 0 or values[-1] < end - 1e-12:
        values = np.append(values, end)
    else:
        values[-1] = end

    return values

def format_size(size_bytes: int) -> str:

    value = float(size_bytes)

    if value < 1024:
        return f"{value:.0f} B"

    value /= 1024

    if value < 1024:
        return f"{value:.2f} KB"

    value /= 1024

    if value < 1024:
        return f"{value:.2f} MB"

    value /= 1024

    if value < 1024:
        return f"{value:.2f} GB"

    value /= 1024

    return f"{value:.2f} TB"
