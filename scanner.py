# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: scanner.py
# ============================================================
# File discovery, ffprobe, selection, validation, metadata and initial pair reporting.
#
# This module is part of the modular version of the project.
# Do not run it directly; start the pipeline with start_sync.py.
# ============================================================

from __future__ import annotations

import json
import math
import re
import subprocess
import time

from config import *
from common import *

def check_ffprobe() -> None:
    """
    Проверяет наличие ffprobe.
    """

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-version",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    except FileNotFoundError:
        print()
        print("=" * 78)
        print("ОШИБКА")
        print("=" * 78)
        print(
            "Программа ffprobe не найдена."
        )
        print()
        print(
            "Убедись, что FFmpeg установлен "
            "и ffprobe доступен из PATH."
        )
        print()

        raise RuntimeError(
            "ffprobe не найден."
        )

    if result.returncode != 0:
        raise RuntimeError(
            "ffprobe установлен, "
            "но не запускается."
        )


# ============================================================
# FFPROBE — ОДИН ФАЙЛ
# ============================================================

def probe_audio_file(
    path: Path,
) -> Optional[AudioInfo]:
    """
    Проверяет файл через ffprobe.

    Возвращает AudioInfo, если найден аудиопоток.
    Иначе None.
    """

    command = [
        "ffprobe",
        "-v",
        "error",

        "-show_entries",
        (
            "stream="
            "index,"
            "codec_name,"
            "codec_long_name,"
            "sample_fmt,"
            "sample_rate,"
            "channels,"
            "channel_layout,"
            "bit_rate,"
            "duration,"
            "codec_type,"
            ":format="
            "format_name,"
            "format_long_name,"
            "duration,"
            "size"
        ),

        "-of",
        "json",

        str(path),
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    except Exception:
        return None

    if result.returncode != 0:
        return None

    try:
        data = json.loads(
            result.stdout
        )
    except Exception:
        return None

    streams = data.get(
        "streams",
        [],
    )

    audio_stream = None

    for stream in streams:
        if (
            stream.get("codec_type")
            == "audio"
        ):
            audio_stream = stream
            break

    if audio_stream is None:
        return None

    fmt = data.get(
        "format",
        {},
    )

    def as_int(
        value,
        default=0,
    ):
        try:
            return int(float(value))
        except Exception:
            return default

    def as_float(
        value,
        default=0.0,
    ):
        try:
            return float(value)
        except Exception:
            return default

    return AudioInfo(
        path=path,

        codec_name=str(
            audio_stream.get(
                "codec_name",
                "",
            )
        ),

        codec_long_name=str(
            audio_stream.get(
                "codec_long_name",
                "",
            )
        ),

        sample_format=str(
            audio_stream.get(
                "sample_fmt",
                "",
            )
        ),

        sample_rate=as_int(
            audio_stream.get(
                "sample_rate",
                0,
            )
        ),

        channels=as_int(
            audio_stream.get(
                "channels",
                0,
            )
        ),

        channel_layout=str(
            audio_stream.get(
                "channel_layout",
                "",
            )
        ),

        bit_rate=as_int(
            audio_stream.get(
                "bit_rate",
                0,
            )
        ),

        duration=as_float(
            audio_stream.get(
                "duration",
                fmt.get(
                    "duration",
                    0,
                ),
            )
        ),

        stream_index=as_int(
            audio_stream.get(
                "index",
                0,
            )
        ),

        format_name=str(
            fmt.get(
                "format_name",
                "",
            )
        ),

        format_long_name=str(
            fmt.get(
                "format_long_name",
                "",
            )
        ),

        size_bytes=as_int(
            fmt.get(
                "size",
                0,
            )
        ),
    )


# ============================================================
# ФОРМАТ ОПИСАНИЯ ДОРОЖКИ
# ============================================================

def describe_audio(
    info: AudioInfo,
) -> str:
    """
    Короткое описание для меню.

    Если контейнер/демультиплексор не сообщает длительность,
    показываем это явно, а не подменяем неизвестное значение
    нулём. Реальная длительность будет получена после decode.
    """

    codec = (
        info.codec_name.upper()
        if info.codec_name
        else "?"
    )

    if info.channels == 1:
        channel_text = "mono"
    elif info.channels == 2:
        channel_text = "stereo"
    else:
        channel_text = f"{info.channels} ch"

    rate = (
        f"{info.sample_rate / 1000:.1f} kHz"
        if info.sample_rate
        else "? kHz"
    )

    if math.isfinite(info.duration) and info.duration > 0:
        duration = format_seconds(info.duration)
    else:
        duration = "duration unknown"

    return (
        f"{codec}, "
        f"{channel_text}, "
        f"{rate}, "
        f"{duration}"
    )


# ============================================================
# ПОИСК АУДИОФАЙЛОВ
# ============================================================

def scan_audio_files(
    progress: Progress,
) -> list[AudioInfo]:
    """
    Сканирует папку скрипта и определяет аудиофайлы через ffprobe.

    Расширение больше НЕ является обязательным условием:
    если ffprobe видит аудиопоток внутри файла, файл попадает
    в список. Это позволяет принимать raw DTS/AC3/PCM, контейнеры
    MKA/MOV/MP4/MKV и другие форматы без постоянного расширения
    списка вручную.
    """

    progress.stage(
        "СКАНИРОВАНИЕ ПАПКИ",
        1.0,
    )

    try:
        entries = sorted(
            BASE_DIR.iterdir(),
            key=lambda p: p.name.lower(),
        )
    except Exception as exc:
        raise RuntimeError(
            "Не удалось прочитать папку:\n"
            f"{BASE_DIR}\n"
            f"{exc}"
        )

    candidates = []

    for path in entries:
        if not path.is_file():
            continue

        if path.suffix.lower() in IGNORED_EXTENSIONS:
            continue

        if path.stem.endswith(RENDER_OUTPUT_SUFFIX):
            continue

        # Производственные файлы проекта и явные временные raw-файлы
        # не нужны в пользовательском меню.
        if path.suffix.lower() in {
            ".pcm", ".raw", ".npy", ".bin",
        }:
            continue

        candidates.append(path)

    if not candidates:
        raise RuntimeError(
            "В папке нет файлов для проверки через ffprobe.\n"
            f"Папка: {BASE_DIR}"
        )

    print()
    print(
        f"Найдено файлов-кандидатов: {len(candidates)}"
    )
    print()

    result = []
    total = len(candidates)

    for index, path in enumerate(candidates, start=1):
        percent = index / total * 100.0
        progress.update(
            percent,
            f"Проверка: {path.name}",
        )

        info = probe_audio_file(path)
        if info is not None:
            result.append(info)

    progress.done(
        100.0,
        f"Найдено аудиодорожек: {len(result)}",
    )

    return result


# ============================================================
# ВЫВОД СПИСКА ФАЙЛОВ
# ============================================================

def print_audio_list(
    audio_files: list[AudioInfo],
):
    print()
    print("=" * 78)
    print("ДОСТУПНЫЕ АУДИОДОРОЖКИ")
    print("=" * 78)
    print()

    for index, info in enumerate(
        audio_files,
        start=1,
    ):

        print(
            f"{index}. "
            f"{info.path.name}"
        )

        print(
            f"   {describe_audio(info)}"
        )

        print()


# ============================================================
# ПОЛУЧЕНИЕ ВЫБОРА ПОЛЬЗОВАТЕЛЯ
# ============================================================

def ask_selection(
    audio_files: list[AudioInfo],
    title: str,
    excluded: Optional[AudioInfo] = None,
) -> AudioInfo:
    """
    Универсальное меню выбора.

    excluded используется для того, чтобы Target нельзя
    было выбрать тем же файлом, что и Reference.
    """

    available = []

    for info in audio_files:

        if (
            excluded is not None
            and info.path == excluded.path
        ):
            continue

        available.append(info)

    if not available:
        raise RuntimeError(
            "Недостаточно аудиофайлов "
            "для выбора двух разных дорожек."
        )

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()

    for index, info in enumerate(
        available,
        start=1,
    ):

        print(
            f"{index}. "
            f"{info.path.name}"
        )

        print(
            f"   {describe_audio(info)}"
        )

        print()

    while True:

        try:
            answer = input(
                "Введите номер и нажмите Enter: "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print()
            print(
                "Выбор отменён пользователем."
            )
            raise SystemExit(1)

        if not answer:
            print(
                "Нужно ввести номер."
            )
            continue

        try:
            number = int(answer)

        except ValueError:
            print(
                "Введите именно номер из списка."
            )
            continue

        if number < 1 or number > len(
            available
        ):
            print(
                "Такого номера нет."
            )
            continue

        selected = available[
            number - 1
        ]

        print()
        print(
            "Выбрано:"
        )
        print(
            selected.path.name
        )

        print()

        return selected


# ============================================================
# ФИНАЛЬНОЕ ПОДТВЕРЖДЕНИЕ ПАРЫ
# ============================================================

def confirm_selected_tracks(
    selected: SelectedTracks,
):
    print()
    print("=" * 78)
    print("ВЫБРАННЫЕ ДОРОЖКИ")
    print("=" * 78)

    print()
    print(
        "REFERENCE:"
    )
    print(
        selected.reference.path.name
    )
    print(
        "  ",
        describe_audio(
            selected.reference
        ),
    )

    print()
    print(
        "TARGET / РАССИНХРОН:"
    )
    print(
        selected.target.path.name
    )
    print(
        "  ",
        describe_audio(
            selected.target
        ),
    )

    print()

    answer = input(
        "Начать анализ? [Y/N]: "
    ).strip().lower()

    if answer not in {
        "y",
        "yes",
        "д",
        "да",
    }:

        print()
        print(
            "Анализ отменён."
        )

        raise SystemExit(0)


# ============================================================
# СОЗДАНИЕ РАБОЧЕЙ ПАПКИ
# ============================================================

def sanitize_filename(
    value: str,
) -> str:
    """
    Убирает опасные символы из имени папки.
    """

    value = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        value,
    )

    value = value.strip()

    if not value:
        value = "audio_sync"

    return value


def create_output_directory(
    target: AudioInfo,
) -> Path:
    """
    Создаёт отдельную рабочую папку.

    Например:

        Barry..._DELAY 0ms_WAVEFORM_SYNC
    """

    stem = target.path.stem

    dirname = (
        sanitize_filename(
            stem
        )
        + OUTPUT_FOLDER_SUFFIX
    )

    output_dir = (
        BASE_DIR
        / dirname
    )

    # A second run must not overwrite previous audio or diagnostic files.
    index = 2
    while output_dir.exists():
        output_dir = BASE_DIR / (dirname + f"_{index}")
        index += 1
    output_dir.mkdir(parents=True, exist_ok=False)

    return output_dir


# ============================================================
# ПРОВЕРКА СОСТОЯНИЯ ДОРОЖЕК
# ============================================================

def validate_selected_tracks(
    selected: SelectedTracks,
):
    """
    Базовая проверка пары.

    Длительность не считается обязательной на этапе выбора:
    некоторые raw audio demuxer'ы (например DTS elementary stream)
    могут не сообщать её в ffprobe. В таком случае она будет получена
    по фактическому числу декодированных samples на ШАГЕ 2.
    """

    reference = selected.reference
    target = selected.target

    if reference.channels < 1 or target.channels < 1:
        raise RuntimeError(
            "У одной из дорожек не определено количество каналов."
        )

    if reference.sample_rate <= 0:
        raise RuntimeError(
            "У Reference не определена частота дискретизации."
        )

    if target.sample_rate <= 0:
        raise RuntimeError(
            "У Target не определена частота дискретизации."
        )


# ============================================================
# ИНФОРМАЦИЯ О ПАРЕ
# ============================================================

def print_pair_summary(
    selected: SelectedTracks,
):
    reference = selected.reference
    target = selected.target

    ref_known = math.isfinite(reference.duration) and reference.duration > 0
    tgt_known = math.isfinite(target.duration) and target.duration > 0

    print()
    print("=" * 78)
    print("АНАЛИЗ ВЫБРАННОЙ ПАРЫ")
    print("=" * 78)
    print()

    if ref_known:
        print(
            "Reference duration :",
            f"{reference.duration:.3f} s",
            "(",
            format_seconds(reference.duration),
            ")",
        )
    else:
        print("Reference duration : неизвестна до декодирования")

    if tgt_known:
        print(
            "Target duration    :",
            f"{target.duration:.3f} s",
            "(",
            format_seconds(target.duration),
            ")",
        )
    else:
        print("Target duration    : неизвестна до декодирования")

    if ref_known and tgt_known:
        duration_difference = target.duration - reference.duration
        print("Difference         :", f"{duration_difference:+.3f} s")
    else:
        print("Difference         : будет рассчитана после декодирования")

    print()
    print("Reference:")
    print("  codec      :", reference.codec_name)
    print("  channels   :", reference.channels)
    print("  layout     :", reference.channel_layout or "unknown")
    print("  sample rate:", reference.sample_rate)

    print()
    print("Target:")
    print("  codec      :", target.codec_name)
    print("  channels   :", target.channels)
    print("  layout     :", target.channel_layout or "unknown")
    print("  sample rate:", target.sample_rate)

    print()


# ============================================================
# СОХРАНЕНИЕ ОСНОВНОЙ ИНФОРМАЦИИ
# ============================================================

def audio_info_to_dict(
    info: AudioInfo,
) -> dict:

    return {
        "path": str(
            info.path
        ),

        "filename": info.path.name,

        "codec_name":
            info.codec_name,

        "codec_long_name":
            info.codec_long_name,

        "sample_format":
            info.sample_format,

        "sample_rate":
            info.sample_rate,

        "channels":
            info.channels,

        "channel_layout":
            info.channel_layout,

        "bit_rate":
            info.bit_rate,

        "duration":
            info.duration,

        "stream_index":
            info.stream_index,

        "format_name":
            info.format_name,

        "format_long_name":
            info.format_long_name,

        "size_bytes":
            info.size_bytes,
    }


def save_initial_metadata(
    output_dir: Path,
    selected: SelectedTracks,
):
    """
    Сохраняет исходную информацию.
    """

    metadata = {
        "created": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "script":
            Path(__file__).name,

        "analysis_sample_rate":
            ANALYSIS_SAMPLE_RATE,

        "analysis_block_sec":
            ANALYSIS_BLOCK_SEC,

        "map_step_sec":
            MAP_STEP_SEC,

        "reference":
            audio_info_to_dict(
                selected.reference
            ),

        "target":
            audio_info_to_dict(
                selected.target
            ),
    }

    path = (
        output_dir
        / "initial_metadata.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# ВРЕМЕННАЯ ИНФОРМАЦИЯ О ПЛАНЕ РАБОТЫ
# ============================================================

def print_algorithm_plan():
    print()
    print("=" * 78)
    print("ПЛАН АЛГОРИТМА")
    print("=" * 78)

    print(
        """
1. Декодировать первый канал Reference.
2. Декодировать первый канал Target.
3. Привести их к единой системе анализа.
4. Построить waveform / амплитудную огибающую.
5. Нормализовать громкость каждого локального участка.
6. Выполнить грубое глобальное сопоставление.
7. Разбить фильм на независимые временные окна.
8. Для каждого окна искать лучшее соответствие волн.
9. Проверять силу найденного совпадения.
10. Отбрасывать ложные совпадения.
11. Строить нелинейную временную карту.
12. Сглаживать карту без уничтожения реального drift.
13. Проверять монотонность времени.
14. Строить графические JPEG-отчёты.
15. Выдавать оценку качества карты.
16. Предлагать применить карту к исходному Target.
17. Создать новую синхронизированную аудиодорожку.
18. Проверить финальный файл.
"""
    )
