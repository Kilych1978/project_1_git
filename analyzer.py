# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: analyzer.py
# ============================================================
# Audio decoding, first-channel extraction, waveform/envelope construction, and preprocessing.
#
# This module is part of the modular version of the project.
# Do not run it directly; start the pipeline with start_sync.py.
# ============================================================

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time

from config import *
from common import *

# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC
#
# ЧАСТЬ 2
#
# ДЕКОДИРОВАНИЕ ПЕРВОГО КАНАЛА
# + ПОТОКОВОЕ ПОСТРОЕНИЕ WAVEFORM / ENVELOPE
#
# ============================================================

# ============================================================
# ДОПОЛНИТЕЛЬНЫЕ КОНСТАНТЫ АНАЛИЗА
# ============================================================

# Размер блока, которым ffmpeg отдаёт PCM в Python.
#
# 48000 samples = ровно 1 секунда при 48 kHz.
#
# Это удобно:
# обработка идёт кусками,
# RAM не раздувается,
# прогресс можно показывать постоянно.
PCM_BLOCK_SECONDS = 1.0

PCM_BLOCK_SAMPLES = int(
    ANALYSIS_SAMPLE_RATE
    * PCM_BLOCK_SECONDS
)


# Размер окна для амплитудной огибающей.
#
# 20 ms достаточно мало, чтобы сохранить форму
# исходного сигнала, но достаточно много, чтобы
# убрать часть высокочастотного мусора.
ENVELOPE_WINDOW_MS = 20.0

ENVELOPE_WINDOW_SAMPLES = max(
    1,
    int(
        ANALYSIS_SAMPLE_RATE
        * ENVELOPE_WINDOW_MS
        / 1000.0
    ),
)


# Шаг между точками огибающей.
#
# В отличие от WAV waveform мы не обязаны хранить
# каждый sample.
#
# 10 ms -> 100 точек в секунду.
#
# Для трёхчасового фильма это примерно 1.08 млн точек,
# что для NumPy вполне нормально.
ENVELOPE_HOP_MS = 10.0

ENVELOPE_HOP_SAMPLES = max(
    1,
    int(
        ANALYSIS_SAMPLE_RATE
        * ENVELOPE_HOP_MS
        / 1000.0
    ),
)


# ============================================================
# НАЗВАНИЯ РАБОЧИХ ФАЙЛОВ
# ============================================================

REFERENCE_PCM_NAME = (
    "reference_first_channel_s16le.pcm"
)

TARGET_PCM_NAME = (
    "target_first_channel_s16le.pcm"
)

REFERENCE_WAVEFORM_NAME = (
    "reference_waveform.npy"
)

TARGET_WAVEFORM_NAME = (
    "target_waveform.npy"
)

REFERENCE_ENVELOPE_NAME = (
    "reference_envelope.npy"
)

TARGET_ENVELOPE_NAME = (
    "target_envelope.npy"
)

ANALYSIS_INFO_NAME = (
    "analysis_info.json"
)


# ============================================================
# ОПРЕДЕЛЕНИЕ КАНАЛА
# ============================================================

def get_analysis_channel(
    info: AudioInfo,
) -> int:
    """
    Определяет номер физического канала, который
    используется для математического анализа.

    Правило проекта:

        mono:
            канал 0

        stereo / 5.1 / 7.1 / multi-channel:
            первый канал = канал 0

    Таким образом мы не пытаемся анализировать
    центральный канал, где может находиться перевод.
    """

    if info.channels <= 0:
        raise RuntimeError(
            "Количество каналов не определено."
        )

    # Для mono это тоже 0.
    return 0


# ============================================================
# ДЕКОДИРОВАНИЕ ОДНОГО ФАЙЛА
# ============================================================

def decode_first_channel_stream(
    info: AudioInfo,
    output_pcm: Path,
    progress: Progress,
    global_start: float,
    global_end: float,
    estimated_duration: float = 0.0,
):
    """
    Потоковое декодирование первого канала.

    Специально защищено от зависаний raw DTS/AC3 и других
    elementary streams, у которых ffprobe может не сообщить duration.

    stderr ffmpeg непрерывно вычитывается отдельным потоком,
    stdout читается через Queue, а при отсутствии нового PCM
    дольше DECODE_STALL_TIMEOUT_SEC процесс аварийно останавливается
    вместо бесконечного обновления 0.00%.
    """

    channel_index = get_analysis_channel(info)

    decode_role = "REFERENCE" if global_start < 1.0 else "TARGET"
    # Use the other track only as a progress estimate. Actual duration is
    # always measured from the decoded PCM; metadata is never invented.
    progress_duration = info.duration if info.duration > 0 else estimated_duration

    if output_pcm.exists():
        try:
            output_pcm.unlink()
        except Exception:
            pass

    print()
    print("Декодирование:")
    print("  ", info.path.name)

    if info.channels == 1:
        print("  Анализируется: единственный канал")
    else:
        print("  Анализируется: первый канал (channel 0)")

    print("  Исходный codec:", info.codec_name)
    print("  Исходный sample rate:", info.sample_rate)
    if info.duration > 0:
        print("  Длительность:", format_seconds(info.duration))
    else:
        print("  Длительность: неизвестна — будет определена после decode")
    print()

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(info.path),
        "-map",
        f"0:{info.stream_index}",
        "-af",
        f"pan=mono|c0=c{channel_index}",
        "-ar",
        str(ANALYSIS_SAMPLE_RATE),
        "-ac",
        "1",
        "-f",
        "s16le",
        "pipe:1",
    ]

    started = time.time()
    last_data_time = started
    last_progress_time = 0.0
    stdout_queue = queue.Queue(maxsize=8)
    stderr_tail = []
    stderr_lock = threading.Lock()
    stdout_done = False
    bytes_per_sample = 2
    block_bytes = PCM_BLOCK_SAMPLES * bytes_per_sample

    process = None

    def read_stdout():
        try:
            while True:
                raw = process.stdout.read(block_bytes)
                if not raw:
                    break
                stdout_queue.put(raw)
        finally:
            stdout_queue.put(None)

    def drain_stderr():
        try:
            while True:
                raw = process.stderr.readline()
                if not raw:
                    break
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    line = str(raw).rstrip()
                if line:
                    with stderr_lock:
                        stderr_tail.append(line)
                        if len(stderr_tail) > 200:
                            del stderr_tail[:-200]
        except Exception:
            pass

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except FileNotFoundError:
        raise RuntimeError("Не найден ffmpeg.")

    stdout_thread = threading.Thread(
        target=read_stdout,
        name="ffmpeg-stdout-reader",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain_stderr,
        name="ffmpeg-stderr-reader",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    total_samples = 0

    try:
        with output_pcm.open("wb") as pcm_file:
            while not stdout_done:
                try:
                    raw = stdout_queue.get(timeout=0.50)
                except queue.Empty:
                    now = time.time()

                    if process.poll() is None and now - last_data_time > DECODE_STALL_TIMEOUT_SEC:
                        try:
                            process.kill()
                        except Exception:
                            pass
                        with stderr_lock:
                            diagnostic = "\n".join(stderr_tail[-40:])
                        raise RuntimeError(
                            "FFmpeg decode завис: "
                            f"{DECODE_STALL_TIMEOUT_SEC:.0f} секунд нет нового PCM.\n"
                            f"Файл: {info.path.name}\n"
                            "Последние сообщения ffmpeg:\n"
                            + (diagnostic or "(нет сообщения)")
                        )

                    processed_sec = total_samples / ANALYSIS_SAMPLE_RATE
                    if now - last_progress_time >= DECODE_PROGRESS_INTERVAL_SEC:
                        speed = progress.speed(processed_sec)
                        if progress_duration > 0:
                            percent = processed_sec / progress_duration * 100.0
                            eta = progress.eta(processed_sec, progress_duration)
                            progress.update(
                                global_start + (global_end - global_start) * max(0.0, min(1.0, percent / 100.0)),
                                f"Decode {decode_role} | ETA {format_seconds(eta)}" + (' | оценка' if info.duration <= 0 else ''),
                                processed_sec,
                                progress_duration,
                                speed,
                            )
                        else:
                            progress.update(
                                global_start,
                                f"Decode {decode_role} | duration unknown",
                                processed_sec,
                                None,
                                speed,
                            )
                        last_progress_time = now
                    continue

                if raw is None:
                    stdout_done = True
                    break

                pcm_file.write(raw)
                total_samples += len(raw) // bytes_per_sample
                last_data_time = time.time()

                now = time.time()
                if now - last_progress_time < DECODE_PROGRESS_INTERVAL_SEC:
                    continue

                processed_sec = total_samples / ANALYSIS_SAMPLE_RATE
                speed = progress.speed(processed_sec)

                if progress_duration > 0:
                    percent = processed_sec / progress_duration * 100.0
                    eta = progress.eta(processed_sec, progress_duration)
                    progress.update(
                        global_start + (global_end - global_start) * max(0.0, min(1.0, percent / 100.0)),
                        f"Decode {decode_role} | ETA {format_seconds(eta)}" + (' | оценка' if info.duration <= 0 else ''),
                        processed_sec,
                        progress_duration,
                        speed,
                    )
                else:
                    progress.update(
                        global_start,
                        f"Decode {decode_role} | duration unknown",
                        processed_sec,
                        None,
                        speed,
                    )
                last_progress_time = now

    except KeyboardInterrupt:
        try:
            process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=2)
        except Exception:
            pass
        raise

    return_code = process.wait()
    try:
        stdout_thread.join(timeout=2)
    except Exception:
        pass
    try:
        stderr_thread.join(timeout=2)
    except Exception:
        pass

    if return_code != 0:
        with stderr_lock:
            diagnostic = "\n".join(stderr_tail[-80:])
        raise RuntimeError(
            "ffmpeg завершился ошибкой при decode:\n"
            + (diagnostic or "неизвестная ошибка")
        )

    if not output_pcm.exists():
        raise RuntimeError("ffmpeg не создал PCM-файл.")

    size_bytes = output_pcm.stat().st_size
    if size_bytes <= 0:
        with stderr_lock:
            diagnostic = "\n".join(stderr_tail[-80:])
        raise RuntimeError(
            "Полученный PCM-файл пуст.\n"
            + (diagnostic or "ffmpeg не сообщил дополнительной причины.")
        )

    actual_samples = size_bytes // bytes_per_sample
    actual_duration = actual_samples / ANALYSIS_SAMPLE_RATE
    elapsed = time.time() - started
    final_speed = actual_duration / elapsed if elapsed > 0 else 0.0

    print()
    print("  Декодирование завершено:")
    print("    Размер PCM :", format_size(size_bytes))
    print("    Длительность:", f"{actual_duration:.3f} s")
    print("    Скорость    :", f"{final_speed:.2f}x")
    if info.duration > 0:
        print("    Разница     :", f"{actual_duration - info.duration:+.3f} s")
    else:
        print("    Разница     : duration была неизвестна до decode")
    print()

    progress.update(
        global_end,
        f"Decode completed: {decode_role}",
        actual_duration,
        actual_duration,
        final_speed,
    )
    print()

    return {
        "path": str(output_pcm),
        "sample_rate": ANALYSIS_SAMPLE_RATE,
        "channels": 1,
        "sample_format": "s16le",
        "samples": actual_samples,
        "duration": actual_duration,
        "size_bytes": size_bytes,
        "elapsed": elapsed,
    }


# ============================================================
# ЧТЕНИЕ PCM БЛОКАМИ
# ============================================================

def read_pcm_blocks(
    pcm_path: Path,
    progress: Progress,
    stage_start: float,
    stage_end: float,
):
    """
    Читает PCM постепенно и превращает его
    в numpy-массив float32.

    Здесь мы уже можем спокойно держать
    один канал в RAM.

    Для трёх часов:

        48000 * 11100 * 4 bytes

    ≈ 2.1 GB float32

    Поэтому ВСЁ исходное waveform хранить
    в памяти не будем.

    Вместо этого функция сразу строит
    downsampled waveform.

    Это существенно экономит RAM.
    """

    if not pcm_path.exists():
        raise FileNotFoundError(
            str(pcm_path)
        )

    size_bytes = (
        pcm_path.stat().st_size
    )

    bytes_per_sample = 2

    total_samples = (
        size_bytes
        // bytes_per_sample
    )

    total_duration = (
        total_samples
        / ANALYSIS_SAMPLE_RATE
    )

    # --------------------------------------------------------
    # Для визуализации нам не нужен каждый sample.
    #
    # Будем хранить максимум примерно 1000 точек
    # на каждые 10 секунд.
    #
    # Для математического анализа ниже будет использована
    # отдельная envelope с фиксированным шагом.
    # --------------------------------------------------------

    visualization_hz = 100.0

    visual_step = max(
        1,
        int(
            ANALYSIS_SAMPLE_RATE
            / visualization_hz
        ),
    )

    # Временные списки.
    visual_values = []
    visual_times = []

    processed_samples = 0

    started = time.time()

    with pcm_path.open(
        "rb"
    ) as file:

        while True:

            raw = file.read(
                PCM_BLOCK_SAMPLES
                * bytes_per_sample
            )

            if not raw:
                break

            block = (
                __import__("numpy")
                .frombuffer(
                    raw,
                    dtype="<i2",
                )
                .astype(
                    __import__("numpy").float32
                )
            )

            if block.size == 0:
                continue

            # Нормируем в [-1; +1].
            block /= 32768.0

            # ------------------------------------------------
            # Для waveform берём максимум и минимум блока
            # через визуальный sampling.
            #
            # Здесь сохраняется форма, но память не раздувается.
            # ------------------------------------------------

            for local_index in range(
                0,
                len(block),
                visual_step,
            ):

                value = float(
                    block[local_index]
                )

                absolute_sample = (
                    processed_samples
                    + local_index
                )

                time_sec = (
                    absolute_sample
                    / ANALYSIS_SAMPLE_RATE
                )

                visual_values.append(
                    value
                )

                visual_times.append(
                    time_sec
                )

            processed_samples += (
                len(block)
            )

            processed_sec = (
                processed_samples
                / ANALYSIS_SAMPLE_RATE
            )

            percent = (
                processed_sec
                / total_duration
                * 100.0
                if total_duration > 0
                else 0.0
            )

            speed = (
                processed_sec
                / max(
                    0.001,
                    time.time() - started,
                )
            )

            progress.update(
                stage_start
                + (
                    stage_end
                    - stage_start
                )
                * (
                    percent
                    / 100.0
                ),
                "Построение waveform",
                processed_sec,
                total_duration,
                speed,
            )

    import numpy as np

    waveform = np.asarray(
        visual_values,
        dtype=np.float32,
    )

    times = np.asarray(
        visual_times,
        dtype=np.float32,
    )

    progress.update(
        stage_end,
        "Waveform готов",
        total_duration,
        total_duration,
        (
            total_duration
            / max(
                0.001,
                time.time() - started,
            )
        ),
    )

    print()

    return (
        times,
        waveform,
    )


# ============================================================
# ПОСТРОЕНИЕ ОГИБАЮЩЕЙ НАПРЯМУЮ ИЗ PCM
# ============================================================

def build_envelope_from_pcm(
    pcm_path: Path,
    progress: Progress,
    stage_start: float,
    stage_end: float,
):
    """
    Главный математический preprocessing.

    Для каждого окна:

        RMS = sqrt(mean(x²))

    затем полученная огибающая
    нормализуется.

    Ключевой момент:
    абсолютная громкость Reference и Target
    нам НЕ важна.

    Например:

        Reference:
        0.85 0.72 0.31 0.17 ...

        Target:
        0.21 0.18 0.08 0.04 ...

    после локальной нормализации форма будет
    сопоставима.

    ------------------------------------------------------------

    ОГИБАЮЩАЯ НУЖНА ДЛЯ ПОИСКА.

    САМА WAVEOFRM НУЖНА ДЛЯ ВИЗУАЛИЗАЦИИ.

    ------------------------------------------------------------
    """

    import numpy as np

    if not pcm_path.exists():
        raise FileNotFoundError(
            str(pcm_path)
        )

    size_bytes = (
        pcm_path.stat().st_size
    )

    bytes_per_sample = 2

    total_samples = (
        size_bytes
        // bytes_per_sample
    )

    total_duration = (
        total_samples
        / ANALYSIS_SAMPLE_RATE
    )

    window = ENVELOPE_WINDOW_SAMPLES
    hop = ENVELOPE_HOP_SAMPLES

    # --------------------------------------------------------
    # Количество точек огибающей.
    # --------------------------------------------------------

    if total_samples <= window:

        envelope_count = 1

    else:

        envelope_count = (
            (
                total_samples
                - window
            )
            // hop
        ) + 1

    envelope = np.zeros(
        envelope_count,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Вместо загрузки всего PCM сразу:
    # читаем блоками.
    #
    # Нам нужна небольшая перекрывающая область между
    # соседними блоками.
    # --------------------------------------------------------

    chunk_samples = max(
        PCM_BLOCK_SAMPLES
        * 10,
        window
        * 4,
    )

    overlap_samples = (
        window
        + hop
    )

    processed_samples = 0
    envelope_index = 0

    carry = np.empty(
        0,
        dtype=np.float32,
    )

    started = time.time()

    with pcm_path.open(
        "rb"
    ) as file:

        while True:

            raw = file.read(
                chunk_samples
                * bytes_per_sample
            )

            if not raw:
                break

            block = np.frombuffer(
                raw,
                dtype="<i2",
            ).astype(
                np.float32
            )

            block /= 32768.0

            if carry.size:
                data = np.concatenate(
                    (
                        carry,
                        block,
                    )
                )
            else:
                data = block

            # ------------------------------------------------
            # Сколько полных окон можно обработать.
            # ------------------------------------------------

            available = len(data)

            if available >= window:

                max_start = (
                    (
                        available
                        - window
                    )
                    // hop
                ) * hop

                starts = range(
                    0,
                    max_start + 1,
                    hop,
                )

                for start in starts:

                    if envelope_index >= envelope_count:
                        break

                    segment = data[
                        start:
                        start + window
                    ]

                    if segment.size == 0:
                        continue

                    # RMS.
                    rms = np.sqrt(
                        np.mean(
                            segment
                            * segment
                        )
                    )

                    envelope[
                        envelope_index
                    ] = rms

                    envelope_index += 1

                # ------------------------------------------------
                # Оставляем хвост для следующего блока.
                # ------------------------------------------------

                consumed = (
                    max_start
                    + hop
                )

                if consumed < len(data):

                    carry = data[
                        consumed:
                    ]

                else:

                    carry = np.empty(
                        0,
                        dtype=np.float32,
                    )

            else:

                carry = data

            processed_samples += len(
                block
            )

            processed_sec = (
                processed_samples
                / ANALYSIS_SAMPLE_RATE
            )

            percent = (
                processed_sec
                / total_duration
                * 100.0
                if total_duration > 0
                else 0.0
            )

            speed = (
                processed_sec
                / max(
                    0.001,
                    time.time() - started,
                )
            )

            progress.update(
                stage_start
                + (
                    stage_end
                    - stage_start
                )
                * (
                    percent
                    / 100.0
                ),

                "Построение огибающей",

                processed_sec,
                total_duration,
                speed,
            )

    # --------------------------------------------------------
    # Удаляем ещё неиспользованные нулевые элементы.
    # --------------------------------------------------------

    if envelope_index < len(envelope):

        envelope = envelope[
            :envelope_index
        ]

    # --------------------------------------------------------
    # Защита от полной тишины.
    # --------------------------------------------------------

    if envelope.size == 0:

        raise RuntimeError(
            "Не удалось построить "
            "амплитудную огибающую."
        )

    # --------------------------------------------------------
    # Нормализация.
    #
    # ВАЖНО:
    # это глобальная нормализация.
    #
    # Локальная нормализация каждого окна будет
    # выполнена непосредственно перед корреляционным
    # сопоставлением.
    # --------------------------------------------------------

    envelope -= np.mean(
        envelope
    )

    std = np.std(
        envelope
    )

    if std > 1e-12:

        envelope /= std

    # --------------------------------------------------------
    # Небольшое ограничение выбросов.
    #
    # Не удаляем реальные пики полностью,
    # только уменьшаем влияние экстремальных выбросов
    # на корреляцию.
    # --------------------------------------------------------

    median = np.median(
        envelope
    )

    mad = np.median(
        np.abs(
            envelope - median
        )
    )

    if mad > 1e-12:

        limit = (
            12.0
            * mad
        )

        envelope = np.clip(
            envelope,
            median - limit,
            median + limit,
        )

    elapsed = (
        time.time()
        - started
    )

    speed = (
        total_duration
        / max(
            0.001,
            elapsed,
        )
    )

    progress.update(
        stage_end,
        "Огибающая готова",
        total_duration,
        total_duration,
        speed,
    )

    print()

    return {
        "values": envelope,
        "sample_rate": (
            1000.0
            / ENVELOPE_HOP_MS
        ),
        "hop_sec": (
            ENVELOPE_HOP_MS
            / 1000.0
        ),
        "window_sec": (
            ENVELOPE_WINDOW_MS
            / 1000.0
        ),
        "duration": (
            len(envelope)
            * ENVELOPE_HOP_MS
            / 1000.0
        ),
    }


# ============================================================
# ПОЛНАЯ ПОДГОТОВКА ОБЕИХ ДОРОЖЕК
# ============================================================

def prepare_audio_analysis(
    selected: SelectedTracks,
    output_dir: Path,
    progress: Progress,
):
    """
    Полностью выполняет preprocessing:

        Reference -> first channel -> PCM
        Target    -> first channel -> PCM

        затем:

        PCM -> waveform
        PCM -> envelope

    """

    import numpy as np

    print()
    print("=" * 78)
    print("ПОДГОТОВКА AUDIO DATA")
    print("=" * 78)

    # --------------------------------------------------------
    # Файлы
    # --------------------------------------------------------

    reference_pcm = (
        output_dir
        / REFERENCE_PCM_NAME
    )

    target_pcm = (
        output_dir
        / TARGET_PCM_NAME
    )

    # --------------------------------------------------------
    # 1. REFERENCE DECODE
    # --------------------------------------------------------

    progress.stage(
        "ДЕКОДИРОВАНИЕ REFERENCE",
        0.0,
    )

    reference_decode = (
        decode_first_channel_stream(
            selected.reference,
            reference_pcm,
            progress,
            0.0,
            25.0,
            estimated_duration=selected.target.duration,
        )
    )

    progress.done(
        25.0,
        "Reference decoded",
    )

    # --------------------------------------------------------
    # 2. TARGET DECODE
    # --------------------------------------------------------

    progress.stage(
        "ДЕКОДИРОВАНИЕ TARGET",
        25.0,
    )

    target_decode = (
        decode_first_channel_stream(
            selected.target,
            target_pcm,
            progress,
            25.0,
            50.0,
            estimated_duration=reference_decode['duration'],
        )
    )

    progress.done(
        50.0,
        "Target decoded",
    )

    # --------------------------------------------------------
    # 3. Reference waveform
    # --------------------------------------------------------

    progress.stage(
        "ПОСТРОЕНИЕ REFERENCE WAVEFORM",
        50.0,
    )

    reference_times, reference_waveform = (
        read_pcm_blocks(
            reference_pcm,
            progress,
            50.0,
            62.5,
        )
    )

    np.save(
        output_dir
        / REFERENCE_WAVEFORM_NAME,
        reference_waveform,
    )

    np.save(
        output_dir
        / "reference_waveform_times.npy",
        reference_times,
    )

    progress.done(
        62.5,
        "Reference waveform готов",
    )

    # --------------------------------------------------------
    # 4. Target waveform
    # --------------------------------------------------------

    progress.stage(
        "ПОСТРОЕНИЕ TARGET WAVEFORM",
        62.5,
    )

    target_times, target_waveform = (
        read_pcm_blocks(
            target_pcm,
            progress,
            62.5,
            75.0,
        )
    )

    np.save(
        output_dir
        / TARGET_WAVEFORM_NAME,
        target_waveform,
    )

    np.save(
        output_dir
        / "target_waveform_times.npy",
        target_times,
    )

    progress.done(
        75.0,
        "Target waveform готов",
    )

    # --------------------------------------------------------
    # 5. Reference envelope
    # --------------------------------------------------------

    progress.stage(
        "ПОСТРОЕНИЕ REFERENCE ENVELOPE",
        75.0,
    )

    reference_envelope = (
        build_envelope_from_pcm(
            reference_pcm,
            progress,
            75.0,
            87.5,
        )
    )

    np.save(
        output_dir
        / REFERENCE_ENVELOPE_NAME,
        reference_envelope[
            "values"
        ],
    )

    progress.done(
        87.5,
        "Reference envelope готов",
    )

    # --------------------------------------------------------
    # 6. Target envelope
    # --------------------------------------------------------

    progress.stage(
        "ПОСТРОЕНИЕ TARGET ENVELOPE",
        87.5,
    )

    target_envelope = (
        build_envelope_from_pcm(
            target_pcm,
            progress,
            87.5,
            100.0,
        )
    )

    np.save(
        output_dir
        / TARGET_ENVELOPE_NAME,
        target_envelope[
            "values"
        ],
    )

    progress.done(
        100.0,
        "Target envelope готов",
    )

    # --------------------------------------------------------
    # Если ffprobe не дал duration, используем фактическую
    # длительность декодированного PCM. Это особенно важно
    # для raw DTS/AC3 и других elementary streams.
    # --------------------------------------------------------

    if reference_decode["duration"] > 0:
        selected.reference.duration = float(
            reference_decode["duration"]
        )

    if target_decode["duration"] > 0:
        selected.target.duration = float(
            target_decode["duration"]
        )

    # --------------------------------------------------------
    # Сохраняем описание анализа.
    # --------------------------------------------------------

    analysis_info = {

        "analysis_sample_rate":
            ANALYSIS_SAMPLE_RATE,

        "pcm_format":
            "s16le",

        "pcm_channels":
            1,

        "envelope_window_ms":
            ENVELOPE_WINDOW_MS,

        "envelope_hop_ms":
            ENVELOPE_HOP_MS,

        "reference_decode":
            reference_decode,

        "target_decode":
            target_decode,

        "reference_envelope":
            {
                "points":
                    len(
                        reference_envelope[
                            "values"
                        ]
                    ),

                "duration":
                    reference_envelope[
                        "duration"
                    ],

                "hop_sec":
                    reference_envelope[
                        "hop_sec"
                    ],
            },

        "target_envelope":
            {
                "points":
                    len(
                        target_envelope[
                            "values"
                        ]
                    ),

                "duration":
                    target_envelope[
                        "duration"
                    ],

                "hop_sec":
                    target_envelope[
                        "hop_sec"
                    ],
            },
    }

    with (
        output_dir
        / ANALYSIS_INFO_NAME
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            analysis_info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 78)
    print("AUDIO PREPROCESSING COMPLETE")
    print("=" * 78)

    print()
    print(
        "Reference envelope:",
        f"{len(reference_envelope['values']):,}",
        "points",
    )

    print(
        "Target envelope   :",
        f"{len(target_envelope['values']):,}",
        "points",
    )

    print()
    print(
        "Рабочие данные сохранены в:"
    )

    print(
        output_dir
    )

    print()

    return {
        "reference_pcm":
            reference_pcm,

        "target_pcm":
            target_pcm,

        "reference_waveform":
            reference_waveform,

        "reference_waveform_times":
            reference_times,

        "target_waveform":
            target_waveform,

        "target_waveform_times":
            target_times,

        "reference_envelope":
            reference_envelope,

        "target_envelope":
            target_envelope,
    }
