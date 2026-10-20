# ============================================================
# UNIVERSAL AUDIO WAVEFORM SYNC — MODULAR COMPONENT
# MODULE: renderer.py
# ============================================================
# Full-target adaptive render, codec selection, duration/format checking and final rendered validation.
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
import numpy as np

from config import *
from common import *
from scanner import *
from analyzer import *
from validator import *

def _finite_positive(value) -> bool:
    try:
        value = float(value)
    except Exception:
        return False
    return math.isfinite(value) and value > 0.0


def _available_ffmpeg_encoders() -> set[str]:
    """Возвращает набор доступных encoder'ов текущего ffmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return set()

    if result.returncode != 0:
        return set()

    available = set()
    for line in result.stdout.splitlines():
        match = re.search(r"^\s*[A-Z.]{6}\s+([A-Za-z0-9_]+)\s", line)
        if match:
            available.add(match.group(1).lower())
    return available


def _get_render_codec_plan(info: AudioInfo) -> dict:
    """
    Определяет безопасный формат результата.

    Обычные кодеки сохраняются по возможности в том же codec family.
    DTS-HD/MA не переводится молча в lossy DTS: вместо этого выбирается
    PCM WAV/RF64, что сохраняет decoded signal losslessly.
    """
    codec = (info.codec_name or "").lower().strip()
    long_name = (info.codec_long_name or "").lower()

    aliases = {
        "dts": "dca",
        "mp3": "libmp3lame",
        "opus": "libopus",
        "vorbis": "libvorbis",
    }
    candidate = aliases.get(codec, codec)
    available = _available_ffmpeg_encoders()

    is_dts_hd = codec == "dts" and (
        "dts-hd" in long_name
        or "dts hd" in long_name
        or "master audio" in long_name
    )

    if is_dts_hd:
        return {
            "source_codec": codec,
            "encoder": "pcm_s32le",
            "container": "wav",
            "extension": ".wav",
            "pcm_fallback": True,
            "reason": (
                "TARGET is DTS-HD / DTS-HD MA. FFmpeg can decode it losslessly, "
                "but a safe lossless DTS-HD re-encode is not available in the "
                "current encoder set. PCM WAV/RF64 preserves the decoded signal losslessly."
            ),
        }

    if candidate.lower() not in available:
        # Универсальная безопасная ветка: если исходный codec нельзя собрать
        # обратно, не теряем сигнал из-за forced lossy fallback.
        return {
            "source_codec": codec,
            "encoder": "pcm_s32le",
            "container": "wav",
            "extension": ".wav",
            "pcm_fallback": True,
            "reason": (
                f"No suitable encoder for source codec '{codec}' is available "
                "in the current FFmpeg build. PCM WAV/RF64 was selected to preserve "
                "the decoded signal losslessly instead of using an unsafe codec downgrade."
            ),
        }

    return {
        "source_codec": codec,
        "encoder": candidate,
        "container": info.format_name.split(",")[0] if info.format_name else "",
        "extension": info.path.suffix or "",
        "pcm_fallback": False,
        "reason": "Original codec family preserved.",
    }


def _select_encoder_for_target(info: AudioInfo) -> str:
    return str(_get_render_codec_plan(info)["encoder"])


def _build_encoder_options(
    info: AudioInfo,
    encoder: str,
    codec_plan: Optional[dict] = None,
    output_path: Optional[Path] = None,
) -> list[str]:
    """Строит параметры финального encoder'а."""
    codec_plan = codec_plan or _get_render_codec_plan(info)
    encoder = str(encoder)
    options = [
        "-c:a", encoder,
        "-ar", str(info.sample_rate),
        "-ac", str(info.channels),
    ]

    if codec_plan.get("pcm_fallback"):
        # WAV/RF64: для больших PCM (>4 GiB) FFmpeg автоматически переключится
        # на RF64, сохраняя корректную длительность в плеерах.
        options += [
            "-f", "wav",
            "-rf64", "auto",
        ]
        if info.channel_layout:
            options += ["-channel_layout", info.channel_layout]
        return options

    if (
        info.bit_rate > 0
        and info.codec_name.lower() not in {
            "flac", "alac", "wavpack", "tta", "truehd", "mlp",
        }
        and not info.codec_name.lower().startswith("pcm_")
    ):
        options += ["-b:a", str(info.bit_rate)]

    if info.codec_name.lower() in {"flac", "alac"}:
        if info.sample_format in {"s16", "s32", "s16p", "s32p"}:
            options += ["-sample_fmt", info.sample_format.replace("p", "")]

    if info.channel_layout:
        options += ["-channel_layout", info.channel_layout]

    if encoder.lower() == "dca":
        options += ["-strict", "-2"]

    if encoder.lower().startswith("pcm_") and output_path is not None and output_path.suffix.lower() in {".wav", ".wave"}:
        options += ["-f", "wav", "-rf64", "auto"]

    return options


def _known_codec_frame_duration(codec_name: str, sample_rate: int):
    """Известная длительность одного аудиофрейма распространённых frame-based codec."""
    if sample_rate <= 0:
        return None
    frame_samples={"ac3":1536,"dca":512,"dts":512,"aac":1024,"mp3":1152,"opus":960}.get((codec_name or "").lower())
    return None if frame_samples is None else float(frame_samples)/float(sample_rate)


def _duration_diagnostics(output_info: AudioInfo, reference_duration: float)->dict:
    """Отделяет encoded duration от ошибки временной синхронизации."""
    difference=float(output_info.duration-reference_duration)
    frame_duration=_known_codec_frame_duration(output_info.codec_name,output_info.sample_rate)
    result={"reference_duration":float(reference_duration),"output_duration":float(output_info.duration),"difference_sec":difference,"codec_frame_duration_sec":frame_duration,"nearest_frame_count":None,"nearest_frame_multiple_sec":None,"frame_granularity_residual_sec":None}
    if frame_duration and math.isfinite(difference):
        count=int(round(difference/frame_duration))
        result["nearest_frame_count"]=count
        result["nearest_frame_multiple_sec"]=float(count*frame_duration)
        result["frame_granularity_residual_sec"]=float(difference-count*frame_duration)
    return result


def _choose_render_output_path(
    target: AudioInfo,
    codec_plan: Optional[dict] = None,
) -> Path:
    codec_plan = codec_plan or _get_render_codec_plan(target)
    if codec_plan.get("pcm_fallback"):
        suffix = ".wav"
    else:
        suffix = target.path.suffix or ".audio"

    return target.path.with_name(
        target.path.stem + RENDER_OUTPUT_SUFFIX + suffix
    )

def _map_error_for_candidate(
    times: list[float],
    references: list[float],
    left_index: int,
    right_index: int,
) -> float:
    """Максимальная ошибка линейной интерполяции на участке."""

    if right_index <= left_index + 1:
        return 0.0

    t0 = times[left_index]
    t1 = times[right_index]
    r0 = references[left_index]
    r1 = references[right_index]
    dt = t1 - t0

    if dt <= 0:
        return float("inf")

    max_error = 0.0

    for index in range(left_index + 1, right_index):
        alpha = (times[index] - t0) / dt
        predicted = r0 + alpha * (r1 - r0)
        error = abs(references[index] - predicted)
        if error > max_error:
            max_error = error

    return float(max_error)


def build_render_segments(apply_map):
    """
    Сжимает густую 0.25 s карту до адаптивных piecewise-linear
    сегментов. На каждом сегменте offset/reftime аппроксимируется
    прямой, а затем сегмент получает один atempo коэффициент.
    """

    import numpy as np

    target = np.asarray(apply_map["target_time"], dtype=np.float64)
    reference = np.asarray(apply_map["reference_time"], dtype=np.float64)

    if len(target) < 2:
        raise RuntimeError("Apply map слишком короткая для render.")

    segments = []
    left = 0
    total = len(target)

    max_segment_sec = RENDER_MAX_SEGMENT_SEC
    tolerance = RENDER_MAX_MAPPING_ERROR_SEC
    min_segment_sec = RENDER_MIN_SEGMENT_SEC

    while left < total - 1:
        right = left + 1
        best_right = right

        while right < total:
            elapsed = target[right] - target[left]

            if elapsed > max_segment_sec:
                break

            error = _map_error_for_candidate(
                target,
                reference,
                left,
                right,
            )

            if error <= tolerance or elapsed < min_segment_sec:
                best_right = right
                right += 1
                continue

            break

        right = best_right

        if right <= left:
            raise RuntimeError("Не удалось построить render segment.")

        target_start = float(target[left])
        target_end = float(target[right])
        reference_start = float(reference[left])
        reference_end = float(reference[right])

        target_duration = target_end - target_start
        reference_duration = reference_end - reference_start

        if target_duration <= 0 or reference_duration <= 0:
            raise RuntimeError("Некорректный render segment с нулевой длительностью.")

        tempo = target_duration / reference_duration

        if not (MIN_APPLY_TEMPO <= tempo <= MAX_APPLY_TEMPO):
            raise RuntimeError(
                "Render segment требует tempo вне допустимого диапазона: "
                f"{tempo:.9f}"
            )

        segments.append({
            "target_start": target_start,
            "target_end": target_end,
            "reference_start": reference_start,
            "reference_end": reference_end,
            "duration": target_duration,
            "reference_duration": reference_duration,
            "tempo": float(tempo),
            "mapping_error": _map_error_for_candidate(
                target,
                reference,
                left,
                right,
            ),
        })

        left = right

    return segments


def save_render_plan(
    output_dir: Path,
    selected: SelectedTracks,
    segments,
    output_path: Path,
):
    data = {
        "target_file": str(selected.target.path),
        "output_file": str(output_path),
        "segment_count": len(segments),
        "max_mapping_error_sec": max(
            (float(item["mapping_error"]) for item in segments),
            default=0.0,
        ),
        "segments": segments,
    }

    path = output_dir / RENDER_PLAN_JSON
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return path


def _render_filtergraph(segments, start_offset, reference_duration, mapped_end, sample_rate):
    """
    Строит один ffmpeg filter_complex для всей дорожки.

    Входной stream сначала последовательно разбивается через
    asegment. Это принципиально лучше, чем повторять [0:a] в каждом
    branch: один decoded stream проходит по сегментам последовательно,
    без гигантского asplit и без повторного decode.
    """

    if not segments:
        raise RuntimeError("Нет render segments.")

    # Границы между сегментами. asegment создаёт N+1 выходов для N
    # timestamp boundaries.
    boundaries = [
        float(item["target_end"])
        for item in segments[:-1]
    ]

    timestamp_text = "|".join(
        f"{value:.9f}" for value in boundaries
    )

    input_labels = [
        f"seg{index}"
        for index in range(len(segments))
    ]

    chains = []

    if timestamp_text:
        chains.append(
            f"[0:{{STREAM}}]asegment=timestamps='{timestamp_text}'"
            + ""
            + "".join(f"[{label}]" for label in input_labels)
        )
    else:
        chains.append(
            "[0:{STREAM}]" + f"asetpts=PTS-STARTPTS[{input_labels[0]}]"
        )

    tempo_labels = []

    for index, segment in enumerate(segments):
        in_label = input_labels[index]
        out_label = f"tempo{index}"
        tempo_labels.append(out_label)

        chains.append(
            f"[{in_label}]"
            "asetpts=PTS-STARTPTS,"
            f"atempo={segment['tempo']:.12f},"
            f"apad=whole_dur={segment['reference_duration']:.9f},"
            f"atrim=end={segment['reference_duration']:.9f},"
            "asetpts=PTS-STARTPTS"
            f"[{out_label}]"
        )

    concat_inputs = "".join(
        f"[{label}]" for label in tempo_labels
    )

    chains.append(
        f"{concat_inputs}concat=n={len(segments)}:v=0:a=1[warped]"
    )

    # Время размещения определяется ТОЛЬКО синхронизационной картой.
    # Нельзя добавлять произвольные 1024 samples: для одно-сегментного
    # zero-map это создало бы универсальный ~21.333 ms сдвиг.
    placement_offset = float(start_offset)

    if placement_offset > 0:
        chains.append(
            f"[warped]adelay=delays={placement_offset * 1000.0:.3f}:all=1,"
            "asetpts=PTS-STARTPTS[placed]"
        )
    elif placement_offset < 0:
        chains.append(
            f"[warped]atrim=start={abs(placement_offset):.9f},"
            "asetpts=PTS-STARTPTS[placed]"
        )
    else:
        chains.append(
            "[warped]asetpts=PTS-STARTPTS[placed]"
        )

    if reference_duration > 0:
        # Последний trim выполняется ПОСЛЕ компенсации concat latency.
        # Поэтому итоговая дорожка имеет ту же duration timeline,
        # что и Reference: короткий результат дополняется тишиной,
        # длинный — обрезается.
        chains.append(
            f"[placed]apad=whole_dur={reference_duration:.9f},"
            f"atrim=end={reference_duration:.9f},"
            "asetpts=PTS-STARTPTS[outa]"
        )
    else:
        chains.append(
            "[placed]asetpts=PTS-STARTPTS[outa]"
        )

    return ";".join(chains).replace("{STREAM}", "STREAM")


def render_target_with_sync_map(
    selected: SelectedTracks,
    apply_map,
    output_dir: Path,
    progress: Progress,
    codec_plan: Optional[dict] = None,
):
    """
    Применяет нелинейную карту к полной исходной Target-дорожке.

    Декодирование -> один фильтрованный поток -> одно финальное кодирование.
    DTS-HD/MA и неподдерживаемые encoder'ом исходные кодеки автоматически
    переходят в PCM WAV/RF64 вместо unsafe lossy downgrade.
    """
    progress.stage("ПРИМЕНЕНИЕ ФИНАЛЬНОЙ КАРТЫ", 0.0)

    codec_plan = codec_plan or _get_render_codec_plan(selected.target)
    output_path = output_dir / _choose_render_output_path(selected.target, codec_plan).name

    if output_path.resolve() == selected.target.path.resolve():
        raise RuntimeError("Render output совпал с исходным TARGET.")

    if output_path.exists():
        try:
            output_path.unlink()
        except Exception as exc:
            raise RuntimeError(f"Не удалось удалить старый render output: {output_path}\n{exc}")

    progress.update(5.0, "Подготовка adaptive render segments")
    segments = build_render_segments(apply_map)
    progress.update(15.0, f"Render segments: {len(segments)}")

    encoder = str(codec_plan["encoder"])
    encoder_options = _build_encoder_options(
        selected.target,
        encoder,
        codec_plan=codec_plan,
        output_path=output_path,
    )

    filtergraph_template = _render_filtergraph(
        segments,
        float(apply_map["offset"][0]),
        float(selected.reference.duration),
        float(apply_map["reference_time"][-1]),
        int(selected.target.sample_rate),
    )
    filtergraph = filtergraph_template.replace(
        "[0:STREAM]",
        f"[0:{selected.target.stream_index}]",
    )

    filtergraph_path = output_dir / "render_filtergraph.txt"
    filtergraph_path.write_text(filtergraph, encoding="utf-8")

    save_render_plan(output_dir, selected, segments, output_path)
    progress.update(20.0, f"Encoder: {encoder}")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-stats_period", "1",
        "-progress", "pipe:2",
        "-i", str(selected.target.path),
        "-filter_complex_script", str(filtergraph_path),
        "-map", "[outa]",
        "-vn",
        "-map_metadata", "0",
    ]
    command.extend(encoder_options)

    if selected.reference.duration > 0:
        command += ["-t", f"{selected.reference.duration:.9f}"]

    command += ["-y", str(output_path)]

    started = time.time()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError:
        raise RuntimeError("Не найден ffmpeg для render.")

    last_message = ""
    last_out_time = 0.0
    stderr_lines = []

    while True:
        line = process.stderr.readline()
        if line == "":
            if process.poll() is not None:
                break
            time.sleep(0.05)
            continue

        stripped = line.strip()
        if not stripped:
            continue
        stderr_lines.append(stripped)
        if len(stderr_lines) > 40:
            stderr_lines.pop(0)

        if stripped.startswith("out_time="):
            value = stripped.split("=", 1)[1].strip()
            try:
                hh, mm, ss = value.split(":")
                last_out_time = int(hh) * 3600 + int(mm) * 60 + float(ss)
            except Exception:
                pass

            duration = max(0.001, float(selected.reference.duration))
            percent = min(99.0, max(20.0, 20.0 + 75.0 * last_out_time / duration))
            eta = progress.eta(last_out_time, duration)
            progress.update(
                percent,
                f"Render {format_seconds(last_out_time)} / {format_seconds(duration)} | ETA {format_seconds(eta)}",
            )
        elif not stripped.startswith((
            "frame=", "fps=", "stream_", "bitrate=", "total_size=",
            "dup_frames=", "drop_frames=", "speed=", "progress=",
        )):
            last_message = stripped

    return_code = process.wait()
    if return_code != 0:
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        tail = "\n".join(stderr_lines[-20:])
        raise RuntimeError(
            "ffmpeg завершился ошибкой при применении карты:\n"
            + (tail or last_message or "неизвестная ошибка")
        )

    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError("ffmpeg не создал корректный render output.")

    if encoder.lower().startswith('pcm_') and output_path.suffix.lower() in {'.wav', '.wave'}:
        from pcm_core.wav import finalize_wave
        written = probe_audio_file(output_path)
        if written is None or written.duration <= 0:
            raise RuntimeError('Не удалось проверить заголовок записанного WAV.')
        finalize_wave(output_path, round(written.duration*written.sample_rate),
                      written.sample_rate, written.channels)

    elapsed = time.time() - started
    progress.done(100.0, f"Render завершён: {output_path.name}")

    report_path = output_dir / RENDER_REPORT
    with report_path.open("w", encoding="utf-8") as file:
        file.write("FINAL AUDIO RENDER\n")
        file.write("=" * 78 + "\n\n")
        file.write(f"SOURCE TARGET:\n{selected.target.path}\n\n")
        file.write(f"OUTPUT:\n{output_path}\n\n")
        file.write(f"SOURCE CODEC: {selected.target.codec_name}\n")
        file.write(f"OUTPUT ENCODER: {encoder}\n")
        file.write(f"OUTPUT CONTAINER: {codec_plan.get('container')}\n")
        file.write(f"PCM LOSSLESS FALLBACK: {bool(codec_plan.get('pcm_fallback'))}\n")
        file.write(f"FORMAT DECISION: {codec_plan.get('reason')}\n")
        file.write(f"SEGMENTS: {len(segments)}\n")
        file.write(
            "CONCAT LATENCY COMPENSATION: 0.000000000 s (map-only placement)\n"
        )
        file.write(f"FILTERGRAPH: {filtergraph_path}\n")
        file.write(
            "MAX MAP APPROXIMATION ERROR: "
            f"{max((float(x['mapping_error']) for x in segments), default=0.0):.9f} s\n"
        )
        file.write(f"RENDER TIME: {elapsed:.3f} s\n")
        try:
            output_info_for_report=probe_audio_file(output_path)
            if output_info_for_report is not None:
                diag=_duration_diagnostics(output_info_for_report,float(selected.reference.duration))
                file.write(f"REFERENCE TIMELINE DURATION: {diag['reference_duration']:.6f} s\n")
                file.write(f"ENCODED OUTPUT DURATION: {diag['output_duration']:.6f} s\n")
                file.write(f"ENCODED DURATION DELTA: {diag['difference_sec']:+.6f} s\n")
                if diag['codec_frame_duration_sec'] is not None:
                    file.write(f"KNOWN CODEC FRAME DURATION: {diag['codec_frame_duration_sec']:.6f} s\n")
                    file.write(f"NEAREST CODEC FRAME MULTIPLE: {diag['nearest_frame_count']}\n")
                    file.write(f"FRAME-GRANULARITY RESIDUAL: {diag['frame_granularity_residual_sec']:+.6f} s\n")
        except Exception:
            pass
        file.write("\nThe original TARGET was not modified.\n")

    return {
        "output_path": output_path,
        "encoder": encoder,
        "segments": segments,
        "render_seconds": elapsed,
        "report_path": report_path,
        "codec_plan": codec_plan,
    }



def verify_rendered_format(
    output_path: Path,
    source_info: AudioInfo,
    codec_plan: Optional[dict] = None,
):
    info = probe_audio_file(output_path)
    if info is None:
        raise RuntimeError("Не удалось проверить созданный render через ffprobe.")

    codec_plan = codec_plan or _get_render_codec_plan(source_info)
    problems = []
    expected_encoder = str(codec_plan.get("encoder", ""))
    pcm_fallback = bool(codec_plan.get("pcm_fallback"))

    if pcm_fallback:
        if not info.codec_name.lower().startswith("pcm_"):
            problems.append(
                f"codec: expected PCM fallback, output={info.codec_name}"
            )
        if not info.format_name.lower().startswith("wav"):
            problems.append(
                f"container: expected WAV/RF64, output={info.format_name}"
            )
    elif expected_encoder and info.codec_name.lower() != source_info.codec_name.lower():
        # Для dts->dca ffprobe всё равно должен вернуть dts; для aliases this is expected.
        if not (
            source_info.codec_name.lower() == "dts"
            and expected_encoder == "dca"
            and info.codec_name.lower() == "dts"
        ):
            problems.append(
                f"codec: source={source_info.codec_name}, output={info.codec_name}"
            )

    if info.sample_rate != source_info.sample_rate:
        problems.append(
            f"sample_rate: source={source_info.sample_rate}, output={info.sample_rate}"
        )

    if info.channels != source_info.channels:
        problems.append(
            f"channels: source={source_info.channels}, output={info.channels}"
        )

    if (
        source_info.codec_name.lower().startswith("pcm_")
        and not pcm_fallback
        and source_info.sample_format
        and info.sample_format
        and info.sample_format != source_info.sample_format
    ):
        problems.append(
            f"sample_format: source={source_info.sample_format}, output={info.sample_format}"
        )

    if (
        not pcm_fallback
        and source_info.bit_rate > 0
        and info.bit_rate > 0
        and not source_info.codec_name.lower() in {
            "flac", "alac", "wavpack", "tta", "truehd", "mlp",
        }
        and not source_info.codec_name.lower().startswith("pcm_")
    ):
        if abs(info.bit_rate - source_info.bit_rate) > max(1000, source_info.bit_rate * 0.03):
            problems.append(
                f"bitrate: source={source_info.bit_rate}, output={info.bit_rate}"
            )

    if problems:
        raise RuntimeError(
            "Render изменил или нарушил исходные технические параметры:\n"
            + "\n".join(f"  - {item}" for item in problems)
        )

    return info



def choose_render_output_path(
    target: AudioInfo,
    codec_plan: Optional[dict] = None,
) -> Path:
    """Публичная обёртка над внутренним выбором пути результата."""
    return _choose_render_output_path(target, codec_plan=codec_plan)


def copy_target_without_reencode(
    target: AudioInfo,
    codec_plan: dict,
    output_dir: Path,
):
    """Точная копия уже синхронного TARGET без повторного кодирования."""
    # Even an undecodable-to-original-codec format can be copied exactly
    # when the map is zero. A fallback encoder is relevant only to rendering.
    codec_plan = {**codec_plan, "encoder": "copy", "pcm_fallback": False,
                  "container": target.format_name, "extension": target.path.suffix}
    output_path = output_dir / (target.path.stem + RENDER_OUTPUT_SUFFIX + target.path.suffix)
    import shutil
    started = time.time()
    shutil.copy2(target.path, output_path)
    elapsed = time.time() - started

    report_path = output_dir / RENDER_REPORT
    with report_path.open("w", encoding="utf-8") as file:
        file.write("FINAL AUDIO RENDER\n")
        file.write("=" * 78 + "\n\n")
        file.write(f"SOURCE TARGET:\n{target.path}\n\n")
        file.write(f"OUTPUT:\n{output_path}\n\n")
        file.write(f"SOURCE CODEC: {target.codec_name}\n")
        file.write("OUTPUT ENCODER: copy\n")
        file.write(f"OUTPUT CONTAINER: {codec_plan.get('container')}\n")
        file.write("PCM LOSSLESS FALLBACK: False\n")
        file.write("FORMAT DECISION: Exact copy; no re-encoding because the non-regression guard found no safe timing correction.\n")
        file.write("SEGMENTS: 0\n")
        file.write("CONCAT LATENCY COMPENSATION: 0.000000000 s (no render)\n")
        file.write("MAX MAP APPROXIMATION ERROR: 0.000000000 s\n")
        file.write(f"COPY TIME: {elapsed:.3f} s\n\n")
        file.write("The original TARGET was not modified.\n")

    return {
        "output_path": output_path,
        "encoder": "copy",
        "segments": [],
        "render_seconds": elapsed,
        "report_path": report_path,
        "codec_plan": codec_plan,
        "copied_without_reencode": True,
    }


def build_zero_sync_map(duration: float):
    import numpy as np
    return {
        "target_time": np.array([0.0, float(duration)], dtype=np.float64),
        "offset": np.array([0.0, 0.0], dtype=np.float64),
        "reference_time": np.array([0.0, float(duration)], dtype=np.float64),
        "confidence": np.array([1.0, 1.0], dtype=np.float64),
    }


def verify_rendered_sync(
    selected: SelectedTracks,
    analysis,
    output_path: Path,
    output_dir: Path,
    validation_interval_sec: float = 60.0,
    progress: Progress | None = None,
):
    """
    Финальная независимая проверка уже готового файла.

    Здесь проверяется не только синхронизация, но и критическая для WAV/RF64
    вещь: реальная длительность output должна соответствовать Reference timeline.
    """
    verify_info = probe_audio_file(output_path)
    if verify_info is None or verify_info.duration <= 0:
        raise RuntimeError("Не удалось получить duration rendered output.")

    expected_duration = float(selected.reference.duration)
    duration_difference = float(verify_info.duration - expected_duration)
    duration_ok = abs(duration_difference) <= 0.50

    if expected_duration > 0 and not duration_ok:
        report_path = output_dir / FINAL_RENDER_VALIDATION_REPORT
        with report_path.open("w", encoding="utf-8") as file:
            file.write("FINAL RENDERED AUDIO VALIDATION\n")
            file.write("=" * 78 + "\n\n")
            file.write(f"OUTPUT: {output_path}\n")
            file.write(f"EXPECTED REFERENCE DURATION: {expected_duration:.6f} s\n")
            file.write(f"OUTPUT DURATION: {verify_info.duration:.6f} s\n")
            file.write(f"DURATION DIFFERENCE: {duration_difference:+.6f} s\n")
            file.write("STATUS: FAILED — rendered duration is inconsistent with Reference.\n")
        raise RuntimeError(
            "Критическая ошибка render: длительность готового файла не совпадает "
            f"с Reference на {duration_difference:+.3f} s. "
            "Исходный TARGET не изменён. Подробности в FINAL_RENDER_VALIDATION_REPORT.txt."
        )

    temp_dir = output_dir / "_render_verify"
    temp_dir.mkdir(exist_ok=True)
    temp_pcm = temp_dir / "rendered_first_channel_s16le.pcm"

    progress = progress or Progress()
    decoded = decode_first_channel_stream(
        verify_info,
        temp_pcm,
        progress,
        0.0,
        50.0,
    )

    target_envelope = build_envelope_from_pcm(
        temp_pcm,
        progress,
        50.0,
        100.0,
    )

    verify_analysis = {
        "reference_envelope": analysis["reference_envelope"],
        "target_envelope": target_envelope,
    }

    duration = float(decoded["duration"])
    zero_map = build_zero_sync_map(duration)

    if duration >= 45.0:
        validation = validate_sync_map(
            verify_analysis,
            zero_map,
            duration,
            progress,
            validation_interval_sec=float(validation_interval_sec),
        )
    else:
        window_sec = max(2.0, min(20.0, duration * 0.40))
        radius_sec = max(0.20, min(3.0, duration * 0.10))
        center_points = [duration * 0.25, duration * 0.50, duration * 0.75]
        validation = []

        for target_time in center_points:
            after = validation_search_offset(
                verify_analysis["reference_envelope"]["values"],
                verify_analysis["target_envelope"]["values"],
                target_time,
                0.0,
                window_sec,
                radius_sec,
                VALIDATION_SEARCH_STEP_SEC,
                verify_analysis["target_envelope"]["hop_sec"],
            )
            actual = after["offset"]
            residual = None if actual is None else abs(float(actual))
            validation.append({
                "target_time": float(target_time),
                "predicted_offset": 0.0,
                "before_offset": actual,
                "before_correlation": float(after["correlation"]),
                "after_offset": actual,
                "after_correlation": float(after["correlation"]),
                "before_error": residual,
                "after_error": residual,
                "raw_after_offset": actual,
                "raw_after_correlation": float(after["correlation"]),
                "bidir_left_offset": None,
                "bidir_left_correlation": 0.0,
                "bidir_right_offset": None,
                "bidir_right_correlation": 0.0,
                "bidir_consensus_offset": actual,
                "bidir_consensus_correlation": float(after["correlation"]),
                "bidir_disagreement": None,
                "bidir_prediction_error": residual,
                "bidir_status": "SHORT_FILE_SINGLE_WINDOW",
            })

    metrics = calculate_validation_metrics(validation)
    hard = find_validation_outliers(validation)

    report_path = output_dir / FINAL_RENDER_VALIDATION_REPORT
    with report_path.open("w", encoding="utf-8") as file:
        file.write("FINAL RENDERED AUDIO VALIDATION\n")
        file.write("=" * 78 + "\n\n")
        file.write(f"OUTPUT: {output_path}\n\n")
        file.write(f"REFERENCE DURATION: {selected.reference.duration:.6f} s\n")
        file.write(f"TARGET ORIGINAL DURATION: {selected.target.duration:.6f} s\n")
        file.write(f"OUTPUT DURATION: {verify_info.duration:.6f} s\n")
        diag=_duration_diagnostics(verify_info,selected.reference.duration)
        file.write(f"OUTPUT - REFERENCE: {duration_difference:+.6f} s\n")
        file.write(f"DURATION CHECK: {'PASS' if duration_ok else 'FAIL'}\n")
        file.write("DURATION INTERPRETATION: encoded output duration is reported separately from synchronization residuals.\n")
        if diag['codec_frame_duration_sec'] is not None:
            file.write(f"KNOWN CODEC FRAME DURATION: {diag['codec_frame_duration_sec']:.6f} s\n")
            file.write(f"NEAREST CODEC FRAME MULTIPLE: {diag['nearest_frame_count']}\n")
            file.write(f"FRAME-GRANULARITY RESIDUAL: {diag['frame_granularity_residual_sec']:+.6f} s\n")
        file.write("\n")
        file.write(f"VALIDATION INTERVAL: {float(validation_interval_sec):.3f} s\n")
        file.write(f"POINTS: {metrics.get('points',0)}\n")
        file.write(f"AFTER mean residual: {metrics.get('after_mean_error')} s\n")
        file.write(f"AFTER median residual: {metrics.get('after_median_error')} s\n")
        file.write(f"AFTER max residual: {metrics.get('after_max_error')} s\n")
        file.write(f"AFTER mean correlation: {metrics.get('after_mean_correlation')}\n")
        file.write(f"AFTER within 50ms: {metrics.get('after_within_50ms_percent')} %\n")
        bidir=summarize_bidirectional_validation(validation)
        file.write(f"VALIDATION OUTLIER CANDIDATES: {len(hard)}\n")
        file.write(f"CANDIDATE HARD OUTLIERS: {bidir['hard_candidates']}\n")
        file.write(f"CONFIRMED DISCREPANCIES: {bidir['confirmed_discrepancy']}\n")
        file.write(f"AMBIGUOUS VALIDATION POINTS: {bidir['ambiguous']}\n")
        file.write(f"HARD CANDIDATES REQUIRING MANUAL REVIEW: {bidir['hard_confirmed_discrepancy']}\n")
        file.write(f"AMBIGUOUS HARD CANDIDATES: {bidir['hard_ambiguous']}\n\n")
        file.write("For the rendered file the expected synchronization offset is 0 s.\n")

    json_path = output_dir / FINAL_RENDER_VALIDATION_JSON
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "output_file": str(output_path),
                "reference_duration": float(selected.reference.duration),
                "target_original_duration": float(selected.target.duration),
                "output_duration": float(verify_info.duration),
                "duration_difference_from_reference": duration_difference,
                "duration_check": duration_ok,
                "duration_diagnostics": diag,
                "metrics": metrics,
                "validation_points": validation,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "metrics": metrics,
        "validation_points": validation,
        "outliers": hard,
        "output_info": verify_info,
        "duration_difference": duration_difference,
        "duration_ok": duration_ok,
        "duration_diagnostics": _duration_diagnostics(verify_info,selected.reference.duration),
        "rendered_first_channel_pcm": temp_pcm,
        "report_path": report_path,
        "json_path": json_path,
    }




# ============================================================
# РУССКИЕ КОПИИ TXT-ОТЧЁТОВ
# ============================================================

RUSSIAN_REPORT_SUFFIX = "_RU.txt"
