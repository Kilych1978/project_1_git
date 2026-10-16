"""One multichannel lossless render; partial files are never labeled SYNCED."""
import subprocess
import time
import numpy as np
from .audio import check_disk, decode
from .dsp import sinc_sample


def render(info, mapping, reference_duration, directory, settings, log=print, progress=None, decode_progress=None):
    mapping.validate(settings.max_rate_change)
    raw_path = directory / "target_multichannel.f32"
    destination = directory / "CANDIDATE_TARGET.wav"
    sr = info.sample_rate
    channels = info.channels
    # Float WAV avoids clipping/bit-depth reduction of decoded or interpolated PCM.
    # It is lossless with respect to the float32 pipeline, not a copy of codec bits.
    frames = round(reference_duration * sr)
    check_disk(directory, int((info.duration * sr * channels + frames * channels) * 4 * 1.1) + 64 * 2**20)
    source = None
    process = None
    started = time.monotonic()
    last_log = 0.
    try:
        log("  Декодирование всех каналов исходной дорожки.")
        source = decode(info, raw_path, progress=decode_progress)
        layout = ["-channel_layout", info.channel_layout] if info.channel_layout else []
        command = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-n",
                   "-f", "f32le", "-ar", str(sr), "-ac", str(channels), *layout,
                   "-i", "pipe:0", "-map", "0:a:0", "-c:a", "pcm_f32le", "-rf64", "auto", str(destination)]
        # A file for stderr avoids a deadlock if FFmpeg reports more than a pipe buffer.
        with (directory / "ffmpeg_render.log").open("wb") as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                for begin in range(0, frames, 16384):
                    end = min(frames, begin + 16384)
                    ref_times = np.arange(begin, end, dtype=np.float64) / sr
                    source_times, delta = mapping.inverse_with_rate(ref_times)
                    positions = source_times * sr
                    # A discontinuity must not become an anti-alias rate estimate.
                    cutoff = np.minimum(1., 1 / np.maximum(np.nan_to_num(delta, nan=1.), 1.))
                    values = sinc_sample(source, positions, settings.sinc_radius, cutoff)
                    process.stdin.write(values.astype("<f4", copy=False).tobytes())
                    now = time.monotonic()
                    if begin == 0 or now - last_log >= 5 or end == frames:
                        if progress:
                            progress(end / sr, frames / sr)
                        else:
                            log(f"  Рендер всех каналов: {end / sr:.1f}/{frames / sr:.1f} с; прошло {now - started:.0f} с.")
                        last_log = now
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("Ошибка кодирования FFmpeg. См. ffmpeg_render.log.")
            except BaseException:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if source is not None:
            source._mmap.close()
        raw_path.unlink(missing_ok=True)
    return destination
