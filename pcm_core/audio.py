import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np


@dataclass
class AudioInfo:
    path: Path
    stream_index: int
    sample_rate: int
    channels: int
    channel_layout: str
    codec: str
    duration: float

    def to_dict(self):
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def find_tools():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "tools" / "ffmpeg" / "bin", root / "tools" / "ffmpeg"]
    for p in paths:
        if p.is_dir():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise RuntimeError("Не найдены " + ", ".join(missing) + ". См. README_RU.md, раздел установки FFmpeg.")
    return {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}


def run_ffmpeg(args):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError("FFmpeg: " + result.stderr.decode("utf-8", errors="replace")[-4000:])
    return result.stdout


def probe(path, stream_index=None):
    path = Path(path).expanduser().resolve(strict=True)
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError("Не удалось прочитать " + path.name + ": " + result.stderr[-2000:])
    data = json.loads(result.stdout)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if stream_index is not None:
        streams = [s for s in streams if s["index"] == stream_index]
    if not streams:
        raise ValueError("В файле не найден выбранный звуковой поток: " + path.name)
    if stream_index is None and len(streams) > 1:
        raise ValueError("В файле несколько звуковых потоков. Укажите индекс с помощью --reference-stream или --target-stream.")
    s = streams[0]
    try:
        duration = float(s.get("duration", data.get("format", {}).get("duration", 0)))
    except (ValueError, TypeError):
        duration = 0.
    rate = int(s.get("sample_rate", 0))
    channels = int(s.get("channels", 0))
    if rate <= 0 or not 1 <= channels <= 32:
        raise ValueError("Неподдерживаемая частота дискретизации или число каналов.")
    return AudioInfo(path, int(s["index"]), rate, channels, s.get("channel_layout", ""), s.get("codec_name", ""), duration)


def decode(info, path, sample_rate=None, first_channel=False, progress=None):
    args = ["-i", str(info.path), "-map", f"0:{info.stream_index}", "-vn", "-sn", "-dn"]
    if first_channel:
        args += ["-af", "pan=mono|c0=c0"]
    args += ["-ar", str(sample_rate or info.sample_rate), "-c:a", "pcm_f32le", "-f", "f32le", "-n", str(path)]
    if progress is None:
        run_ffmpeg(args)
    else:
        # Progress uses stdout, errors go to a file; neither pipe can fill
        # while Windows waits for the other stream to be consumed.
        error_path = path.with_suffix(path.suffix + '.log')
        with error_path.open('wb') as errors:
            process = subprocess.Popen(['ffmpeg', '-hide_banner', '-nostdin', '-v', 'error',
                                        '-progress', 'pipe:1', *args],
                                       stdout=subprocess.PIPE, stderr=errors, text=True)
            try:
                for line in process.stdout:
                    if line.startswith('out_time_us='):
                        try:
                            progress(float(line.partition('=')[2]) / 1e6, info.duration)
                        except ValueError:
                            pass
                if process.wait() != 0:
                    raise RuntimeError('Ошибка декодирования FFmpeg. См. ' + str(error_path))
            except BaseException:
                process.kill()
                process.wait()
                path.unlink(missing_ok=True)
                raise
            finally:
                process.stdout.close()
    channels = 1 if first_channel else info.channels
    if path.stat().st_size == 0 or path.stat().st_size % (4 * channels):
        raise RuntimeError("FFmpeg создал пустой или неполный PCM-файл.")
    return np.memmap(path, mode="r", dtype="<f4", shape=(path.stat().st_size // (4 * channels),)
                     if first_channel else (path.stat().st_size // (4 * channels), channels))


def check_disk(directory, required):
    available = shutil.disk_usage(directory).free
    if available < required:
        raise RuntimeError(f"Недостаточно места: нужно около {required / 2**30:.1f} ГиБ, доступно {available / 2**30:.1f} ГиБ.")


def describe(info):
    return f"{info.path.name} | {info.codec} | {info.sample_rate} Гц | {info.channels} кан. | {info.duration:.1f} с"
