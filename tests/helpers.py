from pathlib import Path
import subprocess
import numpy as np
from scipy.signal import butter, sosfiltfilt
from wavesync.dsp import sinc_sample


SR = 48000


def soundtrack(duration=24., seed=21):
    rng = np.random.default_rng(seed)
    t = np.arange(round(duration * SR)) / SR
    x = sosfiltfilt(butter(4, 6000, fs=SR, output="sos"), rng.normal(size=len(t)))
    env = np.full(len(t), .03)
    for center, width in zip(rng.uniform(0, duration, round(duration * 3.3)), rng.uniform(.025, .16, round(duration * 3.3))):
        env += .18 * np.exp(-((t - center) / width) ** 2)
    return (x * env).astype(np.float32)


def warp(reference, positions):
    out = np.empty(len(positions), np.float32)
    for start in range(0, len(out), 16384):
        out[start:start + 16384] = sinc_sample(reference, positions[start:start + 16384])
    return out


def write_audio(path, samples, rate=SR, layout=None, codec="pcm_f32le"):
    samples = np.asarray(samples, dtype="<f4")
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    layout_args = ["-channel_layout", layout] if layout else []
    result = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "f32le", "-ar", str(rate),
                             "-ac", str(channels), *layout_args, "-i", "pipe:0", "-c:a", codec, str(path)],
                            input=samples.tobytes(), capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())


def read_audio(path):
    from wavesync.audio import probe
    info = probe(path)
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    return info, np.frombuffer(result.stdout, dtype="<f4").reshape(-1, info.channels)
