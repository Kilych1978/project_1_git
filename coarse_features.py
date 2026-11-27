"""Streaming, phase-safe spectral features for coarse audio correspondence.

This optional analysis never edits a source or the existing first-channel PCM.
Only base band RMS is stored; a matcher derives log/whitened features locally.
Frame i is centred at ``origin_sec + i * hop_sec``; there is no end padding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import rfft
from scipy.signal import resample_poly

if TYPE_CHECKING:
    from common import AudioInfo

SAMPLE_RATE = 12000
FFT_SIZE = 512
HOP_SAMPLES = 120
_PCM_CHUNK_SAMPLES = 12000
_FFT_BATCH_FRAMES = 64
_QUEUE_BLOCKS = 2
_DECODE_STALL_SECONDS = 90.0
_BAND_EDGES = np.geomspace(40.0, 5800.0, 29)
_FREQUENCIES = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
_BAND_BINS = [np.flatnonzero((_FREQUENCIES >= a) & (_FREQUENCIES < b))
              for a, b in zip(_BAND_EDGES[:-1], _BAND_EDGES[1:])]
_BAND_BINS = [indices for indices in _BAND_BINS if len(indices)]
_WINDOW = np.hanning(FFT_SIZE + 1)[:-1].astype(np.float32)
_WINDOW_SUM = float(np.sum(_WINDOW))
_LAYOUTS = {
    'mono': ['FC'], 'stereo': ['FL', 'FR'], '2.1': ['FL', 'FR', 'LFE'],
    '3.0': ['FL', 'FR', 'FC'], '3.0(back)': ['FL', 'FR', 'BC'],
    '4.0': ['FL', 'FR', 'FC', 'BC'], 'quad': ['FL', 'FR', 'BL', 'BR'],
    'quad(side)': ['FL', 'FR', 'SL', 'SR'],
    '5.0': ['FL', 'FR', 'FC', 'BL', 'BR'],
    '5.0(side)': ['FL', 'FR', 'FC', 'SL', 'SR'],
    '5.1': ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR'],
    '5.1(side)': ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR'],
    '6.1': ['FL', 'FR', 'FC', 'LFE', 'BC', 'SL', 'SR'],
    '7.1': ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'SL', 'SR'],
    '7.1(wide)': ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'FLC', 'FRC'],
}


@dataclass
class FeatureBank:
    values: np.ndarray
    energy: np.ndarray
    hop_sec: float = HOP_SAMPLES / SAMPLE_RATE
    origin_sec: float = FFT_SIZE / (2 * SAMPLE_RATE)
    duration: float = 0.0
    views: list[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    storage_path: Path | None = None
    metadata_path: Path | None = None

    def __post_init__(self):
        if self.values.ndim != 3 or self.energy.shape != self.values.shape[:2]:
            raise ValueError('FeatureBank expects values[T,V,B] and energy[T,V].')
        if self.views and len(self.views) != self.values.shape[1]:
            raise ValueError('The number of view labels does not match the feature bank.')
        if not math.isfinite(self.hop_sec) or self.hop_sec <= 0:
            raise ValueError('Feature hop must be positive and finite.')
        if not math.isfinite(self.origin_sec) or not math.isfinite(self.duration) or self.duration < 0:
            raise ValueError('Feature timing must be finite and duration nonnegative.')

    @property
    def frame_times(self):
        return self.origin_sec + np.arange(len(self.values), dtype=np.float64) * self.hop_sec

    def close(self):
        """Release file handles, including on Windows; closing twice is safe."""
        shape = self.values.shape
        for array in (self.values, self.energy):
            mmap = getattr(array, '_mmap', None)
            if mmap is not None:
                mmap.close()
        self.values = np.empty((0, shape[1], shape[2]), dtype=np.float32)
        self.energy = np.empty((0, shape[1]), dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @classmethod
    def load(cls, metadata_path):
        path = Path(metadata_path)
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema') != 1 or data.get('dtype') != '<f4':
            raise ValueError('Unsupported spectral feature bank schema.')
        dimensions = data['shape']
        if not isinstance(dimensions, list) or len(dimensions) != 3 or any(type(n) is not int for n in dimensions):
            raise ValueError('Invalid spectral feature bank dimensions.')
        shape = tuple(dimensions)
        if any(n < 0 for n in shape) or shape[1] < 1 or shape[2] < 1:
            raise ValueError('Invalid spectral feature bank dimensions.')
        paths = [path.parent / name for name in ('band_rms.f32', 'energy.f32')]
        expected = [math.prod(shape) * 4, math.prod(shape[:2]) * 4]
        if any(p.stat().st_size != n for p, n in zip(paths, expected)):
            raise ValueError('Truncated or inconsistent spectral feature bank.')
        values = np.memmap(paths[0], mode='r', dtype='<f4', shape=shape) if shape[0] else np.empty(shape, np.float32)
        energy = np.memmap(paths[1], mode='r', dtype='<f4', shape=shape[:2]) if shape[0] else np.empty(shape[:2], np.float32)
        try:
            return cls(values, energy, float(data['hop_sec']), float(data['origin_sec']),
                       float(data['duration']), list(data['views']), data['parameters'], paths[0], path)
        except BaseException:
            for array in (values, energy):
                mmap = getattr(array, '_mmap', None)
                if mmap is not None:
                    mmap.close()
            raise


def _view_definitions(channels, channel_layout):
    if isinstance(channel_layout, (list, tuple)):
        physical = [str(label) for label in channel_layout]
        if len(physical) != channels or len(set(physical)) != channels or not all(physical):
            raise ValueError('Explicit channel labels must be unique and match the samples.')
    else:
        physical = _LAYOUTS.get(str(channel_layout or '').strip().lower())
        if physical is None or len(physical) != channels:
            physical = [f'c{i}' for i in range(channels)]
        else:
            physical = list(physical)
    views = list(physical)
    stereo = None
    if 'FL' in physical and 'FR' in physical:
        stereo = physical.index('FL'), physical.index('FR')
        views.extend(['M', 'S'])
    views.append('energy_union')
    return views, physical, stereo


class _Writer:
    def __init__(self, directory):
        self.directory = directory
        self.count = 0
        self.chunks = []
        self.files = []
        if directory is not None:
            try:
                for name in ('band_rms.f32', 'energy.f32'):
                    self.files.append((directory / name).open('wb'))
            except BaseException:
                self.close()
                raise

    def append(self, values, energy):
        values = np.asarray(values, dtype=np.float32)
        energy = np.asarray(energy, dtype=np.float32)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(energy)):
            raise ValueError('Spectral analysis received nonfinite or overflowing PCM.')
        if self.files:
            for handle, array in zip(self.files, (values, energy)):
                handle.write(np.asarray(array, dtype='<f4').tobytes(order='C'))
        else:
            self.chunks.append((values.copy(), energy.copy()))
        self.count += len(values)

    def close(self):
        for handle in self.files:
            handle.close()

    def finish(self, views, parameters, samples):
        self.close()
        shape = (self.count, len(views), len(_BAND_BINS))
        duration = samples / SAMPLE_RATE
        if self.directory is None:
            values = np.concatenate([a for a, _ in self.chunks]) if self.chunks else np.empty(shape, np.float32)
            energy = np.concatenate([a for _, a in self.chunks]) if self.chunks else np.empty(shape[:2], np.float32)
            return FeatureBank(values, energy, duration=duration, views=views, parameters=parameters)
        metadata = {'schema': 1, 'dtype': '<f4', 'shape': shape, 'duration': duration,
                    'hop_sec': HOP_SAMPLES / SAMPLE_RATE, 'origin_sec': FFT_SIZE / (2 * SAMPLE_RATE),
                    'views': views, 'parameters': parameters}
        temporary = self.directory / 'FEATURES.json.tmp'
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        path = self.directory / 'FEATURES.json'
        temporary.replace(path)
        return FeatureBank.load(path)


class _Accumulator:
    def __init__(self, channels, layout, writer):
        self.channels = channels
        self.views, physical, self.stereo = _view_definitions(channels, layout)
        self.writer = writer
        self.pending = np.empty((0, channels), np.float32)
        self.samples = 0
        self.peak_buffer = 0
        self.peak_batch = 0
        self.parameters = {
            'sample_rate': SAMPLE_RATE, 'fft_size': FFT_SIZE, 'hop_samples': HOP_SAMPLES,
            'window': 'periodic_hann', 'padding': 'none', 'physical_channels': physical,
            'band_edges_hz': _BAND_EDGES.tolist(),
            'band_centers_hz': [float(np.mean(_FREQUENCIES[i])) for i in _BAND_BINS],
            'band_bins': [indices.tolist() for indices in _BAND_BINS],
            'values_units': 'RMS of STFT magnitude per frequency band; FFT scaled by Hann sum',
            'energy_units': 'unweighted time-domain window RMS, normalized physical PCM amplitude',
            'view_definitions': {name: 'physical channel' for name in physical},
        }
        if self.stereo is not None:
            self.parameters['view_definitions'].update(M='(FL+FR)/2', S='(FL-FR)/2')
        self.parameters['view_definitions']['energy_union'] = 'sqrt(mean physical-channel power); not an amplitude downmix'

    def feed(self, samples):
        # Bound temporary FFT/window storage even if an upstream reader supplies
        # a huge block. Only <FFT_SIZE trailing samples survive each iteration.
        for start in range(0, len(samples), _PCM_CHUNK_SAMPLES):
            block = np.asarray(samples[start:start + _PCM_CHUNK_SAMPLES], dtype=np.float32)
            if block.ndim != 2 or block.shape[1] != self.channels or not np.all(np.isfinite(block)):
                raise ValueError('Invalid interleaved PCM block.')
            self.samples += len(block)
            buffer = np.concatenate([self.pending, block])
            self.peak_buffer = max(self.peak_buffer, len(buffer))
            count = max(0, 1 + (len(buffer) - FFT_SIZE) // HOP_SAMPLES)
            if count:
                windows = np.lib.stride_tricks.sliding_window_view(buffer, FFT_SIZE, axis=0)[::HOP_SAMPLES]
                windows = windows.transpose(0, 2, 1)
                for index in range(0, count, _FFT_BATCH_FRAMES):
                    self._frames(windows[index:index + _FFT_BATCH_FRAMES])
            self.pending = buffer[count * HOP_SAMPLES:].copy()

    def _frames(self, windows):
        self.peak_batch = max(self.peak_batch, len(windows))
        spectra = rfft(windows * _WINDOW[None, :, None], axis=1) / _WINDOW_SUM
        energy = np.sqrt(np.mean(windows * windows, axis=1, dtype=np.float64)).astype(np.float32)
        if self.stereo is not None:
            left, right = self.stereo
            spectra = np.concatenate([spectra, ((spectra[:, :, left] + spectra[:, :, right]) / 2)[:, :, None],
                                      ((spectra[:, :, left] - spectra[:, :, right]) / 2)[:, :, None]], axis=2)
            mid = (windows[:, :, left] + windows[:, :, right]) / 2
            side = (windows[:, :, left] - windows[:, :, right]) / 2
            extra = np.stack([np.sqrt(np.mean(mid * mid, axis=1, dtype=np.float64)),
                              np.sqrt(np.mean(side * side, axis=1, dtype=np.float64))], axis=1)
            energy = np.concatenate([energy, extra], axis=1)
        power = np.abs(spectra) ** 2
        bands = np.stack([np.sqrt(np.mean(power[:, indices, :], axis=1)) for indices in _BAND_BINS], axis=2)
        union = np.sqrt(np.mean(bands[:, :self.channels, :] ** 2, axis=1))
        union_energy = np.sqrt(np.mean(energy[:, :self.channels] ** 2, axis=1))
        self.writer.append(np.concatenate([bands, union[:, None, :]], axis=1),
                           np.concatenate([energy, union_energy[:, None]], axis=1))

    def finish(self, extra_parameters=None):
        self.parameters.update(extra_parameters or {})
        self.parameters.update(decoded_samples=self.samples, frame_count=self.writer.count,
                               trailing_samples=len(self.pending), peak_buffer_samples=self.peak_buffer,
                               peak_fft_batch_frames=self.peak_batch, queue_blocks=_QUEUE_BLOCKS)
        return self.writer.finish(self.views, self.parameters, self.samples)


def _new_directory(directory):
    if directory is None:
        return None
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='coarse-features-', dir=path))


def _normalized_samples(samples):
    source = np.asarray(samples)
    if source.ndim == 1:
        source = source[:, None]
    if source.ndim != 2 or source.shape[1] < 1:
        raise ValueError('PCM samples must have shape [samples] or [samples, channels].')
    if source.dtype.kind in 'iu':
        info = np.iinfo(source.dtype)
        if source.dtype.kind == 'u':
            midpoint = (info.max + 1) / 2
            return (source.astype(np.float32) - midpoint) / midpoint
        return source.astype(np.float32) / max(abs(info.min), info.max)
    if source.dtype.kind != 'f':
        raise ValueError('PCM samples must have an integer or real floating dtype.')
    return np.asarray(source, dtype=np.float32)


def build_features_array(samples, sample_rate, channel_layout=None, directory=None):
    """Build a bank from normalized float PCM or integer PCM without mutating it.

    Supplying ``directory`` writes a disk-backed bank in a unique child folder.
    Arrays use scipy's rational resampling; file input streams through FFmpeg.
    """
    if not math.isfinite(float(sample_rate)) or int(sample_rate) != sample_rate or sample_rate <= 0:
        raise ValueError('PCM sample rate must be a positive integer.')
    sample_rate = int(sample_rate)
    pcm = _normalized_samples(samples)
    original_count = len(pcm)
    if sample_rate != SAMPLE_RATE and len(pcm):
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        pcm = resample_poly(pcm, SAMPLE_RATE // divisor, sample_rate // divisor, axis=0)
    _view_definitions(pcm.shape[1], channel_layout)
    folder = _new_directory(directory)
    writer = None
    try:
        writer = _Writer(folder)
        accumulator = _Accumulator(pcm.shape[1], channel_layout, writer)
        accumulator.feed(pcm)
        return accumulator.finish({'source_sample_rate': sample_rate, 'source_samples': original_count})
    except BaseException:
        if writer is not None:
            writer.close()
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
        raise


def _notify(progress, done_seconds, total_seconds, complete=False):
    if progress is None:
        return
    if callable(progress):
        progress(done_seconds, total_seconds)
    else:
        percent = 100.0 if complete else min(99.9, 100 * done_seconds / total_seconds) if total_seconds > 0 else 0.0
        progress.update(percent, 'Многополосный анализ звука', done_seconds,
                        total_seconds if total_seconds > 0 else None)


def build_features_file(info: 'AudioInfo', directory, progress=None):
    """Decode the selected stream once to 12kHz f32 PCM with bounded memory.

    ``progress`` is either callback(done_seconds, total_seconds), with total=0
    if unavailable, or the project's Progress object. Duration is always the
    actual decoded sample count, never a DTS container duration estimate.
    """
    channels = int(info.channels)
    if channels < 1:
        raise ValueError('The selected audio stream has no channels.')
    _view_definitions(channels, info.channel_layout)
    path = Path(info.path)
    before = path.stat()
    folder = _new_directory(directory)
    if folder is None:
        raise ValueError('File feature extraction requires a destination directory.')
    total = max(0.0, float(info.duration or 0.0))
    if not math.isfinite(total):
        total = 0.0
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(path),
               '-map', f'0:{int(info.stream_index)}', '-vn', '-sn', '-dn', '-ar', str(SAMPLE_RATE),
               '-acodec', 'pcm_f32le', '-f', 'f32le', 'pipe:1']
    process = None
    reader = None
    bank = None
    writer = None
    log = None
    stop = threading.Event()
    blocks = queue.Queue(maxsize=_QUEUE_BLOCKS)

    def put(value):
        while not stop.is_set():
            try:
                blocks.put(value, timeout=.2)
                return
            except queue.Full:
                continue

    def read_stdout():
        try:
            while not stop.is_set():
                block = process.stdout.read(_PCM_CHUNK_SAMPLES * channels * 4)
                if not block:
                    break
                put(block)
        except BaseException as error:
            put(error)
        finally:
            put(None)

    try:
        writer = _Writer(folder)
        accumulator = _Accumulator(channels, info.channel_layout, writer)
        log = (folder / 'decode.log').open('wb')
        _notify(progress, 0.0, total)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, bufsize=0)
        reader = threading.Thread(target=read_stdout, daemon=True, name='coarse-feature-pcm-reader')
        reader.start()
        last_data = time.monotonic()
        remainder = b''
        bytes_per_frame = channels * 4
        while True:
            try:
                block = blocks.get(timeout=.5)
            except queue.Empty:
                if time.monotonic() - last_data > _DECODE_STALL_SECONDS:
                    raise RuntimeError('FFmpeg не выдаёт PCM при многополосном анализе звука.')
                continue
            if block is None:
                break
            if isinstance(block, BaseException):
                raise block
            last_data = time.monotonic()
            raw = remainder + block
            length = len(raw) - len(raw) % bytes_per_frame
            if length:
                accumulator.feed(np.frombuffer(raw[:length], dtype='<f4').reshape(-1, channels))
            remainder = raw[length:]
            _notify(progress, accumulator.samples / SAMPLE_RATE, total)
        result = process.wait(timeout=10)
        if result != 0:
            log.flush()
            with (folder / 'decode.log').open('rb') as error_file:
                error_file.seek(max(0, (folder / 'decode.log').stat().st_size - 8000))
                diagnostic = error_file.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'FFmpeg завершил многополосный анализ с кодом {result}: {diagnostic}')
        if remainder:
            raise RuntimeError('FFmpeg вернул неполный interleaved PCM frame.')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Исходный аудиофайл изменился во время многополосного анализа.')
        bank = accumulator.finish({'source_sample_rate': int(info.sample_rate),
                                   'source_identity': {'path': str(path.resolve()), 'size_bytes': before.st_size,
                                                       'mtime_ns': before.st_mtime_ns},
                                   'source_stream_index': int(info.stream_index)})
        _notify(progress, bank.duration, bank.duration, complete=True)
        return bank
    except BaseException:
        if writer is not None:
            writer.close()
        if bank is not None:
            bank.close()
        raise
    finally:
        stop.set()
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            if process.stdout is not None:
                process.stdout.close()
        if reader is not None:
            reader.join(timeout=2)
        if log is not None:
            log.close()
        # Metadata is published only once the entire decode succeeds.
        if not (folder / 'FEATURES.json').exists():
            shutil.rmtree(folder, ignore_errors=True)
