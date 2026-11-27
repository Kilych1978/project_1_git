"""Continuous native-rate rendering of an already approved coarse time map.

All channels share one inverse clock and one bandlimited sampling kernel.
There is no WSOLA/atempo stage, map re-estimation, or lossy output encoding.
The established PCM renderer serves S16/S24/F32. Double-precision PCM keeps a
separate double-precision decode/sampling path so integer low bits and F64
values are not rounded through a float32 intermediate.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from pcm_core.audio import AudioInfo as CoreAudioInfo, check_disk
from pcm_core.config import Settings
from pcm_core.model import Piece, TimeMap
from pcm_core.render import render
from pcm_core.wav import finalize_wave


_PCM = {'pcm_s16le', 'pcm_s24le', 'pcm_s32le', 'pcm_f32le', 'pcm_f64le'}
_BYTES = {'pcm_s16le': 2, 'pcm_s24le': 3, 'pcm_s32le': 4,
          'pcm_f32le': 4, 'pcm_f64le': 8}


def _codec_plan(info, requested):
    source = str(info.codec_name or '').lower()
    # Endianness is a storage detail; output WAV uses little endian PCM.
    native = source[:-2] + 'le' if source.endswith('be') else source
    plan = copy.deepcopy(requested or {})
    if native in _PCM:
        encoder = native
    elif plan.get('encoder') in _PCM:
        encoder = plan['encoder']
    else:
        encoder = 'pcm_f32le'
    plan.update(source_codec=source, encoder=encoder, container='wav', extension='.wav',
                pcm_fallback=(native != encoder),
                reason='Continuous native-rate sampling; PCM output without a second lossy encode.')
    return plan


def _mapping(sync_map, settings):
    target = np.asarray(sync_map['target_time'], dtype=np.float64)
    reference = np.asarray(sync_map['reference_time'], dtype=np.float64)
    mapping = TimeMap([Piece(target, reference)])
    mapping.validate(settings.max_rate_change)
    if 'offset' in sync_map:
        offsets = np.asarray(sync_map['offset'], dtype=np.float64)
        if offsets.shape != target.shape or not np.isfinite(offsets).all() or not np.allclose(
                reference - target, offsets, atol=1e-8, rtol=0.):
            raise ValueError('Карта содержит несовместимые reference_time и offset.')
    return mapping


def _sample_f64(source, positions, radius=16, cutoff=1.):
    """Double Lanczos sampling; integer/unit-rate positions copy exactly.

    This deliberately does not call the existing float32 DSP API. Temporary
    storage depends on output block size and channels, never track duration.
    """
    positions = np.asarray(positions, dtype=np.float64)
    out = np.zeros((len(positions), source.shape[1]), dtype=np.float64)
    valid = np.isfinite(positions) & (positions >= 0) & (positions <= len(source) - 1)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return out
    p = positions[valid]
    c = np.broadcast_to(np.asarray(cutoff), positions.shape)[valid]
    integer = np.rint(p).astype(np.int64)
    direct = (np.abs(p - integer) < 1e-6) & (c >= 1 - 1e-12)
    out[indices[direct]] = source[integer[direct]]
    fractional = ~direct
    if fractional.any():
        use, p, c = indices[fractional], p[fractional], c[fractional]
        floor = np.floor(p).astype(np.int64)
        result = np.zeros((len(p), source.shape[1]), dtype=np.float64)
        weights = np.zeros(len(p), dtype=np.float64)
        for tap in range(-radius + 1, radius + 1):
            j = floor + tap
            dx = p - j
            w = c * np.sinc(c * dx) * np.sinc(dx / radius)
            inside = (j >= 0) & (j < len(source))
            result[inside] += source[j[inside]] * w[inside, None]
            weights += w
        out[use] = result / np.where(np.abs(weights) > 1e-12, weights, 1.)[:, None]
    return out


def _decode_f64(info, path, callback):
    errors_path = path.parent / 'ffmpeg_decode.log'
    command = ['ffmpeg', '-hide_banner', '-nostdin', '-v', 'error', '-progress', 'pipe:1',
               '-i', str(info.path), '-map', f'0:{info.stream_index}', '-vn', '-sn', '-dn',
               '-ar', str(info.sample_rate), '-c:a', 'pcm_f64le', '-f', 'f64le', '-n', str(path)]
    with errors_path.open('wb') as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            for line in process.stdout:
                if line.startswith('out_time_us=') and callback is not None:
                    value = line.partition('=')[2].strip()
                    if value.lstrip('-').isdigit():
                        callback(int(value) / 1e6, info.duration)
            if process.wait() != 0:
                raise RuntimeError('Ошибка декодирования F64. См. ' + str(errors_path))
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
    size = path.stat().st_size
    if size == 0 or size % (8 * info.channels):
        raise RuntimeError('Декодирование создало пустой или неполный F64 PCM.')
    return np.memmap(path, mode='r', dtype='<f8', shape=(size // (8 * info.channels), info.channels))


def _render_f64(info, mapping, duration, directory, settings, encoder, progress, decode_progress):
    frames = round(duration * info.sample_rate)
    # Conservative allowance for two native-double scratch buffers and output.
    required = (2 * info.duration * info.sample_rate * info.channels * 8
                + frames * info.channels * _BYTES[encoder])
    check_disk(directory, math.ceil(required * 1.1) + 64 * 2**20)
    raw = directory / 'target_multichannel.f64'
    destination = directory / 'CANDIDATE_TARGET.wav'
    source = process = None
    try:
        source = _decode_f64(info, raw, decode_progress)
        # The actual decoded frame count may differ from a container estimate.
        check_disk(directory, math.ceil(frames * info.channels * _BYTES[encoder] * 1.1) + 64 * 2**20)
        layout = ['-channel_layout', info.channel_layout] if info.channel_layout else []
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-v', 'error', '-n',
                   '-f', 'f64le', '-ar', str(info.sample_rate), '-ac', str(info.channels), *layout,
                   '-i', 'pipe:0', '-map', '0:a:0', '-c:a', encoder, '-rf64', 'auto', str(destination)]
        with (directory / 'ffmpeg_render.log').open('wb') as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                for begin in range(0, frames, 16384):
                    end = min(frames, begin + 16384)
                    times = np.arange(begin, end, dtype=np.float64) / info.sample_rate
                    positions, rate = mapping.inverse_with_rate(times)
                    cutoff = np.minimum(1., 1. / np.maximum(np.nan_to_num(rate, nan=1.), 1.))
                    values = _sample_f64(source, positions * info.sample_rate, settings.sinc_radius, cutoff)
                    process.stdin.write(values.astype('<f8', copy=False).tobytes())
                    if progress is not None:
                        progress(end / info.sample_rate, frames / info.sample_rate)
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError('Ошибка кодирования native PCM. См. ffmpeg_render.log.')
                finalize_wave(destination, frames, info.sample_rate, info.channels)
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
        raw.unlink(missing_ok=True)
    return destination


def render_native_coarse(selected, sync_map, output_dir, progress, codec_plan=None):
    """Render the current map at native precision, returning the legacy schema."""
    settings = Settings(max_rate_change=.25)
    mapping = _mapping(sync_map, settings)
    reference_duration = float(selected.reference.duration)
    if not math.isfinite(reference_duration) or reference_duration <= 0:
        raise ValueError('Длительность эталона должна быть конечной и положительной.')
    target = selected.target
    if target.sample_rate <= 0 or target.channels <= 0 or not math.isfinite(float(target.duration)) or target.duration <= 0:
        raise ValueError('Некорректные параметры исходного аудио.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (Path(target.path).stem + '_SYNCED.wav')
    if output_path.resolve() == Path(target.path).resolve():
        raise ValueError('Нельзя записывать результат поверх исходной дорожки.')
    if output_path.exists():
        raise FileExistsError('Результат уже существует: ' + str(output_path))
    source_stat = Path(target.path).stat()
    source_identity = source_stat.st_size, source_stat.st_mtime_ns
    plan = _codec_plan(target, codec_plan)
    encoder = plan['encoder']
    info = CoreAudioInfo(Path(target.path), int(target.stream_index), int(target.sample_rate),
                         int(target.channels), target.channel_layout, encoder, float(target.duration))
    work = Path(tempfile.mkdtemp(prefix='_native_render_', dir=output_dir))
    started = time.monotonic()
    if progress is not None:
        progress.stage('ПРИМЕНЕНИЕ ФИНАЛЬНОЙ КАРТЫ: НЕПРЕРЫВНЫЙ PCM', 0.)

    def decode_progress(done, total):
        if progress is not None:
            progress.update(min(20., 20. * done / max(total, 1e-9)),
                            'Декодирование исходных каналов', processed_sec=done, total_sec=total)

    def render_progress(done, total):
        if progress is not None:
            progress.update(min(99., 20. + 79. * done / max(total, 1e-9)),
                            'Непрерывное применение одной карты ко всем каналам', processed_sec=done, total_sec=total)

    candidate = None
    try:
        if encoder in {'pcm_s32le', 'pcm_f64le'}:
            candidate = _render_f64(info, mapping, reference_duration, work, settings, encoder,
                                    render_progress, decode_progress)
        else:
            candidate = render(info, mapping, reference_duration, work, settings,
                               log=lambda *_: None, progress=render_progress, decode_progress=decode_progress)
        current = Path(target.path).stat()
        if source_identity != (current.st_size, current.st_mtime_ns):
            raise RuntimeError('Исходная дорожка изменилась во время рендера; результат не опубликован.')
        if os.name == 'nt':
            # Windows rename is atomic and refuses an existing destination;
            # unlike hard links it also works on FAT/exFAT output volumes.
            candidate.rename(output_path)
        else:
            # Same-volume link publishes only the complete finalized WAV and
            # refuses an existing destination on POSIX as well.
            output_path.hardlink_to(candidate)
            candidate.unlink()
    except BaseException:
        if candidate is not None:
            candidate.unlink(missing_ok=True)
        raise
    elapsed = time.monotonic() - started
    pieces = mapping.pieces[0]
    segments = [dict(target_start=float(a), target_end=float(b), reference_start=float(c),
                     reference_end=float(d), duration=float(b-a), reference_duration=float(d-c),
                     tempo=float((b-a)/(d-c)), mapping_error=0.)
                for a, b, c, d in zip(pieces.target[:-1], pieces.target[1:], pieces.reference[:-1], pieces.reference[1:])]
    report_path = output_dir / 'NATIVE_COARSE_RENDER_REPORT.json'
    report_path.write_text(json.dumps({
        'renderer': 'continuous_native_sampling', 'source': str(target.path), 'output': str(output_path),
        'output_encoder': encoder, 'sample_rate': info.sample_rate, 'channels': info.channels,
        'sample_frames': round(reference_duration * info.sample_rate), 'max_mapping_error_sec': 0.,
        'sampling_precision': 'float64' if encoder in {'pcm_s32le', 'pcm_f64le'} else 'float32',
        'render_seconds': elapsed, 'codec_plan': plan, 'map': mapping.to_dict(),
        'source_unchanged': True, 'scratch_directory': str(work),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    if progress is not None:
        progress.done(100., 'Непрерывный PCM-рендер завершён')
    return dict(output_path=output_path, encoder=encoder, segments=segments,
                render_seconds=elapsed, report_path=report_path, codec_plan=plan)
