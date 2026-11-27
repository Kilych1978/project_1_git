"""Real multichannel native renders, PCM precision, and immutable sources."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from coarse_native_render import render_native_coarse
from scanner import probe_audio_file


def _write(path, values, codec, rate=48000, layout=None):
    formats = {'pcm_s16le': ('s16le', '<i2'), 'pcm_s32le': ('s32le', '<i4'),
               'pcm_f64le': ('f64le', '<f8'), 'pcm_f32le': ('f32le', '<f4')}
    fmt, dtype = formats[codec]
    values = np.asarray(values, dtype=dtype)
    channels = 1 if values.ndim == 1 else values.shape[1]
    args = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', fmt, '-ar', str(rate), '-ac', str(channels)]
    if layout:
        args += ['-channel_layout', layout]
    args += ['-i', 'pipe:0', '-c:a', codec, str(path)]
    result = subprocess.run(args, input=values.tobytes(), capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))


def _read(path, fmt, codec, dtype, channels):
    result = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:a:0',
                             '-c:a', codec, '-f', fmt, 'pipe:1'], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return np.frombuffer(result.stdout, dtype=dtype).reshape(-1, channels)


def _clock(duration, offset=0., rate=1.):
    target = np.array([0., duration])
    reference = rate * target + offset
    return {'target_time': target, 'reference_time': reference, 'offset': reference - target}


def _identity(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()


class CoarseNativeRenderTests(unittest.TestCase):
    def test_affine_map_samples_all_six_channels_at_native_s16_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sr, duration, reference_duration = 48000, 1.2, 1.25
            rate, offset = .999, .031
            t = np.arange(round(duration * sr)) / sr
            frequencies = np.array([211, 477, 731, 1043, 1297, 1601])
            amplitude = .12 + np.arange(6) * .01
            phase = np.arange(6) * np.pi / 7
            signal = amplitude * np.sin(2*np.pi*t[:, None]*frequencies + phase)
            values = np.rint(signal * 32768).astype('<i2')
            source_path = root/'any_input.wav'
            _write(source_path, values, 'pcm_s16le', sr, '5.1')
            original = _identity(source_path)
            selected = SimpleNamespace(target=probe_audio_file(source_path),
                                       reference=SimpleNamespace(duration=reference_duration))
            result = render_native_coarse(selected, _clock(duration, offset, rate), root/'output', None)
            output = probe_audio_file(result['output_path'])
            self.assertEqual(output.codec_name, 'pcm_s16le')
            self.assertEqual(output.sample_rate, sr)
            self.assertEqual(output.channels, 6)
            self.assertEqual(output.channel_layout, '5.1')
            samples = _read(result['output_path'], 's16le', 'pcm_s16le', '<i2', 6)
            self.assertEqual(len(samples), round(reference_duration * sr))
            reference_time = np.arange(len(samples)) / sr
            source_time = (reference_time - offset) / rate
            interior = (source_time > .01) & (source_time < duration - .01)
            expected = amplitude * np.sin(2*np.pi*source_time[interior, None]*frequencies + phase)
            # Includes source S16 quantization, finite Lanczos support, and
            # output S16 rounding; the independent analytic clock must agree
            # within three S16 steps in every channel.
            np.testing.assert_allclose(samples[interior].astype(float)/32768., expected, atol=3/32768., rtol=0.)
            np.testing.assert_array_equal(samples[reference_time < offset-1/sr], 0)
            np.testing.assert_array_equal(samples[reference_time > offset+rate*duration], 0)
            self.assertEqual(_identity(source_path), original)
            self.assertTrue(all(segment['mapping_error'] == 0. for segment in result['segments']))
            report = json.loads(result['report_path'].read_text())
            self.assertEqual(report['sample_frames'], round(reference_duration * sr))
            self.assertFalse(list((root/'output').rglob('target_multichannel.f32')))

    def test_integer_maps_preserve_s32_low_bits_and_non_f32_double_values(self):
        rng = np.random.default_rng(986)
        sr, frames, channels, shift = 44100, 4097, 2, 13
        integers = rng.integers(-2**30, 2**30, size=(frames, channels), dtype=np.int32)
        doubles = rng.uniform(-1.7, 1.7, (frames, channels)) + 2.**-43
        self.assertTrue(np.any(doubles != doubles.astype(np.float32).astype(np.float64)))
        for codec, samples, fmt, dtype in [('pcm_s32le', integers, 's32le', '<i4'),
                                           ('pcm_f64le', doubles, 'f64le', '<f8')]:
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root/'source.wav'
                _write(source, samples, codec, sr, 'stereo')
                original = _identity(source)
                info = probe_audio_file(source)
                selected = SimpleNamespace(target=info, reference=SimpleNamespace(duration=frames/sr))
                for delay in (0, shift):
                    with self.subTest(delay=delay):
                        output_dir = root/str(delay)
                        result = render_native_coarse(selected, _clock(frames/sr, delay/sr), output_dir, None)
                        output = _read(result['output_path'], fmt, codec, dtype, channels)
                        expected = samples if delay == 0 else np.vstack([np.zeros((delay, channels), dtype=dtype), samples[:-delay]])
                        np.testing.assert_array_equal(output, expected)
                        actual = probe_audio_file(result['output_path'])
                        self.assertEqual(actual.codec_name, codec)
                        self.assertEqual(actual.sample_rate, sr)
                        self.assertEqual(actual.channels, channels)
                        self.assertEqual(len(output), frames)
                        self.assertFalse(list(output_dir.rglob('target_multichannel.f64')))
                self.assertEqual(_identity(source), original)

    def test_lossy_input_gets_one_float_pcm_output_without_second_lossy_encode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sr = 48000
            t = np.arange(sr) / sr
            wav = root/'source.wav'
            _write(wav, .1*np.sin(2*np.pi*420*t), 'pcm_f32le', sr)
            ac3 = root/'compressed.ac3'
            process = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(wav),
                                      '-c:a', 'ac3', str(ac3)], capture_output=True)
            self.assertEqual(process.returncode, 0, process.stderr.decode(errors='replace'))
            original = _identity(ac3)
            info = probe_audio_file(ac3)
            decoded = _read(ac3, 'f32le', 'pcm_f32le', '<f4', info.channels)
            duration = len(decoded)/sr
            selected = SimpleNamespace(target=replace(info, duration=duration), reference=SimpleNamespace(duration=duration))
            result = render_native_coarse(selected, _clock(duration), root/'output', None,
                                          codec_plan={'encoder': 'ac3', 'container': 'ac3'})
            output = _read(result['output_path'], 'f32le', 'pcm_f32le', '<f4', info.channels)
            self.assertEqual(result['encoder'], 'pcm_f32le')
            self.assertEqual(result['codec_plan']['container'], 'wav')
            np.testing.assert_array_equal(output, decoded)
            self.assertEqual(_identity(ac3), original)

    def test_invalid_maps_and_existing_destination_never_replace_source_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source.wav'
            _write(source, np.zeros((4800, 2), dtype='<i2'), 'pcm_s16le')
            selected = SimpleNamespace(target=probe_audio_file(source), reference=SimpleNamespace(duration=.1))
            original = _identity(source)
            for mapping in [{'target_time': [0., .1], 'reference_time': [.1, 0.]},
                            _clock(.1, rate=.5),
                            {'target_time': [0., 0.], 'reference_time': [0., .1]},
                            {'target_time': [0., .1], 'reference_time': [0., .1], 'offset': [0., .01]}]:
                with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                    render_native_coarse(selected, mapping, root/'output', None)
            self.assertFalse((root/'output').exists())
            output_dir = root/'output'
            output_dir.mkdir()
            existing = output_dir/'source_SYNCED.wav'
            existing.write_bytes(b'Previously completed result, never replaced.')
            marker = existing.read_bytes()
            with self.assertRaises(FileExistsError):
                render_native_coarse(selected, _clock(.1), output_dir, None)
            self.assertEqual(existing.read_bytes(), marker)
            self.assertEqual(_identity(source), original)


if __name__ == '__main__':
    unittest.main()
