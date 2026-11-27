import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

import coarse_features as features
from coarse_features import FeatureBank, build_features_array, build_features_file


def audio_info(path, rate, channels, layout, duration=0.):
    return SimpleNamespace(path=Path(path), sample_rate=rate, channels=channels,
                           channel_layout=layout, stream_index=0, duration=duration)


def signal(rate, seconds=2.):
    time = np.arange(int(rate * seconds), dtype=np.float64) / rate
    left = .2 * np.sin(2 * np.pi * 713 * time) * (.6 + .3 * np.sin(2 * np.pi * 2.1 * time))
    right = .1 * np.sin(2 * np.pi * 1801 * time) * (.7 + .2 * np.cos(2 * np.pi * 1.3 * time))
    return np.column_stack([left, right]).astype(np.float32)


class CoarseFeaturesTests(unittest.TestCase):
    def test_stream_frames_are_identical_for_different_chunk_boundaries(self):
        samples = signal(12000, 2.7)
        original = samples.copy()
        with patch.object(features, '_PCM_CHUNK_SAMPLES', 12000):
            whole = build_features_array(samples, 12000, 'stereo')
        for chunk in (1, 73, 511, 997, 17003):
            with patch.object(features, '_PCM_CHUNK_SAMPLES', chunk):
                split = build_features_array(samples, 12000, 'stereo')
            np.testing.assert_array_equal(whole.values, split.values)
            np.testing.assert_array_equal(whole.energy, split.energy)
            self.assertEqual(split.parameters['decoded_samples'], len(samples))
            self.assertLess(split.parameters['trailing_samples'], features.FFT_SIZE)
        np.testing.assert_array_equal(samples, original)

    def test_frame_origin_duration_and_physical_energy_are_exact(self):
        samples = signal(12000, 1.)
        bank = build_features_array(samples, 12000, 'stereo')
        expected = 1 + (len(samples) - 512) // 120
        self.assertEqual(bank.values.shape, (expected, 5, 25))
        self.assertEqual(bank.values.dtype, np.float32)
        self.assertEqual(bank.energy.dtype, np.float32)
        self.assertEqual(bank.duration, 1.)
        self.assertEqual(bank.origin_sec, 256 / 12000)
        self.assertEqual(bank.frame_times[0], bank.origin_sec)
        self.assertEqual(bank.frame_times[-1], bank.origin_sec + (expected - 1) * .01)
        np.testing.assert_allclose(bank.energy[0, :2], np.sqrt(np.mean(samples[:512] ** 2, axis=0)), rtol=2e-7)

    def test_44100_and_48000_use_exact_rational_resampling(self):
        for rate, up, down in ((44100, 40, 147), (48000, 1, 4)):
            samples = signal(rate, 1.317)
            actual = build_features_array(samples, rate, 'stereo')
            expected_samples = resample_poly(samples, up, down, axis=0)
            expected = build_features_array(expected_samples, 12000, 'stereo')
            np.testing.assert_array_equal(actual.values, expected.values)
            np.testing.assert_array_equal(actual.energy, expected.energy)
            self.assertEqual(actual.parameters['source_samples'], len(samples))
            self.assertEqual(actual.duration, len(expected_samples) / 12000)
        # Equal band-limited content retains the same band envelope at both
        # source rates, rather than using a rounded or incorrect integer ratio.
        a = build_features_array(signal(44100), 44100, 'stereo')
        b = build_features_array(signal(48000), 48000, 'stereo')
        np.testing.assert_allclose(a.values[3:-3], b.values[3:-3], rtol=.005, atol=2e-6)

    def test_known_channel_permutation_preserves_named_physical_views(self):
        rng = np.random.default_rng(724)
        samples = rng.normal(0, .03, (18000, 6)).astype(np.float32)
        names = ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR']
        original = build_features_array(samples, 12000, names)
        permutation = [2, 4, 0, 5, 1, 3]
        swapped = build_features_array(samples[:, permutation], 12000, [names[n] for n in permutation])
        for name in names + ['M', 'S', 'energy_union']:
            a, b = original.views.index(name), swapped.views.index(name)
            np.testing.assert_allclose(original.values[:, a], swapped.values[:, b], rtol=2e-7, atol=1e-9)
            np.testing.assert_allclose(original.energy[:, a], swapped.energy[:, b], rtol=2e-7, atol=1e-9)

    def test_antiphase_union_and_silence_remain_finite(self):
        mono = signal(12000)[:, 0]
        bank = build_features_array(np.column_stack([mono, -mono]), 12000, 'stereo')
        mid, side, union, left = [bank.views.index(name) for name in ('M', 'S', 'energy_union', 'FL')]
        self.assertFalse(np.any(bank.energy[:, mid]))
        np.testing.assert_array_equal(bank.energy[:, side], bank.energy[:, left])
        np.testing.assert_array_equal(bank.values[:, union], bank.values[:, left])
        self.assertGreater(float(np.max(bank.energy[:, union])), .01)
        silence = build_features_array(np.zeros(18000, np.int16), 12000)
        self.assertTrue(np.all(np.isfinite(silence.values)))
        self.assertFalse(np.any(silence.values))
        self.assertFalse(np.any(silence.energy))
        self.assertEqual(silence.views, ['c0', 'energy_union'])

    def test_unknown_layout_does_not_invent_front_channels_or_downmix(self):
        bank = build_features_array(signal(12000), 12000, 'unknown')
        self.assertEqual(bank.views, ['c0', 'c1', 'energy_union'])
        self.assertNotIn('M', bank.views)
        self.assertNotIn('S', bank.views)

    def test_empty_and_short_arrays_have_no_padded_frames(self):
        for count in (0, 1, 511):
            with tempfile.TemporaryDirectory() as directory:
                bank = build_features_array(np.zeros(count, np.float32), 12000, directory=directory)
                self.assertEqual(bank.values.shape, (0, 2, 25))
                self.assertEqual(bank.energy.shape, (0, 2))
                self.assertEqual(bank.duration, count / 12000)
                self.assertEqual(len(bank.frame_times), 0)
                bank.close()

    def test_integer_normalization_and_source_immutability(self):
        samples = np.round(signal(12000) * 32767).astype(np.int16)
        digest = hashlib.sha256(samples.tobytes()).hexdigest()
        actual = build_features_array(samples, 12000, 'stereo')
        expected = build_features_array(samples.astype(np.float32) / 32768, 12000, 'stereo')
        np.testing.assert_array_equal(actual.values, expected.values)
        self.assertEqual(digest, hashlib.sha256(samples.tobytes()).hexdigest())

    def test_disk_bank_roundtrip_and_truncation_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = build_features_array(signal(12000), 12000, 'stereo', directory)
            self.assertIsInstance(bank.values, np.memmap)
            path = bank.metadata_path
            restored = FeatureBank.load(path)
            np.testing.assert_array_equal(bank.values, restored.values)
            np.testing.assert_array_equal(bank.energy, restored.energy)
            self.assertEqual(restored.views, bank.views)
            restored.close()
            bank.close()
            bank.close()
            (path.parent / 'energy.f32').write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                FeatureBank.load(path)

    def test_streaming_disk_buffers_do_not_grow_with_audio_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            banks = [build_features_array(signal(12000, seconds), 12000, 'stereo', directory)
                     for seconds in (2., 20.)]
            for bank in banks:
                self.assertIsInstance(bank.values, np.memmap)
                self.assertLess(bank.parameters['peak_buffer_samples'], 12000 + 512)
                self.assertLessEqual(bank.parameters['peak_fft_batch_frames'], 64)
                self.assertEqual(bank.parameters['queue_blocks'], 2)
            self.assertGreater(len(banks[1].values), 10 * banks[1].parameters['peak_fft_batch_frames'])
            self.assertEqual(banks[0].parameters['peak_buffer_samples'], banks[1].parameters['peak_buffer_samples'])
            for bank in banks:
                bank.close()

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for real streamed decode.')
    def test_real_streamed_decode_chunk_invariance_and_source_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.wav'
            source = np.round(signal(44100, 1.7) * 32767).astype(np.int16)
            wavfile.write(path, 44100, source)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            identity = path.stat().st_mtime_ns
            notifications = []
            info = audio_info(path, 44100, 2, 'stereo', duration=0.)
            with patch.object(features, '_PCM_CHUNK_SAMPLES', 257):
                a = build_features_file(info, directory, lambda done, total: notifications.append((done, total)))
            b = build_features_file(info, directory)
            np.testing.assert_array_equal(a.values, b.values)
            np.testing.assert_array_equal(a.energy, b.energy)
            self.assertEqual(a.parameters['decoded_samples'], 20400)
            self.assertEqual(a.duration, 1.7)
            self.assertEqual(a.views, ['FL', 'FR', 'M', 'S', 'energy_union'])
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(identity, path.stat().st_mtime_ns)
            self.assertEqual(notifications[0], (0., 0.))
            self.assertEqual(notifications[-1], (1.7, 1.7))
            self.assertTrue(all(first[0] <= second[0] for first, second in zip(notifications, notifications[1:])))
            self.assertNotEqual(a.metadata_path, b.metadata_path)
            a.close()
            b.close()

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for the decoder failure check.')
    def test_failed_decode_does_not_publish_partial_banks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.wav'
            path.write_bytes(b'This is not an audio stream.')
            output = Path(directory) / 'features'
            with self.assertRaisesRegex(RuntimeError, 'FFmpeg'):
                build_features_file(audio_info(path, 48000, 2, 'stereo'), output)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(path.read_bytes(), b'This is not an audio stream.')


if __name__ == '__main__':
    unittest.main()
