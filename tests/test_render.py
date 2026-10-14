import tempfile
import unittest
from pathlib import Path
import numpy as np
from wavesync.audio import find_tools, probe
from wavesync.config import Settings
from wavesync.dsp import sinc_sample
from wavesync.model import Piece, TimeMap
from wavesync.render import render
from .helpers import SR, read_audio, soundtrack, write_audio


class RenderTests(unittest.TestCase):
    def test_integer_positions_copy_exactly_including_values_above_one(self):
        source = np.array([[0., 1.5], [-1.3, .7], [.3, -.9]], dtype="f4")
        actual = sinc_sample(source, np.array([0., 1., 2.]))
        np.testing.assert_array_equal(actual, source)

    def test_out_of_range_is_silence(self):
        actual = sinc_sample(np.ones((10, 2), "f4"), np.array([-1., np.nan, 20.]))
        np.testing.assert_array_equal(actual, np.zeros((3, 2)))

    def test_antialias_filter_at_compression(self):
        t = np.arange(10000)
        source = np.sin(2 * np.pi * .48 * t).astype("f4")
        pos = np.arange(200., 9800., 1.08)
        unfiltered = sinc_sample(source, pos)
        filtered = sinc_sample(source, pos, cutoff=1 / 1.08)
        self.assertLess(np.std(filtered), np.std(unfiltered) * .50)

    def test_invalid_maps_and_overlapping_edits_are_rejected(self):
        with self.assertRaises(ValueError):
            TimeMap([Piece([0, 2], [1, 0])])
        with self.assertRaises(ValueError):
            TimeMap([Piece([0, 2], [0, 2]), Piece([1, 4], [2, 4])])
        with self.assertRaises(ValueError):
            TimeMap.affine(10, rate=1.2).validate()

    def test_6_channels_share_one_map_and_native_rate_is_preserved(self):
        find_tools()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            t = np.arange(2 * SR) / SR
            source = np.stack([.15 * np.sin(2 * np.pi * (130 + i * 210) * t) for i in range(6)], axis=1).astype("f4")
            path = root / "target.wav"
            write_audio(path, source, layout="5.1")
            work = root / "output"
            work.mkdir()
            shift = 288
            result = render(probe(path), TimeMap.affine(2, offset=-shift / SR), 2., work, Settings(), lambda *a: None)
            info, values = read_audio(result)
            self.assertEqual(info.channels, 6)
            self.assertEqual(info.channel_layout, "5.1")
            self.assertEqual(info.sample_rate, SR)
            np.testing.assert_allclose(values[:-shift], source[shift:], atol=2e-6, rtol=0)
            np.testing.assert_array_equal(values[-shift:], 0)
            self.assertFalse((work / "target_multichannel.f32").exists())

    def test_explicit_edit_map_omits_target_scene_and_inserts_reference_silence(self):
        find_tools()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = np.repeat(np.array([.1, .2, .3, .4], dtype="f4"), SR)
            path = root / "target.wav"
            write_audio(path, source)
            output = root / "output"
            output.mkdir()
            mapping = TimeMap([Piece([0, 1], [0, 1]), Piece([2, 4], [1.5, 3.5])])
            result = render(probe(path), mapping, 3.5, output, Settings(), lambda *a: None)
            _, actual = read_audio(result)
            # Ignore 2 samples at discontinuities where resampling has finite support.
            self.assertAlmostEqual(float(actual[100:SR-100].mean()), .1, places=5)
            np.testing.assert_array_equal(actual[SR+100:round(1.5*SR)-100], 0)
            self.assertAlmostEqual(float(actual[round(1.5*SR)+100:round(2.5*SR)-100].mean()), .3, places=5)

    def test_first_channel_extraction_does_not_downmix_center_dialogue(self):
        from wavesync.audio import decode
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rng = np.random.default_rng(17)
            source = np.zeros((SR, 6), dtype="f4")
            source[:, 0] = np.linspace(-.2, .2, SR)
            source[:, 2] = rng.normal(size=SR)
            path = root / "movie.wav"
            write_audio(path, source, layout="5.1")
            mono = decode(probe(path), root / "mono.f32", SR, True)
            try:
                np.testing.assert_array_equal(mono, source[:, 0])
            finally:
                mono._mmap.close()

    def test_44100_hz_is_preserved_and_float_peaks_are_not_clipped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rate = 44100
            t = np.arange(rate) / rate
            source = np.stack([1.2 * np.sin(2 * np.pi * 310 * t), .3 * np.cos(2 * np.pi * 610 * t)], axis=1).astype("f4")
            path = root / "target.wav"
            write_audio(path, source, rate)
            out = root / "out"
            out.mkdir()
            result = render(probe(path), TimeMap.affine(1), 1., out, Settings(), lambda *a: None)
            info, actual = read_audio(result)
            self.assertEqual(info.sample_rate, rate)
            self.assertGreater(np.max(actual), 1.1)
            np.testing.assert_array_equal(actual, source)

    def test_cubic_map_roundtrip_and_exact_numerical_inverse(self):
        t = np.array([0., 2., 4., 6., 8.])
        delta = np.array([0., .005, .010, .006, 0.])
        mapping = TimeMap([Piece(t, t + delta, [{"target": t.tolist(), "delta": delta.tolist()}])]).validate()
        restored = TimeMap.from_dict(mapping.to_dict()).validate()
        probes = np.linspace(.01, 7.99, 1000)
        np.testing.assert_allclose(mapping.forward(probes), restored.forward(probes), atol=1e-12, rtol=0)
        np.testing.assert_allclose(mapping.inverse(mapping.forward(probes)), probes, atol=1e-11, rtol=0)

    def test_cubic_rate_extrema_are_checked_between_anchors(self):
        # Average segment speeds ±6% are allowed, but the cubic edge speed is 12%.
        piece = Piece([0., 1., 2.], [0., 1.06, 2.], [{"target": [0., 1., 2.], "delta": [0., .06, 0.]}])
        with self.assertRaises(ValueError):
            TimeMap([piece]).validate()


if __name__ == "__main__":
    unittest.main()
