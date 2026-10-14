import unittest
import numpy as np
from scipy.signal import butter, sosfiltfilt
from pcm_core.config import Settings
from pcm_core.guard import gate_micro, same_map
from pcm_core.micro import Measurement, collect, measure
from pcm_core.model import TimeMap
from .helpers import SR, soundtrack, warp


QUIET = lambda *args: None


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = soundtrack(24.)
        cls.settings = Settings(anchor_step=2.)
        cls.identity = TimeMap.affine(24.)

    def test_exact_6_and_10_ms_and_negative_shifts(self):
        for shift in [.006, .010, -.006]:
            with self.subTest(shift=shift):
                n = round(abs(shift) * SR)
                x = self.reference
                target = np.r_[np.zeros(n), x[:-n]] if shift > 0 else np.r_[x[n:], np.zeros(n)]
                measurements = collect(x, target, self.identity, self.settings, QUIET)
                final, checks, _ = gate_micro(x, target, self.identity, measurements, self.settings, QUIET)
                ts = np.linspace(0, 24, 100)
                error = final.forward(ts) - ts + shift
                self.assertLess(np.max(np.abs(error)), .1 / SR)
                self.assertTrue(any(r["status"] == "IMPROVED" for r in checks))
                self.assertFalse(any(r["status"] == "REGRESSION" for r in checks))

    def test_fractional_sample_delay(self):
        shift = 6.0037 / 1000
        positions = np.arange(len(self.reference)) - shift * SR
        target = warp(self.reference, positions)
        m = measure(self.reference, target, self.identity, 10., self.settings)
        self.assertEqual(m.status, "MEASURED")
        self.assertLess(abs(m.correction_sec + shift), .3 / SR)

    def test_identity_does_not_change_already_synced_audio(self):
        ms = collect(self.reference, self.reference, self.identity, self.settings, QUIET)
        final, _, _ = gate_micro(self.reference, self.reference, self.identity, ms, self.settings, QUIET)
        self.assertTrue(same_map(final, self.identity))

    def test_periodic_signal_is_ambiguous(self):
        t = np.arange(8 * SR) / SR
        reference = np.sin(2 * np.pi * 1000 * t).astype("f4")
        target = np.sin(2 * np.pi * 1000 * (t - .006)).astype("f4")
        measured = measure(reference, target, TimeMap.affine(8), 4., self.settings)
        self.assertNotEqual(measured.status, "MEASURED")

    def test_silence_cannot_confirm_a_shift(self):
        silent = np.zeros(8 * SR, dtype="f4")
        m = measure(silent, silent, TimeMap.affine(8), 4., self.settings)
        self.assertIsNone(m.correction_sec)

    def test_unrelated_tracks_are_not_refined(self):
        target = soundtrack(24, seed=91)
        ms = collect(self.reference, target, self.identity, self.settings, QUIET)
        final, _, _ = gate_micro(self.reference, target, self.identity, ms, self.settings, QUIET)
        self.assertTrue(same_map(final, self.identity))

    def test_wrong_strong_anchors_are_rejected_by_real_holdouts(self):
        ms = [Measurement(t, t, .007, .99, .9, 0., 3, 1, "MEASURED", "raw") for t in np.arange(.8, 23.2, 2.)]
        final, rows, audit = gate_micro(self.reference, self.reference, self.identity, ms, self.settings, QUIET)
        self.assertTrue(same_map(final, self.identity))
        self.assertTrue(any(r["status"] == "REGRESSION" for r in audit))



    def test_polarity_inversion_and_gain_do_not_reverse_delay(self):
        n = round(.006 * SR)
        target = -.35 * np.r_[np.zeros(n), self.reference[:-n]]
        m = measure(self.reference, target, self.identity, 12., self.settings)
        self.assertEqual(m.status, "MEASURED")
        self.assertEqual(m.polarity, -1)
        self.assertLess(abs(m.correction_sec + .006), .1 / SR)


    def test_multiband_recovers_common_effects_under_different_low_frequency_mix(self):
        rng = np.random.default_rng(710)
        count = 8 * SR
        common = sosfiltfilt(butter(3, [1800, 11000], btype="bandpass", fs=SR, output="sos"), rng.normal(size=count)).astype("f4") * .04
        left_mix = sosfiltfilt(butter(4, 500, fs=SR, output="sos"), rng.normal(size=count)) * 2
        right_mix = sosfiltfilt(butter(4, 500, fs=SR, output="sos"), rng.normal(size=count)) * 2
        n = round(.006 * SR)
        reference = common + left_mix
        target = np.r_[np.zeros(n), common[:-n]] + right_mix
        measured = measure(reference, target, TimeMap.affine(8), 4., self.settings)
        self.assertEqual(measured.status, "MEASURED")
        self.assertEqual(measured.method, "multiband")
        self.assertLess(abs(measured.correction_sec + .006), 1 / SR)

    def test_dense_refinement_improves_a_smooth_local_10_ms_error(self):
        t = np.arange(len(self.reference)) / SR
        correction = -.010 * np.exp(-((t - 12.) / 5.) ** 2)
        target = warp(self.reference, (t + correction) * SR)
        settings = Settings(anchor_step=1.)
        ms = collect(self.reference, target, self.identity, settings, QUIET)
        final, checks, _ = gate_micro(self.reference, target, self.identity, ms, settings, QUIET)
        improved = [r for r in checks if r["status"] == "IMPROVED"]
        self.assertGreater(len(improved), 8)
        self.assertLess(np.median([abs(r["after_residual_sec"]) for r in improved]), 1 / SR)
        self.assertFalse(any(r["status"] == "REGRESSION" for r in checks))


if __name__ == "__main__":
    unittest.main()
