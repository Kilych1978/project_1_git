import unittest
import numpy as np
from scipy.signal import butter, sosfiltfilt
from wavesync.alignment import AlignmentError, coarse_alignment, validate_coarse
from wavesync.config import Settings
from wavesync.features import envelope
from wavesync.guard import gate_micro, same_map
from wavesync.micro import Measurement, collect, measure
from wavesync.model import TimeMap
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

    def test_coarse_fft_delay_and_independent_validation(self):
        delay = round(1.275 * SR)
        target = np.r_[np.zeros(delay), self.reference[:-delay]].astype("f4")
        ref_env = envelope(self.reference, SR, self.settings)
        tar_env = envelope(target, SR, self.settings)
        base, records = coarse_alignment(ref_env, tar_env, 24, 24, self.settings, QUIET)
        self.assertLess(abs(float(base.forward([12])[0]) - 12 + 1.275), .003)
        checks = validate_coarse(base, ref_env, tar_env, 24, self.settings)
        self.assertGreaterEqual(sum(r["status"] == "VERIFIED" for r in checks), 3)

    def test_pal_speed_is_estimated_and_refined(self):
        reference = soundtrack(32, seed=33)
        rate = 25 / 24
        offset = .013
        target_duration = 30.
        times = np.arange(round(target_duration * SR)) / SR
        target = warp(reference, (rate * times + offset) * SR)
        settings = Settings(anchor_step=2.)
        ref_env, tar_env = envelope(reference, SR, settings), envelope(target, SR, settings)
        base, _ = coarse_alignment(ref_env, tar_env, 32, target_duration, settings, QUIET)
        self.assertLess(abs(base.local_rate(15) - rate), .0002)
        ms = collect(reference, target, base, settings, QUIET)
        final, checks, _ = gate_micro(reference, target, base, ms, settings, QUIET)
        trusted = [m.target_time for m in ms if m.status == "MEASURED"]
        self.assertGreater(len(trusted), 8)
        error = final.forward(trusted) - (rate * np.array(trusted) + offset)
        self.assertLess(np.median(np.abs(error)), 1 / SR)

    def test_polarity_inversion_and_gain_do_not_reverse_delay(self):
        n = round(.006 * SR)
        target = -.35 * np.r_[np.zeros(n), self.reference[:-n]]
        m = measure(self.reference, target, self.identity, 12., self.settings)
        self.assertEqual(m.status, "MEASURED")
        self.assertEqual(m.polarity, -1)
        self.assertLess(abs(m.correction_sec + .006), .1 / SR)

    def test_montage_jump_is_not_smeared_into_tempo(self):
        reference = soundtrack(40, seed=14)
        times = np.arange(36 * SR) / SR
        target = warp(reference, (times + np.where(times < 18, .5, 2.5)) * SR)
        settings = Settings(anchor_step=2.)
        with self.assertRaises(AlignmentError) as context:
            coarse_alignment(envelope(reference, SR, settings), envelope(target, SR, settings), 40, 36, settings, QUIET)
        self.assertTrue(context.exception.records)

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
