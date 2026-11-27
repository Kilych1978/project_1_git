"""Independent affine-envelope matcher tests, without movie-specific fixtures."""
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

from coarse_matcher import match_window


def bank(values, names=None, energy=None):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, None, :]
    if energy is None:
        energy = np.sqrt(np.mean(values ** 2, axis=2))
    return SimpleNamespace(values=values, energy=np.asarray(energy, dtype=np.float32),
                           hop_sec=.01, origin_sec=0., duration=len(values) * .01,
                           views=names or [f'c{i}' for i in range(values.shape[1])],
                           parameters={'band_centers_hz': np.geomspace(60, 5000, values.shape[2]).tolist()})


def latent(seed=17, seconds=48, bands=12):
    rng = np.random.default_rng(seed)
    z = gaussian_filter1d(rng.normal(size=(round(seconds * 100), bands)), 2.0, axis=0)
    z /= np.std(z, axis=0)
    return .035 * np.exp(.6 * z)


def warped(values, offset, rate=1.0):
    t = np.arange(len(values)) * .01
    return np.column_stack([np.interp(rate * t + offset, t, values[:, j])
                            for j in range(values.shape[1])])


class CoarseMatcherTests(unittest.TestCase):
    def assert_match(self, result, offset, rate=1., query=24.):
        self.assertEqual(result['status'], 'CONFIRMED', result)
        self.assertAlmostEqual(result['offset'], offset + (rate - 1) * query, delta=.035)
        self.assertAlmostEqual(result['rate'], rate, delta=.009)
        self.assertGreaterEqual(len([p for p in result['disjoint_holdouts']
                                     if p['confirms_frozen_model']]), 2)
        self.assertGreaterEqual(result['uncertainty_sec'], .02)

    def test_random_positive_and_negative_offsets_beyond_three_seconds(self):
        x = latent()
        for offset in (5.37, -7.21, .34):
            with self.subTest(offset=offset):
                result = match_window(bank(x), bank(warped(x, offset)), 24., search_radius=10.)
                self.assert_match(result, offset)

    def test_affine_speed_in_both_directions(self):
        x = latent(seed=38)
        for rate in (.96, 1.04):
            with self.subTest(rate=rate):
                result = match_window(bank(x), bank(warped(x, 2.4, rate)), 24., search_radius=10.)
                self.assert_match(result, 2.4, rate)

    def test_unrelated_and_silent_banks_are_not_confirmed(self):
        x = latent()
        for y in (latent(seed=190), np.zeros_like(x)):
            with self.subTest(silence=not np.any(y)):
                result = match_window(bank(x), bank(y), 24.)
                self.assertNotEqual(result['status'], 'CONFIRMED', result)

    def test_repeating_pattern_has_competing_lags(self):
        motif = latent(seconds=1.)
        x = np.tile(motif, (48, 1))
        result = match_window(bank(x), bank(warped(x, 4.2)), 24., search_radius=10.)
        self.assertEqual(result['status'], 'AMBIGUOUS', result)

    def test_training_only_decoy_fails_disjoint_holdouts(self):
        x = latent(seed=771)
        y = latent(seed=951)
        offset = 4.7
        correct = warped(x, offset)
        # Exact positive training data with *no* common sound in holdout windows.
        for t in 24 + 6 * np.array([-.375, -.125, .125, .375]):
            lo, hi = round((t - .45) * 100), round((t + .45) * 100)
            y[lo:hi] = correct[lo:hi]
        result = match_window(bank(x), bank(y), 24., search_radius=10.)
        self.assertNotEqual(result['status'], 'CONFIRMED', result)
        self.assertTrue(result['training_candidate_count'] > 0, result)
        self.assertIn('holdout', result['reason'])

    def test_one_conflicting_strong_holdout_blocks_affine_confirmation(self):
        x = latent(seed=731)
        y = warped(x, 4.7)
        other_edit = warped(x, -3.1)
        # Train and the other holdouts all agree; a cut in the central holdout
        # is positive contradictory evidence, not a missing/quiet observation.
        lo, hi = round(23.6 * 100), round(24.4 * 100)
        y[lo:hi] = other_edit[lo:hi]
        result = match_window(bank(x), bank(y), 24., search_radius=10.)
        self.assertEqual(result['status'], 'AMBIGUOUS', result)
        self.assertIn('contradict', result['reason'])

    def test_channel_permutation_and_dual_mono_do_not_add_votes(self):
        x, unrelated = latent(), latent(seed=144)
        y = warped(x, 4.2)
        reference = bank(np.stack([unrelated, x, x], axis=1), ['FR', 'FL', 'M'])
        target = bank(np.stack([y, y, y * 1e-4], axis=1), ['R', 'L', 'S'])
        result = match_window(reference, target, 24., search_radius=10.)
        self.assert_match(result, 4.2)
        self.assertEqual(result['reference_view'], 'FL')
        self.assertNotIn('S', result['active_views']['target'])
        self.assertEqual(result['duplicate_views']['target'], {'L': 'R'})
        self.assertEqual(len(result['probes']), 4)

    def test_frozen_views_cannot_switch_to_unrelated_new_channel(self):
        x, noise = latent(), latent(seed=44)
        y = warped(x, 4.2)
        reference = bank(np.stack([x, noise], axis=1), ['FL', 'FC'])
        result = match_window(reference, bank(y, ['M']), 24., search_radius=10.,
                              frozen_views={'reference_view': 'FC', 'target_view': 'M', 'band': 'all'})
        self.assertNotEqual(result['status'], 'CONFIRMED', result)

    def test_dual_mono_noise_cannot_manufacture_shared_signal(self):
        rng = np.random.default_rng(404)
        x = latent()
        n = .04 * np.exp(.6 * gaussian_filter1d(rng.normal(size=x.shape), 2, axis=0))
        values = np.stack([n, n, rng.random(x.shape) * 1e-7], axis=1)
        result = match_window(bank(x), bank(values, ['FL', 'FR', 'S']), 24.)
        self.assertNotEqual(result['status'], 'CONFIRMED', result)
        self.assertNotIn('S', result['active_views']['target'])

    def test_freezing_a_duplicate_preserves_the_requested_channel(self):
        x = latent()
        y = warped(x, 4.2)
        target = bank(np.stack([y, y], axis=1), ['FL', 'FR'])
        result = match_window(bank(x, ['FC']), target, 24., search_radius=10.,
                              frozen_views={'reference_view': 'FC', 'target_view': 'FR', 'band': 'all'})
        self.assert_match(result, 4.2)
        self.assertEqual(result['target_view'], 'FR')
        self.assertEqual(result['frozen_selection'],
                         {'reference_view': 'FC', 'target_view': 'FR', 'band': 'all'})

    def test_pcm_polarity_and_added_independent_dub_voice(self):
        # Exercise actual PCM extraction rather than only ideal feature arrays.
        # Independent band modulation prevents a single repetitive loudness
        # envelope from making all bands appear to be separate evidence.
        from coarse_features import build_features_array
        rng = np.random.default_rng(7713)
        sr, seconds = 12000, 40
        n = sr * seconds
        time = np.arange(n) / sr
        original = np.zeros(n)
        for lo, hi in [(200, 500), (600, 1000), (1300, 1900), (2100, 3000), (3500, 5000)]:
            mod = gaussian_filter1d(rng.normal(size=4000), 3)
            mod /= mod.std()
            envelope = np.interp(time, np.arange(4000) / 100, np.exp(.8 * mod))
            carrier = sosfiltfilt(butter(3, [lo, hi], btype='bandpass', fs=sr, output='sos'),
                                 rng.normal(size=n))
            original += .025 * carrier * envelope
        shifted = np.interp(time + 4.2, time, original)
        voice = .25 * sosfiltfilt(butter(5, 900, fs=sr, output='sos'), rng.normal(size=n))
        reference = build_features_array(original, sr)
        try:
            for label, pcm in [('negative_polarity', -shifted), ('dub_voice', shifted + voice)]:
                with self.subTest(label=label):
                    target = build_features_array(pcm, sr)
                    try:
                        result = match_window(reference, target, 20., search_radius=8.)
                        self.assert_match(result, 4.2, query=20.)
                    finally:
                        target.close()
        finally:
            reference.close()

    def test_leading_render_silence_cannot_steal_best_or_second_peak(self):
        x = latent(seed=941, seconds=40)
        rendered = x.copy()
        rendered[:333] = 0.0
        # A positive synchronization offset legitimately pads the output.
        # FFT roundoff in these zero windows used to turn into NCC=1 and hide
        # every real match until the padding left the broad search region.
        output_bank = bank(rendered, ['output'])
        reference_bank = bank(x, ['reference'])
        frozen = {'reference_view': 'output', 'target_view': 'reference', 'band': 'all'}
        for query in (5.2, 7.8, 10.4, 13., 15.6, 18.2, 20.8):
            with self.subTest(query=query):
                result = match_window(output_bank, reference_bank, query,
                                      search_radius=15., frozen_views=frozen)
                self.assert_match(result, 0.0, query=query)
                for point in result['probes'] + result['disjoint_holdouts']:
                    self.assertGreaterEqual(point['reference_rms'], 1e-5)

    def test_near_search_edge_is_not_a_confirmed_interior_match(self):
        x = latent()
        result = match_window(bank(x), bank(warped(x, 5.0)), 24., search_radius=5.)
        self.assertEqual(result['status'], 'BOUNDARY', result)


if __name__ == '__main__':
    unittest.main()
