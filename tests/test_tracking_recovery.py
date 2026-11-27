"""A weak dubbed interval must not drag the next search away from common sound."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter1d

import aligner
from common import Progress


class TrackingRecoveryTests(unittest.TestCase):
    def test_unrelated_interval_does_not_advance_clock_and_common_audio_is_reacquired(self):
        rng = np.random.default_rng(97)
        times = np.arange(0., 100., .01)
        reference = (gaussian_filter1d(rng.normal(size=len(times)), 3)
                     + .2 * gaussian_filter1d(rng.normal(size=len(times)), 50))
        target = np.interp(times + .25, times, reference)
        unrelated = (times >= 20) & (times < 45)
        target[unrelated] = gaussian_filter1d(rng.normal(size=np.count_nonzero(unrelated)), 3)
        with patch.multiple(aligner, LOCAL_MATCH_WINDOW_SEC=4., LOCAL_SEARCH_RADIUS_SEC=2.,
                            LOCAL_ANCHOR_INTERVAL_SEC=5.), contextlib.redirect_stdout(io.StringIO()):
            rows = aligner.build_local_anchor_table(reference, target, 100., .25, Progress())
        weak = [row for row in rows if 22 <= row['target_time'] <= 42]
        self.assertGreaterEqual(len(weak), 4)
        # Keep the measurements for diagnostics while withholding their clock.
        self.assertTrue(all(row['offset'] is not None for row in weak))
        self.assertFalse(any(row['accepted_for_tracking'] for row in weak))
        recovered = [row for row in rows if row['target_time'] > 47]
        self.assertTrue(all(row['accepted_for_tracking'] for row in recovered))
        self.assertLess(max(abs(row['offset'] - .25) for row in recovered), .002)

    def test_observed_weak_peak_runaway_cannot_become_next_search_center(self):
        # Sanitized measurements from the failing report: the previous tracker
        # accepted .18/.16 correlations and walked from .114 to +22.482 seconds.
        observed = [
            (.131, .845176, .015677), (.114, .813390, .029076),
            (10.222, .182572, .011786), (17.662, .428938, .012525),
            (18.242, .238938, .010302), (22.482, .161550, .011662),
            (15.592, .166758, .007203), (18.292, .230664, .008398),
        ]
        calls = []
        def search(*args):
            calls.append(args[3])
            offset, correlation, margin = observed[len(calls)-1]
            return dict(offset=offset, correlation=correlation, margin=margin,
                        second_correlation=correlation-margin)
        initial = 1.493
        start = 15. + 8. + abs(initial) + 2.
        duration = 2*start + (len(observed)-1)*30.
        with patch.object(aligner, 'search_best_offset_bruteforce', side_effect=search), \
             contextlib.redirect_stdout(io.StringIO()):
            rows = aligner.build_local_anchor_table(np.ones(10), np.ones(10), duration, initial, Progress())
        self.assertEqual(len(rows), len(observed))
        self.assertEqual(calls[:2], [initial, initial])
        np.testing.assert_allclose(calls[2:], .114)
        self.assertFalse(any(row['accepted_for_tracking'] for row in rows[2:]))

    def test_smooth_nonlinear_delay_is_still_tracked(self):
        rng = np.random.default_rng(132)
        times = np.arange(0., 100., .01)
        reference = gaussian_filter1d(rng.normal(size=len(times)), 3)
        true_offset = .2 + .35 * np.sin(times/30.)
        target = np.interp(times + true_offset, times, reference)
        with patch.multiple(aligner, LOCAL_MATCH_WINDOW_SEC=4., LOCAL_SEARCH_RADIUS_SEC=2.,
                            LOCAL_ANCHOR_INTERVAL_SEC=5.), contextlib.redirect_stdout(io.StringIO()):
            rows = aligner.build_local_anchor_table(reference, target, 100., .2, Progress())
        accepted = [r for r in rows if r['accepted_for_tracking']]
        self.assertGreater(len(accepted), len(rows)*.8)
        error = [abs(row['offset'] - (.2+.35*np.sin(row['target_time']/30.))) for row in accepted]
        self.assertLess(max(error), .015)


if __name__ == '__main__':
    unittest.main()
