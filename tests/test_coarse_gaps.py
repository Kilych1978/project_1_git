"""UNKNOWN spectral events must remain gaps, without invented source clocks."""
import contextlib
import copy
import io
import unittest

import numpy as np

from common import Progress
from safety import build_non_regressive_sync_map


def observed(source_time, reference_time=None, *, feature=True):
    reference_time = source_time + 1. if reference_time is None else reference_time
    row = dict(target_time=float(source_time), predicted_offset=1., before_error=1., after_error=0.,
               before_offset=1., before_selected_offset=1., bidir_status='CONFIRMED_MAP',
               bidir_left_correlation=.9, bidir_right_correlation=.9, bidir_disagreement=0.)
    if feature:
        row.update(feature_backend=True, feature_status='CONFIRMED', uncertainty_sec=.02,
                   source_event_time=float(source_time), source_event_known=True,
                   reference_time=float(reference_time), comparison_reference_time=float(reference_time))
    return row


def unknown(reference_time):
    return dict(target_time=float(reference_time), predicted_offset=1., before_error=None, after_error=None,
                before_selected_offset=None, bidir_status='UNRESOLVED', feature_backend=True,
                feature_status='UNVERIFIABLE', source_event_time=None, source_event_known=False,
                reference_time=float(reference_time), comparison_reference_time=float(reference_time))


class CoarseGapTests(unittest.TestCase):
    def build(self, rows, offset=1.):
        times = np.arange(101, dtype=float)
        candidate = dict(target_time=times, offset=np.full(len(times), offset), confidence=np.ones(len(times)))
        with contextlib.redirect_stdout(io.StringIO()):
            return build_non_regressive_sync_map(candidate, rows, 100., Progress())

    def assert_zero_gap(self, result, begin=30., end=70.):
        query = np.linspace(begin, end, 1001)
        np.testing.assert_array_equal(np.interp(query, result['target_time'], result['offset']), np.zeros(len(query)))

    def test_unknown_middle_events_split_support_and_preserve_both_healthy_centers(self):
        rows = [observed(t) if t in (10, 20, 30, 70, 80, 90) else unknown(t + 1.) for t in range(10, 100, 10)]
        original = copy.deepcopy(rows)
        result, report = self.build(rows)
        self.assert_zero_gap(result)
        self.assertEqual(np.interp(20., result['target_time'], result['offset']), 1.)
        self.assertEqual(np.interp(80., result['target_time'], result['offset']), 1.)
        support = report['feature_support']
        self.assertEqual(support['unknown_checkpoints'], 3)
        self.assertEqual(support['supported_source_checkpoints'], 6)
        self.assertEqual([(r['start_time_sec'], r['end_time_sec']) for r in support['source_intervals']], [(10., 30.), (70., 90.)])
        self.assertEqual(support['source_coverage_seconds'], 40.)
        self.assertEqual(rows, original)
        self.assertIsNone(rows[4]['source_event_time'])
        self.assertIsNone(rows[4]['before_error'])

    def test_pcm_checkpoints_cannot_glue_spectral_unknown_intervals(self):
        spectral = [observed(t) if t in (10, 20, 30, 70, 80, 90) else unknown(t + 1.) for t in range(10, 100, 10)]
        pcm = [observed(t, feature=False) for t in range(5, 100, 5)]
        for row in pcm:
            row['evidence_kind'] = 'protected_first_channel_pcm'
        result, report = self.build(spectral + pcm)
        self.assert_zero_gap(result)
        self.assertEqual(report['feature_support']['unknown_checkpoints'], 3)
        self.assertEqual(len(report['feature_support']['source_intervals']), 2)
        self.assertEqual(np.interp(20., result['target_time'], result['offset']), 1.)

    def test_fractional_boundaries_are_exact_zero_knots(self):
        rows = [observed(t) for t in (10.25, 20.25, 30.25)]
        result, report = self.build(rows)
        self.assertIn(10.25, result['target_time'])
        self.assertIn(30.25, result['target_time'])
        query = np.r_[np.linspace(0., 10.25, 501), np.linspace(30.25, 100., 501)]
        np.testing.assert_array_equal(np.interp(query, result['target_time'], result['offset']), np.zeros(len(query)))
        self.assertEqual(np.interp(20.25, result['target_time'], result['offset']), 1.)
        self.assertEqual(len(result['confidence']), len(result['target_time']))
        self.assertEqual(len(report['feature_support']['source_intervals']), 1)

    def test_reference_coordinates_are_never_substituted_for_unknown_source(self):
        rows = [observed(t, t + 10.) for t in (10, 20, 30)] + [unknown(t) for t in (50, 60, 70)]
        result, report = self.build(rows)
        self.assert_zero_gap(result, 30., 100.)
        self.assertEqual(np.interp(20., result['target_time'], result['offset']), 1.)
        self.assertEqual(report['feature_support']['source_intervals'][0]['end_time_sec'], 30.)

    def test_nonmonotone_source_events_break_reference_ordered_support(self):
        rows = [observed(source, reference) for source, reference in [(10, 10), (20, 20), (30, 30),
                (25, 40), (40, 50), (50, 60)]]
        result, report = self.build(rows)
        self.assert_zero_gap(result, 30., 100.)
        self.assertEqual(report['feature_support']['nonmonotone_checkpoints'], 1)
        self.assertEqual(len(report['feature_support']['source_intervals']), 1)

    def test_isolated_feature_runs_cannot_create_support(self):
        rows = [observed(10), observed(20), unknown(30), observed(40), observed(50)]
        result, report = self.build(rows)
        self.assertFalse(np.any(result['offset']))
        self.assertEqual(report['feature_support']['source_intervals'], [])

    def test_dynamic_span_keeps_the_existing_short_grid_rule(self):
        rows = [observed(10), observed(15), observed(20), unknown(25)]
        result, report = self.build(rows)
        self.assertEqual(report['feature_support']['source_intervals'][0]['span_sec'], 10.)
        # Existing tempo limits can attenuate a supported short correction;
        # accepting the run must not bypass that separate geometry guard.
        self.assertGreater(np.interp(15., result['target_time'], result['offset']), 0.)
        self.assertLessEqual(np.interp(15., result['target_time'], result['offset']), 1.)

    def test_three_close_events_do_not_bypass_the_existing_span_requirement(self):
        rows = [observed(0, feature=False), observed(5, feature=False),
                observed(10), observed(11), observed(12)]
        result, report = self.build(rows)
        self.assertFalse(np.any(result['offset']))
        self.assertEqual(report['feature_support']['source_intervals'], [])

    def test_protected_pcm_rejection_wins_at_identical_source_time(self):
        protected = observed(20, feature=False)
        protected.update(evidence_kind='protected_first_channel_pcm', before_error=.006,
                         after_error=1., bidir_status='CONFIRMED_DISCREPANCY')
        rows = [protected] + [observed(t) for t in (10, 20, 30, 70, 80, 90)]
        rows += [unknown(t) for t in (40, 50, 60)]
        result, report = self.build(rows)
        at_twenty = [row for row in report['validation_decisions'] if row['target_time'] == 20.]
        self.assertEqual(len(at_twenty), 1)
        self.assertEqual(at_twenty[0]['weight'], 0.)
        self.assertEqual(np.interp(20., result['target_time'], result['offset']), 0.)
        self.assertEqual(np.interp(80., result['target_time'], result['offset']), 1.)

    def test_negative_offsets_are_zero_outside_the_same_source_intervals(self):
        rows = [observed(t, t-1.) for t in (10, 20, 30)]
        for row in rows:
            row.update(before_selected_offset=-1., before_offset=-1., predicted_offset=-1.)
        result, _ = self.build(rows, offset=-1.)
        self.assert_zero_gap(result, 30., 100.)
        self.assertEqual(np.interp(20., result['target_time'], result['offset']), -1.)

    def test_legacy_without_feature_backend_is_bitwise_unchanged(self):
        rows = [observed(t, feature=False) for t in (10, 20, 30, 70, 80, 90)]
        ordinary, report = self.build(rows)
        # An equivalent explicitly non-feature record remains the same legacy
        # path; no support mask or additional boundary knots are introduced.
        explicit = copy.deepcopy(rows)
        for row in explicit:
            row['feature_backend'] = False
            row['source_event_time'] = None
        checked, explicit_report = self.build(explicit)
        for field in ('target_time', 'offset', 'reference_time', 'confidence'):
            self.assertEqual(ordinary[field].tobytes(), checked[field].tobytes())
        self.assertEqual(np.interp(50., ordinary['target_time'], ordinary['offset']), 1.)
        self.assertFalse(report['feature_support']['applied'])
        self.assertFalse(explicit_report['feature_support']['applied'])


if __name__ == '__main__':
    unittest.main()
