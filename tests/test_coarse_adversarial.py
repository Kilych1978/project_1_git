"""Adversarial checks of coarse-clock exceptions and fallback routing."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from coarse_alignment import NoCommonAudio, run_general_alignment_stage
from coarse_clock import fit_global_clock, map_with_global_clock, verify_global_clock
from common import Progress


class CoarseAdversarialTests(unittest.TestCase):
    def clock_case(self):
        duration = 80.
        anchors = [dict(target_time=float(t), offset=3.31, uncertainty_sec=.02)
                   for t in np.linspace(4., 76., 12)]
        model = fit_global_clock(anchors, duration)
        self.assertIsNotNone(model)
        times = np.arange(81, dtype=float)
        mapping = map_with_global_clock(dict(target_time=times, confidence=np.ones(len(times))), model)
        rows = []
        for time in np.linspace(5., 75., 10):
            rows.append(dict(target_time=float(time), source_event_time=float(time),
                             source_event_known=True, reference_time=float(time + 3.31),
                             comparison_reference_time=float(time + 3.31),
                             feature_backend=True, feature_status='CONFIRMED', coarse_global_clock=model,
                             before_selected_offset=3.31, before_offset=3.31, before_error=3.31,
                             after_error=0., predicted_offset=3.31, uncertainty_sec=.02,
                             bidir_status='CONFIRMED_MAP', bidir_left_correlation=.9,
                             bidir_right_correlation=.9, bidir_disagreement=0.))
        return mapping, rows

    def make_unknown(self, row, status='UNVERIFIABLE'):
        row.update(feature_status=status, source_event_time=None, source_event_known=False,
                   before_selected_offset=None, before_offset=None, before_error=None,
                   after_error=None, bidir_status='UNRESOLVED')

    def test_positive_conflicting_holdouts_veto_even_with_seventy_percent_agreement(self):
        mapping, rows = self.clock_case()
        for row in rows[4:7]:
            self.make_unknown(row, 'AMBIGUOUS')
            row.update(feature_reason='Independent holdout events contradict the frozen affine proposal.',
                       holdouts=[dict(target_time=row['reference_time'], offset=-4.01, score=.95,
                                      margin=.30, passes_peak_gate=True, confirms_frozen_model=False,
                                      residual_sec=-.70)])
        self.assertIsNone(verify_global_clock(mapping, rows, 80.))

    def test_prominent_search_boundary_vetoes_global_extrapolation(self):
        mapping, rows = self.clock_case()
        for row in rows[4:7]:
            self.make_unknown(row, 'BOUNDARY')
            row.update(feature_hit_boundary=True,
                       feature_reason='Prominent training or holdout events touch the search boundary.',
                       probes=[dict(target_time=row['reference_time'], offset=-15., score=.96,
                                    margin=.40, hit_boundary=True, passes_peak_gate=False)])
        self.assertIsNone(verify_global_clock(mapping, rows, 80.))

    def test_lost_output_proof_is_not_dropped_as_an_ordinary_unknown(self):
        mapping, rows = self.clock_case()
        self.make_unknown(rows[4])
        rows[4]['feature_verification_lost'] = True
        self.assertIsNone(verify_global_clock(mapping, rows, 80.))

    def test_quiet_unknowns_are_not_fabricated_as_contradictory_measurements(self):
        mapping, rows = self.clock_case()
        for row in rows[4:7]:
            self.make_unknown(row)
            row.update(feature_reason='No reliable common spectral events.', probes=[], holdouts=[])
        snapshot = copy.deepcopy(rows)
        proof = verify_global_clock(mapping, rows, 80.)
        self.assertIsNotNone(proof)
        self.assertEqual(proof['independent_validation_points'], 7)
        self.assertEqual(rows, snapshot)
        self.assertIsNone(rows[5]['before_error'])
        self.assertIsNone(rows[5]['source_event_time'])

    def test_an_isolated_false_peak_does_not_become_proof_in_either_direction(self):
        mapping, rows = self.clock_case()
        row = rows[4]
        self.make_unknown(row)
        row.update(feature_reason='Not enough independent training events.',
                   probes=[dict(target_time=row['reference_time'], offset=-4.01, score=.95,
                                margin=.30, window_sec=.6,
                                passes_peak_gate=True, hit_boundary=False)], holdouts=[])
        self.assertIsNotNone(verify_global_clock(mapping, rows, 80.))
        self.assertIsNone(row['before_error'])
        self.assertIsNone(row['source_event_time'])
        # Nor may a single agreeing event fill the missing support of other
        # UNKNOWN windows and manufacture a confirmed global relation.
        row['probes'][0]['offset'] = -3.31
        for other in rows[5:]:
            self.make_unknown(other)
        self.assertIsNone(verify_global_clock(mapping, rows, 80.))

    def test_three_independent_conflicting_events_veto_the_global_clock(self):
        mapping, rows = self.clock_case()
        row = rows[4]
        self.make_unknown(row)
        row['probes'] = [dict(target_time=row['reference_time'] + dt, offset=-4.01,
                              reference_time=row['reference_time'] + dt - 4.01,
                              window_sec=.6, score=.95, margin=.30,
                              passes_peak_gate=True, hit_boundary=False)
                         for dt in (-1.5, 0., 1.5)]
        row['holdouts'] = []
        self.assertIsNone(verify_global_clock(mapping, rows, 80.))

    def test_overlapping_partial_probes_cannot_multiply_one_event(self):
        mapping, rows = self.clock_case()
        row = rows[4]
        self.make_unknown(row)
        row['probes'] = [dict(target_time=row['reference_time'] + dt, offset=-4.01,
                              reference_time=row['reference_time'] + dt - 4.01,
                              window_sec=.6, score=.95, margin=.30,
                              passes_peak_gate=True, hit_boundary=False)
                         for dt in (-.1, 0., .1)]
        row['holdouts'] = []
        self.assertIsNotNone(verify_global_clock(mapping, rows, 80.))

    def test_three_reference_queries_matching_one_source_event_are_not_independent(self):
        mapping, rows = self.clock_case()
        row = rows[4]
        self.make_unknown(row)
        repeated_source = row['reference_time'] - 5.
        row['probes'] = [dict(target_time=row['reference_time'] + dt,
                              offset=repeated_source - (row['reference_time'] + dt),
                              reference_time=repeated_source,
                              window_sec=.6, score=.95, margin=.30,
                              passes_peak_gate=True, hit_boundary=False)
                         for dt in (-1.5, 0., 1.5)]
        row['holdouts'] = []
        self.assertIsNotNone(verify_global_clock(mapping, rows, 80.))

    def test_global_clock_cannot_be_reused_for_a_different_source_duration(self):
        mapping, rows = self.clock_case()
        model = rows[0]['coarse_global_clock']
        times = np.linspace(0., 800., 801)
        unrelated_scope = map_with_global_clock(dict(target_time=times, confidence=np.ones(len(times))), model)
        self.assertIsNone(verify_global_clock(unrelated_scope, rows, 800.))

    def test_a_partial_or_extended_candidate_domain_cannot_claim_the_full_source(self):
        mapping, rows = self.clock_case()
        model = rows[0]['coarse_global_clock']
        for begin, end in ((5., 75.), (0., 120.), (-10., 80.)):
            with self.subTest(begin=begin, end=end):
                times = np.linspace(begin, end, 101)
                candidate = map_with_global_clock(dict(target_time=times, confidence=np.ones(len(times))), model)
                self.assertIsNone(verify_global_clock(candidate, rows, 80.))

    def test_unavailable_legacy_windows_do_not_block_the_feature_backend(self):
        # 160s passes the optional classic-path duration gate, but no 120s
        # template fits its ±15s search at the classic fractional checkpoints.
        # The genuine classic geometry runs here; only the new decoder is a
        # sentinel to prove the fallback routing is reached.
        analysis = dict(reference_envelope={'values': np.ones(16000), 'hop_sec': .01},
                        target_envelope={'values': np.ones(16000), 'hop_sec': .01},
                        reference_pcm=Path('reference.pcm'), target_pcm=Path('target.pcm'))
        selected = SimpleNamespace(reference=SimpleNamespace(duration=160.),
                                   target=SimpleNamespace(duration=160.))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch('coarse_features.build_features_file', side_effect=OSError('Sentinel feature decoder unavailable')) as decoder:
            with self.assertRaises(NoCommonAudio):
                run_general_alignment_stage(selected, Path(directory), analysis, Progress())
        decoder.assert_called_once()


if __name__ == '__main__':
    unittest.main()
