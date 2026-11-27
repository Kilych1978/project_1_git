"""Completed processing, unchanged audio and established alignment differ."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diagnostics import diagnostic_session, register_output, register_sync_outcome
from pcm_core.config import Settings
from pcm_core.guard import verify_candidate_pcm
from pcm_core.model import TimeMap
from sync_outcome import assess_sync_outcome, save_sync_outcome
from .helpers import SR, soundtrack


def pcm_window(time, residual=0.000001, before=.010, changed=True):
    return {'target_time': time, 'reference_time': time, 'status': 'VERIFIED',
            'residual_sec': residual, 'baseline_residual_sec': before,
            'micro_changed': changed, 'score': .99, 'verification_window_sec': .125}


class SynchronizationOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recorded = json.loads((Path(__file__).parent / 'fixtures/sync_outcome_20261127.json').read_text(encoding='utf-8'))

    def spectral_rows(self, residual=.005, uncertainty=.060):
        return [dict(target_time=float(t),reference_time=float(t),bidir_status='CONFIRMED_MAP',
                     bidir_consensus_offset=residual,bidir_left_correlation=.9,
                     bidir_right_correlation=.9,bidir_disagreement=.01,
                     feature_backend=True,uncertainty_sec=uncertainty)
                for t in (10,30,50,70,90,110)]

    def test_coarse_uncertainty_cannot_be_presented_as_precise_alignment(self):
        rows=self.spectral_rows()
        result=assess_sync_outcome({'validation_points':rows},duration=120.,output_created=False)
        self.assertEqual(result['status'],'UNVERIFIED')
        self.assertFalse(result['alignment_confirmed'])
        self.assertEqual(result['spectral_measured_windows'],6)
        self.assertEqual(result['spectral_evidence']['trusted_windows'],0)

    def test_spectral_residual_lower_bound_proves_a_remaining_large_offset(self):
        rows=self.spectral_rows(.15,.02)
        result=assess_sync_outcome({'validation_points':rows},duration=120.,output_created=False)
        self.assertEqual(result['status'],'NOT_SYNCHRONIZED')
        self.assertEqual(result['confirmed_mismatch_count'],6)

    def test_uncertain_feature_peak_does_not_invalidate_precise_verified_pcm(self):
        rows=self.spectral_rows(.035,.060)
        pcm=[pcm_window(t) for t in (10.,30.,50.,70.,90.,110.)]
        result=assess_sync_outcome({'validation_points':rows},
            {'status':'PCM_REFINED','actual_output_windows':pcm},duration=120.,output_created=True)
        self.assertEqual(result['status'],'SYNCHRONIZED')

    def test_recorded_retained_source_is_not_declared_synchronized(self):
        data = self.recorded
        result = assess_sync_outcome(data['verification'], data['pcm_report'],
            output_created=data['output_created'], render_guard=data['render_guard'],
            duration=data['verification']['reference_duration'])
        self.assertEqual(result['status'], 'NOT_SYNCHRONIZED')
        self.assertFalse(result['alignment_confirmed'])
        self.assertFalse(result['output_created'])
        self.assertFalse(result['improvement_verified'])
        self.assertTrue(result['non_regression_passed'])
        self.assertGreaterEqual(result['confirmed_mismatch_count'], 63)
        self.assertLessEqual(result['confirmed_mismatch_count'], 71)
        self.assertEqual(result['ambiguous_envelope_windows'], 440)
        self.assertIn('новый аудиофайл не создан', result['message_ru'])

    def test_bad_ambiguous_summary_is_not_presented_as_proven_seconds_of_drift(self):
        rows = [{'target_time': float(t), 'bidir_status': 'AMBIGUOUS',
                 'bidir_consensus_offset': 1.946, 'after_error': 1.946}
                for t in (10, 30, 50, 70, 90, 110)]
        result = assess_sync_outcome({'validation_points': rows,
            'metrics': {'after_median_error': 1.946}}, output_created=False, duration=120.)
        self.assertEqual(result['status'], 'UNVERIFIED')
        self.assertEqual(result['confirmed_mismatch_count'], 0)
        self.assertFalse(result['alignment_confirmed'])

    def test_real_identical_pcm_with_measurable_waveform_is_already_aligned(self):
        samples = soundtrack(3., 582)
        mapping = TimeMap.affine(3.)
        settings = Settings(anchor_step=.125)
        rows = verify_candidate_pcm(samples, samples, mapping, mapping, settings)
        self.assertGreater(sum(row['status'] == 'VERIFIED' for row in rows), 10)
        result = assess_sync_outcome({}, {'status': 'BASELINE_KEPT',
            'actual_output_windows': [], 'selected_settings': asdict(settings)},
            baseline_plan={'fine_map': mapping.to_dict(), 'native_preview_windows': rows},
            output_created=False, duration=3.)
        self.assertEqual(result['status'], 'ALREADY_ALIGNED')
        self.assertTrue(result['alignment_confirmed'])
        self.assertFalse(result['output_created'])

    def test_real_silence_is_unverifiable_instead_of_already_aligned(self):
        silence = np.zeros(2 * SR, dtype=np.float32)
        mapping = TimeMap.affine(2.)
        settings = Settings(anchor_step=.25)
        rows = verify_candidate_pcm(silence, silence, mapping, mapping, settings)
        self.assertTrue(rows)
        self.assertTrue(all(row['status'] == 'UNVERIFIABLE' for row in rows))
        result = assess_sync_outcome({}, {'status': 'BASELINE_KEPT',
            'actual_output_windows': rows, 'selected_settings': asdict(settings)},
            output_created=False, duration=2.)
        self.assertEqual(result['status'], 'UNVERIFIED')
        self.assertFalse(result['alignment_confirmed'])

    def test_local_improvement_does_not_hide_another_confirmed_large_residual(self):
        windows = [pcm_window(t) for t in (10., 30., 50., 70., 90., 110.)]
        verification = {'validation_points': [{'target_time': 100.,
            'bidir_status': 'CONFIRMED_DISCREPANCY', 'bidir_consensus_offset': .9,
            'bidir_left_correlation': .95, 'bidir_right_correlation': .94,
            'bidir_disagreement': .01}]}
        result = assess_sync_outcome(verification, {'status': 'PCM_REFINED',
            'actual_output_windows': windows}, output_created=True, duration=120.)
        self.assertEqual(result['status'], 'IMPROVED_NOT_FULLY_ALIGNED')
        self.assertTrue(result['improvement_verified'])
        self.assertFalse(result['alignment_confirmed'])

    def test_three_verified_improvements_in_one_corner_do_not_prove_full_alignment(self):
        result = assess_sync_outcome({}, {'status': 'PCM_REFINED',
            'actual_output_windows': [pcm_window(t) for t in (1., 2., 3.)]},
            output_created=True, duration=120.)
        self.assertEqual(result['status'], 'IMPROVED_UNVERIFIED')
        self.assertTrue(result['improvement_verified'])
        self.assertFalse(result['alignment_confirmed'])

    def test_verified_pcm_is_not_overruled_by_ambiguous_envelope_peaks(self):
        windows = [pcm_window(t) for t in (1., 3., 5., 7., 9., 11.)]
        verification = {'metrics': {'after_median_error': 1.946},
            'validation_points': [{'target_time': t, 'bidir_status': 'AMBIGUOUS',
                                   'bidir_consensus_offset': 1.946} for t in (1., 3., 5., 7., 9., 11.)]}
        result = assess_sync_outcome(verification, {'status': 'PCM_REFINED',
            'actual_output_windows': windows}, output_created=True, duration=12.)
        self.assertEqual(result['status'], 'SYNCHRONIZED')
        self.assertTrue(result['alignment_confirmed'])
        self.assertTrue(result['improvement_verified'])

    def test_unrendered_rejected_pcm_proposal_cannot_validate_the_retained_source(self):
        proposal = TimeMap.affine(12., offset=-.006)
        rows = [pcm_window(t) for t in (1., 3., 5., 7., 9., 11.)]
        result = assess_sync_outcome({}, {'status': 'BASELINE_KEPT', 'actual_output_windows': []},
            output_created=False, duration=12.,
            baseline_plan={'fine_map': proposal.to_dict(), 'native_preview_windows': rows})
        self.assertEqual(result['status'], 'UNVERIFIED')
        self.assertFalse(result['alignment_confirmed'])

    def test_written_outcome_keeps_measurements_and_explains_geometry_score(self):
        data = self.recorded
        outcome = assess_sync_outcome(data['verification'], data['pcm_report'],
            output_created=False, render_guard=data['render_guard'], duration=5252.)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'FINAL_SYNC_REPORT.txt').write_text('MAP VERDICT: GOOD\nScore: 100.00/100\n', encoding='utf-8')
            (root / 'FINAL_RENDER_VALIDATION.json').write_text(json.dumps({
                'duration_check': False, 'duration_check_enforced': False,
                'metrics': {'after_median_error': 1.7345}}), encoding='utf-8')
            save_sync_outcome(root, outcome)
            written = json.loads((root / 'FINAL_RENDER_VALIDATION.json').read_text())
            self.assertEqual(written['metrics']['after_median_error'], 1.7345)
            self.assertFalse(written['duration_check_enforced'])
            self.assertEqual(written['synchronization_outcome']['status'], 'NOT_SYNCHRONIZED')
            report = (root / 'FINAL_SYNC_REPORT.txt').read_text()
            self.assertIn('СИНХРОНИЗАЦИЯ НЕ ДОСТИГНУТА', report[:250])
            self.assertIn('сами по себе не подтверждают синхронизацию', report)
            self.assertTrue((root / 'SYNC_OUTCOME_RU.txt').is_file())

    def test_run_info_separates_execution_completion_from_sync_status(self):
        data = self.recorded
        outcome = assess_sync_outcome(data['verification'], data['pcm_report'],
            output_created=False, render_guard=data['render_guard'], duration=5252.)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            @diagnostic_session
            def completed_analysis():
                register_output(root)
                register_sync_outcome(outcome)

            with contextlib.redirect_stdout(io.StringIO()):
                completed_analysis()
            info = json.loads((root / 'RUN_INFO.json').read_text())
            self.assertEqual(info['status'], 'COMPLETED')
            self.assertEqual(info['synchronization_status'], 'NOT_SYNCHRONIZED')
            self.assertFalse(info['synchronization_confirmed'])


if __name__ == '__main__':
    unittest.main()
