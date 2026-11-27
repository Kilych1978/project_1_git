import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter1d

from common import Progress
from render_guard import compare_rendered_against_original
from safety import _validation_weights, assess_safe_validation
from validator import calculate_validation_metrics, validate_sync_map


def bidirectional(status,offset,correlation=.8):
    spread = .2 if status == 'AMBIGUOUS' else .001
    return {'status':status,'consensus_offset':offset,'consensus_correlation':correlation,
            'left_offset':offset-spread,'right_offset':offset+spread,'left_correlation':correlation,
            'right_correlation':correlation,'disagreement':2*spread,'prediction_error':abs(offset)}


class EstimatorSymmetryTests(unittest.TestCase):
    def test_identity_with_unrelated_envelopes_is_not_a_regression(self):
        # Real independent searches on unrelated signals give ambiguous halves.
        # Their average differs from the full-window peak, but BEFORE and AFTER
        # still describe exactly the same unmodified input under an identity map.
        rng=np.random.default_rng(731)
        ref=gaussian_filter1d(rng.lognormal(0,.6,9000),2)
        target=gaussian_filter1d(rng.lognormal(0,.6,9000),2)
        analysis={'reference_envelope':{'values':ref,'hop_sec':.01},
                  'target_envelope':{'values':target,'hop_sec':.01}}
        identity={'target_time':np.array([0.,90.]),'offset':np.array([0.,0.])}
        with contextlib.redirect_stdout(io.StringIO()):
            rows=validate_sync_map(analysis,identity,90.,Progress(),validation_interval_sec=30.)
        ambiguous=[r for r in rows if r['bidir_status']=='AMBIGUOUS']
        self.assertTrue(ambiguous)
        self.assertTrue(any(abs(r['bidir_consensus_offset']-r['after_offset'])>.01 for r in ambiguous))
        for row in rows:
            self.assertEqual(row['before_error'],row['after_error'])
            self.assertEqual(row['before_selected_offset'],row['after_offset'])
            self.assertEqual(row['before_selected_correlation'],row['after_correlation'])
            self.assertEqual(row['before_estimator'],row['after_estimator'])
        self.assertTrue(assess_safe_validation(rows)['passed'])
        metrics = calculate_validation_metrics(rows)
        self.assertEqual(metrics['before_mean_error'], metrics['after_mean_error'])
        self.assertEqual(metrics['before_mean_correlation'], metrics['after_mean_correlation'])
        comparison=compare_rendered_against_original(rows,rows)
        self.assertTrue(comparison['passed'])
        self.assertTrue(all(row['error_delta_sec']==0. for row in comparison['rows']))

    def test_reported_ambiguous_half_average_is_not_used_as_baseline(self):
        # Representative values from the user's retained-source report. The
        # halves averaged to -1.1735s; the same full window found +0.136s.
        raw={'offset':.136,'correlation':.6452884498}
        halves=bidirectional('AMBIGUOUS',-1.1735,.4768531108)
        analysis={'reference_envelope':{'values':np.ones(8000),'hop_sec':.01},
                  'target_envelope':{'values':np.ones(8000),'hop_sec':.01}}
        identity={'target_time':np.array([0.,80.]),'offset':np.array([0.,0.])}
        with contextlib.redirect_stdout(io.StringIO()), \
                patch('validator.build_validation_points',return_value=[18.]), \
                patch('validator.validation_search_offset',return_value=raw), \
                patch('validator.bidirectional_validation_check',return_value=halves):
            row=validate_sync_map(analysis,identity,80.,Progress())[0]
        self.assertEqual(row['bidir_status'],'AMBIGUOUS')
        self.assertEqual(row['before_bidirectional_offset'],-1.1735)
        self.assertEqual(row['before_error'],.136)
        self.assertEqual(row['after_error'],.136)
        self.assertEqual(row['before_estimator'],'full_window')

    def test_confident_consensus_still_checks_the_actual_candidate(self):
        analysis={'reference_envelope':{'values':np.ones(8000),'hop_sec':.01},
                  'target_envelope':{'values':np.ones(8000),'hop_sec':.01}}
        candidate={'target_time':np.array([0.,80.]),'offset':np.array([.2,.2])}
        halves_before=bidirectional('CONFIRMED_DISCREPANCY',.2,.95)
        halves_after=bidirectional('CONFIRMED_MAP',.2,.95)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch('validator.build_validation_points',return_value=[18.]), \
                patch('validator.validation_search_offset',return_value={'offset':.12,'correlation':.8}), \
                patch('validator.bidirectional_validation_check',side_effect=[halves_before,halves_after]):
            row=validate_sync_map(analysis,candidate,80.,Progress())[0]
        self.assertEqual(row['before_error'],.2);self.assertEqual(row['after_error'],0.)
        self.assertEqual(row['before_estimator'],'bidirectional_consensus')
        self.assertEqual(row['after_estimator'],'bidirectional_consensus')

    def test_render_comparison_does_not_trust_an_ambiguous_consensus(self):
        baseline={'target_time':20.,'before_offset':.1,'before_correlation':.8,
                  'before_bidirectional_offset':2.,'before_bidirectional_correlation':.9,
                  'before_bidirectional_status':'AMBIGUOUS'}
        rendered={'target_time':20.,'after_offset':.3,'after_correlation':.8,
                  'bidir_consensus_offset':0.,'bidir_consensus_correlation':.95,'bidir_status':'AMBIGUOUS'}
        comparison=compare_rendered_against_original([baseline],[rendered])
        self.assertFalse(comparison['passed']);self.assertEqual(comparison['regression_count'],1)
        self.assertEqual(comparison['rows'][0]['baseline_error_sec'],.1)
        self.assertEqual(comparison['rows'][0]['rendered_error_sec'],.3)

    def test_ambiguous_zero_half_average_does_not_mean_already_synced(self):
        rows = [{'target_time': t, 'before_offset': .5, 'before_correlation': .8,
                 'before_bidirectional_offset': 0., 'before_bidirectional_correlation': .8,
                 'before_bidirectional_status': 'AMBIGUOUS', 'before_error': .5,
                 'after_error': .5, 'bidir_status': 'AMBIGUOUS'} for t in (20., 30., 40.)]
        self.assertFalse(_validation_weights(rows)[-1])

    def test_already_synced_detection_uses_the_selected_baseline(self):
        rows = [{'target_time': t, 'before_offset': .5, 'before_correlation': .8,
                 'before_bidirectional_offset': .5, 'before_bidirectional_correlation': .8,
                 'before_bidirectional_status': 'AMBIGUOUS', 'before_error': 0.,
                 'before_selected_offset': 0., 'before_selected_correlation': .8,
                 'after_error': 0., 'bidir_status': 'AMBIGUOUS'} for t in (20., 30., 40.)]
        self.assertTrue(_validation_weights(rows)[-1])
        for row in rows:
            row['before_selected_offset'] = .5
            row['before_offset'] = row['before_bidirectional_offset'] = 0.
            row['before_error'] = .5
        self.assertFalse(_validation_weights(rows)[-1])


if __name__=='__main__':unittest.main()
