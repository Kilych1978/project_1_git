"""Global clock confirmation must not manufacture tempo ramps at file edges."""
import contextlib
import copy
import io
import unittest

import numpy as np

from coarse_clock import fit_global_clock, map_with_global_clock, verify_global_clock
from common import Progress
from safety import build_non_regressive_sync_map


class GlobalClockTests(unittest.TestCase):
    def setup_case(self, offset=3.31, rate=.999, duration=80.):
        train=np.linspace(4.,duration-4.,12)
        anchors=[dict(target_time=float(t),offset=(rate-1)*t+offset,
                      uncertainty_sec=.02) for t in train]
        model=fit_global_clock(anchors,duration)
        self.assertIsNotNone(model)
        times=np.linspace(0.,duration,161)
        candidate=map_with_global_clock({'target_time':times,'confidence':np.ones(len(times))},model)
        # Different times from training: heldout observations of the source.
        rows=[]
        for t in np.linspace(5.,duration-5.,10):
            observed=(rate-1)*t+offset
            rows.append(dict(target_time=float(t),source_event_time=float(t),
                reference_time=float(t+observed),feature_backend=True,feature_status='CONFIRMED',
                coarse_global_clock=model,before_selected_offset=observed,before_offset=observed,
                before_error=abs(observed),after_error=0.,predicted_offset=observed,
                uncertainty_sec=.02,bidir_status='CONFIRMED_MAP',
                bidir_left_correlation=.9,bidir_right_correlation=.9,bidir_disagreement=0.))
        return candidate,rows,model

    def test_constant_delay_and_rate_keep_their_edges_without_artificial_stretch(self):
        for offset,rate in ((3.31,.999),(-4.7,1.001),(.5,.96),(-.5,1.04)):
            candidate,rows,model=self.setup_case(offset,rate)
            with contextlib.redirect_stdout(io.StringIO()):
                safe,report=build_non_regressive_sync_map(candidate,rows,80.,Progress())
            self.assertEqual(report['mode'],'VERIFIED_GLOBAL_CLOCK')
            np.testing.assert_array_equal(safe['target_time'],candidate['target_time'])
            np.testing.assert_array_equal(safe['offset'],candidate['offset'])
            self.assertAlmostEqual((safe['reference_time'][-1]-safe['reference_time'][0])/80.,rate)

    def test_training_alone_cannot_confirm_a_global_clock(self):
        candidate,rows,_=self.setup_case()
        self.assertIsNone(verify_global_clock(candidate,[],80.))
        for row in rows[2:]:
            row.update(feature_status='UNVERIFIABLE',source_event_time=None,before_error=None,after_error=None)
        self.assertIsNone(verify_global_clock(candidate,rows,80.))

    def test_heldout_contradiction_and_quarantine_prevent_global_exemption(self):
        candidate,rows,_=self.setup_case()
        self.assertIsNone(verify_global_clock(candidate,rows,80.,quarantined=True))
        rows[4]['before_selected_offset']+=.2
        self.assertIsNone(verify_global_clock(candidate,rows,80.))

    def test_protected_pcm_veto_is_not_relaxed_by_global_feature_evidence(self):
        candidate,rows,_=self.setup_case()
        rows.append(dict(evidence_kind='protected_first_channel_pcm',before_error=.006,after_error=.030))
        self.assertIsNone(verify_global_clock(candidate,rows,80.))

    def test_different_candidate_cannot_use_proof_for_another_clock(self):
        candidate,rows,_=self.setup_case()
        candidate['offset']=candidate['offset']+.1
        self.assertIsNone(verify_global_clock(candidate,rows,80.))

    def test_real_nonlinear_edits_are_not_mislabelled_as_one_global_rate(self):
        t=np.linspace(4.,116.,24)
        anchors=[dict(target_time=float(x),offset=float(.2+(x>55)*2.5),uncertainty_sec=.02) for x in t]
        self.assertIsNone(fit_global_clock(anchors,120.))


if __name__=='__main__':unittest.main()
