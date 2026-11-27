"""A feature winner must not sacrifice an already measurable PCM clock."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from coarse_pcm_guard import PCMProtectedEvidence
from render_guard import compare_rendered_against_original
from safety import _validation_weights
from .helpers import SR, soundtrack


class DummyEvidence:
    def validate(self, *args, **kwargs):
        return []
    def for_output(self, *args, **kwargs):
        return DummyEvidence()
    def close(self):
        pass


class CoarsePCMGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = np.clip(np.rint(soundtrack(10., 912)*24000), -32768, 32767).astype('<i2')
        self.target = np.r_[np.zeros(round(.006*SR), dtype='<i2'), self.reference[:-round(.006*SR)]]
        self.reference.tofile(self.root/'reference.pcm')
        self.target.tofile(self.root/'target.pcm')
        self.guard = PCMProtectedEvidence(DummyEvidence(), self.root/'reference.pcm', self.root/'target.pcm')
        # Dummy banks are not owned by the production close path.
        self.addCleanup(self._close_pcm)

    def _close_pcm(self):
        for name in ('_reference', '_target'):
            arr=getattr(self.guard,name)
            if arr is not None:arr._mmap.close();setattr(self.guard,name,None)

    def mapping(self, offset):
        return {'target_time': np.array([0.,10.]), 'offset': np.array([offset,offset])}

    def test_identity_keeps_baseline_error_exactly_and_correct_clock_improves(self):
        original = self.guard.validate(self.mapping(0.), 10., interval_sec=2.)
        self.assertGreater(len(original), 2)
        self.assertTrue(all(r['before_error']==r['after_error'] for r in original))
        fixed = self.guard.validate(self.mapping(-.006), 10., interval_sec=2.)
        self.assertLess(max(r['after_error'] for r in fixed), 1/SR)
        self.assertTrue(np.all(_validation_weights(fixed)[1] > .5))

    def test_wrong_feature_clock_cannot_move_known_six_ms_pcm_to_35_ms(self):
        wrong = self.guard.validate(self.mapping(.035), 10., interval_sec=2.)
        self.assertGreater(len(wrong), 2)
        self.assertTrue(all(r['after_error'] > r['before_error']+.030 for r in wrong))
        self.assertTrue(np.all(_validation_weights(wrong)[1] == 0))

    def test_actual_written_worse_pcm_is_a_regression(self):
        baseline = self.guard.validate(self.mapping(0.), 10., interval_sec=2.)
        # A renderer that added 8 ms instead of removing the original 6 ms.
        actual = np.r_[np.zeros(round(.008*SR),dtype='<i2'), self.target[:-round(.008*SR)]]
        actual.tofile(self.root/'actual.pcm')
        checked = self.guard.for_output(None, self.root)
        checked.set_rendered_pcm(self.root/'actual.pcm')
        try:
            rows=checked.validate(self.mapping(0.),10.,interval_sec=2.)
            compared=compare_rendered_against_original(baseline,rows)
            self.assertFalse(compared['passed'])
            self.assertGreater(compared['regression_count'],2)
        finally:
            checked.close()

    def test_lost_pcm_evidence_cannot_silently_pass_a_blank_output(self):
        baseline=self.guard.validate(self.mapping(0.),10.,interval_sec=2.)
        np.zeros(10*SR,dtype='<i2').tofile(self.root/'blank.pcm')
        checked=self.guard.for_output(None,self.root)
        checked.set_rendered_pcm(self.root/'blank.pcm')
        try:
            rows=checked.validate(self.mapping(0.),10.,interval_sec=2.)
            self.assertTrue(all(r['feature_verification_lost'] for r in rows))
            self.assertFalse(compare_rendered_against_original(baseline,rows)['passed'])
        finally:checked.close()

    def test_spectral_uncertainty_cannot_authorize_small_unproven_changes(self):
        records=[dict(target_time=float(t),before_error=.025,after_error=.005,
                      predicted_offset=.020,before_offset=.025,bidir_status='CONFIRMED_MAP',
                      bidir_left_correlation=.9,bidir_right_correlation=.9,bidir_disagreement=0.,
                      feature_backend=True,uncertainty_sec=.020) for t in (2,4,6)]
        self.assertTrue(np.all(_validation_weights(records)[1] == 0))
        # Exactly the same clear difference is sufficient for precise PCM.
        for row in records:row['feature_backend']=False
        self.assertTrue(np.all(_validation_weights(records)[1] > .5))


if __name__=='__main__':unittest.main()
