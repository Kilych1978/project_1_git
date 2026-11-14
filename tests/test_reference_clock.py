import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pcm_core.model import TimeMap
from pcm_core.blend import BlendMap,sample_map
from pcm_core.recovery import RestoredMap
from pcm_core.micro import measure_window
from pcm_core.guard import verify_candidate_pcm,verify_render
from pcm_core.preflight import settle,failed
from pcm_core.config import Settings
from pcm_core.render import render
from pcm_core.audio import probe
from .helpers import soundtrack,warp,write_audio,SR


class ReferenceClockTests(unittest.TestCase):
    def test_protected_restored_pcm_change_is_detected_despite_equal_nominal_clock(self):
        from pcm_core.recovery import extent
        from pcm_core.choice import verify_reference_grid
        r,t,s=self.fixture()
        clock=TimeMap.affine(6.,offset=-.006)
        protected=RestoredMap(clock,TimeMap.affine(6.),[[2.5,3.5]])
        for proposal in [clock,BlendMap(clock,[])]:
            self.assertGreater(extent(proposal,protected,[3.]),.0059)
            rows=verify_reference_grid(r,t,protected,proposal,s,[3.])
            self.assertTrue(rows[0]['micro_changed'])

    def fixture(self):
        reference=soundtrack(6.,848)
        true=TimeMap.affine(6.,rate=.997)
        target=warp(reference,true.forward(np.arange(len(reference))/SR)*SR)
        reference=np.clip(np.rint(reference*32768),-32768,32767).astype(np.int16)
        target=np.clip(np.rint(target*32768),-32768,32767).astype(np.int16)
        settings=Settings(anchor_step=.125,min_improvement=1/SR,verification_pcm16=True)
        return reference,target,settings

    def test_restored_pcm_with_different_nominal_clock_is_not_false_regression(self):
        r,t,s=self.fixture()
        base=TimeMap.affine(6.,offset=-.002)
        current=RestoredMap(TimeMap.affine(6.,offset=.008),base,[[2.5,3.5]])
        # The previous verifier compared different waveform windows and would
        # reject byte-identical audio by more than its one-sample threshold.
        before=measure_window(r,t,base,3.,s,quantize_pcm16=True)
        after=measure_window(r,t,current,3.,s,quantize_pcm16=True)
        self.assertGreater(abs(after['correction_sec'])-abs(before['correction_sec']),1/SR)
        row=verify_candidate_pcm(r,t,base,current,s,target_times=[3.])[0]
        self.assertFalse(row['micro_changed'])
        self.assertEqual(row['status'],'VERIFIED')
        self.assertEqual(row['baseline_residual_sec'],row['residual_sec'])

    def test_baseline_cache_tracks_changed_reference_center(self):
        r,t,s=self.fixture();base=TimeMap.affine(6.,offset=-.002)
        cache={};rows=[]
        for offset in [.008,.018]:
            current=RestoredMap(TimeMap.affine(6.,offset=offset),base,[[2.5,3.5]])
            row=verify_candidate_pcm(r,t,base,current,s,target_times=[3.],baseline_cache=cache)[0]
            self.assertEqual(row['status'],'VERIFIED')
            self.assertEqual(row['baseline_residual_sec'],row['residual_sec'])
            rows.append(row)
        self.assertNotEqual(rows[0]['baseline_residual_sec'],rows[1]['baseline_residual_sec'])
        self.assertEqual(len(cache),2)

    def test_corrupted_native_pcm_is_rejected_even_when_nominal_change_is_zero(self):
        r,t,s=self.fixture();base=TimeMap.affine(6.,offset=-.002)
        current=RestoredMap(TimeMap.affine(6.,offset=.008),base,[[2.5,3.5]])
        actual=t.astype(np.float32)/32768;actual[round(2.9*SR):round(3.1*SR)]=0
        row=verify_render(r,t,actual,base,current,s,target_times=[3.])[0]
        self.assertFalse(row['micro_changed'])
        self.assertEqual(row['status'],'REGRESSION')

    def test_complete_gate_preserves_distant_gain_with_restored_nominal_clock_in_native_six_channels(self):
        r,t,s=self.fixture();base=TimeMap.affine(6.,offset=-.002)
        clock=TimeMap.affine(6.,offset=.008)
        layer={'reference':[1.,1.012,1.988,2.],'weight':[0.,1.,1.,0.],
               'rate':.997,'offset':0.}
        proposal=RestoredMap(BlendMap(clock,[layer]),base,[[0.,.98],[2.02,6.]])
        source=TimeMap.affine(6.);times=[1.3,1.5,1.7,3.]
        final,rows,history=settle(r,t,source,proposal,base,[(s,times)],s,times,protected=True)
        self.assertEqual(history[-1]['failed'],0)
        self.assertFalse(any(x.get('status')=='COMPLETE_PLAN_FALLBACK' for x in history))
        self.assertGreaterEqual(history[-1]['improved'],3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audio=root/'source.wav'
            channels=np.stack([t.astype(np.float32)/32768*(1-i*.1) for i in range(6)],axis=1)
            write_audio(audio,channels,codec='pcm_s16le');unchanged=audio.read_bytes()
            work=root/'out';work.mkdir()
            path=render(probe(audio),final,6.,work,s,log=lambda _:None)
            _,written=wavfile.read(path);actual=written[:,0].astype(np.float32)/32768
            checks=verify_render(r,t,actual,base,final,s,target_times=times)
            self.assertFalse(failed(checks))
            good=[x for x in checks if x['target_time']<2]
            self.assertTrue(all(abs(x['residual_sec'])<1/SR for x in good))
            self.assertEqual(written.shape,(6*SR,6));self.assertEqual(audio.read_bytes(),unchanged)
            _,original=wavfile.read(audio)
            ts=3.008+np.arange(-200,201)/SR
            expected=np.clip(np.rint(sample_map(original,base,ts,SR,16)),-32768,32767).astype(np.int16)
            np.testing.assert_allclose(written[np.rint(ts*SR).astype(int)],expected,atol=1)


if __name__=='__main__':unittest.main()
