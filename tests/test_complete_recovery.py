import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from pcm_core.blend import BlendMap, sample_map
from pcm_core.config import Settings
from pcm_core.guard import _change_extent, _signature, verify_render, render_verdict
from pcm_core.model import TimeMap
from pcm_core.preflight import settle, failed
from pcm_core.recovery import RestoredMap, restore_failures
from pcm_core.choice import verify_reference_grid
from pcm_best import ProtectedPlan
from pcm_refiner import refine_rendered_audio, export_diagnostic_clips
from common import SelectedTracks
from scanner import probe_audio_file
from .helpers import SR, soundtrack, write_audio


class Quiet:
    def __call__(self,*args): pass
    def finish(self): pass


class CompleteRecoveryTests(unittest.TestCase):
    def test_complete_window_restores_inner_clock_and_all_layers(self):
        source=np.random.default_rng(81).integers(-10000,10000,(6*SR,6),dtype=np.int16)
        protected=TimeMap.affine(6.,offset=-.004)
        inner=TimeMap.affine(6.,offset=-.010)
        layer={'reference':[.5,.512,5.488,5.5],'weight':[0.,1.,1.,0.],
               'rate':1.,'offset':-.006}
        stack=BlendMap(BlendMap(inner,[layer]),[layer])
        bad=[{'reference_time':3.,'verification_window_sec':.125}]
        # Removing overlap weights alone exposes the wrong inner -10 ms clock.
        old,_=stack.reduced([3.],.0635,protected_base=protected)
        times=3.+np.arange(-2000,2001)/SR
        self.assertGreater(np.max(np.abs(sample_map(source,old,times,SR,16)-
                                        sample_map(source,protected,times,SR,16))),100.)
        restored=restore_failures(stack,protected,bad)
        expected=sample_map(source,protected,times,SR,16)
        np.testing.assert_array_equal(sample_map(source,restored,times,SR,16),expected)
        distant=np.arange(1.,1.02,1/SR)
        np.testing.assert_array_equal(sample_map(source,restored,distant,SR,16),
                                      sample_map(source,stack,distant,SR,16))
        roundtrip=TimeMap.from_dict(json.loads(json.dumps(restored.to_dict())))
        np.testing.assert_array_equal(sample_map(source,roundtrip,times,SR,16),expected)
        config=Settings(anchor_step=.125)
        self.assertEqual(_signature(stack,1.,config),_signature(restored,1.,config))
        self.assertNotEqual(_signature(stack,3.,config),_signature(restored,3.,config))
        self.assertEqual(restored.extent(protected,times),0.)

    def test_empty_pcm_stack_cannot_change_reference_window_change_detection(self):
        from pcm_core.model import Piece
        knots=np.array([0.,.9,1.,1.1,2.])
        delta=np.array([0.,0.,-.012,0.,0.])
        clock=TimeMap([Piece(knots,knots+delta,[{'target':knots.tolist(),'delta':delta.tolist()}])])
        source=TimeMap.affine(2.)
        for t in [.92,1.,1.06]:
            a=_change_extent(source,clock,t,Settings(anchor_step=1.),edge=True)
            wrapped=clock
            for _ in range(4):wrapped=BlendMap(wrapped,[])
            b=_change_extent(source,wrapped,t,Settings(anchor_step=1.),edge=True)
            self.assertAlmostEqual(a,b,places=12)
            self.assertGreater(a,1/SR)

    def test_full_refiner_repairs_before_multichannel_write_and_passes_actual_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'result';out.mkdir()
            reference=soundtrack(6.,844)
            target=np.r_[np.zeros(288),reference[:-288]]
            rp,tp=root/'reference.wav',root/'target.wav'
            write_audio(rp,reference,codec='pcm_s16le')
            write_audio(tp,np.stack([target*(1-.1*i) for i in range(6)],axis=1),codec='pcm_s16le')
            original=tp.read_bytes()
            _,ref=wavfile.read(rp);_,src=wavfile.read(tp)
            ref.tofile(root/'ref.pcm');src[:,0].tofile(root/'src.pcm')
            config=Settings(anchor_step=.5,min_improvement=1/SR,max_rate_change=.02,verification_pcm16=True)
            times=np.arange(.3,5.7,.125).tolist();fixed=np.arange(.3,5.7,.0625).tolist()
            source=TimeMap.affine(6.);protected=TimeMap.affine(6.,offset=-.004)
            inner=TimeMap.affine(6.,offset=-.010)
            layer={'reference':[.3,.312,2.188,2.2],'weight':[0.,1.,1.,0.],
                   'rate':1.,'offset':-.006}
            candidate=BlendMap(inner,[layer])
            protection=ProtectedPlan(protected,[(config,times)],fixed,config,True)
            plan=(candidate,[],[],config,[],[(config,times)],protection)
            selected=SelectedTracks(probe_audio_file(rp),probe_audio_file(tp))
            zero={'target_time':np.array([0.,6.]),'reference_time':np.array([0.,6.]),'offset':np.zeros(2)}
            from pcm_core.render import render as actual_render
            with patch('pcm_best.build_best_plan',return_value=plan),patch('pcm_refiner.render',wraps=actual_render) as writer,contextlib.redirect_stdout(io.StringIO()):
                _,result,report=refine_rendered_audio(selected,{'reference_pcm':root/'ref.pcm'},zero,
                    {'output_path':tp,'encoder':'copy','copied_without_reencode':True,
                     'verified_first_channel_pcm':root/'src.pcm'},out,config,make_plots=False)
            self.assertEqual(report['status'],'PCM_REFINED',report.get('reason'))
            self.assertEqual(writer.call_count,1)
            self.assertTrue((out/report['proposal_map_file']).is_file())
            clips=json.loads((out/'_diagnostic_clips'/'CLIPS.json').read_text())
            # The final WAV passed, but evidence of its pre-render rejected
            # proposal must still be included for diagnosis.
            self.assertIn('proposal_target',clips['clips'][0]['files'])
            self.assertGreater(report['complete_plan_audit'][0]['failed'],0)
            self.assertEqual(report['complete_plan_audit'][-1]['failed'],0)
            self.assertFalse(failed(report['actual_output_windows']))
            self.assertFalse(failed(report['protected_output_windows']))
            _,pcm=wavfile.read(result['output_path'])
            self.assertEqual(pcm.shape,(6*SR,6))
            final=TimeMap.from_dict(report['fine_map'])
            virtual=sample_map(src,final,np.arange(len(src))/SR,SR,16)
            expected=np.clip(np.rint(virtual),-32768,32767).astype(np.int16)
            self.assertLessEqual(np.max(np.abs(pcm.astype(float)-expected)),1.)
            self.assertEqual(tp.read_bytes(),original)
            # The failed scene retains the protected 4 ms correction, while
            # the distant 6 ms correction remains substantially better.
            from pcm_core.micro import measure_window
            actual=pcm[:,0].astype(np.float32)/32768
            before=measure_window(ref,src[:,0],protected,1.,config)
            after=measure_window(ref,actual,TimeMap.affine(6.),1.,config)
            self.assertLess(abs(after['correction_sec']),1/SR)
            self.assertGreater(abs(before['correction_sec']),.001)

    def test_diagnostic_bundle_contains_failed_window_and_rejected_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=np.arange(30*SR,dtype=np.int16);candidate=source.copy();candidate[10*SR:11*SR]=0
            export_diagnostic_clips(source,source,root,SR,times=[],final=source,
                include_uniform=False,failure_times=[10.5],candidate=candidate)
            info=json.loads((root/'_diagnostic_clips'/'CLIPS.json').read_text())
            self.assertEqual(len(info['clips']),1)
            clip=info['clips'][0];self.assertIn('candidate_target',clip['files'])
            _,written=wavfile.read(root/'_diagnostic_clips'/clip['files']['candidate_target'])
            begin=round(clip['start_sec']*SR)
            np.testing.assert_array_equal(written,candidate[begin:begin+len(written)])


if __name__=='__main__': unittest.main()
