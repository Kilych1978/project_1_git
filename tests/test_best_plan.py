import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from pcm_best import build_best_plan,bad_rows
from pcm_core.audio import probe,decode
from pcm_core.choice import choose_local,replace_map,verify_reference_grid
from pcm_core.config import Settings
from pcm_core.guard import verify_render
from pcm_core.model import Piece,TimeMap
from pcm_core.render import render
from pcm_core.blend import BlendMap,gate_patches,sample_map
from .helpers import SR,soundtrack,warp,write_audio
from common import SelectedTracks
from scanner import probe_audio_file
from pcm_refiner import refine_rendered_audio


class QuietProgress:
    def __call__(self,*args,**kwargs):
        pass
    def finish(self):
        pass
    def begin_pass(self,*args):
        pass


class BestPlanTests(unittest.TestCase):
    def test_wide_window_smear_still_reaches_short_waveform_patch_search(self):
        from pcm_core.micro import measure_window
        reference=soundtrack(6.,777)
        true=TimeMap.affine(6.,rate=.999,offset=-.003)
        target=warp(reference,true.forward(np.arange(len(reference))/SR)*SR)
        base=TimeMap.affine(6.)
        settings=Settings(anchor_step=1.,min_improvement=1/SR,max_rate_change=.02)
        before=measure_window(reference,target,base,3.,settings)
        self.assertGreater(before['score'],settings.min_raw_score)
        self.assertLess(before['score'],.85)
        def planner(ref,tgt,mapping,config,report,progress,classic=False):
            return mapping,[],[],config,[],[(config,[2.875,3.,3.125])]
        with tempfile.TemporaryDirectory() as tmp:
            report={}
            final,*_=build_best_plan(reference,target,base,settings,Path(tmp),report,
                lambda *args,**kwargs:QuietProgress(),planner)
            after=measure_window(reference,target,final,3.,Settings(anchor_step=.125))
            self.assertLess(abs(after['correction_sec']),1/SR)
            self.assertTrue(report.get('residual_clock_measurements'))
            self.assertTrue(report['additional_improvement_confirmed'])

    def test_pcm_result_does_not_collide_with_accepted_coarse_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            reference=soundtrack(4.,914)
            target=np.r_[np.zeros(round(.006*SR)),reference[:-round(.006*SR)]]
            rp,tp=root/'reference.wav',root/'target.wav'
            write_audio(rp,reference,codec='pcm_s16le');write_audio(tp,target,codec='pcm_s16le')
            _,r=wavfile.read(rp);_,t=wavfile.read(tp)
            r.tofile(root/'reference.pcm');t.tofile(root/'target.pcm')
            out=root/'result';out.mkdir()
            coarse=out/'target_SYNCED.wav';coarse.write_bytes(tp.read_bytes())
            before=coarse.read_bytes()
            selected=SelectedTracks(probe_audio_file(rp),probe_audio_file(tp))
            times=np.array([0.,4.])
            zero={'target_time':times,'reference_time':times.copy(),'offset':np.zeros(2)}
            result={'output_path':coarse,'encoder':'pcm_s16le','segments':[],
                    'verified_first_channel_pcm':root/'target.pcm'}
            import contextlib,io
            with contextlib.redirect_stdout(io.StringIO()):
                _,result,report=refine_rendered_audio(selected,{'reference_pcm':root/'reference.pcm'},
                    zero,result,out,Settings(anchor_step=.5,min_improvement=1/SR,max_rate_change=.02),
                    make_plots=False,protect_best=False)
            self.assertEqual(report['status'],'PCM_REFINED',report.get('reason'))
            self.assertEqual(result['output_path'].name,'target_PCM_SYNCED.wav')
            self.assertEqual(coarse.read_bytes(),before)

    def test_pcm_patches_preserve_bad_scene_and_render_identical_weights_on_all_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            reference=soundtrack(6.,471)
            target=np.r_[np.zeros(round(.006*SR),dtype=np.float32),reference[:-round(.006*SR)]]
            source=root/'source.wav'
            write_audio(source,np.stack([target,target*.5],axis=1),codec='pcm_s16le')
            original=source.read_bytes()
            _,pcm=wavfile.read(source)
            reference=np.clip(np.rint(reference*32768),-32768,32767).astype(np.int16)
            old=TimeMap.affine(6.,offset=-.004)
            layers=[{'reference':[lo,lo+.012,hi-.012,hi],'weight':[0.,1.,1.,0.],
                     'rate':1.,'offset':offset} for lo,hi,offset in [(.5,2.5,-.006),(3.5,5.5,.004)]]
            proposal=BlendMap(old,layers)
            settings=Settings(anchor_step=.5,min_improvement=1/SR,max_rate_change=.02,verification_pcm16=True)
            times=np.arange(.25,5.75,.125).tolist()
            final,rows,history=gate_patches(reference,pcm[:,0],old,proposal,[(settings,times)],settings,times)
            self.assertFalse(bad_rows(rows))
            self.assertTrue(any(row['bad'] for row in history))
            from pcm_core.micro import measure_window
            self.assertLess(abs(measure_window(reference,pcm[:,0],final,1.5,settings)['correction_sec']),1/SR)
            self.assertAlmostEqual(measure_window(reference,pcm[:,0],final,4.5,settings)['correction_sec'],
                                   measure_window(reference,pcm[:,0],old,4.5,settings)['correction_sec'],delta=1e-10)
            restored=TimeMap.from_dict(final.to_dict())
            probes=np.arange(0.,6.,.00137)
            np.testing.assert_array_equal(sample_map(pcm,final,probes,SR,16),sample_map(pcm,restored,probes,SR,16))
            work=root/'render';work.mkdir()
            result=render(probe(source),final,6.,work,settings,log=lambda _:None)
            _,written=wavfile.read(result)
            virtual=sample_map(pcm,final,np.arange(len(written))/SR,SR,16)
            expected=np.clip(np.rint(virtual),-32768,32767).astype(np.int16)
            self.assertLessEqual(np.max(np.abs(written.astype(float)-expected)),1.)
            self.assertEqual(source.read_bytes(),original)
            actual=written[:,0].astype(np.float32)/32768
            checks=verify_render(reference,pcm[:,0],actual,old,final,settings,target_times=times)
            checks+=verify_reference_grid(reference,pcm[:,0],old,final,settings,times,rendered=actual)
            self.assertFalse(bad_rows(checks))

    def test_exact_replacement_and_rollback_preserve_the_established_curve(self):
        t=np.array([0.,1.,3.,5.,8.])
        a=np.array([0.,-.004,-.006,-.003,0.])
        base=TimeMap([Piece(t,t+a,[{'target':t.tolist(),'delta':a.tolist()}])])
        u=np.array([0.,2.,4.,6.,8.])
        b=np.array([0.,-.007,-.002,-.008,0.])
        proposed=TimeMap([Piece(u,u+b,[{'target':u.tolist(),'delta':b.tolist()}])])
        result=replace_map(base,proposed)
        times=np.arange(0.,8.,.00137)
        np.testing.assert_allclose(result.forward(times),proposed.forward(times),atol=1e-12)
        np.testing.assert_allclose(TimeMap.from_dict(result.to_dict()).forward(times),
                                   proposed.forward(times),atol=1e-12)
        self.assertEqual(result.pieces[0].corrections[0],base.pieces[0].corrections[0])

    def test_native_file_keeps_old_good_scene_and_adds_new_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            t=np.array([0.,1.,5.,6.,8.,9.,13.,14.])
            def mapping(early,late):
                d=np.array([0.,early,early,0.,0.,late,late,0.])
                return TimeMap([Piece(t,t+d,[{'target':t.tolist(),'delta':d.tolist()}])])
            true=mapping(-.006,-.006)
            old=mapping(-.004,-.006)
            proposal=mapping(-.006,0.)
            reference=soundtrack(14.,811)
            target=warp(reference,true.forward(np.arange(len(reference))/SR)*SR)
            source=root/'source.wav'
            write_audio(source,target,codec='pcm_s16le')
            _,pcm=wavfile.read(source)
            reference=np.clip(np.rint(reference*32768),-32768,32767).astype(np.int16)
            settings=Settings(anchor_step=.5,min_improvement=1/SR,max_rate_change=.02,verification_pcm16=True)
            times=np.arange(.25,13.75,.25).tolist()
            fixed=np.arange(.25,13.75,.0625).tolist()
            final,rows,_=choose_local(reference,pcm,old,proposal,[(settings,times)],settings,fixed,log=lambda _:None)
            self.assertFalse(bad_rows(rows))
            self.assertAlmostEqual(float(final.forward([3.])[0])-3.,-.006,delta=1/SR)
            self.assertAlmostEqual(float(final.forward([11.])[0])-11.,-.006,delta=1/SR)
            work=root/'render';work.mkdir()
            written=render(probe(source),final,14.,work,settings,log=lambda _:None)
            actual=decode(probe(written),work/'check.f32',first_channel=True)
            try:
                checks=verify_render(reference,pcm,actual,old,final,settings,target_times=times)
                checks+=verify_reference_grid(reference,pcm,old,final,settings,fixed,rendered=actual)
                self.assertFalse(bad_rows(checks))
            finally:
                actual._mmap.close()

    def test_fresh_common_delay_proposal_without_old_folders(self):
        reference=soundtrack(4.,353)
        target=np.r_[np.zeros(round(.006*SR),dtype=np.float32),reference[:-round(.006*SR)]]
        base=TimeMap.affine(4.)
        settings=Settings(anchor_step=.5,min_improvement=1/SR,max_rate_change=.02)
        calls=[]
        def planner(ref,tgt,mapping,config,report,progress,classic=False):
            calls.append(classic)
            return mapping,[],[],config,[],[(config,np.arange(.25,3.75,.25).tolist())]
        with tempfile.TemporaryDirectory() as tmp:
            report={}
            final,_,_,_,_,_,protection=build_best_plan(reference,target,base,settings,Path(tmp),report,
                lambda *args,**kwargs:QuietProgress(),planner)
            self.assertEqual(calls,[True,False])
            self.assertTrue(protection.additional_confirmed)
            self.assertAlmostEqual(float(final.forward([2.])[0])-2.,-.006,delta=1/SR)
            self.assertTrue((Path(tmp)/'BASELINE_PCM_PLAN.json').is_file())


if __name__=='__main__':
    unittest.main()
