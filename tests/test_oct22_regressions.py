"""Movie-duration roundoff and retention of a verified PCM baseline."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
from scipy.interpolate import PchipInterpolator

from common import Progress,SelectedTracks
from pcm_core.config import Settings
from pcm_core.model import Piece,TimeMap
from pcm_core.guard import cancel_unsafe_components
from pcm_core.choice import choose_local
from pcm_core.micro import collect
from pcm_core.micro import Measurement
from pcm_core.blend import BlendMap,affine_patches,sample_map
from pcm_core.guard import _signature
from pcm_core.render import render as render_pcm
from pcm_core.audio import probe
from pcm_refiner import refine_rendered_audio
from render_guard import _discard_rejected_files,_publish_guard_output,run_render_aware_safety_loop
from renderer import copy_target_without_reencode
from scanner import probe_audio_file
from .helpers import SR,soundtrack,write_audio


class Oct22RegressionTests(unittest.TestCase):
    def test_coherent_band_anchors_can_bridge_one_unknown_but_not_an_edit(self):
        times=[1.,1.125,1.25,1.375,1.5]
        rows=[Measurement(t,t,-.006-.001*t,.99,.2,1e-6,3,1,'MEASURED','raw') for t in times]
        rows[2]=Measurement(1.25,1.25,None,status='UNVERIFIABLE')
        rows[3].method='multiband'
        base=TimeMap.affine(3.)
        self.assertIs(affine_patches(rows,base,SR),base)
        proposal=affine_patches(rows,base,SR,coherent_gaps=True)
        self.assertIsInstance(proposal,BlendMap)
        self.assertEqual(proposal.layers[0]['measured_anchors'],4)
        self.assertAlmostEqual(proposal.layers[0]['rate'],.999,places=12)
        rows[2].status='INCONSISTENT'
        self.assertIs(affine_patches(rows,base,SR,coherent_gaps=True),base)
        rows[2].status='UNVERIFIABLE';rows[3].score=.93
        self.assertIs(affine_patches(rows,base,SR,coherent_gaps=True),base)
        self.assertIsInstance(affine_patches(rows,base,SR,coherent_gaps=True,
                                           max_anchor_gap=.501,min_band_score=.85),BlendMap)

    def test_nested_patch_changes_remain_visible_to_validation_and_cache(self):
        base=TimeMap.affine(4.)
        a={'reference':[.5,.512,1.488,1.5],'weight':[0.,1.,1.,0.],'rate':1.,'offset':-.006}
        b={'reference':[2.5,2.512,3.488,3.5],'weight':[0.,1.,1.,0.],'rate':1.,'offset':-.004}
        old=BlendMap(base,[a]);new=BlendMap(old,[b])
        self.assertGreater(new.extent(base,[1.]),.0059)
        self.assertEqual(new.extent(old,[1.]),0.)
        altered=BlendMap(BlendMap(base,[{**a,'offset':-.007}]),[b])
        self.assertNotEqual(_signature(new,1.,Settings()),_signature(altered,1.,Settings()))
        restored=TimeMap.from_dict(new.to_dict())
        np.testing.assert_allclose(restored.sample(np.arange(4*SR,dtype=float),[1.,3.],SR,16),
                                   new.sample(np.arange(4*SR,dtype=float),[1.,3.],SR,16),atol=0,rtol=0)
        reduced,removed=cancel_unsafe_components(base,new,[1.],protected_half_window=.04)
        self.assertTrue(removed)
        source=np.arange(4*SR,dtype=float)
        np.testing.assert_allclose(sample_map(source,reduced,np.array([.97,1.,1.03]),SR,16),
                                   sample_map(source,base,np.array([.97,1.,1.03]),SR,16),atol=0,rtol=0)
        self.assertGreater(reduced.extent(base,[3.]),.0039)
        kept,_=cancel_unsafe_components(old,new,[1.],protected_half_window=.04)
        np.testing.assert_allclose(sample_map(source,kept,np.array([1.]),SR,16),
                                   sample_map(source,old,np.array([1.]),SR,16),atol=0,rtol=0)

    def test_two_pcm_layers_write_the_same_s16_samples_on_all_six_channels(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source.wav';out=root/'render';out.mkdir()
            x=soundtrack(4.,812)
            channels=np.stack([x,x*.7,x*.3,x*.1,x*.4,x*.6],axis=1)
            write_audio(source,channels,codec='pcm_s16le',layout='5.1')
            before=source.read_bytes();samples=wavfile.read(source)[1]
            a={'reference':[.5,.512,2.988,3.],'weight':[0.,1.,1.,0.],'rate':.999,'offset':-.006}
            b={'reference':[1.,1.012,2.488,2.5],'weight':[0.,1.,1.,0.],'rate':.9991,'offset':-.0061}
            c={'reference':[1.5,1.512,1.788,1.8],'weight':[0.,1.,1.,0.],'rate':.9991,'offset':-.0062}
            mapping=BlendMap(BlendMap(BlendMap(TimeMap.affine(4.),[a]),[b]),[c]).validate(.02)
            path=render_pcm(probe(source),mapping,4.,out,Settings(anchor_step=.125),log=lambda _:None)
            rate,actual=wavfile.read(path)
            expected=np.clip(np.rint(sample_map(samples,mapping,np.arange(len(samples))/SR,SR,16)),-32768,32767).astype('<i2')
            self.assertEqual(rate,SR);self.assertEqual(actual.shape,samples.shape)
            self.assertLessEqual(np.max(np.abs(actual.astype(int)-expected.astype(int))),1)
            self.assertEqual(source.read_bytes(),before)

    def test_no_gain_creates_only_a_report_and_references_the_original(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'original.wav';write_audio(source,soundtrack(2.))
            before=source.read_bytes();output=root/'result';output.mkdir()
            result=copy_target_without_reencode(probe_audio_file(source),{'pcm_fallback':True},output)
            self.assertEqual(result['output_path'],source)
            self.assertFalse(result['output_created']);self.assertTrue(result['uses_original_directly'])
            self.assertFalse(list(output.rglob('*.wav')))
            self.assertEqual(source.read_bytes(),before)
            self.assertIn('Новый аудиофайл не создан',result['report_path'].read_text())

    def test_all_rejected_passes_and_safe_but_unimproved_render_leave_no_audio_copy(self):
        for failing in (True,False):
            with self.subTest(regression=failing),tempfile.TemporaryDirectory() as td:
                root=Path(td);source=root/'original.wav';write_audio(source,soundtrack(4.))
                before=source.read_bytes();output=root/'result';output.mkdir()
                info=probe_audio_file(source);info.duration=120.  # Decoder is stubbed; allow all quarantine passes.
                selected=SelectedTracks(info,info)
                t=np.array([0.,120.]);mapping={'target_time':t,'reference_time':t+.006,
                                            'offset':np.full(2,.006),'confidence':np.ones(2)}
                baseline=[{'target_time':2.,'before_offset':.006,'before_correlation':.99}]
                def render(selected,mapping,directory,progress,codec_plan):
                    path=directory/'candidate.wav';path.write_bytes(b'rejected generated audio')
                    return {'output_path':path,'encoder':'pcm_s16le','segments':[],'codec_plan':codec_plan}
                def verify(selected,analysis,path,directory,**kwargs):
                    pcm=directory/'decoded.pcm';pcm.write_bytes(b'pcm')
                    offset=.006 if path==source or not failing else .020
                    return {'validation_points':[{'target_time':2.,'after_offset':offset,'before_correlation':.99}],
                            'rendered_first_channel_pcm':pcm}
                with patch('render_guard.build_non_regressive_sync_map',return_value=(mapping,{})), \
                     patch('render_guard.build_apply_map',return_value=mapping), \
                     patch('render_guard.render_target_with_sync_map',side_effect=render), \
                     patch('render_guard.verify_rendered_sync',side_effect=verify), \
                     patch('render_guard.compare_micro_rendered_against_original',return_value={}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    _,result,report,_=run_render_aware_safety_loop(selected,{},mapping,mapping,baseline,[],
                                                               output,{},Progress())
                self.assertTrue(report['fallback_to_baseline'])
                self.assertEqual(result['output_path'],source)
                self.assertFalse(report['output_created'])
                self.assertEqual(report['passes'],4 if failing else 2)
                self.assertFalse(list(output.rglob('*.wav')))
                self.assertEqual(source.read_bytes(),before)
                self.assertTrue((output/'RENDER_AWARE_NON_REGRESSION.json').exists())

    def test_unchanged_pcm_report_does_not_call_the_source_a_new_output(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'original.wav';write_audio(source,soundtrack(2.),codec='pcm_s16le')
            before=source.read_bytes();samples=wavfile.read(source)[1];samples.tofile(root/'input.pcm')
            output=root/'result';output.mkdir();info=probe_audio_file(source)
            base=TimeMap.affine(2.);settings=Settings(anchor_step=.5)
            t=np.array([0.,2.]);mapping={'target_time':t,'reference_time':t.copy(),'offset':t*0}
            baseline={'output_path':source,'encoder':'copy','segments':[],'copied_without_reencode':True,
                      'verified_first_channel_pcm':root/'input.pcm'}
            with patch('pcm_refiner.plan_refinement',return_value=(base,[],[],settings,[],[])), \
                 contextlib.redirect_stdout(io.StringIO()):
                _,result,report=refine_rendered_audio(SelectedTracks(info,info),{'reference_pcm':root/'input.pcm'},
                    mapping,baseline,output,settings,make_plots=False,protect_best=False)
            saved=json.loads((output/'PCM_MICRO_REPORT.json').read_text())
            self.assertEqual(report['status'],'BASELINE_KEPT')
            self.assertIsNone(saved['output_file']);self.assertFalse(saved['output_created'])
            self.assertEqual(saved['retained_file'],str(source))
            self.assertEqual(result['output_path'],source)
            self.assertFalse(list(output.rglob('*.wav')));self.assertEqual(source.read_bytes(),before)
            self.assertIn('Новый аудиофайл не создан',(output/'PCM_MICRO_REPORT_RU.txt').read_text(encoding='utf-8-sig'))

    def test_report_roundoff_pair_at_40_minutes_is_not_a_real_time_reversal(self):
        # Numeric excerpt from the failing rebuild; no audio or private paths.
        x=np.array([2428.956305555555,2428.962388888888,
                    2428.9623888888887,2428.9669999999996])
        y=np.array([2428.970161958543,2428.976218186013,
                    2428.976218186013,2428.9808039358095])
        model=TimeMap([Piece(x,y)]).validate(.08)
        self.assertEqual(len(model.pieces[0].target),3)
        np.testing.assert_allclose(model.forward(x),y,atol=1e-11,rtol=0)
        self.assertEqual(model.pieces[0].target[0],x[0])
        self.assertEqual(model.pieces[0].target[-1],x[-1])
        times=np.linspace(x[0],x[-1],1001,endpoint=False)
        np.testing.assert_allclose(model.inverse(model.forward(times)),times,atol=1e-11,rtol=0)

    def test_roundoff_merge_keeps_the_exact_cubic_and_real_edit_checks(self):
        t=np.array([2400.,2401.,2401.+np.spacing(2401.),2402.,2403.])
        d=np.array([0.,-.003,-.003,-.006,0.])
        curve={'target':t.tolist(),'delta':d.tolist()}
        expected=PchipInterpolator(t,d)
        model=TimeMap([Piece(t,t+d,[curve])]).validate(.08)
        times=np.linspace(2400.,2403.,12001)
        np.testing.assert_allclose(model.forward(times),times+expected(times),atol=1e-11,rtol=0)
        restored=TimeMap.from_dict(model.to_dict())
        np.testing.assert_allclose(restored.forward(times),model.forward(times),atol=1e-11,rtol=0)
        for bad in ([0.,1.,1.],[0.,1.,.999]):
            with self.assertRaises(ValueError):TimeMap([Piece(np.array([0.,1.,2.]),np.array(bad))])
        # An actual 10-ms jump across close nodes must not be silently merged.
        x=np.array([2400.,2401.,2401.+np.spacing(2401.),2402.])
        y=np.array([2400.,2401.,2401.01,2402.01])
        with self.assertRaises(ValueError):TimeMap([Piece(x,y)]).validate(.08)

    def test_optional_rollback_exception_does_not_escape_selection(self):
        x=soundtrack(4.,814)
        y=np.r_[np.zeros(round(.006*SR)),x[:-round(.006*SR)]]
        base=TimeMap.affine(4.)
        wrong=TimeMap.affine(4.,offset=.006)
        settings=Settings(anchor_step=.5,min_improvement=1/SR,max_rate_change=.02)
        with patch('pcm_core.choice.cancel_unsafe_components',side_effect=ValueError('invalid optional transition')):
            result,_,history=choose_local(x,y,base,wrong,[(settings,[1.,2.,3.])],settings,[],log=lambda _:None)
        np.testing.assert_array_equal(result.forward([1.,2.,3.]),[1.,2.,3.])
        self.assertTrue(any(row.get('status')=='INVALID_OPTIONAL_ROLLBACK' for row in history))

    def test_failed_optional_planner_still_writes_and_verifies_the_baseline_wav(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            x=soundtrack(4.,819)
            y=np.r_[np.zeros(round(.006*SR)),x[:-round(.006*SR)]]
            rp,tp=root/'reference.wav',root/'target.wav'
            write_audio(rp,x,codec='pcm_s16le');write_audio(tp,y,codec='pcm_s16le')
            before=tp.read_bytes()
            r=wavfile.read(rp)[1];t=wavfile.read(tp)[1]
            r.tofile(root/'reference.pcm');t.tofile(root/'target.pcm')
            selected=SelectedTracks(probe_audio_file(rp),probe_audio_file(tp))
            settings=Settings(anchor_step=.5,min_improvement=1/SR,max_rate_change=.02)
            def planner(ref,target,base,config,report,progress,classic=False):
                if not classic:raise ValueError('invalid optional movie map')
                measurements=collect(ref,target,base,config,log=lambda _:None)
                return TimeMap.affine(4.,offset=-.006),[],measurements,config,[],[(config,[.5,1.,1.5,2.,2.5,3.])]
            out=root/'result';out.mkdir()
            times=np.array([0.,4.]);coarse={'target_time':times,'reference_time':times.copy(),'offset':times*0}
            baseline={'output_path':tp,'encoder':'copy','segments':[],
                      'copied_without_reencode':True,'verified_first_channel_pcm':root/'target.pcm'}
            with patch('pcm_refiner.plan_refinement',side_effect=planner),contextlib.redirect_stdout(io.StringIO()):
                _,result,report=refine_rendered_audio(selected,{'reference_pcm':root/'reference.pcm'},
                    coarse,baseline,out,settings,make_plots=False)
            self.assertEqual(report['status'],'PCM_REFINED',report.get('reason'))
            self.assertEqual(report['optional_recovery'],'fresh_verified_pcm_baseline')
            self.assertTrue((out/'PCM_OPTIONAL_FAILURE.txt').is_file())
            self.assertTrue((out/'FAILED_OPTIONAL_PLAN.json').is_file())
            rows=[row for row in report['actual_output_windows'] if row['micro_changed'] and row['status']=='VERIFIED']
            self.assertGreaterEqual(len(rows),3)
            self.assertLess(max(abs(row['residual_sec']) for row in rows),1/SR)
            self.assertEqual(tp.read_bytes(),before)
            self.assertNotEqual(result['output_path'],tp)

    def test_failed_extra_phase_keeps_the_prior_pcm_gain_in_the_written_wav(self):
        import pcm_best
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);x=soundtrack(4.,824)
            y=np.r_[np.zeros(round(.006*SR)),x[:-round(.006*SR)]]
            rp,tp=root/'reference.wav',root/'target.wav'
            write_audio(rp,x,codec='pcm_s16le');write_audio(tp,y,codec='pcm_s16le')
            before=tp.read_bytes();wavfile.read(rp)[1].tofile(root/'r.pcm');wavfile.read(tp)[1].tofile(root/'t.pcm')
            info=SelectedTracks(probe_audio_file(rp),probe_audio_file(tp))
            settings=Settings(anchor_step=.5,min_improvement=1/SR,max_rate_change=.02)
            def planner(ref,target,base,config,report,progress,classic=False):
                return TimeMap.affine(4.,offset=-.003),[],[],config,[],[(config,[.5,1.,1.5,2.,2.5,3.])]
            original_collect=pcm_best.collect;original_patches=pcm_best.affine_patches
            def collection(*args,**kwargs):
                rows=original_collect(*args,**kwargs)
                for row in rows:row.method='multiband'
                return rows
            def patches(*args,**kwargs):
                if kwargs.get('coherent_gaps'):raise ValueError('optional phase failure')
                return original_patches(*args,**kwargs)
            out=root/'result';out.mkdir();t=np.array([0.,4.])
            coarse={'target_time':t,'reference_time':t.copy(),'offset':t*0}
            baseline={'output_path':tp,'encoder':'copy','segments':[],'copied_without_reencode':True,
                      'verified_first_channel_pcm':root/'t.pcm'}
            with patch('pcm_refiner.plan_refinement',side_effect=planner), \
                 patch('pcm_best.collect',side_effect=collection),patch('pcm_best.affine_patches',side_effect=patches), \
                 contextlib.redirect_stdout(io.StringIO()):
                _,result,report=refine_rendered_audio(info,{'reference_pcm':root/'r.pcm'},coarse,
                    baseline,out,settings,make_plots=False)
            self.assertEqual(report['status'],'PCM_REFINED',report['reason'])
            self.assertTrue('phase_failure' in report,{k:report[k] for k in ('reason','optional_failure','phase_choice_audit') if k in report})
            self.assertEqual(report['phase_failure']['type'],'ValueError')
            self.assertNotIn('optional_recovery',report)
            rows=[r for r in report['actual_output_windows'] if r['micro_changed'] and r['status']=='VERIFIED']
            self.assertGreaterEqual(len(rows),3)
            self.assertLess(np.median([abs(r['residual_sec']) for r in rows]),.0031)
            self.assertTrue(report['output_created']);self.assertTrue(result['output_created'])
            self.assertEqual(tp.read_bytes(),before)

    def test_selected_output_is_at_root_and_rejected_files_never_include_source(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'original.wav';source.write_bytes(b'original audio')
            output=root/'result';guard=output/'_render_guard';first=guard/'pass_1';first.mkdir(parents=True)
            bad=first/'rejected.wav';bad.write_bytes(b'rejected audio');pcm=first/'decode.pcm';pcm.write_bytes(b'pcm')
            result={'output_path':bad,'verified_first_channel_pcm':pcm}
            removed=_discard_rejected_files(result,guard,source)
            self.assertEqual(len(removed),2);self.assertFalse(bad.exists());self.assertFalse(pcm.exists())
            _discard_rejected_files({'output_path':source},guard,source)
            selected=guard/'fallback_baseline'/'selected.wav';selected.parent.mkdir();selected.write_bytes(b'accepted audio')
            result={'output_path':selected}
            _publish_guard_output(result,output,source)
            self.assertEqual(result['output_path'],output/'selected.wav')
            self.assertEqual(result['output_path'].read_bytes(),b'accepted audio')
            self.assertFalse(selected.exists());self.assertEqual(source.read_bytes(),b'original audio')
            _publish_guard_output({'output_path':source},output,source)
            another=guard/'pass_2'/'selected.wav';another.parent.mkdir();another.write_bytes(b'new audio')
            with self.assertRaises(RuntimeError):_publish_guard_output({'output_path':another},output,source)
            self.assertEqual((output/'selected.wav').read_bytes(),b'accepted audio')


if __name__=='__main__':unittest.main()
