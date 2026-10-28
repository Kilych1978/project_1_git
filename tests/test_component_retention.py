"""A movie must retain safe local PCM when a distant island exhausts the gate."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from pcm_core.blend import BlendMap,gate_patches,sample_map
from pcm_core.config import Settings
from pcm_core.guard import _signature
from pcm_core.model import TimeMap
from pcm_core.audio import probe
from pcm_core.render import render
from .helpers import SR,soundtrack,write_audio


class ComponentRetentionTests(unittest.TestCase):
    def model(self):
        layer={'reference':[.5,.512,2.,2.01,2.6,2.612,4.488,4.5],
               'weight':[0.,1.,1.,0.,0.,1.,1.,0.],
               'rate':1.,'offset':-.006,'measured_anchors':20,'fit_error_sec':0.}
        return BlendMap(TimeMap.affine(6.),[layer])

    def test_support_islands_are_sample_identical_and_keep_local_cache(self):
        mapping=self.model();split=BlendMap(mapping.base,mapping.support_components()).validate(.02)
        self.assertEqual(len(split.layers),2)
        x=np.stack([soundtrack(6.,890)*v for v in [1.,.7,.1]],axis=1)
        times=np.arange(len(x))/SR
        np.testing.assert_array_equal(sample_map(x,mapping,times,SR,16),sample_map(x,split,times,SR,16))
        settings=Settings(anchor_step=.125)
        self.assertEqual(_signature(mapping,1.,settings),_signature(split,1.,settings))
        self.assertEqual(_signature(mapping,3.5,settings),_signature(split,3.5,settings))
        reduced,removed=mapping.discard_failed_components([{'reference_time':3.5,'window_sec':.125}])
        self.assertEqual(len(removed),1)
        self.assertEqual(_signature(mapping,1.,settings),_signature(reduced,1.,settings))
        self.assertNotEqual(_signature(mapping,3.5,settings),_signature(reduced,3.5,settings))

    def test_bad_window_removes_every_island_it_intersects(self):
        mapping=self.model()
        # A broad comparison can cross the zero plateau; neither side is safe
        # to retain on the evidence of that failed window alone.
        reduced,removed=mapping.discard_failed_components([{'reference_time':2.3,'window_sec':1.}])
        self.assertIs(reduced,mapping.base)
        self.assertEqual(len(removed),2)

    def test_exhausted_gate_retains_and_rechecks_distant_good_pcm_in_native_wav(self):
        base=TimeMap.affine(8.)
        good={'reference':[.4,.412,2.088,2.1],'weight':[0.,1.,1.,0.],
              'rate':1.,'offset':-.006}
        bad={'reference':[3.,3.012,6.488,6.5],'weight':[0.,1.,1.,0.],
             'rate':1.,'offset':.009}
        proposal=BlendMap(base,[good,bad]);settings=Settings(anchor_step=.125,min_improvement=1/SR)
        calls=[];reported_bad=[]
        def verify(reference,target,base,candidate,config,**kwargs):
            calls.append(candidate)
            rows=[{'target_time':t,'reference_time':t,'micro_changed':True,
                   'baseline_residual_sec':-.006,'residual_sec':0.,'status':'VERIFIED',
                   'verification_window_sec':.015625} for t in [.7,1.,1.3]]
            # A persistent bad boundary walks along one distant clock patch.
            # The first six local masks cannot erase that entire patch.
            for layer in getattr(candidate,'layers',[]):
                if layer['offset']<=0:continue
                for t in np.arange(3.05,6.45,.08):
                    if candidate.weight(layer,[t])[0]>.5:
                        reported_bad.append(float(t))
                        rows.append({'target_time':float(t),'reference_time':float(t),
                                     'micro_changed':True,'baseline_residual_sec':0.,
                                     'residual_sec':.009,'status':'REGRESSION',
                                     'verification_window_sec':.015625})
                        return rows
            return rows
        with patch('pcm_core.guard.verify_candidate_pcm',side_effect=verify), \
             patch('pcm_core.choice.verify_reference_grid',return_value=[]):
            result,rows,history=gate_patches(np.zeros(8*SR),np.zeros(8*SR),base,proposal,
                                            [(settings,[.7,1.,1.3])],settings,[])
        self.assertGreater(len(calls),6)
        self.assertIsNot(result,base)
        self.assertTrue(any(r.get('discarded_components') for r in history))
        self.assertTrue(any(r.get('failed_windows') for r in history))
        self.assertFalse(any(row['status']=='REGRESSION' for row in rows))
        self.assertTrue(any(layer['offset']==-.006 for layer in result.layers))
        failed=reported_bad[-1]
        for layer in result.layers:
            self.assertFalse(np.any(result.weight(layer,np.linspace(failed-.007,failed+.007,31))>0))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source.wav';out=root/'out';out.mkdir()
            x=soundtrack(8.,893);samples=np.stack([x*v for v in [1.,.6,.2,.1,.4,.8]],axis=1)
            write_audio(source,samples,codec='pcm_s16le',layout='5.1');before=source.read_bytes()
            original=wavfile.read(source)[1]
            path=render(probe(source),result,8.,out,settings,log=lambda _:None)
            rate,actual=wavfile.read(path)
            self.assertEqual(rate,SR);self.assertEqual(actual.shape,original.shape)
            lo,hi=round((failed-.007)*SR),round((failed+.007)*SR)
            np.testing.assert_array_equal(actual[lo:hi],original[lo:hi])
            np.testing.assert_array_equal(actual[round(.7*SR):round(1.3*SR)],original[round(.706*SR):round(1.306*SR)])
            self.assertEqual(source.read_bytes(),before)


if __name__=='__main__':unittest.main()
