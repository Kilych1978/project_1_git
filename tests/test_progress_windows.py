import contextlib
import io
import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from pcm_refiner import _ProgressCallback
from pcm_core.progress import slice_progress,timecode
from pcm_core.guard import verify_candidate_pcm
from pcm_core.choice import verify_reference_grid
from pcm_core.model import TimeMap
from pcm_core.config import Settings
from .helpers import soundtrack,SR


class Recorder:
    def __init__(self):self.windows=[];self.counts=[]
    def window(self,start,end,cached=False):self.windows.append((start,end,cached))
    def __call__(self,done,total):self.counts.append((done,total))


class ProgressWindowTests(unittest.TestCase):
    def test_offset_counter_preserves_absolute_reference_window_and_cache_flag(self):
        r=soundtrack(3.,891);pcm=np.clip(np.rint(r*32768),-32768,32767).astype(np.int16)
        base=TimeMap.affine(3.);final=TimeMap.affine(3.,offset=.007)
        s=Settings(anchor_step=.125);record=Recorder();callback=slice_progress(record,10,30)
        cache={};before={}
        for repeat in range(2):
            rows=verify_candidate_pcm(pcm,pcm,base,final,s,target_times=[1.25],
                progress=callback,row_cache=cache,baseline_cache=before)
            a,b,cached=record.windows[-1]
            self.assertAlmostEqual((a+b)/2,1.257)
            self.assertAlmostEqual(b-a,rows[0]['verification_window_sec'])
            self.assertEqual(cached,bool(repeat))
            self.assertEqual(record.counts[-1],(11,30))
        fixed=Recorder()
        verify_reference_grid(pcm,pcm,base,final,s,[1.5],progress=fixed)
        a,b,_=fixed.windows[-1];self.assertAlmostEqual((a+b)/2,1.5)

    def test_terminal_shows_milliseconds_arrow_and_keeps_single_line_within_width(self):
        class TTY(io.StringIO):
            def isatty(self):return True
        output=TTY()
        with contextlib.redirect_stdout(output),patch('shutil.get_terminal_size',return_value=os.terminal_size((80,24))):
            p=_ProgressCallback('Проверка | проход 2/3 | сетка 1/2')
            p.window(912.098,912.223);p(500,1000);p.finish()
        records=output.getvalue().split('\r')[1:]
        self.assertTrue(records)
        for record in records:
            line=record.split('\n')[0]
            self.assertLessEqual(len(line),79)
        self.assertIn('00:15:12,098 → 00:15:12,223',output.getvalue())
        self.assertEqual(timecode(59.9996),'00:01:00,000')
        self.assertEqual(timecode(3612.098),'01:00:12,098')


if __name__=='__main__':unittest.main()
