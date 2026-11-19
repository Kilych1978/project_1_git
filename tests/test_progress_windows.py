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
from progress_display import words,scope_lines
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

    def test_terminal_shows_readable_position_and_keeps_single_line_within_width(self):
        class TTY(io.StringIO):
            def isatty(self):return True
        output=TTY()
        with contextlib.redirect_stdout(output),patch('shutil.get_terminal_size',return_value=os.terminal_size((80,24))):
            p=_ProgressCallback('Проверка | проход 2/11 | сетка 1/2',duration=5252.01,
                                target_times=[.2,5251.8])
            p.window(4056.476,4056.601);p(8114,10504);p.finish()
        records=output.getvalue().split('\r')[1:]
        self.assertTrue(records)
        for record in records:
            line=record.split('\n')[0]
            self.assertLessEqual(len(line),79)
        self.assertIn('Сейчас: 1 ч 07 мин 36 с',output.getvalue())
        self.assertIn('Длительность дорожки: 1 ч 27 мин 32 с',output.getvalue())
        self.assertIn('Область: весь фильм (контрольные точки)',output.getvalue())
        self.assertIn('Проход: 2, максимум 11 | Сетка: 1 из 2',output.getvalue())
        self.assertNotIn('→',output.getvalue())
        self.assertNotIn('4056.',output.getvalue())
        # Diagnostic millisecond formatting is deliberately unchanged.
        self.assertEqual(timecode(59.9996),'00:01:00,000')
        self.assertEqual(timecode(3612.098),'01:00:12,098')

    def test_local_points_are_described_without_inventing_a_contiguous_chunk(self):
        lines=scope_lines(5252.,[912.098,913.,1090.,1140.1])
        self.assertEqual(lines[1],'Контрольные точки: с 15 мин 12 с по 19 мин 01 с')
        self.assertNotIn('весь фильм',' '.join(lines))
        self.assertEqual(words(33.2),'33 с')
        self.assertEqual(words(97.8,'nearest'),'1 мин 38 с')

    def test_audio_render_progress_uses_seconds_position_and_real_duration(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            p=_ProgressCallback('Запись всех каналов',seconds=True)
            p(4056.5,5252.01)
        self.assertIn('Сейчас: 1 ч 07 мин 36 с',out.getvalue())
        self.assertIn('Длительность дорожки: 1 ч 27 мин 32 с',out.getvalue())


if __name__=='__main__':unittest.main()
