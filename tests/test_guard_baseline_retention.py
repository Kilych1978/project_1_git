"""A rejected coarse proposal must not be revived during render preparation."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common import Progress,SelectedTracks
from render_guard import run_render_aware_safety_loop
from safety import build_zero_sync_map,_validation_weights
from scanner import probe_audio_file
from .helpers import soundtrack,write_audio


class GuardBaselineRetentionTests(unittest.TestCase):
    def test_render_retains_explicitly_rejected_coarse_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'original.wav'
            write_audio(source,soundtrack(4.))
            before=source.read_bytes();info=probe_audio_file(source)
            selected=SelectedTracks(info,info)
            zero=build_zero_sync_map(info.duration)
            proposed={**zero,'offset':np.full_like(zero['offset'],.006),
                      'reference_time':zero['target_time']+.006}
            baseline=[{'target_time':2.,'before_offset':.006,'before_correlation':.99}]
            def verify(selected,analysis,path,folder,**kwargs):
                self.assertEqual(path,source)
                pcm=folder/'first_channel.pcm';pcm.write_bytes(b'')
                return {'validation_points':[{'target_time':2.,'after_offset':.006,'after_correlation':.99}],
                        'rendered_first_channel_pcm':pcm}
            with patch('render_guard.build_non_regressive_sync_map') as rebuild, \
                 patch('render_guard.render_target_with_sync_map') as render, \
                 patch('render_guard.verify_rendered_sync',side_effect=verify), \
                 patch('render_guard.compare_micro_rendered_against_original',return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()):
                final,result,report,_=run_render_aware_safety_loop(selected,{},proposed,zero,baseline,[],
                    root/'result',{},Progress())
            rebuild.assert_not_called();render.assert_not_called()
            self.assertEqual(result['output_path'],source)
            self.assertFalse(report['output_created']);self.assertTrue(report['fallback_to_baseline'])
            self.assertTrue(np.all(final['offset']==0))
            self.assertFalse(list((root/'result').rglob('*.wav')))
            self.assertEqual(source.read_bytes(),before)

    def test_validation_without_optional_absolute_offset_uses_measured_errors(self):
        rows=[{'target_time':float(t),'before_error':.2,'after_error':.001,
               'bidir_status':'CONFIRMED_MAP','predicted_offset':-.006,
               'bidir_left_correlation':.99,'bidir_right_correlation':.99,'bidir_disagreement':0.}
              for t in (10,20,30)]
        times,weights,_,_,already_synced=_validation_weights(rows)
        np.testing.assert_array_equal(times,[10.,20.,30.])
        self.assertTrue(np.all(weights==1));self.assertFalse(already_synced)


if __name__=='__main__':unittest.main()
