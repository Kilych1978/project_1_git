import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import renderer
from renderer import verify_rendered_sync


class BaselineDurationTests(unittest.TestCase):
    def call(self,root,actual,reference,baseline=False,expected=None):
        source=root/'source.ac3'
        source.write_bytes(b'Original source placeholder: decoding is mocked in this duration-only test.')
        output=source if baseline else root/'new_render.wav'
        selected=SimpleNamespace(reference=SimpleNamespace(duration=reference),
                                 target=SimpleNamespace(path=source,duration=5247.264))
        info=SimpleNamespace(path=output,duration=actual,codec_name='ac3' if baseline else 'pcm_s16le',
                             sample_rate=48000,channels=2)
        envelope={'values':np.zeros(4),'hop_sec':.01}
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(renderer,'probe_audio_file',return_value=info))
            stack.enter_context(patch.object(renderer,'decode_first_channel_stream',return_value={'duration':actual}))
            stack.enter_context(patch.object(renderer,'build_envelope_from_pcm',return_value=envelope))
            stack.enter_context(patch.object(renderer,'validate_sync_map',return_value=[]))
            return verify_rendered_sync(selected,{'reference_envelope':envelope},output,root,
                                        expected_output_duration=expected)

    def test_original_shorter_baseline_is_retained_without_pretending_duration_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);result=self.call(root,5247.264,5252.010666666667,baseline=True)
            self.assertFalse(result['duration_ok'])
            self.assertFalse(result['reference_duration_ok'])
            self.assertFalse(result['duration_check_enforced'])
            self.assertTrue(result['is_original_baseline'])
            self.assertAlmostEqual(result['duration_difference'],-4.746666666667,places=8)
            data=json.loads((root/renderer.FINAL_RENDER_VALIDATION_JSON).read_text())
            self.assertFalse(data['duration_check']);self.assertFalse(data['duration_check_enforced'])
            self.assertEqual(data['expected_duration_basis'],'reference')
            self.assertEqual(data['expected_output_duration'],5252.010666666667)
            self.assertIn('NOT ENFORCED',(root/renderer.FINAL_RENDER_VALIDATION_REPORT).read_text())

    def test_new_file_with_the_same_short_duration_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(RuntimeError):self.call(root,5247.264,5252.010666666667)
            data=json.loads((root/renderer.FINAL_RENDER_VALIDATION_JSON).read_text())
            self.assertTrue(data['duration_check_enforced']);self.assertFalse(data['is_original_baseline'])
            self.assertFalse(data['reference_duration_ok']);self.assertFalse(data['duration_check'])

    def test_explicit_declared_plan_is_checked_separately_from_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);result=self.call(root,5247.264,5252.010666666667,expected=5247.264)
            self.assertTrue(result['duration_ok']);self.assertTrue(result['duration_check_enforced'])
            self.assertFalse(result['reference_duration_ok']);self.assertFalse(result['is_original_baseline'])
            self.assertEqual(result['duration_difference_from_expected'],0.)
            self.assertAlmostEqual(result['duration_difference'],-4.746666666667,places=8)
            data=json.loads((root/renderer.FINAL_RENDER_VALIDATION_JSON).read_text())
            self.assertEqual(data['expected_duration_basis'],'declared_pcm_plan')
            self.assertEqual(data['expected_output_duration'],5247.264)
            self.assertFalse(data['reference_duration_ok']);self.assertTrue(data['duration_check'])

    def test_declared_plan_mismatch_cannot_be_hidden_by_matching_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(RuntimeError):self.call(root,5252.010666666667,5252.010666666667,expected=5247.264)
            data=json.loads((root/renderer.FINAL_RENDER_VALIDATION_JSON).read_text())
            self.assertTrue(data['reference_duration_ok']);self.assertFalse(data['duration_check'])
            self.assertAlmostEqual(data['duration_difference_from_expected'],4.746666666667,places=8)

    def test_existing_half_second_tolerance_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            result=self.call(Path(temp),100.5,100.)
            self.assertTrue(result['duration_ok']);self.assertTrue(result['reference_duration_ok'])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(RuntimeError):self.call(Path(temp),100.5001,100.)

    def test_invalid_declared_expectation_is_rejected(self):
        for value in (0.,-1.,float('nan'),float('inf')):
            with self.subTest(expected=value),tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):self.call(Path(temp),5247.264,5252.010666666667,expected=value)


if __name__=='__main__':unittest.main()
