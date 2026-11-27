import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from coarse_alignment import NoCommonAudio, classic_is_reliable, run_general_alignment_stage
from diagnostics import diagnostic_session, register_analysis_resource


class CoarseAlignmentTests(unittest.TestCase):
    def selected(self, duration=240.):
        return SimpleNamespace(reference=SimpleNamespace(duration=duration),
                               target=SimpleNamespace(duration=duration))

    def alignment(self, strong=True):
        return {'local_results': [dict(target_time=float(t), offset=.006,
            correlation=.95 if strong else .20, margin=.10) for t in range(10,210,20)]}

    def test_reliable_first_channel_alignment_is_returned_unchanged(self):
        legacy=self.alignment()
        with tempfile.TemporaryDirectory() as tmp, \
             patch('coarse_alignment.aligner.run_alignment_stage',return_value=legacy), \
             patch('coarse_features.build_features_file',side_effect=AssertionError('Unexpected new analysis')) as feature:
            actual=run_general_alignment_stage(self.selected(),Path(tmp),{},None)
        self.assertIs(actual,legacy)
        feature.assert_not_called()

    def test_optional_feature_failure_preserves_existing_candidate_and_validation(self):
        legacy=self.alignment(False)
        analysis={'reference_pcm':'original_ref.pcm','target_pcm':'original_target.pcm'}
        with tempfile.TemporaryDirectory() as tmp, \
             patch('coarse_alignment.aligner.run_alignment_stage',return_value=legacy), \
             patch('coarse_features.build_features_file',side_effect=OSError('Analysis disk unavailable')), \
             contextlib.redirect_stdout(io.StringIO()):
            actual=run_general_alignment_stage(self.selected(),Path(tmp),analysis,None)
            report=json.loads((Path(tmp)/'COARSE_MATCH_REPORT.json').read_text())
        self.assertIs(actual,legacy)
        self.assertNotIn('coarse_evidence',analysis)
        self.assertEqual(report['mode'],'FEATURE_BACKEND_UNAVAILABLE')

    def test_short_audio_without_common_measurements_is_an_explicit_refusal(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('coarse_features.build_features_file',side_effect=ValueError('No usable audio')), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(NoCommonAudio):
                run_general_alignment_stage(self.selected(8.),Path(tmp),{},None)

    def test_a_few_strong_points_do_not_certify_an_entire_weak_first_channel(self):
        mixed=self.alignment()
        mixed['local_results']+=self.alignment(False)['local_results']
        self.assertFalse(classic_is_reliable(mixed))
        self.assertTrue(classic_is_reliable(self.alignment()))

    def test_session_releases_analysis_handles_after_success_and_error(self):
        for fail in (False,True):
            closed=[]
            class Resource:
                def close(self):closed.append(True)
            @diagnostic_session
            def run():
                register_analysis_resource(Resource())
                if fail:raise RuntimeError('Deliberate interruption')
            with contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaises(RuntimeError):run()
                else:run()
            self.assertEqual(closed,[True])


if __name__=='__main__':unittest.main()
