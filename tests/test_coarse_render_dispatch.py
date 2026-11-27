"""The real render guard selects a backend and still judges actual output."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from common import Progress
from render_guard import run_render_aware_safety_loop


def _map(offset):
    times = np.array([0., 90.])
    return {'target_time': times, 'reference_time': times + offset,
            'offset': np.full(2, offset, dtype=float)}


class CoarseRenderDispatchTests(unittest.TestCase):
    def run_guard(self, root, *, feature, zero=False, wrong=False):
        source = root/'original.wav'
        original = b'Original is never opened for writing by a render dispatch.'
        source.write_bytes(original)
        info = SimpleNamespace(path=source, duration=90., codec_name='pcm_s16le', format_name='wav')
        selected = SimpleNamespace(target=info, reference=SimpleNamespace(duration=90.))
        analysis = {'coarse_evidence': object()} if feature else {}
        rows = [{'target_time': 30., 'before_selected_offset': .3,
                 'before_selected_correlation': .95, 'feature_backend': feature,
                 'before_uncertainty_sec': .02 if feature else None,
                 'comparison_key': 'fixed-reference:30'}]
        candidate = _map(.3)
        initial = _map(0. if zero else .3)
        rendered_paths = []

        def write_candidate(selected, mapping, directory, progress, codec_plan=None):
            destination = directory/'original_SYNCED.wav'
            destination.write_bytes(b'Complete rendered fixture; timing is independently verified.')
            rendered_paths.append(destination)
            return {'output_path': destination, 'encoder': 'pcm_s16le', 'segments': [],
                    'codec_plan': codec_plan or {}, 'render_seconds': 0.}

        def verify(selected, analysis, path, directory, **kwargs):
            pcm = directory/'actual_first_channel.pcm'
            pcm.write_bytes(b'Actual decoded output; never predicted by the map.')
            residual = .3 if path == source else (.6 if wrong else 0.)
            return {'rendered_first_channel_pcm': pcm,
                    'validation_points': [{'target_time': 30., 'after_offset': residual,
                        'after_correlation': .95, 'feature_backend': feature,
                        'uncertainty_sec': .02 if feature else None,
                        'comparison_key': 'fixed-reference:30'}]}

        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch('render_guard.RENDER_GUARD_MAX_PASSES', 1))
            rebuild = stack.enter_context(patch('render_guard.build_non_regressive_sync_map', return_value=(candidate, {})))
            apply = stack.enter_context(patch('render_guard.build_apply_map', side_effect=lambda mapping, *args: mapping))
            native = stack.enter_context(patch('coarse_native_render.render_native_coarse', side_effect=write_candidate))
            legacy = stack.enter_context(patch('render_guard.render_target_with_sync_map', side_effect=write_candidate))
            verification = stack.enter_context(patch('render_guard.verify_rendered_sync', side_effect=verify))
            stack.enter_context(patch('render_guard.compare_micro_rendered_against_original', return_value={}))
            result = run_render_aware_safety_loop(selected, analysis, candidate, initial, rows, [],
                                                 root/'result', {}, Progress())
        self.assertEqual(source.read_bytes(), original)
        return result, native, legacy, apply, rebuild, verification, rendered_paths

    def test_nonzero_spectral_map_uses_native_and_verifies_actual_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (_, rendered, report, comparison), native, legacy, apply, _, verify, paths = self.run_guard(root, feature=True)
            native.assert_called_once()
            legacy.assert_not_called()
            apply.assert_not_called()
            verify.assert_called_once()
            self.assertEqual(verify.call_args.args[2], paths[0])
            self.assertTrue(report['passed'])
            self.assertTrue(report['output_created'])
            self.assertEqual(comparison['improvement_count'], 1)
            self.assertTrue(rendered['output_path'].is_file())

    def test_nonzero_legacy_map_keeps_existing_renderer_and_output_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            (_, rendered, report, _), native, legacy, apply, _, verify, paths = self.run_guard(Path(temp), feature=False)
            native.assert_not_called()
            legacy.assert_called_once()
            apply.assert_called_once()
            verify.assert_called_once()
            self.assertEqual(verify.call_args.args[2], paths[0])
            self.assertTrue(report['passed'])
            self.assertTrue(rendered['output_path'].is_file())

    def test_zero_spectral_map_reuses_source_without_creating_duplicate_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (final, rendered, report, _), native, legacy, apply, rebuild, verify, _ = self.run_guard(root, feature=True, zero=True)
            native.assert_not_called()
            legacy.assert_not_called()
            apply.assert_not_called()
            rebuild.assert_not_called()
            verify.assert_called_once()
            self.assertEqual(verify.call_args.args[2], root/'original.wav')
            self.assertEqual(rendered['output_path'], root/'original.wav')
            self.assertFalse(report['output_created'])
            self.assertTrue(report['fallback_to_baseline'])
            self.assertTrue(np.all(final['offset'] == 0.))
            self.assertFalse(list((root/'result').rglob('*.wav')))

    def test_wrong_native_render_is_rejected_and_source_is_reverified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (final, rendered, report, _), native, legacy, apply, _, verify, paths = self.run_guard(root, feature=True, wrong=True)
            native.assert_called_once()
            legacy.assert_not_called()
            apply.assert_not_called()
            self.assertEqual(verify.call_count, 2)
            self.assertEqual(verify.call_args_list[0].args[2], paths[0])
            self.assertEqual(verify.call_args_list[1].args[2], root/'original.wav')
            self.assertEqual(report['pass_records'][0]['comparison']['regression_count'], 1)
            self.assertTrue(report['fallback_to_baseline'])
            self.assertFalse(report['output_created'])
            self.assertEqual(rendered['output_path'], root/'original.wav')
            self.assertTrue(np.all(final['offset'] == 0.))
            self.assertFalse(paths[0].exists())
            self.assertFalse(list((root/'result').rglob('*.wav')))


if __name__ == '__main__':
    unittest.main()
