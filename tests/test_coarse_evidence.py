"""Independent clocks, frozen output checks, and legacy dispatch contracts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from common import AudioInfo, Progress
from coarse_evidence import CoarseEvidence
from coarse_features import build_features_array
from render_guard import compare_rendered_against_original, save_render_guard_csv, _write_pass_summary
from renderer import verify_rendered_sync
from validator import validate_sync_map
from tests.test_coarse_matcher import bank, latent, warped


def clock(offset=0., rate=1., duration=48.):
    t = np.array([0., duration])
    return {'target_time': t, 'offset': offset + (rate - 1.) * t}


def middle(rows, time=24.):
    return min(rows, key=lambda row: abs(row['reference_time'] - time))


class CoarseEvidenceTests(unittest.TestCase):
    def test_independent_cached_observations_and_identity_symmetry(self):
        x = latent()
        evidence = CoarseEvidence(bank(x), bank(warped(x, 5.37)), search_radius=10.)
        with patch('coarse_evidence.match_window', wraps=__import__('coarse_matcher').match_window) as matcher:
            before = evidence.validate(clock(), 48., interval_sec=24.)
            calls = matcher.call_count
            candidate = evidence.validate(clock(5.37), 48., interval_sec=24.)
            self.assertEqual(matcher.call_count, calls)
        row, candidate_row = middle(before), middle(candidate)
        self.assertEqual(row['feature_status'], 'CONFIRMED')
        self.assertAlmostEqual(row['before_offset'], 5.37, delta=.035)
        self.assertEqual(row['before_error'], row['after_error'])
        self.assertEqual(row['before_offset'], candidate_row['before_offset'])
        self.assertEqual(row['probes'], candidate_row['probes'])
        self.assertLess(candidate_row['after_error'], .035)
        self.assertEqual(row['comparison_reference_time'], row['reference_time'])
        self.assertAlmostEqual(row['target_time'], row['reference_time'] - 5.37, delta=.035)

    def test_affine_speed_is_not_confused_with_disagreement(self):
        x = latent(seed=38)
        evidence = CoarseEvidence(bank(x), bank(warped(x, 2.4, 1.04)), search_radius=10.)
        row = middle(evidence.validate(clock(2.4, 1.04), 48., interval_sec=24.))
        self.assertEqual(row['feature_status'], 'CONFIRMED', row)
        self.assertAlmostEqual(row['feature_rate'], 1.04, delta=.009)
        self.assertLess(row['after_error'], .035)
        self.assertLess(row['bidir_disagreement'], .04)
        # Real speed drift changes absolute lag across the six-second support.
        self.assertGreater(abs(row['probes'][-1]['offset'] - row['probes'][0]['offset']), .1)

    def test_proposal_grid_differs_from_validation_and_is_not_mutable(self):
        x = latent(seconds=90)
        evidence = CoarseEvidence(bank(x), bank(x))
        with patch('coarse_evidence.match_window', return_value={'status': 'UNVERIFIABLE'}):
            proposal = evidence.observations(30., phase=.5)
            validation = evidence.validate(clock(duration=90), 90., interval_sec=30.)
            self.assertTrue(set(r['reference_time'] for r in proposal).isdisjoint(
                r['reference_time'] for r in validation))
            proposal[0]['status'] = 'CONFIRMED'
            self.assertEqual(evidence.observations(30., phase=.5)[0]['status'], 'UNVERIFIABLE')

    def test_silence_or_unrelated_source_has_unknown_baseline(self):
        x = latent()
        for target in (np.zeros_like(x), latent(seed=140)):
            with self.subTest(silent=not np.any(target)):
                evidence = CoarseEvidence(bank(x), bank(target))
                row = evidence.validate(clock(4.), 48., interval_sec=24.)[0]
                self.assertIsNone(row['before_error'])
                self.assertIsNone(row['after_error'])
                self.assertIsNone(row['before_selected_offset'])
                self.assertFalse(row['source_event_known'])
                self.assertNotEqual(row['bidir_status'], 'CONFIRMED_MAP')
                output = CoarseEvidence(evidence.reference_bank, bank(x), _source=evidence)
                with patch('coarse_evidence.match_window', side_effect=AssertionError('Must not choose fresh channels')):
                    actual = output.validate(clock(), 48., interval_sec=24.)[0]
                self.assertIsNone(actual['before_error'])
                self.assertIsNone(actual['after_error'])

    def test_actual_output_follows_original_event_keys_and_frozen_band(self):
        x = latent()
        source = CoarseEvidence(bank(x, ['REF']), bank(warped(x, 4.2), ['TGT']))
        baseline = source.validate(clock(), 48., interval_sec=24.)
        actual = CoarseEvidence(source.reference_bank, bank(x, ['TGT']), _source=source)
        with patch('coarse_evidence.match_window', wraps=__import__('coarse_matcher').match_window) as matcher:
            rendered = actual.validate(clock(), 48., interval_sec=24.)
        baseline_row, rendered_row = middle(baseline), middle(rendered)
        selected = baseline_row['frozen_selection']
        baseline_by_reference = {row['reference_time']: row for row in baseline}
        for call in matcher.call_args_list:
            self.assertEqual(call.kwargs['frozen_views'], baseline_by_reference[call.args[2]]['frozen_selection'])
        self.assertIn('band', selected)
        self.assertEqual(rendered_row['target_time'], baseline_row['target_time'])
        self.assertEqual(rendered_row['comparison_key'], baseline_row['comparison_key'])
        self.assertAlmostEqual(rendered_row['rendered_target_time'], rendered_row['reference_time'], delta=.035)
        self.assertLess(rendered_row['after_error'], .035)
        comparison = compare_rendered_against_original(baseline, rendered)
        self.assertTrue(comparison['passed'])
        self.assertGreater(comparison['improvement_count'], 0)

    def test_actual_identity_reuses_same_clock_without_numeric_regression(self):
        x = latent()
        source_bank = bank(warped(x, -5.37), ['TGT'])
        source = CoarseEvidence(bank(x, ['REF']), source_bank)
        baseline = source.validate(clock(), 48., interval_sec=24.)
        actual = CoarseEvidence(source.reference_bank, source_bank, _source=source)
        rendered = actual.validate(clock(), 48., interval_sec=24.)
        self.assertEqual(middle(baseline)['before_error'], middle(rendered)['after_error'])
        self.assertEqual(middle(rendered)['target_time'], middle(baseline)['target_time'])
        comparison = compare_rendered_against_original(baseline, rendered)
        self.assertTrue(comparison['passed'])
        self.assertEqual(comparison['rows'][0]['error_delta_sec'], 0.)
        self.assertEqual(comparison['improvement_count'], 0)

    def test_output_wav_retains_original_view_identity_without_mutating_info(self):
        x = latent()
        for physical, output_layout in [(['c0', 'c1'], 'stereo'), (['FL', 'FR'], '')]:
            with self.subTest(physical=physical), tempfile.TemporaryDirectory() as temp:
                target = bank(np.stack([x, x], axis=1), physical)
                target.parameters['physical_channels'] = physical
                source = CoarseEvidence(bank(x), target)
                info = AudioInfo(Path(temp)/'output.wav', 'pcm_s16le', '', 's16', 48000,
                                 2, output_layout, 1536000, 48., 0, 'wav', '', 0)
                decoded = bank(np.stack([x, x], axis=1), physical)
                decoded.close = Mock()
                with patch('coarse_evidence.build_features_file', return_value=decoded) as build:
                    actual = source.for_output(info, temp)
                self.assertEqual(build.call_args.args[0].channel_layout, physical)
                self.assertEqual(info.channel_layout, output_layout)
                actual.close()
                decoded.close.assert_called_once()

    def test_actual_output_cannot_switch_to_new_common_channel(self):
        x, other = latent(), latent(seed=391)
        source = CoarseEvidence(bank(x, ['REF']), bank(warped(x, 4.2), ['TGT']))
        baseline = source.validate(clock(), 48., interval_sec=24.)
        output_bank = bank(np.stack([other, x], axis=1), ['TGT', 'DECOY_NEW_CHANNEL'])
        actual = CoarseEvidence(source.reference_bank, output_bank, _source=source)
        rows = actual.validate(clock(), 48., interval_sec=24.)
        self.assertTrue(middle(rows)['feature_verification_lost'])
        self.assertIsNone(middle(rows)['after_offset'])
        comparison = compare_rendered_against_original(baseline, rows)
        self.assertFalse(comparison['passed'])
        self.assertGreater(comparison['considered_points'], 0)
        self.assertIsNone(comparison['regressions'][0]['rendered_error_sec'])
        self.assertIsNone(comparison['regressions'][0]['error_delta_sec'])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_render_guard_csv(root, comparison, 1)
            report = _write_pass_summary(root, 1, comparison, [rows[0]['target_time']])
            self.assertIn('не подтверждён', report.read_text())

    def test_wrong_actual_file_fails_despite_correct_mathematical_map(self):
        # Exercise actual FFmpeg output decoding, not a recheck of the source's
        # predicted map. Only source and output paths are filesystem fixtures.
        from tests.helpers import soundtrack, write_audio
        from scanner import probe_audio_file
        sr, seconds = 12000, 48
        full_sound = soundtrack(seconds + 16.)[::4]
        full_time = np.arange(len(full_sound)) / sr
        original = full_sound[:sr * seconds]
        t = np.arange(len(original)) / sr
        shifted = np.interp(t + 4.2, full_time, full_sound)
        reference_bank = build_features_array(original, sr, channel_layout='mono')
        target_bank = build_features_array(shifted, sr, channel_layout='mono')
        source = CoarseEvidence(reference_bank, target_bank, search_radius=12.)
        try:
            candidate = source.validate(clock(4.2), seconds, interval_sec=24.)
            self.assertEqual(middle(candidate)['feature_status'], 'CONFIRMED', candidate)
            self.assertLess(middle(candidate)['after_error'], .04)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                output_path = root / 'arbitrary_name.wav'
                write_audio(output_path, np.interp(t + 8.1, full_time, full_sound), rate=sr)
                actual = source.for_output(probe_audio_file(output_path), root)
                try:
                    rows = actual.validate(clock(), seconds, interval_sec=24.)
                    self.assertEqual(middle(rows)['feature_status'], 'CONFIRMED', rows)
                    self.assertGreater(middle(rows)['after_error'], 8.)
                    comparison = compare_rendered_against_original(candidate, rows)
                    self.assertFalse(comparison['passed'])
                finally:
                    actual.close()
        finally:
            reference_bank.close()
            target_bank.close()

    def test_feature_improvement_must_exceed_both_uncertainties(self):
        baseline = {'target_time': 10., 'feature_backend': True,
                    'before_selected_offset': .02, 'before_selected_correlation': .9,
                    'before_uncertainty_sec': .02}
        output = {'target_time': 10., 'feature_backend': True, 'after_offset': 0.,
                  'after_correlation': .9, 'uncertainty_sec': .02}
        result = compare_rendered_against_original([baseline], [output])
        self.assertTrue(result['passed'])
        self.assertEqual(result['improvement_count'], 0)
        baseline['before_selected_offset'] = .2
        self.assertEqual(compare_rendered_against_original([baseline], [output])['improvement_count'], 1)

    def test_short_reference_has_enough_controls_with_identical_output_keys(self):
        # Cut two real overlapping windows from a longer sound, rather than
        # extending the last amplitude with interpolation's constant padding.
        full = latent(seconds=48)
        target = bank(warped(full, 3.31)[:3200], ['TGT'])
        source = CoarseEvidence(bank(full[:3200], ['REF']), target)
        baseline = source.validate(clock(duration=32.), 32., interval_sec=10.)
        self.assertEqual(len(baseline), 27)
        self.assertTrue(all(3. <= row['reference_time'] <= 29. for row in baseline))
        self.assertTrue(all(abs(row['feature_validation_interval_sec'] - 1.0) < 1e-12 for row in baseline))
        confirmed = [row for row in baseline if row['feature_status'] == 'CONFIRMED']
        self.assertGreaterEqual(len(confirmed), 8)
        self.assertGreaterEqual(max(row['reference_time'] for row in confirmed)
                                - min(row['reference_time'] for row in confirmed), 20.)
        actual = CoarseEvidence(source.reference_bank, target, _source=source)
        output = actual.validate(clock(duration=32.), 31., interval_sec=10.)
        self.assertEqual([row['comparison_key'] for row in baseline],
                         [row['comparison_key'] for row in output])
        self.assertEqual([row['target_time'] for row in baseline],
                         [row['target_time'] for row in output])
        comparison = compare_rendered_against_original(baseline, output)
        self.assertTrue(comparison['passed'])
        self.assertTrue(all(row['error_delta_sec'] == 0. for row in comparison['rows']))

    def test_long_dense_interval_is_unchanged(self):
        # Verify scheduling without doing a full recording's FFT searches.
        reference = bank(latent(seconds=1.))
        reference.duration = 106.
        evidence = CoarseEvidence(reference, reference)
        with patch('coarse_evidence.match_window', return_value={'status': 'UNVERIFIABLE'}):
            rows = evidence.validate(clock(duration=106.), 106., interval_sec=10.)
        self.assertEqual([row['reference_time'] for row in rows], list(np.arange(10., 104., 10.)))
        self.assertTrue(all(row['feature_validation_interval_sec'] == 10. for row in rows))

    def test_verification_loss_also_protects_pcm_wrapper_rows(self):
        baseline = {'target_time': 10., 'before_selected_offset': 0., 'before_selected_correlation': 1.}
        output = {'target_time': 10., 'feature_backend': False,
                  'evidence_kind': 'protected_first_channel_pcm',
                  'feature_verification_lost': True, 'after_offset': None}
        result = compare_rendered_against_original([baseline], [output])
        self.assertFalse(result['passed'])
        self.assertTrue(result['regressions'][0]['verification_lost'])

    def test_optional_validator_dispatch_and_legacy_default(self):
        backend = Mock()
        backend.validate.return_value = [{'backend': True}]
        sync_map = clock()
        result = validate_sync_map({'coarse_evidence': backend}, sync_map, 48., None, 12.)
        self.assertEqual(result, [{'backend': True}])
        backend.validate.assert_called_once_with(sync_map, 48., None, interval_sec=12.)
        envelope = {'values': np.ones(8000), 'hop_sec': .01}
        with contextlib.redirect_stdout(io.StringIO()), \
                patch('validator.build_validation_points', return_value=[24.]), \
                patch('validator.validation_search_offset', return_value={'offset': .3, 'correlation': .8}), \
                patch('validator.bidirectional_validation_check', return_value={
                    'consensus_offset': None, 'consensus_correlation': 0., 'status': 'UNRESOLVED',
                    'left_offset': None, 'right_offset': None, 'left_correlation': 0.,
                    'right_correlation': 0., 'disagreement': None, 'prediction_error': None}):
            legacy = validate_sync_map({'reference_envelope': envelope, 'target_envelope': envelope},
                                       clock(), 80., Progress())[0]
        self.assertNotIn('feature_backend', legacy)
        self.assertEqual(legacy['before_estimator'], 'full_window')
        self.assertEqual(legacy['before_error'], .3)
        self.assertEqual(legacy['after_error'], .3)

    def test_short_output_still_decodes_first_pcm_and_dispatches_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source.wav', root / 'output.wav'
            selected = SimpleNamespace(reference=SimpleNamespace(duration=32.),
                                       target=SimpleNamespace(path=source, duration=32.))
            info = SimpleNamespace(path=output, duration=32., codec_name='pcm_s16le',
                                   sample_rate=48000, channels=2)
            envelope = {'values': np.ones(3200), 'hop_sec': .01}
            actual = Mock()
            actual.validate.return_value = []
            backend = Mock()
            backend.for_output.return_value = actual
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch('renderer.probe_audio_file', return_value=info), \
                    patch('renderer.decode_first_channel_stream', return_value={'duration': 32.}) as decode, \
                    patch('renderer.build_envelope_from_pcm', return_value=envelope) as build:
                result = verify_rendered_sync(selected, {'reference_envelope': envelope, 'coarse_evidence': backend},
                                              output, root, validation_interval_sec=8.)
            decode.assert_called_once()
            build.assert_called_once()
            actual.validate.assert_called_once()
            self.assertEqual(actual.validate.call_args.kwargs['interval_sec'], 8.)
            actual.set_rendered_pcm.assert_called_once_with(root / '_render_verify' / 'rendered_first_channel_s16le.pcm')
            actual.close.assert_called_once()
            self.assertEqual(result['validation_points'], [])
            self.assertTrue(json.loads((root / 'FINAL_RENDER_VALIDATION.json').read_text())['duration_check'])

    def test_output_feature_handles_close_if_pcm_attachment_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = SimpleNamespace(reference=SimpleNamespace(duration=32.),
                                       target=SimpleNamespace(path=root/'source.wav', duration=32.))
            info = SimpleNamespace(path=root/'output.wav', duration=32.)
            envelope = {'values': np.ones(3200), 'hop_sec': .01}
            actual = Mock()
            actual.set_rendered_pcm.side_effect = RuntimeError('PCM attachment failed')
            backend = Mock()
            backend.for_output.return_value = actual
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch('renderer.probe_audio_file', return_value=info), \
                    patch('renderer.decode_first_channel_stream', return_value={'duration': 32.}), \
                    patch('renderer.build_envelope_from_pcm', return_value=envelope):
                with self.assertRaisesRegex(RuntimeError, 'PCM attachment failed'):
                    verify_rendered_sync(selected, {'reference_envelope': envelope, 'coarse_evidence': backend},
                                         info.path, root)
            actual.close.assert_called_once()
            actual.validate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
