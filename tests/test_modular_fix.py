import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from common import Progress, SelectedTracks
from pcm_core.config import Settings
from pcm_core.guard import render_verdict, verify_render
from pcm_core.model import TimeMap
from pcm_refiner import refine_rendered_audio
from scanner import probe_audio_file
from renderer import copy_target_without_reencode
from renderer import build_render_segments
from safety import _build_interval_gate
from applier import build_apply_map, validate_apply_map_limits
from mapper import build_final_sync_map
from micro_refiner import refine_sync_map_with_micro_waveform
from .helpers import SR, soundtrack, write_audio


class ModularFixTests(unittest.TestCase):
    def test_safety_edge_taper_does_not_create_an_impossible_render_tempo(self):
        t = np.arange(0., 241., 1.)
        checks = np.arange(18., 219., 10.)
        gate, _ = _build_interval_gate(t, checks, np.ones_like(checks))
        offset = -.265 * gate
        mapping = {'target_time': t, 'offset': offset, 'reference_time': t + offset}
        with contextlib.redirect_stdout(io.StringIO()):
            apply = build_apply_map(mapping, 240., Progress())
            validate_apply_map_limits(apply)
            segments = build_render_segments(apply)
        self.assertTrue(segments)
        self.assertLess(max(s['tempo'] for s in segments), 1.20)

    def test_user_failure_measurements_no_long_gap_global_abort(self):
        # Numeric-only fixture from the failing run; no personal paths/audio.
        path = Path(__file__).parent / 'fixtures/failed_measurements_20261014.csv'
        rows = []
        with path.open() as file:
            for row in csv.DictReader(file):
                rows.append({key: (value == 'True' if key == 'accepted_for_tracking' else float(value))
                             for key, value in row.items()})
        self.assertEqual(len(rows), 1051)
        with contextlib.redirect_stdout(io.StringIO()):
            result = build_final_sync_map({'local_results': rows}, 5252.01, Progress())
        clean = result['cleaned']['clean']
        self.assertGreater(len(clean['time']), 900)
        self.assertLess(np.max(np.abs(clean['offset'])), 1.)
        self.assertLess(abs(np.median(result['map']['offset'])), .010)

    def test_micro_residual_subtracts_the_existing_nonzero_map(self):
        t = np.arange(0., 8., 1.)
        base = {'target_time': t, 'reference_time': t + .5,
                'offset': np.full_like(t, .5), 'confidence': np.ones_like(t)}
        dense = [{'target_time': 4., 'after_offset': .506, 'after_error': .006}]
        with tempfile.TemporaryDirectory() as td, patch('micro_refiner.micro_search_at_time', return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            out = Path(td)
            refine_sync_map_with_micro_waveform({'reference_pcm': 'unused', 'target_pcm': 'unused'},
                                                base, dense, out, Progress())
            with (out / 'micro_waveform_refinement.csv').open(encoding='utf-8-sig') as file:
                row = next(csv.DictReader(file, delimiter=';'))
            self.assertAlmostEqual(float(row['baseline_residual_sec']), .006, places=6)

    def run_fine(self, directory, reference, target, *, channels=1, settings=None):
        rp, tp = directory / 'эталон.wav', directory / 'перевод.wav'
        write_audio(rp, reference)
        if channels == 6:
            # Distinct centre dialogue must not enter the channel-0 analysis.
            target = np.stack([target, target * .5, soundtrack(len(target) / SR, 300),
                               target * .2, target * .7, target * .3], axis=1)
        write_audio(tp, target, layout='5.1' if channels == 6 else None)
        target_first = target[:, 0] if channels == 6 else target
        np.asarray(np.clip(reference * 32768, -32768, 32767), dtype='<i2').tofile(directory / 'ref.pcm')
        np.asarray(np.clip(target_first * 32768, -32768, 32767), dtype='<i2').tofile(directory / 'target.pcm')
        selected = SelectedTracks(probe_audio_file(rp), probe_audio_file(tp))
        out = directory / 'result'
        out.mkdir()
        t = np.arange(0., selected.target.duration, .25)
        t = np.r_[t, selected.target.duration]
        zero = {'target_time': t, 'reference_time': t.copy(),
                'offset': np.zeros_like(t), 'confidence': np.ones_like(t)}
        result = {'output_path': tp, 'encoder': 'copy', 'segments': [],
                  'codec_plan': {'pcm_fallback': False, 'encoder': 'copy'},
                  'copied_without_reencode': True, 'verified_first_channel_pcm': directory / 'target.pcm'}
        original_hash = hashlib.sha256(tp.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            final, result, report = refine_rendered_audio(selected,
                {'reference_pcm': directory / 'ref.pcm'}, zero, result, out,
                settings or Settings(anchor_step=2., min_improvement=1. / SR), make_plots=False)
        self.assertEqual(hashlib.sha256(tp.read_bytes()).hexdigest(), original_hash)
        return final, result, report, out

    def test_actual_6_and_10_ms_shifts_with_6_channels(self):
        reference = soundtrack(14.)
        for shift in (.006, .010):
            with self.subTest(shift=shift), tempfile.TemporaryDirectory() as td:
                n = round(shift * SR)
                target = np.r_[np.zeros(n), reference[:-n]]
                final, result, report, out = self.run_fine(Path(td), reference, target, channels=6)
                self.assertEqual(report['status'], 'PCM_REFINED', report['reason'])
                actual = report['actual_output_windows']
                verified = [r for r in actual if r['micro_changed'] and r['status'] == 'VERIFIED']
                self.assertGreaterEqual(len(verified), 6)
                self.assertLess(max(abs(r['residual_sec']) for r in verified), 1. / SR)
                output = probe_audio_file(result['output_path'])
                self.assertEqual((output.channels, output.sample_rate, output.codec_name), (6, SR, 'pcm_f32le'))
                self.assertTrue((out / 'DIAGNOSTIC_CLIPS.zip').exists())

    def test_identity_and_unrelated_audio_keep_baseline(self):
        x = soundtrack(12)
        for target in (x, soundtrack(12, 177)):
            with tempfile.TemporaryDirectory() as td:
                _, result, report, _ = self.run_fine(Path(td), x, target)
                self.assertEqual(report['status'], 'BASELINE_KEPT')
                self.assertTrue(result['copied_without_reencode'])

    def test_35_seconds_without_common_audio_do_not_cancel_other_improvements(self):
        x = soundtrack(80)
        n = round(.006 * SR)
        y = np.r_[np.zeros(n), x[:-n]]
        y[25 * SR:60 * SR] = soundtrack(35, 823)
        with tempfile.TemporaryDirectory() as td:
            final, _, report, _ = self.run_fine(Path(td), x, y,
                                               settings=Settings(min_improvement=1. / SR))
            self.assertEqual(report['status'], 'PCM_REFINED', report['reason'])
            times = final['target_time']
            self.assertLess(abs(np.interp(40., times, final['offset'])), 1e-10)
            self.assertAlmostEqual(np.interp(10., times, final['offset']), -.006, delta=1 / SR)
            self.assertAlmostEqual(np.interp(70., times, final['offset']), -.006, delta=1 / SR)

    def test_failed_actual_render_is_discarded_without_losing_baseline(self):
        x = soundtrack(14)
        n = round(.010 * SR)
        y = np.r_[np.zeros(n), x[:-n]]
        with tempfile.TemporaryDirectory() as td, patch('pcm_refiner.render_verdict', return_value='REJECTED'):
            _, result, report, out = self.run_fine(Path(td), x, y)
            self.assertEqual(report['status'], 'BASELINE_KEPT')
            self.assertTrue(result['copied_without_reencode'])
            self.assertFalse(list(out.glob('*_SYNCED.wav')))
            self.assertFalse((out / '_pcm_micro/CANDIDATE_TARGET.wav').exists())

    def test_error_in_fine_analysis_keeps_baseline_and_reports_reason(self):
        x = soundtrack(12)
        with tempfile.TemporaryDirectory() as td, patch('pcm_refiner.collect', side_effect=RuntimeError('test decoder failure')):
            _, result, report, out = self.run_fine(Path(td), x, x)
            self.assertEqual(report['status'], 'BASELINE_KEPT_ERROR')
            self.assertIn('test decoder failure', report['reason'])
            self.assertTrue(result['copied_without_reencode'])
            self.assertTrue((out / 'PCM_MICRO_REPORT_RU.txt').exists())

    def test_corrupted_previously_measurable_region_is_rejected(self):
        x = soundtrack(14.)
        corrupted = x.copy()
        corrupted[4 * SR:7 * SR] = 0
        identity = TimeMap.affine(14)
        rows = verify_render(x, x, corrupted, identity, identity, Settings(anchor_step=2))
        self.assertTrue(any(r['status'] == 'REGRESSION' for r in rows))
        self.assertEqual(render_verdict(rows), 'REJECTED')

    def test_pcm_copy_fallback_never_overwrites_source_or_previous_sibling(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_audio(d / 'target.wav', soundtrack(4))
            old = d / 'target_SYNCED.wav'
            old.write_bytes(b'previous output')
            out = d / 'pass'
            out.mkdir()
            result = copy_target_without_reencode(probe_audio_file(d / 'target.wav'),
                                                  {'pcm_fallback': True, 'encoder': 'pcm_s32le'}, out)
            self.assertEqual(old.read_bytes(), b'previous output')
            self.assertEqual(result['output_path'].read_bytes(), (d / 'target.wav').read_bytes())
            self.assertFalse(result['codec_plan']['pcm_fallback'])


if __name__ == '__main__':
    unittest.main()
