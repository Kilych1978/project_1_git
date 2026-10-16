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
from pcm_core.guard import render_verdict, verify_render, cancel_unsafe_components, verify_candidate_pcm
from pcm_core.model import TimeMap
from pcm_core.model import Piece
from pcm_core.micro import Measurement, measure, refined_map
from pcm_refiner import refine_rendered_audio
from scanner import probe_audio_file
from renderer import copy_target_without_reencode
from renderer import build_render_segments
from safety import _build_interval_gate
from applier import build_apply_map, validate_apply_map_limits
from mapper import build_final_sync_map
from micro_refiner import refine_sync_map_with_micro_waveform
from .helpers import SR, soundtrack, write_audio, warp


class ModularFixTests(unittest.TestCase):
    def test_real_failed_clock_cancellation_keeps_valid_rate(self):
        f = json.loads((Path(__file__).parent/'fixtures/cancel_rate_20261016.json').read_text())
        t, delta = np.array(f['target']), np.array(f['delta'])
        candidate = TimeMap([Piece(t, t+delta, [{'target': t.tolist(), 'delta': delta.tolist()}])])
        base = TimeMap.affine(t[-1])
        candidate.validate(.02)
        corrected, cancelled = cancel_unsafe_components(base, candidate, f['failed_times'], .02)
        corrected.validate(.02)
        self.assertTrue(cancelled)
        self.assertTrue(np.any(np.abs(corrected.forward(t)-t)>1/SR))
        for failed in f['failed_times']:
            points=np.array([failed-.125,failed,failed+.125])
            points=points[(points>=0)&(points<=t[-1])]
            np.testing.assert_allclose(corrected.forward(points),points,atol=1e-10)
        fallback, events = cancel_unsafe_components(base,candidate,f['failed_times'],.02,whole_components=True)
        fallback.validate(.02)
        self.assertTrue(any(r.get('mode')=='whole_component_rate_fallback' for r in events))

    def test_preview_rejects_worse_timing_before_any_multichannel_render(self):
        x = soundtrack(14., 812)
        n = round(.006*SR)
        y = np.r_[np.zeros(n), x[:-n]]
        base = TimeMap.affine(14.)
        wrong = TimeMap.affine(14., offset=.006)
        rows = verify_candidate_pcm(x, y, base, wrong, Settings(anchor_step=1.))
        self.assertEqual(render_verdict(rows), 'REJECTED')
        good = TimeMap.affine(14., offset=-.006)
        rows = verify_candidate_pcm(x, y, base, good, Settings(anchor_step=1.))
        self.assertEqual(render_verdict(rows), 'VERIFIED_PCM')

    def test_saved_run_resumes_without_repeating_map_analysis(self):
        from resume_launcher import resume
        from scanner import save_initial_metadata
        import shutil
        x = soundtrack(14., 987)
        y = np.r_[np.zeros(round(.006*SR)), x[:-round(.006*SR)]]
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            _, _, report, previous = self.run_fine(directory, x, y, channels=6)
            self.assertEqual(report['status'], 'PCM_REFINED')
            report['status'] = 'BASELINE_KEPT_ERROR'
            (previous/'PCM_MICRO_REPORT.json').write_text(json.dumps(report))
            selected = SelectedTracks(probe_audio_file(directory/'эталон.wav'),probe_audio_file(directory/'перевод.wav'))
            save_initial_metadata(previous, selected)
            analysis_info = {'analysis_sample_rate': SR, 'pcm_format': 's16le'}
            for role,name in [('reference','ref.pcm'),('target','target.pcm')]:
                p=previous/f'{role}_first_channel_s16le.pcm'
                shutil.copyfile(directory/name, p)
                analysis_info[f'{role}_decode'] = {'path':str(p),'size_bytes':p.stat().st_size,'duration':14.}
            (previous/'analysis_info.json').write_text(json.dumps(analysis_info))
            (previous/'FINAL_SYNC_MAP.json').write_text(json.dumps({'coarse_map_before_pcm':{
                'target_time':[0.,14.],'reference_time':[0.,14.],'offset':[0.,0.]}}))
            old_hash=hashlib.sha256((previous/'PCM_MICRO_REPORT.json').read_bytes()).hexdigest()
            with patch('pcm_refiner._analyze_candidate',side_effect=AssertionError('analysis repeated')), \
                 patch('builtins.input',return_value='y'), contextlib.redirect_stdout(io.StringIO()):
                result=resume(previous)
            self.assertEqual(result['pcm_micro_refinement']['status'],'PCM_REFINED')
            self.assertTrue(result['pcm_micro_refinement']['resumed_candidate'])
            self.assertEqual(result['pcm_micro_refinement']['analysis_steps_sec'],[])
            self.assertEqual(hashlib.sha256((previous/'PCM_MICRO_REPORT.json').read_bytes()).hexdigest(),old_hash)
            self.assertEqual(probe_audio_file(Path(result['output_file'])).channels,6)

    def test_decoded_failure_removes_only_its_component_and_rerenders_baseline(self):
        x = soundtrack(14., 921)
        from scipy.signal import butter, sosfiltfilt
        x = sosfiltfilt(butter(3, 1000, fs=SR, output='sos'), x).astype(np.float32)
        knots = np.array([0., 1., 5., 6., 8., 9., 13., 14.])
        delta = np.array([0., -.006, -.006, 0., 0., -.006, -.006, 0.])
        mapping = TimeMap([Piece(knots, knots + delta,
                                [{'target': knots.tolist(), 'delta': delta.tolist()}])])
        # The source really has these two independent clock curves. A constant
        # delay with artificial map tapers is already wrong at both boundaries
        # and cannot be used to test preservation of a VERIFIED component.
        y = warp(x, mapping.forward(np.arange(len(x))/SR)*SR)
        reduced, removed = cancel_unsafe_components(TimeMap.affine(14.), mapping, [10.])
        np.testing.assert_array_equal(reduced.forward(np.linspace(0., 6., 101)),
                                      mapping.forward(np.linspace(0., 6., 101)))
        np.testing.assert_allclose(reduced.forward([10.]), [10.], atol=1e-12)
        self.assertEqual(len(removed), 1)
        checks = [{'target_time': t, 'status': 'IMPROVED', 'change_sec': -.006,
                   'before_residual_sec': -.006, 'after_residual_sec': 0.,
                   'before_score': .99, 'after_score': .99} for t in (2., 4., 10., 12.)]
        calls = []
        def verify(*args, **kwargs):
            rows = verify_render(*args, **kwargs)
            calls.append(1)
            if len(calls) == 1:
                min(rows, key=lambda r: abs(r['target_time'] - 10.))['status'] = 'REGRESSION'
            return rows
        with tempfile.TemporaryDirectory() as td, \
             patch('pcm_refiner.gate_micro', return_value=(mapping, checks, [])), \
             patch('pcm_refiner.verify_render', side_effect=verify):
            final, result, report, _ = self.run_fine(Path(td), x, y, settings=Settings(anchor_step=1., min_improvement=1/SR))
            self.assertEqual(report['status'], 'PCM_REFINED', report['reason'])
            self.assertGreaterEqual(len(calls), 2)
            self.assertTrue(any(r['status'] == 'ACTUAL_LOCAL_CANCEL' for r in report['audit']))
            self.assertAlmostEqual(np.interp(3., final['target_time'], final['offset']), -.006, delta=1/SR)
            self.assertAlmostEqual(np.interp(10., final['target_time'], final['offset']), 0., delta=1e-10)
            self.assertEqual(probe_audio_file(result['output_path']).codec_name, 'pcm_f32le')

    def test_one_invalid_local_transition_keeps_other_movie_corrections(self):
        times = np.arange(1., 100.)
        rows = [Measurement(float(t), float(t), float(-.001 * ((t % 30) - 15)),
                            .99, .8, .00001, 3, 1, 'MEASURED', 'raw') for t in times]
        mapping = refined_map(TimeMap.affine(100.), rows,
                              Settings(anchor_step=1., max_rate_change=.02, min_improvement=1/SR))
        values = mapping.forward(times) - times
        self.assertGreater(np.count_nonzero(np.abs(values) > 1/SR), 80)
        self.assertLessEqual(max(np.max(np.abs(p.rate_extrema() - 1)) for p in mapping.pieces), .02 + 1e-10)

    def test_consensus_undoes_known_candidate_curve_before_clock_check(self):
        x = soundtrack(12.)
        n = round(.006 * SR)
        y = np.r_[np.zeros(n), x[:-n]]
        knots = np.array([0., 2., 6., 12.])
        delta = np.array([0., -.006, -.006, 0.])
        candidate = TimeMap([Piece(knots, knots + delta,
                                  [{'target': knots.tolist(), 'delta': delta.tolist()}])])
        row = measure(x, y, candidate, 1., Settings(anchor_step=5.))
        self.assertEqual(row.status, 'MEASURED')
        self.assertLess(abs(float(candidate.forward([1.])[0]) - 1. + row.correction_sec + .006), 1/SR)

    def test_splice_center_uses_reciprocal_pcm_instead_of_false_linear_clock(self):
        x = soundtrack(12., 746)
        positions = np.arange(len(x))
        times = positions / SR
        lag = np.where(np.abs(times - 5.5) < .05, .004, .006)
        y = x[np.clip(positions - np.rint(lag * SR).astype(int), 0, len(x) - 1)]
        row = measure(x, y, TimeMap.affine(12.), 5.5, Settings(anchor_step=.5))
        self.assertEqual(row.status, 'MEASURED')
        self.assertIn('boundary', row.method)
        self.assertAlmostEqual(row.correction_sec, -.004, delta=1/SR)

    def test_failing_finer_grid_preserves_previously_verified_candidate(self):
        x = soundtrack(14.)
        n = round(.006 * SR)
        y = np.r_[np.zeros(n), x[:-n]]
        calls = []
        def gate(reference, target, base, measurements, settings, **kwargs):
            calls.append(settings.anchor_step)
            checks = [{'target_time': t, 'status': 'IMPROVED', 'change_sec': -.006,
                       'before_residual_sec': -.006, 'after_residual_sec': .0001,
                       'before_score': .99, 'after_score': .99} for t in (1., 6., 11.)]
            if len(calls) == 1:
                return TimeMap.affine(14., offset=-.006), checks, checks
            return base, checks, [{'status': 'INVALID_MAP'}]
        with tempfile.TemporaryDirectory() as td, patch('pcm_refiner.gate_micro', side_effect=gate):
            _, _, report, _ = self.run_fine(Path(td), x, y, settings=Settings(anchor_step=5., min_improvement=1/SR))
            self.assertGreater(len(calls), 1)
            self.assertEqual(report['status'], 'PCM_REFINED', report['reason'])
            self.assertEqual(report['selected_step_sec'], 5.)

    def test_micro_only_original_menus_create_verified_multichannel_output(self):
        import scanner
        from micro_launcher import main
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            x = soundtrack(14.)
            n = round(.010 * SR)
            y = np.r_[np.zeros(n), x[:-n]]
            rp, tp = directory/'01_reference.wav', directory/'02_target.wav'
            write_audio(rp, x)
            write_audio(tp, np.stack([y, y*.5, soundtrack(14., 772), y*.2, y*.7, y*.3], axis=1), layout='5.1')
            before = hashlib.sha256(tp.read_bytes()).hexdigest()
            output = io.StringIO()
            with patch.object(scanner, 'BASE_DIR', directory), patch('builtins.input', side_effect=['1','1','y']), contextlib.redirect_stdout(output):
                main()
            self.assertIn('ШАГ 4/4', output.getvalue())
            self.assertIn('[100.00%]', output.getvalue())
            report_path = next(directory.glob('*_WAVEFORM_SYNC/FINAL_SYNC_MAP.json'))
            report = json.loads(report_path.read_text())
            self.assertEqual(report['pcm_micro_refinement']['status'], 'PCM_REFINED')
            self.assertEqual(probe_audio_file(Path(report['output_file'])).channels, 6)
            self.assertEqual(hashlib.sha256(tp.read_bytes()).hexdigest(), before)

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
