import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from common import Progress
from diagnostics import build_bundle, diagnostic_session, register_output
from diagnose_points import parse_time


class _Terminal(io.StringIO):
    def isatty(self):
        return True


class DiagnosticTests(unittest.TestCase):
    def test_diagnostic_timecodes_and_exact_first_channel_export(self):
        import numpy as np
        import scanner
        from diagnose_points import main
        from .helpers import SR, write_audio
        self.assertEqual(parse_time('7:28,755'),448.755)
        self.assertEqual(parse_time('00:52:15.301'),3135.301)
        for value in ('nan','-1','1:61','1:2:3:4'):
            with self.assertRaises(ValueError):
                parse_time(value)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            t = np.arange(6*SR)/SR
            signal = .1*np.sin(t*1234)
            reference, target = root/'01_reference.wav',root/'02_target.wav'
            write_audio(reference,signal,codec='pcm_s16le')
            write_audio(target,np.stack([signal,signal*.5,signal*0],axis=1),codec='pcm_s16le')
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (reference,target)]
            with patch.object(scanner,'BASE_DIR',root),patch('builtins.input',side_effect=['1','1','y']),contextlib.redirect_stdout(io.StringIO()):
                main(['0:03.125'])
            output = next(root.glob('*_WAVEFORM_SYNC'))
            self.assertFalse(list(output.rglob('*.pcm')))
            with zipfile.ZipFile(output/'REPORT_BUNDLE.zip') as archive:
                self.assertIn('REQUESTED_POINTS.json',archive.namelist())
                clips = archive.read('DIAGNOSTIC_CLIPS.zip')
                with zipfile.ZipFile(io.BytesIO(clips)) as inner:
                    manifest = json.loads(inner.read('CLIPS.json'))
                    self.assertEqual(len(manifest['clips']),1)
                    self.assertEqual(manifest['channels'],1)
            self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in (reference,target)],hashes)

    def test_progress_fits_terminal_and_clears_previous_message(self):
        output = _Terminal()
        with contextlib.redirect_stdout(output), patch('common.shutil.get_terminal_size', return_value=os.terminal_size((80, 30))):
            progress = Progress()
            progress.update(34.5, 'Очень длинное название этапа '*6+' | 123/64065 точек | ETA 00:14:12')
            progress.update(35, 'Коротко')
        frames = output.getvalue().split('\r')[1:]
        self.assertTrue(all(len(frame) <= 79 for frame in frames))
        self.assertIn('123/64065 точек', frames[0])
        self.assertIn('ETA 00:14:12', frames[0])
        self.assertTrue(frames[1].rstrip().endswith('Коротко'))
        self.assertEqual(len(frames[1]), len(frames[0]))

    def test_redirected_output_is_readable_and_throttled(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch('common.time.monotonic', return_value=0):
            progress = Progress()
            for index in range(101):
                progress.update(index, 'Поиск')
            progress.done(100, 'Готово')
        self.assertNotIn('\r', output.getvalue())
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertIn('100.00%', output.getvalue())

    def test_complete_bundle_contains_final_map_logs_and_clips_without_full_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'FINAL_SYNC_MAP.json').write_text('{"done": true}')
            (root/'PCM_MICRO_REPORT.json').write_text('{}')
            (root/'REPORT_RU.txt').write_text('Отчёт', encoding='utf-8')
            guard = root/'_render_guard'
            guard.mkdir()
            (guard/'ffmpeg.log').write_text('Render log')
            (guard/'audio.f32').write_bytes(b'not a diagnostic excerpt')
            (root/'original.wav').write_bytes(b'full audio')
            (root/'analysis.npy').write_bytes(b'full array')
            (root/'reference.pcm').write_bytes(b'full pcm')
            (root/'DIAGNOSTIC_CLIPS.zip').write_bytes(b'small clips')
            paths = build_bundle(root)
            self.assertEqual([p.name for p in paths], ['REPORT_BUNDLE.zip'])
            with zipfile.ZipFile(paths[0]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertTrue({'FINAL_SYNC_MAP.json', 'PCM_MICRO_REPORT.json', 'REPORT_RU.txt',
                                 '_render_guard/ffmpeg.log', 'DIAGNOSTIC_CLIPS.zip', 'MANIFEST.json'} <= set(archive.namelist()))
                self.assertFalse(any(name.endswith(('.wav', '.npy', '.pcm', '.f32')) for name in archive.namelist()))
                manifest = json.loads(archive.read('MANIFEST.json'))
                for entry in manifest['originals']:
                    self.assertEqual(hashlib.sha256(archive.read(entry['path'])).hexdigest(), entry['sha256'])

    def test_large_report_is_split_losslessly_into_independent_uploadable_zips(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original = os.urandom(160000)
            (root/'huge.log').write_bytes(original)
            (root/'summary.json').write_text('{}')
            paths = build_bundle(root, limit=64*1024)
            self.assertGreater(len(paths), 1)
            self.assertTrue(all(p.stat().st_size <= 64*1024 for p in paths))
            restored = {}
            for path in paths:
                with zipfile.ZipFile(path) as archive:
                    self.assertIsNone(archive.testzip())
                    manifest = json.loads(archive.read('MANIFEST.json'))
                    self.assertEqual(manifest['volumes'], [p.name for p in paths])
                    for member in manifest['members']:
                        if member['volume'] == path.name:
                            restored.setdefault(member['original'], []).append((member['offset'], archive.read(member['path'])))
            joined = b''.join(data for offset, data in sorted(restored['huge.log']))
            self.assertEqual(joined, original)

    def test_session_packages_after_final_reports_and_on_failure(self):
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                @diagnostic_session
                def launch():
                    print('Начало запуска')
                    register_output(root)
                    (root/'FINAL_SYNC_MAP.json').write_text('{"final": true}')
                    print('Карта сохранена')
                    if fails:
                        raise ValueError('Диагностическая ошибка')
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    if fails:
                        with self.assertRaisesRegex(ValueError, 'Диагностическая ошибка'):
                            launch()
                    else:
                        launch()
                with zipfile.ZipFile(root/'REPORT_BUNDLE.zip') as archive:
                    self.assertIn('FINAL_SYNC_MAP.json', archive.namelist())
                    log = archive.read('RUN_LOG.txt').decode('utf-8-sig')
                    self.assertIn('Начало запуска', log)
                    self.assertIn('Карта сохранена', log)
                    self.assertEqual(json.loads(archive.read('RUN_INFO.json'))['status'], 'ERROR' if fails else 'COMPLETED')
                    if fails:
                        self.assertIn('ValueError', log)
                self.assertIn('REPORT_BUNDLE.zip', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
