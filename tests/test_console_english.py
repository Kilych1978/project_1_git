import builtins
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from console_en import english,english_console
from pcm_refiner import _ProgressCallback


class EnglishConsoleTests(unittest.TestCase):
    def test_repeated_grid_headers_are_compact_and_audio_specific(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            for grid in (1,2):
                callback=_ProgressCallback(f'СОХРАННОСТЬ БАЗОВОГО РЕЗУЛЬТАТА | сетка {grid}/2',duration=5252.01)
                callback.window(4056.476,4056.601);callback(8114,10504);callback.finish()
        text=out.getvalue()
        self.assertEqual(text.count('RENDER VALIDATION: BASELINE REGRESSION CHECK'),1)
        self.assertEqual(text.count('Track: 01:27:32'),1)
        self.assertIn('Grid: 1/2',text);self.assertIn('Grid: 2/2',text)
        self.assertIn('8114/10504 probes | At 01:07:36 | ETA',text)
        self.assertTrue(text.isascii())
        self.assertNotIn('film',text.lower());self.assertNotIn('movie',text.lower())

    def test_filenames_and_stored_report_text_are_preserved(self):
        for value in ('Готово.wav','1. Готово.wav',r'C:\Дорожки\Готово_мс.wav'):
            self.assertEqual(english(value),value)
        self.assertEqual(english('Render завершён: Готово.wav'),'Render completed: Готово.wav')
        out=io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'REPORT_RU.txt'
            with contextlib.redirect_stdout(out),english_console():
                print('СОХРАННОСТЬ БАЗОВОГО РЕЗУЛЬТАТА')
                report.write_text('СОХРАННОСТЬ БАЗОВОГО РЕЗУЛЬТАТА',encoding='utf-8')
            self.assertEqual(report.read_text(encoding='utf-8'),'СОХРАННОСТЬ БАЗОВОГО РЕЗУЛЬТАТА')
        self.assertIn('BASELINE REGRESSION CHECK',out.getvalue())

    def test_input_prompt_is_translated_and_original_input_is_restored(self):
        with patch('builtins.input',return_value='1') as original:
            with english_console():self.assertEqual(input('Введите номер и нажмите Enter: '),'1')
            self.assertIs(builtins.input,original)
            original.assert_called_once_with('Enter a number and press Enter: ')


if __name__=='__main__':unittest.main()
