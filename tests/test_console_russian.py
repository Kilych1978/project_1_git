import builtins
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from console_ru import russian,russian_console
from pcm_refiner import _ProgressCallback


class RussianConsoleTests(unittest.TestCase):
    def test_repeated_grid_headers_are_compact_and_audio_specific(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            for grid in (1,2):
                callback=_ProgressCallback(f'СОХРАННОСТЬ БАЗОВОГО РЕЗУЛЬТАТА | сетка {grid}/2',duration=5252.01)
                callback.window(4056.476,4056.601);callback(8114,10504);callback.finish()
        text=out.getvalue()
        self.assertEqual(text.count('ПРОВЕРКА СОХРАНЁННОГО ЗВУКА: СРАВНЕНИЕ С БАЗОЙ'),1)
        self.assertEqual(text.count('Длительность: 01:27:32'),1)
        self.assertIn('Сетка: 1/2',text);self.assertIn('Сетка: 2/2',text)
        self.assertIn('8114/10504 точек | Позиция 01:07:36 | Осталось',text)
        self.assertNotIn('фильм',text.lower())
        self.assertNotIn('BASELINE',text);self.assertNotIn('ETA',text)

    def test_filenames_and_stored_report_text_are_preserved(self):
        for value in ('Reference.wav','1. Reference.wav',r'C:\Дорожки\Reference.wav'):
            self.assertEqual(russian(value),value)
        out=io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'REPORT_RU.txt'
            with contextlib.redirect_stdout(out),russian_console():
                print('Validation verdict:')
                print(r'C:\Дорожки\Reference.wav')
                report.write_text('Validation verdict: BAD',encoding='utf-8')
            self.assertEqual(report.read_text(encoding='utf-8'),'Validation verdict: BAD')
        self.assertIn('Результат независимой проверки:',out.getvalue())
        self.assertIn(r'C:\Дорожки\Reference.wav',out.getvalue())

    def test_input_is_not_patched_or_translated(self):
        with patch('builtins.input',return_value='1') as original:
            with russian_console():
                self.assertIs(builtins.input,original)
                self.assertEqual(input('Введите номер и нажмите Enter: '),'1')
            self.assertIs(builtins.input,original)
            original.assert_called_once_with('Введите номер и нажмите Enter: ')

    def test_reference_target_labels_are_russian_without_changing_codes(self):
        self.assertEqual(russian('Reference:'),'Эталон:')
        self.assertEqual(russian('TARGET / РАССИНХРОН:'),'ИСПРАВЛЯЕМАЯ ДОРОЖКА / РАССИНХРОН:')
        self.assertEqual(russian('Mean residual: 0.006 s'),'Средний остаточный сдвиг: 0.006 s')
        self.assertEqual(russian('BAD'),'НЕ ПРОШЛА ПРОВЕРКУ')
        self.assertNotIn('фильм',russian('7. Разбить фильм на независимые временные окна.'))


if __name__=='__main__':unittest.main()
