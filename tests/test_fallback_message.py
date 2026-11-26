import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from console_ru import russian
from reports import prompt_before_apply


class FallbackMessageTests(unittest.TestCase):
    def test_rejected_map_is_not_presented_as_already_synchronized(self):
        output=io.StringIO()
        selected=SimpleNamespace(target=SimpleNamespace(codec_name='ac3'))
        with contextlib.redirect_stdout(output),patch('builtins.input',return_value='y'):
            accepted=prompt_before_apply(selected,{'offset':np.zeros(2)}, {},{'encoder':'pcm_f32le'},
                guard_report={'mode':'FALLBACK_BASELINE'})
        self.assertTrue(accepted)
        self.assertIn('не прошла защитную проверку',output.getvalue())
        self.assertNotIn('уже практически синхронна',output.getvalue())
        self.assertEqual(russian('Verdict: BAD'),'Оценка: НЕ ПРОШЛА ПРОВЕРКУ')
        self.assertEqual(russian('BAD.wav'),'BAD.wav')


if __name__=='__main__':unittest.main()
