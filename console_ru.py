"""Known terminal labels in Russian; data, paths, reports and input stay intact."""
from contextlib import contextmanager
from functools import wraps
import re
import sys


_LABELS = {
    'Final output': 'Готовый аудиофайл',
    'Final verification': 'Итоговая проверка',
    'Local anchor search complete.': 'Поиск локальных опорных точек завершён.',
    'predicted offset': 'Предсказанный сдвиг', 'actual offset': 'Измеренный сдвиг',
    'residual': 'Остаточный сдвиг', 'correlation': 'Корреляция', 'bidirectional': 'Двусторонняя проверка',
    'BASELINE FALLBACK': 'СОХРАНЕНА БАЗОВАЯ КАРТА',
    'ПОДГОТОВКА AUDIO DATA': 'ПОДГОТОВКА ЗВУКОВЫХ ДАННЫХ',
    'ПОСТРОЕНИЕ REFERENCE WAVEFORM': 'ПОСТРОЕНИЕ ВОЛНЫ ЭТАЛОНА',
    'ПОСТРОЕНИЕ TARGET WAVEFORM': 'ПОСТРОЕНИЕ ВОЛНЫ ИСПРАВЛЯЕМОЙ ДОРОЖКИ',
    'ПОСТРОЕНИЕ REFERENCE ENVELOPE': 'ПОСТРОЕНИЕ ОГИБАЮЩЕЙ ЭТАЛОНА',
    'ПОСТРОЕНИЕ TARGET ENVELOPE': 'ПОСТРОЕНИЕ ОГИБАЮЩЕЙ ИСПРАВЛЯЕМОЙ ДОРОЖКИ',
    'ПОСТРОЕНИЕ WAVEFORM ГРАФИКА': 'ГРАФИК ЗВУКОВЫХ ВОЛН',
    'ПОСТРОЕНИЕ ENVELOPE ГРАФИКА': 'ГРАФИК ОГИБАЮЩЕЙ',
    'ПОСТРОЕНИЕ GLOBAL ALIGNMENT ГРАФИКА': 'ГРАФИК ГЛОБАЛЬНОГО СОВМЕЩЕНИЯ',
    'ПОСТРОЕНИЕ LOCAL ANCHOR ГРАФИКА': 'ГРАФИК ЛОКАЛЬНЫХ ОПОРНЫХ ТОЧЕК',
    'ПОСТРОЕНИЕ CONFIDENCE-AWARE MAP': 'КАРТА С УЧЁТОМ НАДЁЖНОСТИ',
    'СРАВНЕНИЕ BASE / CONFIDENCE-AWARE': 'СРАВНЕНИЕ БАЗОВОЙ КАРТЫ И КАРТЫ С УЧЁТОМ НАДЁЖНОСТИ',
    'СОХРАНЕНИЕ CONFIDENCE CLASSIFICATION': 'СОХРАНЕНИЕ КЛАССИФИКАЦИИ НАДЁЖНОСТИ',
    'СОХРАНЕНИЕ CONFIDENCE-AWARE MAP': 'СОХРАНЕНИЕ КАРТЫ С УЧЁТОМ НАДЁЖНОСТИ',
    'ГРАФИК CONFIDENCE-AWARE MAP': 'ГРАФИК КАРТЫ С УЧЁТОМ НАДЁЖНОСТИ',
    'ГРАФИК BEFORE / AFTER': 'ГРАФИК ДО И ПОСЛЕ КОРРЕКЦИИ',
    'СОХРАНЕНИЕ VALIDATION ТОЧЕК': 'СОХРАНЕНИЕ КОНТРОЛЬНЫХ ТОЧЕК',
    'АНАЛИЗ VALIDATION OUTLIERS': 'АНАЛИЗ ВЫБРОСОВ ПРОВЕРКИ',
    'СОХРАНЕНИЕ ROBUST MAP': 'СОХРАНЕНИЕ ПРОВЕРЕННОЙ КАРТЫ',
    'ПОСТРОЕНИЕ ROBUST MAP ГРАФИКА': 'ГРАФИК ПРОВЕРЕННОЙ КАРТЫ',
    'АНАЛИЗ DRIFT': 'АНАЛИЗ ИЗМЕНЕНИЯ СДВИГА',
    'ПОДТВЕРЖДЕНИЕ ПЕРЕД APPLY': 'ПОДТВЕРЖДЕНИЕ ПЕРЕД ЗАПИСЬЮ',
    'Исходный codec': 'Исходный кодек', 'Исходный sample rate': 'Исходная частота дискретизации',
    'TARGET codec': 'Кодек исправляемой дорожки', 'Выходной encoder': 'Кодировщик результата',
    'Реальные длительности после decode': 'Длительности после декодирования',
    'начальный offset': 'начальный сдвиг', 'конечный offset': 'конечный сдвиг', 'общий drift': 'общее изменение сдвига',
    'Начальный offset': 'Начальный сдвиг', 'Конечный offset': 'Конечный сдвиг',
    'Исходный TARGET НЕ изменён.': 'Исходная исправляемая дорожка НЕ изменена.',
    'Исходный TARGET не изменён.': 'Исходная исправляемая дорожка не изменена.',
    'Декодировать первый канал Reference.': 'Декодировать первый канал эталона.',
    'Декодировать первый канал Target.': 'Декодировать первый канал исправляемой дорожки.',
    'Построить waveform / амплитудную огибающую.': 'Построить звуковую волну и амплитудную огибающую.',
    'Сглаживать карту без уничтожения реального drift.': 'Сглаживать карту с сохранением реального изменения сдвига.',
    'Предлагать применить карту к исходному Target.': 'Предлагать применить карту к исходной исправляемой дорожке.',
    'Следующий этап может переходить к APPLY.': 'Карта готова к этапу записи.',
    'CONFIDENCE-AWARE NONLINEAR SYNCHRONIZATION': 'НЕЛИНЕЙНАЯ СИНХРОНИЗАЦИЯ С ПРОВЕРКОЙ НАДЁЖНОСТИ',
    'CONFIDENCE CLASSIFICATION': 'НАДЁЖНОСТЬ ЛОКАЛЬНЫХ СОВПАДЕНИЙ',
    'VALIDATION DIAGNOSTICS': 'ДИАГНОСТИКА ПРОВЕРКИ',
    'GLOBAL WAVEFORM ALIGNMENT': 'ГЛОБАЛЬНОЕ СОВМЕЩЕНИЕ ЗВУКОВЫХ ВОЛН',
    'GLOBAL RESULTS': 'РЕЗУЛЬТАТ ГЛОБАЛЬНОГО СОВМЕЩЕНИЯ',
    'LOCAL WAVEFORM ANCHOR SEARCH': 'ПОИСК ЛОКАЛЬНЫХ ОПОРНЫХ ТОЧЕК',
    'AUDIO PREPROCESSING COMPLETE': 'ПОДГОТОВКА ЗВУКА ЗАВЕРШЕНА',
    'FINAL ROBUST MAP VERDICT': 'ИТОГ ПРОВЕРКИ КАРТЫ ВРЕМЕНИ',
    'ENVELOPE GRAPH': 'ГРАФИК ОГИБАЮЩЕЙ', 'GLOBAL ALIGNMENT GRAPH': 'ГРАФИК ГЛОБАЛЬНОГО СОВМЕЩЕНИЯ',
    'LOCAL ANCHOR GRAPH': 'ГРАФИК ЛОКАЛЬНЫХ ОПОРНЫХ ТОЧЕК',
    'Reference duration': 'Длительность эталона', 'Target duration': 'Длительность исправляемой дорожки',
    'Создаются только короткие монофрагменты. Синхронизация фильма не запускается.': 'Создаются только короткие монофрагменты. Синхронизация дорожек не запускается.',
    'Разбить фильм на независимые временные окна.': 'Разбить звуковую дорожку на независимые временные окна.',
    'Difference from Reference': 'Отличие от длительности эталона',
    'Final rendered duration': 'Длительность записанного результата', 'Output format': 'Формат результата',
    'Final sync median residual': 'Медиана остаточного сдвига',
    'Final sync mean residual': 'Средний остаточный сдвиг', 'Final sync max residual': 'Наибольший остаточный сдвиг',
    'Reference envelope': 'Огибающая эталона', 'Target envelope': 'Огибающая исправляемой дорожки',
    'Reference decoded': 'Эталон декодирован', 'Target decoded': 'Исправляемая дорожка декодирована',
    'Decode completed: REFERENCE': 'Декодирование эталона завершено',
    'Decode completed: TARGET': 'Декодирование исправляемой дорожки завершено',
    'Decode REFERENCE': 'Декодирование эталона', 'Decode TARGET': 'Декодирование исправляемой дорожки',
    'Robust global offset': 'Устойчивый глобальный сдвиг', 'Global alignment complete': 'Глобальное совмещение завершено',
    'Local anchor search complete': 'Поиск локальных опорных точек завершён',
    'Total points': 'Всего точек', 'Accepted track': 'Принято точек', 'Global offset': 'Глобальный сдвиг',
    'Local points': 'Локальных точек', 'Valid points': 'Надёжных точек', 'Tracked points': 'Прослеженных точек',
    'Correlation': 'Корреляция', 'mean': 'среднее', 'median': 'медиана', 'minimum': 'минимум', 'maximum': 'максимум',
    'WARNING': 'ПРЕДУПРЕЖДЕНИЕ', 'Offset min': 'Минимальный сдвиг', 'Offset median': 'Медианный сдвиг',
    'Offset max': 'Максимальный сдвиг', 'Validation verdict': 'Результат независимой проверки',
    'BEFORE': 'ДО КОРРЕКЦИИ', 'AFTER': 'ПОСЛЕ КОРРЕКЦИИ',
    'Mean error': 'Средняя ошибка', 'Median error': 'Медианная ошибка', 'Max error': 'Наибольшая ошибка',
    'Mean correlation': 'Средняя корреляция', 'Mean residual': 'Средний остаточный сдвиг',
    'Median residual': 'Медианный остаточный сдвиг', 'Max residual': 'Наибольший остаточный сдвиг',
    'P90 residual': '90-й процентиль остатка', 'P95 residual': '95-й процентиль остатка',
    'Residual <= 50ms': 'Остаток не больше 50 мс', 'Verdict': 'Оценка', 'Score': 'Баллы',
    'Validation outlier candidates': 'Кандидатов в выбросы проверки',
    'Candidate HARD outliers': 'Кандидатов в сильные расхождения',
    'Confirmed discrepancies': 'Подтверждённых расхождений',
    'Ambiguous validation points': 'Неоднозначных контрольных точек',
    'HARD candidates requiring manual review': 'Сильных расхождений для ручной проверки',
    'Ambiguous HARD candidates': 'Неоднозначных сильных расхождений',
    'Error improvement': 'Уменьшение ошибки', 'Correlation gain': 'Прирост корреляции',
    'Raw points loaded': 'Исходные точки загружены', 'Metrics saved': 'Метрики сохранены',
    'Median map error': 'Медианная ошибка карты', 'Max map error': 'Наибольшая ошибка карты',
    'Trusted anchors': 'Надёжных опорных точек', 'Base start offset': 'Начальный сдвиг базовой карты',
    'Confidence start offset': 'Начальный сдвиг карты с проверкой надёжности',
    'Base end offset': 'Конечный сдвиг базовой карты',
    'Confidence end offset': 'Конечный сдвиг карты с проверкой надёжности',
    'Mean absolute change': 'Среднее абсолютное изменение', 'Maximum absolute change': 'Наибольшее абсолютное изменение',
    'Safe-map guard status': 'Итог защиты карты', 'Local quarantine centers': 'Центров исключённых участков',
    'Local guard repair passes': 'Проходов локального восстановления',
    'Start offset': 'Начальный сдвиг', 'End offset': 'Конечный сдвиг', 'Map change': 'Изменение карты',
    'Robust verdict': 'Оценка проверенной карты', 'Robust score': 'Баллы проверенной карты',
    'Lossless PCM fallback: YES': 'Сохранение без потерь: PCM',
    'STRONG': 'НАДЁЖНЫЕ', 'ACCEPTABLE': 'ДОПУСТИМЫЕ', 'WEAK': 'СЛАБЫЕ', 'INVALID': 'ОТКЛОНЁННЫЕ',
    'Render segments': 'Отрезков записи', 'Encoder': 'Кодировщик', 'Render': 'Запись',
    'Map points': 'Точек карты', 'Drift graph': 'График изменения сдвига',
    'Validation graph': 'График проверки', 'Error graph': 'График ошибки',
    'Validation': 'Проверка', 'Runs': 'Проходов', 'points': 'точек',
    'codec': 'кодек', 'channels': 'каналов', 'layout': 'схема каналов', 'sample rate': 'частота дискретизации',
    'REFERENCE': 'ЭТАЛОН', 'Reference': 'Эталон', 'TARGET': 'ИСПРАВЛЯЕМАЯ ДОРОЖКА',
    'Target': 'Исправляемая дорожка', 'Difference': 'Разница',
    'BAD': 'НЕ ПРОШЛА ПРОВЕРКУ', 'GOOD': 'ХОРОШО', 'EXCELLENT': 'ОТЛИЧНО',
    'PASSED': 'ПРОЙДЕНО', 'FAILED': 'НЕ ПРОЙДЕНО',
    'NO_CHANGE': 'БЕЗ ОСНОВНОЙ КОРРЕКЦИИ', 'PRESERVE_TARGET': 'СОХРАНЁН ИСХОДНИК',
    'FALLBACK_BASELINE': 'ОСНОВНАЯ КАРТА ОТКЛОНЕНА',
    'FULL_CANDIDATE': 'КАРТА ПОДТВЕРЖДЕНА',
    'MIXED_SAFE_MAP_DENSE': 'СОХРАНЕНЫ ПОДТВЕРЖДЁННЫЕ УЧАСТКИ',
    'LOCAL_QUARANTINE_SAFE_MAP': 'ПРОБЛЕМНЫЕ УЧАСТКИ ИСКЛЮЧЕНЫ',
}

_PREFIX = re.compile(r'(?m)^(?P<space>[ \t]*)(?P<number>\d+\.\s*)?(?P<label>'+ '|'.join(re.escape(x) for x in sorted(_LABELS,key=len,reverse=True))+r')(?=$|[ \t:|])')
_PATH = re.compile(r'^(?:[A-Za-z]:[\\/]|/|\\\\)')
_FILE = re.compile(r'^(?:\d+\.\s*)?[^<>:"\r\n|]+\.(?:wav|wave|flac|dts|ac3|eac3|mka|mp3|m4a|ogg|opus|aac|jpg|png|zip|json|csv|txt)$',re.I)
_STATUS=re.compile(r'(?<=: )(?P<code>BAD|GOOD|EXCELLENT|PASSED|FAILED|NO_CHANGE|PRESERVE_TARGET|FALLBACK_BASELINE|FULL_CANDIDATE|MIXED_SAFE_MAP_DENSE|LOCAL_QUARANTINE_SAFE_MAP)(?=$|\r?\n)')


def russian(text):
    """Translate known label prefixes only; never rewrite path/name chunks."""
    text=str(text)
    if _PATH.match(text.strip()) or _FILE.fullmatch(text.strip()):return text
    text=_PREFIX.sub(lambda match:match['space']+(match['number'] or '')+_LABELS[match['label']],text)
    return _STATUS.sub(lambda match:_LABELS[match['code']],text)


class RussianStream:
    _russian_terminal=True
    def __init__(self,stream):self.stream=stream
    def write(self,text):
        self.stream.write(russian(text))
        return len(text)
    def flush(self):return self.stream.flush()
    def __getattr__(self,key):return getattr(self.stream,key)


@contextmanager
def russian_console():
    if getattr(sys.stdout,'_russian_terminal',False):
        yield
        return
    stdout,stderr=sys.stdout,sys.stderr
    sys.stdout,sys.stderr=RussianStream(stdout),RussianStream(stderr)
    try:yield
    finally:sys.stdout,sys.stderr=stdout,stderr


def russian_session(function):
    @wraps(function)
    def run(*args,**kwargs):
        with russian_console():return function(*args,**kwargs)
    return run
