"""Terminal presentation only; no alignment, sampling or acceptance rules."""
import math
import re
from console_ru import russian

_TITLES = {
    'Render завершён': 'Запись завершена',
    'Apply map подготовлена': 'Карта для записи подготовлена',
    'Apply points:': 'Точек карты для записи:',
    'Подготовка adaptive render segments': 'Подготовка отрезков записи',
    'Confidence-aware analysis завершён': 'Анализ надёжности завершён',
    'Запись validation CSV': 'Запись таблицы проверки CSV',
    'СОХРАННОСТЬ БАЗОВОГО РЕЗУЛЬТАТА': 'ПРОВЕРКА СОХРАНЁННОГО ЗВУКА: СРАВНЕНИЕ С БАЗОЙ',
    'ПРОВЕРКА НА ОДИНАКОВОМ ВРЕМЕНИ ЭТАЛОНА': 'ПРОВЕРКА СОХРАНЁННОГО ЗВУКА: ОБЩЕЕ ВРЕМЯ ЭТАЛОНА',
    'НЕЗАВИСИМАЯ ПРОВЕРКА МИКРОПОПРАВОК': 'ПРОВЕРКА МИКРОПОПРАВОК: НЕЗАВИСИМЫЕ ТОЧКИ',
    'ПОИСК ОСТАТОЧНОГО СДВИГА ПО PCM-ОТСЧЁТАМ': 'АНАЛИЗ PCM: ОСТАТОЧНЫЙ СДВИГ',
    'ТОЧЕЧНОЕ УТОЧНЕНИЕ ОСТАВШИХСЯ ПЕРЕХОДОВ': 'УТОЧНЕНИЕ PCM: ОСТАВШИЕСЯ ПЕРЕХОДЫ',
    'ДЕКОДИРОВАНИЕ ВСЕХ КАНАЛОВ ДЛЯ МИКРОУТОЧНЕНИЯ': 'ДЕКОДИРОВАНИЕ PCM: ВСЕ КАНАЛЫ',
    'ПРИМЕНЕНИЕ ПОДТВЕРЖДЁННЫХ МИКРОПОПРАВОК': 'ЗАПИСЬ PCM: ПРОВЕРЕННЫЕ ПОПРАВКИ',
    'СРАВНЕНИЕ ГОТОВОГО ФАЙЛА С ОСНОВНЫМ РЕЗУЛЬТАТОМ': 'ПРОВЕРКА СОХРАНЁННОГО ЗВУКА: ОСНОВНОЙ РЕЗУЛЬТАТ',
    'ЗАЩИТА ОТ РЕГРЕССИЙ: DENSE GRID': 'ПРОВЕРКА УХУДШЕНИЙ: ПЛОТНАЯ СЕТКА',
    'ПОДГОТОВКА APPLY MAP': 'ПОДГОТОВКА КАРТЫ ДЛЯ ЗАПИСИ',
    'ПРИМЕНЕНИЕ ФИНАЛЬНОЙ КАРТЫ': 'ЗАПИСЬ ЗВУКА: ФИНАЛЬНАЯ КАРТА',
    'ДЕКОДИРОВАНИЕ REFERENCE': 'ДЕКОДИРОВАНИЕ ЭТАЛОНА',
    'ДЕКОДИРОВАНИЕ TARGET': 'ДЕКОДИРОВАНИЕ ИСПРАВЛЯЕМОЙ ДОРОЖКИ',
}
_TITLE_PATTERN = re.compile('|'.join(re.escape(k) for k in sorted(_TITLES,key=len,reverse=True)))


def display_text(text):
    """Render progress labels only. No stdout/input monkeypatch or file writes."""
    text=russian(_TITLE_PATTERN.sub(lambda match:_TITLES[match[0]],str(text)))
    return text.replace(' | ETA ',' | Осталось ').replace(' | duration unknown',' | длительность неизвестна')


def words(seconds, rounding='floor'):
    """Fixed HH:MM:SS clock. Diagnostic millisecond timecodes stay separate."""
    if seconds is None or not math.isfinite(float(seconds)):
        return 'неизвестно'
    value=max(0.,float(seconds))
    if rounding=='ceil':value=math.ceil(value)
    elif rounding=='nearest':value=math.floor(value+.5)
    else:value=math.floor(value)
    minutes,s=divmod(int(value),60);h,m=divmod(minutes,60)
    return f'{h:02}:{m:02}:{s:02}'


def heading(title):
    operation=[];details=[]
    for item in title.split(' | '):
        found=re.fullmatch(r'проход (\d+)/(\d+)',item,re.I)
        grid=re.fullmatch(r'сетка (\d+)/(\d+)(.*)',item,re.I)
        window=re.fullmatch(r'окно (.+)',item,re.I)
        if found:details.append(f'Проход: {found[1]}/{found[2]} (предел)')
        elif grid:
            extra=grid[3].lstrip(', ')
            details.append(f'Сетка: {grid[1]}/{grid[2]}'+('; '+extra if extra else ''))
        elif window:details.append('Окно: '+window[1])
        else:operation.append(display_text(item))
    return ' | '.join(operation),' | '.join(details)


def scope_lines(duration,target_times=None,seconds=False):
    known=duration is not None and math.isfinite(float(duration)) and duration>0
    duration_text=' | Длительность: '+words(duration,'nearest') if known else ''
    if seconds:return ['Область: все каналы'+duration_text]
    if target_times is not None and len(target_times):
        lo,hi=float(min(target_times)),float(max(target_times))
        if known and lo<=1. and hi>=duration-1.:
            return ['Область: вся эталонная дорожка; контрольные точки'+duration_text]
        if lo==hi:return ['Контрольная точка: '+words(lo)+duration_text]
        return ['Контрольные точки: '+words(lo)+' - '+words(hi,'ceil')+duration_text]
    return ['Область: контрольные точки эталонной дорожки'+duration_text]
