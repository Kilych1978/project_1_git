"""Presentation helpers only; no alignment, sampling, or acceptance rules."""
import math
import re


def words(seconds, rounding='floor'):
    """Readable movie position without fractional-second micro-window bounds."""
    if seconds is None or not math.isfinite(float(seconds)):
        return 'неизвестно'
    value=max(0.,float(seconds))
    if rounding=='ceil':value=math.ceil(value)
    elif rounding=='nearest':value=math.floor(value+.5)
    else:value=math.floor(value)
    minutes,s=divmod(int(value),60)
    h,m=divmod(minutes,60)
    if h:return f'{h} ч {m:02d} мин {s:02d} с'
    if m:return f'{m} мин {s:02d} с'
    return f'{s} с'


def heading(title):
    """Move pass/grid counters to a dedicated header line, preserving details."""
    operation=[];details=[]
    for item in title.split(' | '):
        found=re.fullmatch(r'проход (\d+)/(\d+)',item,re.I)
        grid=re.fullmatch(r'сетка (\d+)/(\d+)(.*)',item,re.I)
        if found:details.append(f'Проход: {found[1]}, максимум {found[2]}')
        elif grid:
            extra=grid[3].lstrip(', ')
            details.append(f'Сетка: {grid[1]} из {grid[2]}'+('; '+extra if extra else ''))
        else:operation.append(item)
    return ' | '.join(operation),' | '.join(details)


def scope_lines(duration,target_times=None,seconds=False):
    lines=[]
    known=duration is not None and math.isfinite(float(duration)) and duration>0
    if known:lines.append('Длительность дорожки: '+words(duration,'nearest'))
    if seconds:return lines
    if target_times is not None and len(target_times):
        # These are scheduled points, sometimes from several disconnected
        # regions. Do not label their bounding range a continuous audio chunk.
        lo,hi=float(min(target_times)),float(max(target_times))
        if known and lo<=1. and hi>=duration-1.:
            lines.append('Область: весь фильм (контрольные точки)')
        elif lo==hi:lines.append('Контрольная точка: '+words(lo))
        else:lines.append('Контрольные точки: с '+words(lo)+' по '+words(hi,'ceil'))
    else:lines.append('Область: контрольные точки фильма')
    return lines
