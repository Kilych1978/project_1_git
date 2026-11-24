"""Terminal presentation only; no alignment, sampling or acceptance rules."""
import math
import re

from console_en import english


def words(seconds, rounding='floor'):
    """Fixed HH:MM:SS clock. Diagnostic millisecond timecodes stay separate."""
    if seconds is None or not math.isfinite(float(seconds)):
        return 'unknown'
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
        if found:details.append(f'Pass: {found[1]}/{found[2]} (limit)')
        elif grid:
            extra=english(grid[3].lstrip(', '))
            details.append(f'Grid: {grid[1]}/{grid[2]}'+('; '+extra if extra else ''))
        elif window:details.append('Window: '+english(window[1]))
        else:operation.append(english(item))
    return ' | '.join(operation),' | '.join(details)


def scope_lines(duration,target_times=None,seconds=False):
    known=duration is not None and math.isfinite(float(duration)) and duration>0
    duration_text=' | Track: '+words(duration,'nearest') if known else ''
    if seconds:return ['Scope: all channels'+duration_text]
    if target_times is not None and len(target_times):
        lo,hi=float(min(target_times)),float(max(target_times))
        if known and lo<=1. and hi>=duration-1.:
            return ['Scope: full reference track; scheduled checkpoints'+duration_text]
        if lo==hi:return ['Checkpoint: '+words(lo)+duration_text]
        return ['Checkpoints: '+words(lo)+' - '+words(hi,'ceil')+duration_text]
    return ['Scope: scheduled reference checkpoints'+duration_text]
