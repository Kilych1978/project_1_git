"""Keep absolute reference-window metadata through combined progress counters."""
import math


def timecode(seconds):
    if not math.isfinite(float(seconds)):
        return 'неизвестно'
    millis=max(0,round(float(seconds)*1000))
    seconds,ms=divmod(millis,1000)
    minutes,s=divmod(seconds,60)
    h,m=divmod(minutes,60)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def window(progress, center, length, cached=False):
    if progress is not None and hasattr(progress,'window'):
        progress.window(center-length/2,center+length/2,cached=cached)


class _Slice:
    def __init__(self,callback,offset,total):
        self.callback,self.offset,self.total=callback,offset,total

    def __call__(self,done,total):
        self.callback(self.offset+done,self.total)

    def window(self,start,end,cached=False):
        if hasattr(self.callback,'window'):
            self.callback.window(start,end,cached=cached)


def slice_progress(progress,offset,total):
    return _Slice(progress,offset,total) if progress is not None else None
