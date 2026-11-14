"""Restore complete reference windows, including every inner sampling layer.

The nominal clock stays fixed while unsafe audio is replaced by the protected
plan. Only amplitude is crossfaded; a rollback never creates a delay ramp.
"""
import numpy as np


def extent(mapping, other, times):
    if mapping is other:
        return 0.
    if isinstance(other,RestoredMap) and not isinstance(mapping,RestoredMap):
        # A protected restored plan can share a nominal clock with different
        # actual PCM. Compare its sampled branches, not just that clock.
        return other.extent(mapping,times)
    if hasattr(mapping, 'extent'):
        return mapping.extent(other, times)
    if hasattr(other, 'extent'):
        return other.extent(mapping, times)
    a, b = mapping.inverse(times), other.inverse(times)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.any(np.isfinite(a) != np.isfinite(b)):
        return float('inf')
    return float(np.max(np.abs(a[valid]-b[valid]))) if valid.any() else 0.


class RestoredMap:
    def __init__(self, candidate, fallback, intervals, fade=.012):
        self.candidate, self.fallback = candidate, fallback
        self.fade = float(fade)
        merged = []
        for lo, hi in sorted((float(a), float(b)) for a, b in intervals):
            if not np.isfinite([lo, hi]).all() or hi <= lo:
                raise ValueError('Некорректное окно восстановления PCM.')
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        self.intervals = merged
        self._starts = np.array([x[0] for x in merged])
        self._ends = np.array([x[1] for x in merged])
        self.pieces = candidate.pieces
        self.notes = [*candidate.notes, 'Full PCM windows restored to the protected plan.']

    def forward(self, times): return self.candidate.forward(times)
    def inverse(self, times): return self.candidate.inverse(times)
    def inverse_with_rate(self, times): return self.candidate.inverse_with_rate(times)
    def inverse_rate(self, times): return self.candidate.inverse_rate(times)
    def local_rate(self, t): return self.candidate.local_rate(t)

    def local_intervals(self, lo, hi):
        start=np.searchsorted(self._ends,lo-self.fade,side='left')
        end=np.searchsorted(self._starts,hi+self.fade,side='right')
        return self.intervals[start:end]

    def weight(self, times):
        times = np.asarray(times)
        weights = np.zeros(times.shape)
        if not times.size: return weights
        finite=times[np.isfinite(times)]
        if not finite.size:return weights
        for lo, hi in self.local_intervals(np.min(finite),np.max(finite)):
            distance = np.maximum(np.maximum(lo-times, times-hi), 0.)
            weights = np.maximum(weights, np.clip(1.-distance/self.fade, 0., 1.))
        return weights

    def sample(self, source, times, sample_rate, radius):
        from .blend import sample_map
        w = self.weight(times)
        if np.all(w == 1): return sample_map(source, self.fallback, times, sample_rate, radius)
        values = sample_map(source, self.candidate, times, sample_rate, radius)
        active = w > 0
        if active.any():
            restored = sample_map(source, self.fallback, np.asarray(times)[active], sample_rate, radius)
            weights = w[active] if values.ndim == 1 else w[active, None]
            values[active] = values[active]*(1-weights)+restored*weights
        return values

    def extent(self, other, times):
        if self is other: return 0.
        w = self.weight(times)
        a = extent(self.candidate, other, times) if np.any(w < 1) else 0.
        b = extent(self.fallback, other, times) if np.any(w > 0) else 0.
        return a*float(np.max(1-w)) + b*float(np.max(w)) if len(w) else 0.

    def reduced(self, bad_reference_times, half_window=.06, protected_base=None):
        fallback = self.fallback if protected_base is None else protected_base
        same = fallback is self.fallback
        intervals = list(self.intervals) if same else []
        added = [[float(t-half_window), float(t+half_window)] for t in bad_reference_times]
        return RestoredMap(self.candidate if same else self, fallback, intervals+added, self.fade), added

    def validate(self, max_rate_change=.08):
        if not np.isfinite(self.fade) or self.fade <= 0: raise ValueError('Некорректный переход PCM.')
        self.candidate.validate(max_rate_change)
        self.fallback.validate(max_rate_change)
        return self

    def to_dict(self):
        return {'representation': 'pcm_window_restore', 'candidate': self.candidate.to_dict(),
                'fallback': self.fallback.to_dict(), 'intervals': self.intervals, 'fade_sec': self.fade}


def restore_failures(candidate, fallback, failures, padding=0.):
    intervals = []
    for row in failures:
        half = row.get('window_sec', row.get('verification_window_sec', .125))/2 + .001 + padding
        intervals.append([row['reference_time']-half, row['reference_time']+half])
    if isinstance(candidate, RestoredMap) and candidate.fallback is fallback:
        return RestoredMap(candidate.candidate, fallback, candidate.intervals+intervals, candidate.fade)
    return RestoredMap(candidate, fallback, intervals)
