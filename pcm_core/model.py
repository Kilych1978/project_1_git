"""All offsets use ONE convention: reference_time = target_time + offset.

A discontinuity is represented by separate pieces, never by a large local tempo.
Reference gaps produce silence; target gaps omit unmatched target material.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class Piece:
    target: np.ndarray
    reference: np.ndarray
    corrections: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.target = np.asarray(self.target, dtype=np.float64)
        self.reference = np.asarray(self.reference, dtype=np.float64)
        if (self.target.ndim != 1 or self.target.shape != self.reference.shape
                or len(self.target) < 2
                or not np.isfinite(self.target).all()
                or not np.isfinite(self.reference).all()
                or not (np.diff(self.target) > 0).all()
                or not (np.diff(self.reference) > 0).all()):
            raise ValueError("Некорректная или немонотонная карта времени.")
        self._curves = []
        for curve in self.corrections:
            x = np.asarray(curve["target"], dtype=float)
            y = np.asarray(curve["delta"], dtype=float)
            if (x.ndim != 1 or x.shape != y.shape or len(x) < 2 or not np.isfinite(x).all()
                    or not np.isfinite(y).all() or not (np.diff(x) > 0).all()
                    or x[0] < self.target[0] or x[-1] > self.target[-1]):
                raise ValueError("Некорректная кривая микропоправки.")
            if ((x[0] > self.target[0] and abs(y[0]) > 1e-12)
                    or (x[-1] < self.target[-1] and abs(y[-1]) > 1e-12)):
                raise ValueError("Поправка должна возвращаться к нулю на неподтверждённой границе.")
            self._curves.append((x, PchipInterpolator(x, y, extrapolate=False)))
        # Serialized reference nodes are the FINAL map. Retain the original
        # piecewise-linear base separately so cubic corrections do not smooth
        # untrusted base intervals or cross a rejected-anchor barrier.
        self._base_reference = self.reference - self.correction_at(self.target)
        if self._curves:
            knots = np.unique(np.r_[self.target, *[x for x, _ in self._curves]])
            baseline = np.interp(knots, self.target, self._base_reference)
            corrected = baseline + self.correction_at(knots)
            if not (np.diff(corrected) > 0).all():
                raise ValueError("Микропоправка нарушила монотонность карты.")
            self.target, self.reference, self._base_reference = knots, corrected, baseline

    def correction_at(self, times, derivative=0):
        t = np.asarray(times, dtype=float)
        result = np.zeros(t.shape)
        for x, spline in self._curves:
            valid = (t >= x[0]) & (t <= x[-1])
            result[valid] += spline(t[valid], nu=derivative)
        return result

    def reference_at(self, times):
        return np.interp(times, self.target, self._base_reference) + self.correction_at(times)

    def rate_at(self, times):
        t = np.asarray(times, dtype=float)
        index = np.clip(np.searchsorted(self.target, t, side="right") - 1, 0, len(self.target) - 2)
        slopes = np.diff(self._base_reference) / np.diff(self.target)
        return slopes[index] + self.correction_at(t, derivative=1)

    def inverse_at(self, ref_times):
        r = np.asarray(ref_times, dtype=float)
        t = np.interp(r, self.reference, self.target)
        if self._curves:
            # Invert the ACTUAL cubic map, not an independently fitted spline.
            for _ in range(4):
                t = np.clip(t - (self.reference_at(t) - r) / self.rate_at(t), self.target[0], self.target[-1])
        return t

    def rate_extrema(self):
        lo = np.nextafter(self.target[:-1], self.target[1:])
        hi = np.nextafter(self.target[1:], self.target[:-1])
        # On each joint interval, the sum of cubic second derivatives is
        # linear. Its root is the only possible interior rate extremum.
        d0 = self.correction_at(lo, derivative=2)
        d1 = self.correction_at(hi, derivative=2)
        crossing = (d0 * d1 < 0) & (np.abs(d1 - d0) > 1e-15)
        roots = lo[crossing] - d0[crossing] * (hi[crossing] - lo[crossing]) / (d1[crossing] - d0[crossing])
        return self.rate_at(np.r_[lo, hi, roots])

    @property
    def rate(self):
        return np.diff(self.reference) / np.diff(self.target)


@dataclass
class TimeMap:
    pieces: list[Piece]
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.pieces:
            raise ValueError("Карта времени пуста.")
        for left, right in zip(self.pieces, self.pieces[1:]):
            if (left.target[-1] > right.target[0] + 1e-9
                    or left.reference[-1] > right.reference[0] + 1e-9):
                raise ValueError("Участки карты перекрываются или меняют порядок звука.")

    def validate(self, max_rate_change=0.08):
        for p in self.pieces:
            if np.max(np.abs(p.rate_extrema() - 1)) > max_rate_change + 1e-10:
                raise ValueError("Скачок монтажа нельзя исправлять растяжением звука. Нужна отдельная граница.")
        return self

    @classmethod
    def affine(cls, duration, rate=1.0, offset=0.0):
        return cls([Piece(np.array([0., duration]), np.array([offset, rate * duration + offset]))])

    def forward(self, times):
        x = np.asarray(times, dtype=np.float64)
        out = np.full(x.shape, np.nan)
        for p in self.pieces:
            valid = (x >= p.target[0] - 1e-10) & (x <= p.target[-1] + 1e-10)
            out[valid] = p.reference_at(x[valid])
        return out

    def inverse(self, times):
        x = np.asarray(times, dtype=np.float64)
        out = np.full(x.shape, np.nan)
        for p in self.pieces:
            valid = (x >= p.reference[0] - 1e-10) & (x < p.reference[-1])
            out[valid] = p.inverse_at(x[valid])
        return out

    def inverse_rate(self, times):
        """Derivative dt/dr, without mistaking an edit discontinuity for tempo."""
        x = np.asarray(times, dtype=np.float64)
        out = np.full(x.shape, np.nan)
        for p in self.pieces:
            valid = (x >= p.reference[0]) & (x < p.reference[-1])
            positions = p.inverse_at(x[valid])
            out[valid] = 1 / p.rate_at(positions)
        return out

    def local_rate(self, target_time):
        for p in self.pieces:
            if p.target[0] <= target_time <= p.target[-1]:
                return float(p.rate_at(np.array([target_time]))[0])
        return float("nan")

    def to_dict(self):
        return {"convention": "reference_time = target_time + offset", "notes": self.notes,
                "pieces": [{"target": p.target.tolist(), "reference": p.reference.tolist(),
                            **({"corrections": p.corrections} if p.corrections else {})} for p in self.pieces]}

    @classmethod
    def from_dict(cls, value):
        return cls([Piece(p["target"], p["reference"], p.get("corrections", [])) for p in value["pieces"]], value.get("notes", []))
