"""Normalized FFT search inherited from v3.3; bounded-memory bandlimited rendering."""
import numpy as np
from scipy.signal import correlate


def ncc_valid(search, template):
    """Pearson correlation at every valid placement; silence cannot be a match."""
    a = np.asarray(search, dtype=np.float64)
    b = np.asarray(template, dtype=np.float64)
    if len(b) < 16 or len(a) < len(b):
        return np.empty(0)
    b = b - b.mean()
    energy_b = np.dot(b, b)
    if energy_b < 1e-16:
        return np.zeros(len(a) - len(b) + 1)
    cross = correlate(a, b, mode="valid", method="fft")
    s = np.r_[0., np.cumsum(a)]
    s2 = np.r_[0., np.cumsum(a * a)]
    n = len(b)
    sums = s[n:] - s[:-n]
    energy = np.maximum((s2[n:] - s2[:-n]) - sums * sums / n, 0)
    denom = np.sqrt(energy * energy_b)
    result = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 1e-14)
    return np.clip(result, -1, 1)


def peak(curve, exclusion=1, absolute=False):
    if len(curve) < 3:
        return None
    scores = np.abs(curve) if absolute else np.asarray(curve)
    index = int(np.argmax(scores))
    if index == 0 or index == len(scores) - 1:
        return None
    y0, y1, y2 = scores[index - 1:index + 2]
    denominator = y0 - 2 * y1 + y2
    fraction = float(np.clip(0.5 * (y0 - y2) / denominator, -.5, .5)) if denominator < -1e-12 else 0.
    mask = np.abs(np.arange(len(scores)) - index) > exclusion
    second = float(np.max(scores[mask])) if mask.any() else 0.
    return {"index": index, "fraction": fraction, "score": float(y1),
            "margin": float(y1 - second), "polarity": 1 if curve[index] >= 0 else -1}


def sinc_sample(source, positions, radius=16, cutoff=1.0):
    """Lanczos windowed sinc. Same kernel for micro A/B and final multichannel output.

    Low-pass cutoff prevents aliasing when a map compresses source time.
    NaN/outside positions are silence. Integer positions at unit rate copy exactly.
    """
    mono = source.ndim == 1
    x = source[:, None] if mono else source
    pos = np.asarray(positions, dtype=np.float64)
    out = np.zeros((len(pos), x.shape[1]), dtype=np.float32)
    valid = np.isfinite(pos) & (pos >= 0) & (pos <= len(x) - 1)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return out[:, 0] if mono else out
    p = pos[valid]
    integer = np.rint(p).astype(np.int64)
    c = np.broadcast_to(np.asarray(cutoff), pos.shape)[valid]
    direct = (np.abs(p - integer) < 1e-6) & (c >= 1 - 1e-12)
    out[indices[direct]] = x[integer[direct]]
    left = ~direct
    if left.any():
        use = indices[left]
        p = p[left]
        c = c[left]
        floor = np.floor(p).astype(np.int64)
        result = np.zeros((len(p), x.shape[1]), dtype=np.float64)
        weights = np.zeros(len(p))
        for tap in range(-radius + 1, radius + 1):
            j = floor + tap
            dx = p - j
            w = c * np.sinc(c * dx) * np.sinc(dx / radius)
            good = (j >= 0) & (j < len(x))
            result[good] += np.asarray(x[j[good]], dtype=np.float64) * w[good, None]
            # Outside the source is zero padding, not renormalized missing audio.
            weights += w
        result /= np.where(np.abs(weights) > 1e-12, weights, 1)[:, None]
        out[use] = result.astype(np.float32)
    return out[:, 0] if mono else out


def mapped_window(source, mapping, ref_center, length, sample_rate, radius=16):
    # Round the REFERENCE start once. Returned start makes sample rounding explicit.
    start = int(round(ref_center * sample_rate - length / 2))
    ref_times = (start + np.arange(length)) / sample_rate
    positions = mapping.inverse(ref_times) * sample_rate
    finite = np.isfinite(positions)
    if not finite.all():
        return None, start
    step = mapping.inverse_rate(ref_times)
    cutoff = np.minimum(1., 1. / np.maximum(step, 1.))
    return sinc_sample(source, positions, radius, cutoff), start
