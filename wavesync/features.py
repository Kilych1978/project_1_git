import numpy as np


def envelope(pcm, sample_rate, settings):
    """v3.3 RMS contour, vectorized without a film-sized sample array in RAM."""
    window = round(settings.envelope_window * sample_rate)
    hop = round(settings.envelope_hop * sample_rate)
    if len(pcm) < window:
        raise ValueError("Дорожка слишком короткая для анализа.")
    count = (len(pcm) - window) // hop + 1
    values = np.empty(count, dtype=np.float32)
    for begin in range(0, count, 20000):
        end = min(count, begin + 20000)
        data = np.asarray(pcm[begin * hop:(end - 1) * hop + window], dtype=np.float64)
        cs = np.r_[0., np.cumsum(data * data)]
        starts = np.arange(end - begin) * hop
        values[begin:end] = np.sqrt(np.maximum((cs[starts + window] - cs[starts]) / window, 0))
    # Compress extreme loudness differences, retaining local temporal structure.
    scale = float(np.percentile(values, 75))
    if scale > 1e-12:
        values = np.log1p(values / scale)
    return values
