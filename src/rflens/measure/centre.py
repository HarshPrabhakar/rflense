"""Spectral center-frequency estimates."""

import numpy as np


def spectral_centres(
    freq_hz: np.ndarray, psd_linear: np.ndarray, noise_linear: float = 0.0
) -> tuple[float, float]:
    """Return power-weighted centroid and peak frequency, both in Hz."""
    f, p = np.asarray(freq_hz, dtype=float), np.asarray(psd_linear, dtype=float)
    if f.ndim != 1 or f.size < 1 or p.shape != f.shape:
        raise ValueError("frequency and PSD arrays must have matching 1-D shapes")
    weights = np.maximum(p - noise_linear, 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("cannot estimate center with zero noise-subtracted power")
    centroid = float(np.sum(f * weights) / total)
    peak = float(f[int(np.argmax(p))])
    return centroid, peak
