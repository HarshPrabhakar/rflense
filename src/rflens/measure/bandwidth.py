"""Occupied and x-dB spectral bandwidth measurements."""

import numpy as np
from scipy.ndimage import uniform_filter1d


def obw99(freq_hz: np.ndarray, psd_linear: np.ndarray, noise_linear: float = 0.0,
          *, tail_fraction: float = 0.005) -> tuple[float, float, float]:
    """Return (bandwidth, low edge, high edge) containing 99% signal power.

    Frequencies are in Hz; PSD and ``noise_linear`` are full-scale-squared/Hz.
    Noise is subtracted and clipped at zero before numerical integration.
    """
    f, p = _validate(freq_hz, psd_linear)
    if noise_linear < 0 or not np.isfinite(noise_linear):
        raise ValueError("noise_linear must be finite and non-negative")
    if not 0 <= tail_fraction < 0.5:
        raise ValueError("tail_fraction must be in [0, 0.5)")
    signal = np.maximum(p - noise_linear, 0.0)
    if not np.any(signal > 0):
        raise ValueError("cannot measure bandwidth with zero noise-subtracted power")
    cumulative = np.zeros(signal.size, dtype=float)
    cumulative[1:] = np.cumsum((signal[:-1] + signal[1:]) * 0.5 * np.diff(f))
    total = cumulative[-1]
    low = float(np.interp(tail_fraction * total, cumulative, f))
    high = float(np.interp((1.0 - tail_fraction) * total, cumulative, f))
    return high - low, low, high


def x_db_bandwidth(freq_hz: np.ndarray, psd_db: np.ndarray, level_db: float,
                   *, smooth_bins: int = 3, noise_linear: float = 0.0
                   ) -> tuple[float, float, float]:
    """Return (bandwidth, low crossing, high crossing) at ``level_db`` below peak.

    Frequency is in Hz and PSD is in dBFS/Hz. Crossings use linear interpolation
    in dB between adjacent bins. The caller should provide one signal with margin.
    """
    f, p = _validate(freq_hz, psd_db)
    if level_db <= 0 or smooth_bins < 1:
        raise ValueError("level_db must be positive and smooth_bins at least one")
    if noise_linear < 0 or not np.isfinite(noise_linear):
        raise ValueError("noise_linear must be finite and non-negative")
    power = np.maximum(10.0 ** (p / 10.0) - noise_linear, np.finfo(float).tiny)
    signal_db = 10.0 * np.log10(power)
    smooth = uniform_filter1d(signal_db, size=min(smooth_bins, p.size), mode="nearest")
    peak = int(np.argmax(smooth))
    target = smooth[peak] - level_db
    lefts = np.flatnonzero(smooth[:peak] <= target)
    rights = np.flatnonzero(smooth[peak + 1:] <= target)
    if not lefts.size or not rights.size:
        raise ValueError(f"spectrum does not cross peak - {level_db:g} dB on both sides")
    left_i = int(lefts[-1])
    right_i = int(peak + 1 + rights[0])
    low = _crossing(f[left_i], f[left_i + 1], smooth[left_i], smooth[left_i + 1], target)
    high = _crossing(f[right_i - 1], f[right_i], smooth[right_i - 1], smooth[right_i], target)
    return high - low, float(low), float(high)


def _validate(freq_hz: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, v = np.asarray(freq_hz, dtype=float), np.asarray(values, dtype=float)
    if f.ndim != 1 or v.shape != f.shape or f.size < 3:
        raise ValueError("frequency and power arrays must be matching 1-D arrays of at least 3 bins")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(v)) or np.any(np.diff(f) <= 0):
        raise ValueError("frequency and power values must be finite with strictly increasing frequency")
    return f, v


def _crossing(f0: float, f1: float, p0: float, p1: float, level: float) -> float:
    if p1 == p0:
        return (f0 + f1) / 2.0
    return float(f0 + (level - p0) * (f1 - f0) / (p1 - p0))
