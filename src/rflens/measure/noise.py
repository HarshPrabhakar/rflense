"""Robust spectral noise density estimation."""

import numpy as np
from scipy.stats import gamma

from rflens.types import Spectrum


def estimate_noise_floor(
    spectrum: Spectrum,
    *,
    edge_fraction: float = 0.05,
    mask_threshold_db: float = 6.0,
    dilation_bins: int = 3,
    iterations: int = 2,
) -> float:
    """Estimate noise density in dBFS/Hz from PSD bins.

    Excludes the outer ``edge_fraction`` at each band edge and corrects the median
    upward to a mean-power estimate using a Gamma periodogram model with the
    recorded Welch segment count as an approximate number of averages.
    """
    psd = 10.0 ** (np.asarray(spectrum.psd_db, dtype=float) / 10.0)
    if psd.ndim != 1 or psd.size < 4 or not np.all(np.isfinite(psd)):
        raise ValueError("spectrum must contain at least four finite PSD bins")
    if not 0 <= edge_fraction < 0.5:
        raise ValueError("edge_fraction must be in [0, 0.5)")
    if dilation_bins < 0 or iterations < 0:
        raise ValueError("dilation_bins and iterations must be non-negative")
    n = psd.size
    edge = int(np.floor(n * edge_fraction))
    eligible = np.zeros(n, dtype=bool)
    eligible[edge : n - edge if edge else n] = True
    averages = max(1, int(spectrum.n_averages))
    median_factor = float(gamma.median(a=averages, scale=1.0 / averages))
    correction = 1.0 / median_factor
    estimate = float(np.median(psd[eligible])) * correction
    for _ in range(iterations):
        above = psd > estimate * 10.0 ** (mask_threshold_db / 10.0)
        if dilation_bins:
            kernel = np.ones(2 * dilation_bins + 1, dtype=int)
            above = np.convolve(above.astype(int), kernel, mode="same") > 0
        candidate = eligible & ~above
        if np.count_nonzero(candidate) < max(4, n // 20):
            break
        estimate = float(np.median(psd[candidate])) * correction
    return float(10.0 * np.log10(max(estimate, np.finfo(float).tiny)))
