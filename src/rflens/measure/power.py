"""Signal power and in-band SNR measurements."""

import numpy as np


def signal_power_and_snr(
    psd_linear: np.ndarray, noise_density: float, bin_width_hz: float
) -> tuple[float, float, float]:
    """Return (signal dBFS, level above noise dB, in-band SNR dB).

    PSD and noise density are in full-scale-squared/Hz; bin width is in Hz.
    Noise density is subtracted per bin, then integrated across the supplied band.
    The latter two outputs are equal in this v1 definition: signal power divided
    by estimated noise power over the selected measurement bandwidth.
    """
    p = np.asarray(psd_linear, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("psd_linear must be a non-empty finite non-negative vector")
    if noise_density < 0 or not np.isfinite(noise_density) or bin_width_hz <= 0:
        raise ValueError("noise_density must be non-negative and bin_width_hz positive")
    signal = float(np.sum(np.maximum(p - noise_density, 0.0)) * bin_width_hz)
    noise = float(noise_density * p.size * bin_width_hz)
    tiny = np.finfo(float).tiny
    power_dbfs = 10.0 * np.log10(max(signal, tiny))
    snr_db = 10.0 * np.log10(max(signal, tiny) / max(noise, tiny))
    return float(power_dbfs), float(snr_db), float(snr_db)
