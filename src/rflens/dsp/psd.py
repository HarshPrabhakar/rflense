"""Welch PSD estimation for complex baseband IQ."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import get_window

from rflens.types import IQRecording, Spectrum


def welch_psd(
    recording: IQRecording,
    *,
    nperseg: int = 4096,
    noverlap: int | None = None,
    window: str = "hann",
    chunk_segments: int = 128,
) -> Spectrum:
    """Estimate two-sided PSD in dBFS/Hz; frequencies are absolute Hz and fft-shifted."""
    samples = np.asarray(recording.samples)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("recording samples must be non-empty 1-D")
    if nperseg < 2:
        raise ValueError("nperseg must be at least 2 samples")
    nperseg = min(int(nperseg), samples.size)
    if noverlap is None:
        noverlap = nperseg // 2
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must be in [0, nperseg)")
    if noverlap >= samples.size:
        raise ValueError("noverlap must be smaller than recording length")
    if chunk_segments < 1:
        raise ValueError("chunk_segments must be positive")
    win = get_window(window, nperseg, fftbins=True)
    step = nperseg - noverlap
    segments = sliding_window_view(samples, nperseg)[::step]
    n_averages = segments.shape[0]
    psd_sum = np.zeros(nperseg, dtype=float)
    scale = recording.fs * float(np.sum(np.abs(win) ** 2))
    for start in range(0, n_averages, chunk_segments):
        block = segments[start : start + chunk_segments]
        transforms = np.fft.fft(block * win, axis=-1)
        psd_sum += np.sum(np.abs(transforms) ** 2, axis=0) / scale
    psd = psd_sum / n_averages
    freq = np.fft.fftfreq(nperseg, d=1.0 / recording.fs)
    freq, psd = np.fft.fftshift(freq) + recording.fc, np.fft.fftshift(psd)
    enbw = recording.fs * np.sum(np.abs(win) ** 2) / np.abs(np.sum(win)) ** 2
    psd_db = 10.0 * np.log10(np.maximum(psd, np.finfo(float).tiny))
    return Spectrum(
        freq.astype(float),
        psd_db.astype(float),
        float(enbw),
        n_averages,
        {
            "window": window,
            "nperseg": nperseg,
            "noverlap": noverlap,
            "scaling": "density",
            "units": "dBFS/Hz",
            "chunk_segments": chunk_segments,
        },
    )
