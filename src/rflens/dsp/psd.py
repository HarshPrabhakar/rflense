"""Welch PSD estimation for complex baseband IQ."""
import numpy as np
from scipy.signal import get_window, welch
from rflens.types import IQRecording, Spectrum

def welch_psd(recording: IQRecording, *, nperseg: int = 4096, noverlap: int | None = None,
              window: str = "hann") -> Spectrum:
    """Estimate two-sided PSD in dBFS/Hz; frequencies are absolute Hz and fft-shifted."""
    samples = np.asarray(recording.samples)
    if samples.ndim != 1 or samples.size == 0: raise ValueError("recording samples must be non-empty 1-D")
    if nperseg < 2: raise ValueError("nperseg must be at least 2 samples")
    nperseg = min(int(nperseg), samples.size)
    if noverlap is None: noverlap = nperseg // 2
    if noverlap < 0 or noverlap >= nperseg: raise ValueError("noverlap must be in [0, nperseg)")
    if noverlap >= samples.size: raise ValueError("noverlap must be smaller than recording length")
    freq, psd = welch(samples, fs=recording.fs, window=window, nperseg=nperseg,
        noverlap=noverlap, detrend=False, return_onesided=False, scaling="density", average="mean")
    freq, psd = np.fft.fftshift(freq) + recording.fc, np.fft.fftshift(psd)
    win = get_window(window, nperseg, fftbins=True)
    enbw = recording.fs * np.sum(np.abs(win) ** 2) / np.abs(np.sum(win)) ** 2
    step = nperseg - noverlap
    averages = max(1, 1 + (samples.size - nperseg) // step)
    psd_db = 10.0 * np.log10(np.maximum(psd, np.finfo(float).tiny))
    return Spectrum(freq.astype(float), psd_db.astype(float), float(enbw), averages,
                    {"window": window, "nperseg": nperseg, "noverlap": noverlap,
                     "scaling": "density", "units": "dBFS/Hz"})
