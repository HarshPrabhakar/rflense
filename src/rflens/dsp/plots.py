"""Basic plotting helpers for spectra and spectrograms."""
import numpy as np
from matplotlib.axes import Axes
from rflens.types import IQRecording, Spectrum

def plot_psd(spectrum: Spectrum, *, ax: Axes | None = None) -> Axes:
    """Plot PSD in dBFS/Hz against absolute frequency in Hz."""
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    ax.plot(spectrum.freq_hz, spectrum.psd_db)
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (dBFS/Hz)", title="Power spectral density")
    ax.grid(True, alpha=0.25)
    return ax

def plot_spectrogram(recording: IQRecording, *, nperseg: int = 1024,
                     ax: Axes | None = None) -> Axes:
    """Plot IQ spectrogram with absolute frequency (Hz) and time (s) axes."""
    from scipy.signal import spectrogram
    if nperseg < 2: raise ValueError("nperseg must be at least 2 samples")
    size = min(nperseg, recording.samples.size)
    freq, times, power = spectrogram(recording.samples, fs=recording.fs, window="hann",
        nperseg=size, noverlap=size // 2, detrend=False, return_onesided=False,
        scaling="density", mode="psd")
    freq, power = np.fft.fftshift(freq) + recording.fc, np.fft.fftshift(power, axes=0)
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    ax.pcolormesh(times, freq, 10 * np.log10(np.maximum(power, np.finfo(float).tiny)),
                  shading="auto", cmap="viridis")
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", title="IQ spectrogram")
    return ax
