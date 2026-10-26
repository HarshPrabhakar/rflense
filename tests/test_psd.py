import numpy as np

from rflens.dsp.psd import welch_psd
from rflens.types import IQRecording


def test_tone_peak_and_absolute_frequency_axis():
    fs, fc, offset, n = 1_000_000.0, 433_000_000.0, 125_000.0, 8192
    samples = np.exp(2j * np.pi * offset * np.arange(n) / fs).astype(np.complex64)
    spectrum = welch_psd(IQRecording(samples, fs, fc), nperseg=1024)
    bin_width = fs / 1024
    assert spectrum.freq_hz[np.argmax(spectrum.psd_db)] == fc + offset
    assert np.all(np.diff(spectrum.freq_hz) > 0)
    assert np.isclose(spectrum.rbw_hz, 1.5 * bin_width, rtol=1e-12)


def test_parseval_integrated_psd_matches_mean_power():
    rng = np.random.default_rng(31)
    samples = ((rng.standard_normal(32768) + 1j * rng.standard_normal(32768)) / 2**0.5).astype(np.complex64)
    recording = IQRecording(samples, 2_000_000.0)
    spectrum = welch_psd(recording, nperseg=1024)
    measured = np.sum(10 ** (spectrum.psd_db / 10)) * (recording.fs / 1024)
    assert np.isclose(measured, np.mean(np.abs(samples) ** 2), rtol=0.01)


def test_short_recording_adapts_segment_length():
    rec = IQRecording(np.ones(64, dtype=np.complex64), 1000.0)
    spectrum = welch_psd(rec, nperseg=4096)
    assert len(spectrum.freq_hz) == 64
    assert spectrum.params["nperseg"] == 64
