import numpy as np
from scipy.signal import welch

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
    samples = ((rng.standard_normal(32768) + 1j * rng.standard_normal(32768)) / 2**0.5).astype(
        np.complex64
    )
    recording = IQRecording(samples, 2_000_000.0)
    spectrum = welch_psd(recording, nperseg=1024)
    measured = np.sum(10 ** (spectrum.psd_db / 10)) * (recording.fs / 1024)
    assert np.isclose(measured, np.mean(np.abs(samples) ** 2), rtol=0.01)


def test_short_recording_adapts_segment_length():
    rec = IQRecording(np.ones(64, dtype=np.complex64), 1000.0)
    spectrum = welch_psd(rec, nperseg=4096)
    assert len(spectrum.freq_hz) == 64
    assert spectrum.params["nperseg"] == 64


def test_chunked_estimator_matches_scipy_welch():
    rng = np.random.default_rng(72)
    samples = (rng.standard_normal(16384) + 1j * rng.standard_normal(16384)).astype(np.complex64)
    rec = IQRecording(samples, 2.4e6, 915e6)
    actual = welch_psd(rec, nperseg=1024, noverlap=768, chunk_segments=7)
    freq, expected = welch(
        samples,
        fs=rec.fs,
        window="hann",
        nperseg=1024,
        noverlap=768,
        detrend=False,
        return_onesided=False,
        scaling="density",
        average="mean",
    )
    np.testing.assert_allclose(actual.freq_hz, np.fft.fftshift(freq) + rec.fc, rtol=0, atol=1e-8)
    np.testing.assert_allclose(10 ** (actual.psd_db / 10), np.fft.fftshift(expected), rtol=1e-7)
