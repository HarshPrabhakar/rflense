import numpy as np

from rflens.measure import (
    analyze_spectrum,
    detect_regions,
    estimate_noise_floor,
    obw99,
    spectral_centres,
    x_db_bandwidth,
)
from rflens.measure.power import signal_power_and_snr
from rflens.types import Spectrum


def gaussian_spectrum():
    fs, nfft, fc = 1_000_000.0, 2048, 433_000_000.0
    f = fc + np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    df = fs / nfft
    sigma = 20 * df
    noise = 1e-9
    signal = 1e-6 * np.exp(-0.5 * ((f - (fc + 100 * df)) / sigma) ** 2)
    spectrum = Spectrum(
        f,
        10 * np.log10(noise + signal),
        1.5 * df,
        32,
        {"window": "hann", "nperseg": nfft, "noverlap": nfft // 2},
    )
    return spectrum, noise, sigma, df


def test_noise_estimate_and_detection_of_known_gaussian():
    spectrum, noise, _, df = gaussian_spectrum()
    floor = estimate_noise_floor(spectrum)
    assert abs(floor - 10 * np.log10(noise)) < 0.1
    regions = detect_regions(spectrum, floor, threshold_db=6, min_width_bins=3)
    assert len(regions) == 1
    assert regions[0].width_bins * df > 10 * df


def test_obw_and_xdb_bandwidth_match_gaussian_ground_truth():
    spectrum, noise, sigma, _ = gaussian_spectrum()
    p = 10 ** (spectrum.psd_db / 10)
    bw, low, high = obw99(spectrum.freq_hz, p, noise)
    expected_obw = 2 * sigma * 2.575829
    assert abs(bw / expected_obw - 1) < 0.01
    assert np.isclose((low + high) / 2, spectrum.freq_hz[np.argmax(p)], atol=1)
    bw3, _, _ = x_db_bandwidth(spectrum.freq_hz, spectrum.psd_db, 3, noise_linear=noise)
    expected_bw3 = 2 * sigma * np.sqrt(2 * np.log(10) * 3 / 10)
    assert abs(bw3 / expected_bw3 - 1) < 0.03
    bw26, _, _ = x_db_bandwidth(spectrum.freq_hz, spectrum.psd_db, 26, noise_linear=noise)
    expected_bw26 = 2 * sigma * np.sqrt(2 * np.log(10) * 26 / 10)
    assert abs(bw26 / expected_bw26 - 1) < 0.03


def test_centroid_and_integrated_power_for_uniform_band():
    f = np.arange(100.0, 111.0)
    p = np.full(f.shape, 1e-3)
    centroid, peak = spectral_centres(f, p)
    assert centroid == 105.0
    assert peak == 100.0


def test_power_and_in_band_snr_have_known_value():
    psd = np.full(100, 2e-6)
    floor = 1e-6
    power_dbfs, above_noise_db, snr_db = signal_power_and_snr(psd, floor, 1000.0)
    assert np.isclose(power_dbfs, 10 * np.log10(0.1))
    assert np.isclose(above_noise_db, 0.0)
    assert np.isclose(snr_db, 0.0)


def test_detector_marks_capture_edge_regions():
    f = np.arange(-5.0, 6.0)
    p = np.full(11, -100.0)
    p[0:3] = -80.0
    spec = Spectrum(f, p, 1.0, 1, {})
    regions = detect_regions(spec, -100.0, threshold_db=6.0)
    assert regions[0].flags == ("near_band_edge",)


def test_measurements_from_detected_gaussian():
    spectrum, noise, sigma, _ = gaussian_spectrum()
    floor, signals = analyze_spectrum(spectrum, threshold_db=6)
    assert abs(floor - 10 * np.log10(noise)) < 0.1
    assert len(signals) == 1
    measured = signals[0]
    assert abs(measured.obw99_hz / (2 * sigma * 2.575829) - 1) < 0.05
    assert (
        abs(
            measured.centre_freq_hz
            - (
                spectrum.freq_hz[len(spectrum.freq_hz) // 2]
                + 100 * (spectrum.freq_hz[1] - spectrum.freq_hz[0])
            )
        )
        < 2
    )
    assert measured.snr_db > 20
    assert measured.power_dbfs < 0
