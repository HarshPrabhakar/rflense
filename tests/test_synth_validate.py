import numpy as np

from rflens.synth import (SceneSignal, add_awgn, apply_cfo, apply_iq_imbalance,
                          apply_multipath, flat_fading, generate_fsk, generate_ofdm, generate_psk,
                          generate_scene, rrc_taps)
from rflens.validate.metrics import error_metrics
from rflens.validate.montecarlo import run_psk_validation
from rflens.validate.truth import build_signal_truth, numerical_obw99


def test_psk_generators_are_seeded_and_normalized_at_symbol_rate():
    for modulation in ("BPSK", "QPSK"):
        first = generate_psk(modulation, symbol_rate_hz=100_000, sample_rate_hz=800_000,
                             n_symbols=128, seed=7)
        again = generate_psk(modulation, symbol_rate_hz=100_000, sample_rate_hz=800_000,
                             n_symbols=128, seed=7)
        np.testing.assert_array_equal(first, again)
        assert np.iscomplexobj(first) and first.size > 128 * 8
    taps = rrc_taps(0.35, 8)
    assert np.isclose(np.sum(taps ** 2), 1.0)


def test_fsk_and_ofdm_return_expected_lengths():
    fsk = generate_fsk(symbol_rate_hz=100_000, sample_rate_hz=800_000,
                       n_symbols=16, deviation_hz=25_000, seed=2)
    ofdm = generate_ofdm(n_subcarriers=32, sample_rate_hz=640_000,
                         subcarrier_spacing_hz=20_000, n_symbols=4,
                         cyclic_prefix_samples=8, seed=2)
    assert fsk.size == 128
    assert ofdm.size == 4 * (32 + 8)
    assert np.allclose(np.abs(fsk), 1)


def test_channel_effects_are_repeatable_and_awgn_hits_sample_snr():
    samples = np.ones(100_000, dtype=np.complex64)
    noisy = add_awgn(samples, snr_db=10, seed=12)
    estimated_snr = 10 * np.log10(np.mean(np.abs(samples) ** 2) /
                                   np.mean(np.abs(noisy - samples) ** 2))
    assert abs(estimated_snr - 10) < 0.1
    inband_noisy = add_awgn(samples, snr_db=10, seed=12, in_bandwidth_hz=100_000,
                            sample_rate_hz=1_000_000)
    inband_noise = np.mean(np.abs(inband_noisy - samples) ** 2) * 0.1
    assert abs(10 * np.log10(np.mean(np.abs(samples) ** 2) / inband_noise) - 10) < 0.1
    cfo = apply_cfo(samples[:8], offset_hz=125_000, sample_rate_hz=1_000_000)
    assert np.isclose(np.angle(cfo[1] / cfo[0]), 2 * np.pi * 0.125)
    changed = apply_iq_imbalance(samples[:4], gain_db=3, dc=0.2 + 0.1j)
    assert not np.array_equal(changed, samples[:4])
    faded1, coeff1 = flat_fading(samples, seed=9)
    faded2, coeff2 = flat_fading(samples, seed=9)
    np.testing.assert_array_equal(faded1, faded2)
    assert coeff1 == coeff2
    multipath = apply_multipath(samples[:4], np.array([1.0, 0.5j]))
    assert multipath.size == 5


def test_scene_and_numerical_truth_are_deterministic():
    scene = generate_scene([SceneSignal("QPSK", 50_000, -10, 128, 3,
                         symbol_rate_hz=100_000)], sample_rate_hz=800_000,
                         center_frequency_hz=433e6)
    again = generate_scene([SceneSignal("QPSK", 50_000, -10, 128, 3,
                         symbol_rate_hz=100_000)], sample_rate_hz=800_000,
                         center_frequency_hz=433e6)
    np.testing.assert_array_equal(scene.samples, again.samples)
    assert scene.fc == 433e6 and scene.meta["signals"][0]["seed"] == 3
    obw = numerical_obw99(scene.samples, fs=scene.fs, fc=scene.fc, nperseg=256)
    assert 0 < obw < scene.fs
    truth = build_signal_truth("QPSK", scene.samples, fs=scene.fs, fc=scene.fc,
        seed=3, symbol_rate_hz=100_000, alpha=0.35, nperseg=256)
    assert truth.theoretical_bandwidth_hz == 135_000
    assert truth.numerical_obw99_hz == obw


def test_error_metrics_and_small_monte_carlo_smoke():
    result = error_metrics(np.array([9.0, 10.0, 11.0]), 10.0)
    assert result.bias == 0 and np.isclose(result.rmse, np.sqrt(2 / 3))
    trials, summary = run_psk_validation(snr_values_db=(15,), trials_per_snr=2,
        symbol_rate_hz=100_000, sample_rate_hz=800_000, n_symbols=128, nperseg=256, seed=18)
    assert len(trials) == 2 and summary[15.0]["total_trials"] == 2
