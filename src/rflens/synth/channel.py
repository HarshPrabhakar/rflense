"""Channel effects for synthetic IQ."""

import numpy as np


def add_awgn(
    samples: np.ndarray,
    *,
    snr_db: float,
    seed: int = 0,
    in_bandwidth_hz: float | None = None,
    sample_rate_hz: float | None = None,
) -> np.ndarray:
    """Add complex AWGN at sample SNR, or at a requested in-band SNR in dB.

    When ``in_bandwidth_hz`` is given, ``sample_rate_hz`` is also required and
    total noise power is scaled so its density integrated over that bandwidth
    gives the requested signal-to-noise ratio.
    """
    x = np.asarray(samples, dtype=np.complex128)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(snr_db):
        raise ValueError("samples must be a non-empty vector and snr_db finite")
    power = float(np.mean(np.abs(x) ** 2))
    if in_bandwidth_hz is not None:
        if sample_rate_hz is None or not 0 < in_bandwidth_hz <= sample_rate_hz:
            raise ValueError("in-band AWGN requires 0 < in_bandwidth_hz <= sample_rate_hz")
        noise_power = power * sample_rate_hz / in_bandwidth_hz / 10 ** (snr_db / 10)
    else:
        noise_power = power / 10 ** (snr_db / 10)
    rng = np.random.default_rng(seed)
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
    )
    return (x + noise).astype(np.result_type(samples, np.complex64))


def apply_cfo(
    samples: np.ndarray, *, offset_hz: float, sample_rate_hz: float, phase_rad: float = 0.0
) -> np.ndarray:
    """Apply carrier-frequency and initial-phase offsets."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    n = np.arange(np.asarray(samples).size)
    return np.asarray(samples) * np.exp(
        1j * (2 * np.pi * offset_hz * n / sample_rate_hz + phase_rad)
    )


def apply_iq_imbalance(
    samples: np.ndarray, *, gain_db: float = 0.0, phase_error_rad: float = 0.0, dc: complex = 0j
) -> np.ndarray:
    """Apply gain/phase imbalance and complex DC offset to IQ samples."""
    x = np.asarray(samples)
    q_gain = 10 ** (gain_db / 20)
    i, q = x.real, x.imag * q_gain
    q2 = q * np.cos(phase_error_rad) + i * np.sin(phase_error_rad)
    return (i + 1j * q2 + dc).astype(np.result_type(samples, np.complex64))


def flat_fading(samples: np.ndarray, *, seed: int = 0) -> tuple[np.ndarray, complex]:
    """Apply one seeded unit-mean-power Rayleigh fading coefficient and return it."""
    rng = np.random.default_rng(seed)
    coefficient = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    return np.asarray(samples) * coefficient, complex(coefficient)


def apply_multipath(samples: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply a complex tapped-delay channel by linear convolution."""
    x, h = np.asarray(samples), np.asarray(taps, dtype=np.complex128)
    if x.ndim != 1 or x.size == 0 or h.ndim != 1 or h.size == 0:
        raise ValueError("samples and taps must be non-empty 1-D arrays")
    if not np.all(np.isfinite(h)) or np.sum(np.abs(h) ** 2) == 0:
        raise ValueError("channel taps must be finite and non-zero")
    return np.convolve(x, h).astype(np.result_type(x.dtype, np.complex64))
