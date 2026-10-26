"""BPSK, QPSK, continuous-phase FSK, and OFDM baseband modulators."""

import numpy as np
from scipy.signal import upfirdn

from rflens.synth.pulse import rrc_taps


def generate_psk(
    modulation: str,
    *,
    symbol_rate_hz: float,
    sample_rate_hz: float,
    n_symbols: int,
    alpha: float = 0.35,
    seed: int = 0,
    span_symbols: int = 10,
) -> np.ndarray:
    """Generate pulse-shaped BPSK or QPSK complex baseband samples."""
    sps = _samples_per_symbol(symbol_rate_hz, sample_rate_hz)
    if n_symbols < 1:
        raise ValueError("n_symbols must be positive")
    rng = np.random.default_rng(seed)
    name = modulation.upper()
    if name == "BPSK":
        symbols = (2 * rng.integers(0, 2, n_symbols) - 1).astype(np.complex128)
    elif name == "QPSK":
        bits = rng.integers(0, 4, n_symbols)
        symbols = np.exp(1j * (np.pi / 4 + bits * np.pi / 2))
    else:
        raise ValueError("modulation must be BPSK or QPSK")
    return upfirdn(rrc_taps(alpha, sps, span_symbols), symbols, up=sps).astype(np.complex64)


def generate_fsk(
    *,
    symbol_rate_hz: float,
    sample_rate_hz: float,
    n_symbols: int,
    deviation_hz: float,
    seed: int = 0,
) -> np.ndarray:
    """Generate continuous-phase binary FSK with rectangular symbol pulses."""
    sps = _samples_per_symbol(symbol_rate_hz, sample_rate_hz)
    if n_symbols < 1 or deviation_hz <= 0:
        raise ValueError("n_symbols and deviation_hz must be positive")
    bits = np.random.default_rng(seed).integers(0, 2, n_symbols)
    frequencies = np.repeat((2 * bits - 1) * deviation_hz, sps)
    phase = 2 * np.pi * np.cumsum(frequencies) / sample_rate_hz
    return np.exp(1j * phase).astype(np.complex64)


def generate_ofdm(
    *,
    n_subcarriers: int,
    sample_rate_hz: float,
    subcarrier_spacing_hz: float,
    n_symbols: int,
    cyclic_prefix_samples: int = 0,
    modulation: str = "QPSK",
    seed: int = 0,
) -> np.ndarray:
    """Generate CP-OFDM samples; carrier spacing and sample rate are in Hz."""
    if n_subcarriers < 2 or n_symbols < 1 or cyclic_prefix_samples < 0:
        raise ValueError("invalid OFDM dimensions")
    nfft = int(round(sample_rate_hz / subcarrier_spacing_hz))
    if nfft < n_subcarriers or not np.isclose(nfft * subcarrier_spacing_hz, sample_rate_hz):
        raise ValueError("sample_rate_hz must be an integer FFT multiple of subcarrier spacing")
    if cyclic_prefix_samples > nfft:
        raise ValueError("cyclic_prefix_samples cannot exceed the FFT length")
    rng = np.random.default_rng(seed)
    name = modulation.upper()
    if name == "QPSK":
        symbols = np.exp(
            1j * (np.pi / 4 + rng.integers(0, 4, (n_symbols, n_subcarriers)) * np.pi / 2)
        )
    elif name == "16QAM":
        levels = np.array([-3, -1, 1, 3])
        symbols = (
            rng.choice(levels, (n_symbols, n_subcarriers))
            + 1j * rng.choice(levels, (n_symbols, n_subcarriers))
        ) / np.sqrt(10)
    else:
        raise ValueError("OFDM modulation must be QPSK or 16QAM")
    grid = np.zeros((n_symbols, nfft), dtype=np.complex128)
    start = (nfft - n_subcarriers) // 2
    grid[:, start : start + n_subcarriers] = symbols
    useful = np.fft.ifft(np.fft.ifftshift(grid, axes=1), axis=1) * np.sqrt(nfft)
    if cyclic_prefix_samples:
        useful = np.concatenate((useful[:, -cyclic_prefix_samples:], useful), axis=1)
    return useful.ravel().astype(np.complex64)


def _samples_per_symbol(symbol_rate_hz: float, sample_rate_hz: float) -> int:
    if symbol_rate_hz <= 0 or sample_rate_hz <= 0:
        raise ValueError("sample and symbol rates must be positive Hz")
    ratio = sample_rate_hz / symbol_rate_hz
    if not np.isclose(ratio, round(ratio)) or ratio < 2:
        raise ValueError("sample rate must be an integer multiple of symbol rate and at least 2x")
    return int(round(ratio))
