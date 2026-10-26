"""Numerical ground truth for synthetic waveform scenarios."""

from dataclasses import dataclass

import numpy as np

from rflens.dsp.psd import welch_psd
from rflens.measure.bandwidth import obw99
from rflens.types import IQRecording


@dataclass(frozen=True)
class SignalTruth:
    """Ground-truth frequency and bandwidth values; frequencies in Hz."""

    center_frequency_hz: float
    theoretical_bandwidth_hz: float | None
    numerical_obw99_hz: float
    seed: int
    snr_db: float | None = None


def numerical_obw99(
    samples: np.ndarray, *, fs: float, fc: float = 0.0, nperseg: int = 4096
) -> float:
    """Compute noiseless numerical OBW99 in Hz from a generated waveform."""
    rec = IQRecording(np.asarray(samples), fs, fc, "synthetic:truth")
    spectrum = welch_psd(rec, nperseg=nperseg)
    power = 10 ** (spectrum.psd_db / 10)
    bandwidth, _, _ = obw99(spectrum.freq_hz, power)
    return bandwidth


def build_signal_truth(
    modulation: str,
    samples: np.ndarray,
    *,
    fs: float,
    fc: float,
    seed: int,
    symbol_rate_hz: float | None = None,
    alpha: float | None = None,
    deviation_hz: float | None = None,
    n_subcarriers: int | None = None,
    subcarrier_spacing_hz: float | None = None,
    nperseg: int = 4096,
    snr_db: float | None = None,
) -> SignalTruth:
    """Build theoretical and numerical OBW truth for common modulation families."""
    name = modulation.upper()
    if name in ("BPSK", "QPSK"):
        if symbol_rate_hz is None or alpha is None:
            raise ValueError("PSK truth requires symbol_rate_hz and alpha")
        theoretical = symbol_rate_hz * (1.0 + alpha)
    elif name == "FSK":
        if symbol_rate_hz is None or deviation_hz is None:
            raise ValueError("FSK truth requires symbol_rate_hz and deviation_hz")
        theoretical = 2.0 * (deviation_hz + symbol_rate_hz / 2.0)
    elif name == "OFDM":
        if n_subcarriers is None or subcarrier_spacing_hz is None:
            raise ValueError("OFDM truth requires n_subcarriers and subcarrier_spacing_hz")
        theoretical = n_subcarriers * subcarrier_spacing_hz
    else:
        raise ValueError(f"Unsupported modulation for truth: {modulation}")
    return SignalTruth(
        float(fc),
        float(theoretical),
        numerical_obw99(samples, fs=fs, fc=fc, nperseg=nperseg),
        int(seed),
        snr_db,
    )
