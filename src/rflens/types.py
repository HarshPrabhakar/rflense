"""Typed data structures shared across RF-Lens layers."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IQRecording:
    """Complex IQ recording; ``fs`` and ``fc`` are in Hz."""
    samples: np.ndarray
    fs: float
    fc: float = 0.0
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Spectrum:
    """Two-sided PSD; ``freq_hz`` is absolute Hz and ``psd_db`` is dBFS/Hz."""
    freq_hz: np.ndarray
    psd_db: np.ndarray
    rbw_hz: float
    n_averages: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalMeasurement:
    """Measured signal quantities, with frequencies and bandwidths in Hz."""
    id: int
    centre_freq_hz: float
    centre_freq_edge_hz: float
    peak_freq_hz: float
    obw99_hz: float
    bw_3db_hz: float
    bw_26db_hz: float
    f_low_hz: float
    f_high_hz: float
    power_dbfs: float
    level_above_noise_db: float
    snr_db: float
    flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisResult:
    """Analysis output for one recording."""
    recording: IQRecording
    spectrum: Spectrum
    noise_floor_db: float
    signals: list[SignalMeasurement] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
