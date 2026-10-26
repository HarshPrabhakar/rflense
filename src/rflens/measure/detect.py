"""Threshold-based spectral region detection."""

from dataclasses import dataclass

import numpy as np

from rflens.types import Spectrum


@dataclass(frozen=True)
class SignalRegion:
    """Inclusive PSD bin span and flags for one detected signal."""

    start: int
    stop: int
    flags: tuple[str, ...] = ()

    @property
    def width_bins(self) -> int:
        return self.stop - self.start + 1


def detect_regions(
    spectrum: Spectrum,
    noise_floor_db: float,
    *,
    threshold_db: float = 6.0,
    min_width_bins: int = 3,
    merge_gap_hz: float = 0.0,
    dc_guard_hz: float | None = None,
) -> list[SignalRegion]:
    """Detect contiguous PSD regions above noise floor plus ``threshold_db``.

    ``merge_gap_hz`` and ``dc_guard_hz`` are in Hz. Frequency edge overlap is
    flagged because an out-of-band signal may be truncated by the capture.
    """
    freq = np.asarray(spectrum.freq_hz)
    psd = np.asarray(spectrum.psd_db)
    if freq.ndim != 1 or psd.shape != freq.shape or freq.size < 2:
        raise ValueError("spectrum frequency and PSD arrays must be matching 1-D arrays")
    if min_width_bins < 1 or merge_gap_hz < 0:
        raise ValueError("min_width_bins must be positive and merge_gap_hz non-negative")
    above = psd >= noise_floor_db + threshold_db
    edges = np.diff(np.r_[False, above, False].astype(np.int8))
    spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))
    df = float(np.median(np.diff(freq)))
    gap_bins = int(np.floor(merge_gap_hz / df))
    merged: list[list[int]] = []
    for start, stop in spans:
        if merged and start - merged[-1][1] - 1 <= gap_bins:
            merged[-1][1] = int(stop)
        else:
            merged.append([int(start), int(stop)])
    result = []
    for start, stop in merged:
        if stop - start + 1 < min_width_bins:
            continue
        flags = []
        if start == 0 or stop == freq.size - 1:
            flags.append("near_band_edge")
        tuned_center = (freq[0] + freq[-1]) / 2.0
        if (
            dc_guard_hz is not None
            and dc_guard_hz > 0
            and freq[start] <= tuned_center + dc_guard_hz
            and freq[stop] >= tuned_center - dc_guard_hz
        ):
            flags.append("dc_spike_overlap")
        result.append(SignalRegion(start, stop, tuple(flags)))
    return result
