"""Combine detection and measurements for an existing spectrum."""

import numpy as np
from scipy.signal import find_peaks

from rflens.measure.bandwidth import obw99, x_db_bandwidth
from rflens.measure.centre import spectral_centres
from rflens.measure.detect import SignalRegion, detect_regions
from rflens.measure.noise import estimate_noise_floor
from rflens.measure.power import signal_power_and_snr
from rflens.types import SignalMeasurement, Spectrum


def analyze_spectrum(
    spectrum: Spectrum,
    *,
    threshold_db: float = 6.0,
    edge_fraction: float = 0.05,
    min_width_bins: int = 3,
    merge_gap_hz: float = 0.0,
    margin_bins: int | None = 4,
    dc_guard_hz: float | None = None,
    centroid_tolerance_fraction: float = 0.1,
) -> tuple[float, list[SignalMeasurement]]:
    """Estimate the noise floor and measure all detections in ``spectrum``.

    Spectrum frequencies are in Hz and PSD is in dBFS/Hz. Returns floor dBFS/Hz
    and measurements; SNR is measured over the expanded detected region.
    """
    floor_db = estimate_noise_floor(spectrum, edge_fraction=edge_fraction)
    regions = detect_regions(
        spectrum,
        floor_db,
        threshold_db=threshold_db,
        min_width_bins=min_width_bins,
        merge_gap_hz=merge_gap_hz,
        dc_guard_hz=dc_guard_hz,
    )
    return floor_db, [
        _measure_region(spectrum, region, floor_db, margin_bins, idx, centroid_tolerance_fraction)
        for idx, region in enumerate(regions, start=1)
    ]


def _measure_region(
    spectrum: Spectrum,
    region: SignalRegion,
    floor_db: float,
    margin_bins: int | None,
    signal_id: int,
    tolerance: float,
) -> SignalMeasurement:
    f, p_db = np.asarray(spectrum.freq_hz), np.asarray(spectrum.psd_db)
    if margin_bins is None:
        margin_bins = 4
    if margin_bins < 0:
        raise ValueError("margin_bins must be non-negative")
    start, stop = max(0, region.start - margin_bins), min(len(f) - 1, region.stop + margin_bins)
    indices = slice(start, stop + 1)
    f_band, p_db_band = f[indices], p_db[indices]
    p_band = 10 ** (p_db_band / 10.0)
    noise = 10 ** (floor_db / 10.0)
    flags = list(region.flags)
    try:
        obw, low, high = obw99(f_band, p_band, noise, smooth_bins=9)
    except ValueError:
        obw, low, high = float("nan"), float(f_band[0]), float(f_band[-1])
        flags.append("obw_unavailable")
    try:
        bw3, _, _ = x_db_bandwidth(f_band, p_db_band, 3.0, noise_linear=noise)
    except ValueError:
        bw3 = float("nan")
        flags.append("bw_3db_unavailable")
    try:
        bw26, _, _ = x_db_bandwidth(f_band, p_db_band, 26.0, noise_linear=noise)
    except ValueError:
        bw26 = float("nan")
        flags.append("bw_26db_unavailable")
    try:
        centroid, peak = spectral_centres(f_band, p_band, noise)
    except ValueError:
        peak = float(f_band[int(np.argmax(p_band))])
        centroid = peak
        flags.append("centre_unavailable")
    df = float(np.median(np.diff(f_band)))
    power_dbfs, level_noise, snr = signal_power_and_snr(p_band, noise, df)
    centre_edges = (low + high) / 2.0
    if (
        np.isfinite(obw)
        and max(abs(centroid - peak), abs(centroid - centre_edges)) > max(obw, df) * tolerance
    ):
        flags.append("centre_estimates_disagree")
    noise_subtracted_db = 10.0 * np.log10(np.maximum(p_band - noise, np.finfo(float).tiny))
    peak_indices, peak_props = find_peaks(noise_subtracted_db, prominence=3.0)
    if peak_indices.size >= 2 and np.count_nonzero(peak_props["prominences"] >= 3.0) >= 2:
        flags.append("non_unimodal")
    if snr < 6.0:
        flags.append("low_snr")
    return SignalMeasurement(
        signal_id,
        centroid,
        centre_edges,
        peak,
        obw,
        bw3,
        bw26,
        low,
        high,
        power_dbfs,
        level_noise,
        snr,
        flags,
    )
