"""High-level analysis API."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from rflens.dsp.psd import welch_psd
from rflens.measure.analyze import analyze_spectrum
from rflens.types import AnalysisResult, IQRecording, SignalMeasurement, Spectrum


def analyze(
    recording: IQRecording,
    *,
    nperseg: int = 4096,
    noverlap: int | None = None,
    threshold_db: float = 6.0,
    window: str = "hann",
    min_width_bins: int = 3,
    merge_gap_hz: float = 0.0,
) -> AnalysisResult:
    """Run PSD estimation, noise estimation, detection, and signal measurements."""
    spectrum = welch_psd(recording, nperseg=nperseg, noverlap=noverlap, window=window)
    noise, signals = analyze_spectrum(
        spectrum,
        threshold_db=threshold_db,
        min_width_bins=min_width_bins,
        merge_gap_hz=merge_gap_hz,
    )
    return AnalysisResult(
        recording,
        spectrum,
        noise,
        signals,
        {
            "nperseg": spectrum.params["nperseg"],
            "noverlap": spectrum.params["noverlap"],
            "window": window,
            "threshold_db": threshold_db,
            "min_width_bins": min_width_bins,
            "merge_gap_hz": merge_gap_hz,
        },
    )


def result_from_json(path: str | Path) -> AnalysisResult:
    """Reconstruct reportable result data from RF-Lens JSON without IQ samples."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rec_data, spec_data = data["recording"], data["spectrum"]
    sample_count = int(rec_data.get("n_samples", 0))
    recording = IQRecording(
        np.empty(0, dtype=np.complex64),
        float(rec_data["fs_hz"]),
        float(rec_data["fc_hz"]),
        str(rec_data["source"]),
        {**rec_data.get("meta", {}), "n_samples": sample_count},
    )
    spectrum = Spectrum(
        np.asarray(spec_data["freq_hz"], dtype=float),
        np.asarray(spec_data["psd_db"], dtype=float),
        float(spec_data["rbw_hz"]),
        int(spec_data["n_averages"]),
        spec_data.get("params", {}),
    )
    signals = []
    for row in data.get("signals", []):
        signals.append(
            SignalMeasurement(
                id=int(row["id"]),
                centre_freq_hz=_float_or_nan(row.get("centre_freq_hz")),
                centre_freq_edge_hz=_float_or_nan(row.get("centre_freq_edge_hz")),
                peak_freq_hz=_float_or_nan(row.get("peak_freq_hz")),
                obw99_hz=_float_or_nan(row.get("obw99_hz")),
                bw_3db_hz=_float_or_nan(row.get("bw_3db_hz")),
                bw_26db_hz=_float_or_nan(row.get("bw_26db_hz")),
                f_low_hz=_float_or_nan(row.get("f_low_hz")),
                f_high_hz=_float_or_nan(row.get("f_high_hz")),
                power_dbfs=_float_or_nan(row.get("power_dbfs")),
                level_above_noise_db=_float_or_nan(row.get("level_above_noise_db")),
                snr_db=_float_or_nan(row.get("snr_db")),
                flags=list(row.get("flags", [])),
            )
        )
    floor = float(data["noise_floor_dbfs_per_hz"])
    return AnalysisResult(
        recording, spectrum, floor, signals, data.get("config", {}), data.get("warnings", [])
    )


def _float_or_nan(value: Any) -> float:
    return float("nan") if value is None else float(value)
