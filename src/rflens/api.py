"""High-level analysis API."""

from pathlib import Path
from typing import Any

import numpy as np

from rflens.dsp.psd import welch_psd
from rflens.measure.analyze import analyze_spectrum
from rflens.types import AnalysisResult, IQRecording, SignalMeasurement, Spectrum


def analyze(recording: IQRecording, *, nperseg: int = 4096, noverlap: int | None = None,
            threshold_db: float = 6.0, window: str = "hann",
            min_width_bins: int = 3, merge_gap_hz: float = 0.0) -> AnalysisResult:
    """Run PSD estimation, noise estimation, detection, and signal measurements."""
    spectrum = welch_psd(recording, nperseg=nperseg, noverlap=noverlap, window=window)
    noise, signals = analyze_spectrum(spectrum, threshold_db=threshold_db,
        min_width_bins=min_width_bins, merge_gap_hz=merge_gap_hz)
    return AnalysisResult(recording, spectrum, noise, signals,
        {"nperseg": spectrum.params["nperseg"], "noverlap": spectrum.params["noverlap"],
         "window": window, "threshold_db": threshold_db,
         "min_width_bins": min_width_bins, "merge_gap_hz": merge_gap_hz})


def result_from_json(path: str | Path) -> AnalysisResult:
    """Reconstruct reportable result data from RF-Lens JSON without IQ samples."""
    import json
    from rflens.report.tables import MEASUREMENT_FIELDS

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rec_data, spec_data = data["recording"], data["spectrum"]
    sample_count = int(rec_data.get("n_samples", 0))
    recording = IQRecording(np.empty(0, dtype=np.complex64), float(rec_data["fs_hz"]),
        float(rec_data["fc_hz"]), str(rec_data["source"]),
        {**rec_data.get("meta", {}), "n_samples": sample_count})
    spectrum = Spectrum(np.asarray(spec_data["freq_hz"], dtype=float),
        np.asarray(spec_data["psd_db"], dtype=float), float(spec_data["rbw_hz"]),
        int(spec_data["n_averages"]), spec_data.get("params", {}))
    signals = [SignalMeasurement(**{key: (float("nan") if value is None and key not in ("flags",) else value)
        for key, value in {field: row.get(field) for field in MEASUREMENT_FIELDS}.items()})
        for row in data.get("signals", [])]
    floor = float(data["noise_floor_dbfs_per_hz"])
    return AnalysisResult(recording, spectrum, floor, signals,
                          data.get("config", {}), data.get("warnings", []))
