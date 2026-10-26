"""CSV and JSON serialization for analysis results."""

import csv
import io
import json
import math
from dataclasses import fields
from typing import Any

import numpy as np

from rflens.types import AnalysisResult, SignalMeasurement

MEASUREMENT_FIELDS = [field.name for field in fields(SignalMeasurement)]


def measurements_csv(result: AnalysisResult) -> str:
    """Serialize measurement rows to CSV with frequency units in field names."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MEASUREMENT_FIELDS)
    writer.writeheader()
    for signal in result.signals:
        writer.writerow({key: _json_safe(getattr(signal, key)) for key in MEASUREMENT_FIELDS})
    return output.getvalue()


def result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    """Serialize metadata, spectrum, analysis configuration, and measurements."""
    return _json_safe({
        "recording": {"fs_hz": result.recording.fs, "fc_hz": result.recording.fc,
                      "source": result.recording.source, "n_samples": int(result.recording.samples.size),
                      "meta": result.recording.meta},
        "spectrum": {"freq_hz": result.spectrum.freq_hz, "psd_db": result.spectrum.psd_db,
                     "rbw_hz": result.spectrum.rbw_hz, "n_averages": result.spectrum.n_averages,
                     "params": result.spectrum.params},
        "noise_floor_dbfs_per_hz": result.noise_floor_db,
        "signals": [{field: getattr(signal, field) for field in MEASUREMENT_FIELDS}
                    for signal in result.signals],
        "config": result.config,
        "warnings": result.warnings,
    })


def result_to_json(result: AnalysisResult, *, indent: int = 2) -> str:
    """Return standards-compliant JSON; undefined measurements are JSON null."""
    return json.dumps(result_to_dict(result), indent=indent, allow_nan=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
