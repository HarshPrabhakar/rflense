import json

import matplotlib

matplotlib.use("Agg")

import numpy as np

from rflens.api import analyze, result_from_json
from rflens.report.pdf_builder import build_report
from rflens.report.tables import measurements_csv, result_to_json
from rflens.types import AnalysisResult, IQRecording, SignalMeasurement, Spectrum


def test_analysis_json_csv_round_trip_and_pdf(tmp_path):
    rng = np.random.default_rng(40)
    samples = 0.2 * np.exp(2j * np.pi * 0.1 * np.arange(8192)) + 0.01 * (
        rng.standard_normal(8192) + 1j * rng.standard_normal(8192)
    )
    result = analyze(
        IQRecording(samples.astype(np.complex64), 1e6, 433e6, "synthetic:test"), nperseg=512
    )
    payload = result_to_json(result)
    decoded = json.loads(payload)
    assert decoded["recording"]["n_samples"] == 8192
    assert decoded["spectrum"]["freq_hz"]
    assert measurements_csv(result).startswith("id,centre_freq_hz")
    restored_path = tmp_path / "results.json"
    restored_path.write_text(payload, encoding="utf-8")
    restored = result_from_json(restored_path)
    assert restored.recording.meta["n_samples"] == 8192
    assert np.array_equal(restored.spectrum.freq_hz, result.spectrum.freq_hz)
    pdf = build_report(result, tmp_path / "reports" / "analysis.pdf")
    assert pdf.is_file() and pdf.stat().st_size > 1000


def test_json_converts_nonfinite_measurements_to_null():
    rec = IQRecording(np.zeros(8, dtype=np.complex64), 8.0)
    spectrum = Spectrum(np.arange(8.0), np.zeros(8), 1.5, 1)
    signal = SignalMeasurement(
        1, 1.0, 1.0, 1.0, float("nan"), float("nan"), float("nan"), 0.0, 2.0, -20.0, 10.0, 10.0
    )
    result = AnalysisResult(rec, spectrum, -100.0, [signal], {}, [])
    assert json.loads(result_to_json(result))["signals"][0]["obw99_hz"] is None
