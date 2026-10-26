"""Technical PDF report builder."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rflens.report.plots import (measurement_zoom_figure, result_spectrogram_figure,
                                 result_spectrum_figure)
from rflens.report.tables import result_to_dict
from rflens.types import AnalysisResult


def build_report(result: AnalysisResult, output_path: str | Path, *, include_plots: bool = True) -> Path:
    """Build a multi-page technical PDF report and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result_data = result_to_dict(result)
    config_hash = hashlib.sha256(json.dumps(result_data["config"], sort_keys=True).encode()).hexdigest()[:12]
    with PdfPages(path, metadata={"Title": "RF-Lens IQ Analysis Report", "Creator": "RF-Lens"}) as pdf:
        _text_page(pdf, "RF-Lens IQ Signal Characterization Report", [
            f"Source: {result.recording.source}",
            f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"RF-Lens version: 0.1.0", f"Configuration hash: {config_hash}", "",
            "Objective", "Characterize the recorded IQ spectrum using the configured spectral analysis pipeline.",
        ])
        sample_count = int(result.recording.meta.get("n_samples", result.recording.samples.size))
        duration = sample_count / result.recording.fs
        _text_page(pdf, "Data and Methodology", [
            f"Sample rate: {result.recording.fs:g} Hz",
            f"Center frequency: {result.recording.fc:g} Hz",
            f"Samples: {sample_count:,}", f"Duration: {duration:.6g} s",
            f"Source metadata: {json.dumps(result.recording.meta, default=str, sort_keys=True)}", "",
            "Processing: two-sided Welch PSD with configured window, segment length, and overlap.",
            f"Window: {result.spectrum.params.get('window', 'unknown')}",
            f"Resolution bandwidth (ENBW): {result.spectrum.rbw_hz:g} Hz",
            f"Welch segments: {result.spectrum.n_averages}",
            f"Noise density: {result.noise_floor_db:.3f} dBFS/Hz",
            "OBW99 uses 0.5% tail exclusion on each side after noise subtraction.",
            "SNR is in-band signal power divided by estimated noise over the measured bandwidth.",
            "Absolute calibrated power in dBm is not available in v1.",
        ])
        if include_plots:
            _figure_page(pdf, result_spectrum_figure(result))
            if result.recording.samples.size:
                _figure_page(pdf, result_spectrogram_figure(result))
        _measurements_page(pdf, result)
        for signal in result.signals:
            if include_plots:
                _figure_page(pdf, measurement_zoom_figure(result, signal))
            _text_page(pdf, f"Signal {signal.id}: observations", [
                f"Measured center: {signal.centre_freq_hz:.3f} Hz",
                f"OBW99: {_format(signal.obw99_hz)} Hz",
                f"Power: {signal.power_dbfs:.3f} dBFS",
                f"In-band SNR: {signal.snr_db:.3f} dB",
                f"Flags: {', '.join(signal.flags) if signal.flags else 'none'}",
                "These observations are generated from the measurement values above.",
            ])
        _text_page(pdf, "Limitations and Conclusions", [
            "Limitations", "No absolute power calibration is applied. Frequency and bandwidth precision are limited by the PSD bin spacing and selected analysis settings.",
            "Noise-floor and measurement estimates depend on the signal and noise assumptions documented in the methodology.", "",
            "Conclusions", f"Detected signals: {len(result.signals)}",
            f"Warnings: {', '.join(result.warnings) if result.warnings else 'none'}",
            "The conclusions on this page are restricted to the measurements and warnings reported in this analysis.",
        ])
        versions = []
        for package in ("numpy", "scipy", "matplotlib", "rflens"):
            try:
                versions.append(f"{package}: {version(package)}")
            except PackageNotFoundError:
                versions.append(f"{package}: source checkout")
        _text_page(pdf, "Appendix: Configuration and Software", [
            f"Python: {platform.python_version()}", *versions, "",
            "Analysis configuration:", json.dumps(result.config, sort_keys=True, default=str), "",
            "Formulas: PSD power conversion uses 10 log10(P). OBW99 retains 99% of integrated noise-subtracted power.",
            "Signal power integrates positive noise-subtracted PSD across the selected measurement region.",
            "SNR is signal power divided by estimated noise density integrated over that region.",
        ])
    return path


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", va="top")
    y = 0.88
    for line in lines:
        wrap = textwrap.wrap(line, width=92) or [""]
        for piece in wrap:
            fig.text(0.08, y, piece, fontsize=10, va="top")
            y -= 0.027
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _figure_page(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _measurements_page(pdf: PdfPages, result: AnalysisResult) -> None:
    columns = ["ID", "Center (Hz)", "OBW99 (Hz)", "-3 dB (Hz)", "-26 dB (Hz)",
               "Power (dBFS)", "SNR (dB)", "Flags"]
    rows = [[str(s.id), f"{s.centre_freq_hz:.3f}", _format(s.obw99_hz), _format(s.bw_3db_hz),
             _format(s.bw_26db_hz), f"{s.power_dbfs:.2f}", f"{s.snr_db:.2f}", ", ".join(s.flags)]
            for s in result.signals]
    fig, ax = plt.subplots(figsize=(11.7, max(2.5, 1.2 + 0.45 * len(rows))))
    ax.axis("off")
    ax.set_title("Signal Measurements", loc="left", fontsize=16, pad=14)
    if rows:
        table = ax.table(cellText=rows, colLabels=columns, loc="upper left", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.5)
    else:
        ax.text(0.02, 0.85, "No signals were detected.", transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _format(value: float) -> str:
    return f"{value:.3f}" if value == value and abs(value) != float("inf") else "unavailable"
