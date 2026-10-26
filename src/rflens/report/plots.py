"""Analysis-result plots."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from rflens.dsp.plots import plot_psd, plot_spectrogram
from rflens.types import AnalysisResult, SignalMeasurement


def result_spectrum_figure(result: AnalysisResult) -> Figure:
    """Create a full-span PSD figure with OBW edges annotated."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_psd(result.spectrum, ax=ax)
    for signal in result.signals:
        ax.axvspan(signal.f_low_hz, signal.f_high_hz, alpha=0.18)
        ax.axvline(signal.centre_freq_hz, linestyle="--", linewidth=0.8)
        ax.annotate(
            f"{signal.id}",
            (signal.centre_freq_hz, ax.get_ylim()[1]),
            xytext=(2, -4),
            textcoords="offset points",
            va="top",
        )
    return fig


def result_spectrogram_figure(result: AnalysisResult) -> Figure:
    """Create an IQ spectrogram figure for a non-empty recording."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_spectrogram(result.recording, ax=ax)
    return fig


def measurement_zoom_figure(result: AnalysisResult, measurement: SignalMeasurement) -> Figure:
    """Create a signal zoom with measured center and OBW edges."""
    f, p = result.spectrum.freq_hz, result.spectrum.psd_db
    half_span = max(
        measurement.obw99_hz * 1.5 if np.isfinite(measurement.obw99_hz) else 0,
        measurement.bw_26db_hz * 1.2 if np.isfinite(measurement.bw_26db_hz) else 0,
        result.spectrum.rbw_hz * 8,
    )
    mask = np.abs(f - measurement.centre_freq_hz) <= half_span
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(f[mask], p[mask])
    ax.axvline(measurement.f_low_hz, color="tab:orange", linestyle="--", label="OBW99 edges")
    ax.axvline(measurement.f_high_hz, color="tab:orange", linestyle="--")
    ax.axvline(measurement.centre_freq_hz, color="tab:red", linestyle=":", label="Power centroid")
    ax.set(
        xlabel="Frequency (Hz)",
        ylabel="PSD (dBFS/Hz)",
        title=f"Signal {measurement.id}: measured bandwidth markers",
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig
