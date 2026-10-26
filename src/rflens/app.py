"""Optional Streamlit analysis UI. Install with ``pip install rflens[ui]``."""

import tempfile
import wave
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from rflens.api import analyze
from rflens.dsp.plots import plot_psd, plot_spectrogram
from rflens.io import load
from rflens.report.pdf_builder import build_report
from rflens.report.plots import measurement_zoom_figure
from rflens.report.tables import measurements_csv, result_to_json
from rflens.synth.modulators import generate_psk
from rflens.types import IQRecording
from rflens.validate.montecarlo import run_psk_validation


def main() -> None:
    st.set_page_config(page_title="RF-Lens", layout="wide")
    st.title("RF-Lens IQ Signal Characterization")
    source_mode = st.sidebar.radio("Input", ["Upload IQ", "Generate QPSK"])
    fs = st.sidebar.number_input("Sample rate (Hz)", min_value=1.0, value=1_000_000.0)
    fc = st.sidebar.number_input("Center frequency (Hz)", value=0.0)
    nperseg = st.sidebar.number_input("Welch segment length", min_value=8, value=4096, step=256)
    threshold = st.sidebar.slider("Detection threshold above noise (dB)", 1.0, 20.0, 6.0)
    recording = None
    if source_mode == "Upload IQ":
        files = st.file_uploader(
            "Select IQ capture (SigMF: select both metadata and data)",
            accept_multiple_files=True,
            type=["sigmf-meta", "sigmf-data", "cfile", "wav", "npy", "bin"],
        )
        format_name = st.sidebar.selectbox(
            "Format", ["auto", "sigmf", "cfile", "wav", "npy", "rtl", "hackrf"]
        )
        if files and st.button("Analyze capture"):
            with tempfile.TemporaryDirectory(prefix="rflens-") as directory:
                paths = []
                for item in files:
                    target = Path(directory) / item.name
                    target.write_bytes(item.getvalue())
                    paths.append(target)
                selected = next((p for p in paths if p.name.endswith(".sigmf-meta")), paths[0])
                try:
                    recording = load(selected, format=format_name, fs=fs, fc=fc)
                except (ValueError, OSError, wave.Error, TypeError) as exc:
                    st.error(str(exc))
                    return
                _show_analysis(recording, nperseg, threshold)
            return
    else:
        symbol_rate = st.sidebar.number_input("Symbol rate (Hz)", min_value=1.0, value=100_000.0)
        alpha = st.sidebar.slider("RRC roll-off", 0.0, 1.0, 0.35)
        symbols = st.sidebar.number_input("Symbols", min_value=32, value=2048, step=128)
        seed = st.sidebar.number_input("Seed", min_value=0, value=42)
        if st.button("Generate and analyze"):
            samples = generate_psk(
                "QPSK",
                symbol_rate_hz=symbol_rate,
                sample_rate_hz=fs,
                n_symbols=symbols,
                alpha=alpha,
                seed=seed,
            )
            recording = IQRecording(samples, fs, fc, "synthetic:QPSK", {"seed": seed})
            _show_analysis(recording, nperseg, threshold)


def _show_analysis(recording: IQRecording, nperseg: int, threshold: float) -> None:
    result = analyze(recording, nperseg=int(nperseg), threshold_db=threshold)
    spectrum_tab, waterfall_tab, table_tab, validation_tab, report_tab, export_tab = st.tabs(
        ["Spectrum", "Spectrogram", "Measurements", "Validation", "Report", "Export"]
    )
    with spectrum_tab:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_psd(result.spectrum, ax=ax)
        for signal in result.signals:
            ax.axvspan(signal.f_low_hz, signal.f_high_hz, alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
        chosen = st.selectbox("Signal zoom", [None, *[s.id for s in result.signals]])
        if chosen is not None:
            fig = measurement_zoom_figure(result, next(s for s in result.signals if s.id == chosen))
            st.pyplot(fig)
            plt.close(fig)
    with waterfall_tab:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_spectrogram(recording, nperseg=min(int(nperseg), 2048), ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    with table_tab:
        st.metric("Noise floor (dBFS/Hz)", f"{result.noise_floor_db:.2f}")
        st.dataframe(
            [{**signal.__dict__, "flags": ", ".join(signal.flags)} for signal in result.signals],
            use_container_width=True,
        )
    with validation_tab:
        st.caption("Run a small seeded QPSK calibration sweep for the current sample rate.")
        validation_snr = st.select_slider(
            "Target in-band SNR (dB)", options=[0, 5, 10, 15, 20, 25, 30], value=10
        )
        if st.button("Run 10 validation trials"):
            _, summary = run_psk_validation(
                sample_rate_hz=recording.fs,
                snr_values_db=(float(validation_snr),),
                trials_per_snr=10,
                seed=42,
            )
            st.json(summary)
    with report_tab:
        pdf = BytesIO()
        with tempfile.TemporaryDirectory(prefix="rflens-report-") as directory:
            path = build_report(result, Path(directory) / "report.pdf")
            pdf.write(path.read_bytes())
        st.download_button("Download PDF", pdf.getvalue(), "rflens-report.pdf", "application/pdf")
    with export_tab:
        st.download_button(
            "Download JSON", result_to_json(result), "rflens-results.json", "application/json"
        )
        st.download_button(
            "Download CSV", measurements_csv(result), "rflens-measurements.csv", "text/csv"
        )


if __name__ == "__main__":
    main()
