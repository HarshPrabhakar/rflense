"""Command-line interface for RF-Lens."""

import csv
import json
from pathlib import Path

import numpy as np
import typer

from rflens.api import analyze, result_from_json
from rflens.io import load
from rflens.report.pdf_builder import build_report
from rflens.report.tables import measurements_csv, result_to_json
from rflens.synth.channel import add_awgn
from rflens.synth.modulators import generate_fsk, generate_ofdm, generate_psk
from rflens.validate.montecarlo import run_psk_validation

app = typer.Typer(no_args_is_help=True, help="RF-Lens IQ signal characterization toolkit")


@app.command("analyze")
def analyze_file(path: Path, format: str = "auto", fs: float | None = None,
                 fc: float = 0.0, nperseg: int = 4096, threshold_db: float = 6.0,
                 report: Path | None = None, json_out: Path | None = typer.Option(None, "--json"),
                 csv_out: Path | None = typer.Option(None, "--csv"),
                 no_plots: bool = False) -> None:
    """Analyze one recorded IQ file and optionally export PDF/JSON/CSV."""
    recording = load(path, format=format, fs=fs, fc=fc)  # type: ignore[arg-type]
    result = analyze(recording, nperseg=nperseg, threshold_db=threshold_db)
    typer.echo(f"Source: {recording.source}")
    typer.echo(f"Sample rate: {recording.fs:g} Hz; center frequency: {recording.fc:g} Hz")
    typer.echo(f"Noise floor: {result.noise_floor_db:.2f} dBFS/Hz")
    typer.echo(f"Signals detected: {len(result.signals)}")
    for signal in result.signals:
        typer.echo(f"  {signal.id}: center={signal.centre_freq_hz:.3f} Hz, "
                   f"OBW99={signal.obw99_hz:.3f} Hz, SNR={signal.snr_db:.2f} dB")
    if report:
        build_report(result, report, include_plots=not no_plots)
        typer.echo(f"PDF report: {report}")
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(result_to_json(result), encoding="utf-8")
        typer.echo(f"JSON results: {json_out}")
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        csv_out.write_text(measurements_csv(result), encoding="utf-8", newline="")
        typer.echo(f"CSV measurements: {csv_out}")


@app.command("info")
def info(path: Path, format: str = "auto", fs: float | None = None, fc: float = 0.0) -> None:
    """Show IQ file format and recording metadata."""
    recording = load(path, format=format, fs=fs, fc=fc)  # type: ignore[arg-type]
    typer.echo(json.dumps({"source": recording.source, "samples": recording.samples.size,
        "sample_rate_hz": recording.fs, "center_frequency_hz": recording.fc,
        "dtype": str(recording.samples.dtype), "duration_s": recording.samples.size / recording.fs,
        "metadata": recording.meta}, indent=2, default=str))


@app.command("synth")
def synth(mod: str = typer.Option("QPSK", "--mod"), rs: float = typer.Option(100e3, "--rs"),
          alpha: float = 0.35, snr: float | None = None, fs: float = typer.Option(1e6, "--fs"),
          out: Path = typer.Option(..., "--out"), n_symbols: int = 1000, seed: int = 0,
          fc: float = 0.0) -> None:
    """Generate seeded PSK or FSK IQ and save SigMF data and metadata."""
    name = mod.upper()
    if name in ("BPSK", "QPSK"):
        samples = generate_psk(name, symbol_rate_hz=rs, sample_rate_hz=fs,
                               n_symbols=n_symbols, alpha=alpha, seed=seed)
    elif name == "FSK":
        samples = generate_fsk(symbol_rate_hz=rs, sample_rate_hz=fs,
                               n_symbols=n_symbols, deviation_hz=rs / 2, seed=seed)
    else:
        raise typer.BadParameter("Phase 4 synth CLI supports BPSK, QPSK, and FSK")
    if snr is not None:
        samples = add_awgn(samples, snr_db=snr, seed=seed)
    stem = out.with_suffix("") if out.suffix in (".sigmf", ".sigmf-data", ".sigmf-meta") else out
    data_path = stem.with_suffix(".sigmf-data")
    meta_path = stem.with_suffix(".sigmf-meta")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = np.empty(samples.size * 2, dtype="<f4")
    interleaved[0::2], interleaved[1::2] = samples.real, samples.imag
    interleaved.tofile(data_path)
    metadata = {"global": {"core:datatype": "cf32_le", "core:sample_rate": fs,
                           "core:version": "1.0.0", "core:description": f"RF-Lens synthetic {name}; seed={seed}"},
                "captures": [{"core:sample_start": 0, "core:frequency": fc}],
                "annotations": []}
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    typer.echo(f"SigMF data: {data_path}")
    typer.echo(f"SigMF metadata: {meta_path}")


@app.command("validate")
def validate(config: Path, out: Path = typer.Option(..., "--out")) -> None:
    """Run PSK Monte Carlo validation using a YAML config file."""
    try:
        import yaml
    except ImportError as exc:
        raise typer.BadParameter("Install RF-Lens with the validation extra for YAML config support") from exc
    settings = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    trials, summary = run_psk_validation(**settings)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["snr_db", "seed", "detected",
            "estimated_obw99_hz", "estimated_center_hz"])
        writer.writeheader()
        writer.writerows(trial.__dict__ for trial in trials)
    typer.echo(f"Validation outputs saved to {out}")


@app.command("report")
def report(results: Path, out: Path = typer.Option(..., "--out"), no_plots: bool = False) -> None:
    """Build a PDF report from exported RF-Lens JSON results."""
    result = result_from_json(results)
    build_report(result, out, include_plots=not no_plots)
    typer.echo(f"PDF report: {out}")


if __name__ == "__main__":
    app()
