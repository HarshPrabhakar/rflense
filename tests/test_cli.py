from typer.testing import CliRunner

from rflens.cli import app


def test_synth_analyze_and_report_commands(tmp_path):
    runner = CliRunner()
    sample_stem = tmp_path / "demo.sigmf"
    generated = runner.invoke(app, ["synth", "--mod", "qpsk", "--rs", "100000", "--fs", "1000000",
                                    "--out", str(sample_stem), "--n-symbols", "128", "--seed", "9"])
    assert generated.exit_code == 0, generated.output
    meta = tmp_path / "demo.sigmf-meta"
    json_path, csv_path, pdf_path = tmp_path / "results.json", tmp_path / "signals.csv", tmp_path / "analysis.pdf"
    analyzed = runner.invoke(app, ["analyze", str(meta), "--nperseg", "256", "--json", str(json_path),
                                   "--csv", str(csv_path), "--report", str(pdf_path), "--no-plots"])
    assert analyzed.exit_code == 0, analyzed.output
    assert json_path.is_file() and csv_path.is_file() and pdf_path.is_file()
    rebuilt = runner.invoke(app, ["report", str(json_path), "--out", str(tmp_path / "rebuilt.pdf"), "--no-plots"])
    assert rebuilt.exit_code == 0, rebuilt.output
