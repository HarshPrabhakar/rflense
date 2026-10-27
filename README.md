# RF-Lens

RF-Lens is a Python toolkit for loading recorded IQ data, estimating spectra, characterizing detected signals, and validating measurements against seeded synthetic signals.

## Development

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
rflens --help
```

macOS/Linux:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the web UI:

```powershell
streamlit run src/rflens/app.py
```

Analyze an IQ file and write a PDF, JSON, and CSV:

```powershell
rflens analyze capture.sigmf-meta --nperseg 4096 --report report.pdf --json results.json --csv measurements.csv
```

Run tests and code-quality checks:

```powershell
python -m pytest
ruff check src tests
mypy src/rflens
```

## Phase 1 API

Loaders return an `IQRecording` with complex baseband samples, sample rate in Hz, and center frequency in Hz. `welch_psd` returns a two-sided PSD in dB re full-scale-squared per Hz on an absolute, fft-shifted frequency axis.

## Validation snapshot

Phase 3's seeded default QPSK sweep ran 100 trials at each target in-band SNR from -5 to 30 dB in 5 dB steps (symbol rate 100 ksym/s, 800 ksample/s, roll-off 0.35, 512 symbols). The numerically computed noiseless OBW99 was 117.209 kHz. Results below are from the actual run; the noise floor and region threshold were estimated independently for every trial.

| In-band SNR | Detection probability | OBW99 relative RMSE | Centre RMSE | Extra detections / trial |
|---:|---:|---:|---:|---:|
| -5 dB | 0% | — | — | 0.00 |
| 0 dB | 0% | — | — | 0.00 |
| 5 dB | 100% | 77.42% | 20.73 kHz | 4.82 |
| 10 dB | 100% | 1.98% | 1.06 kHz | 0.02 |
| 15 dB | 100% | 2.48% | 0.77 kHz | 0.00 |
| 20 dB | 100% | 2.90% | 0.75 kHz | 0.00 |
| 25 dB | 100% | 3.05% | 0.72 kHz | 0.00 |
| 30 dB | 100% | 3.04% | 0.70 kHz | 0.00 |

For this scenario, the measured OBW99 relative RMSE is below the 5% target at 10 dB and above. At 5 dB, many extra regions are detected and bandwidth error is high; this limitation is retained in the results rather than filtered away. This sweep validates QPSK OBW/center behavior only; it does not establish equivalent performance for BPSK, FSK, OFDM, or impairments.

## Interfaces

```powershell
rflens analyze capture.sigmf-meta --nperseg 4096 --report report.pdf --json results.json --csv measurements.csv
rflens info capture.npy --fs 1000000
rflens synth --mod qpsk --rs 100000 --alpha 0.35 --fs 1000000 --out sample.sigmf
rflens report results.json --out report.pdf
```

The CLI is installed from the package entry point. Install the optional Streamlit interface with `pip install -e ".[ui]"`, then run `streamlit run src/rflens/app.py`. YAML validation configs require `pip install -e ".[validation]"`.

## Large-file throughput observation

On the current Python 3.14.3 environment, the checked-in benchmark processed 10,000,000 complex samples with 4096-sample Welch windows in 0.329 s (4,881 segments; 30.4 million samples/s). This is a single local measurement and may vary substantially across systems.

## Example artifacts

- [Synthetic QPSK SigMF capture](examples/data/synthetic_qpsk.sigmf-meta) and [complex sample data](examples/data/synthetic_qpsk.sigmf-data), generated with seed 20261027, sample rate 1 MHz, center frequency 433.92 MHz, and target in-band SNR 10 dB.
- [Synthetic QPSK PDF report](examples/reports/synthetic_qpsk_report.pdf), [JSON results](examples/reports/synthetic_qpsk_results.json), and [CSV measurements](examples/reports/synthetic_qpsk_measurements.csv). This run measured 111.397 kHz OBW99, a center of 433.919522 MHz, and 10.60 dB in-band SNR.

A real-recording report is not included because the workspace contains no real IQ capture. The example report generator is ready to run when a capture and its metadata are available.

## Generate uploadable practice captures

You can generate five synthetic SigMF captures without an SDR. Each capture
gets its own folder containing a matching `.sigmf-meta` and `.sigmf-data` pair:

```powershell
python examples/generate_captures.py
```

By default, the folders are created under `examples/generated_captures`.
Choose a different output path or generate fewer captures with:

```powershell
python examples/generate_captures.py --out my_captures --count 3
```

Upload the `.sigmf-meta` file from one capture folder; keep its matching
`.sigmf-data` file in the same folder. The set includes BPSK, QPSK, FSK, OFDM,
and a noisier QPSK example. These are software-generated practice signals,
not recordings from a physical radio.
