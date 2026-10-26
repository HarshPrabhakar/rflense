# RF-Lens

RF-Lens is a Python toolkit for loading recorded IQ data, estimating spectra, characterizing detected signals, and validating measurements against seeded synthetic signals.

## Development

```powershell
python -m pip install -e ".[test]"
python -m pytest
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
