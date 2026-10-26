# RF-Lens

RF-Lens is a Python toolkit for loading recorded IQ data and characterizing its spectrum. The first development phase establishes the recording model, file loaders, Welch power spectral density, and plotting helpers.

## Development

```powershell
python -m pip install -e ".[test]"
python -m pytest
```

## Phase 1 API

Loaders return an `IQRecording` with complex baseband samples, sample rate in Hz, and center frequency in Hz. `welch_psd` returns a two-sided PSD in dB re full-scale-squared per Hz on an absolute, fft-shifted frequency axis.
