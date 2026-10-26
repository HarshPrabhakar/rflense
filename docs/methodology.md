# Phase 1 methodology

## IQ normalization

Loaders produce a one-dimensional `complex64` vector. Floating-point complex samples are preserved. RTL-SDR unsigned bytes map with `(x - 127.5) / 127.5`; signed HackRF bytes map with `x / 128`. PCM WAV uses the corresponding signed integer full-scale divisor and stores left/right channels as I/Q.

## Welch power spectral density

The estimator uses SciPy Welch with a periodic Hann window, 50% overlap by default, no segment detrending, two-sided output, and density scaling. The complex baseband frequency bins and density bins are both fft-shifted. The RF center frequency is added to the shifted baseband axis. PSD values are converted with `10 log10(P)` to dBFS/Hz.

The reported equivalent noise bandwidth is `fs * sum(|w|^2) / |sum(w)|^2`, in Hz. `n_averages` records the number of complete Welch segments; overlapping segments are correlated, so it is not an effective-independent-average estimate.
