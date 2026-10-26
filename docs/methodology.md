# Phase 1 methodology

## IQ normalization

Loaders produce a one-dimensional `complex64` vector. Floating-point complex samples are preserved. RTL-SDR unsigned bytes map with `(x - 127.5) / 127.5`; signed HackRF bytes map with `x / 128`. PCM WAV uses the corresponding signed integer full-scale divisor and stores left/right channels as I/Q.

## Welch power spectral density

The estimator uses SciPy Welch with a periodic Hann window, 50% overlap by default, no segment detrending, two-sided output, and density scaling. The complex baseband frequency bins and density bins are both fft-shifted. The RF center frequency is added to the shifted baseband axis. PSD values are converted with `10 log10(P)` to dBFS/Hz.

Welch windows are processed in batches (`chunk_segments=128` by default) so temporary FFT storage does not scale with the total capture length. The source sample vector itself remains accessible through the `IQRecording`; cfile, SigMF complex-float32, and NPY loaders use memory maps where possible.

The reported equivalent noise bandwidth is `fs * sum(|w|^2) / |sum(w)|^2`, in Hz. `n_averages` records the number of complete Welch segments; overlapping segments are correlated, so it is not an effective-independent-average estimate.

## Phase 2 measurements

The noise floor is the median of linear PSD bins after excluding the outer 5% on each side. The median is corrected to a mean estimate with the Gamma-distribution median for the spectrum's Welch segment count. Bins more than 6 dB above that estimate are dilated and excluded for up to two refinements. This is an approximation because overlapping Welch segments are correlated.

Detection groups contiguous bins at or above noise floor + threshold and optionally merges gaps in Hz. OBW99 first smooths linear PSD over 9 bins, then integrates trapezoidally over the detected region plus a four-bin margin after subtracting the estimated noise density; 0.5% of integrated signal power is excluded from each tail. The -3 dB and -26 dB widths use lightly smoothed, noise-subtracted PSD with linearly interpolated crossings in dB. If a crossing is absent, the result is NaN and a flag records that the width could not be measured.

Signal power integrates positive noise-subtracted PSD bins. SNR is the ratio of this power to estimated noise density times the expanded measurement bandwidth, in dB. This is an in-band estimate; it is not an absolute calibrated power measurement.
