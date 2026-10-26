# Design decisions

- Phase 1 uses a small in-package SigMF metadata reader, so loading does not depend on an optional third-party package. The supported subset covers common complex float32, complex int16, complex int8, and complex uint8 sample encodings.
- `IQRecording.samples` are normalized to `complex64`; metadata retains the source datatype. Integer formats are normalized to approximately [-1, 1].
- The PSD is a two-sided density in full-scale-squared per Hz. Its axis includes the supplied or metadata center frequency and is fft-shifted.
- A `.npy` capture must contain a complex one-dimensional array. Its sample rate is supplied by the caller because NumPy files do not define one.
- WAV input must be stereo PCM with I on the left and Q on the right. Its center frequency defaults to 0 Hz because WAV headers do not contain RF tuning metadata.
- SigMF capture selection currently uses the first capture with a center frequency. Segment slicing by `core:sample_start` is deferred until the loader handles annotations and multiple captures explicitly.
- Phase 2 measures each detected region with a margin equal to its detected width in bins by default. This gives tails room beyond the initial detection threshold and is clipped at the capture edges; edge-touching regions receive a `near_band_edge` flag.
- Welch overlap makes the nominal segment count an imperfect effective-average count. Noise median correction therefore uses the nominal segment count and is documented as approximate.
