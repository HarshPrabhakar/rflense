"""Common loader helpers and format dispatch."""

from pathlib import Path
from typing import Literal

import numpy as np

from rflens.types import IQRecording

Format = Literal["auto", "sigmf", "cfile", "wav", "rtl", "hackrf", "npy"]


def make_recording(samples: np.ndarray, fs: float, fc: float, source: str, **meta: object) -> IQRecording:
    """Validate samples and normalize them to complex64."""
    array = np.asarray(samples)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("IQ samples must be a non-empty one-dimensional array")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sample rate fs must be finite and positive (Hz)")
    if not np.isfinite(fc):
        raise ValueError("center frequency fc must be finite (Hz)")
    return IQRecording(array.astype(np.complex64, copy=False), float(fs), float(fc), source, dict(meta))


def load(path: str | Path, *, format: Format = "auto", fs: float | None = None,
         fc: float = 0.0) -> IQRecording:
    """Load IQ data. Frequencies are in Hz; raw formats require ``fs``."""
    from rflens.io.cfile_loader import load_cfile
    from rflens.io.npy_loader import load_npy
    from rflens.io.raw_loader import load_raw
    from rflens.io.sigmf_loader import load_sigmf
    from rflens.io.wav_loader import load_wav
    p = Path(path)
    selected = format
    if selected == "auto":
        if p.suffix.lower() == ".npy": selected = "npy"
        elif p.suffix.lower() == ".wav": selected = "wav"
        elif p.name.endswith(".sigmf-meta") or p.suffix.lower() == ".sigmf-data": selected = "sigmf"
        elif p.suffix.lower() == ".cfile": selected = "cfile"
        else: raise ValueError(f"Cannot infer IQ format from {p.name!r}; provide format explicitly")
    if selected == "sigmf": return load_sigmf(p, fc=fc)
    if selected == "cfile": return load_cfile(p, fs=_required_fs(fs), fc=fc)
    if selected == "wav": return load_wav(p, fc=fc)
    if selected in ("rtl", "hackrf"): return load_raw(p, format=selected, fs=_required_fs(fs), fc=fc)
    if selected == "npy": return load_npy(p, fs=_required_fs(fs), fc=fc)
    raise ValueError(f"Unsupported format: {selected}")


def _required_fs(fs: float | None) -> float:
    if fs is None: raise ValueError("fs in Hz is required for this format")
    return fs
