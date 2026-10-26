"""Loader for interleaved float32 I/Q cfiles."""

from pathlib import Path

import numpy as np

from rflens.io.base import make_recording
from rflens.types import IQRecording


def load_cfile(path: str | Path, *, fs: float, fc: float = 0.0) -> IQRecording:
    """Load interleaved float32 I/Q; ``fs`` and ``fc`` are in Hz."""
    raw = np.memmap(path, dtype="<f4", mode="r")
    if raw.size % 2:
        raise ValueError("cfile must contain an even number of float32 values")
    return make_recording(raw.view(np.complex64), fs, fc, str(path), datatype="cf32_le")
