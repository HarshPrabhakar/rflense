"""Loader for NumPy complex IQ arrays."""

from pathlib import Path

import numpy as np

from rflens.io.base import make_recording
from rflens.types import IQRecording


def load_npy(path: str | Path, *, fs: float, fc: float = 0.0) -> IQRecording:
    """Load a one-dimensional complex .npy array; ``fs`` and ``fc`` are in Hz."""
    samples = np.load(path, mmap_mode="r", allow_pickle=False)
    if not np.iscomplexobj(samples):
        raise ValueError(".npy IQ data must have a complex dtype")
    return make_recording(samples, fs, fc, str(path), datatype=str(samples.dtype))
