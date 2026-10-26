"""Load RTL-SDR uint8 and HackRF signed int8 interleaved IQ."""

from pathlib import Path
from typing import Literal

import numpy as np

from rflens.io.base import make_recording
from rflens.types import IQRecording


def load_raw(
    path: str | Path, *, format: Literal["rtl", "hackrf"], fs: float, fc: float = 0.0
) -> IQRecording:
    """Load byte-interleaved raw IQ; ``fs`` and ``fc`` are in Hz."""
    if format == "rtl":
        raw = np.fromfile(path, dtype=np.uint8).astype(np.float32)
        if raw.size % 2:
            raise ValueError("raw IQ file must contain an even number of bytes")
        i, q = (raw[0::2] - 127.5) / 127.5, (raw[1::2] - 127.5) / 127.5
        datatype = "rtl_u8"
    elif format == "hackrf":
        raw = np.fromfile(path, dtype=np.int8).astype(np.float32)
        if raw.size % 2:
            raise ValueError("raw IQ file must contain an even number of bytes")
        i, q = raw[0::2] / 128.0, raw[1::2] / 128.0
        datatype = "hackrf_i8"
    else:
        raise ValueError(f"Unsupported raw format: {format}")
    return make_recording(i + 1j * q, fs, fc, str(path), datatype=datatype)
