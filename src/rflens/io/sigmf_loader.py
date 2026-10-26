"""Minimal SigMF metadata parser and data reader."""
import json
from pathlib import Path
from typing import Any
import numpy as np
from rflens.io.base import make_recording
from rflens.types import IQRecording

_DTYPES: dict[str, tuple[str, bool]] = {
    "cf32_le": ("<f4", True), "cf32_be": (">f4", True),
    "ci16_le": ("<i2", True), "ci16_be": (">i2", True), "cu8": ("u1", False), "ci8": ("i1", False),
}

def load_sigmf(path: str | Path, *, fc: float = 0.0) -> IQRecording:
    """Read a SigMF pair; sample and center frequencies are in Hz."""
    p = Path(path)
    meta_path = p if p.name.endswith(".sigmf-meta") else p.with_suffix(".sigmf-meta")
    data_path = p.with_suffix(".sigmf-data") if p.name.endswith(".sigmf-meta") else p
    metadata: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    global_meta = metadata.get("global", {})
    fs, datatype = global_meta.get("core:sample_rate"), global_meta.get("core:datatype")
    if fs is None: raise ValueError("SigMF metadata is missing global core:sample_rate")
    if datatype not in _DTYPES: raise ValueError(f"Unsupported or missing SigMF core:datatype: {datatype!r}")
    dtype, interleaved = _DTYPES[datatype]
    raw = np.memmap(data_path, dtype=dtype, mode="r")
    if raw.size % 2: raise ValueError("SigMF data has an incomplete I/Q pair")
    if interleaved:
        scale = 1.0 if datatype.startswith("cf32") else (32768.0 if datatype.startswith("ci16") else 128.0)
        samples = raw[0::2].astype(np.float32) / scale + 1j * raw[1::2].astype(np.float32) / scale
    elif datatype == "cu8":
        vals = (raw.astype(np.float32) - 127.5) / 127.5
        samples = vals[0::2] + 1j * vals[1::2]
    else:
        vals = raw.astype(np.float32) / 128.0
        samples = vals[0::2] + 1j * vals[1::2]
    captures = metadata.get("captures", [])
    capture_fc = next((c.get("core:frequency") for c in captures if c.get("core:frequency") is not None), fc)
    return make_recording(samples, float(fs), float(capture_fc), str(data_path),
                          datatype=datatype, sigmf_global=global_meta)
