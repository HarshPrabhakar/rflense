"""Loader for stereo WAV files with I on left and Q on right."""

import wave
from pathlib import Path

import numpy as np

from rflens.io.base import make_recording
from rflens.types import IQRecording


def load_wav(path: str | Path, *, fc: float = 0.0) -> IQRecording:
    """Load stereo PCM WAV as IQ; WAV sample rate and supplied ``fc`` are in Hz."""
    with wave.open(str(path), "rb") as wav:
        channels, width, fs, count = (
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getframerate(),
            wav.getnframes(),
        )
        if channels != 2:
            raise ValueError("WAV-IQ requires exactly two channels (I left, Q right)")
        raw = wav.readframes(count)
    if width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 127.5) / 127.5
    elif width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = (ints ^ 0x800000) - 0x800000
        values = ints.astype(np.float32) / 8388608.0
    elif width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")
    if values.size != count * 2:
        raise ValueError("WAV data is truncated or malformed")
    return make_recording(
        values[0::2] + 1j * values[1::2], fs, fc, str(path), datatype=f"pcm_{width * 8}_stereo"
    )
