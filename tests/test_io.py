import json
import wave

import numpy as np
import pytest

from rflens.io import load


def test_cfile_round_trip(tmp_path):
    expected = np.array([0.25 - 0.5j, -0.75 + 0.125j], dtype=np.complex64)
    path = tmp_path / "capture.cfile"
    interleaved = np.empty(expected.size * 2, dtype="<f4")
    interleaved[0::2], interleaved[1::2] = expected.real, expected.imag
    interleaved.tofile(path)
    actual = load(path, fs=2e6, fc=433e6)
    np.testing.assert_allclose(actual.samples, expected)
    assert actual.fs == 2e6 and actual.fc == 433e6


def test_npy_round_trip_and_reject_real_array(tmp_path):
    expected = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    path = tmp_path / "capture.npy"
    np.save(path, expected)
    np.testing.assert_array_equal(load(path, fs=10).samples, expected)
    real_path = tmp_path / "real.npy"
    np.save(real_path, np.arange(4))
    with pytest.raises(ValueError, match="complex"):
        load(real_path, fs=10)


def test_raw_normalization(tmp_path):
    rtl = tmp_path / "rtl.bin"
    np.array([255, 0, 127, 128], dtype=np.uint8).tofile(rtl)
    rec = load(rtl, format="rtl", fs=1)
    np.testing.assert_allclose(
        rec.samples,
        [(255 - 127.5) / 127.5 - 1j, (127 - 127.5) / 127.5 + (128 - 127.5) / 127.5 * 1j],
    )
    hackrf = tmp_path / "hackrf.bin"
    np.array([-128, 127, 64, -64], dtype=np.int8).tofile(hackrf)
    rec = load(hackrf, format="hackrf", fs=1)
    np.testing.assert_allclose(rec.samples, [-1 + 127 / 128 * 1j, 0.5 - 0.5j])


def test_wav_iq_and_rate_from_header(tmp_path):
    path = tmp_path / "iq.wav"
    vals = np.array([16384, -16384, -32768, 32767], dtype="<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(48000)
        out.writeframes(vals.tobytes())
    rec = load(path, fc=144e6)
    np.testing.assert_allclose(rec.samples, [0.5 - 0.5j, -1 + 32767 / 32768 * 1j])
    assert rec.fs == 48000 and rec.fc == 144e6


def test_sigmf_metadata_and_interleaved_data(tmp_path):
    metadata = {
        "global": {"core:sample_rate": 1e6, "core:datatype": "cf32_le"},
        "captures": [{"core:frequency": 915e6}],
    }
    (tmp_path / "sample.sigmf-meta").write_text(json.dumps(metadata), encoding="utf-8")
    np.array([1, 0, 0, -1], dtype="<f4").tofile(tmp_path / "sample.sigmf-data")
    rec = load(tmp_path / "sample.sigmf-meta")
    np.testing.assert_array_equal(rec.samples, [1 + 0j, 0 - 1j])
    assert rec.fs == 1e6 and rec.fc == 915e6


def test_requires_fs_for_formats_without_metadata(tmp_path):
    path = tmp_path / "x.cfile"
    np.array([0, 0], dtype="<f4").tofile(path)
    with pytest.raises(ValueError, match="fs"):
        load(path)
