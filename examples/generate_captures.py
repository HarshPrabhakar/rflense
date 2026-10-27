"""Generate independent synthetic SigMF captures for trying RF-Lens.

Each capture is written to its own directory with a matching
``.sigmf-meta`` and ``.sigmf-data`` file. The data is synthetic; no SDR is
needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rflens.synth.channel import add_awgn
from rflens.synth.modulators import generate_fsk, generate_ofdm, generate_psk


SAMPLE_RATE_HZ = 1_000_000.0
CENTER_FREQUENCY_HZ = 433_920_000.0
SYMBOL_RATE_HZ = 100_000.0
N_SYMBOLS = 2_000


def make_captures(output: Path, count: int) -> list[Path]:
    """Write up to ``count`` example captures and return their metadata paths."""
    profiles = [
        ("01_qpsk_clean", "QPSK", 20.0, 20261028),
        ("02_bpsk_clean", "BPSK", 20.0, 20261029),
        ("03_fsk_clean", "FSK", 20.0, 20261030),
        ("04_ofdm_clean", "OFDM", 20.0, 20261031),
        ("05_qpsk_noisy", "QPSK", 8.0, 20261032),
    ]
    if not 1 <= count <= len(profiles):
        raise ValueError(f"count must be between 1 and {len(profiles)}")

    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for folder_name, modulation, snr_db, seed in profiles[:count]:
        folder = output / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        if modulation in ("BPSK", "QPSK"):
            samples = generate_psk(
                modulation,
                symbol_rate_hz=SYMBOL_RATE_HZ,
                sample_rate_hz=SAMPLE_RATE_HZ,
                n_symbols=N_SYMBOLS,
                alpha=0.35,
                seed=seed,
            )
            occupied_bandwidth_hz = SYMBOL_RATE_HZ * 1.35
        elif modulation == "FSK":
            deviation_hz = SYMBOL_RATE_HZ / 2
            samples = generate_fsk(
                symbol_rate_hz=SYMBOL_RATE_HZ,
                sample_rate_hz=SAMPLE_RATE_HZ,
                n_symbols=N_SYMBOLS,
                deviation_hz=deviation_hz,
                seed=seed,
            )
            occupied_bandwidth_hz = 2 * (deviation_hz + SYMBOL_RATE_HZ / 2)
        else:
            n_subcarriers = 64
            spacing_hz = SYMBOL_RATE_HZ / n_subcarriers
            samples = generate_ofdm(
                n_subcarriers=n_subcarriers,
                sample_rate_hz=SAMPLE_RATE_HZ,
                subcarrier_spacing_hz=spacing_hz,
                n_symbols=N_SYMBOLS,
                cyclic_prefix_samples=16,
                modulation="QPSK",
                seed=seed,
            )
            occupied_bandwidth_hz = n_subcarriers * spacing_hz

        samples = add_awgn(
            samples,
            snr_db=snr_db,
            seed=seed + 100,
            in_bandwidth_hz=occupied_bandwidth_hz,
            sample_rate_hz=SAMPLE_RATE_HZ,
        )

        stem = folder / folder_name
        data_path = stem.with_suffix(".sigmf-data")
        meta_path = stem.with_suffix(".sigmf-meta")
        interleaved = np.empty(samples.size * 2, dtype="<f4")
        interleaved[0::2] = samples.real
        interleaved[1::2] = samples.imag
        interleaved.tofile(data_path)

        metadata = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": SAMPLE_RATE_HZ,
                "core:version": "1.0.0",
                "core:description": (
                    f"Synthetic {modulation} example; target in-band SNR={snr_db:g} dB; "
                    f"seed={seed}"
                ),
            },
            "captures": [
                {"core:sample_start": 0, "core:frequency": CENTER_FREQUENCY_HZ}
            ],
            "annotations": [],
        }
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        created.append(meta_path)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create separate synthetic SigMF folders for RF-Lens uploads."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/generated_captures"),
        help="Output folder (default: examples/generated_captures)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of capture folders to create, from 1 to 5 (default: 5)",
    )
    args = parser.parse_args()
    for meta_path in make_captures(args.out, args.count):
        print(f"Created {meta_path.parent}")
        print(f"  Upload {meta_path.name} together with its matching .sigmf-data file")


if __name__ == "__main__":
    main()
