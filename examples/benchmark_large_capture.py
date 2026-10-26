"""Reproducible in-memory throughput check for chunked Welch PSD."""

import argparse
import time

import numpy as np

from rflens.dsp.psd import welch_psd
from rflens.types import IQRecording


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10_000_000)
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    samples = ((rng.standard_normal(args.samples) + 1j * rng.standard_normal(args.samples)) /
               np.sqrt(2)).astype(np.complex64)
    start = time.perf_counter()
    spectrum = welch_psd(IQRecording(samples, 2_000_000.0, 0.0, "synthetic:benchmark"),
                         nperseg=args.nperseg)
    elapsed = time.perf_counter() - start
    print(f"samples={args.samples} nperseg={args.nperseg} "
          f"segments={spectrum.n_averages} elapsed_s={elapsed:.3f} "
          f"samples_per_s={args.samples / elapsed:.0f}")


if __name__ == "__main__":
    main()
