"""Build multi-signal synthetic scenes and their signal truth metadata."""

from dataclasses import dataclass

import numpy as np

from rflens.synth.modulators import generate_fsk, generate_ofdm, generate_psk
from rflens.types import IQRecording


@dataclass(frozen=True)
class SceneSignal:
    """Description of one synthesized signal in a scene."""

    modulation: str
    center_offset_hz: float
    power_db: float
    n_symbols: int
    seed: int
    symbol_rate_hz: float = 0.0
    alpha: float = 0.35
    deviation_hz: float = 0.0
    n_subcarriers: int = 0
    subcarrier_spacing_hz: float = 0.0
    cyclic_prefix_samples: int = 0
    ofdm_modulation: str = "QPSK"


def generate_scene(
    signals: list[SceneSignal],
    *,
    sample_rate_hz: float,
    center_frequency_hz: float = 0.0,
    n_samples: int | None = None,
) -> IQRecording:
    """Synthesize and sum PSK/FSK signals at declared offsets and powers."""
    if not signals:
        raise ValueError("scene must contain at least one signal")
    generated: list[tuple[SceneSignal, np.ndarray]] = []
    for signal in signals:
        name = signal.modulation.upper()
        if name in ("BPSK", "QPSK"):
            wave = generate_psk(
                name,
                symbol_rate_hz=signal.symbol_rate_hz,
                sample_rate_hz=sample_rate_hz,
                n_symbols=signal.n_symbols,
                alpha=signal.alpha,
                seed=signal.seed,
            )
        elif name == "FSK":
            wave = generate_fsk(
                symbol_rate_hz=signal.symbol_rate_hz,
                sample_rate_hz=sample_rate_hz,
                n_symbols=signal.n_symbols,
                deviation_hz=signal.deviation_hz,
                seed=signal.seed,
            )
        elif name == "OFDM":
            wave = generate_ofdm(
                n_subcarriers=signal.n_subcarriers,
                sample_rate_hz=sample_rate_hz,
                subcarrier_spacing_hz=signal.subcarrier_spacing_hz,
                n_symbols=signal.n_symbols,
                cyclic_prefix_samples=signal.cyclic_prefix_samples,
                modulation=signal.ofdm_modulation,
                seed=signal.seed,
            )
        else:
            raise ValueError(f"Unsupported modulation: {signal.modulation}")
        generated.append((signal, wave))
    length = n_samples if n_samples is not None else max(w.size for _, w in generated)
    if length < 1:
        raise ValueError("n_samples must be positive")
    composite = np.zeros(length, dtype=np.complex128)
    for signal, wave in generated:
        n = min(length, wave.size)
        segment = wave[:n]
        mean_power = float(np.mean(np.abs(segment) ** 2))
        if mean_power == 0:
            continue
        scaled = segment / np.sqrt(mean_power) * 10 ** (signal.power_db / 20)
        composite[:n] += scaled * np.exp(
            2j * np.pi * signal.center_offset_hz * np.arange(n) / sample_rate_hz
        )
    return IQRecording(
        composite.astype(np.complex64),
        sample_rate_hz,
        center_frequency_hz,
        "synthetic:scene",
        {"signals": [s.__dict__ for s in signals]},
    )
