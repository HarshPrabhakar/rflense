"""Pulse shaping filters."""

import numpy as np


def rrc_taps(alpha: float, sps: int, span_symbols: int = 10) -> np.ndarray:
    """Return unit-energy root-raised-cosine taps.

    ``alpha`` is dimensionless roll-off in [0, 1], ``sps`` samples/symbol,
    and ``span_symbols`` is the total filter span in symbols.
    """
    if not 0 <= alpha <= 1 or sps < 2 or span_symbols < 2:
        raise ValueError("alpha must be [0,1], sps >= 2, and span_symbols >= 2")
    count = span_symbols * sps + 1
    t = (np.arange(count) - (count - 1) / 2) / sps
    h = np.empty(count, dtype=float)
    for k, value in enumerate(t):
        if np.isclose(value, 0.0):
            h[k] = 1.0 + alpha * (4 / np.pi - 1)
        elif alpha > 0 and np.isclose(abs(value), 1 / (4 * alpha)):
            h[k] = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                                             (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        elif alpha == 0:
            h[k] = np.sinc(value)
        else:
            numerator = np.sin(np.pi * value * (1 - alpha)) + 4 * alpha * value * np.cos(np.pi * value * (1 + alpha))
            denominator = np.pi * value * (1 - (4 * alpha * value) ** 2)
            h[k] = numerator / denominator
    return h / np.sqrt(np.sum(h * h))
