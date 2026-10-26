"""Error metrics for Monte Carlo estimator validation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorMetrics:
    """Bias, population standard deviation, RMSE, and relative RMSE."""

    bias: float
    standard_deviation: float
    rmse: float
    relative_rmse: float


def error_metrics(estimates: np.ndarray | list[float], truth: float) -> ErrorMetrics:
    """Calculate errors in input units against one finite truth value."""
    values = np.asarray(estimates, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("estimates must be a non-empty finite 1-D array")
    if not np.isfinite(truth):
        raise ValueError("truth must be finite")
    errors = values - truth
    bias = float(np.mean(errors))
    std = float(np.std(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    relative = float(rmse / abs(truth)) if truth != 0 else float("nan")
    return ErrorMetrics(bias, std, rmse, relative)
