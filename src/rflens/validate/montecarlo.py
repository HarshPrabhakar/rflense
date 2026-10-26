"""Monte Carlo validation of PSK occupied bandwidth and center estimates."""

from dataclasses import dataclass

from rflens.dsp.psd import welch_psd
from rflens.measure.analyze import analyze_spectrum
from rflens.synth.channel import add_awgn
from rflens.synth.modulators import generate_psk
from rflens.types import IQRecording
from rflens.validate.metrics import ErrorMetrics, error_metrics
from rflens.validate.truth import numerical_obw99


@dataclass(frozen=True)
class ValidationTrial:
    """One seeded validation trial."""

    snr_db: float
    seed: int
    detected: bool
    estimated_obw99_hz: float | None
    estimated_center_hz: float | None


def run_psk_validation(
    *,
    modulation: str = "QPSK",
    snr_values_db: tuple[float, ...] = (-5, 0, 5, 10, 15, 20, 25, 30),
    trials_per_snr: int = 100,
    symbol_rate_hz: float = 100_000,
    sample_rate_hz: float = 800_000,
    alpha: float = 0.35,
    n_symbols: int = 512,
    nperseg: int = 512,
    center_frequency_hz: float = 100_000_000.0,
    threshold_db: float = 6.0,
    seed: int = 1000,
) -> tuple[list[ValidationTrial], dict[float, dict[str, float | int]]]:
    """Run deterministic Monte Carlo trials across target in-band SNRs.

    Returns individual trials and summary bias/RMSE/success-rate values. Metrics
    only use detected trials; detection probability uses all trials.
    """
    if trials_per_snr < 1:
        raise ValueError("trials_per_snr must be at least one")
    clean = generate_psk(
        modulation,
        symbol_rate_hz=symbol_rate_hz,
        sample_rate_hz=sample_rate_hz,
        n_symbols=n_symbols,
        alpha=alpha,
        seed=seed,
    )
    truth = numerical_obw99(clean, fs=sample_rate_hz, fc=center_frequency_hz, nperseg=nperseg)
    trials: list[ValidationTrial] = []
    summary: dict[float, dict[str, float | int]] = {}
    for snr in snr_values_db:
        estimates: list[float] = []
        center_estimates: list[float] = []
        false_detections = 0
        for trial_index in range(trials_per_snr):
            trial_seed = seed + trial_index + round((snr + 100) * 10000)
            noisy = add_awgn(
                clean,
                snr_db=snr,
                seed=trial_seed,
                in_bandwidth_hz=truth,
                sample_rate_hz=sample_rate_hz,
            )
            spectrum = welch_psd(
                IQRecording(noisy, sample_rate_hz, center_frequency_hz, f"synthetic:{modulation}"),
                nperseg=nperseg,
            )
            _, measurements = analyze_spectrum(spectrum, threshold_db=threshold_db)
            detected = bool(measurements)
            best = max(measurements, key=lambda s: s.power_dbfs) if detected else None
            false_detections += max(0, len(measurements) - int(detected))
            trials.append(
                ValidationTrial(
                    float(snr),
                    trial_seed,
                    detected,
                    best.obw99_hz if best else None,
                    best.centre_freq_hz if best else None,
                )
            )
            if best:
                estimates.append(best.obw99_hz)
                center_estimates.append(best.centre_freq_hz)
        rate = len(estimates) / trials_per_snr
        row: dict[str, float | int] = {
            "detection_probability": rate,
            "detected_trials": len(estimates),
            "total_trials": trials_per_snr,
            "false_detections_per_trial": false_detections / trials_per_snr,
            "truth_obw99_hz": truth,
        }
        if estimates:
            metrics: ErrorMetrics = error_metrics(estimates, truth)
            row.update(
                {
                    "bias_hz": metrics.bias,
                    "rmse_hz": metrics.rmse,
                    "std_hz": metrics.standard_deviation,
                    "relative_rmse": metrics.relative_rmse,
                }
            )
            center_metrics: ErrorMetrics = error_metrics(center_estimates, center_frequency_hz)
            row.update(
                {
                    "center_bias_hz": center_metrics.bias,
                    "center_rmse_hz": center_metrics.rmse,
                    "center_std_hz": center_metrics.standard_deviation,
                }
            )
        summary[float(snr)] = row
    return trials, summary
