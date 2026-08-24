"""Explicit truth-side spectral discrepancies for controlled credibility tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probdt.diagnostics.identifiability import acceleration_auto_psd
from probdt.structural import apply_stiffness_multipliers

FloatArray = NDArray[np.float64]


class ModelDiscrepancyError(ValueError):
    """Raised when a declared truth-model discrepancy is invalid."""


@dataclass(frozen=True)
class SpectralModelDiscrepancy:
    """Multiplicative truth perturbations omitted from the inference model."""

    mass_scale: float = 1.0
    damping_scale: float = 1.0
    excitation_standard_deviation_scale: float = 1.0
    excitation_correlation_time_scale: float = 1.0
    sensor_gain_scales: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        scalars = np.asarray(
            [
                self.mass_scale,
                self.damping_scale,
                self.excitation_standard_deviation_scale,
                self.excitation_correlation_time_scale,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0.0):
            raise ModelDiscrepancyError(
                "All physical discrepancy scales must be positive and finite."
            )
        if self.sensor_gain_scales is not None:
            gains = np.asarray(self.sensor_gain_scales, dtype=np.float64)
            if gains.ndim != 1 or gains.size < 1:
                raise ModelDiscrepancyError(
                    "sensor_gain_scales must be a nonempty vector."
                )
            if not np.all(np.isfinite(gains)) or np.any(gains <= 0.0):
                raise ModelDiscrepancyError(
                    "Sensor gain scales must be positive and finite."
                )


def discrepant_acceleration_auto_psd(
    mass_matrix_kg: ArrayLike,
    damping_matrix_n_s_per_m: ArrayLike,
    zone_stiffness_components_n_per_m: ArrayLike,
    stiffness_multipliers: ArrayLike,
    frequencies_hz: ArrayLike,
    force_distribution: ArrayLike,
    sensor_dof_indices: tuple[int, ...],
    *,
    force_standard_deviation_n: float,
    force_correlation_time_s: float,
    discrepancy: SpectralModelDiscrepancy,
    measurement_noise_psd: ArrayLike | None = None,
) -> FloatArray:
    """Evaluate one discrepant truth spectrum with additive sensor noise.

    Sensor gain multiplies physical acceleration before additive electronic
    noise, so its contribution to the signal PSD is the squared gain.
    """

    mass = np.asarray(mass_matrix_kg, dtype=np.float64)
    damping = np.asarray(damping_matrix_n_s_per_m, dtype=np.float64)
    components = np.asarray(
        zone_stiffness_components_n_per_m,
        dtype=np.float64,
    )
    stiffness = apply_stiffness_multipliers(
        components,
        stiffness_multipliers,
    )
    clean_psd = acceleration_auto_psd(
        discrepancy.mass_scale * mass,
        discrepancy.damping_scale * damping,
        stiffness,
        frequencies_hz,
        force_distribution,
        sensor_dof_indices,
        force_standard_deviation_n=(
            discrepancy.excitation_standard_deviation_scale
            * float(force_standard_deviation_n)
        ),
        force_correlation_time_s=(
            discrepancy.excitation_correlation_time_scale
            * float(force_correlation_time_s)
        ),
    )
    n_frequencies, n_sensors = clean_psd.shape
    if discrepancy.sensor_gain_scales is None:
        gains = np.ones(n_sensors, dtype=np.float64)
    else:
        gains = np.asarray(discrepancy.sensor_gain_scales, dtype=np.float64)
        if gains.shape != (n_sensors,):
            raise ModelDiscrepancyError(
                "sensor_gain_scales must contain one value per selected sensor."
            )

    if measurement_noise_psd is None:
        noise = np.zeros_like(clean_psd)
    else:
        noise_input = np.asarray(measurement_noise_psd, dtype=np.float64)
        if noise_input.shape == (n_sensors,):
            noise = np.broadcast_to(noise_input, clean_psd.shape)
        elif noise_input.shape == (n_frequencies, n_sensors):
            noise = noise_input
        else:
            raise ModelDiscrepancyError(
                "measurement_noise_psd has invalid dimensions."
            )
        if not np.all(np.isfinite(noise)) or np.any(noise < 0.0):
            raise ModelDiscrepancyError(
                "measurement_noise_psd must be nonnegative and finite."
            )

    result = gains[None, :] ** 2 * clean_psd + noise
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ModelDiscrepancyError(
            "The discrepant acceleration PSD must be positive and finite."
        )
    return result


def coherent_log_spectral_residual_scores(
    observed_log_spectra: ArrayLike,
    fitted_log_spectra: ArrayLike,
    log_noise_variance: float,
) -> FloatArray:
    """Accumulate persistent signed spectral residuals across windows.

    For each monitoring window, the statistic is the mean squared running
    residual divided by the variance of that running mean. Under an unbiased,
    independent log-spectral observation model its reference mean is one.
    Persistent model-form error grows with the number of accumulated windows,
    whereas incoherent periodogram scatter averages away.
    """

    observed = np.asarray(observed_log_spectra, dtype=np.float64)
    fitted = np.asarray(fitted_log_spectra, dtype=np.float64)
    if observed.shape != fitted.shape or observed.ndim < 2:
        raise ModelDiscrepancyError(
            "Observed and fitted log spectra must have the same windowed shape."
        )
    if observed.shape[0] < 1:
        raise ModelDiscrepancyError("At least one monitoring window is required.")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(fitted)):
        raise ModelDiscrepancyError("Log spectra must be finite.")
    variance = float(log_noise_variance)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ModelDiscrepancyError(
            "log_noise_variance must be positive and finite."
        )

    residuals = (observed - fitted).reshape(observed.shape[0], -1)
    counts = np.arange(1, residuals.shape[0] + 1, dtype=np.float64)
    running_means = np.cumsum(residuals, axis=0) / counts[:, None]
    return np.mean(
        running_means**2 / (variance / counts[:, None]),
        axis=1,
    )
