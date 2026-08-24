"""Frequency-domain sensitivity and local practical-identifiability tools."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class IdentifiabilityError(ValueError):
    """Raised when a sensitivity or spectral-identifiability input is invalid."""


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Local sensitivity strength, correlation, and singular-value diagnostics."""

    parameter_names: tuple[str, ...]
    sensitivity_matrix: FloatArray
    column_norms: FloatArray
    normalized_parameter_correlation: FloatArray
    normalized_singular_values: FloatArray
    effective_rank: int
    normalized_condition_number: float
    least_identifiable_combination: FloatArray

    @property
    def maximum_absolute_parameter_correlation(self) -> float:
        """Return the largest off-diagonal absolute column correlation."""

        correlation = np.abs(self.normalized_parameter_correlation.copy())
        np.fill_diagonal(correlation, 0.0)
        return float(np.max(correlation))


def acceleration_auto_psd(
    mass_matrix_kg: ArrayLike,
    damping_matrix_n_s_per_m: ArrayLike,
    stiffness_matrix_n_per_m: ArrayLike,
    frequencies_hz: ArrayLike,
    force_distribution: ArrayLike,
    sensor_dof_indices: Sequence[int],
    *,
    force_standard_deviation_n: float,
    force_correlation_time_s: float,
    force_amplitude_scale: float = 1.0,
    measurement_noise_psd: ArrayLike | None = None,
) -> FloatArray:
    """Return acceleration auto-PSDs under scalar OU excitation.

    The output has shape ``(n_frequency, n_sensor)``. The OU force spectrum uses
    angular frequency ``omega``:

    ``S_f(omega) = 2 sigma_f**2 tau / (1 + (omega tau)**2)``.
    """

    mass = np.asarray(mass_matrix_kg, dtype=np.float64)
    damping = np.asarray(damping_matrix_n_s_per_m, dtype=np.float64)
    stiffness = np.asarray(stiffness_matrix_n_per_m, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    distribution = np.asarray(force_distribution, dtype=np.float64)
    n_dof = mass.shape[0]

    for matrix, name in (
        (mass, "mass"),
        (damping, "damping"),
        (stiffness, "stiffness"),
    ):
        if matrix.shape != (n_dof, n_dof) or not np.all(np.isfinite(matrix)):
            raise IdentifiabilityError(f"The {name} matrix has invalid dimensions.")
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise IdentifiabilityError("frequencies_hz must be one-dimensional.")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise IdentifiabilityError("frequencies_hz must be finite and positive.")
    if distribution.shape != (n_dof,) or not np.all(np.isfinite(distribution)):
        raise IdentifiabilityError("force_distribution must have shape (n_dof,).")
    if np.all(distribution == 0.0):
        raise IdentifiabilityError("force_distribution must be nonzero.")

    sensor_indices = tuple(sensor_dof_indices)
    if not sensor_indices or len(set(sensor_indices)) != len(sensor_indices):
        raise IdentifiabilityError("Sensor indices must be nonempty and unique.")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < n_dof
        for index in sensor_indices
    ):
        raise IdentifiabilityError("A sensor index lies outside the structural model.")

    force_standard_deviation = float(force_standard_deviation_n)
    correlation_time = float(force_correlation_time_s)
    amplitude_scale = float(force_amplitude_scale)
    positive_scalars = np.asarray(
        [force_standard_deviation, correlation_time, amplitude_scale],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(positive_scalars)) or np.any(positive_scalars <= 0.0):
        raise IdentifiabilityError(
            "Force scale, correlation time, and amplitude scale must be positive."
        )

    if measurement_noise_psd is None:
        noise = np.zeros((frequencies.size, len(sensor_indices)), dtype=np.float64)
    else:
        noise_input = np.asarray(measurement_noise_psd, dtype=np.float64)
        if noise_input.shape == (len(sensor_indices),):
            noise = np.broadcast_to(
                noise_input,
                (frequencies.size, len(sensor_indices)),
            )
        elif noise_input.shape == (frequencies.size, len(sensor_indices)):
            noise = noise_input
        else:
            raise IdentifiabilityError("measurement_noise_psd has invalid dimensions.")
        if not np.all(np.isfinite(noise)) or np.any(noise < 0.0):
            raise IdentifiabilityError("measurement_noise_psd must be nonnegative.")

    angular_frequencies = 2.0 * np.pi * frequencies
    force_psd = (
        2.0
        * (amplitude_scale * force_standard_deviation) ** 2
        * correlation_time
        / (1.0 + (angular_frequencies * correlation_time) ** 2)
    )
    dynamic_stiffness = (
        stiffness[None, :, :]
        - angular_frequencies[:, None, None] ** 2 * mass[None, :, :]
        + 1j * angular_frequencies[:, None, None] * damping[None, :, :]
    )
    distributed_force = np.broadcast_to(
        distribution[None, :, None],
        (frequencies.size, n_dof, 1),
    )
    displacement_transfer = np.linalg.solve(
        dynamic_stiffness,
        distributed_force,
    )[:, :, 0]
    acceleration_transfer = (
        -angular_frequencies[:, None] ** 2 * displacement_transfer
    )
    selected_transfer = acceleration_transfer[:, list(sensor_indices)]
    output_psd = np.abs(selected_transfer) ** 2 * force_psd[:, None] + noise

    if not np.all(np.isfinite(output_psd)) or np.any(output_psd <= 0.0):
        raise IdentifiabilityError(
            "Calculated acceleration PSD is not positive finite."
        )
    return output_psd


def log_spectral_features(auto_psd: ArrayLike) -> FloatArray:
    """Flatten positive auto-PSDs after a natural-log transform."""

    spectral_density = np.asarray(auto_psd, dtype=np.float64)
    if spectral_density.ndim != 2 or not np.all(np.isfinite(spectral_density)):
        raise IdentifiabilityError("auto_psd must be a finite two-dimensional array.")
    if np.any(spectral_density <= 0.0):
        raise IdentifiabilityError("auto_psd must be strictly positive.")
    return np.log(spectral_density).reshape(-1)


def central_difference_sensitivity(
    feature_function: Callable[[FloatArray], FloatArray],
    reference_parameters: ArrayLike,
    parameter_steps: ArrayLike,
) -> FloatArray:
    """Calculate a central finite-difference feature Jacobian."""

    reference = np.asarray(reference_parameters, dtype=np.float64)
    steps = np.asarray(parameter_steps, dtype=np.float64)
    if reference.ndim != 1 or reference.size < 1 or not np.all(np.isfinite(reference)):
        raise IdentifiabilityError("reference_parameters must be a finite vector.")
    if steps.shape != reference.shape or not np.all(np.isfinite(steps)):
        raise IdentifiabilityError("parameter_steps must match the reference vector.")
    if np.any(steps <= 0.0) or np.any(reference - steps <= 0.0):
        raise IdentifiabilityError(
            "Central-difference steps must be positive and preserve positive "
            "parameters."
        )

    baseline_features = np.asarray(feature_function(reference), dtype=np.float64)
    if baseline_features.ndim != 1 or not np.all(np.isfinite(baseline_features)):
        raise IdentifiabilityError("feature_function must return a finite vector.")
    sensitivity = np.empty(
        (baseline_features.size, reference.size),
        dtype=np.float64,
    )
    for parameter_index, step in enumerate(steps):
        upper = reference.copy()
        lower = reference.copy()
        upper[parameter_index] += step
        lower[parameter_index] -= step
        upper_features = np.asarray(feature_function(upper), dtype=np.float64)
        lower_features = np.asarray(feature_function(lower), dtype=np.float64)
        if upper_features.shape != baseline_features.shape:
            raise IdentifiabilityError("feature_function changed output dimensions.")
        if lower_features.shape != baseline_features.shape:
            raise IdentifiabilityError("feature_function changed output dimensions.")
        sensitivity[:, parameter_index] = (
            upper_features - lower_features
        ) / (2.0 * step)
    return sensitivity


def analyze_identifiability(
    sensitivity_matrix: ArrayLike,
    parameter_names: Sequence[str],
    *,
    effective_rank_relative_threshold: float = 0.01,
) -> IdentifiabilityReport:
    """Analyze sensitivity strength and collinearity using normalized columns.

    Column normalization makes the reported condition number a measure of
    confounding rather than a consequence of parameter units. Original column
    norms are retained as a separate sensitivity-strength diagnostic.
    """

    sensitivity = np.asarray(sensitivity_matrix, dtype=np.float64)
    names = tuple(parameter_names)
    threshold = float(effective_rank_relative_threshold)
    if sensitivity.ndim != 2 or sensitivity.shape[0] < 1:
        raise IdentifiabilityError("sensitivity_matrix must be two-dimensional.")
    if sensitivity.shape[1] != len(names) or len(names) < 1:
        raise IdentifiabilityError("parameter_names must match sensitivity columns.")
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise IdentifiabilityError("parameter_names must be nonempty and unique.")
    if not np.all(np.isfinite(sensitivity)):
        raise IdentifiabilityError("sensitivity_matrix must be finite.")
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise IdentifiabilityError(
            "effective_rank_relative_threshold must lie strictly within (0, 1)."
        )

    column_norms = np.linalg.norm(sensitivity, axis=0)
    if np.any(column_norms == 0.0):
        raise IdentifiabilityError("Every parameter must have nonzero sensitivity.")
    normalized_sensitivity = sensitivity / column_norms
    parameter_correlation = normalized_sensitivity.T @ normalized_sensitivity
    _, singular_values, right_vectors = np.linalg.svd(
        normalized_sensitivity,
        full_matrices=False,
    )
    relative_singular_values = singular_values / singular_values[0]
    effective_rank = int(np.sum(relative_singular_values >= threshold))
    if singular_values[-1] <= np.finfo(float).eps * singular_values[0]:
        condition_number = float("inf")
    else:
        condition_number = float(singular_values[0] / singular_values[-1])

    least_identifiable = right_vectors[-1].copy()
    dominant_index = int(np.argmax(np.abs(least_identifiable)))
    if least_identifiable[dominant_index] < 0.0:
        least_identifiable *= -1.0

    return IdentifiabilityReport(
        parameter_names=names,
        sensitivity_matrix=sensitivity.copy(),
        column_norms=column_norms,
        normalized_parameter_correlation=parameter_correlation,
        normalized_singular_values=relative_singular_values,
        effective_rank=effective_rank,
        normalized_condition_number=condition_number,
        least_identifiable_combination=least_identifiable,
    )
