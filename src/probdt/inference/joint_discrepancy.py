"""Joint stiffness and spectral-nuisance parameterization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probdt.diagnostics import (
    ModelDiscrepancyError,
    SpectralModelDiscrepancy,
    discrepant_acceleration_auto_psd,
)

FloatArray = NDArray[np.float64]

JOINT_NUISANCE_PARAMETER_NAMES = (
    "mass_scale",
    "damping_scale",
    "excitation_standard_deviation_scale",
    "excitation_correlation_time_scale",
)


class JointDiscrepancyInferenceError(ValueError):
    """Raised when a joint stiffness--nuisance parameter vector is invalid."""


def joint_spectral_parameter_names(
    stiffness_parameter_names: Sequence[str],
) -> tuple[str, ...]:
    """Append the four supported nuisance names to stiffness parameters."""

    stiffness_names = tuple(str(name) for name in stiffness_parameter_names)
    names = (*stiffness_names, *JOINT_NUISANCE_PARAMETER_NAMES)
    if not stiffness_names or len(set(names)) != len(names):
        raise JointDiscrepancyInferenceError(
            "Stiffness parameter names must be nonempty and jointly unique."
        )
    if any(not name.strip() for name in names):
        raise JointDiscrepancyInferenceError(
            "Joint parameter names cannot be empty."
        )
    return names


def split_joint_spectral_parameters(
    parameters: ArrayLike,
    n_stiffness_parameters: int,
) -> tuple[FloatArray, SpectralModelDiscrepancy]:
    """Split ``[stiffness, mass, damping, force scale, force time scale]``."""

    if (
        isinstance(n_stiffness_parameters, bool)
        or not isinstance(n_stiffness_parameters, int)
        or n_stiffness_parameters < 1
    ):
        raise JointDiscrepancyInferenceError(
            "n_stiffness_parameters must be a positive integer."
        )
    values = np.asarray(parameters, dtype=np.float64)
    expected_size = n_stiffness_parameters + len(
        JOINT_NUISANCE_PARAMETER_NAMES
    )
    if values.shape != (expected_size,) or not np.all(np.isfinite(values)):
        raise JointDiscrepancyInferenceError(
            f"Joint parameter vector must contain {expected_size} finite values."
        )
    stiffness = values[:n_stiffness_parameters]
    if np.any(stiffness <= 0.0):
        raise JointDiscrepancyInferenceError(
            "Stiffness multipliers must be positive."
        )
    try:
        discrepancy = SpectralModelDiscrepancy(
            mass_scale=float(values[n_stiffness_parameters]),
            damping_scale=float(values[n_stiffness_parameters + 1]),
            excitation_standard_deviation_scale=float(
                values[n_stiffness_parameters + 2]
            ),
            excitation_correlation_time_scale=float(
                values[n_stiffness_parameters + 3]
            ),
        )
    except ModelDiscrepancyError as error:
        raise JointDiscrepancyInferenceError(str(error)) from error
    return stiffness.copy(), discrepancy


def joint_discrepancy_acceleration_auto_psd(
    mass_matrix_kg: ArrayLike,
    damping_matrix_n_s_per_m: ArrayLike,
    zone_stiffness_components_n_per_m: ArrayLike,
    joint_parameters: ArrayLike,
    frequencies_hz: ArrayLike,
    force_distribution: ArrayLike,
    sensor_dof_indices: tuple[int, ...],
    *,
    force_standard_deviation_n: float,
    force_correlation_time_s: float,
    measurement_noise_psd: ArrayLike | None = None,
) -> FloatArray:
    """Evaluate the spectrum for jointly inferred stiffness and nuisances."""

    components = np.asarray(
        zone_stiffness_components_n_per_m,
        dtype=np.float64,
    )
    if components.ndim != 3 or components.shape[0] < 1:
        raise JointDiscrepancyInferenceError(
            "zone_stiffness_components_n_per_m must contain zonal matrices."
        )
    stiffness, discrepancy = split_joint_spectral_parameters(
        joint_parameters,
        components.shape[0],
    )
    return discrepant_acceleration_auto_psd(
        mass_matrix_kg,
        damping_matrix_n_s_per_m,
        components,
        stiffness,
        frequencies_hz,
        force_distribution,
        sensor_dof_indices,
        force_standard_deviation_n=force_standard_deviation_n,
        force_correlation_time_s=force_correlation_time_s,
        discrepancy=discrepancy,
        measurement_noise_psd=measurement_noise_psd,
    )
