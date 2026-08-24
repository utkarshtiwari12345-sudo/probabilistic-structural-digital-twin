"""Utilities for paired sensor-layout and damage-localisation studies."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probdt.monitoring.sensors import SensorLayout

FloatArray = NDArray[np.float64]


class SensorInformationError(ValueError):
    """Raised when a sensor-information comparison is invalid."""


def sensor_subset_columns(
    reference_layout: SensorLayout,
    target_layout: SensorLayout,
) -> tuple[int, ...]:
    """Map target sensor DOFs to columns of a reference layout."""

    reference_lookup = {
        dof_index: column
        for column, dof_index in enumerate(reference_layout.dof_indices)
    }
    if any(
        dof_index not in reference_lookup
        for dof_index in target_layout.dof_indices
    ):
        raise SensorInformationError(
            "Every target sensor must occur in the reference layout."
        )
    return tuple(reference_lookup[index] for index in target_layout.dof_indices)


def subset_sensor_spectra(
    reference_spectra: ArrayLike,
    sensor_columns: tuple[int, ...],
) -> FloatArray:
    """Select paired sensor channels along the final spectral array axis."""

    spectra = np.asarray(reference_spectra, dtype=np.float64)
    if spectra.ndim < 2 or not np.all(np.isfinite(spectra)):
        raise SensorInformationError(
            "reference_spectra must be a finite array with a sensor axis."
        )
    if not sensor_columns or len(set(sensor_columns)) != len(sensor_columns):
        raise SensorInformationError(
            "sensor_columns must be nonempty and unique."
        )
    if any(
        isinstance(column, bool)
        or not isinstance(column, int)
        or not 0 <= column < spectra.shape[-1]
        for column in sensor_columns
    ):
        raise SensorInformationError("A sensor column lies outside the spectra.")
    return spectra[..., list(sensor_columns)].copy()


def top_k_localization_success(
    final_damage_probabilities: ArrayLike,
    true_damage_mask: ArrayLike,
) -> bool:
    """Test whether the top-k probabilities exactly recover k damaged zones."""

    probabilities = np.asarray(final_damage_probabilities, dtype=np.float64)
    truth = np.asarray(true_damage_mask)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise SensorInformationError(
            "final_damage_probabilities must contain at least two zones."
        )
    if truth.shape != probabilities.shape or truth.dtype != np.bool_:
        raise SensorInformationError(
            "true_damage_mask must be a Boolean vector matching probabilities."
        )
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise SensorInformationError(
            "Damage probabilities must be finite and lie within [0, 1]."
        )
    n_damaged = int(np.sum(truth))
    if not 1 <= n_damaged < truth.size:
        raise SensorInformationError(
            "Truth must contain at least one damaged and one undamaged zone."
        )
    top_indices = np.argpartition(probabilities, -n_damaged)[-n_damaged:]
    predicted = np.zeros_like(truth)
    predicted[top_indices] = True
    return bool(np.array_equal(predicted, truth))
