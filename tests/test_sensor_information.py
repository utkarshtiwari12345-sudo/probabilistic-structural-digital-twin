"""Tests for paired sensor layouts and damage localisation."""

import numpy as np
import pytest

from probdt.monitoring import (
    SensorInformationError,
    SensorLayout,
    sensor_subset_columns,
    subset_sensor_spectra,
    top_k_localization_success,
)


def test_sensor_subset_mapping_preserves_target_order() -> None:
    reference = SensorLayout("reference", (0, 1, 2, 3))
    target = SensorLayout("target", (3, 1))

    assert sensor_subset_columns(reference, target) == (3, 1)


def test_sensor_spectral_subset_preserves_pairing() -> None:
    spectra = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)

    subset = subset_sensor_spectra(spectra, (3, 1))

    np.testing.assert_array_equal(subset, spectra[..., [3, 1]])
    assert not np.shares_memory(subset, spectra)


def test_top_k_localization_handles_single_and_multiple_damage() -> None:
    assert top_k_localization_success([0.8, 0.2, 0.1], [True, False, False])
    assert top_k_localization_success([0.7, 0.1, 0.6], [True, False, True])
    assert not top_k_localization_success(
        [0.7, 0.6, 0.1],
        [True, False, True],
    )


def test_invalid_sensor_information_inputs_are_rejected() -> None:
    with pytest.raises(SensorInformationError, match="reference layout"):
        sensor_subset_columns(
            SensorLayout("reference", (0, 1)),
            SensorLayout("target", (2,)),
        )
    with pytest.raises(SensorInformationError):
        subset_sensor_spectra(np.ones((3, 2)), (2,))
    with pytest.raises(SensorInformationError):
        top_k_localization_success([0.5, 0.4], [True, True])
