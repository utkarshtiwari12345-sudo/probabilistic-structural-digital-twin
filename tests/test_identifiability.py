"""Tests for spectral sensitivity and local-identifiability diagnostics."""

import numpy as np
import pytest

from probdt.diagnostics import (
    IdentifiabilityError,
    acceleration_auto_psd,
    analyze_identifiability,
    central_difference_sensitivity,
    log_spectral_features,
)


def test_sdof_acceleration_psd_peaks_near_natural_frequency() -> None:
    mass = np.array([[4.0]])
    stiffness = np.array([[100.0]])
    damping = np.array([[0.4]])
    frequencies = np.linspace(0.2, 1.5, 1000)

    spectral_density = acceleration_auto_psd(
        mass,
        damping,
        stiffness,
        frequencies,
        [1.0],
        [0],
        force_standard_deviation_n=10.0,
        force_correlation_time_s=0.5,
    )
    peak_frequency = frequencies[np.argmax(spectral_density[:, 0])]
    natural_frequency = np.sqrt(100.0 / 4.0) / (2.0 * np.pi)

    assert peak_frequency == pytest.approx(natural_frequency, abs=0.02)


def test_measurement_noise_floor_is_added_exactly() -> None:
    arguments = {
        "mass_matrix_kg": [[4.0]],
        "damping_matrix_n_s_per_m": [[0.4]],
        "stiffness_matrix_n_per_m": [[100.0]],
        "frequencies_hz": [0.5, 1.0],
        "force_distribution": [1.0],
        "sensor_dof_indices": [0],
        "force_standard_deviation_n": 10.0,
        "force_correlation_time_s": 0.5,
    }
    clean = acceleration_auto_psd(**arguments)
    noisy = acceleration_auto_psd(**arguments, measurement_noise_psd=[2.0])

    np.testing.assert_allclose(noisy - clean, 2.0)


def test_log_spectral_features_use_frequency_major_flattening() -> None:
    spectral_density = np.array([[1.0, np.e], [np.e**2, np.e**3]])

    features = log_spectral_features(spectral_density)

    np.testing.assert_allclose(features, [0.0, 1.0, 2.0, 3.0])


def test_central_difference_recovers_known_jacobian() -> None:
    def feature(parameters: np.ndarray) -> np.ndarray:
        return np.array(
            [
                2.0 * parameters[0] + 3.0 * parameters[1],
                parameters[0] - 4.0 * parameters[1],
            ]
        )

    sensitivity = central_difference_sensitivity(
        feature,
        [1.0, 1.0],
        [0.01, 0.01],
    )

    np.testing.assert_allclose(sensitivity, [[2.0, 3.0], [1.0, -4.0]])


def test_orthogonal_sensitivities_are_well_identified() -> None:
    report = analyze_identifiability(
        np.eye(2),
        ("first", "second"),
        effective_rank_relative_threshold=0.01,
    )

    assert report.effective_rank == 2
    assert report.normalized_condition_number == pytest.approx(1.0)
    assert report.maximum_absolute_parameter_correlation == pytest.approx(0.0)


def test_collinear_sensitivities_are_rank_deficient() -> None:
    sensitivity = np.column_stack(([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]))

    report = analyze_identifiability(
        sensitivity,
        ("first", "second"),
        effective_rank_relative_threshold=0.01,
    )

    assert report.effective_rank == 1
    assert np.isinf(report.normalized_condition_number)
    assert report.maximum_absolute_parameter_correlation == pytest.approx(1.0)


@pytest.mark.parametrize(
    "spectral_density",
    [
        [[0.0, 1.0]],
        [[-1.0, 1.0]],
        [[np.nan, 1.0]],
    ],
)
def test_nonpositive_or_nonfinite_psd_is_rejected(
    spectral_density: list[list[float]],
) -> None:
    with pytest.raises(IdentifiabilityError):
        log_spectral_features(spectral_density)


def test_central_difference_preserves_positive_parameters() -> None:
    with pytest.raises(IdentifiabilityError, match="preserve positive"):
        central_difference_sensitivity(
            lambda parameters: parameters,
            [0.01],
            [0.02],
        )
