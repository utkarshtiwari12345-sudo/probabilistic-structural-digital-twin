"""Tests for Stage 7 UKF and EnKF parameter filters."""

import numpy as np
import pytest

from probdt.inference import (
    NonlinearFilterError,
    corrected_log_periodogram,
    log_periodogram_noise_variance,
    reflect_parameters,
    run_ensemble_parameter_filter,
    run_unscented_parameter_filter,
)


def measurement_function(parameters: np.ndarray) -> np.ndarray:
    return np.array(
        [
            parameters[0] + 0.2 * parameters[1],
            0.3 * parameters[0] + parameters[1],
            parameters[0] - 0.5 * parameters[1],
        ]
    )


def test_log_periodogram_correction_has_expected_moments() -> None:
    rng = np.random.default_rng(42)
    expected_psd = 3.0
    averages = 8
    samples = expected_psd * rng.gamma(
        averages,
        1.0 / averages,
        size=100000,
    )

    corrected = corrected_log_periodogram(samples, averages)

    assert np.mean(corrected) == pytest.approx(np.log(expected_psd), abs=0.003)
    assert np.var(corrected) == pytest.approx(
        log_periodogram_noise_variance(averages),
        rel=0.02,
    )


def test_parameter_reflection_preserves_bounds_without_clipping() -> None:
    reflected = reflect_parameters(
        [[0.4, 1.1], [0.7, 0.9]],
        [[0.5, 1.0], [0.5, 1.0]],
    )

    np.testing.assert_allclose(reflected, [[0.6, 0.9], [0.7, 0.9]])


def test_ukf_tracks_a_decreasing_parameter_with_positive_covariance() -> None:
    truths = np.array([[0.98, 0.98], [0.92, 0.97], [0.84, 0.96]])
    observations = np.stack([measurement_function(truth) for truth in truths])
    history = run_unscented_parameter_filter(
        observations,
        measurement_function,
        ("first", "second"),
        [(0.5, 1.0), (0.5, 1.0)],
        initial_mean=[0.98, 0.98],
        initial_covariance=np.diag([0.03**2, 0.03**2]),
        process_covariance=np.diag([0.04**2, 0.04**2]),
        observation_covariance=np.eye(3) * 1.0e-4,
    )

    assert history.posterior_means[-1, 0] < history.posterior_means[0, 0]
    assert np.max(np.abs(history.posterior_means[-1] - truths[-1])) < 0.04
    assert np.all(np.linalg.eigvalsh(history.posterior_covariances) > 0.0)


def test_enkf_is_reproducible_and_tracks_the_same_sequence() -> None:
    truths = np.array([[0.98, 0.98], [0.92, 0.97], [0.84, 0.96]])
    observations = np.stack([measurement_function(truth) for truth in truths])
    arguments = {
        "observations": observations,
        "measurement_function": measurement_function,
        "parameter_names": ("first", "second"),
        "bounds": [(0.5, 1.0), (0.5, 1.0)],
        "initial_mean": [0.98, 0.98],
        "initial_covariance": np.diag([0.03**2, 0.03**2]),
        "process_covariance": np.diag([0.04**2, 0.04**2]),
        "observation_covariance": np.eye(3) * 1.0e-4,
        "n_ensemble": 200,
    }
    first = run_ensemble_parameter_filter(
        **arguments,
        rng=np.random.default_rng(7),
    )
    second = run_ensemble_parameter_filter(
        **arguments,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_array_equal(first.posterior_means, second.posterior_means)
    assert np.max(np.abs(first.posterior_means[-1] - truths[-1])) < 0.03


def test_invalid_observation_covariance_is_rejected() -> None:
    with pytest.raises(NonlinearFilterError, match="positive definite"):
        run_unscented_parameter_filter(
            [[1.0, 1.0, 1.0]],
            measurement_function,
            ("first", "second"),
            [(0.5, 1.0), (0.5, 1.0)],
            initial_mean=[0.9, 0.9],
            initial_covariance=np.eye(2),
            process_covariance=np.eye(2),
            observation_covariance=np.zeros((3, 3)),
        )
