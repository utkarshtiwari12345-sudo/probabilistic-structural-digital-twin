"""Tests for the Stage 5 batch spectral-inference baseline."""

import numpy as np
import pytest

from probdt.inference import (
    BatchInferenceError,
    averaged_periodogram_negative_log_likelihood,
    fit_batch_spectral_map,
    log_spectral_jacobian,
    simulate_averaged_periodogram,
)


def exponential_forward(parameters: np.ndarray) -> np.ndarray:
    """Small positive spectral model with two identifiable parameters."""

    design = np.array([[1.0, 0.2], [0.3, 1.0], [1.0, -0.5]])
    return np.exp(design @ parameters)[:, None]


def test_averaged_periodogram_is_reproducible_and_has_expected_mean() -> None:
    expected = np.full((20000, 2), 3.0)
    first = simulate_averaged_periodogram(expected, 8, np.random.default_rng(42))
    second = simulate_averaged_periodogram(expected, 8, np.random.default_rng(42))

    np.testing.assert_array_equal(first, second)
    assert np.mean(first) == pytest.approx(3.0, rel=0.01)
    assert np.var(first / expected) == pytest.approx(1.0 / 8.0, rel=0.04)


def test_whittle_objective_is_minimized_by_observed_spectrum() -> None:
    observed = np.array([[1.0, 2.0], [4.0, 8.0]])

    at_observation = averaged_periodogram_negative_log_likelihood(
        observed,
        observed,
        4,
    )
    away = averaged_periodogram_negative_log_likelihood(observed, 1.3 * observed, 4)

    assert at_observation < away


def test_log_spectral_jacobian_matches_known_design() -> None:
    parameters = np.array([0.8, 0.9])
    jacobian = log_spectral_jacobian(
        exponential_forward,
        parameters,
        [(0.5, 1.0), (0.5, 1.0)],
    )

    np.testing.assert_allclose(
        jacobian,
        [[1.0, 0.2], [0.3, 1.0], [1.0, -0.5]],
        rtol=1.0e-9,
        atol=1.0e-9,
    )


def test_one_sided_jacobian_is_available_at_physical_upper_bound() -> None:
    jacobian = log_spectral_jacobian(
        exponential_forward,
        [1.0, 1.0],
        [(0.5, 1.0), (0.5, 1.0)],
    )

    np.testing.assert_allclose(
        jacobian,
        [[1.0, 0.2], [0.3, 1.0], [1.0, -0.5]],
        rtol=1.0e-9,
        atol=1.0e-9,
    )


def test_batch_map_recovers_exact_expected_spectrum() -> None:
    truth = np.array([0.78, 0.88])
    observed = exponential_forward(truth)
    estimate = fit_batch_spectral_map(
        exponential_forward,
        observed,
        ("first", "second"),
        [(0.5, 1.0), (0.5, 1.0)],
        prior_mean=[0.9, 0.9],
        prior_standard_deviation=[10.0, 10.0],
        n_independent_averages=100,
        initial_points=[[0.9, 0.9], [0.6, 0.6]],
    )

    assert estimate.optimizer_success
    np.testing.assert_allclose(estimate.map_parameters, truth, atol=2.0e-5)
    np.testing.assert_allclose(
        estimate.posterior_covariance,
        estimate.posterior_covariance.T,
    )
    assert np.linalg.eigvalsh(estimate.posterior_covariance)[0] > 0.0
    assert np.all(estimate.credible_interval_lower <= estimate.map_parameters)
    assert np.all(estimate.credible_interval_upper >= estimate.map_parameters)


def test_full_correlated_prior_covariance_is_supported() -> None:
    truth = np.array([0.78, 0.88])
    estimate = fit_batch_spectral_map(
        exponential_forward,
        exponential_forward(truth),
        ("first", "second"),
        [(0.5, 1.0), (0.5, 1.0)],
        prior_mean=[0.9, 0.9],
        prior_standard_deviation=None,
        prior_covariance=[[0.04, 0.01], [0.01, 0.04]],
        n_independent_averages=100,
    )

    assert estimate.optimizer_success
    assert np.linalg.eigvalsh(estimate.posterior_covariance)[0] > 0.0


def test_ambiguous_prior_description_is_rejected() -> None:
    with pytest.raises(BatchInferenceError, match="exactly one"):
        fit_batch_spectral_map(
            exponential_forward,
            exponential_forward(np.array([0.8, 0.9])),
            ("first", "second"),
            [(0.5, 1.0), (0.5, 1.0)],
            prior_mean=[0.9, 0.9],
            prior_standard_deviation=[0.2, 0.2],
            prior_covariance=[[0.04, 0.0], [0.0, 0.04]],
            n_independent_averages=4,
        )


@pytest.mark.parametrize(
    ("expected", "averages"),
    [([[0.0]], 2), ([[np.nan]], 2), ([[1.0]], 0)],
)
def test_invalid_periodogram_inputs_are_rejected(
    expected: list[list[float]],
    averages: int,
) -> None:
    with pytest.raises(BatchInferenceError):
        simulate_averaged_periodogram(
            expected,
            averages,
            np.random.default_rng(1),
        )


def test_mismatched_spectral_shapes_are_rejected() -> None:
    with pytest.raises(BatchInferenceError, match="equal shape"):
        averaged_periodogram_negative_log_likelihood([[1.0]], [[1.0, 2.0]], 2)


def test_invalid_parameter_bounds_are_rejected() -> None:
    with pytest.raises(BatchInferenceError, match="lower bound"):
        fit_batch_spectral_map(
            exponential_forward,
            exponential_forward(np.array([0.8, 0.9])),
            ("first", "second"),
            [(1.0, 0.5), (0.5, 1.0)],
            prior_mean=[0.9, 0.9],
            prior_standard_deviation=[0.2, 0.2],
            n_independent_averages=4,
        )
