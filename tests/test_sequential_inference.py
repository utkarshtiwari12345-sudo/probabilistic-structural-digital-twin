"""Tests for recursive spectral Bayesian updating."""

import numpy as np
import pytest

from probdt.inference import (
    SequentialInferenceError,
    predict_random_walk_covariance,
    run_recursive_spectral_map,
    run_recursive_spectral_map_varying_observations,
)


def exponential_forward(parameters: np.ndarray) -> np.ndarray:
    design = np.array([[1.0, 0.2], [0.3, 1.0], [1.0, -0.5]])
    return np.exp(design @ parameters)[:, None]


def test_random_walk_prediction_adds_process_covariance_exactly() -> None:
    posterior = np.array([[0.04, 0.01], [0.01, 0.03]])
    process = np.diag([0.002, 0.003])

    predicted = predict_random_walk_covariance(posterior, process)

    np.testing.assert_allclose(predicted, posterior + process)
    assert np.linalg.eigvalsh(predicted)[0] > 0.0


def test_recursive_update_propagates_full_posterior() -> None:
    truths = np.array([[0.95, 0.95], [0.90, 0.92], [0.84, 0.90]])
    observations = np.stack([exponential_forward(truth) for truth in truths])
    process = np.diag([0.03**2, 0.03**2])

    history = run_recursive_spectral_map(
        exponential_forward,
        observations,
        ("first", "second"),
        [(0.5, 1.0), (0.5, 1.0)],
        initial_prior_mean=[0.95, 0.95],
        initial_prior_covariance=np.diag([0.2**2, 0.2**2]),
        process_covariance=process,
        n_independent_averages=100,
        fallback_initial_points=[[0.8, 0.8]],
    )

    np.testing.assert_allclose(
        history.prior_covariances[1],
        history.posterior_covariances[0] + process,
    )
    assert np.all(np.diff(history.posterior_means[:, 0]) < 0.0)
    assert np.all(np.diff(history.posterior_means[:, 1]) < 0.0)
    assert np.max(np.abs(history.posterior_means[-1] - truths[-1])) < 0.06
    assert np.all(history.optimizer_success)
    assert np.all(
        np.linalg.eigvalsh(history.posterior_covariances)[:, 0] > 0.0
    )


def test_loss_probability_increases_for_a_deteriorating_parameter() -> None:
    truths = np.array([[0.99, 0.99], [0.90, 0.99], [0.80, 0.99]])
    observations = np.stack([exponential_forward(truth) for truth in truths])
    history = run_recursive_spectral_map(
        exponential_forward,
        observations,
        ("first", "second"),
        [(0.5, 1.0), (0.5, 1.0)],
        initial_prior_mean=[0.99, 0.99],
        initial_prior_covariance=np.diag([0.1**2, 0.1**2]),
        process_covariance=np.diag([0.04**2, 0.04**2]),
        n_independent_averages=100,
    )

    probabilities = history.loss_exceedance_probability(0.05)

    assert probabilities[-1, 0] > probabilities[0, 0]
    assert probabilities[-1, 0] > 0.90
    assert probabilities[-1, 0] > probabilities[-1, 1]


def test_nonpositive_process_covariance_is_rejected() -> None:
    with pytest.raises(SequentialInferenceError, match="semidefinite"):
        predict_random_walk_covariance(
            np.eye(2),
            [[-0.1, 0.0], [0.0, 0.1]],
        )


def test_invalid_spectral_window_shape_is_rejected() -> None:
    with pytest.raises(SequentialInferenceError, match="window, frequency"):
        run_recursive_spectral_map(
            exponential_forward,
            np.ones((3, 2)),
            ("first", "second"),
            [(0.5, 1.0), (0.5, 1.0)],
            initial_prior_mean=[0.9, 0.9],
            initial_prior_covariance=np.eye(2),
            process_covariance=np.eye(2),
            n_independent_averages=4,
        )


def test_varying_observations_support_sensor_dropout() -> None:
    truth = np.array([0.88, 0.92])
    full_observation = exponential_forward(truth)

    def sparse_forward(parameters: np.ndarray) -> np.ndarray:
        return exponential_forward(parameters)[[0, 2]]

    history = run_recursive_spectral_map_varying_observations(
        [exponential_forward, sparse_forward, exponential_forward],
        [full_observation, full_observation[[0, 2]], full_observation],
        ("first", "second"),
        [(0.5, 1.0), (0.5, 1.0)],
        initial_prior_mean=[0.95, 0.95],
        initial_prior_covariance=np.diag([0.1**2, 0.1**2]),
        process_covariance=np.diag([0.01**2, 0.01**2]),
        n_independent_averages=100,
    )

    assert np.all(history.update_performed)
    assert np.all(history.optimizer_success)
    assert np.max(np.abs(history.posterior_means[-1] - truth)) < 0.04


def test_missing_window_skips_update_but_propagates_process_noise() -> None:
    process = np.diag([0.02**2, 0.02**2])
    observation = exponential_forward(np.array([0.9, 0.9]))

    history = run_recursive_spectral_map_varying_observations(
        [exponential_forward, exponential_forward],
        [None, observation],
        ("first", "second"),
        [(0.5, 1.0), (0.5, 1.0)],
        initial_prior_mean=[0.95, 0.95],
        initial_prior_covariance=np.diag([0.1**2, 0.1**2]),
        process_covariance=process,
        n_independent_averages=20,
    )

    assert not history.update_performed[0]
    assert history.update_performed[1]
    np.testing.assert_allclose(history.posterior_means[0], [0.95, 0.95])
    np.testing.assert_allclose(
        history.prior_covariances[1],
        history.posterior_covariances[0] + process,
    )


def test_varying_observation_lengths_must_match() -> None:
    with pytest.raises(SequentialInferenceError, match="equal nonzero"):
        run_recursive_spectral_map_varying_observations(
            [exponential_forward],
            [None, None],
            ("first", "second"),
            [(0.5, 1.0), (0.5, 1.0)],
            initial_prior_mean=[0.95, 0.95],
            initial_prior_covariance=np.eye(2),
            process_covariance=np.zeros((2, 2)),
            n_independent_averages=4,
        )
