"""Tests for augmented time-domain state--parameter inference."""

import numpy as np
import pytest

from probdt.inference import (
    AugmentedStructuralModel,
    JointStateInferenceError,
    run_augmented_ensemble_filter,
    run_augmented_unscented_filter,
    simulate_augmented_truth,
)


def simple_model() -> AugmentedStructuralModel:
    return AugmentedStructuralModel(
        mass_matrix_kg=np.asarray([[2.0]]),
        damping_matrix_n_s_per_m=np.asarray([[0.4]]),
        zone_stiffness_components_n_per_m=np.asarray([[[8.0]]]),
        force_distribution=np.asarray([1.0]),
        force_correlation_time_s=0.8,
    )


def filter_arguments(observations: np.ndarray) -> dict[str, object]:
    model = simple_model()
    return {
        "model": model,
        "observations": observations,
        "sensor_dof_indices": (0,),
        "time_step_s": 0.02,
        "theta_bounds": [(0.6, 1.0)],
        "initial_mean": [0.0, 0.0, 0.0, 0.95],
        "initial_covariance": np.diag([0.02**2, 0.05**2, 1.0, 0.04**2]),
        "process_covariance": np.diag([1.0e-8, 1.0e-6, 0.04, 1.0e-8]),
        "observation_covariance": np.asarray([[0.02**2]]),
        "state_scales": [0.02, 0.05, 1.0, 1.0],
    }


def test_augmented_dynamics_and_acceleration_match_equilibrium() -> None:
    model = simple_model()
    state = np.asarray([0.1, -0.2, 3.0, 0.75])

    derivative = model.derivative(state)
    acceleration = model.acceleration(state, (0,))

    expected = (3.0 - 0.4 * -0.2 - 8.0 * 0.75 * 0.1) / 2.0
    assert derivative[0] == pytest.approx(-0.2)
    assert derivative[1] == pytest.approx(expected)
    assert derivative[2] == pytest.approx(-3.0 / 0.8)
    assert derivative[3] == pytest.approx(0.0)
    assert acceleration[0] == pytest.approx(expected)


def test_truth_simulation_is_reproducible_and_bounded() -> None:
    arguments = {
        "model": simple_model(),
        "stiffness_multipliers": [0.85],
        "n_samples": 80,
        "time_step_s": 0.02,
        "force_stationary_standard_deviation_n": 1.0,
        "burn_in_samples": 20,
    }

    first = simulate_augmented_truth(**arguments, rng=np.random.default_rng(5))
    second = simulate_augmented_truth(**arguments, rng=np.random.default_rng(5))

    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(
        first.acceleration_m_per_s2,
        second.acceleration_m_per_s2,
    )
    assert first.states.shape == (80, 4)
    np.testing.assert_allclose(first.states[:, -1], 0.85)


def test_augmented_ukf_handles_missing_measurements() -> None:
    truth = simulate_augmented_truth(
        simple_model(),
        [0.85],
        n_samples=100,
        time_step_s=0.02,
        force_stationary_standard_deviation_n=1.0,
        burn_in_samples=50,
        rng=np.random.default_rng(6),
    )
    observations = truth.acceleration_m_per_s2.copy()
    observations[::5] = np.nan

    history = run_augmented_unscented_filter(
        **filter_arguments(observations),
    )

    assert history.posterior_means.shape == (100, 4)
    assert np.all(np.isfinite(history.posterior_means))
    assert np.all(np.linalg.eigvalsh(history.posterior_covariances) > 0.0)
    assert np.all(history.posterior_means[:, -1] >= 0.6)
    assert np.all(history.posterior_means[:, -1] <= 1.0)


def test_augmented_enkf_is_reproducible() -> None:
    truth = simulate_augmented_truth(
        simple_model(),
        [0.9],
        n_samples=60,
        time_step_s=0.02,
        force_stationary_standard_deviation_n=1.0,
        burn_in_samples=20,
        rng=np.random.default_rng(8),
    )
    arguments = filter_arguments(truth.acceleration_m_per_s2)

    first = run_augmented_ensemble_filter(
        **arguments,
        n_ensemble=30,
        rng=np.random.default_rng(9),
    )
    second = run_augmented_ensemble_filter(
        **arguments,
        n_ensemble=30,
        rng=np.random.default_rng(9),
    )

    np.testing.assert_array_equal(first.posterior_means, second.posterior_means)
    np.testing.assert_array_equal(
        first.posterior_covariances,
        second.posterior_covariances,
    )


def test_invalid_augmented_filter_inputs_are_rejected() -> None:
    arguments = filter_arguments(np.ones((3, 1)))
    arguments["theta_bounds"] = [(1.0, 0.6)]
    with pytest.raises(JointStateInferenceError, match="theta_bounds"):
        run_augmented_unscented_filter(**arguments)
