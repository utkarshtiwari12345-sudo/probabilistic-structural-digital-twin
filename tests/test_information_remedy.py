"""Tests for heterogeneous Stage 14 observation channels."""

import numpy as np
import pytest

from probdt.inference import (
    AugmentedStructuralModel,
    JointStateInferenceError,
    MixedObservationModel,
    run_augmented_unscented_filter,
    simulate_augmented_truth,
)


def simple_model() -> AugmentedStructuralModel:
    return AugmentedStructuralModel(
        mass_matrix_kg=np.diag([2.0, 3.0]),
        damping_matrix_n_s_per_m=np.diag([0.4, 0.5]),
        zone_stiffness_components_n_per_m=np.asarray(
            [
                [[8.0, -3.0], [-3.0, 3.0]],
                [[0.0, 0.0], [0.0, 5.0]],
            ]
        ),
        force_distribution=np.asarray([0.4, 1.0]),
        force_correlation_time_s=0.8,
    )


def mixed_model() -> MixedObservationModel:
    return MixedObservationModel(
        structural_model=simple_model(),
        acceleration_dof_indices=(0, 1),
        displacement_dof_indices=(1,),
        strain_story_indices=(0, 1),
        level_spacing_m=2.0,
        modal_projection_vectors=np.asarray([[0.25, 0.75]]),
        include_latent_force=True,
    )


def test_mixed_observation_channel_order_and_values() -> None:
    model = mixed_model()
    state = np.asarray([0.2, 0.5, -0.1, 0.3, 4.0, 0.9, 0.8])

    observed = model.observe(state)

    np.testing.assert_allclose(observed[:2], simple_model().acceleration(state, (0, 1)))
    assert observed[2] == pytest.approx(4.0)
    assert observed[3] == pytest.approx(0.5)
    np.testing.assert_allclose(observed[4:6], [0.1, 0.15])
    assert observed[6] == pytest.approx(0.425)
    assert model.channel_types() == (
        "acceleration",
        "acceleration",
        "force",
        "displacement",
        "strain",
        "strain",
        "modal",
    )


def test_augmented_ukf_accepts_mixed_channels_and_missing_values() -> None:
    physical = simple_model()
    model = mixed_model()
    truth = simulate_augmented_truth(
        physical,
        [0.9, 0.85],
        n_samples=50,
        time_step_s=0.02,
        force_stationary_standard_deviation_n=1.0,
        burn_in_samples=20,
        rng=np.random.default_rng(14),
    )
    channel_indices = tuple(range(model.n_channels))
    observations = model.observe(truth.states)
    observations[::7, 1] = np.nan
    history = run_augmented_unscented_filter(
        model,
        observations,
        channel_indices,
        time_step_s=0.02,
        theta_bounds=[(0.6, 1.0), (0.6, 1.0)],
        initial_mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.95],
        initial_covariance=np.diag(
            [0.02**2, 0.02**2, 0.05**2, 0.05**2, 1.0, 0.04**2, 0.04**2]
        ),
        process_covariance=np.diag(
            [1e-8, 1e-8, 1e-6, 1e-6, 0.04, 1e-8, 1e-8]
        ),
        observation_covariance=np.eye(model.n_channels) * 1e-4,
        state_scales=[0.02, 0.02, 0.05, 0.05, 1.0, 1.0, 1.0],
    )

    assert history.posterior_means.shape == (50, physical.n_state)
    assert np.all(np.isfinite(history.posterior_means))
    assert not history.observation_available[0, 1]


def test_invalid_mixed_observation_configuration_is_rejected() -> None:
    with pytest.raises(JointStateInferenceError, match="duplicates"):
        MixedObservationModel(
            structural_model=simple_model(),
            acceleration_dof_indices=(0, 0),
        )
    with pytest.raises(JointStateInferenceError, match="modal_projection"):
        MixedObservationModel(
            structural_model=simple_model(),
            acceleration_dof_indices=(0,),
            modal_projection_vectors=np.ones((2, 3)),
        )
