"""Verification tests for state-space and direct structural dynamics."""

import numpy as np
import pytest
from scipy.linalg import expm

from probdt.structural import (
    DynamicsError,
    assemble_mass_matrix,
    assemble_rayleigh_damping,
    assemble_stiffness_matrix,
    build_continuous_state_space,
    calculate_energy_balance,
    discretize_state_space_zoh,
    fit_rayleigh_coefficients,
    simulate_discrete_state_space,
    simulate_newmark_average_acceleration,
    solve_modal_properties,
)


@pytest.fixture
def two_dof_model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a small damped physical model for state-space checks."""

    mass = assemble_mass_matrix([2.0, 1.0])
    stiffness = assemble_stiffness_matrix([50.0, 30.0])
    damping = 0.05 * mass + 0.002 * stiffness
    return mass, damping, stiffness


def test_state_space_is_equivalent_to_second_order_equilibrium(
    two_dof_model: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    mass, damping, stiffness = two_dof_model
    distribution = np.array([0.0, 1.0])
    model = build_continuous_state_space(
        mass,
        damping,
        stiffness,
        force_distribution=distribution,
    )
    displacement = np.array([0.01, -0.02])
    velocity = np.array([0.2, -0.1])
    force = np.array([12.0])
    state = np.concatenate((displacement, velocity))

    derivative = model.state_matrix_per_s @ state + model.input_matrix @ force
    expected_acceleration = np.linalg.solve(
        mass,
        distribution * force[0] - damping @ velocity - stiffness @ displacement,
    )

    np.testing.assert_allclose(derivative[:2], velocity)
    np.testing.assert_allclose(derivative[2:], expected_acceleration)


def test_damped_state_space_is_asymptotically_stable(
    two_dof_model: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    mass, damping, stiffness = two_dof_model
    model = build_continuous_state_space(mass, damping, stiffness)

    eigenvalues = np.linalg.eigvals(model.state_matrix_per_s)

    assert np.max(eigenvalues.real) < 0.0


def test_zero_order_hold_state_transition_matches_matrix_exponential(
    two_dof_model: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    mass, damping, stiffness = two_dof_model
    continuous = build_continuous_state_space(mass, damping, stiffness)
    time_step = 0.01
    discrete = discretize_state_space_zoh(continuous, time_step)

    np.testing.assert_allclose(
        discrete.state_matrix,
        expm(continuous.state_matrix_per_s * time_step),
        rtol=1.0e-13,
        atol=1.0e-14,
    )


def test_discrete_propagation_preserves_initial_state_for_one_sample(
    two_dof_model: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    mass, damping, stiffness = two_dof_model
    continuous = build_continuous_state_space(mass, damping, stiffness)
    discrete = discretize_state_space_zoh(continuous, 0.01)
    initial = np.arange(4, dtype=float)

    states = simulate_discrete_state_space(discrete, np.zeros((1, 2)), initial)

    np.testing.assert_array_equal(states, initial[np.newaxis, :])


def test_newmark_matches_undamped_sdof_closed_form() -> None:
    mass = np.array([[4.0]])
    damping = np.array([[0.0]])
    stiffness = np.array([[100.0]])
    time = np.arange(0.0, 2.0 + 0.0005, 0.001)
    force = np.zeros((time.size, 1))
    initial_displacement = np.array([0.01])

    response = simulate_newmark_average_acceleration(
        mass,
        damping,
        stiffness,
        time,
        force,
        initial_displacement_m=initial_displacement,
    )
    exact = initial_displacement[0] * np.cos(5.0 * time)

    np.testing.assert_allclose(
        response.displacement_m[:, 0],
        exact,
        rtol=0.0,
        atol=2.0e-7,
    )


def test_unforced_damped_response_loses_mechanical_energy() -> None:
    mass = np.array([[2.0]])
    damping = np.array([[0.4]])
    stiffness = np.array([[50.0]])
    time = np.arange(0.0, 5.0 + 0.0005, 0.001)
    force = np.zeros((time.size, 1))
    response = simulate_newmark_average_acceleration(
        mass,
        damping,
        stiffness,
        time,
        force,
        initial_displacement_m=[0.01],
    )
    energy = calculate_energy_balance(response, mass, damping, stiffness, force)

    assert np.max(np.diff(energy.mechanical_energy_j)) <= 1.0e-12
    normalized_balance_error = np.max(np.abs(energy.balance_residual_j)) / (
        energy.mechanical_energy_j[0]
    )
    assert normalized_balance_error < 1.0e-5


def test_forced_response_satisfies_equilibrium_at_every_sample(
    two_dof_model: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    mass, damping, stiffness = two_dof_model
    time = np.arange(0.0, 2.0 + 0.005, 0.01)
    force = np.zeros((time.size, 2))
    force[:, 1] = 10.0 * np.sin(3.0 * time)
    response = simulate_newmark_average_acceleration(
        mass,
        damping,
        stiffness,
        time,
        force,
    )

    residual = (
        response.acceleration_m_per_s2 @ mass.T
        + response.velocity_m_per_s @ damping.T
        + response.displacement_m @ stiffness.T
        - force
    )

    assert np.max(np.abs(residual)) < 1.0e-10


def test_newmark_displacement_error_decreases_at_second_order() -> None:
    mass = np.array([[4.0]])
    damping = np.array([[0.0]])
    stiffness = np.array([[100.0]])
    errors: list[float] = []

    for time_step in (0.04, 0.02, 0.01):
        time = np.arange(0.0, 4.0 + 0.5 * time_step, time_step)
        force = np.zeros((time.size, 1))
        response = simulate_newmark_average_acceleration(
            mass,
            damping,
            stiffness,
            time,
            force,
            initial_displacement_m=[0.01],
        )
        exact = 0.01 * np.cos(5.0 * time)
        displacement_error = response.displacement_m[:, 0] - exact
        errors.append(float(np.sqrt(np.mean(displacement_error**2))))

    first_order = np.log(errors[0] / errors[1]) / np.log(2.0)
    second_order = np.log(errors[1] / errors[2]) / np.log(2.0)
    assert first_order > 1.9
    assert second_order > 1.9


def test_reference_damped_model_can_be_integrated() -> None:
    mass = assemble_mass_matrix([420000.0, 400000.0, 380000.0])
    stiffness = assemble_stiffness_matrix([30000000.0, 28000000.0, 26000000.0])
    modes = solve_modal_properties(mass, stiffness)
    alpha, beta = fit_rayleigh_coefficients(
        modes.angular_frequencies_rad_s[0],
        modes.angular_frequencies_rad_s[2],
        0.015,
        0.015,
    )
    damping = assemble_rayleigh_damping(
        mass,
        stiffness,
        alpha_per_s=alpha,
        beta_s=beta,
    ).damping_matrix_n_s_per_m
    time = np.arange(0.0, 1.0 + 0.005, 0.01)
    force = np.zeros((time.size, 3))

    response = simulate_newmark_average_acceleration(
        mass,
        damping,
        stiffness,
        time,
        force,
        initial_displacement_m=[0.001, 0.002, 0.003],
    )

    assert np.all(np.isfinite(response.displacement_m))


@pytest.mark.parametrize(
    "time",
    [
        [0.0],
        [0.0, 0.1, 0.1],
        [0.0, 0.1, 0.21],
        [0.0, np.nan],
    ],
)
def test_invalid_time_grids_are_rejected(
    two_dof_model: tuple[np.ndarray, np.ndarray, np.ndarray],
    time: list[float],
) -> None:
    mass, damping, stiffness = two_dof_model
    force = np.zeros((len(time), 2))

    with pytest.raises(DynamicsError, match=r"time_s|uniform"):
        simulate_newmark_average_acceleration(
            mass,
            damping,
            stiffness,
            time,
            force,
        )
