"""State-space construction and direct integration of structural dynamics.

The physical equation is

``M q_ddot + C q_dot + K q = f``.

The module provides two complementary representations:

- a continuous first-order state-space model for sequential estimators;
- the Newmark average-acceleration method for transparent second-order response
  simulation and verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, expm

from probdt.structural.matrices import (
    StructuralMatrixError,
    validate_structural_matrices,
)

FloatArray = NDArray[np.float64]


class DynamicsError(ValueError):
    """Raised when a response simulation is not physically well defined."""


@dataclass(frozen=True)
class ContinuousStateSpace:
    """Continuous model ``x_dot = A x + B u`` with ``x = [q, q_dot]``."""

    state_matrix_per_s: FloatArray
    input_matrix: FloatArray
    force_distribution: FloatArray


@dataclass(frozen=True)
class DiscreteStateSpace:
    """Exact zero-order-hold discretization ``x[k+1] = A_d x[k] + B_d u[k]``."""

    state_matrix: FloatArray
    input_matrix: FloatArray
    time_step_s: float


@dataclass(frozen=True)
class TimeResponse:
    """Physical response histories stored as ``(n_time, n_dof)`` arrays."""

    time_s: FloatArray
    displacement_m: FloatArray
    velocity_m_per_s: FloatArray
    acceleration_m_per_s2: FloatArray


@dataclass(frozen=True)
class EnergyBalance:
    """Mechanical energy, work, dissipation, and balance residual histories."""

    kinetic_energy_j: FloatArray
    strain_energy_j: FloatArray
    mechanical_energy_j: FloatArray
    cumulative_dissipation_j: FloatArray
    cumulative_external_work_j: FloatArray
    balance_residual_j: FloatArray


def _validated_dynamic_matrices(
    mass_matrix_kg: ArrayLike,
    damping_matrix_n_s_per_m: ArrayLike,
    stiffness_matrix_n_per_m: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate and copy the three physical matrices."""

    mass = np.asarray(mass_matrix_kg, dtype=np.float64)
    damping = np.asarray(damping_matrix_n_s_per_m, dtype=np.float64)
    stiffness = np.asarray(stiffness_matrix_n_per_m, dtype=np.float64)
    try:
        validate_structural_matrices(mass, stiffness)
    except StructuralMatrixError as error:
        raise DynamicsError(str(error)) from error

    if damping.shape != mass.shape:
        raise DynamicsError("The damping matrix must match M and K dimensions.")
    if not np.all(np.isfinite(damping)):
        raise DynamicsError("The damping matrix must contain finite values.")
    if not np.allclose(damping, damping.T, rtol=0.0, atol=1.0e-10):
        raise DynamicsError("The damping matrix must be symmetric.")
    if np.linalg.eigvalsh(damping)[0] < -1.0e-12:
        raise DynamicsError("The damping matrix must be positive semidefinite.")

    return mass.copy(), damping.copy(), stiffness.copy()


def build_continuous_state_space(
    mass_matrix_kg: ArrayLike,
    damping_matrix_n_s_per_m: ArrayLike,
    stiffness_matrix_n_per_m: ArrayLike,
    *,
    force_distribution: ArrayLike | None = None,
) -> ContinuousStateSpace:
    """Construct the exact first-order representation of the second-order model.

    ``force_distribution`` has shape ``(n_dof, n_input)``. A one-dimensional
    vector is interpreted as a single input. The default is one independent
    physical force input at every DOF.
    """

    mass, damping, stiffness = _validated_dynamic_matrices(
        mass_matrix_kg,
        damping_matrix_n_s_per_m,
        stiffness_matrix_n_per_m,
    )
    n_dof = mass.shape[0]

    if force_distribution is None:
        distribution = np.eye(n_dof, dtype=np.float64)
    else:
        distribution = np.asarray(force_distribution, dtype=np.float64)
        if distribution.ndim == 1:
            distribution = distribution[:, np.newaxis]
        if distribution.ndim != 2 or distribution.shape[0] != n_dof:
            raise DynamicsError(
                "force_distribution must have shape (n_dof,) or "
                "(n_dof, n_input)."
            )
        if distribution.shape[1] == 0 or not np.all(np.isfinite(distribution)):
            raise DynamicsError("force_distribution must define finite inputs.")

    state_matrix = np.zeros((2 * n_dof, 2 * n_dof), dtype=np.float64)
    state_matrix[:n_dof, n_dof:] = np.eye(n_dof)
    state_matrix[n_dof:, :n_dof] = -np.linalg.solve(mass, stiffness)
    state_matrix[n_dof:, n_dof:] = -np.linalg.solve(mass, damping)

    input_matrix = np.zeros((2 * n_dof, distribution.shape[1]), dtype=np.float64)
    input_matrix[n_dof:] = np.linalg.solve(mass, distribution)

    return ContinuousStateSpace(
        state_matrix_per_s=state_matrix,
        input_matrix=input_matrix,
        force_distribution=distribution.copy(),
    )


def discretize_state_space_zoh(
    state_space: ContinuousStateSpace,
    time_step_s: float,
) -> DiscreteStateSpace:
    """Compute an exact zero-order-hold discretization using a block exponential."""

    if not np.isfinite(time_step_s) or time_step_s <= 0.0:
        raise DynamicsError("time_step_s must be finite and positive.")

    state_matrix = np.asarray(state_space.state_matrix_per_s, dtype=np.float64)
    input_matrix = np.asarray(state_space.input_matrix, dtype=np.float64)
    n_state = state_matrix.shape[0]
    if state_matrix.shape != (n_state, n_state):
        raise DynamicsError("The continuous state matrix must be square.")
    if input_matrix.ndim != 2 or input_matrix.shape[0] != n_state:
        raise DynamicsError("The continuous input matrix has invalid dimensions.")

    n_input = input_matrix.shape[1]
    augmented = np.zeros((n_state + n_input, n_state + n_input), dtype=np.float64)
    augmented[:n_state, :n_state] = state_matrix
    augmented[:n_state, n_state:] = input_matrix
    exponential = expm(augmented * float(time_step_s))

    return DiscreteStateSpace(
        state_matrix=exponential[:n_state, :n_state],
        input_matrix=exponential[:n_state, n_state:],
        time_step_s=float(time_step_s),
    )


def simulate_discrete_state_space(
    model: DiscreteStateSpace,
    input_history: ArrayLike,
    initial_state: ArrayLike,
) -> FloatArray:
    """Propagate a discrete model using one input row per reported state.

    Input row ``k`` is held over ``[t_k, t_{k+1})``. The last input row is
    retained only so input and state histories have equal length.
    """

    state_matrix = np.asarray(model.state_matrix, dtype=np.float64)
    input_matrix = np.asarray(model.input_matrix, dtype=np.float64)
    inputs = np.asarray(input_history, dtype=np.float64)
    initial = np.asarray(initial_state, dtype=np.float64)
    n_state = state_matrix.shape[0]
    n_input = input_matrix.shape[1]

    if state_matrix.shape != (n_state, n_state):
        raise DynamicsError("The discrete state matrix must be square.")
    if input_matrix.shape != (n_state, n_input):
        raise DynamicsError("The discrete input matrix has invalid dimensions.")
    if inputs.ndim != 2 or inputs.shape[1] != n_input or inputs.shape[0] < 1:
        raise DynamicsError("input_history must have shape (n_time, n_input).")
    if initial.shape != (n_state,):
        raise DynamicsError(f"initial_state must have shape ({n_state},).")
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(initial)):
        raise DynamicsError("Inputs and initial state must be finite.")

    states = np.empty((inputs.shape[0], n_state), dtype=np.float64)
    states[0] = initial
    for time_index in range(inputs.shape[0] - 1):
        states[time_index + 1] = (
            state_matrix @ states[time_index]
            + input_matrix @ inputs[time_index]
        )
    return states


def _uniform_time_grid(time_s: ArrayLike) -> tuple[FloatArray, float]:
    """Validate an increasing, uniformly spaced time grid."""

    time = np.asarray(time_s, dtype=np.float64)
    if time.ndim != 1 or time.size < 2:
        raise DynamicsError("time_s must contain at least two samples.")
    if not np.all(np.isfinite(time)):
        raise DynamicsError("time_s must contain finite values.")
    increments = np.diff(time)
    if np.any(increments <= 0.0):
        raise DynamicsError("time_s must be strictly increasing.")
    time_step = float(increments[0])
    if not np.allclose(increments, time_step, rtol=1.0e-10, atol=1.0e-14):
        raise DynamicsError("Newmark integration requires a uniform time grid.")
    return time.copy(), time_step


def simulate_newmark_average_acceleration(
    mass_matrix_kg: ArrayLike,
    damping_matrix_n_s_per_m: ArrayLike,
    stiffness_matrix_n_per_m: ArrayLike,
    time_s: ArrayLike,
    force_history_n: ArrayLike,
    *,
    initial_displacement_m: ArrayLike | None = None,
    initial_velocity_m_per_s: ArrayLike | None = None,
) -> TimeResponse:
    """Integrate response using Newmark ``beta=1/4`` and ``gamma=1/2``.

    This average-acceleration scheme is second-order accurate and unconditionally
    stable for the linear model. The force is represented at every grid point.
    """

    mass, damping, stiffness = _validated_dynamic_matrices(
        mass_matrix_kg,
        damping_matrix_n_s_per_m,
        stiffness_matrix_n_per_m,
    )
    time, time_step = _uniform_time_grid(time_s)
    n_time = time.size
    n_dof = mass.shape[0]
    forces = np.asarray(force_history_n, dtype=np.float64)
    if forces.shape != (n_time, n_dof):
        raise DynamicsError(
            f"force_history_n must have shape ({n_time}, {n_dof})."
        )
    if not np.all(np.isfinite(forces)):
        raise DynamicsError("force_history_n must contain finite values.")

    displacement = np.zeros((n_time, n_dof), dtype=np.float64)
    velocity = np.zeros((n_time, n_dof), dtype=np.float64)
    acceleration = np.zeros((n_time, n_dof), dtype=np.float64)

    if initial_displacement_m is not None:
        initial_displacement = np.asarray(initial_displacement_m, dtype=np.float64)
        if initial_displacement.shape != (n_dof,) or not np.all(
            np.isfinite(initial_displacement)
        ):
            raise DynamicsError(
                f"initial_displacement_m must be finite with shape ({n_dof},)."
            )
        displacement[0] = initial_displacement
    if initial_velocity_m_per_s is not None:
        initial_velocity = np.asarray(initial_velocity_m_per_s, dtype=np.float64)
        if initial_velocity.shape != (n_dof,) or not np.all(
            np.isfinite(initial_velocity)
        ):
            raise DynamicsError(
                f"initial_velocity_m_per_s must be finite with shape ({n_dof},)."
            )
        velocity[0] = initial_velocity

    acceleration[0] = np.linalg.solve(
        mass,
        forces[0] - damping @ velocity[0] - stiffness @ displacement[0],
    )

    beta = 0.25
    gamma = 0.5
    inverse_beta_dt2 = 1.0 / (beta * time_step**2)
    inverse_beta_dt = 1.0 / (beta * time_step)
    effective_stiffness = (
        stiffness
        + gamma * inverse_beta_dt * damping
        + inverse_beta_dt2 * mass
    )
    factorized_stiffness = cho_factor(effective_stiffness, check_finite=False)

    for time_index in range(n_time - 1):
        q_n = displacement[time_index]
        v_n = velocity[time_index]
        a_n = acceleration[time_index]

        effective_force = forces[time_index + 1].copy()
        effective_force += mass @ (
            inverse_beta_dt2 * q_n
            + inverse_beta_dt * v_n
            + (1.0 / (2.0 * beta) - 1.0) * a_n
        )
        effective_force += damping @ (
            gamma * inverse_beta_dt * q_n
            + (gamma / beta - 1.0) * v_n
            + time_step * (gamma / (2.0 * beta) - 1.0) * a_n
        )

        q_next = cho_solve(
            factorized_stiffness,
            effective_force,
            check_finite=False,
        )
        a_next = (
            inverse_beta_dt2 * (q_next - q_n)
            - inverse_beta_dt * v_n
            - (1.0 / (2.0 * beta) - 1.0) * a_n
        )
        v_next = v_n + time_step * ((1.0 - gamma) * a_n + gamma * a_next)

        displacement[time_index + 1] = q_next
        velocity[time_index + 1] = v_next
        acceleration[time_index + 1] = a_next

    return TimeResponse(
        time_s=time,
        displacement_m=displacement,
        velocity_m_per_s=velocity,
        acceleration_m_per_s2=acceleration,
    )


def calculate_energy_balance(
    response: TimeResponse,
    mass_matrix_kg: ArrayLike,
    damping_matrix_n_s_per_m: ArrayLike,
    stiffness_matrix_n_per_m: ArrayLike,
    force_history_n: ArrayLike,
) -> EnergyBalance:
    """Calculate ``E + D - W - E(0)`` using trapezoidal power integration."""

    mass, damping, stiffness = _validated_dynamic_matrices(
        mass_matrix_kg,
        damping_matrix_n_s_per_m,
        stiffness_matrix_n_per_m,
    )
    time, _ = _uniform_time_grid(response.time_s)
    displacement = np.asarray(response.displacement_m, dtype=np.float64)
    velocity = np.asarray(response.velocity_m_per_s, dtype=np.float64)
    forces = np.asarray(force_history_n, dtype=np.float64)
    expected_shape = (time.size, mass.shape[0])
    if displacement.shape != expected_shape or velocity.shape != expected_shape:
        raise DynamicsError("Response arrays do not match time and matrix dimensions.")
    if forces.shape != expected_shape:
        raise DynamicsError("force_history_n does not match the response dimensions.")

    kinetic = 0.5 * np.einsum("ti,ij,tj->t", velocity, mass, velocity)
    strain = 0.5 * np.einsum(
        "ti,ij,tj->t",
        displacement,
        stiffness,
        displacement,
    )
    mechanical = kinetic + strain
    damping_power = np.einsum("ti,ij,tj->t", velocity, damping, velocity)
    external_power = np.einsum("ti,ti->t", velocity, forces)

    increments = np.diff(time)
    cumulative_dissipation = np.zeros(time.size, dtype=np.float64)
    cumulative_work = np.zeros(time.size, dtype=np.float64)
    cumulative_dissipation[1:] = np.cumsum(
        0.5 * increments * (damping_power[:-1] + damping_power[1:])
    )
    cumulative_work[1:] = np.cumsum(
        0.5 * increments * (external_power[:-1] + external_power[1:])
    )
    residual = (
        mechanical
        + cumulative_dissipation
        - cumulative_work
        - mechanical[0]
    )

    return EnergyBalance(
        kinetic_energy_j=kinetic,
        strain_energy_j=strain,
        mechanical_energy_j=mechanical,
        cumulative_dissipation_j=cumulative_dissipation,
        cumulative_external_work_j=cumulative_work,
        balance_residual_j=residual,
    )
