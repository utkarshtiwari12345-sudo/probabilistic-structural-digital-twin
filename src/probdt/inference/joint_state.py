"""Augmented time-domain state--stiffness filtering from acceleration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve

FloatArray = NDArray[np.float64]


class JointStateInferenceError(ValueError):
    """Raised when an augmented state--parameter model or filter is invalid."""


def _stabilized_covariance(covariance: FloatArray) -> FloatArray:
    """Symmetrize and floor covariance eigenvalues in scaled coordinates."""

    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    stabilized = (
        eigenvectors * np.maximum(eigenvalues, 1.0e-12 * scale)
    ) @ eigenvectors.T
    return 0.5 * (stabilized + stabilized.T)


def _covariance(
    covariance: ArrayLike,
    size: int,
    name: str,
    *,
    allow_semidefinite: bool,
) -> FloatArray:
    """Validate one physical-coordinate covariance matrix."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise JointStateInferenceError(
            f"{name} must be a finite ({size}, {size}) matrix."
        )
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise JointStateInferenceError(f"{name} must be symmetric.")
    minimum = float(np.linalg.eigvalsh(matrix)[0])
    if allow_semidefinite:
        if minimum < -1.0e-14:
            raise JointStateInferenceError(
                f"{name} must be positive semidefinite."
            )
    elif minimum <= 0.0:
        raise JointStateInferenceError(f"{name} must be positive definite.")
    return matrix.copy()


def _matrix_square_root(covariance: FloatArray) -> FloatArray:
    """Return a stable symmetric square root for semidefinite covariance."""

    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (covariance + covariance.T)
    )
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T


@dataclass(frozen=True)
class AugmentedStructuralModel:
    """Physical model for ``[q, q_dot, latent load, theta]``."""

    mass_matrix_kg: FloatArray
    damping_matrix_n_s_per_m: FloatArray
    zone_stiffness_components_n_per_m: FloatArray
    force_distribution: FloatArray
    force_correlation_time_s: float

    def __post_init__(self) -> None:
        mass = np.asarray(self.mass_matrix_kg, dtype=np.float64)
        damping = np.asarray(self.damping_matrix_n_s_per_m, dtype=np.float64)
        components = np.asarray(
            self.zone_stiffness_components_n_per_m,
            dtype=np.float64,
        )
        force = np.asarray(self.force_distribution, dtype=np.float64)
        if mass.ndim != 2 or mass.shape[0] < 1 or mass.shape[0] != mass.shape[1]:
            raise JointStateInferenceError("mass_matrix_kg must be square.")
        n_dof = mass.shape[0]
        if damping.shape != mass.shape:
            raise JointStateInferenceError("Damping must match the mass matrix.")
        if (
            components.ndim != 3
            or components.shape[0] < 1
            or components.shape[1:] != mass.shape
        ):
            raise JointStateInferenceError(
                "Stiffness components must have shape (zone, dof, dof)."
            )
        if force.shape != (n_dof,):
            raise JointStateInferenceError(
                "force_distribution must contain one entry per DOF."
            )
        arrays = (mass, damping, components, force)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise JointStateInferenceError("Model arrays must be finite.")
        if not np.allclose(mass, mass.T, rtol=0.0, atol=1.0e-10):
            raise JointStateInferenceError("Mass must be symmetric.")
        if not np.allclose(damping, damping.T, rtol=0.0, atol=1.0e-10):
            raise JointStateInferenceError("Damping must be symmetric.")
        if np.linalg.eigvalsh(mass)[0] <= 0.0:
            raise JointStateInferenceError("Mass must be positive definite.")
        correlation_time = float(self.force_correlation_time_s)
        if not np.isfinite(correlation_time) or correlation_time <= 0.0:
            raise JointStateInferenceError(
                "force_correlation_time_s must be finite and positive."
            )

    @property
    def n_dof(self) -> int:
        return int(np.asarray(self.mass_matrix_kg).shape[0])

    @property
    def n_zones(self) -> int:
        return int(np.asarray(self.zone_stiffness_components_n_per_m).shape[0])

    @property
    def n_state(self) -> int:
        return 2 * self.n_dof + 1 + self.n_zones

    @property
    def load_index(self) -> int:
        return 2 * self.n_dof

    @property
    def theta_slice(self) -> slice:
        return slice(self.load_index + 1, self.n_state)

    def state_names(self) -> tuple[str, ...]:
        return (
            *(f"q_{index + 1}" for index in range(self.n_dof)),
            *(f"v_{index + 1}" for index in range(self.n_dof)),
            "latent_force",
            *(f"theta_{index + 1}" for index in range(self.n_zones)),
        )

    def derivative(self, states: ArrayLike) -> FloatArray:
        """Evaluate the deterministic augmented continuous dynamics."""

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1] != self.n_state or not np.all(np.isfinite(values)):
            raise JointStateInferenceError(
                "states must be finite with the augmented state trailing size."
            )
        flattened = values.reshape(-1, self.n_state)
        q = flattened[:, : self.n_dof]
        velocity = flattened[:, self.n_dof : 2 * self.n_dof]
        load = flattened[:, self.load_index]
        theta = flattened[:, self.theta_slice]
        stiffness_force = np.einsum(
            "ez,zij,ej->ei",
            theta,
            self.zone_stiffness_components_n_per_m,
            q,
        )
        physical_force = load[:, None] * self.force_distribution[None, :]
        net_force = (
            physical_force
            - velocity @ self.damping_matrix_n_s_per_m.T
            - stiffness_force
        )
        acceleration = np.linalg.solve(
            self.mass_matrix_kg,
            net_force.T,
        ).T
        derivative = np.zeros_like(flattened)
        derivative[:, : self.n_dof] = velocity
        derivative[:, self.n_dof : 2 * self.n_dof] = acceleration
        derivative[:, self.load_index] = -load / self.force_correlation_time_s
        return derivative.reshape(values.shape)

    def propagate_rk4(self, states: ArrayLike, time_step_s: float) -> FloatArray:
        """Advance one deterministic step while holding stiffness constant."""

        time_step = float(time_step_s)
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise JointStateInferenceError(
                "time_step_s must be finite and positive."
            )
        values = np.asarray(states, dtype=np.float64)
        first = self.derivative(values)
        second = self.derivative(values + 0.5 * time_step * first)
        third = self.derivative(values + 0.5 * time_step * second)
        fourth = self.derivative(values + time_step * third)
        propagated = values + time_step * (
            first + 2.0 * second + 2.0 * third + fourth
        ) / 6.0
        propagated[..., self.theta_slice] = values[..., self.theta_slice]
        return propagated

    def acceleration(
        self,
        states: ArrayLike,
        sensor_dof_indices: tuple[int, ...],
    ) -> FloatArray:
        """Evaluate acceleration at the selected physical DOFs."""

        values = np.asarray(states, dtype=np.float64)
        derivative = self.derivative(values)
        acceleration = derivative[..., self.n_dof : 2 * self.n_dof]
        if (
            not sensor_dof_indices
            or len(set(sensor_dof_indices)) != len(sensor_dof_indices)
            or min(sensor_dof_indices) < 0
            or max(sensor_dof_indices) >= self.n_dof
        ):
            raise JointStateInferenceError("Sensor DOF indices are invalid.")
        return acceleration[..., list(sensor_dof_indices)]


@dataclass(frozen=True)
class AugmentedTruth:
    """One stationary-response window with known latent augmented states."""

    states: FloatArray
    acceleration_m_per_s2: FloatArray


@dataclass(frozen=True)
class JointStateFilterHistory:
    """Posterior Gaussian approximations at every fast sample."""

    state_names: tuple[str, ...]
    posterior_means: FloatArray
    posterior_covariances: FloatArray
    observation_available: NDArray[np.bool_]

    @property
    def posterior_standard_deviations(self) -> FloatArray:
        return np.sqrt(
            np.diagonal(self.posterior_covariances, axis1=1, axis2=2)
        )


def reflect_stiffness_parameters(
    states: FloatArray,
    theta_slice: slice,
    theta_bounds: FloatArray,
) -> FloatArray:
    """Reflect only stiffness entries into their physical bounds."""

    reflected = np.asarray(states, dtype=np.float64).copy()
    theta = reflected[..., theta_slice]
    lower = theta_bounds[:, 0]
    widths = theta_bounds[:, 1] - lower
    folded = np.mod(theta - lower, 2.0 * widths)
    reflected[..., theta_slice] = lower + np.where(
        folded <= widths,
        folded,
        2.0 * widths - folded,
    )
    return reflected


def simulate_augmented_truth(
    model: AugmentedStructuralModel,
    stiffness_multipliers: ArrayLike,
    *,
    n_samples: int,
    time_step_s: float,
    force_stationary_standard_deviation_n: float,
    burn_in_samples: int,
    rng: np.random.Generator,
) -> AugmentedTruth:
    """Simulate one response window under an unmeasured stationary OU load."""

    theta = np.asarray(stiffness_multipliers, dtype=np.float64)
    if theta.shape != (model.n_zones,) or np.any(theta <= 0.0):
        raise JointStateInferenceError(
            "stiffness_multipliers must contain one positive value per zone."
        )
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise JointStateInferenceError("n_samples must be an integer of at least two.")
    if (
        isinstance(burn_in_samples, bool)
        or not isinstance(burn_in_samples, int)
        or burn_in_samples < 0
    ):
        raise JointStateInferenceError("burn_in_samples must be nonnegative.")
    if not isinstance(rng, np.random.Generator):
        raise JointStateInferenceError("rng must be a numpy.random.Generator.")
    force_scale = float(force_stationary_standard_deviation_n)
    if not np.isfinite(force_scale) or force_scale <= 0.0:
        raise JointStateInferenceError(
            "force_stationary_standard_deviation_n must be positive."
        )
    time_step = float(time_step_s)
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise JointStateInferenceError("time_step_s must be finite and positive.")

    total_samples = n_samples + burn_in_samples
    states = np.zeros((total_samples, model.n_state), dtype=np.float64)
    states[0, model.load_index] = rng.normal(scale=force_scale)
    states[:, model.theta_slice] = theta
    correlation = np.exp(-time_step / model.force_correlation_time_s)
    innovation_scale = force_scale * np.sqrt(1.0 - correlation**2)
    for index in range(total_samples - 1):
        states[index + 1] = model.propagate_rk4(states[index], time_step)
        states[index + 1, model.load_index] += rng.normal(
            scale=innovation_scale
        )
        states[index + 1, model.theta_slice] = theta
    retained = states[burn_in_samples:].copy()
    acceleration = model.acceleration(
        retained,
        tuple(range(model.n_dof)),
    )
    return AugmentedTruth(
        states=retained,
        acceleration_m_per_s2=acceleration,
    )


def _validated_filter_inputs(
    model: AugmentedStructuralModel,
    observations: ArrayLike,
    sensor_dof_indices: tuple[int, ...],
    theta_bounds: ArrayLike,
    initial_mean: ArrayLike,
    initial_covariance: ArrayLike,
    process_covariance: ArrayLike,
    observation_covariance: ArrayLike,
    state_scales: ArrayLike,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """Validate and scale common augmented-filter inputs."""

    observed = np.asarray(observations, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[0] < 2:
        raise JointStateInferenceError(
            "observations must have shape (time, sensor)."
        )
    if observed.shape[1] != len(sensor_dof_indices):
        raise JointStateInferenceError(
            "observations must match sensor_dof_indices."
        )
    if np.any(np.isinf(observed)) or np.all(np.isnan(observed)):
        raise JointStateInferenceError(
            "observations may contain NaN missing values but not infinities."
        )
    model.acceleration(np.zeros((1, model.n_state)), sensor_dof_indices)
    bounds = np.asarray(theta_bounds, dtype=np.float64)
    if (
        bounds.shape != (model.n_zones, 2)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise JointStateInferenceError("theta_bounds are invalid.")
    mean = np.asarray(initial_mean, dtype=np.float64)
    scales = np.asarray(state_scales, dtype=np.float64)
    if mean.shape != (model.n_state,) or not np.all(np.isfinite(mean)):
        raise JointStateInferenceError("initial_mean must match the state.")
    if scales.shape != mean.shape or np.any(scales <= 0.0) or not np.all(
        np.isfinite(scales)
    ):
        raise JointStateInferenceError("state_scales must be positive and complete.")
    if np.any(mean[model.theta_slice] < bounds[:, 0]) or np.any(
        mean[model.theta_slice] > bounds[:, 1]
    ):
        raise JointStateInferenceError("Initial stiffness lies outside bounds.")
    initial = _covariance(
        initial_covariance,
        model.n_state,
        "initial_covariance",
        allow_semidefinite=False,
    )
    process = _covariance(
        process_covariance,
        model.n_state,
        "process_covariance",
        allow_semidefinite=True,
    )
    observation = _covariance(
        observation_covariance,
        observed.shape[1],
        "observation_covariance",
        allow_semidefinite=False,
    )
    scale_outer = np.outer(scales, scales)
    return (
        observed,
        bounds,
        mean / scales,
        initial / scale_outer,
        process / scale_outer,
        observation,
    )


def _physical_history(
    model: AugmentedStructuralModel,
    normalized_means: FloatArray,
    normalized_covariances: FloatArray,
    scales: FloatArray,
    observations: FloatArray,
) -> JointStateFilterHistory:
    """Convert normalized filter outputs back to physical coordinates."""

    scale_outer = np.outer(scales, scales)
    return JointStateFilterHistory(
        state_names=model.state_names(),
        posterior_means=normalized_means * scales,
        posterior_covariances=normalized_covariances * scale_outer,
        observation_available=np.isfinite(observations),
    )


def run_augmented_unscented_filter(
    model: AugmentedStructuralModel,
    observations: ArrayLike,
    sensor_dof_indices: tuple[int, ...],
    *,
    time_step_s: float,
    theta_bounds: ArrayLike,
    initial_mean: ArrayLike,
    initial_covariance: ArrayLike,
    process_covariance: ArrayLike,
    observation_covariance: ArrayLike,
    state_scales: ArrayLike,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> JointStateFilterHistory:
    """Run an augmented UKF with channel-wise missing acceleration support."""

    (
        observed,
        bounds,
        posterior_mean,
        posterior_covariance,
        process_covariance_scaled,
        observation_covariance_checked,
    ) = _validated_filter_inputs(
        model,
        observations,
        sensor_dof_indices,
        theta_bounds,
        initial_mean,
        initial_covariance,
        process_covariance,
        observation_covariance,
        state_scales,
    )
    time_step = float(time_step_s)
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise JointStateInferenceError("time_step_s must be finite and positive.")
    scaling = np.asarray((alpha, beta, kappa), dtype=np.float64)
    if not np.all(np.isfinite(scaling)) or alpha <= 0.0 or beta < 0.0:
        raise JointStateInferenceError("UKF scaling parameters are invalid.")
    n_state = model.n_state
    lambda_value = alpha**2 * (n_state + kappa) - n_state
    sigma_scale = n_state + lambda_value
    if sigma_scale <= 0.0:
        raise JointStateInferenceError("UKF sigma-point scale must be positive.")
    mean_weights = np.full(2 * n_state + 1, 0.5 / sigma_scale)
    covariance_weights = mean_weights.copy()
    mean_weights[0] = lambda_value / sigma_scale
    covariance_weights[0] = mean_weights[0] + 1.0 - alpha**2 + beta
    scales = np.asarray(state_scales, dtype=np.float64)
    normalized_theta_bounds = bounds / scales[model.theta_slice, None]

    means = np.empty((observed.shape[0], n_state), dtype=np.float64)
    covariances = np.empty(
        (observed.shape[0], n_state, n_state),
        dtype=np.float64,
    )

    def sigma_points(mean: FloatArray, covariance: FloatArray) -> FloatArray:
        cholesky = np.linalg.cholesky(sigma_scale * covariance)
        points = np.vstack((mean, mean + cholesky.T, mean - cholesky.T))
        return reflect_stiffness_parameters(
            points,
            model.theta_slice,
            normalized_theta_bounds,
        )

    for index, observation_row in enumerate(observed):
        if index > 0:
            points = sigma_points(posterior_mean, posterior_covariance)
            propagated = model.propagate_rk4(points * scales, time_step) / scales
            propagated = reflect_stiffness_parameters(
                propagated,
                model.theta_slice,
                normalized_theta_bounds,
            )
            predicted_mean = mean_weights @ propagated
            deviations = propagated - predicted_mean
            predicted_covariance = _stabilized_covariance(
                deviations.T @ (covariance_weights[:, None] * deviations)
                + process_covariance_scaled
            )
        else:
            predicted_mean = posterior_mean.copy()
            predicted_covariance = posterior_covariance.copy()

        available = np.isfinite(observation_row)
        if np.any(available):
            points = sigma_points(predicted_mean, predicted_covariance)
            predicted_observations = model.acceleration(
                points * scales,
                sensor_dof_indices,
            )[:, available]
            observation_mean = mean_weights @ predicted_observations
            state_deviations = points - predicted_mean
            observation_deviations = predicted_observations - observation_mean
            cross_covariance = state_deviations.T @ (
                covariance_weights[:, None] * observation_deviations
            )
            observation_covariance_available = observation_covariance_checked[
                np.ix_(available, available)
            ]
            innovation_covariance = _stabilized_covariance(
                observation_deviations.T
                @ (covariance_weights[:, None] * observation_deviations)
                + observation_covariance_available
            )
            factorized = cho_factor(innovation_covariance, check_finite=False)
            gain = cho_solve(
                factorized,
                cross_covariance.T,
                check_finite=False,
            ).T
            posterior_mean = predicted_mean + gain @ (
                observation_row[available] - observation_mean
            )
            posterior_mean = reflect_stiffness_parameters(
                posterior_mean,
                model.theta_slice,
                normalized_theta_bounds,
            )
            posterior_covariance = _stabilized_covariance(
                predicted_covariance
                - gain @ innovation_covariance @ gain.T
            )
        else:
            posterior_mean = predicted_mean
            posterior_covariance = predicted_covariance
        means[index] = posterior_mean
        covariances[index] = posterior_covariance

    return _physical_history(
        model,
        means,
        covariances,
        scales,
        observed,
    )


def run_augmented_ensemble_filter(
    model: AugmentedStructuralModel,
    observations: ArrayLike,
    sensor_dof_indices: tuple[int, ...],
    *,
    time_step_s: float,
    theta_bounds: ArrayLike,
    initial_mean: ArrayLike,
    initial_covariance: ArrayLike,
    process_covariance: ArrayLike,
    observation_covariance: ArrayLike,
    state_scales: ArrayLike,
    n_ensemble: int,
    rng: np.random.Generator,
) -> JointStateFilterHistory:
    """Run an augmented stochastic EnKF with missing acceleration support."""

    (
        observed,
        bounds,
        normalized_mean,
        normalized_initial_covariance,
        normalized_process_covariance,
        observation_covariance_checked,
    ) = _validated_filter_inputs(
        model,
        observations,
        sensor_dof_indices,
        theta_bounds,
        initial_mean,
        initial_covariance,
        process_covariance,
        observation_covariance,
        state_scales,
    )
    if (
        isinstance(n_ensemble, bool)
        or not isinstance(n_ensemble, int)
        or n_ensemble <= model.n_state + 1
    ):
        raise JointStateInferenceError(
            "n_ensemble must exceed the augmented state dimension plus one."
        )
    if not isinstance(rng, np.random.Generator):
        raise JointStateInferenceError("rng must be a numpy.random.Generator.")
    time_step = float(time_step_s)
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise JointStateInferenceError("time_step_s must be finite and positive.")
    scales = np.asarray(state_scales, dtype=np.float64)
    normalized_theta_bounds = bounds / scales[model.theta_slice, None]
    ensemble = rng.multivariate_normal(
        normalized_mean,
        normalized_initial_covariance,
        size=n_ensemble,
    )
    ensemble = reflect_stiffness_parameters(
        ensemble,
        model.theta_slice,
        normalized_theta_bounds,
    )
    process_root = _matrix_square_root(normalized_process_covariance)
    means = np.empty((observed.shape[0], model.n_state), dtype=np.float64)
    covariances = np.empty(
        (observed.shape[0], model.n_state, model.n_state),
        dtype=np.float64,
    )

    for index, observation_row in enumerate(observed):
        if index > 0:
            ensemble = model.propagate_rk4(ensemble * scales, time_step) / scales
            ensemble += rng.standard_normal(ensemble.shape) @ process_root.T
            ensemble = reflect_stiffness_parameters(
                ensemble,
                model.theta_slice,
                normalized_theta_bounds,
            )
        available = np.isfinite(observation_row)
        if np.any(available):
            predicted_observations = model.acceleration(
                ensemble * scales,
                sensor_dof_indices,
            )[:, available]
            state_mean = np.mean(ensemble, axis=0)
            state_anomalies = ensemble - state_mean
            observation_mean = np.mean(predicted_observations, axis=0)
            observation_anomalies = predicted_observations - observation_mean
            normalization = float(n_ensemble - 1)
            cross_covariance = (
                state_anomalies.T @ observation_anomalies / normalization
            )
            observation_covariance_available = observation_covariance_checked[
                np.ix_(available, available)
            ]
            innovation_covariance = _stabilized_covariance(
                observation_anomalies.T @ observation_anomalies / normalization
                + observation_covariance_available
            )
            factorized = cho_factor(innovation_covariance, check_finite=False)
            gain = cho_solve(
                factorized,
                cross_covariance.T,
                check_finite=False,
            ).T
            observation_root = np.linalg.cholesky(
                observation_covariance_available
            )
            perturbations = (
                rng.standard_normal((n_ensemble, int(np.sum(available))))
                @ observation_root.T
            )
            innovations = (
                observation_row[available]
                + perturbations
                - predicted_observations
            )
            ensemble += innovations @ gain.T
            ensemble = reflect_stiffness_parameters(
                ensemble,
                model.theta_slice,
                normalized_theta_bounds,
            )
        means[index] = np.mean(ensemble, axis=0)
        covariances[index] = _stabilized_covariance(
            np.cov(ensemble, rowvar=False, ddof=1)
        )

    return _physical_history(
        model,
        means,
        covariances,
        scales,
        observed,
    )
