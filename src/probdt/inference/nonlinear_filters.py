"""Bounded UKF and EnKF approximations for slow structural parameters."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma, polygamma
from scipy.stats import norm

from probdt.inference.sequential import SequentialHistory

FloatArray = NDArray[np.float64]
MeasurementFunction = Callable[[FloatArray], FloatArray]


class NonlinearFilterError(ValueError):
    """Raised when a UKF or EnKF input is invalid."""


def corrected_log_periodogram(
    averaged_periodogram: ArrayLike,
    n_independent_averages: int,
) -> FloatArray:
    """Return log spectra corrected for the exact log-Gamma mean bias."""

    spectral = np.asarray(averaged_periodogram, dtype=np.float64)
    if spectral.ndim < 1 or not np.all(np.isfinite(spectral)):
        raise NonlinearFilterError("averaged_periodogram must be finite.")
    if np.any(spectral <= 0.0):
        raise NonlinearFilterError("averaged_periodogram must be positive.")
    averages = _positive_integer(n_independent_averages, "n_independent_averages")
    log_bias = float(digamma(averages) - np.log(averages))
    return np.log(spectral) - log_bias


def log_periodogram_noise_variance(n_independent_averages: int) -> float:
    """Return the exact variance of a log averaged-periodogram ordinate."""

    averages = _positive_integer(n_independent_averages, "n_independent_averages")
    return float(polygamma(1, averages))


def _positive_integer(value: int, name: str) -> int:
    """Validate one positive integer without accepting Boolean values."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NonlinearFilterError(f"{name} must be a positive integer.")
    return value


def _bounds_array(
    bounds: Sequence[tuple[float, float]],
    n_parameters: int,
) -> FloatArray:
    """Return checked finite parameter bounds."""

    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (n_parameters, 2):
        raise NonlinearFilterError("bounds must contain one pair per parameter.")
    if not np.all(np.isfinite(values)) or np.any(values[:, 0] >= values[:, 1]):
        raise NonlinearFilterError("Parameter bounds are invalid.")
    return values


def _covariance(
    values: ArrayLike,
    size: int,
    name: str,
    *,
    allow_semidefinite: bool,
) -> FloatArray:
    """Return one checked symmetric covariance matrix."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise NonlinearFilterError(f"{name} must be a finite ({size}, {size}) matrix.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise NonlinearFilterError(f"{name} must be symmetric.")
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
    if allow_semidefinite:
        if minimum_eigenvalue < -1.0e-14:
            raise NonlinearFilterError(f"{name} must be positive semidefinite.")
    elif minimum_eigenvalue <= 0.0:
        raise NonlinearFilterError(f"{name} must be positive definite.")
    return matrix.copy()


def _stabilized_covariance(covariance: FloatArray) -> FloatArray:
    """Symmetrize covariance and floor only numerical negative eigenvalues."""

    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = 1.0e-12 * scale
    stabilized = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return 0.5 * (stabilized + stabilized.T)


def reflect_parameters(parameters: ArrayLike, bounds: ArrayLike) -> FloatArray:
    """Reflect parameter values into bounds without boundary point masses."""

    values = np.asarray(parameters, dtype=np.float64)
    bound_array = np.asarray(bounds, dtype=np.float64)
    if values.shape[-1] != bound_array.shape[0] or bound_array.shape[1:] != (2,):
        raise NonlinearFilterError("Parameter values and bounds do not match.")
    lower = bound_array[:, 0]
    widths = bound_array[:, 1] - lower
    folded = np.mod(values - lower, 2.0 * widths)
    return lower + np.where(folded <= widths, folded, 2.0 * widths - folded)


def _validated_filter_inputs(
    observations: ArrayLike,
    parameter_names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    initial_mean: ArrayLike,
    initial_covariance: ArrayLike,
    process_covariance: ArrayLike,
    observation_covariance: ArrayLike,
) -> tuple[
    FloatArray,
    tuple[str, ...],
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """Validate common Gaussian-filter arguments."""

    observed = np.asarray(observations, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[0] < 1 or observed.shape[1] < 1:
        raise NonlinearFilterError(
            "observations must have shape (window, observation)."
        )
    if not np.all(np.isfinite(observed)):
        raise NonlinearFilterError("observations must be finite.")
    names = tuple(parameter_names)
    n_parameters = len(names)
    if n_parameters < 1 or len(set(names)) != n_parameters:
        raise NonlinearFilterError("parameter_names must be nonempty and unique.")
    bounds_array = _bounds_array(bounds, n_parameters)
    mean = np.asarray(initial_mean, dtype=np.float64)
    if mean.shape != (n_parameters,) or not np.all(np.isfinite(mean)):
        raise NonlinearFilterError("initial_mean must match the parameters.")
    if np.any(mean < bounds_array[:, 0]) or np.any(mean > bounds_array[:, 1]):
        raise NonlinearFilterError("initial_mean must lie within bounds.")
    initial_cov = _covariance(
        initial_covariance,
        n_parameters,
        "initial_covariance",
        allow_semidefinite=False,
    )
    process_cov = _covariance(
        process_covariance,
        n_parameters,
        "process_covariance",
        allow_semidefinite=True,
    )
    observation_cov = _covariance(
        observation_covariance,
        observed.shape[1],
        "observation_covariance",
        allow_semidefinite=False,
    )
    return (
        observed,
        names,
        bounds_array,
        mean,
        initial_cov,
        process_cov,
        observation_cov,
    )


def _intervals(
    means: FloatArray,
    covariances: FloatArray,
    bounds: FloatArray,
    credible_level: float,
) -> tuple[FloatArray, FloatArray]:
    """Return clipped marginal Gaussian intervals."""

    level = float(credible_level)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise NonlinearFilterError("credible_level must lie within (0, 1).")
    standard_deviations = np.sqrt(
        np.diagonal(covariances, axis1=1, axis2=2)
    )
    quantile = float(norm.ppf(0.5 + level / 2.0))
    lower = np.maximum(
        bounds[:, 0],
        means - quantile * standard_deviations,
    )
    upper = np.minimum(
        bounds[:, 1],
        means + quantile * standard_deviations,
    )
    return lower, upper


def run_unscented_parameter_filter(
    observations: ArrayLike,
    measurement_function: MeasurementFunction,
    parameter_names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    *,
    initial_mean: ArrayLike,
    initial_covariance: ArrayLike,
    process_covariance: ArrayLike,
    observation_covariance: ArrayLike,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    credible_level: float = 0.95,
) -> SequentialHistory:
    """Run a bounded additive-noise unscented parameter filter."""

    (
        observed,
        names,
        bound_array,
        posterior_mean,
        posterior_covariance,
        process_cov,
        observation_cov,
    ) = _validated_filter_inputs(
        observations,
        parameter_names,
        bounds,
        initial_mean,
        initial_covariance,
        process_covariance,
        observation_covariance,
    )
    scaling_parameters = np.asarray([alpha, beta, kappa], dtype=np.float64)
    if not np.all(np.isfinite(scaling_parameters)) or alpha <= 0.0 or beta < 0.0:
        raise NonlinearFilterError("UKF scaling parameters are invalid.")
    n_parameters = len(names)
    lambda_value = alpha**2 * (n_parameters + kappa) - n_parameters
    sigma_scale = n_parameters + lambda_value
    if sigma_scale <= 0.0:
        raise NonlinearFilterError("UKF sigma-point scale must be positive.")
    mean_weights = np.full(2 * n_parameters + 1, 0.5 / sigma_scale)
    covariance_weights = mean_weights.copy()
    mean_weights[0] = lambda_value / sigma_scale
    covariance_weights[0] = mean_weights[0] + 1.0 - alpha**2 + beta

    n_windows = observed.shape[0]
    n_observations = observed.shape[1]
    prior_means = np.empty((n_windows, n_parameters), dtype=np.float64)
    prior_covariances = np.empty(
        (n_windows, n_parameters, n_parameters),
        dtype=np.float64,
    )
    posterior_means = np.empty_like(prior_means)
    posterior_covariances = np.empty_like(prior_covariances)

    for window_index in range(n_windows):
        predicted_mean = posterior_mean.copy()
        predicted_covariance = _stabilized_covariance(
            posterior_covariance + process_cov
        )
        prior_means[window_index] = predicted_mean
        prior_covariances[window_index] = predicted_covariance
        scaled_cholesky = np.linalg.cholesky(sigma_scale * predicted_covariance)
        sigma_points = np.vstack(
            (
                predicted_mean,
                predicted_mean + scaled_cholesky.T,
                predicted_mean - scaled_cholesky.T,
            )
        )
        bounded_sigma_points = reflect_parameters(sigma_points, bound_array)
        predicted_observations = np.stack(
            [measurement_function(point) for point in bounded_sigma_points]
        )
        if predicted_observations.shape != (
            2 * n_parameters + 1,
            n_observations,
        ):
            raise NonlinearFilterError(
                "measurement_function returned inconsistent dimensions."
            )
        observation_mean = mean_weights @ predicted_observations
        state_deviations = bounded_sigma_points - predicted_mean
        observation_deviations = predicted_observations - observation_mean
        cross_covariance = (
            state_deviations.T
            @ (covariance_weights[:, None] * observation_deviations)
        )
        innovation_covariance = (
            observation_deviations.T
            @ (covariance_weights[:, None] * observation_deviations)
            + observation_cov
        )
        innovation_covariance = 0.5 * (
            innovation_covariance + innovation_covariance.T
        )
        factorized = cho_factor(innovation_covariance, check_finite=False)
        gain = cho_solve(
            factorized,
            cross_covariance.T,
            check_finite=False,
        ).T
        posterior_mean = reflect_parameters(
            predicted_mean + gain @ (observed[window_index] - observation_mean),
            bound_array,
        )
        posterior_covariance = _stabilized_covariance(
            predicted_covariance
            - gain @ innovation_covariance @ gain.T
        )
        posterior_means[window_index] = posterior_mean
        posterior_covariances[window_index] = posterior_covariance

    lower, upper = _intervals(
        posterior_means,
        posterior_covariances,
        bound_array,
        credible_level,
    )
    return SequentialHistory(
        parameter_names=names,
        parameter_bounds=bound_array,
        prior_means=prior_means,
        prior_covariances=prior_covariances,
        posterior_means=posterior_means,
        posterior_covariances=posterior_covariances,
        credible_interval_lower=lower,
        credible_interval_upper=upper,
        optimizer_success=np.ones(n_windows, dtype=np.bool_),
        n_function_evaluations=np.full(
            n_windows,
            2 * n_parameters + 1,
            dtype=np.int64,
        ),
    )


def run_ensemble_parameter_filter(
    observations: ArrayLike,
    measurement_function: MeasurementFunction,
    parameter_names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    *,
    initial_mean: ArrayLike,
    initial_covariance: ArrayLike,
    process_covariance: ArrayLike,
    observation_covariance: ArrayLike,
    n_ensemble: int,
    rng: np.random.Generator,
    credible_level: float = 0.95,
) -> SequentialHistory:
    """Run a bounded stochastic ensemble Kalman parameter filter."""

    (
        observed,
        names,
        bound_array,
        initial_location,
        initial_cov,
        process_cov,
        observation_cov,
    ) = _validated_filter_inputs(
        observations,
        parameter_names,
        bounds,
        initial_mean,
        initial_covariance,
        process_covariance,
        observation_covariance,
    )
    ensemble_size = _positive_integer(n_ensemble, "n_ensemble")
    if ensemble_size <= len(names) + 1:
        raise NonlinearFilterError(
            "n_ensemble must exceed the parameter dimension plus one."
        )
    if not isinstance(rng, np.random.Generator):
        raise NonlinearFilterError("rng must be a numpy.random.Generator.")

    n_windows = observed.shape[0]
    n_parameters = len(names)
    n_observations = observed.shape[1]
    ensemble = reflect_parameters(
        rng.multivariate_normal(initial_location, initial_cov, size=ensemble_size),
        bound_array,
    )
    prior_means = np.empty((n_windows, n_parameters), dtype=np.float64)
    prior_covariances = np.empty(
        (n_windows, n_parameters, n_parameters),
        dtype=np.float64,
    )
    posterior_means = np.empty_like(prior_means)
    posterior_covariances = np.empty_like(prior_covariances)
    process_cholesky = np.linalg.cholesky(
        _stabilized_covariance(process_cov)
    )
    observation_cholesky = np.linalg.cholesky(observation_cov)

    for window_index in range(n_windows):
        process_draws = rng.standard_normal((ensemble_size, n_parameters))
        ensemble = reflect_parameters(
            ensemble + process_draws @ process_cholesky.T,
            bound_array,
        )
        prior_mean = np.mean(ensemble, axis=0)
        prior_covariance = _stabilized_covariance(
            np.cov(ensemble, rowvar=False, ddof=1)
        )
        prior_means[window_index] = prior_mean
        prior_covariances[window_index] = prior_covariance
        predicted_observations = np.stack(
            [measurement_function(member) for member in ensemble]
        )
        if predicted_observations.shape != (ensemble_size, n_observations):
            raise NonlinearFilterError(
                "measurement_function returned inconsistent dimensions."
            )
        state_anomalies = ensemble - prior_mean
        observation_mean = np.mean(predicted_observations, axis=0)
        observation_anomalies = predicted_observations - observation_mean
        normalization = float(ensemble_size - 1)
        cross_covariance = (
            state_anomalies.T @ observation_anomalies / normalization
        )
        innovation_covariance = (
            observation_anomalies.T @ observation_anomalies / normalization
            + observation_cov
        )
        innovation_covariance = 0.5 * (
            innovation_covariance + innovation_covariance.T
        )
        factorized = cho_factor(innovation_covariance, check_finite=False)
        gain = cho_solve(
            factorized,
            cross_covariance.T,
            check_finite=False,
        ).T
        observation_perturbations = (
            rng.standard_normal((ensemble_size, n_observations))
            @ observation_cholesky.T
        )
        innovations = (
            observed[window_index]
            + observation_perturbations
            - predicted_observations
        )
        ensemble = reflect_parameters(
            ensemble + innovations @ gain.T,
            bound_array,
        )
        posterior_means[window_index] = np.mean(ensemble, axis=0)
        posterior_covariances[window_index] = _stabilized_covariance(
            np.cov(ensemble, rowvar=False, ddof=1)
        )

    lower, upper = _intervals(
        posterior_means,
        posterior_covariances,
        bound_array,
        credible_level,
    )
    return SequentialHistory(
        parameter_names=names,
        parameter_bounds=bound_array,
        prior_means=prior_means,
        prior_covariances=prior_covariances,
        posterior_means=posterior_means,
        posterior_covariances=posterior_covariances,
        credible_interval_lower=lower,
        credible_interval_upper=upper,
        optimizer_success=np.ones(n_windows, dtype=np.bool_),
        n_function_evaluations=np.full(
            n_windows,
            ensemble_size,
            dtype=np.int64,
        ),
    )
