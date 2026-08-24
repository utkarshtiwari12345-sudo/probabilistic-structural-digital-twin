"""Recursive Laplace updating for slowly evolving structural parameters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from probdt.inference.batch import (
    BatchInferenceError,
    SpectralForwardModel,
    fit_batch_spectral_map,
)

FloatArray = NDArray[np.float64]


class SequentialInferenceError(ValueError):
    """Raised when a recursive inference input is invalid."""


@dataclass(frozen=True)
class SequentialHistory:
    """Prior predictions and posterior approximations at monitoring windows."""

    parameter_names: tuple[str, ...]
    parameter_bounds: FloatArray
    prior_means: FloatArray
    prior_covariances: FloatArray
    posterior_means: FloatArray
    posterior_covariances: FloatArray
    credible_interval_lower: FloatArray
    credible_interval_upper: FloatArray
    optimizer_success: NDArray[np.bool_]
    n_function_evaluations: NDArray[np.int64]
    update_performed: NDArray[np.bool_] | None = None

    @property
    def posterior_standard_deviations(self) -> FloatArray:
        """Return the marginal posterior standard deviation at every window."""

        diagonal = np.diagonal(
            self.posterior_covariances,
            axis1=1,
            axis2=2,
        )
        return np.sqrt(diagonal)

    def loss_exceedance_probability(self, loss_threshold: float) -> FloatArray:
        """Return bounded-normal ``P(1 - theta > threshold)`` by window and zone."""

        threshold = float(loss_threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise SequentialInferenceError(
                "loss_threshold must lie strictly within (0, 1)."
            )
        multiplier_threshold = 1.0 - threshold
        lower = self.parameter_bounds[:, 0][None, :]
        upper = self.parameter_bounds[:, 1][None, :]
        mean = self.posterior_means
        scale = self.posterior_standard_deviations
        lower_cdf = norm.cdf((lower - mean) / scale)
        upper_cdf = norm.cdf((upper - mean) / scale)
        threshold_cdf = norm.cdf((multiplier_threshold - mean) / scale)
        normalization = upper_cdf - lower_cdf
        probabilities = (threshold_cdf - lower_cdf) / normalization
        return np.clip(probabilities, 0.0, 1.0)


def _positive_definite_covariance(
    covariance: ArrayLike,
    n_parameters: int,
    name: str,
    *,
    allow_semidefinite: bool,
) -> FloatArray:
    """Return one checked symmetric covariance matrix."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (n_parameters, n_parameters):
        raise SequentialInferenceError(
            f"{name} must have shape ({n_parameters}, {n_parameters})."
        )
    if not np.all(np.isfinite(matrix)):
        raise SequentialInferenceError(f"{name} must be finite.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise SequentialInferenceError(f"{name} must be symmetric.")
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
    if allow_semidefinite:
        if minimum_eigenvalue < -1.0e-14:
            raise SequentialInferenceError(
                f"{name} must be positive semidefinite."
            )
    elif minimum_eigenvalue <= 0.0:
        raise SequentialInferenceError(f"{name} must be positive definite.")
    return matrix.copy()


def predict_random_walk_covariance(
    posterior_covariance: ArrayLike,
    process_covariance: ArrayLike,
) -> FloatArray:
    """Apply the random-walk prediction ``P[k|k-1] = P[k-1|k-1] + Q``."""

    posterior = np.asarray(posterior_covariance, dtype=np.float64)
    if posterior.ndim != 2 or posterior.shape[0] != posterior.shape[1]:
        raise SequentialInferenceError("posterior_covariance must be square.")
    n_parameters = posterior.shape[0]
    checked_posterior = _positive_definite_covariance(
        posterior,
        n_parameters,
        "posterior_covariance",
        allow_semidefinite=False,
    )
    process = _positive_definite_covariance(
        process_covariance,
        n_parameters,
        "process_covariance",
        allow_semidefinite=True,
    )
    predicted = checked_posterior + process
    return 0.5 * (predicted + predicted.T)


def run_recursive_spectral_map(
    forward_model: SpectralForwardModel,
    observed_psd_windows: ArrayLike,
    parameter_names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    *,
    initial_prior_mean: ArrayLike,
    initial_prior_covariance: ArrayLike,
    process_covariance: ArrayLike,
    n_independent_averages: int,
    fallback_initial_points: Sequence[ArrayLike] | None = None,
    credible_level: float = 0.95,
    derivative_relative_step: float = 1.0e-4,
    optimizer_max_iterations: int = 300,
) -> SequentialHistory:
    """Apply prediction and bounded MAP/Laplace updating to spectral windows."""

    observations = np.asarray(observed_psd_windows, dtype=np.float64)
    if observations.ndim != 3 or observations.shape[0] < 1:
        raise SequentialInferenceError(
            "observed_psd_windows must have shape (window, frequency, sensor)."
        )
    if not np.all(np.isfinite(observations)) or np.any(observations <= 0.0):
        raise SequentialInferenceError(
            "observed_psd_windows must be positive and finite."
        )
    names = tuple(parameter_names)
    n_parameters = len(names)
    if n_parameters < 1 or len(set(names)) != n_parameters:
        raise SequentialInferenceError(
            "parameter_names must be nonempty and unique."
        )
    bound_array = np.asarray(bounds, dtype=np.float64)
    if bound_array.shape != (n_parameters, 2):
        raise SequentialInferenceError("bounds must contain one pair per parameter.")
    lower = bound_array[:, 0]
    upper = bound_array[:, 1]
    if not np.all(np.isfinite(bound_array)) or np.any(lower >= upper):
        raise SequentialInferenceError("Parameter bounds are invalid.")

    initial_mean = np.asarray(initial_prior_mean, dtype=np.float64)
    if initial_mean.shape != (n_parameters,) or not np.all(np.isfinite(initial_mean)):
        raise SequentialInferenceError(
            "initial_prior_mean must match the parameter vector."
        )
    if np.any(initial_mean < lower) or np.any(initial_mean > upper):
        raise SequentialInferenceError("initial_prior_mean must lie within bounds.")
    initial_covariance = _positive_definite_covariance(
        initial_prior_covariance,
        n_parameters,
        "initial_prior_covariance",
        allow_semidefinite=False,
    )
    process = _positive_definite_covariance(
        process_covariance,
        n_parameters,
        "process_covariance",
        allow_semidefinite=True,
    )

    fallback = [] if fallback_initial_points is None else [
        np.asarray(point, dtype=np.float64) for point in fallback_initial_points
    ]
    if any(point.shape != (n_parameters,) for point in fallback):
        raise SequentialInferenceError(
            "Every fallback initial point must match the parameter vector."
        )
    if any(not np.all(np.isfinite(point)) for point in fallback):
        raise SequentialInferenceError("Fallback initial points must be finite.")

    n_windows = observations.shape[0]
    prior_means = np.empty((n_windows, n_parameters), dtype=np.float64)
    prior_covariances = np.empty(
        (n_windows, n_parameters, n_parameters),
        dtype=np.float64,
    )
    posterior_means = np.empty_like(prior_means)
    posterior_covariances = np.empty_like(prior_covariances)
    interval_lower = np.empty_like(prior_means)
    interval_upper = np.empty_like(prior_means)
    success = np.empty(n_windows, dtype=np.bool_)
    evaluations = np.empty(n_windows, dtype=np.int64)

    predicted_mean = initial_mean.copy()
    predicted_covariance = initial_covariance.copy()
    for window_index, observed_psd in enumerate(observations):
        prior_means[window_index] = predicted_mean
        prior_covariances[window_index] = predicted_covariance
        starts = [predicted_mean, *fallback]
        try:
            estimate = fit_batch_spectral_map(
                forward_model,
                observed_psd,
                names,
                bounds,
                prior_mean=predicted_mean,
                prior_standard_deviation=None,
                prior_covariance=predicted_covariance,
                n_independent_averages=n_independent_averages,
                initial_points=starts,
                credible_level=credible_level,
                derivative_relative_step=derivative_relative_step,
                optimizer_max_iterations=optimizer_max_iterations,
            )
        except BatchInferenceError as error:
            raise SequentialInferenceError(str(error)) from error

        posterior_means[window_index] = estimate.map_parameters
        posterior_covariances[window_index] = estimate.posterior_covariance
        interval_lower[window_index] = estimate.credible_interval_lower
        interval_upper[window_index] = estimate.credible_interval_upper
        success[window_index] = estimate.optimizer_success
        evaluations[window_index] = estimate.n_function_evaluations
        predicted_mean = estimate.map_parameters.copy()
        predicted_covariance = predict_random_walk_covariance(
            estimate.posterior_covariance,
            process,
        )

    return SequentialHistory(
        parameter_names=names,
        parameter_bounds=bound_array.copy(),
        prior_means=prior_means,
        prior_covariances=prior_covariances,
        posterior_means=posterior_means,
        posterior_covariances=posterior_covariances,
        credible_interval_lower=interval_lower,
        credible_interval_upper=interval_upper,
        optimizer_success=success,
        n_function_evaluations=evaluations,
        update_performed=np.ones(n_windows, dtype=np.bool_),
    )


def run_recursive_spectral_map_varying_observations(
    forward_models: Sequence[SpectralForwardModel],
    observed_psd_windows: Sequence[ArrayLike | None],
    parameter_names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    *,
    initial_prior_mean: ArrayLike,
    initial_prior_covariance: ArrayLike,
    process_covariance: ArrayLike,
    n_independent_averages: int,
    credible_level: float = 0.95,
    derivative_relative_step: float = 1.0e-4,
    optimizer_max_iterations: int = 300,
) -> SequentialHistory:
    """Recursively update with changing sensor sets or entirely missing windows.

    A forward model is supplied for every monitoring window because sensor
    dropout can change the observation dimension. A ``None`` observation skips
    the likelihood update but still propagates process uncertainty.
    """

    forwards = tuple(forward_models)
    observations = tuple(observed_psd_windows)
    if not forwards or len(forwards) != len(observations):
        raise SequentialInferenceError(
            "forward_models and observed_psd_windows must have equal nonzero length."
        )
    if any(not callable(forward) for forward in forwards):
        raise SequentialInferenceError("Every forward model must be callable.")
    names = tuple(parameter_names)
    n_parameters = len(names)
    if n_parameters < 1 or len(set(names)) != n_parameters:
        raise SequentialInferenceError(
            "parameter_names must be nonempty and unique."
        )
    bound_array = np.asarray(bounds, dtype=np.float64)
    if bound_array.shape != (n_parameters, 2):
        raise SequentialInferenceError("bounds must contain one pair per parameter.")
    lower = bound_array[:, 0]
    upper = bound_array[:, 1]
    if not np.all(np.isfinite(bound_array)) or np.any(lower >= upper):
        raise SequentialInferenceError("Parameter bounds are invalid.")
    initial_mean = np.asarray(initial_prior_mean, dtype=np.float64)
    if initial_mean.shape != (n_parameters,) or not np.all(np.isfinite(initial_mean)):
        raise SequentialInferenceError(
            "initial_prior_mean must match the parameter vector."
        )
    if np.any(initial_mean < lower) or np.any(initial_mean > upper):
        raise SequentialInferenceError("initial_prior_mean must lie within bounds.")
    initial_covariance = _positive_definite_covariance(
        initial_prior_covariance,
        n_parameters,
        "initial_prior_covariance",
        allow_semidefinite=False,
    )
    process = _positive_definite_covariance(
        process_covariance,
        n_parameters,
        "process_covariance",
        allow_semidefinite=True,
    )
    if not np.isfinite(credible_level) or not 0.0 < credible_level < 1.0:
        raise SequentialInferenceError(
            "credible_level must lie strictly within (0, 1)."
        )

    n_windows = len(observations)
    prior_means = np.empty((n_windows, n_parameters), dtype=np.float64)
    prior_covariances = np.empty(
        (n_windows, n_parameters, n_parameters),
        dtype=np.float64,
    )
    posterior_means = np.empty_like(prior_means)
    posterior_covariances = np.empty_like(prior_covariances)
    interval_lower = np.empty_like(prior_means)
    interval_upper = np.empty_like(prior_means)
    success = np.ones(n_windows, dtype=np.bool_)
    evaluations = np.zeros(n_windows, dtype=np.int64)
    update_performed = np.zeros(n_windows, dtype=np.bool_)
    normal_quantile = float(norm.ppf(0.5 + credible_level / 2.0))

    predicted_mean = initial_mean.copy()
    predicted_covariance = initial_covariance.copy()
    for window_index, (forward_model, observation) in enumerate(
        zip(forwards, observations, strict=True)
    ):
        prior_means[window_index] = predicted_mean
        prior_covariances[window_index] = predicted_covariance
        if observation is None:
            posterior_mean = predicted_mean.copy()
            posterior_covariance = predicted_covariance.copy()
            posterior_scale = np.sqrt(np.diag(posterior_covariance))
            lower_interval = np.maximum(
                lower,
                posterior_mean - normal_quantile * posterior_scale,
            )
            upper_interval = np.minimum(
                upper,
                posterior_mean + normal_quantile * posterior_scale,
            )
        else:
            try:
                estimate = fit_batch_spectral_map(
                    forward_model,
                    observation,
                    names,
                    bounds,
                    prior_mean=predicted_mean,
                    prior_standard_deviation=None,
                    prior_covariance=predicted_covariance,
                    n_independent_averages=n_independent_averages,
                    initial_points=[predicted_mean],
                    credible_level=credible_level,
                    derivative_relative_step=derivative_relative_step,
                    optimizer_max_iterations=optimizer_max_iterations,
                )
            except BatchInferenceError as error:
                raise SequentialInferenceError(str(error)) from error
            posterior_mean = estimate.map_parameters
            posterior_covariance = estimate.posterior_covariance
            lower_interval = estimate.credible_interval_lower
            upper_interval = estimate.credible_interval_upper
            success[window_index] = estimate.optimizer_success
            evaluations[window_index] = estimate.n_function_evaluations
            update_performed[window_index] = True
        posterior_means[window_index] = posterior_mean
        posterior_covariances[window_index] = posterior_covariance
        interval_lower[window_index] = lower_interval
        interval_upper[window_index] = upper_interval
        predicted_mean = posterior_mean.copy()
        predicted_covariance = predict_random_walk_covariance(
            posterior_covariance,
            process,
        )

    return SequentialHistory(
        parameter_names=names,
        parameter_bounds=bound_array.copy(),
        prior_means=prior_means,
        prior_covariances=prior_covariances,
        posterior_means=posterior_means,
        posterior_covariances=posterior_covariances,
        credible_interval_lower=interval_lower,
        credible_interval_upper=interval_upper,
        optimizer_success=success,
        n_function_evaluations=evaluations,
        update_performed=update_performed,
    )
