"""Batch spectral parameter inference for the Phase I digital twin.

The baseline estimator uses an averaged-periodogram Whittle likelihood, bounded
optimization, and an expected-Fisher (Laplace) covariance approximation.  It is
deliberately small and transparent so later sequential estimators can be judged
against a verified batch reference.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.stats import norm

FloatArray = NDArray[np.float64]
SpectralForwardModel = Callable[[FloatArray], FloatArray]


class BatchInferenceError(ValueError):
    """Raised when a batch-inference input is mathematically invalid."""


@dataclass(frozen=True)
class BatchEstimate:
    """Bounded MAP estimate and local Gaussian uncertainty diagnostics."""

    parameter_names: tuple[str, ...]
    map_parameters: FloatArray
    posterior_covariance: FloatArray
    posterior_standard_deviations: FloatArray
    credible_interval_lower: FloatArray
    credible_interval_upper: FloatArray
    objective_value: float
    optimizer_success: bool
    optimizer_message: str
    n_function_evaluations: int
    active_lower_bounds: NDArray[np.bool_]
    active_upper_bounds: NDArray[np.bool_]
    fisher_condition_number: float

    @property
    def posterior_correlation(self) -> FloatArray:
        """Return the covariance converted to a parameter-correlation matrix."""

        scale = np.outer(
            self.posterior_standard_deviations,
            self.posterior_standard_deviations,
        )
        return self.posterior_covariance / scale


def _positive_spectral_array(values: ArrayLike, name: str) -> FloatArray:
    """Return one checked positive two-dimensional spectral array."""

    spectral = np.asarray(values, dtype=np.float64)
    if spectral.ndim != 2 or spectral.size == 0:
        raise BatchInferenceError(f"{name} must be a nonempty two-dimensional array.")
    if not np.all(np.isfinite(spectral)) or np.any(spectral <= 0.0):
        raise BatchInferenceError(f"{name} must be positive and finite.")
    return spectral


def simulate_averaged_periodogram(
    expected_psd: ArrayLike,
    n_independent_averages: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw an averaged periodogram with the requested Gamma degrees of freedom.

    If ``K`` independent raw periodograms are averaged, every normalized
    ordinate follows ``Gamma(shape=K, scale=1/K)`` under the Whittle model.
    """

    expected = _positive_spectral_array(expected_psd, "expected_psd")
    if (
        isinstance(n_independent_averages, bool)
        or not isinstance(n_independent_averages, int)
        or n_independent_averages < 1
    ):
        raise BatchInferenceError("n_independent_averages must be a positive integer.")
    if not isinstance(rng, np.random.Generator):
        raise BatchInferenceError("rng must be a numpy.random.Generator.")

    normalized_ordinates = rng.gamma(
        shape=float(n_independent_averages),
        scale=1.0 / float(n_independent_averages),
        size=expected.shape,
    )
    return expected * normalized_ordinates


def averaged_periodogram_negative_log_likelihood(
    observed_psd: ArrayLike,
    model_psd: ArrayLike,
    n_independent_averages: int,
) -> float:
    """Return the parameter-dependent averaged-periodogram Whittle objective."""

    observed = _positive_spectral_array(observed_psd, "observed_psd")
    model = _positive_spectral_array(model_psd, "model_psd")
    if observed.shape != model.shape:
        raise BatchInferenceError(
            "Observed and model PSD arrays must have equal shape."
        )
    if (
        isinstance(n_independent_averages, bool)
        or not isinstance(n_independent_averages, int)
        or n_independent_averages < 1
    ):
        raise BatchInferenceError("n_independent_averages must be a positive integer.")

    objective = n_independent_averages * np.sum(np.log(model) + observed / model)
    return float(objective)


def log_spectral_jacobian(
    forward_model: SpectralForwardModel,
    parameters: ArrayLike,
    bounds: Sequence[tuple[float, float]],
    *,
    relative_step: float = 1.0e-4,
) -> FloatArray:
    """Differentiate flattened log spectra while respecting parameter bounds."""

    point = np.asarray(parameters, dtype=np.float64)
    lower, upper = _validated_bounds(bounds, point.size)
    step_fraction = float(relative_step)
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise BatchInferenceError("parameters must be a finite vector.")
    if np.any(point < lower) or np.any(point > upper):
        raise BatchInferenceError("parameters must lie within bounds.")
    if not np.isfinite(step_fraction) or step_fraction <= 0.0:
        raise BatchInferenceError("relative_step must be positive and finite.")

    baseline = np.log(_positive_spectral_array(forward_model(point), "model PSD"))
    baseline_features = baseline.reshape(-1)
    jacobian = np.empty((baseline_features.size, point.size), dtype=np.float64)

    for index in range(point.size):
        nominal_step = step_fraction * max(abs(point[index]), 1.0)
        forward_room = upper[index] - point[index]
        backward_room = point[index] - lower[index]
        if forward_room >= nominal_step and backward_room >= nominal_step:
            plus = point.copy()
            minus = point.copy()
            plus[index] += nominal_step
            minus[index] -= nominal_step
            plus_features = np.log(
                _positive_spectral_array(forward_model(plus), "model PSD")
            ).reshape(-1)
            minus_features = np.log(
                _positive_spectral_array(forward_model(minus), "model PSD")
            ).reshape(-1)
            jacobian[:, index] = (
                plus_features - minus_features
            ) / (2.0 * nominal_step)
        elif backward_room >= nominal_step:
            minus = point.copy()
            minus[index] -= nominal_step
            minus_features = np.log(
                _positive_spectral_array(forward_model(minus), "model PSD")
            ).reshape(-1)
            jacobian[:, index] = (
                baseline_features - minus_features
            ) / nominal_step
        elif forward_room >= nominal_step:
            plus = point.copy()
            plus[index] += nominal_step
            plus_features = np.log(
                _positive_spectral_array(forward_model(plus), "model PSD")
            ).reshape(-1)
            jacobian[:, index] = (
                plus_features - baseline_features
            ) / nominal_step
        else:
            raise BatchInferenceError(
                "A parameter bound is too narrow for the derivative step."
            )

    return jacobian


def _validated_bounds(
    bounds: Sequence[tuple[float, float]],
    n_parameters: int,
) -> tuple[FloatArray, FloatArray]:
    """Return checked lower and upper bound vectors."""

    if len(bounds) != n_parameters:
        raise BatchInferenceError("bounds must contain one pair per parameter.")
    bound_array = np.asarray(bounds, dtype=np.float64)
    if bound_array.shape != (n_parameters, 2):
        raise BatchInferenceError("Every bound must contain a lower and upper value.")
    if not np.all(np.isfinite(bound_array)):
        raise BatchInferenceError("Parameter bounds must be finite.")
    lower = bound_array[:, 0]
    upper = bound_array[:, 1]
    if np.any(lower >= upper):
        raise BatchInferenceError("Every lower bound must be below its upper bound.")
    return lower, upper


def _validated_prior_precision(
    prior_standard_deviation: ArrayLike | None,
    prior_covariance: ArrayLike | None,
    n_parameters: int,
) -> FloatArray:
    """Return a positive-definite prior precision from one prior description."""

    if (prior_standard_deviation is None) == (prior_covariance is None):
        raise BatchInferenceError(
            "Provide exactly one of prior_standard_deviation or prior_covariance."
        )
    if prior_covariance is None:
        prior_scale = np.asarray(prior_standard_deviation, dtype=np.float64)
        if prior_scale.shape != (n_parameters,) or not np.all(
            np.isfinite(prior_scale)
        ):
            raise BatchInferenceError(
                "prior_standard_deviation must match the parameter vector."
            )
        if np.any(prior_scale <= 0.0):
            raise BatchInferenceError("Prior standard deviations must be positive.")
        return np.diag(1.0 / prior_scale**2)

    covariance = np.asarray(prior_covariance, dtype=np.float64)
    if covariance.shape != (n_parameters, n_parameters):
        raise BatchInferenceError(
            "prior_covariance must be square and match the parameter vector."
        )
    if not np.all(np.isfinite(covariance)):
        raise BatchInferenceError("prior_covariance must be finite.")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12):
        raise BatchInferenceError("prior_covariance must be symmetric.")
    if np.linalg.eigvalsh(covariance)[0] <= 0.0:
        raise BatchInferenceError("prior_covariance must be positive definite.")
    return np.linalg.inv(covariance)


def fit_batch_spectral_map(
    forward_model: SpectralForwardModel,
    observed_psd: ArrayLike,
    parameter_names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    *,
    prior_mean: ArrayLike,
    prior_standard_deviation: ArrayLike | None,
    n_independent_averages: int,
    prior_covariance: ArrayLike | None = None,
    initial_points: Sequence[ArrayLike] | None = None,
    credible_level: float = 0.95,
    derivative_relative_step: float = 1.0e-4,
    optimizer_max_iterations: int = 300,
) -> BatchEstimate:
    """Fit bounded spectral parameters and calculate local posterior uncertainty."""

    observed = _positive_spectral_array(observed_psd, "observed_psd")
    names = tuple(parameter_names)
    n_parameters = len(names)
    if n_parameters < 1 or len(set(names)) != n_parameters:
        raise BatchInferenceError("parameter_names must be nonempty and unique.")
    if any(not name.strip() for name in names):
        raise BatchInferenceError("parameter_names cannot contain empty names.")
    lower, upper = _validated_bounds(bounds, n_parameters)

    prior_location = np.asarray(prior_mean, dtype=np.float64)
    if prior_location.shape != (n_parameters,) or not np.all(
        np.isfinite(prior_location)
    ):
        raise BatchInferenceError("prior_mean must match the parameter vector.")
    prior_precision = _validated_prior_precision(
        prior_standard_deviation,
        prior_covariance,
        n_parameters,
    )
    if not np.isfinite(credible_level) or not 0.0 < credible_level < 1.0:
        raise BatchInferenceError("credible_level must lie strictly within (0, 1).")
    if (
        isinstance(optimizer_max_iterations, bool)
        or not isinstance(optimizer_max_iterations, int)
        or optimizer_max_iterations < 1
    ):
        raise BatchInferenceError("optimizer_max_iterations must be positive.")

    if initial_points is None:
        starts = [np.clip(prior_location, lower, upper)]
    else:
        starts = [np.asarray(start, dtype=np.float64) for start in initial_points]
        if not starts:
            raise BatchInferenceError("initial_points cannot be empty.")
        if any(start.shape != (n_parameters,) for start in starts):
            raise BatchInferenceError("Every initial point must match the parameters.")
        if any(not np.all(np.isfinite(start)) for start in starts):
            raise BatchInferenceError("Initial points must be finite.")
        starts = [np.clip(start, lower, upper) for start in starts]

    def objective(parameters: FloatArray) -> float:
        model = forward_model(parameters)
        likelihood = averaged_periodogram_negative_log_likelihood(
            observed,
            model,
            n_independent_averages,
        )
        prior_difference = parameters - prior_location
        prior_penalty = 0.5 * float(
            prior_difference @ prior_precision @ prior_difference
        )
        return likelihood + prior_penalty

    results = []
    for start in starts:
        primary_result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper, strict=True)),
            options={"maxiter": optimizer_max_iterations, "ftol": 1.0e-12},
        )
        results.append(primary_result)
        if not primary_result.success:
            results.append(
                minimize(
                    objective,
                    primary_result.x,
                    method="Powell",
                    bounds=list(zip(lower, upper, strict=True)),
                    options={
                        "maxiter": optimizer_max_iterations,
                        "ftol": 1.0e-10,
                        "xtol": 1.0e-8,
                    },
                )
            )
    finite_results = [result for result in results if np.isfinite(result.fun)]
    if not finite_results:
        raise BatchInferenceError(
            "Every optimization start returned a nonfinite value."
        )
    successful_results = [result for result in finite_results if result.success]
    eligible_results = successful_results or finite_results
    result = min(eligible_results, key=lambda candidate: float(candidate.fun))
    map_parameters = np.asarray(result.x, dtype=np.float64)

    jacobian = log_spectral_jacobian(
        forward_model,
        map_parameters,
        bounds,
        relative_step=derivative_relative_step,
    )
    precision = (
        n_independent_averages * (jacobian.T @ jacobian)
        + prior_precision
    )
    precision = 0.5 * (precision + precision.T)
    precision_eigenvalues = np.linalg.eigvalsh(precision)
    if precision_eigenvalues[0] <= 0.0:
        raise BatchInferenceError("Posterior precision is not positive definite.")
    covariance = np.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)
    standard_deviations = np.sqrt(np.diag(covariance))

    normal_quantile = float(norm.ppf(0.5 + credible_level / 2.0))
    interval_lower = np.maximum(
        lower,
        map_parameters - normal_quantile * standard_deviations,
    )
    interval_upper = np.minimum(
        upper,
        map_parameters + normal_quantile * standard_deviations,
    )
    bound_tolerance = 1.0e-7 * np.maximum(1.0, upper - lower)

    return BatchEstimate(
        parameter_names=names,
        map_parameters=map_parameters,
        posterior_covariance=covariance,
        posterior_standard_deviations=standard_deviations,
        credible_interval_lower=interval_lower,
        credible_interval_upper=interval_upper,
        objective_value=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        n_function_evaluations=int(result.nfev),
        active_lower_bounds=np.abs(map_parameters - lower) <= bound_tolerance,
        active_upper_bounds=np.abs(map_parameters - upper) <= bound_tolerance,
        fisher_condition_number=float(np.linalg.cond(precision)),
    )
