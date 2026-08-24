"""Post-hoc uncertainty calibration that preserves posterior means."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]

UNCERTAINTY_CALIBRATION_METHODS = (
    "raw",
    "scalar_inflation",
    "zonewise_inflation",
    "additive_model_error",
)


class UncertaintyCalibrationError(ValueError):
    """Raised when interval-calibration inputs are invalid."""


@dataclass(frozen=True)
class UncertaintyCalibration:
    """Frozen scale and additive corrections learned from calibration truth."""

    nominal_coverage: float
    gaussian_quantile: float
    scalar_inflation: float
    zonewise_inflation: FloatArray
    additive_model_error_standard_deviation: FloatArray


def _validated_arrays(
    posterior_means: ArrayLike,
    posterior_standard_deviations: ArrayLike,
    truth: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    means = np.asarray(posterior_means, dtype=np.float64)
    scales = np.asarray(posterior_standard_deviations, dtype=np.float64)
    true_values = np.asarray(truth, dtype=np.float64)
    if (
        means.ndim < 2
        or means.shape != scales.shape
        or means.shape != true_values.shape
        or means.shape[-1] < 1
    ):
        raise UncertaintyCalibrationError(
            "Means, standard deviations, and truth must share shape (..., zone)."
        )
    if (
        not np.all(np.isfinite(means))
        or not np.all(np.isfinite(scales))
        or not np.all(np.isfinite(true_values))
        or np.any(scales <= 0.0)
    ):
        raise UncertaintyCalibrationError(
            "Calibration arrays must be finite with positive standard deviations."
        )
    return means, scales, true_values


def _higher_quantile(values: FloatArray, probability: float) -> float:
    """Return a conservative empirical quantile without interpolation."""

    return float(np.quantile(values, probability, method="higher"))


def fit_uncertainty_calibration(
    posterior_means: ArrayLike,
    posterior_standard_deviations: ArrayLike,
    truth: ArrayLike,
    *,
    nominal_coverage: float = 0.95,
) -> UncertaintyCalibration:
    """Fit scalar, zonewise, and additive interval corrections.

    Corrections use higher empirical quantiles of absolute calibration errors.
    They affect posterior standard deviations only and never posterior means.
    """

    means, scales, true_values = _validated_arrays(
        posterior_means,
        posterior_standard_deviations,
        truth,
    )
    coverage = float(nominal_coverage)
    if not np.isfinite(coverage) or not 0.5 < coverage < 1.0:
        raise UncertaintyCalibrationError(
            "nominal_coverage must be finite and strictly between 0.5 and 1."
        )
    gaussian_quantile = float(norm.ppf(0.5 + 0.5 * coverage))
    absolute_error = np.abs(means - true_values)
    ratios = absolute_error / (gaussian_quantile * scales)
    scalar = max(1.0, _higher_quantile(ratios.reshape(-1), coverage))
    zonewise = np.asarray(
        [
            max(1.0, _higher_quantile(ratios[..., zone].reshape(-1), coverage))
            for zone in range(means.shape[-1])
        ],
        dtype=np.float64,
    )
    required_variance = np.maximum(
        (absolute_error / gaussian_quantile) ** 2 - scales**2,
        0.0,
    )
    additive = np.asarray(
        [
            np.sqrt(
                _higher_quantile(
                    required_variance[..., zone].reshape(-1),
                    coverage,
                )
            )
            for zone in range(means.shape[-1])
        ],
        dtype=np.float64,
    )
    return UncertaintyCalibration(
        nominal_coverage=coverage,
        gaussian_quantile=gaussian_quantile,
        scalar_inflation=scalar,
        zonewise_inflation=zonewise,
        additive_model_error_standard_deviation=additive,
    )


def calibrated_standard_deviations(
    posterior_standard_deviations: ArrayLike,
    method: str,
    calibration: UncertaintyCalibration,
) -> FloatArray:
    """Apply one frozen calibration rule to posterior standard deviations."""

    scales = np.asarray(posterior_standard_deviations, dtype=np.float64)
    if (
        scales.ndim < 1
        or scales.shape[-1] != calibration.zonewise_inflation.size
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise UncertaintyCalibrationError(
            "posterior_standard_deviations must be positive and end in zone."
        )
    if method not in UNCERTAINTY_CALIBRATION_METHODS:
        raise UncertaintyCalibrationError(f"Unknown calibration method: {method}.")
    if method == "raw":
        return scales.copy()
    if method == "scalar_inflation":
        return scales * calibration.scalar_inflation
    if method == "zonewise_inflation":
        return scales * calibration.zonewise_inflation
    return np.sqrt(
        scales**2 + calibration.additive_model_error_standard_deviation**2
    )
