"""Transparent stochastic excitation models for virtual monitoring data."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class ExcitationError(ValueError):
    """Raised when a stochastic excitation specification is invalid."""


def _time_grid(time_s: ArrayLike) -> FloatArray:
    """Validate and copy a strictly increasing time grid."""

    time = np.asarray(time_s, dtype=np.float64)
    if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
        raise ExcitationError("time_s must be a finite one-dimensional grid.")
    if np.any(np.diff(time) <= 0.0):
        raise ExcitationError("time_s must be strictly increasing.")
    return time.copy()


def generate_ornstein_uhlenbeck_force(
    time_s: ArrayLike,
    *,
    stationary_standard_deviation_n: float,
    correlation_time_s: float,
    rng: np.random.Generator,
    mean_n: float = 0.0,
) -> FloatArray:
    """Generate a stationary Ornstein-Uhlenbeck force using exact transitions.

    For an interval ``dt``, the discrete transition is

    ``x[k+1] = mean + rho * (x[k] - mean) + sigma*sqrt(1-rho**2)*epsilon``

    with ``rho = exp(-dt/tau)``. Drawing the initial sample from the stationary
    distribution avoids a stochastic-process burn-in transient.
    """

    time = _time_grid(time_s)
    standard_deviation = float(stationary_standard_deviation_n)
    correlation_time = float(correlation_time_s)
    mean = float(mean_n)
    if not isinstance(rng, np.random.Generator):
        raise ExcitationError("rng must be a numpy.random.Generator.")
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ExcitationError(
            "stationary_standard_deviation_n must be finite and positive."
        )
    if not np.isfinite(correlation_time) or correlation_time <= 0.0:
        raise ExcitationError("correlation_time_s must be finite and positive.")
    if not np.isfinite(mean):
        raise ExcitationError("mean_n must be finite.")

    force = np.empty(time.size, dtype=np.float64)
    force[0] = mean + standard_deviation * rng.normal()
    for index, time_increment in enumerate(np.diff(time), start=1):
        correlation = np.exp(-time_increment / correlation_time)
        innovation_scale = standard_deviation * np.sqrt(1.0 - correlation**2)
        force[index] = (
            mean
            + correlation * (force[index - 1] - mean)
            + innovation_scale * rng.normal()
        )
    return force


def distribute_scalar_force(
    scalar_force_n: ArrayLike,
    spatial_distribution: ArrayLike,
) -> FloatArray:
    """Map one scalar excitation to physical DOF forces in newtons."""

    scalar_force = np.asarray(scalar_force_n, dtype=np.float64)
    distribution = np.asarray(spatial_distribution, dtype=np.float64)
    if scalar_force.ndim != 1 or scalar_force.size < 2:
        raise ExcitationError("scalar_force_n must be a one-dimensional history.")
    if distribution.ndim != 1 or distribution.size < 1:
        raise ExcitationError("spatial_distribution must be one-dimensional.")
    if not np.all(np.isfinite(scalar_force)) or not np.all(np.isfinite(distribution)):
        raise ExcitationError("Force and spatial distribution must be finite.")
    if np.all(distribution == 0.0):
        raise ExcitationError("spatial_distribution must apply a nonzero force.")
    return np.outer(scalar_force, distribution)
