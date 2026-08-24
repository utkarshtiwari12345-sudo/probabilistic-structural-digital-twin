"""Measurement noise, sensor bias, drift, and explicit missing-data masks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probdt.monitoring.sensors import SensorLayout
from probdt.structural import TimeResponse

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class MeasurementError(ValueError):
    """Raised when a synthetic measurement specification is invalid."""


@dataclass(frozen=True)
class ContiguousOutage:
    """Inclusive outage interval for one zero-based sensor channel."""

    sensor_index: int
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.sensor_index, bool)
            or not isinstance(self.sensor_index, int)
            or self.sensor_index < 0
        ):
            raise MeasurementError("sensor_index must be a nonnegative integer.")
        if (
            not np.isfinite(self.start_s)
            or not np.isfinite(self.end_s)
            or self.start_s >= self.end_s
        ):
            raise MeasurementError("An outage must have finite start_s < end_s.")


@dataclass(frozen=True)
class AccelerationMeasurements:
    """Clean and imperfect acceleration channels for one monitoring window."""

    layout: SensorLayout
    time_s: FloatArray
    clean_m_per_s2: FloatArray
    observed_m_per_s2: FloatArray
    availability_mask: BoolArray
    noise_standard_deviation_m_per_s2: FloatArray
    bias_history_m_per_s2: FloatArray

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=np.float64)
        clean = np.asarray(self.clean_m_per_s2, dtype=np.float64)
        observed = np.asarray(self.observed_m_per_s2, dtype=np.float64)
        availability = np.asarray(self.availability_mask, dtype=bool)
        noise_standard_deviation = np.asarray(
            self.noise_standard_deviation_m_per_s2,
            dtype=np.float64,
        )
        bias = np.asarray(self.bias_history_m_per_s2, dtype=np.float64)
        expected_shape = (time.size, self.layout.n_sensors)
        if time.ndim != 1 or time.size < 2 or np.any(np.diff(time) <= 0.0):
            raise MeasurementError("time_s must be strictly increasing.")
        if clean.shape != expected_shape or not np.all(np.isfinite(clean)):
            raise MeasurementError("Clean measurements have invalid values or shape.")
        if observed.shape != expected_shape or availability.shape != expected_shape:
            raise MeasurementError("Observed data and mask must match clean data.")
        if bias.shape != expected_shape or not np.all(np.isfinite(bias)):
            raise MeasurementError("Bias history must be finite and match clean data.")
        if noise_standard_deviation.shape != (self.layout.n_sensors,):
            raise MeasurementError("Noise standard deviations must match sensors.")
        if np.any(noise_standard_deviation < 0.0) or not np.all(
            np.isfinite(noise_standard_deviation)
        ):
            raise MeasurementError("Noise standard deviations must be nonnegative.")
        if not np.all(np.isfinite(observed[availability])):
            raise MeasurementError("Available measurements must be finite.")
        if not np.all(np.isnan(observed[~availability])):
            raise MeasurementError("Unavailable measurements must be stored as NaN.")

    @property
    def available_fraction(self) -> FloatArray:
        """Return the available fraction for each sensor."""

        return np.mean(self.availability_mask, axis=0)


def generate_availability_mask(
    time_s: ArrayLike,
    n_sensors: int,
    *,
    random_dropout_probability: float,
    rng: np.random.Generator,
    contiguous_outages: Sequence[ContiguousOutage] = (),
) -> BoolArray:
    """Generate independent random dropout plus declared contiguous outages."""

    time = np.asarray(time_s, dtype=np.float64)
    probability = float(random_dropout_probability)
    if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
        raise MeasurementError("time_s must be a finite one-dimensional grid.")
    if np.any(np.diff(time) <= 0.0):
        raise MeasurementError("time_s must be strictly increasing.")
    if isinstance(n_sensors, bool) or not isinstance(n_sensors, int) or n_sensors < 1:
        raise MeasurementError("n_sensors must be a positive integer.")
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise MeasurementError("random_dropout_probability must lie within [0, 1].")
    if not isinstance(rng, np.random.Generator):
        raise MeasurementError("rng must be a numpy.random.Generator.")

    availability = rng.random((time.size, n_sensors)) >= probability
    for outage in contiguous_outages:
        if outage.sensor_index >= n_sensors:
            raise MeasurementError("Outage references a sensor outside the layout.")
        if outage.start_s < time[0] or outage.end_s > time[-1]:
            raise MeasurementError(
                "Outage interval lies outside the monitoring window."
            )
        interval = (time >= outage.start_s) & (time <= outage.end_s)
        availability[interval, outage.sensor_index] = False
    return availability


def create_acceleration_measurements(
    response: TimeResponse,
    layout: SensorLayout,
    *,
    rng: np.random.Generator,
    noise_standard_deviation_fraction: float,
    constant_bias_fraction: float = 0.0,
    end_of_window_drift_fraction: float = 0.0,
    availability_mask: ArrayLike | None = None,
) -> AccelerationMeasurements:
    """Add calibrated Gaussian noise, constant bias, drift, and missingness.

    Fractions are relative to the sample standard deviation of each clean sensor
    channel. Drift varies linearly from zero to its configured end-of-window
    fraction. Random noise is independent across times and sensors.
    """

    time = np.asarray(response.time_s, dtype=np.float64)
    acceleration = np.asarray(response.acceleration_m_per_s2, dtype=np.float64)
    if max(layout.dof_indices) >= acceleration.shape[1]:
        raise MeasurementError("Sensor layout references a DOF outside the response.")
    clean = acceleration[:, layout.dof_indices].copy()
    fractions = np.asarray(
        [
            noise_standard_deviation_fraction,
            constant_bias_fraction,
            end_of_window_drift_fraction,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0.0):
        raise MeasurementError("Noise, bias, and drift fractions must be nonnegative.")
    if not isinstance(rng, np.random.Generator):
        raise MeasurementError("rng must be a numpy.random.Generator.")

    signal_scale = np.std(clean, axis=0, ddof=1)
    if np.any(signal_scale <= 0.0) or not np.all(np.isfinite(signal_scale)):
        raise MeasurementError("Every clean sensor channel must have nonzero variance.")
    noise_standard_deviation = fractions[0] * signal_scale
    if fractions[0] == 0.0:
        noise = np.zeros_like(clean)
    else:
        noise = rng.normal(size=clean.shape) * noise_standard_deviation

    normalized_time = (time - time[0]) / (time[-1] - time[0])
    bias = signal_scale * (
        fractions[1] + fractions[2] * normalized_time[:, np.newaxis]
    )
    observed = clean + noise + bias

    if availability_mask is None:
        availability = np.ones(clean.shape, dtype=bool)
    else:
        availability = np.asarray(availability_mask, dtype=bool)
        if availability.shape != clean.shape:
            raise MeasurementError(
                "availability_mask must match the sensor data shape."
            )
    observed[~availability] = np.nan

    return AccelerationMeasurements(
        layout=layout,
        time_s=time.copy(),
        clean_m_per_s2=clean,
        observed_m_per_s2=observed,
        availability_mask=availability.copy(),
        noise_standard_deviation_m_per_s2=noise_standard_deviation,
        bias_history_m_per_s2=bias,
    )
