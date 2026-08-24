"""Sparse sensor layouts and clean response extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from probdt.structural import TimeResponse

FloatArray = NDArray[np.float64]


class SensorError(ValueError):
    """Raised when a sensor layout is invalid for a structural response."""


@dataclass(frozen=True)
class SensorLayout:
    """Named zero-based acceleration-sensor locations."""

    name: str
    dof_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SensorError("Sensor layout name must not be empty.")
        if not self.dof_indices:
            raise SensorError("A sensor layout must contain at least one sensor.")
        if len(set(self.dof_indices)) != len(self.dof_indices):
            raise SensorError("Sensor DOF indices must be unique.")
        for index in self.dof_indices:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise SensorError("Sensor DOF indices must be nonnegative integers.")

    @property
    def n_sensors(self) -> int:
        """Return the number of sensors in the layout."""

        return len(self.dof_indices)


def extract_clean_acceleration(
    response: TimeResponse,
    layout: SensorLayout,
) -> FloatArray:
    """Extract clean acceleration channels in the declared layout order."""

    acceleration = np.asarray(response.acceleration_m_per_s2, dtype=np.float64)
    if acceleration.ndim != 2 or not np.all(np.isfinite(acceleration)):
        raise SensorError("Response acceleration must be a finite 2D array.")
    if max(layout.dof_indices) >= acceleration.shape[1]:
        raise SensorError("Sensor layout references a DOF outside the response.")
    return acceleration[:, layout.dof_indices].copy()
