"""Bounded deterioration histories for controlled digital-twin experiments.

Loss fraction ``d`` and stiffness multiplier ``theta`` are related by

``theta = 1 - d``.

All implemented processes are non-decreasing in loss, so stiffness cannot heal
silently. Independent mechanisms acting in the same zone are combined
multiplicatively in retained stiffness rather than by unbounded addition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]


class DeteriorationError(ValueError):
    """Raised when a deterioration history is invalid or nonphysical."""


@dataclass(frozen=True)
class ConditionThresholds:
    """Descriptive boundaries for known ground-truth severity."""

    healthy_max_loss_fraction: float = 0.01
    early_max_loss_fraction: float = 0.05
    moderate_max_loss_fraction: float = 0.12

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.healthy_max_loss_fraction,
                self.early_max_loss_fraction,
                self.moderate_max_loss_fraction,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise DeteriorationError("Condition thresholds must be finite.")
        if not 0.0 <= values[0] < values[1] < values[2] < 1.0:
            raise DeteriorationError(
                "Condition thresholds must be strictly ordered within [0, 1)."
            )


@dataclass(frozen=True)
class DeteriorationHistory:
    """Known zonal stiffness multipliers at successive monitoring times."""

    scenario_name: str
    time_s: FloatArray
    zone_names: tuple[str, ...]
    stiffness_multipliers: FloatArray

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=np.float64)
        multipliers = np.asarray(self.stiffness_multipliers, dtype=np.float64)
        if not self.scenario_name.strip():
            raise DeteriorationError("scenario_name must not be empty.")
        if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
            raise DeteriorationError("time_s must be a finite one-dimensional grid.")
        if np.any(np.diff(time) <= 0.0):
            raise DeteriorationError("time_s must be strictly increasing.")
        if not self.zone_names or len(set(self.zone_names)) != len(self.zone_names):
            raise DeteriorationError("zone_names must be nonempty and unique.")
        if multipliers.shape != (time.size, len(self.zone_names)):
            raise DeteriorationError(
                "stiffness_multipliers must have shape (n_time, n_zone)."
            )
        if not np.all(np.isfinite(multipliers)):
            raise DeteriorationError("stiffness_multipliers must be finite.")
        if np.any(multipliers <= 0.0) or np.any(multipliers > 1.0):
            raise DeteriorationError(
                "Every stiffness multiplier must satisfy 0 < theta <= 1."
            )

    @property
    def loss_fractions(self) -> FloatArray:
        """Return ``1 - theta`` for every time and zone."""

        return 1.0 - self.stiffness_multipliers


@dataclass(frozen=True)
class DeteriorationValidationReport:
    """Auditable bounds and monotonicity evidence for one truth history."""

    n_time: int
    n_zones: int
    minimum_multiplier: float
    maximum_multiplier: float
    maximum_multiplier_increase: float
    initially_healthy: bool
    monotone_nonincreasing: bool


def _time_grid(time_s: ArrayLike) -> FloatArray:
    """Validate and copy a strictly increasing time grid."""

    time = np.asarray(time_s, dtype=np.float64)
    if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
        raise DeteriorationError("time_s must be a finite one-dimensional grid.")
    if np.any(np.diff(time) <= 0.0):
        raise DeteriorationError("time_s must be strictly increasing.")
    return time.copy()


def _loss_fraction(value: float, *, name: str, allow_zero: bool = True) -> float:
    """Validate a scalar loss fraction."""

    loss = float(value)
    lower_valid = loss >= 0.0 if allow_zero else loss > 0.0
    if not np.isfinite(loss) or not lower_valid or loss >= 1.0:
        lower_symbol = "0 <=" if allow_zero else "0 <"
        raise DeteriorationError(f"{name} must satisfy {lower_symbol} loss < 1.")
    return loss


def smooth_gradual_loss(
    time_s: ArrayLike,
    *,
    onset_s: float,
    completion_s: float,
    final_loss_fraction: float,
) -> FloatArray:
    """Return a monotone cubic smoothstep deterioration trajectory.

    Loss and its slope are continuous at onset and completion. The function is a
    controlled benchmark, not a fatigue crack-growth law.
    """

    time = _time_grid(time_s)
    onset = float(onset_s)
    completion = float(completion_s)
    final_loss = _loss_fraction(
        final_loss_fraction,
        name="final_loss_fraction",
        allow_zero=False,
    )
    if not np.isfinite(onset) or not np.isfinite(completion) or onset >= completion:
        raise DeteriorationError("onset_s must be finite and precede completion_s.")
    if onset < time[0] or completion > time[-1]:
        raise DeteriorationError(
            "Gradual deterioration must begin and complete within the time grid."
        )

    normalized_time = np.clip((time - onset) / (completion - onset), 0.0, 1.0)
    smoothstep = normalized_time**2 * (3.0 - 2.0 * normalized_time)
    return final_loss * smoothstep


def abrupt_loss(
    time_s: ArrayLike,
    *,
    event_time_s: float,
    loss_fraction: float,
) -> FloatArray:
    """Return a permanent step loss beginning at the first sample at the event."""

    time = _time_grid(time_s)
    event_time = float(event_time_s)
    loss = _loss_fraction(
        loss_fraction,
        name="loss_fraction",
        allow_zero=False,
    )
    if not np.isfinite(event_time) or not time[0] < event_time <= time[-1]:
        raise DeteriorationError(
            "event_time_s must occur after the initial sample and within the grid."
        )
    return np.where(time >= event_time, loss, 0.0)


def gamma_process_loss(
    time_s: ArrayLike,
    *,
    onset_s: float,
    mean_final_loss_fraction: float,
    coefficient_of_variation: float,
    maximum_loss_fraction: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw one monotone gamma-process deterioration realization.

    Before explicit saturation, final loss has the configured mean and
    coefficient of variation. Independent gamma increments make the path
    monotone. ``maximum_loss_fraction`` is an explicit physical/numerical cap;
    configured studies should report whether that cap was reached.
    """

    time = _time_grid(time_s)
    onset = float(onset_s)
    mean_final_loss = _loss_fraction(
        mean_final_loss_fraction,
        name="mean_final_loss_fraction",
        allow_zero=False,
    )
    maximum_loss = _loss_fraction(
        maximum_loss_fraction,
        name="maximum_loss_fraction",
        allow_zero=False,
    )
    coefficient = float(coefficient_of_variation)
    if not isinstance(rng, np.random.Generator):
        raise DeteriorationError("rng must be a numpy.random.Generator.")
    if not np.isfinite(onset) or not time[0] <= onset < time[-1]:
        raise DeteriorationError("onset_s must lie within the active time range.")
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise DeteriorationError(
            "coefficient_of_variation must be finite and positive."
        )
    if maximum_loss <= mean_final_loss:
        raise DeteriorationError(
            "maximum_loss_fraction must exceed mean_final_loss_fraction."
        )

    active_duration = time[-1] - onset
    total_shape = 1.0 / coefficient**2
    scale = mean_final_loss / total_shape
    increments = np.zeros(time.size, dtype=np.float64)

    for index in range(1, time.size):
        active_start = max(time[index - 1], onset)
        active_end = max(time[index], onset)
        active_increment = active_end - active_start
        if active_increment > 0.0:
            increment_shape = total_shape * active_increment / active_duration
            increments[index] = rng.gamma(shape=increment_shape, scale=scale)

    return np.minimum(np.cumsum(increments), maximum_loss)


def combine_independent_losses(loss_components: Sequence[ArrayLike]) -> FloatArray:
    """Combine losses as ``1 - product(1 - d_component)``.

    This prevents two mechanisms from producing an unbounded additive loss and
    corresponds to multiplying their retained-stiffness fractions.
    """

    if not loss_components:
        raise DeteriorationError("At least one loss component is required.")
    components = [
        np.asarray(component, dtype=np.float64) for component in loss_components
    ]
    reference_shape = components[0].shape
    if len(reference_shape) != 1 or reference_shape[0] < 2:
        raise DeteriorationError("Loss components must be one-dimensional histories.")
    for component in components:
        if component.shape != reference_shape:
            raise DeteriorationError("All loss components must have identical shape.")
        if not np.all(np.isfinite(component)):
            raise DeteriorationError("Loss components must be finite.")
        if np.any(component < 0.0) or np.any(component >= 1.0):
            raise DeteriorationError("Loss components must satisfy 0 <= loss < 1.")

    retained_fraction = np.prod(1.0 - np.stack(components), axis=0)
    return 1.0 - retained_fraction


def build_deterioration_history(
    scenario_name: str,
    time_s: ArrayLike,
    zone_names: Sequence[str],
    zone_loss_histories: Mapping[str, ArrayLike],
) -> DeteriorationHistory:
    """Construct a complete history, assigning zero loss to unspecified zones."""

    time = _time_grid(time_s)
    names = tuple(str(name) for name in zone_names)
    if not names or any(not name.strip() for name in names):
        raise DeteriorationError("zone_names must contain nonempty names.")
    if len(set(names)) != len(names):
        raise DeteriorationError("zone_names must be unique.")
    unknown_zones = set(zone_loss_histories) - set(names)
    if unknown_zones:
        raise DeteriorationError(
            f"Loss histories reference unknown zones: {sorted(unknown_zones)}"
        )

    losses = np.zeros((time.size, len(names)), dtype=np.float64)
    for zone_index, zone_name in enumerate(names):
        if zone_name not in zone_loss_histories:
            continue
        zone_loss = np.asarray(zone_loss_histories[zone_name], dtype=np.float64)
        if zone_loss.shape != (time.size,):
            raise DeteriorationError(
                f"Loss history for {zone_name!r} must have shape ({time.size},)."
            )
        if not np.all(np.isfinite(zone_loss)):
            raise DeteriorationError("Zone loss histories must be finite.")
        if np.any(zone_loss < 0.0) or np.any(zone_loss >= 1.0):
            raise DeteriorationError("Zone losses must satisfy 0 <= loss < 1.")
        losses[:, zone_index] = zone_loss

    return DeteriorationHistory(
        scenario_name=scenario_name,
        time_s=time,
        zone_names=names,
        stiffness_multipliers=1.0 - losses,
    )


def classify_condition(
    loss_fractions: ArrayLike,
    thresholds: ConditionThresholds,
) -> StringArray:
    """Classify known truth as healthy, early, moderate, or significant."""

    losses = np.asarray(loss_fractions, dtype=np.float64)
    if not np.all(np.isfinite(losses)) or np.any(losses < 0.0) or np.any(losses >= 1.0):
        raise DeteriorationError("loss_fractions must satisfy 0 <= loss < 1.")

    stages = np.full(losses.shape, "significant", dtype="<U11")
    stages[losses <= thresholds.moderate_max_loss_fraction] = "moderate"
    stages[losses <= thresholds.early_max_loss_fraction] = "early"
    stages[losses <= thresholds.healthy_max_loss_fraction] = "healthy"
    return stages


def validate_deterioration_history(
    history: DeteriorationHistory,
    *,
    bound_tolerance: float = 1.0e-12,
    monotonicity_tolerance: float = 1.0e-12,
) -> DeteriorationValidationReport:
    """Check bounds, healthy initialization, and absence of silent healing."""

    if bound_tolerance < 0.0 or monotonicity_tolerance < 0.0:
        raise DeteriorationError("Validation tolerances must be nonnegative.")
    multipliers = history.stiffness_multipliers
    increments = np.diff(multipliers, axis=0)
    maximum_increase = float(max(np.max(increments), 0.0))
    initially_healthy = bool(
        np.allclose(
            multipliers[0],
            1.0,
            rtol=0.0,
            atol=bound_tolerance,
        )
    )
    monotone = maximum_increase <= monotonicity_tolerance

    return DeteriorationValidationReport(
        n_time=history.time_s.size,
        n_zones=len(history.zone_names),
        minimum_multiplier=float(np.min(multipliers)),
        maximum_multiplier=float(np.max(multipliers)),
        maximum_multiplier_increase=maximum_increase,
        initially_healthy=initially_healthy,
        monotone_nonincreasing=monotone,
    )
