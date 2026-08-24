"""Discrete Bayesian tracking for OpenFAST stiffness-state libraries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class OpenFASTTrackingError(ValueError):
    """Raised when a discrete OpenFAST tracking input is invalid."""


@dataclass(frozen=True)
class DiscreteTrackingResult:
    """Posterior histories from a finite-state Bayesian filter."""

    predicted_state_probabilities: FloatArray
    posterior_state_probabilities: FloatArray
    log_likelihoods: FloatArray


@dataclass(frozen=True)
class DamagePosteriorSummary:
    """Decision quantities derived from one discrete posterior."""

    damage_probability: float
    expected_loss_fraction: float
    credible_interval_lower: float
    credible_interval_upper: float
    zone_probabilities: FloatArray
    conditional_zone_probabilities: FloatArray
    map_state_index: int
    posterior_entropy: float


@dataclass(frozen=True)
class HybridTemplateCalibration:
    """Physics-informed local-linear correction fitted on development cases."""

    corrected_state_templates: FloatArray
    zone_names: tuple[str, ...]
    physics_slopes: FloatArray
    empirical_slopes: FloatArray
    blended_slopes: FloatArray
    development_residuals: FloatArray


def _validate_state_library(
    state_zones: Sequence[str],
    state_losses: ArrayLike,
) -> tuple[tuple[str, ...], FloatArray]:
    zones = tuple(str(value) for value in state_zones)
    losses = np.asarray(state_losses, dtype=np.float64)
    if not zones or losses.shape != (len(zones),):
        raise OpenFASTTrackingError("State zones and losses must align.")
    if not np.all(np.isfinite(losses)) or np.any(losses < 0.0):
        raise OpenFASTTrackingError("State losses must be finite and nonnegative.")
    if len(set(zip(zones, losses, strict=True))) != len(zones):
        raise OpenFASTTrackingError("Every zone-loss state must be unique.")
    if int(np.sum(np.isclose(losses, 0.0))) != 1:
        raise OpenFASTTrackingError("The state library must contain one healthy state.")
    return zones, losses


def monotonic_damage_transition(
    state_zones: Sequence[str],
    state_losses: ArrayLike,
    *,
    healthy_persistence_probability: float,
    damage_progression_probability: float,
    zone_switch_probability: float,
    healthy_entry_severity_weights: ArrayLike | None = None,
) -> FloatArray:
    """Construct a non-healing transition matrix over ordered damage states."""

    zones, losses = _validate_state_library(state_zones, state_losses)
    probabilities = (
        healthy_persistence_probability,
        damage_progression_probability,
        zone_switch_probability,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in probabilities):
        raise OpenFASTTrackingError("Transition probabilities must be nonnegative.")
    if healthy_persistence_probability > 1.0:
        raise OpenFASTTrackingError("Healthy persistence cannot exceed one.")
    if damage_progression_probability + zone_switch_probability > 1.0:
        raise OpenFASTTrackingError("Damage transition probabilities exceed one.")

    healthy_index = int(np.flatnonzero(np.isclose(losses, 0.0))[0])
    damaged_zones = tuple(
        dict.fromkeys(
            zones[index] for index in range(len(zones)) if index != healthy_index
        )
    )
    if not damaged_zones:
        raise OpenFASTTrackingError("At least one damaged zone is required.")
    transition = np.zeros((len(zones), len(zones)), dtype=np.float64)

    first_zone_states = sorted(
        [
            index
            for index, (candidate_zone, loss) in enumerate(
                zip(zones, losses, strict=True)
            )
            if candidate_zone == damaged_zones[0] and loss > 0.0
        ],
        key=lambda index: losses[index],
    )
    if healthy_entry_severity_weights is None:
        severity_weights = np.zeros(len(first_zone_states), dtype=np.float64)
        severity_weights[0] = 1.0
    else:
        severity_weights = np.asarray(
            healthy_entry_severity_weights,
            dtype=np.float64,
        )
        if severity_weights.shape != (len(first_zone_states),):
            raise OpenFASTTrackingError(
                "Healthy-entry weights must match the number of loss levels."
            )
        if np.any(severity_weights < 0.0) or not np.isclose(
            np.sum(severity_weights), 1.0, atol=1.0e-12
        ):
            raise OpenFASTTrackingError(
                "Healthy-entry weights must be nonnegative and sum to one."
            )
    entry_probability = (1.0 - healthy_persistence_probability) / len(damaged_zones)
    transition[healthy_index, healthy_index] = healthy_persistence_probability
    for zone in damaged_zones:
        candidates = sorted(
            [
                index
                for index, (candidate_zone, loss) in enumerate(
                    zip(zones, losses, strict=True)
                )
                if candidate_zone == zone and loss > 0.0
            ],
            key=lambda index: losses[index],
        )
        if len(candidates) != len(first_zone_states):
            raise OpenFASTTrackingError(
                "Every damage zone must share the same loss levels."
            )
        for candidate, severity_weight in zip(
            candidates,
            severity_weights,
            strict=True,
        ):
            transition[healthy_index, candidate] = entry_probability * severity_weight

    for index, (zone, loss) in enumerate(zip(zones, losses, strict=True)):
        if index == healthy_index:
            continue
        same_zone_higher = [
            candidate
            for candidate, (candidate_zone, candidate_loss) in enumerate(
                zip(zones, losses, strict=True)
            )
            if candidate_zone == zone and candidate_loss > loss
        ]
        progression = 0.0
        if same_zone_higher:
            next_index = min(same_zone_higher, key=lambda item: losses[item])
            transition[index, next_index] = damage_progression_probability
            progression = damage_progression_probability

        alternatives = [
            candidate
            for candidate, (candidate_zone, candidate_loss) in enumerate(
                zip(zones, losses, strict=True)
            )
            if candidate_zone != zone
            and candidate != healthy_index
            and np.isclose(candidate_loss, loss)
        ]
        switching = zone_switch_probability if alternatives else 0.0
        for candidate in alternatives:
            transition[index, candidate] = switching / len(alternatives)
        transition[index, index] = 1.0 - progression - switching

    if not np.allclose(transition.sum(axis=1), 1.0, atol=1.0e-12):
        raise OpenFASTTrackingError("Transition rows do not sum to one.")
    if np.any(transition < 0.0):
        raise OpenFASTTrackingError("Transition probabilities became negative.")
    return transition


def fit_shrunk_linear_damage_templates(
    physics_state_templates: ArrayLike,
    state_zones: Sequence[str],
    state_losses: ArrayLike,
    development_features: ArrayLike,
    development_zones: Sequence[str],
    development_losses: ArrayLike,
    *,
    empirical_blend_weight: float,
) -> HybridTemplateCalibration:
    """Shrink empirical local damage slopes toward reduced-twin predictions."""

    zones, losses = _validate_state_library(state_zones, state_losses)
    physics = np.asarray(physics_state_templates, dtype=np.float64)
    features = np.asarray(development_features, dtype=np.float64)
    feature_zones = np.asarray([str(value) for value in development_zones])
    feature_losses = np.asarray(development_losses, dtype=np.float64)
    if physics.ndim != 2 or physics.shape[0] != len(zones):
        raise OpenFASTTrackingError("Physics templates must align with the states.")
    if features.ndim != 2 or features.shape[1] != physics.shape[1]:
        raise OpenFASTTrackingError(
            "Development features must align with template features."
        )
    if feature_zones.shape != (features.shape[0],) or feature_losses.shape != (
        features.shape[0],
    ):
        raise OpenFASTTrackingError("Development labels must align with features.")
    if not np.all(np.isfinite(physics)) or not np.all(np.isfinite(features)):
        raise OpenFASTTrackingError("Hybrid template inputs must be finite.")
    if np.any(feature_losses <= 0.0) or not np.all(np.isfinite(feature_losses)):
        raise OpenFASTTrackingError("Development losses must be finite and positive.")
    if not np.isfinite(empirical_blend_weight) or not (
        0.0 <= empirical_blend_weight <= 1.0
    ):
        raise OpenFASTTrackingError("Empirical blend weight must lie in [0, 1].")

    healthy_index = int(np.flatnonzero(np.isclose(losses, 0.0))[0])
    damaged_zones = tuple(
        dict.fromkeys(
            zones[index] for index in range(len(zones)) if index != healthy_index
        )
    )
    corrected = physics.copy()
    physics_slopes = []
    empirical_slopes = []
    blended_slopes = []
    for zone in damaged_zones:
        development_mask = feature_zones == zone
        zone_development_losses = feature_losses[development_mask]
        if np.unique(zone_development_losses).size < 2:
            raise OpenFASTTrackingError(
                "Every zone requires at least two development severities."
            )
        zone_features = features[development_mask]
        empirical_denominator = float(np.sum(zone_development_losses**2))
        empirical_slope = (
            np.sum(
                zone_development_losses[:, np.newaxis] * zone_features,
                axis=0,
            )
            / empirical_denominator
        )

        development_levels = np.unique(zone_development_losses)
        state_indices = [
            index
            for index, (state_zone, state_loss) in enumerate(
                zip(zones, losses, strict=True)
            )
            if state_zone == zone and np.any(np.isclose(state_loss, development_levels))
        ]
        state_loss_values = losses[state_indices]
        physics_denominator = float(np.sum(state_loss_values**2))
        physics_slope = (
            np.sum(
                state_loss_values[:, np.newaxis] * physics[state_indices],
                axis=0,
            )
            / physics_denominator
        )
        blended_slope = (
            1.0 - empirical_blend_weight
        ) * physics_slope + empirical_blend_weight * empirical_slope
        for state_index, (state_zone, state_loss) in enumerate(
            zip(zones, losses, strict=True)
        ):
            if state_zone == zone and state_loss > 0.0:
                corrected[state_index] = physics[state_index] + (
                    empirical_blend_weight
                    * state_loss
                    * (empirical_slope - physics_slope)
                )
        physics_slopes.append(physics_slope)
        empirical_slopes.append(empirical_slope)
        blended_slopes.append(blended_slope)

    residuals = []
    for feature, zone, loss in zip(
        features,
        feature_zones,
        feature_losses,
        strict=True,
    ):
        matching = [
            index
            for index, (state_zone, state_loss) in enumerate(
                zip(zones, losses, strict=True)
            )
            if state_zone == zone and np.isclose(state_loss, loss)
        ]
        if len(matching) != 1:
            raise OpenFASTTrackingError(
                "Each development label must match exactly one damage state."
            )
        residuals.append(feature - corrected[matching[0]])
    return HybridTemplateCalibration(
        corrected_state_templates=corrected,
        zone_names=damaged_zones,
        physics_slopes=np.stack(physics_slopes),
        empirical_slopes=np.stack(empirical_slopes),
        blended_slopes=np.stack(blended_slopes),
        development_residuals=np.stack(residuals),
    )


def student_t_diagonal_log_likelihood(
    observation: ArrayLike,
    state_templates: ArrayLike,
    feature_variances: ArrayLike,
    *,
    degrees_of_freedom: float,
    likelihood_power: float,
) -> FloatArray:
    """Evaluate a robust diagonal Student-t likelihood for every state."""

    values = np.asarray(observation, dtype=np.float64)
    templates = np.asarray(state_templates, dtype=np.float64)
    variances = np.asarray(feature_variances, dtype=np.float64)
    if values.ndim != 1 or templates.ndim != 2 or templates.shape[1] != values.size:
        raise OpenFASTTrackingError("Observation and state templates must align.")
    if variances.shape != values.shape:
        raise OpenFASTTrackingError("Feature variances must match the observation.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(templates)):
        raise OpenFASTTrackingError("Likelihood inputs must be finite.")
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        raise OpenFASTTrackingError("Feature variances must be finite and positive.")
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0.0:
        raise OpenFASTTrackingError("Student-t degrees of freedom must be positive.")
    if not np.isfinite(likelihood_power) or not 0.0 < likelihood_power <= 1.0:
        raise OpenFASTTrackingError("likelihood_power must lie in (0, 1].")

    standardized_square = (templates - values[np.newaxis, :]) ** 2 / variances
    data_term = (
        -0.5
        * (degrees_of_freedom + 1.0)
        * np.sum(
            np.log1p(standardized_square / degrees_of_freedom),
            axis=1,
        )
    )
    return likelihood_power * data_term


def normalized_log_weights(log_weights: ArrayLike) -> FloatArray:
    """Normalize finite log weights without numerical underflow."""

    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise OpenFASTTrackingError("Log weights must be a finite vector.")
    shifted = np.exp(values - np.max(values))
    total = float(np.sum(shifted))
    if not np.isfinite(total) or total <= 0.0:
        raise OpenFASTTrackingError("Log weights could not be normalized.")
    return shifted / total


def run_discrete_damage_filter(
    observations: ArrayLike,
    state_templates: ArrayLike,
    feature_variances: ArrayLike,
    transition_matrix: ArrayLike,
    initial_state_probabilities: ArrayLike,
    *,
    degrees_of_freedom: float,
    likelihood_power: float,
) -> DiscreteTrackingResult:
    """Apply prediction and robust Bayesian updating to sequential features."""

    features = np.asarray(observations, dtype=np.float64)
    templates = np.asarray(state_templates, dtype=np.float64)
    variances = np.asarray(feature_variances, dtype=np.float64)
    transition = np.asarray(transition_matrix, dtype=np.float64)
    initial = np.asarray(initial_state_probabilities, dtype=np.float64)
    if features.ndim != 2 or templates.ndim != 2:
        raise OpenFASTTrackingError("Observations and templates must be matrices.")
    if features.shape[1] != templates.shape[1]:
        raise OpenFASTTrackingError("Observations and templates must share features.")
    n_states = templates.shape[0]
    if transition.shape != (n_states, n_states):
        raise OpenFASTTrackingError("Transition matrix has the wrong shape.")
    if np.any(transition < 0.0) or not np.allclose(
        transition.sum(axis=1), 1.0, atol=1.0e-12
    ):
        raise OpenFASTTrackingError("Transition matrix must be row stochastic.")
    if initial.shape != (n_states,) or np.any(initial < 0.0):
        raise OpenFASTTrackingError("Initial probabilities have the wrong shape.")
    if not np.isclose(np.sum(initial), 1.0, atol=1.0e-12):
        raise OpenFASTTrackingError("Initial probabilities must sum to one.")

    predicted_history = np.empty((features.shape[0], n_states), dtype=np.float64)
    posterior_history = np.empty_like(predicted_history)
    likelihood_history = np.empty_like(predicted_history)
    posterior = initial.copy()
    for window_index, observation in enumerate(features):
        predicted = posterior @ transition
        log_likelihood = student_t_diagonal_log_likelihood(
            observation,
            templates,
            variances,
            degrees_of_freedom=degrees_of_freedom,
            likelihood_power=likelihood_power,
        )
        safe_predicted = np.maximum(predicted, np.finfo(np.float64).tiny)
        posterior = normalized_log_weights(np.log(safe_predicted) + log_likelihood)
        predicted_history[window_index] = predicted
        posterior_history[window_index] = posterior
        likelihood_history[window_index] = log_likelihood
    return DiscreteTrackingResult(
        predicted_state_probabilities=predicted_history,
        posterior_state_probabilities=posterior_history,
        log_likelihoods=likelihood_history,
    )


def _weighted_quantile(
    values: FloatArray,
    weights: FloatArray,
    quantile: float,
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def summarize_damage_posterior(
    posterior: ArrayLike,
    state_zones: Sequence[str],
    state_losses: ArrayLike,
    *,
    damaged_zone_order: Sequence[str],
    meaningful_loss_fraction: float,
    credible_level: float,
) -> DamagePosteriorSummary:
    """Convert a state posterior into damage, severity, and zone quantities."""

    zones, losses = _validate_state_library(state_zones, state_losses)
    probabilities = np.asarray(posterior, dtype=np.float64)
    if probabilities.shape != losses.shape or np.any(probabilities < 0.0):
        raise OpenFASTTrackingError("Posterior probabilities do not match states.")
    if not np.isclose(np.sum(probabilities), 1.0, atol=1.0e-12):
        raise OpenFASTTrackingError("Posterior probabilities must sum to one.")
    if not meaningful_loss_fraction >= 0.0 or not 0.0 < credible_level < 1.0:
        raise OpenFASTTrackingError("Damage threshold or credible level is invalid.")

    damaged = losses >= meaningful_loss_fraction
    damage_probability = float(np.sum(probabilities[damaged]))
    ordered_zones = tuple(str(value) for value in damaged_zone_order)
    if len(set(ordered_zones)) != len(ordered_zones):
        raise OpenFASTTrackingError("Damaged zone order must be unique.")
    zone_probabilities = np.asarray(
        [
            np.sum(
                probabilities[
                    np.asarray(
                        [
                            zone == selected and loss > 0.0
                            for zone, loss in zip(zones, losses, strict=True)
                        ],
                        dtype=np.bool_,
                    )
                ]
            )
            for selected in ordered_zones
        ],
        dtype=np.float64,
    )
    conditional = (
        zone_probabilities / np.sum(zone_probabilities)
        if np.sum(zone_probabilities) > 0.0
        else np.zeros_like(zone_probabilities)
    )
    tail = (1.0 - credible_level) / 2.0
    positive = probabilities > 0.0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return DamagePosteriorSummary(
        damage_probability=damage_probability,
        expected_loss_fraction=float(np.dot(probabilities, losses)),
        credible_interval_lower=_weighted_quantile(losses, probabilities, tail),
        credible_interval_upper=_weighted_quantile(losses, probabilities, 1.0 - tail),
        zone_probabilities=zone_probabilities,
        conditional_zone_probabilities=conditional,
        map_state_index=int(np.argmax(probabilities)),
        posterior_entropy=entropy,
    )
