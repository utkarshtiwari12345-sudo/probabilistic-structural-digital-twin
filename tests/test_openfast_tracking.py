"""Tests for discrepancy-aware discrete OpenFAST tracking."""

from __future__ import annotations

import numpy as np
import pytest

from probdt.openfast import (
    OpenFASTTrackingError,
    fit_shrunk_linear_damage_templates,
    monotonic_damage_transition,
    normalized_log_weights,
    run_discrete_damage_filter,
    student_t_diagonal_log_likelihood,
    summarize_damage_posterior,
)

STATE_ZONES = ("healthy", "lower", "lower", "middle", "middle")
STATE_LOSSES = np.asarray([0.0, 0.1, 0.2, 0.1, 0.2])


def test_monotonic_transition_is_stochastic_and_nonhealing() -> None:
    transition = monotonic_damage_transition(
        STATE_ZONES,
        STATE_LOSSES,
        healthy_persistence_probability=0.98,
        damage_progression_probability=0.03,
        zone_switch_probability=0.01,
    )
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert np.all(transition >= 0.0)
    assert transition[2, 0] == 0.0
    assert transition[4, 0] == 0.0
    assert transition[1, 2] == pytest.approx(0.03)


def test_transition_can_represent_abrupt_entry_severity() -> None:
    transition = monotonic_damage_transition(
        STATE_ZONES,
        STATE_LOSSES,
        healthy_persistence_probability=0.90,
        damage_progression_probability=0.03,
        zone_switch_probability=0.01,
        healthy_entry_severity_weights=(0.7, 0.3),
    )
    assert transition[0, 1] == pytest.approx(0.035)
    assert transition[0, 2] == pytest.approx(0.015)
    assert transition[0, 3] == pytest.approx(0.035)
    assert transition[0, 4] == pytest.approx(0.015)


def test_hybrid_template_fit_reduces_development_residual() -> None:
    physics = np.asarray([[0.0, 0.0], [0.3, 0.1], [0.6, 0.2], [-0.1, 0.2], [-0.2, 0.4]])
    development_zones = ("lower", "lower", "middle", "middle")
    development_losses = np.asarray([0.1, 0.2, 0.1, 0.2])
    development_features = np.asarray(
        [[0.1, 0.2], [0.2, 0.4], [-0.2, 0.1], [-0.4, 0.2]]
    )
    result = fit_shrunk_linear_damage_templates(
        physics,
        STATE_ZONES,
        STATE_LOSSES,
        development_features,
        development_zones,
        development_losses,
        empirical_blend_weight=1.0,
    )
    assert np.array_equal(result.corrected_state_templates[0], physics[0])
    assert np.allclose(result.corrected_state_templates[1], [0.1, 0.2])
    assert np.allclose(result.corrected_state_templates[4], [-0.4, 0.2])
    assert np.max(np.abs(result.development_residuals)) < 1.0e-12


def test_student_t_likelihood_prefers_matching_template() -> None:
    templates = np.asarray([[0.0, 0.0], [1.0, -0.5], [2.0, 1.0]])
    likelihood = student_t_diagonal_log_likelihood(
        np.asarray([1.0, -0.5]),
        templates,
        np.ones(2),
        degrees_of_freedom=4.0,
        likelihood_power=0.5,
    )
    assert int(np.argmax(likelihood)) == 1
    assert np.isclose(normalized_log_weights(likelihood).sum(), 1.0)


def test_discrete_filter_normalizes_and_moves_toward_damage() -> None:
    transition = monotonic_damage_transition(
        STATE_ZONES,
        STATE_LOSSES,
        healthy_persistence_probability=0.95,
        damage_progression_probability=0.03,
        zone_switch_probability=0.01,
    )
    templates = np.asarray(
        [[0.0, 0.0], [1.0, 0.8], [2.0, 1.6], [-1.0, -0.8], [-2.0, -1.6]]
    )
    observations = np.asarray([[0.0, 0.0], [0.9, 0.7], [1.1, 0.9], [2.0, 1.6]])
    result = run_discrete_damage_filter(
        observations,
        templates,
        np.full(2, 0.05),
        transition,
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
        degrees_of_freedom=5.0,
        likelihood_power=1.0,
    )
    assert np.allclose(result.posterior_state_probabilities.sum(axis=1), 1.0)
    assert result.posterior_state_probabilities[-1, 2] > 0.9


def test_posterior_summary_reports_damage_zone_and_interval() -> None:
    summary = summarize_damage_posterior(
        np.asarray([0.05, 0.10, 0.65, 0.05, 0.15]),
        STATE_ZONES,
        STATE_LOSSES,
        damaged_zone_order=("lower", "middle"),
        meaningful_loss_fraction=0.05,
        credible_level=0.95,
    )
    assert summary.damage_probability == pytest.approx(0.95)
    assert summary.expected_loss_fraction == pytest.approx(0.175)
    assert summary.credible_interval_lower == 0.0
    assert summary.credible_interval_upper == 0.2
    assert summary.conditional_zone_probabilities[0] == pytest.approx(0.75 / 0.95)
    assert summary.map_state_index == 2


def test_tracking_rejects_invalid_state_library() -> None:
    with pytest.raises(OpenFASTTrackingError, match="one healthy state"):
        monotonic_damage_transition(
            ("healthy", "also_healthy", "lower"),
            (0.0, 0.0, 0.1),
            healthy_persistence_probability=0.98,
            damage_progression_probability=0.01,
            zone_switch_probability=0.0,
        )
