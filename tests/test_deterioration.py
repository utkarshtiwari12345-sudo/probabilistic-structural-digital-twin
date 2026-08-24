"""Tests for bounded, reproducible deterioration truth histories."""

import numpy as np
import pytest

from probdt.deterioration import (
    ConditionThresholds,
    DeteriorationError,
    abrupt_loss,
    build_deterioration_history,
    classify_condition,
    combine_independent_losses,
    gamma_process_loss,
    smooth_gradual_loss,
    validate_deterioration_history,
)
from probdt.structural import (
    apply_stiffness_multipliers,
    assemble_zonal_stiffness_components,
    validate_structural_matrices,
)


@pytest.fixture
def time_grid() -> np.ndarray:
    return np.arange(0.0, 101.0, 1.0)


def test_smooth_gradual_loss_has_exact_endpoints_and_is_monotone(
    time_grid: np.ndarray,
) -> None:
    loss = smooth_gradual_loss(
        time_grid,
        onset_s=20.0,
        completion_s=80.0,
        final_loss_fraction=0.20,
    )

    assert np.all(loss[time_grid <= 20.0] == 0.0)
    assert np.all(loss[time_grid >= 80.0] == 0.20)
    assert np.all(np.diff(loss) >= 0.0)


def test_abrupt_loss_occurs_at_first_sample_on_or_after_event(
    time_grid: np.ndarray,
) -> None:
    loss = abrupt_loss(time_grid, event_time_s=40.5, loss_fraction=0.10)

    assert np.all(loss[time_grid < 40.5] == 0.0)
    assert np.all(loss[time_grid >= 40.5] == 0.10)


def test_independent_losses_combine_multiplicatively() -> None:
    first = np.array([0.0, 0.10])
    second = np.array([0.0, 0.20])

    combined = combine_independent_losses([first, second])

    np.testing.assert_allclose(combined, [0.0, 0.28])


def test_gamma_process_is_reproducible_monotone_and_bounded(
    time_grid: np.ndarray,
) -> None:
    arguments = {
        "onset_s": 10.0,
        "mean_final_loss_fraction": 0.15,
        "coefficient_of_variation": 0.20,
        "maximum_loss_fraction": 0.35,
    }
    first = gamma_process_loss(
        time_grid,
        rng=np.random.default_rng(42),
        **arguments,
    )
    second = gamma_process_loss(
        time_grid,
        rng=np.random.default_rng(42),
        **arguments,
    )

    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first) >= 0.0)
    assert np.all(first[time_grid <= 10.0] == 0.0)
    assert first[-1] <= 0.35


def test_history_assigns_healthy_multiplier_to_unspecified_zones(
    time_grid: np.ndarray,
) -> None:
    lower_loss = smooth_gradual_loss(
        time_grid,
        onset_s=10.0,
        completion_s=90.0,
        final_loss_fraction=0.15,
    )
    history = build_deterioration_history(
        "lower_only",
        time_grid,
        ("lower", "middle", "upper"),
        {"lower": lower_loss},
    )

    np.testing.assert_array_equal(history.stiffness_multipliers[:, 1:], 1.0)
    np.testing.assert_allclose(history.loss_fractions[:, 0], lower_loss)


def test_validation_detects_artificial_healing(time_grid: np.ndarray) -> None:
    loss = np.zeros(time_grid.size)
    loss[40:60] = 0.1
    history = build_deterioration_history(
        "healing_case",
        time_grid,
        ("lower",),
        {"lower": loss},
    )

    report = validate_deterioration_history(history)

    assert not report.monotone_nonincreasing
    assert report.maximum_multiplier_increase == pytest.approx(0.1)


def test_condition_classification_uses_declared_boundaries() -> None:
    thresholds = ConditionThresholds(0.01, 0.05, 0.12)
    losses = np.array([0.0, 0.01, 0.02, 0.05, 0.08, 0.12, 0.13])

    stages = classify_condition(losses, thresholds)

    assert stages.tolist() == [
        "healthy",
        "healthy",
        "early",
        "early",
        "moderate",
        "moderate",
        "significant",
    ]


def test_deterioration_preserves_positive_definite_stiffness(
    time_grid: np.ndarray,
) -> None:
    story_stiffnesses = [30.0e6, 28.0e6, 26.0e6, 24.0e6, 22.0e6, 20.0e6]
    zones = {"lower": [0, 1], "middle": [2, 3], "upper": [4, 5]}
    zone_names, components = assemble_zonal_stiffness_components(
        story_stiffnesses,
        zones,
    )
    lower_loss = smooth_gradual_loss(
        time_grid,
        onset_s=10.0,
        completion_s=90.0,
        final_loss_fraction=0.30,
    )
    history = build_deterioration_history(
        "physical_case",
        time_grid,
        zone_names,
        {"lower": lower_loss},
    )
    mass = np.diag([420.0e3, 400.0e3, 380.0e3, 360.0e3, 340.0e3, 520.0e3])

    for multipliers in history.stiffness_multipliers:
        stiffness = apply_stiffness_multipliers(components, multipliers)
        report = validate_structural_matrices(mass, stiffness)
        assert report.minimum_stiffness_eigenvalue_n_per_m > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"onset_s": 20.0, "completion_s": 20.0, "final_loss_fraction": 0.1},
        {"onset_s": -1.0, "completion_s": 20.0, "final_loss_fraction": 0.1},
        {"onset_s": 10.0, "completion_s": 20.0, "final_loss_fraction": 1.0},
    ],
)
def test_invalid_gradual_process_is_rejected(
    time_grid: np.ndarray,
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(DeteriorationError):
        smooth_gradual_loss(time_grid, **kwargs)
