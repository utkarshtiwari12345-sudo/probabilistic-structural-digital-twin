"""Regression tests for Stage 13 alarm-calibration analysis."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_joint_state_parameter_study import (  # noqa: E402
    _calibrate_thresholds_allow_failure,
)


def test_uncalibratable_method_is_retained_at_strictest_threshold() -> None:
    always_alarm = np.ones((4, 3), dtype=np.float64)

    thresholds, table = _calibrate_thresholds_allow_failure(
        {"unstable": [always_alarm, always_alarm]},
        [0.8, 0.95, 0.99],
        2,
        0.05,
    )

    selected = table[table["selected"]].iloc[0]
    assert thresholds["unstable"] == 0.99
    assert not bool(selected["calibration_succeeded"])
    assert selected["healthy_false_alarm_fraction"] == 1.0


def test_calibratable_method_uses_least_stringent_passing_threshold() -> None:
    probabilities = np.asarray(
        [
            [0.1, 0.1],
            [0.85, 0.1],
            [0.1, 0.1],
        ]
    )

    thresholds, table = _calibrate_thresholds_allow_failure(
        {"stable": [probabilities]},
        [0.8, 0.95],
        2,
        0.05,
    )

    selected = table[table["selected"]].iloc[0]
    assert thresholds["stable"] == 0.8
    assert bool(selected["calibration_succeeded"])
