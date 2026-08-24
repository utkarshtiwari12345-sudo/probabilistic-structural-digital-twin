"""Regression tests for Stage 14 recovery-gate aggregation."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_information_remedy_study import _evaluate_recovery_gates  # noqa: E402


def gates() -> dict[str, float]:
    return {
        "maximum_displacement_nrmse": 0.65,
        "maximum_velocity_nrmse": 0.75,
        "maximum_stiffness_rmse": 0.06,
        "minimum_matched_stiffness_coverage": 0.75,
        "maximum_healthy_false_alarm_fraction": 0.05,
        "minimum_stressed_detection_fraction": 0.60,
        "maximum_predamage_false_alarm_fraction": 0.05,
    }


def summary() -> pd.DataFrame:
    rows = []
    for scheme in ("restored", "overconfident"):
        for case in ("matched", "stressed"):
            rows.append(
                {
                    "scheme": scheme,
                    "case": case,
                    "displacement_nrmse_mean": 0.1,
                    "velocity_nrmse_mean": 0.2,
                    "stiffness_rmse_mean": 0.02,
                    "stiffness_coverage_mean": (
                        0.9 if scheme == "restored" else 0.2
                    ),
                    "detected_mean": 1.0,
                    "predamage_false_alarm_fraction_mean": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_recovery_requires_every_gate_including_coverage() -> None:
    calibration = pd.DataFrame(
        {
            "scheme": ["restored", "overconfident"],
            "selected": [True, True],
            "healthy_false_alarm_fraction": [0.0, 0.0],
            "calibration_succeeded": [True, True],
        }
    )

    result = _evaluate_recovery_gates(summary(), calibration, gates())

    status = result.groupby("scheme")["passed"].all()
    assert bool(status["restored"])
    assert not bool(status["overconfident"])
    failed = result[
        (result["scheme"] == "overconfident") & ~result["passed"]
    ]
    assert failed["check"].tolist() == ["matched_stiffness_coverage"]


def test_failed_calibration_prevents_restoration() -> None:
    calibration = pd.DataFrame(
        {
            "scheme": ["restored", "overconfident"],
            "selected": [True, True],
            "healthy_false_alarm_fraction": [0.5, 0.0],
            "calibration_succeeded": [False, True],
        }
    )

    result = _evaluate_recovery_gates(summary(), calibration, gates())

    calibration_gate = result[
        (result["scheme"] == "restored")
        & (result["check"] == "healthy_alarm_calibration")
    ].iloc[0]
    assert not bool(calibration_gate["passed"])
    assert not bool(calibration_gate["scheme_restores_joint_recovery"])
