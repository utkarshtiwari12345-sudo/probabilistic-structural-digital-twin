"""Regression tests for Stage 15 uncertainty-recovery gates."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_uncertainty_calibration_study import (  # noqa: E402
    _evaluate_recovery_gates,
)


def gates() -> dict[str, float]:
    return {
        "minimum_evaluation_coverage": 0.85,
        "maximum_evaluation_coverage": 0.99,
        "minimum_coverage_lower_confidence_bound": 0.75,
        "maximum_mean_interval_width": 0.08,
        "maximum_interval_score_ratio_to_raw": 0.75,
        "maximum_healthy_false_alarm_fraction": 0.05,
        "minimum_stressed_detection_fraction": 0.75,
        "maximum_predamage_false_alarm_fraction": 0.05,
        "maximum_posterior_mean_shift": 1e-14,
    }


def evaluation_runs() -> pd.DataFrame:
    rows = []
    for method, coverage, score in (
        ("raw", 0.2, 0.12),
        ("calibrated", 0.94, 0.02),
    ):
        for case in ("matched", "stressed"):
            for replicate in range(4):
                rows.append(
                    {
                        "method": method,
                        "case": case,
                        "replicate": replicate,
                        "coverage": coverage,
                        "mean_interval_width": 0.02,
                        "mean_interval_score": score,
                        "detected": 1.0,
                        "predamage_false_alarm_fraction": 0.0,
                        "posterior_mean_shift": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_corrected_method_must_pass_coverage_sharpness_and_decisions() -> None:
    healthy = pd.DataFrame(
        {
            "method": ["raw", "calibrated"],
            "zone_false_alarm_fraction": [0.0, 0.0],
        }
    )

    result = _evaluate_recovery_gates(evaluation_runs(), healthy, gates())

    recovered = result.groupby("method")["passed"].all()
    assert not bool(recovered["raw"])
    assert bool(recovered["calibrated"])


def test_false_alarm_failure_blocks_otherwise_calibrated_method() -> None:
    healthy = pd.DataFrame(
        {
            "method": ["raw", "calibrated"],
            "zone_false_alarm_fraction": [0.0, 0.5],
        }
    )

    result = _evaluate_recovery_gates(evaluation_runs(), healthy, gates())

    calibrated = result[result["method"] == "calibrated"]
    assert not bool(calibrated["passed"].all())
    failed = calibrated[~calibrated["passed"]]
    assert failed["check"].tolist() == ["healthy_false_alarm_fraction"]
