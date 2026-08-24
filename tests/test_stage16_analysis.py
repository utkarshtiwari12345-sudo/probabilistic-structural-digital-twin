"""Tests for Stage 16 confirmation intervals and summaries."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_phase1_closure_study import (  # noqa: E402
    _summarize_confirmation,
    _wilson_interval,
)


def test_wilson_interval_bounds_zero_success_probability() -> None:
    lower, upper = _wilson_interval(0, 75)
    assert lower == 0.0
    assert upper < 0.05
    with pytest.raises(ValueError, match="Wilson"):
        _wilson_interval(2, 1)


def test_confirmation_summary_uses_wilson_decision_bounds() -> None:
    rows = []
    for replicate in range(10):
        rows.append(
            {
                "method": "scalar_inflation",
                "case": "stressed",
                "replicate": replicate,
                "coverage": 0.94,
                "mean_interval_width": 0.02,
                "mean_interval_score": 0.02,
                "mean_theta_nees": 4.0,
                "theta_rmse": 0.007,
                "displacement_nrmse": 0.005,
                "force_nrmse": 0.05,
                "predamage_false_alarm_fraction": 0.0,
                "detected": 1.0,
                "localized": 1.0,
            }
        )

    summary = _summarize_confirmation(pd.DataFrame(rows)).iloc[0]

    assert summary["detected_fraction"] == 1.0
    assert 0.70 < summary["detected_wilson_lower"] < 1.0
    assert summary["coverage_mean"] == pytest.approx(0.94)
    assert np.isfinite(summary["mean_theta_nees_confidence_upper"])
