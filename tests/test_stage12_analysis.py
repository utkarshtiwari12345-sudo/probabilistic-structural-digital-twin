"""Regression tests for Stage 12 paired response analysis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_monte_carlo_confirmation import (  # noqa: E402
    RESPONSE_NAMES,
    _paired_effect_tables,
)


def test_paired_effect_table_accepts_boolean_responses() -> None:
    rows = []
    for context in ("first", "second"):
        for replicate in range(3):
            for noise in (0.02, 0.10):
                high = noise > 0.05
                row = {
                    "context": context,
                    "replicate": replicate,
                    "measurement_noise_fraction": noise,
                }
                for response in RESPONSE_NAMES:
                    row[response] = (
                        high
                        if response in {"detected", "localized"}
                        else float(replicate + high)
                    )
                rows.append(row)
    runs = pd.DataFrame(rows)
    stage11 = pd.DataFrame(
        [
            {
                "response": response,
                "term": "noise",
                "term_order": 1,
                "high_minus_low_effect": 1.0,
            }
            for response in RESPONSE_NAMES
        ]
    )

    samples, effects, convergence = _paired_effect_tables(
        runs,
        (0.02, 0.10),
        0.95,
        stage11,
    )

    assert np.allclose(samples["high_minus_low_difference"], 1.0)
    assert np.allclose(effects["effect_mean"], 1.0)
    assert convergence["n_replicates"].max() == 3
