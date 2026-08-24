"""Tests for replicated Monte Carlo interval summaries."""

import numpy as np
import pytest

from probdt.diagnostics import (
    MonteCarloAnalysisError,
    monte_carlo_mean_interval,
    paired_monte_carlo_difference,
)


def test_monte_carlo_summary_matches_sample_statistics() -> None:
    samples = np.asarray([1.0, 2.0, 3.0, 4.0])

    summary = monte_carlo_mean_interval(samples)

    assert summary.n_replicates == 4
    assert summary.mean == pytest.approx(2.5)
    assert summary.standard_deviation == pytest.approx(np.std(samples, ddof=1))
    assert summary.standard_error == pytest.approx(np.std(samples, ddof=1) / 2.0)
    assert summary.confidence_interval_lower < summary.mean
    assert summary.confidence_interval_upper > summary.mean


def test_constant_paired_difference_has_zero_width_interval() -> None:
    low = np.asarray([1.0, 3.0, 5.0, 7.0])
    high = low + 2.0

    summary = paired_monte_carlo_difference(high, low)

    assert summary.mean == pytest.approx(2.0)
    assert summary.standard_error == pytest.approx(0.0)
    assert summary.confidence_interval_lower == pytest.approx(2.0)
    assert summary.confidence_interval_upper == pytest.approx(2.0)


@pytest.mark.parametrize(
    "samples",
    [np.asarray([1.0]), np.asarray([1.0, np.nan]), np.ones((2, 2))],
)
def test_invalid_monte_carlo_samples_are_rejected(samples: np.ndarray) -> None:
    with pytest.raises(MonteCarloAnalysisError):
        monte_carlo_mean_interval(samples)


def test_unpaired_shapes_are_rejected() -> None:
    with pytest.raises(MonteCarloAnalysisError, match="identical shapes"):
        paired_monte_carlo_difference(np.ones(3), np.ones(4))
