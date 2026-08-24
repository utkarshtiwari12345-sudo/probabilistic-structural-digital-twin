"""Small-sample Monte Carlo summaries for replicated simulation studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t as student_t


class MonteCarloAnalysisError(ValueError):
    """Raised when replication samples cannot support an interval estimate."""


@dataclass(frozen=True)
class MonteCarloMeanInterval:
    """Mean, sampling uncertainty, and Student-t confidence interval."""

    n_replicates: int
    mean: float
    standard_deviation: float
    standard_error: float
    confidence_level: float
    confidence_interval_lower: float
    confidence_interval_upper: float


def monte_carlo_mean_interval(
    samples: ArrayLike,
    *,
    confidence_level: float = 0.95,
) -> MonteCarloMeanInterval:
    """Summarize independent scalar replication outcomes with a t interval."""

    values = np.asarray(samples, dtype=np.float64)
    confidence = float(confidence_level)
    if values.ndim != 1 or values.size < 2:
        raise MonteCarloAnalysisError(
            "samples must contain at least two one-dimensional replicates."
        )
    if not np.all(np.isfinite(values)):
        raise MonteCarloAnalysisError("samples must be finite.")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise MonteCarloAnalysisError(
            "confidence_level must lie strictly within (0, 1)."
        )

    standard_deviation = float(np.std(values, ddof=1))
    standard_error = standard_deviation / np.sqrt(values.size)
    critical = float(
        student_t.ppf(0.5 + confidence / 2.0, values.size - 1)
    )
    mean = float(np.mean(values))
    half_width = critical * standard_error
    return MonteCarloMeanInterval(
        n_replicates=int(values.size),
        mean=mean,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        confidence_level=confidence,
        confidence_interval_lower=mean - half_width,
        confidence_interval_upper=mean + half_width,
    )


def paired_monte_carlo_difference(
    high_samples: ArrayLike,
    low_samples: ArrayLike,
    *,
    confidence_level: float = 0.95,
) -> MonteCarloMeanInterval:
    """Summarize paired high-minus-low replication differences."""

    high = np.asarray(high_samples, dtype=np.float64)
    low = np.asarray(low_samples, dtype=np.float64)
    if high.shape != low.shape:
        raise MonteCarloAnalysisError(
            "Paired high and low samples must have identical shapes."
        )
    return monte_carlo_mean_interval(
        high - low,
        confidence_level=confidence_level,
    )
