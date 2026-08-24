"""Balanced two-level factorial effect estimation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class FactorialAnalysisError(ValueError):
    """Raised when a two-level factorial design is invalid."""


@dataclass(frozen=True)
class FactorialEffects:
    """Least-squares effects for a balanced two-level factorial design."""

    term_names: tuple[str, ...]
    coefficients: FloatArray
    effects: FloatArray
    fitted_responses: FloatArray
    residuals: FloatArray
    design_rank: int


def estimate_two_level_factorial_effects(
    coded_factors: ArrayLike,
    responses: ArrayLike,
    factor_names: tuple[str, ...],
    *,
    include_two_factor_interactions: bool = True,
) -> FactorialEffects:
    """Estimate high-minus-low effects from a balanced full factorial.

    Factors must be coded as exactly ``-1`` and ``+1``. Repeated full
    factorials are allowed, but every factor combination must occur equally
    often. The returned intercept effect is ``NaN``; all other effects are
    twice their regression coefficient under the conventional coding.
    """

    factors = np.asarray(coded_factors, dtype=np.float64)
    response = np.asarray(responses, dtype=np.float64)
    if factors.ndim != 2 or factors.shape[0] < 1 or factors.shape[1] < 1:
        raise FactorialAnalysisError(
            "coded_factors must be a nonempty two-dimensional array."
        )
    if response.shape != (factors.shape[0],):
        raise FactorialAnalysisError(
            "responses must contain one value for every design row."
        )
    if not np.all(np.isfinite(factors)) or not np.all(np.isfinite(response)):
        raise FactorialAnalysisError("Factors and responses must be finite.")
    if not np.all(np.isin(factors, (-1.0, 1.0))):
        raise FactorialAnalysisError("Every factor must be coded as -1 or +1.")
    if len(factor_names) != factors.shape[1]:
        raise FactorialAnalysisError(
            "factor_names must match the number of factor columns."
        )
    if any(not name.strip() for name in factor_names) or len(
        set(factor_names)
    ) != len(factor_names):
        raise FactorialAnalysisError("Factor names must be nonempty and unique.")

    combinations_present, counts = np.unique(factors, axis=0, return_counts=True)
    if combinations_present.shape[0] != 2 ** factors.shape[1] or not np.all(
        counts == counts[0]
    ):
        raise FactorialAnalysisError(
            "The design must be a balanced full two-level factorial."
        )

    columns = [np.ones(factors.shape[0], dtype=np.float64)]
    terms = ["intercept"]
    for index, name in enumerate(factor_names):
        columns.append(factors[:, index])
        terms.append(name)
    if include_two_factor_interactions:
        for first, second in combinations(range(factors.shape[1]), 2):
            columns.append(factors[:, first] * factors[:, second])
            terms.append(f"{factor_names[first]}:{factor_names[second]}")

    design = np.column_stack(columns)
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ coefficients
    effects = 2.0 * coefficients
    effects[0] = np.nan
    return FactorialEffects(
        term_names=tuple(terms),
        coefficients=coefficients,
        effects=effects,
        fitted_responses=fitted,
        residuals=response - fitted,
        design_rank=int(rank),
    )
