"""Tests for balanced two-level factorial effect estimation."""

import itertools

import numpy as np
import pytest

from probdt.diagnostics import (
    FactorialAnalysisError,
    estimate_two_level_factorial_effects,
)


def test_exact_main_and_two_factor_effects_are_recovered() -> None:
    design = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=3)),
        dtype=np.float64,
    )
    response = (
        4.0
        + 1.5 * design[:, 0]
        - 0.5 * design[:, 1]
        + 0.25 * design[:, 2]
        + 0.75 * design[:, 0] * design[:, 2]
    )

    result = estimate_two_level_factorial_effects(
        design,
        response,
        ("noise", "severity", "data"),
    )

    effects = dict(zip(result.term_names, result.effects, strict=True))
    assert result.coefficients[0] == pytest.approx(4.0)
    assert effects["noise"] == pytest.approx(3.0)
    assert effects["severity"] == pytest.approx(-1.0)
    assert effects["data"] == pytest.approx(0.5)
    assert effects["noise:data"] == pytest.approx(1.5)
    assert effects["noise:severity"] == pytest.approx(0.0, abs=1.0e-12)
    assert result.design_rank == len(result.term_names)
    np.testing.assert_allclose(result.residuals, 0.0, atol=1.0e-12)


def test_replicated_full_factorial_is_supported() -> None:
    base = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=2)),
        dtype=np.float64,
    )
    design = np.repeat(base, 2, axis=0)
    response = 2.0 + design[:, 0] - 2.0 * design[:, 1]

    result = estimate_two_level_factorial_effects(
        design,
        response,
        ("first", "second"),
    )

    np.testing.assert_allclose(result.fitted_responses, response)


@pytest.mark.parametrize(
    ("design", "response"),
    [
        (np.asarray([[-1.0], [0.0]]), np.ones(2)),
        (np.asarray([[-1.0], [1.0]]), np.ones(1)),
        (np.asarray([[-1.0, -1.0], [1.0, 1.0]]), np.ones(2)),
    ],
)
def test_invalid_factorial_inputs_are_rejected(
    design: np.ndarray,
    response: np.ndarray,
) -> None:
    with pytest.raises(FactorialAnalysisError):
        estimate_two_level_factorial_effects(
            design,
            response,
            tuple(f"factor_{index}" for index in range(design.shape[1])),
        )
