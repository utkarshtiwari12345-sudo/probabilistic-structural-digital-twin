"""Tests for Stage 15 posterior uncertainty calibration."""

import numpy as np
import pytest

from probdt.inference import (
    UNCERTAINTY_CALIBRATION_METHODS,
    UncertaintyCalibrationError,
    calibrated_standard_deviations,
    fit_uncertainty_calibration,
)


def calibration_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.asarray(
        [
            [0.98, 0.99],
            [0.92, 0.97],
            [0.86, 0.96],
            [0.82, 0.95],
        ]
    )
    truth = np.asarray(
        [
            [1.00, 1.00],
            [0.96, 1.00],
            [0.90, 1.00],
            [0.85, 1.00],
        ]
    )
    standard_deviation = np.full_like(means, 0.005)
    return means, standard_deviation, truth


def test_calibration_inflates_undercovered_intervals() -> None:
    means, raw_scale, truth = calibration_data()

    calibration = fit_uncertainty_calibration(
        means,
        raw_scale,
        truth,
        nominal_coverage=0.95,
    )

    assert calibration.scalar_inflation > 1.0
    assert np.all(calibration.zonewise_inflation > 1.0)
    assert np.all(calibration.additive_model_error_standard_deviation > 0.0)
    for method in UNCERTAINTY_CALIBRATION_METHODS:
        adjusted = calibrated_standard_deviations(raw_scale, method, calibration)
        assert adjusted.shape == raw_scale.shape
        assert np.all(adjusted >= raw_scale)


def test_calibration_methods_are_deterministic_and_keep_raw_input_immutable() -> None:
    means, raw_scale, truth = calibration_data()
    original = raw_scale.copy()
    first = fit_uncertainty_calibration(means, raw_scale, truth)
    second = fit_uncertainty_calibration(means, raw_scale, truth)

    np.testing.assert_array_equal(first.zonewise_inflation, second.zonewise_inflation)
    np.testing.assert_array_equal(
        first.additive_model_error_standard_deviation,
        second.additive_model_error_standard_deviation,
    )
    np.testing.assert_array_equal(raw_scale, original)


def test_invalid_calibration_inputs_are_rejected() -> None:
    means, raw_scale, truth = calibration_data()
    with pytest.raises(UncertaintyCalibrationError, match="nominal_coverage"):
        fit_uncertainty_calibration(
            means,
            raw_scale,
            truth,
            nominal_coverage=1.0,
        )
    with pytest.raises(UncertaintyCalibrationError, match="Unknown"):
        calibration = fit_uncertainty_calibration(means, raw_scale, truth)
        calibrated_standard_deviations(raw_scale, "unknown", calibration)
