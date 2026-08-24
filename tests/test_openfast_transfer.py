"""Tests for healthy-only OpenFAST-to-reduced-twin calibration utilities."""

import numpy as np
import pytest

from probdt.openfast import (
    OpenFASTTransferError,
    ReducedTwinCalibration,
    averaged_welch_auto_psd,
    calibrate_reduced_twin_nuisance,
    cosine_similarity,
    differential_log_spectral_signature,
)


def test_welch_psd_returns_common_positive_band() -> None:
    time = np.arange(0.0, 60.05, 0.05)
    values = np.column_stack(
        [
            np.sin(2.0 * np.pi * 0.3 * time),
            0.5 * np.sin(2.0 * np.pi * 0.6 * time),
        ]
    )

    frequencies, density = averaged_welch_auto_psd(
        time,
        values,
        start_time_s=10.0,
        segment_samples=400,
        minimum_frequency_hz=0.05,
        maximum_frequency_hz=2.0,
    )

    assert frequencies.shape == (40,)
    assert density.shape == (40, 2)
    assert np.all(density > 0.0)
    assert frequencies[np.argmax(density[:, 0])] == pytest.approx(0.3)
    assert frequencies[np.argmax(density[:, 1])] == pytest.approx(0.6)


def test_nuisance_calibration_recovers_synthetic_spectra() -> None:
    template = np.linspace(0.1, 1.0, 40)[:, None]
    observed = np.column_stack([2.5 * template[:, 0], 0.7 * template[:, 0]])

    def forward(calibration: ReducedTwinCalibration) -> np.ndarray:
        parameters = calibration.as_parameter_vector()
        return np.column_stack(
            [
                parameters[2] ** 2 * template[:, 0],
                parameters[4] ** 2 * template[:, 0],
            ]
        )

    result = calibrate_reduced_twin_nuisance(
        observed,
        forward,
        initial_parameters=np.ones(6),
        lower_bounds=np.full(6, 0.1),
        upper_bounds=np.full(6, 5.0),
        maximum_function_evaluations=500,
    )

    assert result.optimizer_success
    assert result.calibrated_log_spectral_rmse < 1.0e-8
    assert result.calibrated_log_spectral_rmse < result.initial_log_spectral_rmse


def test_differential_signature_and_cosine_similarity() -> None:
    healthy = np.ones((3, 2))
    case = np.exp(np.arange(6, dtype=float).reshape(3, 2) / 10.0)
    signature = differential_log_spectral_signature(case, healthy)

    assert signature == pytest.approx(np.arange(6) / 10.0)
    assert cosine_similarity(signature, signature) == pytest.approx(1.0)
    assert cosine_similarity(signature, -signature) == pytest.approx(-1.0)


def test_transfer_inputs_are_validated() -> None:
    with pytest.raises(OpenFASTTransferError, match="positive"):
        ReducedTwinCalibration.from_parameter_vector([1, 1, 0, 1, 1, 1])
    with pytest.raises(OpenFASTTransferError, match="nonzero"):
        cosine_similarity(np.zeros(3), np.ones(3))
