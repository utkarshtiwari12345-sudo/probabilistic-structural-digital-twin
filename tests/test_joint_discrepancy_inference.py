"""Tests for Stage 9 joint stiffness--nuisance spectral inference."""

import numpy as np
import pytest

from probdt.inference import (
    JOINT_NUISANCE_PARAMETER_NAMES,
    JointDiscrepancyInferenceError,
    joint_discrepancy_acceleration_auto_psd,
    joint_spectral_parameter_names,
    split_joint_spectral_parameters,
)


def test_joint_parameter_names_append_declared_nuisances() -> None:
    names = joint_spectral_parameter_names(("theta_lower", "theta_upper"))

    assert names == (
        "theta_lower",
        "theta_upper",
        *JOINT_NUISANCE_PARAMETER_NAMES,
    )


def test_joint_parameter_split_preserves_order() -> None:
    stiffness, discrepancy = split_joint_spectral_parameters(
        [0.9, 0.8, 1.05, 1.2, 0.95, 1.3],
        2,
    )

    np.testing.assert_allclose(stiffness, [0.9, 0.8])
    assert discrepancy.mass_scale == 1.05
    assert discrepancy.damping_scale == 1.2
    assert discrepancy.excitation_standard_deviation_scale == 0.95
    assert discrepancy.excitation_correlation_time_scale == 1.3


def test_joint_identity_spectrum_is_positive() -> None:
    spectrum = joint_discrepancy_acceleration_auto_psd(
        [[4.0]],
        [[0.4]],
        [[[100.0]]],
        [0.9, 1.0, 1.0, 1.0, 1.0],
        [0.5, 1.0],
        [1.0],
        (0,),
        force_standard_deviation_n=10.0,
        force_correlation_time_s=0.5,
        measurement_noise_psd=[0.01],
    )

    assert spectrum.shape == (2, 1)
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum > 0.0)


@pytest.mark.parametrize(
    ("parameters", "count"),
    [
        ([1.0, 1.0], 1),
        ([1.0, 1.0, 1.0, 1.0, -1.0], 1),
        ([1.0, 1.0, 1.0, 1.0, 1.0], 0),
    ],
)
def test_invalid_joint_parameters_are_rejected(
    parameters: list[float],
    count: int,
) -> None:
    with pytest.raises(JointDiscrepancyInferenceError):
        split_joint_spectral_parameters(parameters, count)
