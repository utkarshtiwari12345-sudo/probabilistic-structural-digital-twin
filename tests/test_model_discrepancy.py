"""Tests for explicit Stage 8 truth-model discrepancies."""

import numpy as np
import pytest

from probdt.diagnostics import (
    ModelDiscrepancyError,
    SpectralModelDiscrepancy,
    acceleration_auto_psd,
    coherent_log_spectral_residual_scores,
    discrepant_acceleration_auto_psd,
)


def common_arguments() -> dict[str, object]:
    return {
        "mass_matrix_kg": [[4.0]],
        "damping_matrix_n_s_per_m": [[0.4]],
        "zone_stiffness_components_n_per_m": [[[100.0]]],
        "stiffness_multipliers": [0.9],
        "frequencies_hz": [0.5, 1.0],
        "force_distribution": [1.0],
        "sensor_dof_indices": (0,),
        "force_standard_deviation_n": 10.0,
        "force_correlation_time_s": 0.5,
    }


def test_identity_discrepancy_matches_nominal_spectrum() -> None:
    arguments = common_arguments()
    discrepant = discrepant_acceleration_auto_psd(
        **arguments,
        discrepancy=SpectralModelDiscrepancy(),
    )
    nominal = acceleration_auto_psd(
        arguments["mass_matrix_kg"],
        arguments["damping_matrix_n_s_per_m"],
        [[90.0]],
        arguments["frequencies_hz"],
        arguments["force_distribution"],
        arguments["sensor_dof_indices"],
        force_standard_deviation_n=10.0,
        force_correlation_time_s=0.5,
    )

    np.testing.assert_allclose(discrepant, nominal)


def test_excitation_and_sensor_gain_scale_signal_power_quadratically() -> None:
    arguments = common_arguments()
    baseline = discrepant_acceleration_auto_psd(
        **arguments,
        discrepancy=SpectralModelDiscrepancy(),
    )
    scaled = discrepant_acceleration_auto_psd(
        **arguments,
        discrepancy=SpectralModelDiscrepancy(
            excitation_standard_deviation_scale=2.0,
            sensor_gain_scales=(1.5,),
        ),
    )

    np.testing.assert_allclose(scaled, baseline * 4.0 * 2.25)


def test_sensor_gain_precedes_additive_noise() -> None:
    arguments = common_arguments()
    baseline = discrepant_acceleration_auto_psd(
        **arguments,
        discrepancy=SpectralModelDiscrepancy(),
    )
    gained = discrepant_acceleration_auto_psd(
        **arguments,
        discrepancy=SpectralModelDiscrepancy(sensor_gain_scales=(2.0,)),
        measurement_noise_psd=[3.0],
    )

    np.testing.assert_allclose(gained, 4.0 * baseline + 3.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass_scale": 0.0},
        {"damping_scale": np.nan},
        {"sensor_gain_scales": (1.0, -1.0)},
    ],
)
def test_invalid_discrepancy_scales_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ModelDiscrepancyError):
        SpectralModelDiscrepancy(**kwargs)


def test_sensor_gain_count_must_match_layout() -> None:
    with pytest.raises(ModelDiscrepancyError, match="one value per"):
        discrepant_acceleration_auto_psd(
            **common_arguments(),
            discrepancy=SpectralModelDiscrepancy(sensor_gain_scales=(1.0, 1.0)),
        )


def test_coherent_residual_accumulates_persistent_bias() -> None:
    observed = np.full((6, 2, 3), 0.2)
    fitted = np.zeros_like(observed)

    scores = coherent_log_spectral_residual_scores(observed, fitted, 0.04)

    np.testing.assert_allclose(scores, np.arange(1.0, 7.0))


def test_coherent_residual_rejects_invalid_inputs() -> None:
    observed = np.zeros((3, 4))

    with pytest.raises(ModelDiscrepancyError):
        coherent_log_spectral_residual_scores(observed, observed[:, :3], 1.0)
    with pytest.raises(ModelDiscrepancyError):
        coherent_log_spectral_residual_scores(observed, observed, 0.0)
    with pytest.raises(ModelDiscrepancyError):
        coherent_log_spectral_residual_scores(
            np.array([[np.nan, 0.0]]),
            np.zeros((1, 2)),
            1.0,
        )
