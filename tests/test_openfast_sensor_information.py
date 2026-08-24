"""Tests for Stage 24 high-fidelity sensor feature transforms."""

import numpy as np
import pytest

from probdt.openfast import (
    OpenFASTTrackingError,
    band_log_energy_ratio,
    fit_linear_damage_state_library,
    log_spectral_ratio,
    paired_log_transmissibility,
)


def test_log_spectral_and_band_ratios_recover_known_scale() -> None:
    frequencies = np.asarray([0.1, 0.2, 0.3, 0.4])
    reference = np.ones((4, 2))
    case = np.full((3, 4, 2), 2.0)

    spectral = log_spectral_ratio(case, reference)
    bands = band_log_energy_ratio(
        frequencies,
        case,
        reference,
        ((0.1, 0.2), (0.3, 0.4)),
    )

    assert spectral.shape == (3, 4, 2)
    assert bands.shape == (3, 2, 2)
    assert np.allclose(spectral, np.log(2.0))
    assert np.allclose(bands, np.log(2.0))


def test_transmissibility_is_a_paired_log_difference() -> None:
    features = np.asarray([[[np.log(4.0), np.log(2.0), np.log(8.0)]]])

    values = paired_log_transmissibility(features, ((0, 1), (2, 0)))

    assert values.shape == (1, 1, 2)
    assert values[0, 0, 0] == pytest.approx(np.log(2.0))
    assert values[0, 0, 1] == pytest.approx(np.log(2.0))


def test_linear_damage_library_scales_zone_means() -> None:
    healthy = np.zeros((4, 2))
    damaged = np.asarray(
        [[3.0, 6.0], [3.0, 6.0], [6.0, 3.0], [6.0, 3.0]]
    )
    zones = ("healthy", "lower", "lower", "upper", "upper")
    losses = np.asarray([0.0, 0.1, 0.3, 0.1, 0.3])

    fitted = fit_linear_damage_state_library(
        healthy,
        damaged,
        ("lower", "lower", "upper", "upper"),
        zones,
        losses,
        training_loss_fraction=0.3,
        minimum_feature_variance=0.01,
    )

    assert np.array_equal(fitted.templates[0], np.zeros(2))
    assert np.allclose(fitted.templates[1], [1.0, 2.0])
    assert np.allclose(fitted.templates[2], [3.0, 6.0])
    assert np.allclose(fitted.templates[3], [2.0, 1.0])
    assert np.allclose(fitted.templates[4], [6.0, 3.0])
    assert np.all(fitted.feature_variance >= 0.01)


def test_band_features_reject_overlapping_bands() -> None:
    with pytest.raises(OpenFASTTrackingError, match="overlapping"):
        band_log_energy_ratio(
            np.asarray([0.1, 0.2, 0.3]),
            np.ones((1, 3, 1)),
            np.ones((3, 1)),
            ((0.1, 0.2), (0.2, 0.3)),
        )
