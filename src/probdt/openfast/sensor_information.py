"""Feature transforms for high-fidelity OpenFAST virtual sensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probdt.openfast.tracking import OpenFASTTrackingError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EmpiricalDamageStateLibrary:
    """Linear-in-loss state templates and residual feature variance."""

    templates: FloatArray
    feature_variance: FloatArray
    zone_training_means: FloatArray


def log_spectral_ratio(case_psd: ArrayLike, reference_psd: ArrayLike) -> FloatArray:
    """Return elementwise log PSD ratios with strict positivity checks."""

    case = np.asarray(case_psd, dtype=np.float64)
    reference = np.asarray(reference_psd, dtype=np.float64)
    try:
        case, reference = np.broadcast_arrays(case, reference)
    except ValueError as error:
        raise OpenFASTTrackingError(
            "Case and reference PSDs cannot broadcast."
        ) from error
    if (
        np.any(case <= 0.0)
        or np.any(reference <= 0.0)
        or not np.all(np.isfinite(case))
        or not np.all(np.isfinite(reference))
    ):
        raise OpenFASTTrackingError("Spectral ratios require positive finite PSDs.")
    return np.log(case / reference)


def band_log_energy_ratio(
    frequencies_hz: ArrayLike,
    case_psd: ArrayLike,
    reference_psd: ArrayLike,
    bands_hz: Sequence[tuple[float, float]],
) -> FloatArray:
    """Aggregate window/channel PSD ratios over disjoint frequency bands.

    PSD arrays have shape ``(window, frequency, channel)``. The reference may
    have either that shape or ``(frequency, channel)``.
    """

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    case = np.asarray(case_psd, dtype=np.float64)
    reference = np.asarray(reference_psd, dtype=np.float64)
    if case.ndim != 3 or case.shape[1] != frequencies.size:
        raise OpenFASTTrackingError("Case PSDs must be window/frequency/channel.")
    if reference.ndim == 2:
        reference = np.broadcast_to(reference, case.shape)
    if reference.shape != case.shape:
        raise OpenFASTTrackingError("Reference PSDs must align with case PSDs.")
    if not bands_hz:
        raise OpenFASTTrackingError("At least one frequency band is required.")

    ratios = []
    occupied = np.zeros(frequencies.size, dtype=np.bool_)
    for lower, upper in bands_hz:
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise OpenFASTTrackingError("Frequency bands must be finite and ordered.")
        selected = (frequencies >= lower - 1.0e-12) & (
            frequencies <= upper + 1.0e-12
        )
        if not np.any(selected) or np.any(occupied & selected):
            raise OpenFASTTrackingError("Frequency bands are empty or overlapping.")
        occupied |= selected
        case_energy = np.mean(case[:, selected, :], axis=1)
        reference_energy = np.mean(reference[:, selected, :], axis=1)
        ratios.append(log_spectral_ratio(case_energy, reference_energy))
    return np.stack(ratios, axis=1)


def paired_log_transmissibility(
    channel_log_features: ArrayLike,
    channel_pairs: Sequence[tuple[int, int]],
) -> FloatArray:
    """Subtract paired channel log features to form transmissibilities."""

    features = np.asarray(channel_log_features, dtype=np.float64)
    if features.ndim != 3 or not np.all(np.isfinite(features)):
        raise OpenFASTTrackingError(
            "Channel features must be finite window/band/channel values."
        )
    if not channel_pairs:
        raise OpenFASTTrackingError("At least one transmissibility pair is required.")
    values = []
    for numerator, denominator in channel_pairs:
        indices_valid = (
            0 <= numerator < features.shape[2]
            and 0 <= denominator < features.shape[2]
        )
        if not indices_valid:
            raise OpenFASTTrackingError("A transmissibility channel index is invalid.")
        if numerator == denominator:
            raise OpenFASTTrackingError("Transmissibility channels must differ.")
        values.append(features[:, :, numerator] - features[:, :, denominator])
    return np.stack(values, axis=2)


def fit_linear_damage_state_library(
    healthy_features: ArrayLike,
    damaged_features: ArrayLike,
    damaged_feature_zones: Sequence[str],
    state_zones: Sequence[str],
    state_losses: ArrayLike,
    *,
    training_loss_fraction: float,
    minimum_feature_variance: float,
) -> EmpiricalDamageStateLibrary:
    """Fit per-zone severe means and scale them over discrete loss states."""

    healthy = np.asarray(healthy_features, dtype=np.float64)
    damaged = np.asarray(damaged_features, dtype=np.float64)
    feature_zones = np.asarray([str(value) for value in damaged_feature_zones])
    zones = tuple(str(value) for value in state_zones)
    losses = np.asarray(state_losses, dtype=np.float64)
    if healthy.ndim != 2 or damaged.ndim != 2 or healthy.shape[1] != damaged.shape[1]:
        raise OpenFASTTrackingError("Healthy and damaged features must align.")
    if feature_zones.shape != (damaged.shape[0],):
        raise OpenFASTTrackingError("Damaged feature labels must align.")
    if losses.shape != (len(zones),) or int(np.sum(np.isclose(losses, 0.0))) != 1:
        raise OpenFASTTrackingError("The state metadata is invalid.")
    if (
        not np.all(np.isfinite(healthy))
        or not np.all(np.isfinite(damaged))
        or training_loss_fraction <= 0.0
        or minimum_feature_variance <= 0.0
    ):
        raise OpenFASTTrackingError(
            "State-library fit inputs must be finite and positive."
        )

    healthy_index = int(np.flatnonzero(np.isclose(losses, 0.0))[0])
    damaged_zones = tuple(
        dict.fromkeys(
            zone
            for index, (zone, loss) in enumerate(zip(zones, losses, strict=True))
            if index != healthy_index and loss > 0.0
        )
    )
    zone_means = []
    residuals = [healthy]
    for zone in damaged_zones:
        selected = damaged[feature_zones == zone]
        if selected.shape[0] < 2:
            raise OpenFASTTrackingError(
                "Every zone needs two or more training features."
            )
        mean = np.mean(selected, axis=0)
        zone_means.append(mean)
        residuals.append(selected - mean)

    templates = np.zeros((len(zones), healthy.shape[1]), dtype=np.float64)
    for state_index, (zone, loss) in enumerate(zip(zones, losses, strict=True)):
        if state_index == healthy_index:
            continue
        try:
            zone_index = damaged_zones.index(zone)
        except ValueError as error:
            raise OpenFASTTrackingError(
                "A state zone has no training features."
            ) from error
        templates[state_index] = (
            zone_means[zone_index] * loss / training_loss_fraction
        )
    residual_matrix = np.concatenate(residuals, axis=0)
    variance = np.maximum(
        np.var(residual_matrix, axis=0, ddof=1),
        minimum_feature_variance,
    )
    return EmpiricalDamageStateLibrary(
        templates=templates,
        feature_variance=variance,
        zone_training_means=np.stack(zone_means),
    )
