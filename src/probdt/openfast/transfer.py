"""Healthy-only calibration of a reduced spectral twin to OpenFAST data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.signal import welch

from probdt.diagnostics import acceleration_auto_psd
from probdt.structural import apply_stiffness_multipliers

FloatArray = NDArray[np.float64]


class OpenFASTTransferError(ValueError):
    """Raised when reduced-twin transfer inputs are invalid."""


@dataclass(frozen=True)
class ReducedTwinCalibration:
    """Frozen nuisance calibration for two directional top-DOF spectra."""

    mass_scale: float
    damping_scale: float
    force_standard_deviation_scales: tuple[float, float]
    force_correlation_times_s: tuple[float, float]

    @classmethod
    def from_parameter_vector(
        cls,
        parameters: ArrayLike,
    ) -> ReducedTwinCalibration:
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise OpenFASTTransferError(
                "Reduced-twin calibration requires six finite parameters."
            )
        if np.any(values <= 0.0):
            raise OpenFASTTransferError("Calibration parameters must be positive.")
        return cls(
            mass_scale=float(values[0]),
            damping_scale=float(values[1]),
            force_standard_deviation_scales=(
                float(values[2]),
                float(values[4]),
            ),
            force_correlation_times_s=(
                float(values[3]),
                float(values[5]),
            ),
        )

    def as_parameter_vector(self) -> FloatArray:
        """Return parameters in the fixed optimization order."""

        return np.asarray(
            [
                self.mass_scale,
                self.damping_scale,
                self.force_standard_deviation_scales[0],
                self.force_correlation_times_s[0],
                self.force_standard_deviation_scales[1],
                self.force_correlation_times_s[1],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ReducedTwinCalibrationResult:
    """Numerical result and reproducible fit diagnostics."""

    calibration: ReducedTwinCalibration
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    function_evaluations: int
    initial_log_spectral_rmse: float
    calibrated_log_spectral_rmse: float
    minimum_bound_margin_fraction: float


def averaged_welch_auto_psd(
    time_s: ArrayLike,
    values: ArrayLike,
    *,
    start_time_s: float,
    segment_samples: int,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
) -> tuple[FloatArray, FloatArray]:
    """Estimate channel-wise auto-PSDs on one common Welch grid."""

    time = np.asarray(time_s, dtype=np.float64)
    observations = np.asarray(values, dtype=np.float64)
    if time.ndim != 1 or observations.shape[0] != time.size:
        raise OpenFASTTransferError("Time and observations have incompatible shapes.")
    if observations.ndim != 2 or observations.shape[1] < 1:
        raise OpenFASTTransferError("Observations must contain one or more channels.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(observations)):
        raise OpenFASTTransferError("Time and observations must be finite.")
    increments = np.diff(time)
    if np.any(increments <= 0.0):
        raise OpenFASTTransferError("Time must be strictly increasing.")
    time_step = float(np.median(increments))
    relative_error = float(
        np.max(np.abs(increments - time_step)) / max(time_step, 1.0e-15)
    )
    if relative_error > 1.0e-10:
        raise OpenFASTTransferError("Welch estimation requires a uniform time grid.")
    if segment_samples < 8:
        raise OpenFASTTransferError("segment_samples must be at least eight.")
    if not 0.0 < minimum_frequency_hz < maximum_frequency_hz:
        raise OpenFASTTransferError("The frequency band must be positive and ordered.")

    retained = time >= float(start_time_s)
    if int(np.sum(retained)) < segment_samples:
        raise OpenFASTTransferError("Too few post-transient samples for Welch PSD.")
    spectra = []
    frequencies: FloatArray | None = None
    for channel_index in range(observations.shape[1]):
        current_frequency, current_density = welch(
            observations[retained, channel_index],
            fs=1.0 / time_step,
            window="hann",
            nperseg=segment_samples,
            noverlap=segment_samples // 2,
            detrend="constant",
            scaling="density",
        )
        if frequencies is None:
            frequencies = np.asarray(current_frequency, dtype=np.float64)
        spectra.append(np.asarray(current_density, dtype=np.float64))
    if frequencies is None:
        raise OpenFASTTransferError("No Welch spectra were calculated.")
    frequency_tolerance = 1.0e-9 * maximum_frequency_hz
    band = (frequencies >= minimum_frequency_hz - frequency_tolerance) & (
        frequencies <= maximum_frequency_hz + frequency_tolerance
    )
    density = np.column_stack(spectra)[band]
    frequencies = frequencies[band]
    if frequencies.size < 5 or np.any(density <= 0.0):
        raise OpenFASTTransferError("The retained Welch spectra are invalid.")
    return frequencies, density


def reduced_twin_directional_psd(
    mass_matrix_kg: ArrayLike,
    damping_matrix_n_s_per_m: ArrayLike,
    zone_stiffness_components_n_per_m: ArrayLike,
    stiffness_multipliers: ArrayLike,
    frequencies_hz: ArrayLike,
    force_distribution: ArrayLike,
    *,
    top_dof_index: int,
    baseline_force_standard_deviation_n: float,
    calibration: ReducedTwinCalibration,
) -> FloatArray:
    """Predict two directional spectra from one shared reduced structure."""

    mass = np.asarray(mass_matrix_kg, dtype=np.float64)
    damping = np.asarray(damping_matrix_n_s_per_m, dtype=np.float64)
    components = np.asarray(
        zone_stiffness_components_n_per_m,
        dtype=np.float64,
    )
    stiffness = apply_stiffness_multipliers(components, stiffness_multipliers)
    channel_spectra = []
    for force_scale, correlation_time in zip(
        calibration.force_standard_deviation_scales,
        calibration.force_correlation_times_s,
        strict=True,
    ):
        spectrum = acceleration_auto_psd(
            calibration.mass_scale * mass,
            calibration.damping_scale * damping,
            stiffness,
            frequencies_hz,
            force_distribution,
            (top_dof_index,),
            force_standard_deviation_n=(
                baseline_force_standard_deviation_n * force_scale
            ),
            force_correlation_time_s=correlation_time,
        )
        channel_spectra.append(spectrum[:, 0])
    return np.column_stack(channel_spectra)


def log_spectral_rmse(observed_psd: ArrayLike, predicted_psd: ArrayLike) -> float:
    """Return root-mean-square natural-log spectral error."""

    observed = np.asarray(observed_psd, dtype=np.float64)
    predicted = np.asarray(predicted_psd, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise OpenFASTTransferError("Observed and predicted PSDs must align.")
    if np.any(observed <= 0.0) or np.any(predicted <= 0.0):
        raise OpenFASTTransferError("Spectra must be strictly positive.")
    return float(np.sqrt(np.mean((np.log(predicted) - np.log(observed)) ** 2)))


def calibrate_reduced_twin_nuisance(
    observed_psd: ArrayLike,
    forward_model,
    *,
    initial_parameters: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    maximum_function_evaluations: int,
    robust_loss: str = "soft_l1",
    robust_scale: float = 0.5,
) -> ReducedTwinCalibrationResult:
    """Fit six nuisance parameters to a healthy log spectrum only."""

    observed = np.asarray(observed_psd, dtype=np.float64)
    initial = np.asarray(initial_parameters, dtype=np.float64)
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[1] != 2 or np.any(observed <= 0.0):
        message = "Healthy observed PSD must be positive and two-channel."
        raise OpenFASTTransferError(message)
    if initial.shape != (6,) or lower.shape != (6,) or upper.shape != (6,):
        raise OpenFASTTransferError("Calibration bounds must contain six entries.")
    if np.any(lower <= 0.0) or np.any(lower >= upper):
        raise OpenFASTTransferError("Calibration bounds must be positive and ordered.")
    if np.any(initial <= lower) or np.any(initial >= upper):
        raise OpenFASTTransferError("Initial parameters must lie inside the bounds.")
    if maximum_function_evaluations < 1:
        raise OpenFASTTransferError("maximum_function_evaluations must be positive.")

    observed_log = np.log(observed)

    def residual(parameters: FloatArray) -> FloatArray:
        calibration = ReducedTwinCalibration.from_parameter_vector(parameters)
        predicted = np.asarray(forward_model(calibration), dtype=np.float64)
        if predicted.shape != observed.shape or np.any(predicted <= 0.0):
            raise OpenFASTTransferError("The calibration forward model is invalid.")
        return (np.log(predicted) - observed_log).ravel()

    initial_rmse = float(np.sqrt(np.mean(residual(initial) ** 2)))
    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss=robust_loss,
        f_scale=float(robust_scale),
        x_scale="jac",
        max_nfev=int(maximum_function_evaluations),
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
    )
    calibrated_rmse = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    margins = np.minimum(
        (result.x - lower) / (upper - lower),
        (upper - result.x) / (upper - lower),
    )
    return ReducedTwinCalibrationResult(
        calibration=ReducedTwinCalibration.from_parameter_vector(result.x),
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        function_evaluations=int(result.nfev),
        initial_log_spectral_rmse=initial_rmse,
        calibrated_log_spectral_rmse=calibrated_rmse,
        minimum_bound_margin_fraction=float(np.min(margins)),
    )


def differential_log_spectral_signature(
    case_psd: ArrayLike,
    healthy_psd: ArrayLike,
) -> FloatArray:
    """Return flattened log spectral change relative to healthy response."""

    case = np.asarray(case_psd, dtype=np.float64)
    healthy = np.asarray(healthy_psd, dtype=np.float64)
    if case.shape != healthy.shape or case.ndim != 2:
        raise OpenFASTTransferError("Case and healthy spectra must align.")
    if np.any(case <= 0.0) or np.any(healthy <= 0.0):
        raise OpenFASTTransferError("Differential spectra must be positive.")
    return np.log(case / healthy).ravel()


def cosine_similarity(first: ArrayLike, second: ArrayLike) -> float:
    """Return bounded cosine similarity for two nonzero signatures."""

    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != second_values.shape or first_values.ndim != 1:
        raise OpenFASTTransferError("Signatures must be aligned vectors.")
    denominator = np.linalg.norm(first_values) * np.linalg.norm(second_values)
    if denominator <= 0.0:
        raise OpenFASTTransferError("Signatures must be nonzero.")
    return float(np.clip(np.dot(first_values, second_values) / denominator, -1.0, 1.0))
