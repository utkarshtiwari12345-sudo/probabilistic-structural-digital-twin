"""Run the Stage 13 multirate joint state--parameter estimation study."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from run_robustness_factorial_study import _load_yaml  # noqa: E402
from run_sequential_tracking_study import (  # noqa: E402
    build_spectral_system,
    first_persistent_alarm_indices,
)

from probdt.deterioration import smooth_gradual_loss  # noqa: E402
from probdt.inference import (  # noqa: E402
    AugmentedStructuralModel,
    AugmentedTruth,
    JointStateFilterHistory,
    run_augmented_ensemble_filter,
    run_augmented_unscented_filter,
    simulate_augmented_truth,
)
from probdt.monitoring import SensorLayout  # noqa: E402

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
JOINT_STATE_CONFIG_PATH = PROJECT_ROOT / "config" / "joint_state_parameter.yaml"

RUNS_PATH = PROJECT_ROOT / "results" / "metrics" / "stage13_joint_state_runs.csv"
SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage13_joint_state_summary.csv"
)
MONITORING_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage13_monitoring_trajectory.csv"
)
FAST_PATH = PROJECT_ROOT / "results" / "metrics" / "stage13_fast_state.csv"
CALIBRATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage13_alarm_calibration.csv"
)
OBSERVABILITY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage13_observability.csv"
)
CHECKS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage13_hypothesis_checks.csv"
)
RUNTIME_PATH = PROJECT_ROOT / "results" / "metrics" / "stage13_runtime.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage13_joint_state_parameter.png"


@dataclass(frozen=True)
class MonitoringDataset:
    """Shared truth and observations for one slow monitoring history."""

    truth_windows: tuple[AugmentedTruth, ...]
    observations: tuple[np.ndarray, ...]
    dropout_fraction: float


@dataclass(frozen=True)
class MultirateFilterResult:
    """One method's fast-state and slow-parameter outputs."""

    monitoring_means: np.ndarray
    monitoring_covariances: np.ndarray
    loss_probabilities: np.ndarray
    displacement_nrmse: float
    velocity_nrmse: float
    force_nrmse: float
    acceleration_nrmse: float
    displacement_coverage: float
    stiffness_rmse: float
    stiffness_coverage: float
    minimum_scaled_covariance_eigenvalue: float
    last_window_history: JointStateFilterHistory
    runtime_seconds: float


def _calibrate_thresholds_allow_failure(
    probability_histories: dict[str, list[np.ndarray]],
    candidate_thresholds: list[float],
    persistence_windows: int,
    maximum_false_alarm_fraction: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Calibrate alarms while retaining methods that miss the healthy target."""

    candidates = [float(value) for value in candidate_thresholds]
    if not candidates or candidates != sorted(candidates):
        raise ValueError("Alarm candidates must be nonempty and increasing.")
    selected: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for method, histories in probability_histories.items():
        chosen: float | None = None
        method_rows: list[dict[str, object]] = []
        for threshold in candidates:
            alarms = [
                first_persistent_alarm_indices(
                    probabilities,
                    threshold,
                    persistence_windows,
                )
                for probabilities in histories
            ]
            false_alarm_fraction = float(np.mean(np.stack(alarms) >= 0))
            passed = false_alarm_fraction <= maximum_false_alarm_fraction
            method_rows.append(
                {
                    "method": method,
                    "candidate_probability_threshold": threshold,
                    "healthy_false_alarm_fraction": false_alarm_fraction,
                    "passes_calibration": passed,
                }
            )
            if chosen is None and passed:
                chosen = threshold
        calibration_succeeded = chosen is not None
        if chosen is None:
            chosen = candidates[-1]
        selected[method] = chosen
        for row in method_rows:
            row["selected"] = bool(
                np.isclose(row["candidate_probability_threshold"], chosen)
            )
            row["calibration_succeeded"] = calibration_succeeded
            rows.append(row)
    return selected, pd.DataFrame(rows)


def _state_scales(
    model: AugmentedStructuralModel,
    configuration: dict[str, object],
) -> np.ndarray:
    """Build the physical scaling vector used by both nonlinear filters."""

    return np.asarray(
        [
            *([float(configuration["displacement_m"])] * model.n_dof),
            *([float(configuration["velocity_m_per_s"])] * model.n_dof),
            float(configuration["latent_force_n"]),
            *(
                [float(configuration["stiffness_multiplier"])]
                * model.n_zones
            ),
        ],
        dtype=np.float64,
    )


def _initial_covariance(
    model: AugmentedStructuralModel,
    configuration: dict[str, object],
    theta_covariance: np.ndarray,
) -> np.ndarray:
    """Reset fast-state uncertainty while retaining slow-parameter belief."""

    variances = np.zeros(model.n_state, dtype=np.float64)
    variances[: model.n_dof] = float(
        configuration["displacement_standard_deviation_m"]
    ) ** 2
    variances[model.n_dof : 2 * model.n_dof] = float(
        configuration["velocity_standard_deviation_m_per_s"]
    ) ** 2
    variances[model.load_index] = float(
        configuration["latent_force_standard_deviation_n"]
    ) ** 2
    covariance = np.diag(variances)
    covariance[model.theta_slice, model.theta_slice] = theta_covariance
    return covariance


def _process_covariance(
    model: AugmentedStructuralModel,
    configuration: dict[str, object],
    time_step_s: float,
    force_standard_deviation_n: float,
) -> np.ndarray:
    """Build per-sample state noise including exact OU innovation variance."""

    variances = np.zeros(model.n_state, dtype=np.float64)
    variances[: model.n_dof] = float(
        configuration["displacement_standard_deviation_per_step_m"]
    ) ** 2
    variances[model.n_dof : 2 * model.n_dof] = float(
        configuration["velocity_standard_deviation_per_step_m_per_s"]
    ) ** 2
    correlation = np.exp(-time_step_s / model.force_correlation_time_s)
    variances[model.load_index] = force_standard_deviation_n**2 * (
        1.0 - correlation**2
    )
    variances[model.theta_slice] = float(
        configuration["stiffness_standard_deviation_per_fast_step"]
    ) ** 2
    return np.diag(variances)


def _loss_probabilities(
    means: np.ndarray,
    covariances: np.ndarray,
    theta_slice: slice,
    theta_bounds: np.ndarray,
    loss_threshold: float,
) -> np.ndarray:
    """Return truncated-normal loss probabilities at monitoring windows."""

    theta_mean = means[:, theta_slice]
    theta_variance = np.diagonal(
        covariances[:, theta_slice, theta_slice],
        axis1=1,
        axis2=2,
    )
    theta_scale = np.sqrt(np.maximum(theta_variance, 1.0e-16))
    lower = theta_bounds[:, 0]
    upper = theta_bounds[:, 1]
    threshold = 1.0 - loss_threshold
    lower_cdf = norm.cdf((lower - theta_mean) / theta_scale)
    upper_cdf = norm.cdf((upper - theta_mean) / theta_scale)
    threshold_cdf = norm.cdf((threshold - theta_mean) / theta_scale)
    normalization = np.maximum(upper_cdf - lower_cdf, 1.0e-15)
    return np.clip((threshold_cdf - lower_cdf) / normalization, 0.0, 1.0)


def _generate_dataset(
    truth_model: AugmentedStructuralModel,
    theta_history: np.ndarray,
    sensor_layout: SensorLayout,
    noise_standard_deviation: np.ndarray,
    dropout_probability: float,
    *,
    n_samples: int,
    burn_in_samples: int,
    time_step_s: float,
    force_standard_deviation_n: float,
    base_seed: int,
    replicate: int,
) -> MonitoringDataset:
    """Generate shared multirate truth, noise, and availability masks."""

    truth_windows: list[AugmentedTruth] = []
    observations: list[np.ndarray] = []
    total_values = 0
    missing_values = 0
    for window, theta in enumerate(theta_history):
        truth_rng = np.random.default_rng(
            np.random.SeedSequence([base_seed, replicate, window, 0])
        )
        noise_rng = np.random.default_rng(
            np.random.SeedSequence([base_seed, replicate, window, 1])
        )
        dropout_rng = np.random.default_rng(
            np.random.SeedSequence([base_seed, replicate, window, 2])
        )
        truth = simulate_augmented_truth(
            truth_model,
            theta,
            n_samples=n_samples,
            time_step_s=time_step_s,
            force_stationary_standard_deviation_n=force_standard_deviation_n,
            burn_in_samples=burn_in_samples,
            rng=truth_rng,
        )
        clean = truth.acceleration_m_per_s2[:, list(sensor_layout.dof_indices)]
        observed = clean + noise_rng.normal(
            scale=noise_standard_deviation,
            size=clean.shape,
        )
        if dropout_probability > 0.0:
            missing = dropout_rng.random(clean.shape) < dropout_probability
            observed[missing] = np.nan
            missing_values += int(np.sum(missing))
        total_values += observed.size
        truth_windows.append(truth)
        observations.append(observed)
    return MonitoringDataset(
        truth_windows=tuple(truth_windows),
        observations=tuple(observations),
        dropout_fraction=missing_values / total_values,
    )


def _run_multirate_filter(
    method_name: str,
    method_configuration: dict[str, object],
    filter_model: AugmentedStructuralModel,
    dataset: MonitoringDataset,
    theta_truth: np.ndarray,
    theta_bounds: np.ndarray,
    sensor_layout: SensorLayout,
    noise_covariance: np.ndarray,
    state_scales: np.ndarray,
    initial_configuration: dict[str, object],
    process_configuration: dict[str, object],
    *,
    time_step_s: float,
    force_standard_deviation_n: float,
    metric_exclusion_samples: int,
    loss_threshold: float,
    rng_seed: int,
) -> MultirateFilterResult:
    """Run one augmented method across reset fast windows and carried theta."""

    started = time.perf_counter()
    n_windows = len(dataset.observations)
    theta_mean = np.asarray(
        initial_configuration["stiffness_initial_mean"],
        dtype=np.float64,
    )
    theta_scale = np.asarray(
        initial_configuration["stiffness_standard_deviation"],
        dtype=np.float64,
    )
    theta_covariance = np.diag(theta_scale**2)
    monitoring_process_scale = np.asarray(
        process_configuration[
            "stiffness_standard_deviation_per_monitoring_window"
        ],
        dtype=np.float64,
    )
    monitoring_means = np.empty(
        (n_windows, filter_model.n_state),
        dtype=np.float64,
    )
    monitoring_covariances = np.empty(
        (n_windows, filter_model.n_state, filter_model.n_state),
        dtype=np.float64,
    )
    q_error_square = q_truth_square = 0.0
    v_error_square = v_truth_square = 0.0
    force_error_square = force_truth_square = 0.0
    acceleration_error_square = acceleration_truth_square = 0.0
    displacement_covered = 0
    displacement_total = 0
    minimum_scaled_eigenvalue = np.inf
    last_history: JointStateFilterHistory | None = None
    process_covariance = _process_covariance(
        filter_model,
        process_configuration,
        time_step_s,
        force_standard_deviation_n,
    )
    ensemble_rng = np.random.default_rng(rng_seed)
    oracle = bool(method_configuration["oracle_stiffness"])

    for window, (observed, truth) in enumerate(
        zip(dataset.observations, dataset.truth_windows, strict=True)
    ):
        initial_mean = np.zeros(filter_model.n_state, dtype=np.float64)
        if oracle:
            theta_mean = theta_truth[window].copy()
            oracle_scale = float(
                initial_configuration["oracle_stiffness_standard_deviation"]
            )
            theta_covariance = np.eye(filter_model.n_zones) * oracle_scale**2
        elif window > 0:
            theta_covariance = theta_covariance + np.diag(
                monitoring_process_scale**2
            )
        initial_mean[filter_model.theta_slice] = theta_mean
        initial_covariance = _initial_covariance(
            filter_model,
            initial_configuration,
            theta_covariance,
        )
        common_arguments = {
            "model": filter_model,
            "observations": observed,
            "sensor_dof_indices": sensor_layout.dof_indices,
            "time_step_s": time_step_s,
            "theta_bounds": theta_bounds,
            "initial_mean": initial_mean,
            "initial_covariance": initial_covariance,
            "process_covariance": process_covariance,
            "observation_covariance": noise_covariance,
            "state_scales": state_scales,
        }
        if method_configuration["type"] == "ukf":
            history = run_augmented_unscented_filter(
                **common_arguments,
                alpha=float(method_configuration["alpha"]),
                beta=float(method_configuration["beta"]),
                kappa=float(method_configuration["kappa"]),
            )
        elif method_configuration["type"] == "enkf":
            history = run_augmented_ensemble_filter(
                **common_arguments,
                n_ensemble=int(method_configuration["ensemble_size"]),
                rng=ensemble_rng,
            )
        else:
            raise ValueError(f"Unknown Stage 13 filter type for {method_name}.")

        monitoring_means[window] = history.posterior_means[-1]
        monitoring_covariances[window] = history.posterior_covariances[-1]
        theta_mean = history.posterior_means[-1, filter_model.theta_slice].copy()
        theta_covariance = history.posterior_covariances[
            -1,
            filter_model.theta_slice,
            filter_model.theta_slice,
        ].copy()
        retained = slice(metric_exclusion_samples, None)
        estimate = history.posterior_means[retained]
        truth_state = truth.states[retained]
        q_error = estimate[:, : filter_model.n_dof] - truth_state[
            :, : filter_model.n_dof
        ]
        velocity_error = estimate[
            :, filter_model.n_dof : 2 * filter_model.n_dof
        ] - truth_state[:, filter_model.n_dof : 2 * filter_model.n_dof]
        force_error = (
            estimate[:, filter_model.load_index]
            - truth_state[:, filter_model.load_index]
        )
        q_error_square += float(np.sum(q_error**2))
        q_truth_square += float(
            np.sum(truth_state[:, : filter_model.n_dof] ** 2)
        )
        v_error_square += float(np.sum(velocity_error**2))
        v_truth_square += float(
            np.sum(
                truth_state[:, filter_model.n_dof : 2 * filter_model.n_dof]
                ** 2
            )
        )
        force_error_square += float(np.sum(force_error**2))
        force_truth_square += float(
            np.sum(truth_state[:, filter_model.load_index] ** 2)
        )
        estimated_acceleration = filter_model.acceleration(
            estimate,
            tuple(range(filter_model.n_dof)),
        )
        true_acceleration = truth.acceleration_m_per_s2[retained]
        acceleration_error_square += float(
            np.sum((estimated_acceleration - true_acceleration) ** 2)
        )
        acceleration_truth_square += float(np.sum(true_acceleration**2))
        displacement_standard_deviation = history.posterior_standard_deviations[
            retained,
            : filter_model.n_dof,
        ]
        displacement_covered += int(
            np.sum(np.abs(q_error) <= 1.96 * displacement_standard_deviation)
        )
        displacement_total += q_error.size
        scaled_covariances = history.posterior_covariances / np.outer(
            state_scales,
            state_scales,
        )
        minimum_scaled_eigenvalue = min(
            minimum_scaled_eigenvalue,
            float(np.min(np.linalg.eigvalsh(scaled_covariances))),
        )
        last_history = history

    assert last_history is not None
    loss_probabilities = _loss_probabilities(
        monitoring_means,
        monitoring_covariances,
        filter_model.theta_slice,
        theta_bounds,
        loss_threshold,
    )
    theta_estimates = monitoring_means[:, filter_model.theta_slice]
    theta_standard_deviations = np.sqrt(
        np.diagonal(
            monitoring_covariances[
                :, filter_model.theta_slice, filter_model.theta_slice
            ],
            axis1=1,
            axis2=2,
        )
    )
    theta_covered = np.abs(theta_estimates - theta_truth) <= (
        1.96 * theta_standard_deviations
    )
    return MultirateFilterResult(
        monitoring_means=monitoring_means,
        monitoring_covariances=monitoring_covariances,
        loss_probabilities=loss_probabilities,
        displacement_nrmse=np.sqrt(q_error_square / q_truth_square),
        velocity_nrmse=np.sqrt(v_error_square / v_truth_square),
        force_nrmse=np.sqrt(force_error_square / force_truth_square),
        acceleration_nrmse=np.sqrt(
            acceleration_error_square / acceleration_truth_square
        ),
        displacement_coverage=displacement_covered / displacement_total,
        stiffness_rmse=float(
            np.sqrt(np.mean((theta_estimates - theta_truth) ** 2))
        ),
        stiffness_coverage=float(np.mean(theta_covered)),
        minimum_scaled_covariance_eigenvalue=minimum_scaled_eigenvalue,
        last_window_history=last_history,
        runtime_seconds=time.perf_counter() - started,
    )


def _decision_metrics(
    result: MultirateFilterResult,
    theta_truth: np.ndarray,
    threshold: float,
    loss_threshold: float,
    persistence: int,
) -> dict[str, object]:
    """Compute monitoring-window damage decisions after frozen calibration."""

    crossing = int(
        np.flatnonzero(1.0 - theta_truth[:, 0] >= loss_threshold)[0]
    )
    relative_alarm = first_persistent_alarm_indices(
        result.loss_probabilities[crossing:, [0]],
        threshold,
        persistence,
    )[0]
    alarm = int(crossing + relative_alarm) if relative_alarm >= 0 else -1
    pre_damage_alarms = first_persistent_alarm_indices(
        result.loss_probabilities[:crossing],
        threshold,
        persistence,
    )
    localized = int(np.argmax(result.loss_probabilities[-1])) == 0
    return {
        "detected": bool(alarm >= 0),
        "truth_crossing_window": crossing,
        "alarm_window": alarm,
        "detection_delay_windows": (
            float(alarm - crossing) if alarm >= 0 else np.nan
        ),
        "localized": bool(localized),
        "predamage_false_alarm_fraction": float(
            np.mean(pre_damage_alarms >= 0)
        ),
    }


def _observability_analysis(
    model: AugmentedStructuralModel,
    reference_state: np.ndarray,
    state_scales: np.ndarray,
    sensor_layout: SensorLayout,
    *,
    time_step_s: float,
    horizon_steps: int,
    finite_difference_step: float,
    rank_threshold: float,
) -> pd.DataFrame:
    """Compute a scaled finite-horizon nonlinear observability Jacobian."""

    reference_scaled = reference_state / state_scales

    def features(scaled_state: np.ndarray) -> np.ndarray:
        state = scaled_state * state_scales
        values: list[np.ndarray] = []
        for _ in range(horizon_steps):
            values.append(model.acceleration(state, sensor_layout.dof_indices))
            state = model.propagate_rk4(state, time_step_s)
        return np.concatenate(values)

    jacobian = np.empty(
        (horizon_steps * sensor_layout.n_sensors, model.n_state),
        dtype=np.float64,
    )
    for index in range(model.n_state):
        step = finite_difference_step * max(abs(reference_scaled[index]), 1.0)
        positive = reference_scaled.copy()
        negative = reference_scaled.copy()
        positive[index] += step
        negative[index] -= step
        jacobian[:, index] = (features(positive) - features(negative)) / (
            2.0 * step
        )
    column_norm = np.linalg.norm(jacobian, axis=0)
    normalized = jacobian / np.maximum(column_norm, 1.0e-15)
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    effective_rank = int(
        np.sum(singular_values >= rank_threshold * singular_values[0])
    )
    condition_number = float(singular_values[0] / singular_values[-1])
    return pd.DataFrame(
        {
            "singular_value_index": np.arange(1, len(singular_values) + 1),
            "normalized_singular_value": singular_values,
            "effective_rank": effective_rank,
            "n_augmented_states": model.n_state,
            "normalized_condition_number": condition_number,
            "minimum_column_norm": np.min(column_norm),
            "maximum_column_norm": np.max(column_norm),
        }
    )


def _summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicate means and standard deviations by case and method."""

    metrics = [
        "displacement_nrmse",
        "velocity_nrmse",
        "force_nrmse",
        "acceleration_nrmse",
        "displacement_coverage",
        "stiffness_rmse",
        "stiffness_coverage",
        "detected",
        "localized",
        "predamage_false_alarm_fraction",
    ]
    rows: list[dict[str, object]] = []
    for (method, case), selected in runs.groupby(["method", "case"], sort=False):
        row: dict[str, object] = {
            "method": method,
            "case": case,
            "replicates": len(selected),
        }
        for metric in metrics:
            values = selected[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_standard_deviation"] = np.std(values, ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def _create_figure(
    summary: pd.DataFrame,
    monitoring: pd.DataFrame,
    fast_state: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    """Plot fast-state tracking, deterioration, errors, and decisions."""

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), constrained_layout=True)
    fast_axis, theta_axis, state_axis, decision_axis = axes.ravel()
    colors = {
        "oracle_ukf": "0.45",
        "joint_ukf": "#0072B2",
        "joint_enkf": "#D55E00",
    }
    fast_axis.plot(
        fast_state["time_s"],
        fast_state["true_top_displacement_m"],
        color="black",
        linewidth=1.6,
        label="Truth",
    )
    for method, color in colors.items():
        fast_axis.plot(
            fast_state["time_s"],
            fast_state[f"estimated_top_displacement_m_{method}"],
            color=color,
            linewidth=1.0,
            label=method.replace("_", " "),
        )
    fast_axis.set_xlabel("Time within final stressed window (s)")
    fast_axis.set_ylabel("Top displacement (m)")
    fast_axis.set_title("Sparse-acceleration fast-state reconstruction")
    fast_axis.legend(fontsize=8)
    fast_axis.grid(alpha=0.25)

    selected_monitoring = monitoring[
        (monitoring["case"] == "stressed")
        & (monitoring["replicate"] == 0)
        & (monitoring["zone"] == "lower")
    ]
    truth = selected_monitoring[
        selected_monitoring["method"] == "joint_ukf"
    ]
    theta_axis.plot(
        truth["time_day"],
        truth["true_theta"],
        color="black",
        linewidth=2.0,
        label="Truth",
    )
    for method in ("joint_ukf", "joint_enkf"):
        selected = selected_monitoring[selected_monitoring["method"] == method]
        theta_axis.plot(
            selected["time_day"],
            selected["posterior_mean"],
            color=colors[method],
            marker="o",
            markersize=3,
            label=method.replace("_", " "),
        )
        theta_axis.fill_between(
            selected["time_day"],
            selected["credible_lower"],
            selected["credible_upper"],
            color=colors[method],
            alpha=0.15,
        )
    theta_axis.axhline(0.95, color="black", linestyle=":", linewidth=1.0)
    theta_axis.set_xlabel("Monitoring time (day)")
    theta_axis.set_ylabel("Lower-zone stiffness multiplier")
    theta_axis.set_title("Carried slow-parameter posterior under stress")
    theta_axis.legend(fontsize=8)
    theta_axis.grid(alpha=0.25)

    methods = list(colors)
    positions = np.arange(len(methods), dtype=np.float64)
    width = 0.34
    case_styles = (
        (-width / 2.0, "matched", 1.0),
        (width / 2.0, "stressed", 0.6),
    )
    for offset, case, alpha in case_styles:
        selected = summary.set_index(["method", "case"])
        q_values = [
            selected.loc[(method, case), "displacement_nrmse_mean"]
            for method in methods
        ]
        v_values = [
            selected.loc[(method, case), "velocity_nrmse_mean"]
            for method in methods
        ]
        state_axis.bar(
            positions + offset,
            q_values,
            width=width,
            color=[colors[method] for method in methods],
            alpha=alpha,
            label=f"Displacement: {case}",
        )
        state_axis.scatter(
            positions + offset,
            v_values,
            marker="x",
            color="black",
            label=f"Velocity: {case}" if case == "matched" else None,
            zorder=3,
        )
    state_axis.set_xticks(
        positions,
        labels=[method.replace("_", " ") for method in methods],
        rotation=12,
    )
    state_axis.set_ylabel("Normalized RMSE")
    state_axis.set_title("Fast-state error: bars q, crosses velocity")
    state_axis.legend(fontsize=7)
    state_axis.grid(axis="y", alpha=0.25)

    joint = summary[summary["method"].isin(["joint_ukf", "joint_enkf"])]
    labels = [
        f"{row.method.replace('_', ' ')}\n{row.case}"
        for row in joint.itertuples(index=False)
    ]
    decision_positions = np.arange(len(joint), dtype=np.float64)
    decision_width = 0.25
    decision_axis.bar(
        decision_positions - decision_width,
        joint["detected_mean"],
        width=decision_width,
        color="#009E73",
        label="Detection",
    )
    decision_axis.bar(
        decision_positions,
        joint["localized_mean"],
        width=decision_width,
        color="#E69F00",
        label="Localisation",
    )
    decision_axis.bar(
        decision_positions + decision_width,
        joint["stiffness_coverage_mean"],
        width=decision_width,
        color="#56B4E9",
        label="Theta coverage",
    )
    decision_axis.set_xticks(
        decision_positions,
        labels=labels,
        rotation=12,
    )
    decision_axis.set_ylim(0.0, 1.05)
    decision_axis.set_ylabel("Fraction")
    decision_axis.set_title("Joint deterioration decisions and calibration")
    decision_axis.legend(fontsize=8)
    decision_axis.grid(axis="y", alpha=0.25)
    failed_methods = calibration.loc[
        calibration["selected"] & ~calibration["calibration_succeeded"],
        "method",
    ].tolist()
    if failed_methods:
        decision_axis.text(
            0.02,
            0.04,
            "Healthy alarm calibration failed at 0.99:\n"
            + ", ".join(name.replace("_", " ") for name in failed_methods),
            transform=decision_axis.transAxes,
            color="#A60000",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#A60000"},
        )

    figure.suptitle(
        "Stage 13: multirate joint structural-state and stiffness estimation",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main() -> None:
    """Execute observability, calibration, and joint state--parameter tests."""

    started = time.perf_counter()
    base_config = _load_yaml(BASE_CONFIG_PATH)
    monitoring_config = _load_yaml(MONITORING_CONFIG_PATH)
    configuration = _load_yaml(JOINT_STATE_CONFIG_PATH)
    study_config = configuration["study"]
    response_config = configuration["response_window"]
    monitoring_sequence = configuration["monitoring"]
    excitation_config = monitoring_config["excitation"]

    spectral_system = build_spectral_system(base_config)
    force_distribution = np.asarray(
        excitation_config["spatial_distribution"],
        dtype=np.float64,
    )
    filter_model = AugmentedStructuralModel(
        mass_matrix_kg=spectral_system.mass_matrix_kg,
        damping_matrix_n_s_per_m=spectral_system.damping_matrix_n_s_per_m,
        zone_stiffness_components_n_per_m=(
            spectral_system.zone_components_n_per_m
        ),
        force_distribution=force_distribution,
        force_correlation_time_s=float(excitation_config["correlation_time_s"]),
    )
    layout_specification = monitoring_config["sensor_layouts"][
        study_config["sensor_layout"]
    ]
    sensor_layout = SensorLayout(
        str(study_config["sensor_layout"]),
        tuple(
            int(index) - 1
            for index in layout_specification["dofs_one_based"]
        ),
    )
    time_step = float(response_config["time_step_s"])
    n_samples = round(float(response_config["duration_s"]) / time_step) + 1
    burn_in_samples = round(float(response_config["burn_in_s"]) / time_step)
    metric_exclusion_samples = round(
        float(response_config["metric_initialization_exclusion_s"])
        / time_step
    )
    force_standard_deviation = float(
        excitation_config["stationary_standard_deviation_n"]
    )
    theta_bounds = np.asarray([[0.65, 1.0]] * filter_model.n_zones)
    state_scales = _state_scales(filter_model, configuration["state_scaling"])

    n_windows = int(monitoring_sequence["number_of_windows"])
    time_days = np.arange(n_windows, dtype=np.float64) * float(
        monitoring_sequence["update_interval_days"]
    )
    loss = smooth_gradual_loss(
        time_days * 86400.0,
        onset_s=float(monitoring_sequence["deterioration_onset_day"]) * 86400.0,
        completion_s=(
            float(monitoring_sequence["deterioration_completion_day"])
            * 86400.0
        ),
        final_loss_fraction=float(
            monitoring_sequence["final_lower_zone_loss_fraction"]
        ),
    )
    damaged_theta = np.ones((n_windows, filter_model.n_zones), dtype=np.float64)
    damaged_theta[:, 0] = 1.0 - loss
    healthy_theta = np.ones_like(damaged_theta)

    measurement_config = configuration["measurement"]
    reference_truth = simulate_augmented_truth(
        filter_model,
        np.ones(filter_model.n_zones),
        n_samples=5 * n_samples,
        time_step_s=time_step,
        force_stationary_standard_deviation_n=force_standard_deviation,
        burn_in_samples=burn_in_samples,
        rng=np.random.default_rng(
            int(measurement_config["reference_noise_seed"])
        ),
    )
    reference_sensor_acceleration = reference_truth.acceleration_m_per_s2[
        :, list(sensor_layout.dof_indices)
    ]
    noise_standard_deviation = float(
        measurement_config["noise_standard_deviation_fraction"]
    ) * np.std(reference_sensor_acceleration, axis=0, ddof=1)
    noise_covariance = np.diag(noise_standard_deviation**2)

    case_models = {
        case_name: AugmentedStructuralModel(
            mass_matrix_kg=spectral_system.mass_matrix_kg,
            damping_matrix_n_s_per_m=spectral_system.damping_matrix_n_s_per_m,
            zone_stiffness_components_n_per_m=(
                spectral_system.zone_components_n_per_m
            ),
            force_distribution=force_distribution,
            force_correlation_time_s=(
                float(excitation_config["correlation_time_s"])
                * float(case_specification["truth_force_correlation_time_scale"])
            ),
        )
        for case_name, case_specification in configuration["cases"].items()
    }

    observability_config = configuration["observability"]
    observability_truth = simulate_augmented_truth(
        filter_model,
        [0.92, 0.97, 0.99],
        n_samples=10,
        time_step_s=time_step,
        force_stationary_standard_deviation_n=force_standard_deviation,
        burn_in_samples=burn_in_samples,
        rng=np.random.default_rng(20261501),
    )
    observability = _observability_analysis(
        filter_model,
        observability_truth.states[0],
        state_scales,
        sensor_layout,
        time_step_s=time_step,
        horizon_steps=int(observability_config["horizon_steps"]),
        finite_difference_step=float(
            observability_config["finite_difference_step_scaled"]
        ),
        rank_threshold=float(
            observability_config["effective_rank_relative_threshold"]
        ),
    )

    loss_threshold = float(monitoring_sequence["meaningful_loss_fraction"])
    persistence = int(monitoring_sequence["persistence_windows"])
    filter_configurations = configuration["filters"]
    joint_method_names = ("joint_ukf", "joint_enkf")
    calibration_started = time.perf_counter()
    calibration_config = configuration["calibration"]
    calibration_probabilities = {name: [] for name in joint_method_names}
    for replicate in range(int(calibration_config["healthy_replicates"])):
        dataset = _generate_dataset(
            case_models["matched"],
            healthy_theta,
            sensor_layout,
            noise_standard_deviation,
            0.0,
            n_samples=n_samples,
            burn_in_samples=burn_in_samples,
            time_step_s=time_step,
            force_standard_deviation_n=force_standard_deviation,
            base_seed=int(calibration_config["random_seed"]),
            replicate=replicate,
        )
        for method_index, method_name in enumerate(joint_method_names):
            result = _run_multirate_filter(
                method_name,
                filter_configurations[method_name],
                filter_model,
                dataset,
                healthy_theta,
                theta_bounds,
                sensor_layout,
                noise_covariance,
                state_scales,
                configuration["initial_uncertainty"],
                configuration["process_uncertainty"],
                time_step_s=time_step,
                force_standard_deviation_n=force_standard_deviation,
                metric_exclusion_samples=metric_exclusion_samples,
                loss_threshold=loss_threshold,
                rng_seed=(
                    int(calibration_config["random_seed"])
                    + 10000 * method_index
                    + replicate
                ),
            )
            calibration_probabilities[method_name].append(
                result.loss_probabilities
            )
    alarm_thresholds, calibration = _calibrate_thresholds_allow_failure(
        calibration_probabilities,
        measurement_config["candidate_alarm_probability_thresholds"],
        persistence,
        float(measurement_config["maximum_calibration_false_alarm_fraction"]),
    )
    calibration_seconds = time.perf_counter() - calibration_started

    evaluation_started = time.perf_counter()
    evaluation_config = configuration["evaluation"]
    run_rows: list[dict[str, object]] = []
    monitoring_rows: list[dict[str, object]] = []
    representative_fast: dict[str, np.ndarray] = {}
    methods = tuple(str(name) for name in filter_configurations)
    for replicate in range(int(evaluation_config["replicates"])):
        for case_index, (case_name, case_specification) in enumerate(
            configuration["cases"].items()
        ):
            dataset = _generate_dataset(
                case_models[case_name],
                damaged_theta,
                sensor_layout,
                noise_standard_deviation,
                float(case_specification["independent_channel_dropout_probability"]),
                n_samples=n_samples,
                burn_in_samples=burn_in_samples,
                time_step_s=time_step,
                force_standard_deviation_n=force_standard_deviation,
                base_seed=int(evaluation_config["random_seed"]),
                replicate=replicate,
            )
            for method_index, method_name in enumerate(methods):
                result = _run_multirate_filter(
                    method_name,
                    filter_configurations[method_name],
                    filter_model,
                    dataset,
                    damaged_theta,
                    theta_bounds,
                    sensor_layout,
                    noise_covariance,
                    state_scales,
                    configuration["initial_uncertainty"],
                    configuration["process_uncertainty"],
                    time_step_s=time_step,
                    force_standard_deviation_n=force_standard_deviation,
                    metric_exclusion_samples=metric_exclusion_samples,
                    loss_threshold=loss_threshold,
                    rng_seed=(
                        int(evaluation_config["random_seed"])
                        + 100000 * case_index
                        + 10000 * method_index
                        + replicate
                    ),
                )
                decision = (
                    _decision_metrics(
                        result,
                        damaged_theta,
                        alarm_thresholds[method_name],
                        loss_threshold,
                        persistence,
                    )
                    if method_name in alarm_thresholds
                    else {
                        "detected": np.nan,
                        "truth_crossing_window": np.nan,
                        "alarm_window": np.nan,
                        "detection_delay_windows": np.nan,
                        "localized": np.nan,
                        "predamage_false_alarm_fraction": np.nan,
                    }
                )
                run_rows.append(
                    {
                        "replicate": replicate,
                        "case": case_name,
                        "method": method_name,
                        "dropout_fraction": dataset.dropout_fraction,
                        "displacement_nrmse": result.displacement_nrmse,
                        "velocity_nrmse": result.velocity_nrmse,
                        "force_nrmse": result.force_nrmse,
                        "acceleration_nrmse": result.acceleration_nrmse,
                        "displacement_coverage": result.displacement_coverage,
                        "stiffness_rmse": result.stiffness_rmse,
                        "stiffness_coverage": result.stiffness_coverage,
                        "minimum_scaled_covariance_eigenvalue": (
                            result.minimum_scaled_covariance_eigenvalue
                        ),
                        "runtime_seconds": result.runtime_seconds,
                        **decision,
                    }
                )
                theta_mean = result.monitoring_means[:, filter_model.theta_slice]
                theta_standard_deviation = np.sqrt(
                    np.diagonal(
                        result.monitoring_covariances[
                            :, filter_model.theta_slice, filter_model.theta_slice
                        ],
                        axis1=1,
                        axis2=2,
                    )
                )
                for window in range(n_windows):
                    for zone, zone_name in enumerate(("lower", "middle", "upper")):
                        monitoring_rows.append(
                            {
                                "replicate": replicate,
                                "case": case_name,
                                "method": method_name,
                                "window": window,
                                "time_day": time_days[window],
                                "zone": zone_name,
                                "true_theta": damaged_theta[window, zone],
                                "posterior_mean": theta_mean[window, zone],
                                "posterior_standard_deviation": (
                                    theta_standard_deviation[window, zone]
                                ),
                                "credible_lower": max(
                                    theta_bounds[zone, 0],
                                    theta_mean[window, zone]
                                    - 1.96 * theta_standard_deviation[window, zone],
                                ),
                                "credible_upper": min(
                                    theta_bounds[zone, 1],
                                    theta_mean[window, zone]
                                    + 1.96 * theta_standard_deviation[window, zone],
                                ),
                                "loss_probability": result.loss_probabilities[
                                    window, zone
                                ],
                            }
                        )
                if case_name == "stressed" and replicate == 0:
                    truth_last = dataset.truth_windows[-1]
                    top_dof = sensor_layout.dof_indices[-1]
                    if not representative_fast:
                        representative_fast = {
                            "time_s": np.arange(n_samples) * time_step,
                            "true_top_displacement_m": truth_last.states[
                                :, top_dof
                            ],
                            "true_top_velocity_m_per_s": truth_last.states[
                                :, filter_model.n_dof + top_dof
                            ],
                            "true_latent_force_n": truth_last.states[
                                :, filter_model.load_index
                            ],
                            "observed_top_acceleration_m_per_s2": (
                                dataset.observations[-1][:, -1]
                            ),
                        }
                    history = result.last_window_history
                    representative_fast[
                        f"estimated_top_displacement_m_{method_name}"
                    ] = history.posterior_means[:, top_dof]
                    representative_fast[
                        f"estimated_top_velocity_m_per_s_{method_name}"
                    ] = history.posterior_means[:, filter_model.n_dof + top_dof]
                    representative_fast[
                        f"estimated_latent_force_n_{method_name}"
                    ] = history.posterior_means[:, filter_model.load_index]
    runs = pd.DataFrame(run_rows)
    monitoring = pd.DataFrame(monitoring_rows)
    fast_state = pd.DataFrame(representative_fast)
    summary = _summarize_runs(runs)
    evaluation_seconds = time.perf_counter() - evaluation_started

    summary_index = summary.set_index(["method", "case"])
    observability_rank = int(observability["effective_rank"].iloc[0])
    joint_runs = runs[runs["method"].isin(joint_method_names)]
    matched_joint = joint_runs[joint_runs["case"] == "matched"]
    stressed_joint = joint_runs[joint_runs["case"] == "stressed"]
    ratios = []
    for case_name in configuration["cases"]:
        oracle_error = summary_index.loc[
            ("oracle_ukf", case_name), "displacement_nrmse_mean"
        ]
        for method_name in joint_method_names:
            ratios.append(
                summary_index.loc[
                    (method_name, case_name), "displacement_nrmse_mean"
                ]
                / oracle_error
            )
    hypotheses = configuration["hypotheses"]
    scientific_checks = [
        (
            "scaled_observability_rank",
            float(observability_rank),
            ">=",
            float(hypotheses["minimum_scaled_observability_rank"]),
        ),
        (
            "oracle_displacement_nrmse",
            float(
                runs[runs["method"] == "oracle_ukf"][
                    "displacement_nrmse"
                ].max()
            ),
            "<=",
            float(hypotheses["maximum_oracle_displacement_nrmse"]),
        ),
        (
            "joint_displacement_nrmse",
            float(joint_runs["displacement_nrmse"].max()),
            "<=",
            float(hypotheses["maximum_joint_displacement_nrmse"]),
        ),
        (
            "joint_velocity_nrmse",
            float(joint_runs["velocity_nrmse"].max()),
            "<=",
            float(hypotheses["maximum_joint_velocity_nrmse"]),
        ),
        (
            "joint_stiffness_rmse",
            float(joint_runs["stiffness_rmse"].max()),
            "<=",
            float(hypotheses["maximum_joint_stiffness_rmse"]),
        ),
        (
            "joint_to_oracle_displacement_nrmse_ratio",
            float(max(ratios)),
            "<=",
            float(
                hypotheses[
                    "maximum_joint_to_oracle_displacement_nrmse_ratio"
                ]
            ),
        ),
        (
            "matched_detection_fraction",
            float(matched_joint["detected"].mean()),
            ">=",
            float(hypotheses["minimum_matched_detection_fraction"]),
        ),
        (
            "stressed_detection_fraction",
            float(stressed_joint["detected"].mean()),
            ">=",
            float(hypotheses["minimum_stressed_detection_fraction"]),
        ),
        (
            "matched_stiffness_coverage",
            float(matched_joint["stiffness_coverage"].mean()),
            ">=",
            float(hypotheses["minimum_matched_stiffness_coverage"]),
        ),
        (
            "predamage_false_alarm_fraction",
            float(joint_runs["predamage_false_alarm_fraction"].mean()),
            "<=",
            float(hypotheses["maximum_predamage_false_alarm_fraction"]),
        ),
    ]
    check_rows = [
        {
            "category": "hypothesis",
            "check": name,
            "observed": observed,
            "operator": operator,
            "criterion": criterion,
            "supported": (
                observed >= criterion if operator == ">=" else observed <= criterion
            ),
        }
        for name, observed, operator, criterion in scientific_checks
    ]
    for row in calibration[calibration["selected"]].itertuples(index=False):
        check_rows.append(
            {
                "category": "calibration",
                "check": f"{row.method}_healthy_alarm_calibration",
                "observed": row.healthy_false_alarm_fraction,
                "operator": "<=",
                "criterion": float(
                    measurement_config[
                        "maximum_calibration_false_alarm_fraction"
                    ]
                ),
                "supported": bool(row.calibration_succeeded),
            }
        )
    verification = configuration["verification"]
    numeric_columns = [
        "displacement_nrmse",
        "velocity_nrmse",
        "force_nrmse",
        "acceleration_nrmse",
        "stiffness_rmse",
        "runtime_seconds",
    ]
    verification_checks = [
        (
            "evaluation_run_count",
            float(len(runs)),
            "==",
            float(verification["expected_evaluation_runs"]),
        ),
        (
            "nonfinite_fraction",
            float(
                1.0
                - np.mean(
                    np.isfinite(runs[numeric_columns].to_numpy(dtype=np.float64))
                )
            ),
            "<=",
            float(verification["maximum_nonfinite_fraction"]),
        ),
        (
            "minimum_scaled_covariance_eigenvalue",
            float(runs["minimum_scaled_covariance_eigenvalue"].min()),
            ">=",
            float(verification["minimum_covariance_eigenvalue"]),
        ),
        (
            "monitoring_row_count",
            float(len(monitoring)),
            "==",
            float(len(runs) * n_windows * filter_model.n_zones),
        ),
    ]
    for name, observed, operator, criterion in verification_checks:
        check_rows.append(
            {
                "category": "verification",
                "check": name,
                "observed": observed,
                "operator": operator,
                "criterion": criterion,
                "supported": (
                    np.isclose(observed, criterion)
                    if operator == "=="
                    else (
                        observed >= criterion
                        if operator == ">="
                        else observed <= criterion
                    )
                ),
            }
        )
    checks = pd.DataFrame(check_rows)

    total_seconds = time.perf_counter() - started
    runtime = pd.DataFrame(
        [
            {"phase": "calibration", "seconds": calibration_seconds},
            {"phase": "evaluation", "seconds": evaluation_seconds},
            {"phase": "total", "seconds": total_seconds},
        ]
        + [
            {
                "phase": f"evaluation_mean_{method}",
                "seconds": seconds,
            }
            for method, seconds in (
                runs.groupby("method")["runtime_seconds"].mean().items()
            )
        ]
    )
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs.drop(columns=["runtime_seconds"]).to_csv(
        RUNS_PATH, index=False, float_format="%.12g"
    )
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    monitoring.to_csv(MONITORING_PATH, index=False, float_format="%.12g")
    fast_state.to_csv(FAST_PATH, index=False, float_format="%.12g")
    calibration.to_csv(CALIBRATION_PATH, index=False, float_format="%.12g")
    observability.to_csv(OBSERVABILITY_PATH, index=False, float_format="%.12g")
    checks.to_csv(CHECKS_PATH, index=False, float_format="%.12g")
    runtime.to_csv(RUNTIME_PATH, index=False, float_format="%.12g")
    _create_figure(summary, monitoring, fast_state, calibration)

    print("STAGE 13 - JOINT STATE--PARAMETER ESTIMATION")
    print(
        f"Scaled finite-horizon observability rank: {observability_rank}/"
        f"{filter_model.n_state}"
    )
    print(
        "Method       | Case     | q NRMSE | v NRMSE | theta RMSE | "
        "Detection | Localisation"
    )
    print(
        "-------------+----------+---------+---------+------------+"
        "-----------+-------------"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.method:<12} | {row.case:<8} | "
            f"{row.displacement_nrmse_mean:>7.3f} | "
            f"{row.velocity_nrmse_mean:>7.3f} | "
            f"{row.stiffness_rmse_mean:>10.3f} | "
            f"{row.detected_mean:>9.3f} | {row.localized_mean:>11.3f}"
        )
    hypothesis_checks = checks[checks["category"] == "hypothesis"]
    print(
        "Predeclared hypotheses supported: "
        f"{int(hypothesis_checks['supported'].sum())}/{len(hypothesis_checks)}"
    )
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Checks: {CHECKS_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    failed_verification = checks[
        (checks["category"] == "verification") & ~checks["supported"]
    ]
    if not failed_verification.empty:
        failed_names = ", ".join(failed_verification["check"])
        raise RuntimeError(f"Stage 13 verification failed: {failed_names}")
    print("RESULT: PASS (joint experiment complete; failures retained)")


if __name__ == "__main__":
    main()
