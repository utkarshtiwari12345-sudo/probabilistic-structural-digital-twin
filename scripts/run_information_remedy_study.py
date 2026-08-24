"""Run the Stage 14 joint-estimation information-remedy study."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from run_joint_state_parameter_study import (  # noqa: E402
    MultirateFilterResult,
    _calibrate_thresholds_allow_failure,
    _decision_metrics,
    _initial_covariance,
    _loss_probabilities,
    _process_covariance,
    _state_scales,
)
from run_robustness_factorial_study import _load_yaml  # noqa: E402
from run_sequential_tracking_study import build_spectral_system  # noqa: E402

from probdt.deterioration import smooth_gradual_loss  # noqa: E402
from probdt.inference import (  # noqa: E402
    AugmentedStructuralModel,
    AugmentedTruth,
    JointStateFilterHistory,
    MixedObservationModel,
    run_augmented_unscented_filter,
    simulate_augmented_truth,
)
from probdt.structural import solve_modal_properties  # noqa: E402

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
REMEDY_CONFIG_PATH = PROJECT_ROOT / "config" / "information_remedy.yaml"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
RUNS_PATH = METRICS_DIR / "stage14_information_remedy_runs.csv"
SUMMARY_PATH = METRICS_DIR / "stage14_information_remedy_summary.csv"
MONITORING_PATH = METRICS_DIR / "stage14_monitoring_trajectory.csv"
CALIBRATION_PATH = METRICS_DIR / "stage14_alarm_calibration.csv"
OBSERVABILITY_PATH = METRICS_DIR / "stage14_observability.csv"
RECOVERY_PATH = METRICS_DIR / "stage14_recovery_gates.csv"
CHECKS_PATH = METRICS_DIR / "stage14_hypothesis_checks.csv"
RUNTIME_PATH = METRICS_DIR / "stage14_runtime.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage14_information_remedy.png"


@dataclass(frozen=True)
class RemedyDataset:
    """Shared truth and full superset observations for one monitoring history."""

    truth_windows: tuple[AugmentedTruth, ...]
    observations: tuple[np.ndarray, ...]


def _scheme_channel_indices(
    observation_model: MixedObservationModel,
    scheme_types: list[str],
) -> tuple[int, ...]:
    """Map a declared channel-type subset to ordered superset indices."""

    requested = {str(value) for value in scheme_types}
    available = set(observation_model.channel_types())
    if not requested or not requested <= available:
        raise ValueError("A Stage 14 scheme requests unavailable channel types.")
    return tuple(
        index
        for index, channel_type in enumerate(observation_model.channel_types())
        if channel_type in requested
    )


def _generate_dataset(
    truth_model: AugmentedStructuralModel,
    observation_model: MixedObservationModel,
    theta_history: np.ndarray,
    noise_standard_deviation: np.ndarray,
    dropout_probability: float,
    *,
    n_samples: int,
    burn_in_samples: int,
    time_step_s: float,
    force_standard_deviation_n: float,
    base_seed: int,
    replicate: int,
) -> RemedyDataset:
    """Generate nested mixed-channel data with common random numbers."""

    truth_windows: list[AugmentedTruth] = []
    observations: list[np.ndarray] = []
    for window, theta in enumerate(theta_history):
        truth = simulate_augmented_truth(
            truth_model,
            theta,
            n_samples=n_samples,
            time_step_s=time_step_s,
            force_stationary_standard_deviation_n=force_standard_deviation_n,
            burn_in_samples=burn_in_samples,
            rng=np.random.default_rng(
                np.random.SeedSequence([base_seed, replicate, window, 0])
            ),
        )
        clean = observation_model.observe(truth.states)
        noise_rng = np.random.default_rng(
            np.random.SeedSequence([base_seed, replicate, window, 1])
        )
        observed = clean + noise_rng.normal(
            scale=noise_standard_deviation,
            size=clean.shape,
        )
        if dropout_probability > 0.0:
            dropout_rng = np.random.default_rng(
                np.random.SeedSequence([base_seed, replicate, window, 2])
            )
            observed[
                dropout_rng.random(clean.shape) < dropout_probability
            ] = np.nan
        truth_windows.append(truth)
        observations.append(observed)
    return RemedyDataset(tuple(truth_windows), tuple(observations))


def _run_remedy_filter(
    observation_model: MixedObservationModel,
    dataset: RemedyDataset,
    theta_truth: np.ndarray,
    theta_bounds: np.ndarray,
    channel_indices: tuple[int, ...],
    noise_covariance: np.ndarray,
    state_scales: np.ndarray,
    initial_configuration: dict[str, object],
    process_configuration: dict[str, object],
    filter_configuration: dict[str, object],
    *,
    time_step_s: float,
    force_standard_deviation_n: float,
    metric_exclusion_samples: int,
    loss_threshold: float,
) -> MultirateFilterResult:
    """Run the fixed joint UKF for one information scheme."""

    started = time.perf_counter()
    model = observation_model.structural_model
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
    n_windows = len(dataset.observations)
    monitoring_means = np.empty((n_windows, model.n_state), dtype=np.float64)
    monitoring_covariances = np.empty(
        (n_windows, model.n_state, model.n_state),
        dtype=np.float64,
    )
    process_covariance = _process_covariance(
        model,
        process_configuration,
        time_step_s,
        force_standard_deviation_n,
    )
    selected_covariance = noise_covariance[np.ix_(channel_indices, channel_indices)]
    q_error_square = q_truth_square = 0.0
    v_error_square = v_truth_square = 0.0
    force_error_square = force_truth_square = 0.0
    acceleration_error_square = acceleration_truth_square = 0.0
    displacement_covered = displacement_total = 0
    minimum_scaled_eigenvalue = np.inf
    last_history: JointStateFilterHistory | None = None

    for window, (full_observation, truth) in enumerate(
        zip(dataset.observations, dataset.truth_windows, strict=True)
    ):
        if window > 0:
            theta_covariance = theta_covariance + np.diag(
                monitoring_process_scale**2
            )
        initial_mean = np.zeros(model.n_state, dtype=np.float64)
        initial_mean[model.theta_slice] = theta_mean
        history = run_augmented_unscented_filter(
            observation_model,
            full_observation[:, list(channel_indices)],
            channel_indices,
            time_step_s=time_step_s,
            theta_bounds=theta_bounds,
            initial_mean=initial_mean,
            initial_covariance=_initial_covariance(
                model,
                initial_configuration,
                theta_covariance,
            ),
            process_covariance=process_covariance,
            observation_covariance=selected_covariance,
            state_scales=state_scales,
            alpha=float(filter_configuration["alpha"]),
            beta=float(filter_configuration["beta"]),
            kappa=float(filter_configuration["kappa"]),
        )
        monitoring_means[window] = history.posterior_means[-1]
        monitoring_covariances[window] = history.posterior_covariances[-1]
        theta_mean = history.posterior_means[-1, model.theta_slice].copy()
        theta_covariance = history.posterior_covariances[
            -1,
            model.theta_slice,
            model.theta_slice,
        ].copy()

        retained = slice(metric_exclusion_samples, None)
        estimate = history.posterior_means[retained]
        truth_state = truth.states[retained]
        q_error = estimate[:, : model.n_dof] - truth_state[:, : model.n_dof]
        velocity_error = (
            estimate[:, model.n_dof : 2 * model.n_dof]
            - truth_state[:, model.n_dof : 2 * model.n_dof]
        )
        force_error = (
            estimate[:, model.load_index] - truth_state[:, model.load_index]
        )
        q_error_square += float(np.sum(q_error**2))
        q_truth_square += float(np.sum(truth_state[:, : model.n_dof] ** 2))
        v_error_square += float(np.sum(velocity_error**2))
        v_truth_square += float(
            np.sum(truth_state[:, model.n_dof : 2 * model.n_dof] ** 2)
        )
        force_error_square += float(np.sum(force_error**2))
        force_truth_square += float(np.sum(truth_state[:, model.load_index] ** 2))
        estimated_acceleration = model.acceleration(
            estimate,
            tuple(range(model.n_dof)),
        )
        true_acceleration = truth.acceleration_m_per_s2[retained]
        acceleration_error_square += float(
            np.sum((estimated_acceleration - true_acceleration) ** 2)
        )
        acceleration_truth_square += float(np.sum(true_acceleration**2))
        displacement_scale = history.posterior_standard_deviations[
            retained,
            : model.n_dof,
        ]
        displacement_covered += int(
            np.sum(np.abs(q_error) <= 1.96 * displacement_scale)
        )
        displacement_total += q_error.size
        scaled_covariance = history.posterior_covariances / np.outer(
            state_scales,
            state_scales,
        )
        minimum_scaled_eigenvalue = min(
            minimum_scaled_eigenvalue,
            float(np.min(np.linalg.eigvalsh(scaled_covariance))),
        )
        last_history = history

    assert last_history is not None
    loss_probabilities = _loss_probabilities(
        monitoring_means,
        monitoring_covariances,
        model.theta_slice,
        theta_bounds,
        loss_threshold,
    )
    theta_estimates = monitoring_means[:, model.theta_slice]
    theta_standard_deviation = np.sqrt(
        np.diagonal(
            monitoring_covariances[:, model.theta_slice, model.theta_slice],
            axis1=1,
            axis2=2,
        )
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
        stiffness_rmse=float(np.sqrt(np.mean((theta_estimates - theta_truth) ** 2))),
        stiffness_coverage=float(
            np.mean(
                np.abs(theta_estimates - theta_truth)
                <= 1.96 * theta_standard_deviation
            )
        ),
        minimum_scaled_covariance_eigenvalue=minimum_scaled_eigenvalue,
        last_window_history=last_history,
        runtime_seconds=time.perf_counter() - started,
    )


def _observability_analysis(
    model: MixedObservationModel,
    reference_state: np.ndarray,
    state_scales: np.ndarray,
    schemes: dict[str, tuple[int, ...]],
    *,
    time_step_s: float,
    horizon_steps: int,
    finite_difference_step: float,
    rank_threshold: float,
) -> pd.DataFrame:
    """Compare scaled finite-horizon observability across nested layouts."""

    reference_scaled = reference_state / state_scales
    rows: list[dict[str, object]] = []
    for scheme, indices in schemes.items():
        def features(
            scaled_state: np.ndarray,
            selected_indices: tuple[int, ...] = indices,
        ) -> np.ndarray:
            state = scaled_state * state_scales
            values: list[np.ndarray] = []
            for _ in range(horizon_steps):
                values.append(model.acceleration(state, selected_indices))
                state = model.propagate_rk4(state, time_step_s)
            return np.concatenate(values)

        jacobian = np.empty(
            (horizon_steps * len(indices), model.n_state),
            dtype=np.float64,
        )
        for state_index in range(model.n_state):
            step = finite_difference_step * max(
                abs(reference_scaled[state_index]),
                1.0,
            )
            positive = reference_scaled.copy()
            negative = reference_scaled.copy()
            positive[state_index] += step
            negative[state_index] -= step
            jacobian[:, state_index] = (
                features(positive) - features(negative)
            ) / (2.0 * step)
        column_norms = np.linalg.norm(jacobian, axis=0)
        normalized = jacobian / np.maximum(column_norms, 1.0e-15)
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        rank = int(
            np.sum(singular_values >= rank_threshold * singular_values[0])
        )
        condition = float(singular_values[0] / singular_values[-1])
        for singular_index, singular_value in enumerate(singular_values):
            rows.append(
                {
                    "scheme": scheme,
                    "channel_count": len(indices),
                    "singular_value_index": singular_index + 1,
                    "normalized_singular_value": singular_value,
                    "effective_rank": rank,
                    "n_augmented_states": model.n_state,
                    "normalized_condition_number": condition,
                    "minimum_column_norm": float(np.min(column_norms)),
                    "maximum_column_norm": float(np.max(column_norms)),
                }
            )
    return pd.DataFrame(rows)


def _summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate deterministic replicate metrics by scheme and case."""

    metrics = [
        "dropout_fraction",
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
    for (scheme, case), selected in runs.groupby(["scheme", "case"], sort=False):
        row: dict[str, object] = {
            "scheme": scheme,
            "case": case,
            "replicates": len(selected),
        }
        for metric in metrics:
            values = selected[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_standard_deviation"] = np.std(values, ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_recovery_gates(
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    gates: dict[str, float],
) -> pd.DataFrame:
    """Apply all frozen recovery gates independently to every scheme."""

    selected_calibration = calibration[calibration["selected"]].set_index("scheme")
    rows: list[dict[str, object]] = []
    definitions = (
        (
            "displacement_nrmse",
            "displacement_nrmse_mean",
            "max",
            "<=",
            "maximum_displacement_nrmse",
        ),
        (
            "velocity_nrmse",
            "velocity_nrmse_mean",
            "max",
            "<=",
            "maximum_velocity_nrmse",
        ),
        (
            "stiffness_rmse",
            "stiffness_rmse_mean",
            "max",
            "<=",
            "maximum_stiffness_rmse",
        ),
        (
            "matched_stiffness_coverage",
            "stiffness_coverage_mean",
            "matched",
            ">=",
            "minimum_matched_stiffness_coverage",
        ),
        (
            "stressed_detection_fraction",
            "detected_mean",
            "stressed",
            ">=",
            "minimum_stressed_detection_fraction",
        ),
        (
            "predamage_false_alarm_fraction",
            "predamage_false_alarm_fraction_mean",
            "max",
            "<=",
            "maximum_predamage_false_alarm_fraction",
        ),
    )
    for scheme, selected in summary.groupby("scheme", sort=False):
        for check, metric, selector, operator, criterion_name in definitions:
            if selector == "max":
                observed = float(selected[metric].max())
            else:
                observed = float(
                    selected.loc[selected["case"] == selector, metric].iloc[0]
                )
            criterion = float(gates[criterion_name])
            passed = (
                observed <= criterion if operator == "<=" else observed >= criterion
            )
            rows.append(
                {
                    "scheme": scheme,
                    "check": check,
                    "observed": observed,
                    "operator": operator,
                    "criterion": criterion,
                    "passed": passed,
                }
            )
        calibration_row = selected_calibration.loc[scheme]
        rows.append(
            {
                "scheme": scheme,
                "check": "healthy_alarm_calibration",
                "observed": float(calibration_row["healthy_false_alarm_fraction"]),
                "operator": "<=",
                "criterion": float(gates["maximum_healthy_false_alarm_fraction"]),
                "passed": bool(calibration_row["calibration_succeeded"]),
            }
        )
    table = pd.DataFrame(rows)
    restored = table.groupby("scheme", sort=False)["passed"].all()
    table["scheme_restores_joint_recovery"] = table["scheme"].map(restored)
    return table


def _create_figure(
    summary: pd.DataFrame,
    monitoring: pd.DataFrame,
    calibration: pd.DataFrame,
    recovery: pd.DataFrame,
    gates: dict[str, float],
) -> None:
    """Plot state recovery, parameter tracking, alarms, and gate outcomes."""

    schemes = list(summary["scheme"].drop_duplicates())
    labels = [value.replace("_", "\n") for value in schemes]
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(schemes)))
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), constrained_layout=True)
    displacement_axis, theta_axis, alarm_axis, gate_axis = axes.ravel()
    x = np.arange(len(schemes))
    width = 0.36
    for case_index, (case, offset) in enumerate(
        (("matched", -width / 2), ("stressed", width / 2))
    ):
        values = [
            float(
                summary.loc[
                    (summary["scheme"] == scheme) & (summary["case"] == case),
                    "displacement_nrmse_mean",
                ].iloc[0]
            )
            for scheme in schemes
        ]
        displacement_axis.bar(
            x + offset,
            values,
            width,
            color=colors,
            alpha=1.0 if case_index == 0 else 0.55,
            edgecolor="black",
            linewidth=0.4,
            label=case.capitalize(),
        )
    displacement_axis.axhline(
        float(gates["maximum_displacement_nrmse"]),
        color="#A60000",
        linestyle="--",
        linewidth=1.2,
        label="Recovery gate",
    )
    displacement_axis.set_xticks(x, labels)
    displacement_axis.set_ylabel("Displacement NRMSE")
    displacement_axis.set_title("Fast-state recovery")
    displacement_axis.legend(fontsize=8)
    displacement_axis.grid(axis="y", alpha=0.3)

    stressed = summary[summary["case"] == "stressed"].set_index("scheme")
    best_scheme = str(stressed["stiffness_rmse_mean"].idxmin())
    for scheme, color, label in (
        ("accel_only", "0.45", "Acceleration only"),
        (best_scheme, "#0072B2", f"Best: {best_scheme.replace('_', ' ')}"),
    ):
        selected = monitoring[
            (monitoring["scheme"] == scheme)
            & (monitoring["case"] == "stressed")
            & (monitoring["replicate"] == 0)
            & (monitoring["zone"] == "lower")
        ]
        theta_axis.plot(
            selected["time_day"],
            selected["posterior_mean"],
            marker="o",
            linewidth=1.4,
            color=color,
            label=label,
        )
        theta_axis.fill_between(
            selected["time_day"],
            selected["credible_lower"],
            selected["credible_upper"],
            color=color,
            alpha=0.12,
        )
    truth = monitoring[
        (monitoring["scheme"] == "accel_only")
        & (monitoring["case"] == "stressed")
        & (monitoring["replicate"] == 0)
        & (monitoring["zone"] == "lower")
    ]
    theta_axis.plot(
        truth["time_day"],
        truth["true_theta"],
        color="black",
        linewidth=1.8,
        label="Truth",
    )
    theta_axis.axhline(0.95, color="black", linestyle=":", linewidth=1.0)
    theta_axis.set_xlabel("Monitoring time (day)")
    theta_axis.set_ylabel("Lower-zone stiffness multiplier")
    theta_axis.set_title("Slow-parameter tracking under stress")
    theta_axis.legend(fontsize=8)
    theta_axis.grid(alpha=0.3)

    selected_calibration = calibration[calibration["selected"]].set_index("scheme")
    false_alarm = [
        float(selected_calibration.loc[scheme, "healthy_false_alarm_fraction"])
        for scheme in schemes
    ]
    detection = [float(stressed.loc[scheme, "detected_mean"]) for scheme in schemes]
    alarm_axis.bar(
        x - width / 2,
        false_alarm,
        width,
        color="#D55E00",
        label="Healthy false alarm",
    )
    alarm_axis.bar(
        x + width / 2,
        detection,
        width,
        color="#009E73",
        label="Stressed detection",
    )
    alarm_axis.axhline(
        float(gates["maximum_healthy_false_alarm_fraction"]),
        color="#A60000",
        linestyle="--",
        linewidth=1.0,
    )
    alarm_axis.set_xticks(x, labels)
    alarm_axis.set_ylim(0.0, 1.05)
    alarm_axis.set_ylabel("Fraction")
    alarm_axis.set_title("Frozen alarm calibration and detection")
    alarm_axis.legend(fontsize=8)
    alarm_axis.grid(axis="y", alpha=0.3)

    checks = list(recovery["check"].drop_duplicates())
    matrix = np.asarray(
        [
            [
                bool(
                    recovery.loc[
                        (recovery["scheme"] == scheme)
                        & (recovery["check"] == check),
                        "passed",
                    ].iloc[0]
                )
                for check in checks
            ]
            for scheme in schemes
        ],
        dtype=float,
    )
    gate_axis.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    gate_axis.set_xticks(
        np.arange(len(checks)),
        [value.replace("_", "\n") for value in checks],
        rotation=35,
        ha="right",
        fontsize=7,
    )
    gate_axis.set_yticks(np.arange(len(schemes)), labels, fontsize=8)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            gate_axis.text(
                column,
                row,
                "PASS" if matrix[row, column] else "FAIL",
                ha="center",
                va="center",
                fontsize=6,
                color="black",
            )
    restored = recovery.groupby("scheme", sort=False)["passed"].all()
    restored_names = [name for name, value in restored.items() if value]
    gate_axis.set_title(
        "Predeclared joint-recovery gates\nRestored: "
        + (", ".join(restored_names) if restored_names else "none")
    )
    figure.suptitle(
        "Stage 14: information remedies for joint state--stiffness inference",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main() -> None:
    """Execute the frozen Stage 14 information-remedy experiment."""

    started = time.perf_counter()
    base_config = _load_yaml(BASE_CONFIG_PATH)
    monitoring_config = _load_yaml(MONITORING_CONFIG_PATH)
    configuration = _load_yaml(REMEDY_CONFIG_PATH)
    excitation_config = monitoring_config["excitation"]
    spectral_system = build_spectral_system(base_config)
    force_distribution = np.asarray(
        excitation_config["spatial_distribution"],
        dtype=np.float64,
    )
    physical_model = AugmentedStructuralModel(
        mass_matrix_kg=spectral_system.mass_matrix_kg,
        damping_matrix_n_s_per_m=spectral_system.damping_matrix_n_s_per_m,
        zone_stiffness_components_n_per_m=spectral_system.zone_components_n_per_m,
        force_distribution=force_distribution,
        force_correlation_time_s=float(excitation_config["correlation_time_s"]),
    )
    healthy_stiffness = np.sum(spectral_system.zone_components_n_per_m, axis=0)
    modal_properties = solve_modal_properties(
        spectral_system.mass_matrix_kg,
        healthy_stiffness,
    )
    measurement = configuration["measurement_superset"]
    n_modal = int(measurement["modal_coordinates"])
    modal_projection = (
        modal_properties.mode_shapes[:, :n_modal].T
        @ spectral_system.mass_matrix_kg
    )
    observation_model = MixedObservationModel(
        structural_model=physical_model,
        acceleration_dof_indices=tuple(
            int(value) - 1 for value in measurement["acceleration_dofs_one_based"]
        ),
        displacement_dof_indices=tuple(
            int(value) - 1 for value in measurement["displacement_dofs_one_based"]
        ),
        strain_story_indices=tuple(
            int(value) - 1
            for value in measurement["strain_story_indices_one_based"]
        ),
        level_spacing_m=float(base_config["model"]["level_spacing_m"]),
        modal_projection_vectors=modal_projection,
        include_latent_force=bool(measurement["include_latent_force"]),
    )
    schemes = {
        str(name): _scheme_channel_indices(observation_model, channel_types)
        for name, channel_types in configuration["schemes"].items()
    }
    response = configuration["response_window"]
    time_step = float(response["time_step_s"])
    n_samples = round(float(response["duration_s"]) / time_step) + 1
    burn_in_samples = round(float(response["burn_in_s"]) / time_step)
    metric_exclusion_samples = round(
        float(response["metric_initialization_exclusion_s"]) / time_step
    )
    force_standard_deviation = float(
        excitation_config["stationary_standard_deviation_n"]
    )
    theta_bounds = np.asarray([[0.65, 1.0]] * physical_model.n_zones)
    state_scales = _state_scales(physical_model, configuration["state_scaling"])

    monitoring_sequence = configuration["monitoring"]
    n_windows = int(monitoring_sequence["number_of_windows"])
    time_days = np.arange(n_windows, dtype=np.float64) * float(
        monitoring_sequence["update_interval_days"]
    )
    loss = smooth_gradual_loss(
        time_days * 86400.0,
        onset_s=float(monitoring_sequence["deterioration_onset_day"]) * 86400.0,
        completion_s=(
            float(monitoring_sequence["deterioration_completion_day"]) * 86400.0
        ),
        final_loss_fraction=float(
            monitoring_sequence["final_lower_zone_loss_fraction"]
        ),
    )
    damaged_theta = np.ones((n_windows, physical_model.n_zones))
    damaged_theta[:, 0] = 1.0 - loss
    healthy_theta = np.ones_like(damaged_theta)

    reference_truth = simulate_augmented_truth(
        physical_model,
        np.ones(physical_model.n_zones),
        n_samples=5 * n_samples,
        time_step_s=time_step,
        force_stationary_standard_deviation_n=force_standard_deviation,
        burn_in_samples=burn_in_samples,
        rng=np.random.default_rng(int(measurement["reference_noise_seed"])),
    )
    reference_observation = observation_model.observe(reference_truth.states)
    noise_fractions = measurement["noise_standard_deviation_fraction"]
    noise_standard_deviation = np.asarray(
        [
            float(noise_fractions[channel_type])
            * np.std(reference_observation[:, index], ddof=1)
            for index, channel_type in enumerate(observation_model.channel_types())
        ]
    )
    if np.any(noise_standard_deviation <= 0.0):
        raise RuntimeError("Every Stage 14 observation channel needs nonzero noise.")
    noise_covariance = np.diag(noise_standard_deviation**2)
    case_models = {
        str(case): AugmentedStructuralModel(
            mass_matrix_kg=spectral_system.mass_matrix_kg,
            damping_matrix_n_s_per_m=spectral_system.damping_matrix_n_s_per_m,
            zone_stiffness_components_n_per_m=spectral_system.zone_components_n_per_m,
            force_distribution=force_distribution,
            force_correlation_time_s=(
                float(excitation_config["correlation_time_s"])
                * float(specification["truth_force_correlation_time_scale"])
            ),
        )
        for case, specification in configuration["cases"].items()
    }

    observability_configuration = configuration["observability"]
    observability_truth = simulate_augmented_truth(
        physical_model,
        [0.92, 0.97, 0.99],
        n_samples=10,
        time_step_s=time_step,
        force_stationary_standard_deviation_n=force_standard_deviation,
        burn_in_samples=burn_in_samples,
        rng=np.random.default_rng(20261601),
    )
    observability = _observability_analysis(
        observation_model,
        observability_truth.states[0],
        state_scales,
        schemes,
        time_step_s=time_step,
        horizon_steps=int(observability_configuration["horizon_steps"]),
        finite_difference_step=float(
            observability_configuration["finite_difference_step_scaled"]
        ),
        rank_threshold=float(
            observability_configuration["effective_rank_relative_threshold"]
        ),
    )

    loss_threshold = float(monitoring_sequence["meaningful_loss_fraction"])
    persistence = int(monitoring_sequence["persistence_windows"])
    calibration_started = time.perf_counter()
    calibration_config = configuration["calibration"]
    calibration_probabilities = {scheme: [] for scheme in schemes}
    for replicate in range(int(calibration_config["healthy_replicates"])):
        dataset = _generate_dataset(
            case_models["matched"],
            observation_model,
            healthy_theta,
            noise_standard_deviation,
            0.0,
            n_samples=n_samples,
            burn_in_samples=burn_in_samples,
            time_step_s=time_step,
            force_standard_deviation_n=force_standard_deviation,
            base_seed=int(calibration_config["random_seed"]),
            replicate=replicate,
        )
        for scheme, channel_indices in schemes.items():
            result = _run_remedy_filter(
                observation_model,
                dataset,
                healthy_theta,
                theta_bounds,
                channel_indices,
                noise_covariance,
                state_scales,
                configuration["initial_uncertainty"],
                configuration["process_uncertainty"],
                configuration["filter"],
                time_step_s=time_step,
                force_standard_deviation_n=force_standard_deviation,
                metric_exclusion_samples=metric_exclusion_samples,
                loss_threshold=loss_threshold,
            )
            calibration_probabilities[scheme].append(result.loss_probabilities)
    alarm_thresholds, calibration = _calibrate_thresholds_allow_failure(
        calibration_probabilities,
        measurement["candidate_alarm_probability_thresholds"],
        persistence,
        float(measurement["maximum_calibration_false_alarm_fraction"]),
    )
    calibration = calibration.rename(columns={"method": "scheme"})
    calibration_seconds = time.perf_counter() - calibration_started

    evaluation_started = time.perf_counter()
    evaluation = configuration["evaluation"]
    run_rows: list[dict[str, object]] = []
    monitoring_rows: list[dict[str, object]] = []
    for replicate in range(int(evaluation["replicates"])):
        for case, case_specification in configuration["cases"].items():
            dataset = _generate_dataset(
                case_models[case],
                observation_model,
                damaged_theta,
                noise_standard_deviation,
                float(case_specification["independent_channel_dropout_probability"]),
                n_samples=n_samples,
                burn_in_samples=burn_in_samples,
                time_step_s=time_step,
                force_standard_deviation_n=force_standard_deviation,
                base_seed=int(evaluation["random_seed"]),
                replicate=replicate,
            )
            for scheme, channel_indices in schemes.items():
                result = _run_remedy_filter(
                    observation_model,
                    dataset,
                    damaged_theta,
                    theta_bounds,
                    channel_indices,
                    noise_covariance,
                    state_scales,
                    configuration["initial_uncertainty"],
                    configuration["process_uncertainty"],
                    configuration["filter"],
                    time_step_s=time_step,
                    force_standard_deviation_n=force_standard_deviation,
                    metric_exclusion_samples=metric_exclusion_samples,
                    loss_threshold=loss_threshold,
                )
                decision = _decision_metrics(
                    result,
                    damaged_theta,
                    alarm_thresholds[scheme],
                    loss_threshold,
                    persistence,
                )
                selected_observations = np.concatenate(
                    [
                        window[:, list(channel_indices)]
                        for window in dataset.observations
                    ]
                )
                run_rows.append(
                    {
                        "replicate": replicate,
                        "case": case,
                        "scheme": scheme,
                        "channel_count": len(channel_indices),
                        "dropout_fraction": float(
                            np.mean(~np.isfinite(selected_observations))
                        ),
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
                theta_mean = result.monitoring_means[:, physical_model.theta_slice]
                theta_scale = np.sqrt(
                    np.diagonal(
                        result.monitoring_covariances[
                            :,
                            physical_model.theta_slice,
                            physical_model.theta_slice,
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
                                "case": case,
                                "scheme": scheme,
                                "window": window,
                                "time_day": time_days[window],
                                "zone": zone_name,
                                "true_theta": damaged_theta[window, zone],
                                "posterior_mean": theta_mean[window, zone],
                                "posterior_standard_deviation": theta_scale[
                                    window,
                                    zone,
                                ],
                                "credible_lower": max(
                                    theta_bounds[zone, 0],
                                    theta_mean[window, zone]
                                    - 1.96 * theta_scale[window, zone],
                                ),
                                "credible_upper": min(
                                    theta_bounds[zone, 1],
                                    theta_mean[window, zone]
                                    + 1.96 * theta_scale[window, zone],
                                ),
                                "loss_probability": result.loss_probabilities[
                                    window,
                                    zone,
                                ],
                            }
                        )
    runs = pd.DataFrame(run_rows)
    summary = _summarize_runs(runs)
    monitoring = pd.DataFrame(monitoring_rows)
    evaluation_seconds = time.perf_counter() - evaluation_started
    gates = configuration["recovery_gates"]
    recovery = _evaluate_recovery_gates(summary, calibration, gates)

    baseline = summary[summary["scheme"] == "accel_only"]
    remedies = summary[summary["scheme"] != "accel_only"]
    baseline_q = float(baseline["displacement_nrmse_mean"].max())
    baseline_theta = float(baseline["stiffness_rmse_mean"].max())
    best_q = float(remedies.groupby("scheme")["displacement_nrmse_mean"].max().min())
    best_theta = float(remedies.groupby("scheme")["stiffness_rmse_mean"].max().min())
    selected_calibration = calibration[calibration["selected"]]
    calibrated_remedies = int(
        selected_calibration.loc[
            selected_calibration["scheme"] != "accel_only",
            "calibration_succeeded",
        ].sum()
    )
    restored = recovery.groupby("scheme", sort=False)["passed"].all()
    restored_remedies = int(restored.drop(index="accel_only").sum())
    ranks = observability.groupby("scheme")["effective_rank"].first()
    hypotheses = configuration["hypotheses"]
    scientific_checks = [
        (
            "best_scaled_observability_rank",
            float(ranks.max()),
            ">=",
            float(hypotheses["minimum_best_scaled_observability_rank"]),
        ),
        (
            "best_displacement_error_reduction_fraction",
            1.0 - best_q / baseline_q,
            ">=",
            float(hypotheses["minimum_displacement_error_reduction_fraction"]),
        ),
        (
            "best_stiffness_error_reduction_fraction",
            1.0 - best_theta / baseline_theta,
            ">=",
            float(hypotheses["minimum_stiffness_error_reduction_fraction"]),
        ),
        (
            "accel_force_maximum_force_nrmse",
            float(
                summary.loc[
                    summary["scheme"] == "accel_force",
                    "force_nrmse_mean",
                ].max()
            ),
            "<=",
            float(hypotheses["maximum_best_force_nrmse"]),
        ),
        (
            "calibrated_remedy_count",
            float(calibrated_remedies),
            ">=",
            float(hypotheses["minimum_number_of_calibrated_remedies"]),
        ),
        (
            "restored_remedy_count",
            float(restored_remedies),
            ">=",
            float(hypotheses["minimum_number_of_restored_remedies"]),
        ),
    ]
    check_rows = [
        {
            "category": "hypothesis",
            "check": check,
            "observed": observed,
            "operator": operator,
            "criterion": criterion,
            "supported": (
                observed >= criterion if operator == ">=" else observed <= criterion
            ),
        }
        for check, observed, operator, criterion in scientific_checks
    ]
    verification = configuration["verification"]
    numeric_columns = [
        "displacement_nrmse",
        "velocity_nrmse",
        "force_nrmse",
        "acceleration_nrmse",
        "stiffness_rmse",
    ]
    verification_checks = [
        (
            "evaluation_run_count",
            float(len(runs)),
            "==",
            float(verification["expected_evaluation_runs"]),
        ),
        (
            "scheme_count",
            float(runs["scheme"].nunique()),
            "==",
            float(verification["expected_schemes"]),
        ),
        (
            "nonfinite_fraction",
            float(
                1.0
                - np.mean(np.isfinite(runs[numeric_columns].to_numpy(dtype=float)))
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
    ]
    for check, observed, operator, criterion in verification_checks:
        check_rows.append(
            {
                "category": "verification",
                "check": check,
                "observed": observed,
                "operator": operator,
                "criterion": criterion,
                "supported": (
                    np.isclose(observed, criterion)
                    if operator == "=="
                    else observed >= criterion
                    if operator == ">="
                    else observed <= criterion
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
            {"phase": f"evaluation_mean_{scheme}", "seconds": seconds}
            for scheme, seconds in (
                runs.groupby("scheme")["runtime_seconds"].mean().items()
            )
        ]
    )
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    runs.drop(columns=["runtime_seconds"]).to_csv(
        RUNS_PATH,
        index=False,
        float_format="%.12g",
    )
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    monitoring.to_csv(MONITORING_PATH, index=False, float_format="%.12g")
    calibration.to_csv(CALIBRATION_PATH, index=False, float_format="%.12g")
    observability.to_csv(OBSERVABILITY_PATH, index=False, float_format="%.12g")
    recovery.to_csv(RECOVERY_PATH, index=False, float_format="%.12g")
    checks.to_csv(CHECKS_PATH, index=False, float_format="%.12g")
    runtime.to_csv(RUNTIME_PATH, index=False, float_format="%.12g")
    _create_figure(summary, monitoring, calibration, recovery, gates)

    summary_index = summary.set_index(["scheme", "case"])
    print("STAGE 14 - INFORMATION REMEDIES FOR JOINT INFERENCE")
    print(
        "Scheme             | q max | theta max | rank | healthy FA | "
        "calibrated | restored"
    )
    print(
        "-------------------+-------+-----------+------+------------+------------+---------"
    )
    for scheme in schemes:
        selected = summary_index.loc[scheme]
        calibration_row = selected_calibration.set_index("scheme").loc[scheme]
        print(
            f"{scheme:<18} | {selected['displacement_nrmse_mean'].max():>5.3f} | "
            f"{selected['stiffness_rmse_mean'].max():>9.3f} | "
            f"{int(ranks[scheme]):>4d} | "
            f"{calibration_row['healthy_false_alarm_fraction']:>10.3f} | "
            f"{bool(calibration_row['calibration_succeeded'])!s:>10} | "
            f"{bool(restored[scheme])!s:>7}"
        )
    hypothesis_rows = checks[checks["category"] == "hypothesis"]
    print(
        "Predeclared hypotheses supported: "
        f"{int(hypothesis_rows['supported'].sum())}/{len(hypothesis_rows)}"
    )
    verification_rows = checks[checks["category"] == "verification"]
    if not bool(verification_rows["supported"].all()):
        raise RuntimeError("Stage 14 mechanical verification failed.")
    print(f"Restored remedies: {restored_remedies}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Recovery gates: {RECOVERY_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS (information-remedy experiment complete; failures retained)")


if __name__ == "__main__":
    main()
