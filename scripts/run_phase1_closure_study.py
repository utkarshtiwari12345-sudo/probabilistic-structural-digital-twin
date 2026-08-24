"""Run the Stage 16 Phase I artifact audit and frozen Monte Carlo closure."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm, t

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from run_information_remedy_study import (  # noqa: E402
    _generate_dataset,
    _run_remedy_filter,
    _scheme_channel_indices,
)
from run_joint_state_parameter_study import (  # noqa: E402
    _decision_metrics,
    _loss_probabilities,
    _state_scales,
)
from run_robustness_factorial_study import _load_yaml  # noqa: E402
from run_sequential_tracking_study import (  # noqa: E402
    build_spectral_system,
    first_persistent_alarm_indices,
)
from run_uncertainty_calibration_study import (  # noqa: E402
    _adjust_covariances,
    _interval_statistics,
)

from probdt.deterioration import smooth_gradual_loss  # noqa: E402
from probdt.diagnostics import (  # noqa: E402
    audit_pipeline_artifacts,
    select_pipeline_stages,
)
from probdt.inference import (  # noqa: E402
    AugmentedStructuralModel,
    MixedObservationModel,
    UncertaintyCalibration,
    calibrated_standard_deviations,
    simulate_augmented_truth,
)
from probdt.structural import solve_modal_properties  # noqa: E402

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
CLOSURE_CONFIG_PATH = PROJECT_ROOT / "config" / "phase1_closure.yaml"
STAGE15_PARAMETERS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage15_calibration_parameters.csv"
)
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
RUNS_PATH = METRICS_DIR / "stage16_confirmation_runs.csv"
SUMMARY_PATH = METRICS_DIR / "stage16_confirmation_summary.csv"
ZONE_PATH = METRICS_DIR / "stage16_zone_coverage.csv"
HEALTHY_PATH = METRICS_DIR / "stage16_healthy_specificity.csv"
CONVERGENCE_PATH = METRICS_DIR / "stage16_convergence.csv"
MANIFEST_PATH = METRICS_DIR / "stage16_phase1_manifest.csv"
CHECKS_PATH = METRICS_DIR / "stage16_confirmation_checks.csv"
RUNTIME_PATH = METRICS_DIR / "stage16_runtime.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage16_phase1_closure.png"


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= trials.")
    z = float(norm.ppf(0.975))
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    lower = float(np.clip(center - half_width, 0.0, 1.0))
    upper = float(np.clip(center + half_width, 0.0, 1.0))
    if successes == 0:
        lower = 0.0
    if successes == trials:
        upper = 1.0
    return lower, upper


def _mean_interval(values: np.ndarray) -> tuple[float, float]:
    """Return a two-sided 95% t interval across independent histories."""

    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean
    half_width = float(
        t.ppf(0.975, values.size - 1)
        * np.std(values, ddof=1)
        / np.sqrt(values.size)
    )
    return mean - half_width, mean + half_width


def _summarize_confirmation(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize continuous and binomial metrics by method and truth case."""

    continuous = [
        "coverage",
        "mean_interval_width",
        "mean_interval_score",
        "mean_theta_nees",
        "theta_rmse",
        "displacement_nrmse",
        "force_nrmse",
        "predamage_false_alarm_fraction",
    ]
    rows: list[dict[str, object]] = []
    for (method, case), selected in runs.groupby(["method", "case"], sort=False):
        row: dict[str, object] = {
            "method": method,
            "case": case,
            "replicates": len(selected),
        }
        for metric in continuous:
            values = selected[metric].to_numpy(dtype=np.float64)
            lower, upper = _mean_interval(values)
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_standard_deviation"] = np.std(values, ddof=1)
            row[f"{metric}_confidence_lower"] = lower
            row[f"{metric}_confidence_upper"] = upper
        for metric in ("detected", "localized"):
            values = selected[metric].to_numpy(dtype=np.float64)
            successes = int(np.sum(values > 0.5))
            lower, upper = _wilson_interval(successes, values.size)
            row[f"{metric}_fraction"] = successes / values.size
            row[f"{metric}_wilson_lower"] = lower
            row[f"{metric}_wilson_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def _zone_summary(zone_runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-history zonal coverage with history-level intervals."""

    rows: list[dict[str, object]] = []
    for (method, case, zone), selected in zone_runs.groupby(
        ["method", "case", "zone"],
        sort=False,
    ):
        coverage = selected["coverage"].to_numpy(dtype=np.float64)
        lower, upper = _mean_interval(coverage)
        rows.append(
            {
                "method": method,
                "case": case,
                "zone": zone,
                "replicates": len(selected),
                "coverage_mean": np.mean(coverage),
                "coverage_standard_deviation": np.std(coverage, ddof=1),
                "coverage_confidence_lower": lower,
                "coverage_confidence_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def _create_convergence(
    runs: pd.DataFrame,
    healthy: pd.DataFrame,
) -> pd.DataFrame:
    """Create cumulative confirmation traces in deterministic replicate order."""

    rows: list[dict[str, object]] = []
    damaged = runs[runs["method"] == "scalar_inflation"].sort_values(
        ["replicate", "case"]
    )
    coverage_values: list[float] = []
    for index, row in enumerate(damaged.itertuples(index=False), start=1):
        coverage_values.append(float(row.coverage))
        rows.append(
            {
                "trace": "damaged_scalar_coverage",
                "history_count": index,
                "cumulative_value": np.mean(coverage_values),
            }
        )
    healthy_scalar = healthy[healthy["method"] == "scalar_inflation"].sort_values(
        "replicate"
    )
    false_alarm_counts = 0.0
    zone_trials = 0
    for index, row in enumerate(healthy_scalar.itertuples(index=False), start=1):
        false_alarm_counts += 3.0 * float(row.zone_false_alarm_fraction)
        zone_trials += 3
        rows.append(
            {
                "trace": "healthy_zone_false_alarm",
                "history_count": index,
                "cumulative_value": false_alarm_counts / zone_trials,
            }
        )
    return pd.DataFrame(rows)


def _create_figure(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    zones: pd.DataFrame,
    healthy: pd.DataFrame,
    convergence: pd.DataFrame,
    checks: pd.DataFrame,
    gates: dict[str, float],
) -> None:
    """Plot larger-sample coverage, accuracy, decisions, and convergence."""

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), constrained_layout=True)
    coverage_axis, accuracy_axis, decision_axis, convergence_axis = axes.ravel()
    methods = ("raw", "scalar_inflation")
    cases = ("matched", "stressed")
    x = np.arange(len(cases))
    width = 0.34
    colors = {"raw": "0.5", "scalar_inflation": "#0072B2"}
    for method_index, method in enumerate(methods):
        values = []
        lower_errors = []
        upper_errors = []
        for case in cases:
            selected = runs[(runs["method"] == method) & (runs["case"] == case)]
            value = float(selected["coverage"].mean())
            lower, upper = _mean_interval(selected["coverage"].to_numpy())
            values.append(value)
            lower_errors.append(value - lower)
            upper_errors.append(upper - value)
        positions = x + (method_index - 0.5) * width
        coverage_axis.bar(
            positions,
            values,
            width,
            color=colors[method],
            label=method.replace("_", " "),
        )
        coverage_axis.errorbar(
            positions,
            values,
            yerr=np.asarray([lower_errors, upper_errors]),
            fmt="none",
            color="black",
            capsize=4,
        )
    coverage_axis.axhspan(
        float(gates["minimum_case_stiffness_coverage"]),
        float(gates["maximum_case_stiffness_coverage"]),
        color="#009E73",
        alpha=0.1,
        label="Confirmation range",
    )
    coverage_axis.axhline(0.95, color="black", linestyle=":", linewidth=1.2)
    coverage_axis.set_xticks(x, [value.capitalize() for value in cases])
    coverage_axis.set_ylim(0.0, 1.05)
    coverage_axis.set_ylabel("History-level marginal coverage")
    coverage_axis.set_title("Frozen scalar calibration on new seeds")
    coverage_axis.legend(fontsize=8)
    coverage_axis.grid(axis="y", alpha=0.3)

    scalar_summary = summary[summary["method"] == "scalar_inflation"].set_index(
        "case"
    )
    accuracy_metrics = (
        ("theta_rmse_mean", "Stiffness RMSE", "#0072B2"),
        ("displacement_nrmse_mean", "Displacement NRMSE", "#009E73"),
        ("force_nrmse_mean", "Force NRMSE", "#D55E00"),
    )
    metric_x = np.arange(len(accuracy_metrics))
    for case_index, case in enumerate(cases):
        accuracy_axis.bar(
            metric_x + (case_index - 0.5) * width,
            [
                float(scalar_summary.loc[case, metric])
                for metric, _, _ in accuracy_metrics
            ],
            width,
            color=[color for _, _, color in accuracy_metrics],
            alpha=1.0 if case == "matched" else 0.55,
            edgecolor="black",
            linewidth=0.4,
            label=case.capitalize(),
        )
    accuracy_axis.set_xticks(metric_x, [label for _, label, _ in accuracy_metrics])
    accuracy_axis.set_ylabel("Normalized / multiplier RMSE")
    accuracy_axis.set_title("Selected-twin accuracy")
    accuracy_axis.legend(fontsize=8)
    accuracy_axis.grid(axis="y", alpha=0.3)

    stressed = scalar_summary.loc["stressed"]
    healthy_scalar = healthy[healthy["method"] == "scalar_inflation"]
    decision_labels = (
        "Detection",
        "Localisation",
        "Healthy zone\nspecificity",
        "Healthy structure\nspecificity",
    )
    decision_values = [
        float(stressed["detected_fraction"]),
        float(stressed["localized_fraction"]),
        1.0 - float(healthy_scalar["zone_false_alarm_fraction"].mean()),
        1.0 - float(healthy_scalar["structure_false_alarm"].mean()),
    ]
    detection_interval = (
        float(stressed["detected_wilson_lower"]),
        float(stressed["detected_wilson_upper"]),
    )
    localization_interval = (
        float(stressed["localized_wilson_lower"]),
        float(stressed["localized_wilson_upper"]),
    )
    zone_failures = round(3 * healthy_scalar["zone_false_alarm_fraction"].sum())
    zone_false_interval = _wilson_interval(zone_failures, 3 * len(healthy_scalar))
    structure_failures = int(healthy_scalar["structure_false_alarm"].sum())
    structure_false_interval = _wilson_interval(
        structure_failures,
        len(healthy_scalar),
    )
    decision_intervals = [
        detection_interval,
        localization_interval,
        (1.0 - zone_false_interval[1], 1.0 - zone_false_interval[0]),
        (1.0 - structure_false_interval[1], 1.0 - structure_false_interval[0]),
    ]
    lower_error = [
        value - interval[0]
        for value, interval in zip(decision_values, decision_intervals, strict=True)
    ]
    upper_error = [
        interval[1] - value
        for value, interval in zip(decision_values, decision_intervals, strict=True)
    ]
    decision_axis.bar(
        np.arange(4),
        decision_values,
        color=["#009E73", "#56B4E9", "#E69F00", "#D55E00"],
    )
    decision_axis.errorbar(
        np.arange(4),
        decision_values,
        yerr=np.asarray([lower_error, upper_error]),
        fmt="none",
        color="black",
        capsize=4,
    )
    decision_axis.set_xticks(np.arange(4), decision_labels)
    decision_axis.set_ylim(0.0, 1.05)
    decision_axis.set_ylabel("Confirmed fraction")
    decision_axis.set_title("Decision performance with Wilson intervals")
    decision_axis.grid(axis="y", alpha=0.3)

    for trace, color, label in (
        ("damaged_scalar_coverage", "#0072B2", "Scalar coverage"),
        ("healthy_zone_false_alarm", "#D55E00", "Healthy zone false alarm"),
    ):
        selected = convergence[convergence["trace"] == trace]
        convergence_axis.plot(
            selected["history_count"],
            selected["cumulative_value"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=color,
            label=label,
        )
    convergence_axis.axhline(0.95, color="black", linestyle=":", linewidth=1.0)
    convergence_axis.axhline(0.05, color="#A60000", linestyle="--", linewidth=1.0)
    convergence_axis.set_xlabel("Independent histories accumulated")
    convergence_axis.set_ylabel("Cumulative fraction")
    artifact_pass = checks.loc[
        checks["check"] == "phase1_artifact_pass_fraction",
        "supported",
    ].iloc[0]
    convergence_axis.set_title(
        f"Monte Carlo convergence; Phase I audit: {'PASS' if artifact_pass else 'FAIL'}"
    )
    convergence_axis.legend(fontsize=8)
    convergence_axis.grid(alpha=0.3)

    figure.suptitle(
        "Stage 16: Phase I closure and larger frozen confirmation",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main() -> None:
    """Audit Stages 1--15 and confirm the frozen Stage 15 selection."""

    started = time.perf_counter()
    base_config = _load_yaml(BASE_CONFIG_PATH)
    monitoring_config = _load_yaml(MONITORING_CONFIG_PATH)
    configuration = _load_yaml(CLOSURE_CONFIG_PATH)
    study = configuration["study"]
    source = _load_yaml(
        PROJECT_ROOT / "config" / str(study["source_information_config"])
    )
    pipeline_document = _load_yaml(
        PROJECT_ROOT / "config" / str(study["source_pipeline_manifest"])
    )
    all_stages = pipeline_document["pipeline"]["stages"]
    prior_stages = select_pipeline_stages(all_stages, through_stage="15")
    artifact_audit = audit_pipeline_artifacts(PROJECT_ROOT, prior_stages)
    manifest = pd.DataFrame([asdict(row) for row in artifact_audit])
    if not bool(manifest["passed"].all()):
        raise RuntimeError("A required Stage 1--15 artifact failed the closure audit.")

    frozen = configuration["frozen_decision"]
    scalar_factor = float(frozen["scalar_inflation_factor"])
    stage15_parameters = pd.read_csv(STAGE15_PARAMETERS_PATH)
    imported_factors = stage15_parameters.loc[
        stage15_parameters["method"] == "scalar_inflation",
        "multiplicative_scale",
    ].to_numpy(dtype=np.float64)
    factor_difference = float(np.max(np.abs(imported_factors - scalar_factor)))
    factor_tolerance = float(frozen["scalar_factor_absolute_tolerance"])
    if factor_difference > factor_tolerance:
        raise RuntimeError("The frozen scalar factor does not match Stage 15.")

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
    measurement = source["measurement_superset"]
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
    scheme_name = str(study["fixed_sensor_scheme"])
    channel_indices = _scheme_channel_indices(
        observation_model,
        source["schemes"][scheme_name],
    )
    response = source["response_window"]
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
    state_scales = _state_scales(physical_model, source["state_scaling"])
    monitoring_sequence = source["monitoring"]
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
    loss_threshold = float(monitoring_sequence["meaningful_loss_fraction"])
    persistence = int(monitoring_sequence["persistence_windows"])
    alarm_threshold = float(frozen["alarm_probability_threshold"])
    nominal_coverage = float(frozen["nominal_coverage"])
    gaussian_quantile = float(norm.ppf(0.5 + 0.5 * nominal_coverage))
    uncertainty = UncertaintyCalibration(
        nominal_coverage=nominal_coverage,
        gaussian_quantile=gaussian_quantile,
        scalar_inflation=scalar_factor,
        zonewise_inflation=np.full(physical_model.n_zones, scalar_factor),
        additive_model_error_standard_deviation=np.zeros(physical_model.n_zones),
    )

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
        for case, specification in source["cases"].items()
    }

    def run_raw_history(
        replicate: int,
        case: str,
        theta_truth: np.ndarray,
        seed: int,
        dropout_probability: float,
    ):
        dataset = _generate_dataset(
            case_models[case],
            observation_model,
            theta_truth,
            noise_standard_deviation,
            dropout_probability,
            n_samples=n_samples,
            burn_in_samples=burn_in_samples,
            time_step_s=time_step,
            force_standard_deviation_n=force_standard_deviation,
            base_seed=seed,
            replicate=replicate,
        )
        return _run_remedy_filter(
            observation_model,
            dataset,
            theta_truth,
            theta_bounds,
            channel_indices,
            noise_covariance,
            state_scales,
            source["initial_uncertainty"],
            source["process_uncertainty"],
            source["filter"],
            time_step_s=time_step,
            force_standard_deviation_n=force_standard_deviation,
            metric_exclusion_samples=metric_exclusion_samples,
            loss_threshold=loss_threshold,
        )

    confirmation_started = time.perf_counter()
    confirmation = configuration["confirmation"]
    run_rows: list[dict[str, object]] = []
    zone_rows: list[dict[str, object]] = []
    for case, case_specification in source["cases"].items():
        for replicate in range(int(confirmation["damaged_replicates_per_case"])):
            raw_result = run_raw_history(
                replicate,
                case,
                damaged_theta,
                int(confirmation["damaged_random_seed"]),
                float(case_specification["independent_channel_dropout_probability"]),
            )
            means = raw_result.monitoring_means[:, physical_model.theta_slice]
            raw_scale = np.sqrt(
                np.diagonal(
                    raw_result.monitoring_covariances[
                        :,
                        physical_model.theta_slice,
                        physical_model.theta_slice,
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            for method in ("raw", "scalar_inflation"):
                adjusted_scale = calibrated_standard_deviations(
                    raw_scale,
                    method,
                    uncertainty,
                )
                adjusted_covariances = _adjust_covariances(
                    raw_result,
                    method,
                    uncertainty,
                    physical_model,
                )
                probabilities = _loss_probabilities(
                    raw_result.monitoring_means,
                    adjusted_covariances,
                    physical_model.theta_slice,
                    theta_bounds,
                    loss_threshold,
                )
                adjusted_result = replace(
                    raw_result,
                    monitoring_covariances=adjusted_covariances,
                    loss_probabilities=probabilities,
                )
                decision = _decision_metrics(
                    adjusted_result,
                    damaged_theta,
                    alarm_threshold,
                    loss_threshold,
                    persistence,
                )
                statistics = _interval_statistics(
                    means,
                    adjusted_scale,
                    damaged_theta,
                    uncertainty,
                    theta_bounds,
                )
                run_rows.append(
                    {
                        "replicate": replicate,
                        "case": case,
                        "method": method,
                        **statistics,
                        "theta_rmse": raw_result.stiffness_rmse,
                        "displacement_nrmse": raw_result.displacement_nrmse,
                        "velocity_nrmse": raw_result.velocity_nrmse,
                        "force_nrmse": raw_result.force_nrmse,
                        "acceleration_nrmse": raw_result.acceleration_nrmse,
                        **decision,
                    }
                )
                half_width = gaussian_quantile * adjusted_scale
                for zone, zone_name in enumerate(("lower", "middle", "upper")):
                    lower = np.maximum(
                        theta_bounds[zone, 0],
                        means[:, zone] - half_width[:, zone],
                    )
                    upper = np.minimum(
                        theta_bounds[zone, 1],
                        means[:, zone] + half_width[:, zone],
                    )
                    zone_rows.append(
                        {
                            "replicate": replicate,
                            "case": case,
                            "method": method,
                            "zone": zone_name,
                            "coverage": float(
                                np.mean(
                                    (damaged_theta[:, zone] >= lower)
                                    & (damaged_theta[:, zone] <= upper)
                                )
                            ),
                        }
                    )
    runs = pd.DataFrame(run_rows)
    zone_runs = pd.DataFrame(zone_rows)
    summary = _summarize_confirmation(runs)
    zones = _zone_summary(zone_runs)
    confirmation_seconds = time.perf_counter() - confirmation_started

    healthy_started = time.perf_counter()
    healthy_rows: list[dict[str, object]] = []
    for replicate in range(int(confirmation["healthy_replicates"])):
        raw_result = run_raw_history(
            replicate,
            "matched",
            healthy_theta,
            int(confirmation["healthy_random_seed"]),
            0.0,
        )
        for method in ("raw", "scalar_inflation"):
            adjusted_covariances = _adjust_covariances(
                raw_result,
                method,
                uncertainty,
                physical_model,
            )
            probabilities = _loss_probabilities(
                raw_result.monitoring_means,
                adjusted_covariances,
                physical_model.theta_slice,
                theta_bounds,
                loss_threshold,
            )
            alarms = first_persistent_alarm_indices(
                probabilities,
                alarm_threshold,
                persistence,
            )
            healthy_rows.append(
                {
                    "replicate": replicate,
                    "method": method,
                    "zone_false_alarm_fraction": float(np.mean(alarms >= 0)),
                    "structure_false_alarm": bool(np.any(alarms >= 0)),
                    "earliest_alarm_window": (
                        int(np.min(alarms[alarms >= 0]))
                        if np.any(alarms >= 0)
                        else -1
                    ),
                }
            )
    healthy = pd.DataFrame(healthy_rows)
    healthy_seconds = time.perf_counter() - healthy_started
    convergence = _create_convergence(runs, healthy)

    scalar_runs = runs[runs["method"] == "scalar_inflation"]
    raw_runs = runs[runs["method"] == "raw"]
    scalar_summary = summary[summary["method"] == "scalar_inflation"].set_index(
        "case"
    )
    scalar_zones = zones[zones["method"] == "scalar_inflation"]
    pooled_coverage = float(scalar_runs["coverage"].mean())
    pooled_coverage_interval = _mean_interval(
        scalar_runs["coverage"].to_numpy(dtype=np.float64)
    )
    stressed = scalar_summary.loc["stressed"]
    healthy_scalar = healthy[healthy["method"] == "scalar_inflation"]
    healthy_zone_failures = round(
        3 * healthy_scalar["zone_false_alarm_fraction"].sum()
    )
    healthy_zone_trials = 3 * len(healthy_scalar)
    healthy_zone_interval = _wilson_interval(
        healthy_zone_failures,
        healthy_zone_trials,
    )
    healthy_structure_failures = int(healthy_scalar["structure_false_alarm"].sum())
    healthy_structure_interval = _wilson_interval(
        healthy_structure_failures,
        len(healthy_scalar),
    )
    gates = configuration["confirmation_gates"]
    confirmation_checks = [
        (
            "pooled_stiffness_coverage_minimum",
            pooled_coverage,
            ">=",
            float(gates["minimum_pooled_stiffness_coverage"]),
        ),
        (
            "pooled_stiffness_coverage_maximum",
            pooled_coverage,
            "<=",
            float(gates["maximum_pooled_stiffness_coverage"]),
        ),
        (
            "minimum_case_stiffness_coverage",
            float(scalar_summary["coverage_mean"].min()),
            ">=",
            float(gates["minimum_case_stiffness_coverage"]),
        ),
        (
            "maximum_case_stiffness_coverage",
            float(scalar_summary["coverage_mean"].max()),
            "<=",
            float(gates["maximum_case_stiffness_coverage"]),
        ),
        (
            "minimum_zone_stiffness_coverage",
            float(scalar_zones["coverage_mean"].min()),
            ">=",
            float(gates["minimum_zone_stiffness_coverage"]),
        ),
        (
            "maximum_zone_stiffness_coverage",
            float(scalar_zones["coverage_mean"].max()),
            "<=",
            float(gates["maximum_zone_stiffness_coverage"]),
        ),
        (
            "pooled_coverage_lower_confidence_bound",
            pooled_coverage_interval[0],
            ">=",
            float(gates["minimum_coverage_lower_confidence_bound"]),
        ),
        (
            "mean_interval_width",
            float(scalar_runs["mean_interval_width"].mean()),
            "<=",
            float(gates["maximum_mean_interval_width"]),
        ),
        (
            "mean_theta_nees",
            float(scalar_runs["mean_theta_nees"].mean()),
            "<=",
            float(gates["maximum_mean_theta_nees"]),
        ),
        (
            "maximum_case_stiffness_rmse",
            float(scalar_summary["theta_rmse_mean"].max()),
            "<=",
            float(gates["maximum_stiffness_rmse"]),
        ),
        (
            "maximum_case_displacement_nrmse",
            float(scalar_summary["displacement_nrmse_mean"].max()),
            "<=",
            float(gates["maximum_displacement_nrmse"]),
        ),
        (
            "maximum_case_force_nrmse",
            float(scalar_summary["force_nrmse_mean"].max()),
            "<=",
            float(gates["maximum_force_nrmse"]),
        ),
        (
            "stressed_detection_fraction",
            float(stressed["detected_fraction"]),
            ">=",
            float(gates["minimum_stressed_detection_fraction"]),
        ),
        (
            "stressed_detection_wilson_lower",
            float(stressed["detected_wilson_lower"]),
            ">=",
            float(gates["minimum_stressed_detection_wilson_lower"]),
        ),
        (
            "stressed_localization_fraction",
            float(stressed["localized_fraction"]),
            ">=",
            float(gates["minimum_stressed_localization_fraction"]),
        ),
        (
            "stressed_localization_wilson_lower",
            float(stressed["localized_wilson_lower"]),
            ">=",
            float(gates["minimum_stressed_localization_wilson_lower"]),
        ),
        (
            "healthy_zone_false_alarm_fraction",
            healthy_zone_failures / healthy_zone_trials,
            "<=",
            float(gates["maximum_healthy_zone_false_alarm_fraction"]),
        ),
        (
            "healthy_zone_false_alarm_wilson_upper",
            healthy_zone_interval[1],
            "<=",
            float(gates["maximum_healthy_zone_false_alarm_wilson_upper"]),
        ),
        (
            "healthy_structure_false_alarm_fraction",
            healthy_structure_failures / len(healthy_scalar),
            "<=",
            float(gates["maximum_healthy_structure_false_alarm_fraction"]),
        ),
        (
            "healthy_structure_false_alarm_wilson_upper",
            healthy_structure_interval[1],
            "<=",
            float(gates["maximum_healthy_structure_false_alarm_wilson_upper"]),
        ),
        (
            "maximum_predamage_false_alarm_fraction",
            float(scalar_summary["predamage_false_alarm_fraction_mean"].max()),
            "<=",
            float(gates["maximum_predamage_false_alarm_fraction"]),
        ),
    ]
    check_rows = [
        {
            "category": "confirmation",
            "check": check,
            "observed": observed,
            "operator": operator,
            "criterion": criterion,
            "supported": (
                observed <= criterion if operator == "<=" else observed >= criterion
            ),
        }
        for check, observed, operator, criterion in confirmation_checks
    ]
    hypotheses = configuration["hypotheses"]
    raw_score = float(raw_runs["mean_interval_score"].mean())
    scalar_score = float(scalar_runs["mean_interval_score"].mean())
    artifact_pass_fraction = float(manifest["passed"].mean())
    scientific_checks = [
        (
            "scalar_pooled_coverage_minimum",
            pooled_coverage,
            ">=",
            float(hypotheses["minimum_scalar_coverage"]),
        ),
        (
            "scalar_pooled_coverage_maximum",
            pooled_coverage,
            "<=",
            float(hypotheses["maximum_scalar_coverage"]),
        ),
        (
            "raw_to_scalar_coverage_gain",
            pooled_coverage - float(raw_runs["coverage"].mean()),
            ">=",
            float(hypotheses["minimum_raw_to_scalar_coverage_gain"]),
        ),
        (
            "scalar_interval_score_ratio_to_raw",
            scalar_score / raw_score,
            "<=",
            float(hypotheses["maximum_scalar_interval_score_ratio_to_raw"]),
        ),
        (
            "scalar_mean_theta_nees",
            float(scalar_runs["mean_theta_nees"].mean()),
            "<=",
            float(hypotheses["maximum_scalar_mean_theta_nees"]),
        ),
        (
            "stressed_detection_fraction",
            float(stressed["detected_fraction"]),
            ">=",
            float(hypotheses["minimum_stressed_detection_fraction"]),
        ),
        (
            "healthy_zone_false_alarm_fraction",
            healthy_zone_failures / healthy_zone_trials,
            "<=",
            float(hypotheses["maximum_healthy_zone_false_alarm_fraction"]),
        ),
        (
            "phase1_artifact_pass_fraction",
            artifact_pass_fraction,
            ">=",
            float(hypotheses["minimum_phase1_artifact_pass_fraction"]),
        ),
    ]
    for check, observed, operator, criterion in scientific_checks:
        check_rows.append(
            {
                "category": "hypothesis",
                "check": check,
                "observed": observed,
                "operator": operator,
                "criterion": criterion,
                "supported": (
                    observed <= criterion
                    if operator == "<="
                    else observed >= criterion
                ),
            }
        )
    verification = configuration["verification"]
    numeric_columns = [
        "coverage",
        "mean_interval_width",
        "mean_interval_score",
        "mean_theta_nees",
        "theta_rmse",
        "displacement_nrmse",
        "force_nrmse",
    ]
    verification_checks = [
        (
            "prior_stage_count",
            float(len(prior_stages)),
            "==",
            float(verification["expected_prior_stages"]),
        ),
        (
            "audited_artifact_count",
            float(len(manifest)),
            "==",
            float(verification["expected_audited_artifacts"]),
        ),
        (
            "confirmation_row_count",
            float(len(runs)),
            "==",
            float(verification["expected_confirmation_rows"]),
        ),
        (
            "healthy_row_count",
            float(len(healthy)),
            "==",
            float(verification["expected_healthy_rows"]),
        ),
        (
            "damaged_history_count",
            float(runs[["case", "replicate"]].drop_duplicates().shape[0]),
            "==",
            float(verification["expected_damaged_histories"]),
        ),
        (
            "healthy_history_count",
            float(healthy["replicate"].nunique()),
            "==",
            float(verification["expected_healthy_histories"]),
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
            "frozen_scalar_factor_difference",
            factor_difference,
            "<=",
            factor_tolerance,
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
                    else observed <= criterion
                ),
            }
        )
    checks = pd.DataFrame(check_rows)
    total_seconds = time.perf_counter() - started
    runtime = pd.DataFrame(
        [
            {"phase": "confirmation", "seconds": confirmation_seconds},
            {"phase": "healthy_holdout", "seconds": healthy_seconds},
            {"phase": "total", "seconds": total_seconds},
        ]
    )
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RUNS_PATH, index=False, float_format="%.12g")
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    zones.to_csv(ZONE_PATH, index=False, float_format="%.12g")
    healthy.to_csv(HEALTHY_PATH, index=False, float_format="%.12g")
    convergence.to_csv(CONVERGENCE_PATH, index=False, float_format="%.12g")
    manifest.to_csv(MANIFEST_PATH, index=False)
    checks.to_csv(CHECKS_PATH, index=False, float_format="%.12g")
    runtime.to_csv(RUNTIME_PATH, index=False, float_format="%.12g")
    _create_figure(runs, summary, zones, healthy, convergence, checks, gates)

    confirmation_rows = checks[checks["category"] == "confirmation"]
    hypothesis_rows = checks[checks["category"] == "hypothesis"]
    verification_rows = checks[checks["category"] == "verification"]
    print("STAGE 16 - PHASE I CLOSURE AND LARGER FROZEN CONFIRMATION")
    print(f"Prior artifacts passing: {int(manifest['passed'].sum())}/{len(manifest)}")
    print(
        f"Scalar coverage: {pooled_coverage:.3f} "
        f"(95% history CI {pooled_coverage_interval[0]:.3f}, "
        f"{pooled_coverage_interval[1]:.3f})"
    )
    print(
        f"Raw coverage: {raw_runs['coverage'].mean():.3f}; "
        f"interval-score ratio: {scalar_score / raw_score:.3f}; "
        f"theta NEES: {scalar_runs['mean_theta_nees'].mean():.3f}"
    )
    print(
        f"Stressed detection/localisation: {stressed['detected_fraction']:.3f}/"
        f"{stressed['localized_fraction']:.3f}"
    )
    print(
        f"Healthy zone/structure false alarm: "
        f"{healthy_zone_failures / healthy_zone_trials:.3f}/"
        f"{healthy_structure_failures / len(healthy_scalar):.3f}"
    )
    print(
        "Confirmation gates supported: "
        f"{int(confirmation_rows['supported'].sum())}/{len(confirmation_rows)}"
    )
    print(
        "Predeclared hypotheses supported: "
        f"{int(hypothesis_rows['supported'].sum())}/{len(hypothesis_rows)}"
    )
    if not bool(verification_rows["supported"].all()):
        raise RuntimeError("Stage 16 mechanical verification failed.")
    print(f"Checks: {CHECKS_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS (Phase I closure experiment complete; failures retained)")


if __name__ == "__main__":
    main()
