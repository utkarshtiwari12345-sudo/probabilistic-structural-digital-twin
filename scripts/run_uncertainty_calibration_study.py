"""Run the Stage 15 independent posterior uncertainty-calibration study."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import t

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
    MultirateFilterResult,
    _decision_metrics,
    _loss_probabilities,
    _state_scales,
)
from run_robustness_factorial_study import _load_yaml  # noqa: E402
from run_sequential_tracking_study import (  # noqa: E402
    build_spectral_system,
    first_persistent_alarm_indices,
)

from probdt.deterioration import smooth_gradual_loss  # noqa: E402
from probdt.inference import (  # noqa: E402
    UNCERTAINTY_CALIBRATION_METHODS,
    AugmentedStructuralModel,
    MixedObservationModel,
    UncertaintyCalibration,
    calibrated_standard_deviations,
    fit_uncertainty_calibration,
    simulate_augmented_truth,
)
from probdt.structural import solve_modal_properties  # noqa: E402

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
STAGE15_CONFIG_PATH = PROJECT_ROOT / "config" / "uncertainty_calibration.yaml"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PARAMETERS_PATH = METRICS_DIR / "stage15_calibration_parameters.csv"
RUNS_PATH = METRICS_DIR / "stage15_uncertainty_runs.csv"
SUMMARY_PATH = METRICS_DIR / "stage15_uncertainty_summary.csv"
HEALTHY_PATH = METRICS_DIR / "stage15_healthy_holdout.csv"
MONITORING_PATH = METRICS_DIR / "stage15_monitoring_intervals.csv"
RECOVERY_PATH = METRICS_DIR / "stage15_uncertainty_recovery_gates.csv"
CHECKS_PATH = METRICS_DIR / "stage15_hypothesis_checks.csv"
RUNTIME_PATH = METRICS_DIR / "stage15_runtime.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage15_uncertainty_calibration.png"


@dataclass(frozen=True)
class PosteriorRun:
    """One unmodified combined-sensor UKF posterior and its known truth."""

    replicate: int
    case: str
    result: MultirateFilterResult
    theta_truth: np.ndarray


def _adjust_covariances(
    result: MultirateFilterResult,
    method: str,
    calibration: UncertaintyCalibration,
    model: AugmentedStructuralModel,
) -> np.ndarray:
    """Return PSD covariance corrections while retaining all posterior means."""

    adjusted = result.monitoring_covariances.copy()
    theta_block = adjusted[:, model.theta_slice, model.theta_slice]
    if method == "scalar_inflation":
        theta_block *= calibration.scalar_inflation**2
    elif method == "zonewise_inflation":
        scaling = np.diag(calibration.zonewise_inflation)
        theta_block[:] = np.einsum(
            "ij,tjk,kl->til",
            scaling,
            theta_block,
            scaling,
        )
    elif method == "additive_model_error":
        theta_block += np.diag(
            calibration.additive_model_error_standard_deviation**2
        )
    elif method != "raw":
        raise ValueError(f"Unknown uncertainty method: {method}.")
    adjusted[:, model.theta_slice, model.theta_slice] = theta_block
    return adjusted


def _interval_statistics(
    means: np.ndarray,
    standard_deviations: np.ndarray,
    truth: np.ndarray,
    calibration: UncertaintyCalibration,
    theta_bounds: np.ndarray,
) -> dict[str, float]:
    """Compute bounded interval coverage, width, score, and theta NEES."""

    half_width = calibration.gaussian_quantile * standard_deviations
    lower = np.maximum(theta_bounds[:, 0], means - half_width)
    upper = np.minimum(theta_bounds[:, 1], means + half_width)
    covered = (truth >= lower) & (truth <= upper)
    alpha = 1.0 - calibration.nominal_coverage
    interval_score = upper - lower
    interval_score += (2.0 / alpha) * (lower - truth) * (truth < lower)
    interval_score += (2.0 / alpha) * (truth - upper) * (truth > upper)
    nees = np.sum(((means - truth) / standard_deviations) ** 2, axis=1)
    return {
        "coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(upper - lower)),
        "mean_interval_score": float(np.mean(interval_score)),
        "mean_theta_nees": float(np.mean(nees)),
        "minimum_standard_deviation": float(np.min(standard_deviations)),
    }


def _confidence_interval(values: np.ndarray) -> tuple[float, float]:
    """Return a two-sided t interval across independent history metrics."""

    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean
    half_width = float(
        t.ppf(0.975, values.size - 1)
        * np.std(values, ddof=1)
        / np.sqrt(values.size)
    )
    return max(0.0, mean - half_width), min(1.0, mean + half_width)


def _summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize evaluation metrics and independent-history coverage intervals."""

    metrics = [
        "coverage",
        "mean_interval_width",
        "mean_interval_score",
        "mean_theta_nees",
        "theta_rmse",
        "posterior_mean_shift",
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
        lower, upper = _confidence_interval(
            selected["coverage"].to_numpy(dtype=np.float64)
        )
        row["coverage_confidence_lower"] = lower
        row["coverage_confidence_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_recovery_gates(
    runs: pd.DataFrame,
    healthy: pd.DataFrame,
    gates: dict[str, float],
) -> pd.DataFrame:
    """Apply frozen uncertainty-recovery gates to all calibration methods."""

    raw_score = float(
        runs.loc[runs["method"] == "raw", "mean_interval_score"].mean()
    )
    rows: list[dict[str, object]] = []
    definitions = (
        (
            "evaluation_coverage_minimum",
            "coverage",
            "mean",
            ">=",
            "minimum_evaluation_coverage",
        ),
        (
            "evaluation_coverage_maximum",
            "coverage",
            "mean",
            "<=",
            "maximum_evaluation_coverage",
        ),
        (
            "coverage_lower_confidence_bound",
            "coverage",
            "confidence_lower",
            ">=",
            "minimum_coverage_lower_confidence_bound",
        ),
        (
            "mean_interval_width",
            "mean_interval_width",
            "mean",
            "<=",
            "maximum_mean_interval_width",
        ),
        (
            "interval_score_ratio_to_raw",
            "mean_interval_score",
            "ratio",
            "<=",
            "maximum_interval_score_ratio_to_raw",
        ),
        (
            "stressed_detection_fraction",
            "detected",
            "stressed",
            ">=",
            "minimum_stressed_detection_fraction",
        ),
        (
            "predamage_false_alarm_fraction",
            "predamage_false_alarm_fraction",
            "max_case_mean",
            "<=",
            "maximum_predamage_false_alarm_fraction",
        ),
        (
            "posterior_mean_shift",
            "posterior_mean_shift",
            "maximum",
            "<=",
            "maximum_posterior_mean_shift",
        ),
    )
    for method, selected in runs.groupby("method", sort=False):
        for check, metric, reduction, operator, criterion_name in definitions:
            if reduction == "confidence_lower":
                observed = _confidence_interval(
                    selected[metric].to_numpy(dtype=np.float64)
                )[0]
            elif reduction == "ratio":
                observed = float(selected[metric].mean()) / raw_score
            elif reduction == "stressed":
                observed = float(
                    selected.loc[selected["case"] == "stressed", metric].mean()
                )
            elif reduction == "max_case_mean":
                observed = float(selected.groupby("case")[metric].mean().max())
            elif reduction == "maximum":
                observed = float(selected[metric].max())
            else:
                observed = float(selected[metric].mean())
            criterion = float(gates[criterion_name])
            passed = (
                observed <= criterion if operator == "<=" else observed >= criterion
            )
            rows.append(
                {
                    "method": method,
                    "check": check,
                    "observed": observed,
                    "operator": operator,
                    "criterion": criterion,
                    "passed": passed,
                }
            )
        false_alarm = float(
            healthy.loc[healthy["method"] == method, "zone_false_alarm_fraction"].mean()
        )
        criterion = float(gates["maximum_healthy_false_alarm_fraction"])
        rows.append(
            {
                "method": method,
                "check": "healthy_false_alarm_fraction",
                "observed": false_alarm,
                "operator": "<=",
                "criterion": criterion,
                "passed": false_alarm <= criterion,
            }
        )
    table = pd.DataFrame(rows)
    recovered = table.groupby("method", sort=False)["passed"].all()
    table["method_restores_uncertainty"] = table["method"].map(recovered)
    return table


def _create_figure(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    monitoring: pd.DataFrame,
    healthy: pd.DataFrame,
    recovery: pd.DataFrame,
    calibration: UncertaintyCalibration,
    gates: dict[str, float],
) -> None:
    """Plot holdout coverage, sharpness, intervals, and decision performance."""

    methods = list(summary["method"].drop_duplicates())
    labels = [value.replace("_", "\n") for value in methods]
    colors = ["0.5", "#0072B2", "#009E73", "#D55E00"]
    x = np.arange(len(methods))
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), constrained_layout=True)
    coverage_axis, score_axis, trajectory_axis, decision_axis = axes.ravel()

    pooled = summary.groupby("method", sort=False).agg(
        coverage=("coverage_mean", "mean"),
        width=("mean_interval_width_mean", "mean"),
        score=("mean_interval_score_mean", "mean"),
    )
    pooled = pooled.loc[methods]
    coverage_errors = []
    for method in methods:
        values = runs.loc[runs["method"] == method, "coverage"].to_numpy()
        lower, upper = _confidence_interval(values)
        method_coverage = float(pooled.loc[method, "coverage"])
        coverage_errors.append(
            [method_coverage - lower, upper - method_coverage]
        )
    coverage_axis.bar(x, pooled["coverage"], color=colors)
    coverage_axis.errorbar(
        x,
        pooled["coverage"],
        yerr=np.asarray(coverage_errors).T,
        fmt="none",
        color="black",
        capsize=4,
    )
    coverage_axis.axhline(
        calibration.nominal_coverage,
        color="black",
        linestyle=":",
        label="Nominal 95%",
    )
    coverage_axis.axhspan(
        float(gates["minimum_evaluation_coverage"]),
        float(gates["maximum_evaluation_coverage"]),
        color="#009E73",
        alpha=0.1,
        label="Recovery range",
    )
    coverage_axis.set_xticks(x, labels)
    coverage_axis.set_ylim(0.0, 1.05)
    coverage_axis.set_ylabel("Independent evaluation coverage")
    coverage_axis.set_title("Posterior interval calibration")
    coverage_axis.legend(fontsize=8)
    coverage_axis.grid(axis="y", alpha=0.3)

    raw_score = float(pooled.loc["raw", "score"])
    score_axis.bar(
        x - 0.18,
        pooled["width"],
        0.36,
        color=colors,
        label="Mean interval width",
    )
    ratio_axis = score_axis.twinx()
    ratio_axis.plot(
        x + 0.18,
        pooled["score"] / raw_score,
        color="black",
        marker="o",
        linewidth=1.4,
        label="Interval-score ratio",
    )
    ratio_axis.axhline(
        float(gates["maximum_interval_score_ratio_to_raw"]),
        color="#A60000",
        linestyle="--",
        linewidth=1.0,
    )
    score_axis.set_xticks(x, labels)
    score_axis.set_ylabel("Mean stiffness-interval width")
    ratio_axis.set_ylabel("Interval score / raw score")
    score_axis.set_title("Sharpness and proper interval score")
    score_axis.grid(axis="y", alpha=0.3)
    handles, legends = score_axis.get_legend_handles_labels()
    second_handles, second_legends = ratio_axis.get_legend_handles_labels()
    score_axis.legend(handles + second_handles, legends + second_legends, fontsize=8)

    recovered = recovery.groupby("method", sort=False)["passed"].all()
    eligible = [method for method in methods if method != "raw" and recovered[method]]
    if eligible:
        best_method = min(eligible, key=lambda value: float(pooled.loc[value, "score"]))
    else:
        best_method = min(
            methods[1:],
            key=lambda value: abs(
                float(pooled.loc[value, "coverage"]) - calibration.nominal_coverage
            ),
        )
    selected = monitoring[
        (monitoring["case"] == "stressed")
        & (monitoring["replicate"] == 0)
        & (monitoring["zone"] == "lower")
    ]
    raw = selected[selected["method"] == "raw"]
    best = selected[selected["method"] == best_method]
    trajectory_axis.fill_between(
        raw["time_day"],
        raw["credible_lower"],
        raw["credible_upper"],
        color="0.5",
        alpha=0.25,
        label="Raw 95% interval",
    )
    trajectory_axis.fill_between(
        best["time_day"],
        best["credible_lower"],
        best["credible_upper"],
        color="#0072B2",
        alpha=0.25,
        label=f"{best_method.replace('_', ' ')} interval",
    )
    trajectory_axis.plot(
        raw["time_day"],
        raw["posterior_mean"],
        color="#0072B2",
        marker="o",
        linewidth=1.3,
        label="Frozen UKF mean",
    )
    trajectory_axis.plot(
        raw["time_day"],
        raw["true_theta"],
        color="black",
        linewidth=1.8,
        label="Truth",
    )
    trajectory_axis.set_xlabel("Monitoring time (day)")
    trajectory_axis.set_ylabel("Lower-zone stiffness multiplier")
    trajectory_axis.set_title("Mean preserved; uncertainty corrected")
    trajectory_axis.legend(fontsize=8)
    trajectory_axis.grid(alpha=0.3)

    healthy_rates = [
        float(
            healthy.loc[
                healthy["method"] == method,
                "zone_false_alarm_fraction",
            ].mean()
        )
        for method in methods
    ]
    stressed_detection = [
        float(
            summary.loc[
                (summary["method"] == method)
                & (summary["case"] == "stressed"),
                "detected_mean",
            ].iloc[0]
        )
        for method in methods
    ]
    decision_axis.bar(
        x - 0.18,
        healthy_rates,
        0.36,
        color="#D55E00",
        label="Healthy false alarm",
    )
    decision_axis.bar(
        x + 0.18,
        stressed_detection,
        0.36,
        color="#009E73",
        label="Stressed detection",
    )
    decision_axis.axhline(
        float(gates["maximum_healthy_false_alarm_fraction"]),
        color="#A60000",
        linestyle="--",
        linewidth=1.0,
    )
    decision_axis.set_xticks(x, labels)
    decision_axis.set_ylim(0.0, 1.05)
    decision_axis.set_ylabel("Fraction")
    decision_axis.set_title(
        "Frozen decision performance\nRecovered: "
        + (", ".join(eligible) if eligible else "none")
    )
    decision_axis.legend(fontsize=8)
    decision_axis.grid(axis="y", alpha=0.3)

    figure.suptitle(
        "Stage 15: independent calibration of joint-UKF uncertainty",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main() -> None:
    """Execute independent calibration, damaged evaluation, and healthy holdout."""

    started = time.perf_counter()
    base_config = _load_yaml(BASE_CONFIG_PATH)
    monitoring_config = _load_yaml(MONITORING_CONFIG_PATH)
    configuration = _load_yaml(STAGE15_CONFIG_PATH)
    source_path = PROJECT_ROOT / "config" / str(
        configuration["study"]["source_information_config"]
    )
    source = _load_yaml(source_path)
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
    scheme_name = str(configuration["study"]["fixed_sensor_scheme"])
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
    ) -> MultirateFilterResult:
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

    calibration_started = time.perf_counter()
    interval_config = configuration["interval_calibration"]
    calibration_runs: list[PosteriorRun] = []
    for case, case_specification in source["cases"].items():
        for replicate in range(int(interval_config["replicates_per_case"])):
            result = run_raw_history(
                replicate,
                case,
                damaged_theta,
                int(interval_config["random_seed"]),
                float(case_specification["independent_channel_dropout_probability"]),
            )
            calibration_runs.append(
                PosteriorRun(replicate, case, result, damaged_theta.copy())
            )
    calibration_means = np.stack(
        [
            run.result.monitoring_means[:, physical_model.theta_slice]
            for run in calibration_runs
        ]
    )
    calibration_raw_standard_deviation = np.stack(
        [
            np.sqrt(
                np.diagonal(
                    run.result.monitoring_covariances[
                        :,
                        physical_model.theta_slice,
                        physical_model.theta_slice,
                    ],
                    axis1=1,
                    axis2=2,
                )
            )
            for run in calibration_runs
        ]
    )
    calibration_truth = np.stack([run.theta_truth for run in calibration_runs])
    calibration = fit_uncertainty_calibration(
        calibration_means,
        calibration_raw_standard_deviation,
        calibration_truth,
        nominal_coverage=float(interval_config["nominal_coverage"]),
    )
    methods = tuple(str(value) for value in interval_config["methods"])
    if methods != UNCERTAINTY_CALIBRATION_METHODS:
        raise RuntimeError(
            "Stage 15 methods must match the frozen implementation order."
        )
    parameter_rows: list[dict[str, object]] = []
    for method in methods:
        adjusted_scale = calibrated_standard_deviations(
            calibration_raw_standard_deviation,
            method,
            calibration,
        )
        statistics = _interval_statistics(
            calibration_means,
            adjusted_scale,
            calibration_truth,
            calibration,
            theta_bounds,
        )
        for zone, zone_name in enumerate(("lower", "middle", "upper")):
            parameter_rows.append(
                {
                    "method": method,
                    "zone": zone_name,
                    "nominal_coverage": calibration.nominal_coverage,
                    "gaussian_quantile": calibration.gaussian_quantile,
                    "multiplicative_scale": (
                        calibration.scalar_inflation
                        if method == "scalar_inflation"
                        else calibration.zonewise_inflation[zone]
                        if method == "zonewise_inflation"
                        else 1.0
                    ),
                    "additive_standard_deviation": (
                        calibration.additive_model_error_standard_deviation[zone]
                        if method == "additive_model_error"
                        else 0.0
                    ),
                    "pooled_calibration_coverage": statistics["coverage"],
                    "pooled_calibration_interval_width": statistics[
                        "mean_interval_width"
                    ],
                    "pooled_calibration_interval_score": statistics[
                        "mean_interval_score"
                    ],
                }
            )
    calibration_parameters = pd.DataFrame(parameter_rows)
    calibration_seconds = time.perf_counter() - calibration_started

    evaluation_started = time.perf_counter()
    evaluation_config = configuration["evaluation"]
    alarm_threshold = float(configuration["decision"]["alarm_probability_threshold"])
    run_rows: list[dict[str, object]] = []
    monitoring_rows: list[dict[str, object]] = []
    for case, case_specification in source["cases"].items():
        for replicate in range(int(evaluation_config["replicates_per_case"])):
            raw_result = run_raw_history(
                replicate,
                case,
                damaged_theta,
                int(evaluation_config["random_seed"]),
                float(case_specification["independent_channel_dropout_probability"]),
            )
            means = raw_result.monitoring_means[:, physical_model.theta_slice]
            raw_standard_deviation = np.sqrt(
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
            for method in methods:
                adjusted_standard_deviation = calibrated_standard_deviations(
                    raw_standard_deviation,
                    method,
                    calibration,
                )
                adjusted_covariances = _adjust_covariances(
                    raw_result,
                    method,
                    calibration,
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
                    adjusted_standard_deviation,
                    damaged_theta,
                    calibration,
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
                        "force_nrmse": raw_result.force_nrmse,
                        "posterior_mean_shift": 0.0,
                        **decision,
                    }
                )
                half_width = (
                    calibration.gaussian_quantile * adjusted_standard_deviation
                )
                for window in range(n_windows):
                    for zone, zone_name in enumerate(("lower", "middle", "upper")):
                        monitoring_rows.append(
                            {
                                "replicate": replicate,
                                "case": case,
                                "method": method,
                                "window": window,
                                "time_day": time_days[window],
                                "zone": zone_name,
                                "true_theta": damaged_theta[window, zone],
                                "posterior_mean": means[window, zone],
                                "posterior_standard_deviation": (
                                    adjusted_standard_deviation[window, zone]
                                ),
                                "credible_lower": max(
                                    theta_bounds[zone, 0],
                                    means[window, zone] - half_width[window, zone],
                                ),
                                "credible_upper": min(
                                    theta_bounds[zone, 1],
                                    means[window, zone] + half_width[window, zone],
                                ),
                                "loss_probability": probabilities[window, zone],
                            }
                        )
    runs = pd.DataFrame(run_rows)
    monitoring = pd.DataFrame(monitoring_rows)
    summary = _summarize_runs(runs)
    evaluation_seconds = time.perf_counter() - evaluation_started

    healthy_started = time.perf_counter()
    healthy_config = configuration["healthy_holdout"]
    healthy_rows: list[dict[str, object]] = []
    for replicate in range(int(healthy_config["replicates"])):
        raw_result = run_raw_history(
            replicate,
            "matched",
            healthy_theta,
            int(healthy_config["random_seed"]),
            0.0,
        )
        for method in methods:
            adjusted_covariances = _adjust_covariances(
                raw_result,
                method,
                calibration,
                physical_model,
            )
            probabilities = _loss_probabilities(
                raw_result.monitoring_means,
                adjusted_covariances,
                physical_model.theta_slice,
                theta_bounds,
                loss_threshold,
            )
            alarm_indices = first_persistent_alarm_indices(
                probabilities,
                alarm_threshold,
                persistence,
            )
            healthy_rows.append(
                {
                    "replicate": replicate,
                    "method": method,
                    "zone_false_alarm_fraction": float(np.mean(alarm_indices >= 0)),
                    "structure_false_alarm": bool(np.any(alarm_indices >= 0)),
                    "earliest_alarm_window": (
                        int(np.min(alarm_indices[alarm_indices >= 0]))
                        if np.any(alarm_indices >= 0)
                        else -1
                    ),
                }
            )
    healthy = pd.DataFrame(healthy_rows)
    healthy_seconds = time.perf_counter() - healthy_started

    gates = configuration["uncertainty_recovery_gates"]
    recovery = _evaluate_recovery_gates(runs, healthy, gates)
    recovered = recovery.groupby("method", sort=False)["passed"].all()
    corrected = [method for method in methods if method != "raw"]
    pooled = runs.groupby("method", sort=False).mean(numeric_only=True)
    raw_coverage = float(pooled.loc["raw", "coverage"])
    best_coverage = float(pooled.loc[corrected, "coverage"].max())
    raw_score = float(pooled.loc["raw", "mean_interval_score"])
    best_score = float(pooled.loc[corrected, "mean_interval_score"].min())
    operational_count = int(recovered.loc[corrected].sum())
    healthy_rates = healthy.groupby("method")["zone_false_alarm_fraction"].mean()
    stressed_detection = runs.loc[runs["case"] == "stressed"].groupby("method")[
        "detected"
    ].mean()
    hypotheses = configuration["hypotheses"]
    scientific_checks = [
        (
            "raw_evaluation_coverage",
            raw_coverage,
            "<=",
            float(hypotheses["maximum_raw_evaluation_coverage"]),
        ),
        (
            "best_corrected_evaluation_coverage",
            best_coverage,
            ">=",
            float(hypotheses["minimum_best_evaluation_coverage"]),
        ),
        (
            "best_interval_score_reduction_fraction",
            1.0 - best_score / raw_score,
            ">=",
            float(hypotheses["minimum_best_interval_score_reduction_fraction"]),
        ),
        (
            "operational_calibration_method_count",
            float(operational_count),
            ">=",
            float(hypotheses["minimum_operational_calibration_methods"]),
        ),
        (
            "best_healthy_false_alarm_fraction",
            float(healthy_rates.loc[corrected].min()),
            "<=",
            float(hypotheses["maximum_best_healthy_false_alarm_fraction"]),
        ),
        (
            "best_stressed_detection_fraction",
            float(stressed_detection.loc[corrected].max()),
            ">=",
            float(hypotheses["minimum_best_stressed_detection_fraction"]),
        ),
        (
            "maximum_posterior_mean_shift",
            float(runs["posterior_mean_shift"].max()),
            "<=",
            float(hypotheses["maximum_mean_shift"]),
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
                observed <= criterion if operator == "<=" else observed >= criterion
            ),
        }
        for check, observed, operator, criterion in scientific_checks
    ]
    verification = configuration["verification"]
    numeric_columns = [
        "coverage",
        "mean_interval_width",
        "mean_interval_score",
        "mean_theta_nees",
        "theta_rmse",
    ]
    verification_checks = [
        (
            "calibration_history_count",
            float(len(calibration_runs)),
            "==",
            float(verification["expected_calibration_histories"]),
        ),
        (
            "evaluation_run_count",
            float(len(runs)),
            "==",
            float(verification["expected_evaluation_runs"]),
        ),
        (
            "healthy_holdout_run_count",
            float(len(healthy)),
            "==",
            float(verification["expected_healthy_holdout_runs"]),
        ),
        (
            "method_count",
            float(runs["method"].nunique()),
            "==",
            float(verification["expected_methods"]),
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
            "minimum_adjusted_standard_deviation",
            float(runs["minimum_standard_deviation"].min()),
            ">",
            float(verification["minimum_adjusted_standard_deviation"]),
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
                    else observed > criterion
                    if operator == ">"
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
            {"phase": "healthy_holdout", "seconds": healthy_seconds},
            {"phase": "total", "seconds": total_seconds},
        ]
    )
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    calibration_parameters.to_csv(
        PARAMETERS_PATH,
        index=False,
        float_format="%.12g",
    )
    runs.to_csv(RUNS_PATH, index=False, float_format="%.12g")
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    healthy.to_csv(HEALTHY_PATH, index=False, float_format="%.12g")
    monitoring.to_csv(MONITORING_PATH, index=False, float_format="%.12g")
    recovery.to_csv(RECOVERY_PATH, index=False, float_format="%.12g")
    checks.to_csv(CHECKS_PATH, index=False, float_format="%.12g")
    runtime.to_csv(RUNTIME_PATH, index=False, float_format="%.12g")
    _create_figure(runs, summary, monitoring, healthy, recovery, calibration, gates)

    print("STAGE 15 - INDEPENDENT POSTERIOR UNCERTAINTY CALIBRATION")
    print(
        "Method                  | coverage | width | score/raw | healthy FA | "
        "stress detect | recovered"
    )
    print(
        "------------------------+----------+-------+-----------+------------+"
        "---------------+----------"
    )
    for method in methods:
        print(
            f"{method:<23} | {pooled.loc[method, 'coverage']:>8.3f} | "
            f"{pooled.loc[method, 'mean_interval_width']:>5.3f} | "
            f"{pooled.loc[method, 'mean_interval_score'] / raw_score:>9.3f} | "
            f"{healthy_rates[method]:>10.3f} | "
            f"{stressed_detection[method]:>13.3f} | "
            f"{bool(recovered[method])!s:>8}"
        )
    hypothesis_rows = checks[checks["category"] == "hypothesis"]
    print(
        "Predeclared hypotheses supported: "
        f"{int(hypothesis_rows['supported'].sum())}/{len(hypothesis_rows)}"
    )
    verification_rows = checks[checks["category"] == "verification"]
    if not bool(verification_rows["supported"].all()):
        raise RuntimeError("Stage 15 mechanical verification failed.")
    print(f"Recovered uncertainty methods: {operational_count}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Recovery gates: {RECOVERY_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS (uncertainty-calibration study complete; failures retained)")


if __name__ == "__main__":
    main()
