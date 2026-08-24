"""Run the Stage 7 UKF and EnKF comparison on the Stage 6 holdout data."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from run_sequential_tracking_study import (  # noqa: E402
    build_spectral_system,
    build_truth_histories,
    calibrate_alarm_thresholds,
    first_persistent_alarm_indices,
    generate_spectral_windows,
    make_forward_model,
    run_method,
)

from probdt.inference import (  # noqa: E402
    SequentialHistory,
    corrected_log_periodogram,
    log_periodogram_noise_variance,
    run_ensemble_parameter_filter,
    run_unscented_parameter_filter,
)
from probdt.monitoring import SensorLayout  # noqa: E402

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
TRACKING_CONFIG_PATH = PROJECT_ROOT / "config" / "sequential_tracking.yaml"
FILTER_CONFIG_PATH = PROJECT_ROOT / "config" / "nonlinear_filters.yaml"
RUNS_PATH = PROJECT_ROOT / "results" / "metrics" / "stage7_filter_runs.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "metrics" / "stage7_filter_summary.csv"
DETECTION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage7_detection_events.csv"
)
CALIBRATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage7_alarm_calibration.csv"
)
RUNTIME_PATH = PROJECT_ROOT / "results" / "metrics" / "stage7_runtime.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage7_filter_comparison.png"


def run_filter(
    method_name: str,
    raw_observations: np.ndarray,
    corrected_observations: np.ndarray,
    twin_forward,
    tracking_config: dict[str, object],
    filter_config: dict[str, object],
    rng: np.random.Generator,
) -> SequentialHistory:
    """Run one configured posterior approximation on one shared dataset."""

    if method_name == "laplace":
        process_name = str(
            filter_config["filters"]["laplace"]["stage6_process_model"]
        )
        return run_method(
            process_name,
            twin_forward,
            raw_observations,
            tracking_config,
        )

    parameter_config = tracking_config["parameter_model"]
    names = tuple(str(name) for name in parameter_config["names"])
    bounds = list(
        zip(
            parameter_config["lower_bounds"],
            parameter_config["upper_bounds"],
            strict=True,
        )
    )
    initial_mean = np.asarray(
        parameter_config["initial_prior_mean"],
        dtype=np.float64,
    )
    initial_scale = np.asarray(
        parameter_config["initial_prior_standard_deviation"],
        dtype=np.float64,
    )
    process_scale = np.asarray(
        filter_config["process_model"]["standard_deviation_per_window"],
        dtype=np.float64,
    )
    n_averages = int(
        tracking_config["monitoring_sequence"][
            "independent_periodogram_averages"
        ]
    )
    observation_variance = log_periodogram_noise_variance(n_averages)
    observation_covariance = np.eye(corrected_observations.shape[1]) * (
        observation_variance
    )

    def measurement_function(parameters: np.ndarray) -> np.ndarray:
        return np.log(twin_forward(parameters)).reshape(-1)

    common_arguments = {
        "observations": corrected_observations,
        "measurement_function": measurement_function,
        "parameter_names": names,
        "bounds": bounds,
        "initial_mean": initial_mean,
        "initial_covariance": np.diag(initial_scale**2),
        "process_covariance": np.diag(process_scale**2),
        "observation_covariance": observation_covariance,
        "credible_level": float(parameter_config["credible_level"]),
    }
    if method_name == "ukf":
        specification = filter_config["filters"]["ukf"]
        return run_unscented_parameter_filter(
            **common_arguments,
            alpha=float(specification["alpha"]),
            beta=float(specification["beta"]),
            kappa=float(specification["kappa"]),
        )
    specification = filter_config["filters"][method_name]
    return run_ensemble_parameter_filter(
        **common_arguments,
        n_ensemble=int(specification["ensemble_size"]),
        rng=rng,
    )


def filter_rng(
    master_seed: int,
    dataset_index: int,
    method_index: int,
) -> np.random.Generator:
    """Create a reproducible method-specific generator without global state."""

    sequence = np.random.SeedSequence(
        [int(master_seed), int(dataset_index), int(method_index)]
    )
    return np.random.default_rng(sequence)


def create_comparison_figure(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    runtime: pd.DataFrame,
    time_days: np.ndarray,
) -> None:
    """Plot representative trajectories, aggregate calibration, and cost."""

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), constrained_layout=True)
    gradual_axis, abrupt_axis, metrics_axis, runtime_axis = axes.ravel()
    colors = {
        "laplace": "#000000",
        "ukf": "#0072B2",
        "enkf_50": "#E69F00",
        "enkf_200": "#009E73",
    }
    labels = {
        "laplace": "Recursive Laplace",
        "ukf": "UKF",
        "enkf_50": "EnKF (50)",
        "enkf_200": "EnKF (200)",
    }

    for axis, scenario, zone, title in (
        (
            gradual_axis,
            "gradual_lower",
            "lower",
            "Gradual lower-zone deterioration",
        ),
        (
            abrupt_axis,
            "abrupt_middle",
            "middle",
            "Abrupt middle-zone deterioration",
        ),
    ):
        subset = runs[
            (runs["scenario"] == scenario)
            & (runs["replicate"] == 0)
            & (runs["zone"] == zone)
        ]
        truth = subset[subset["method"] == "laplace"]
        axis.plot(
            time_days,
            truth["true_parameter"],
            color="0.4",
            linewidth=3.0,
            label="Truth",
        )
        for method_name in colors:
            method_data = subset[subset["method"] == method_name]
            axis.plot(
                time_days,
                method_data["posterior_mean"],
                color=colors[method_name],
                linewidth=1.5,
                label=labels[method_name],
            )
        axis.axhline(0.95, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Monitoring time (days)")
        axis.set_ylabel("Stiffness multiplier")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    method_order = list(colors)
    positions = np.arange(len(method_order))
    rmse_values = [
        summary.set_index("method").loc[method, "pooled_rmse"]
        for method in method_order
    ]
    coverage_values = [
        summary.set_index("method").loc[method, "interior_coverage"]
        for method in method_order
    ]
    metrics_axis.bar(
        positions - 0.18,
        rmse_values,
        width=0.36,
        color=[colors[method] for method in method_order],
        alpha=0.85,
        label="RMSE",
    )
    coverage_scaled = np.asarray(coverage_values) * 0.03
    metrics_axis.bar(
        positions + 0.18,
        coverage_scaled,
        width=0.36,
        color="0.75",
        edgecolor="black",
        label="Coverage / 33.3",
    )
    for position, rmse, coverage in zip(
        positions,
        rmse_values,
        coverage_values,
        strict=True,
    ):
        metrics_axis.text(
            position - 0.18,
            rmse + 0.0007,
            f"{rmse:.3f}",
            ha="center",
            fontsize=8,
        )
        metrics_axis.text(
            position + 0.18,
            0.03 * coverage + 0.0007,
            f"{coverage:.0%}",
            ha="center",
            fontsize=8,
        )
    metrics_axis.set_xticks(
        positions,
        labels=[labels[method] for method in method_order],
        rotation=15,
    )
    metrics_axis.set_ylabel("RMSE (coverage labels shown above grey bars)")
    metrics_axis.set_title("Holdout accuracy and interval calibration")
    metrics_axis.legend(fontsize=8)
    metrics_axis.grid(axis="y", alpha=0.25)

    runtime_summary = runtime.groupby("method", sort=False)["runtime_s"].mean()
    summary_index = summary.set_index("method")
    for method_name in method_order:
        runtime_axis.scatter(
            runtime_summary.loc[method_name],
            summary_index.loc[method_name, "pooled_rmse"],
            s=90,
            color=colors[method_name],
            edgecolor="black",
            label=labels[method_name],
        )
    runtime_axis.set_xlabel("Mean runtime per 31-window sequence (s)")
    runtime_axis.set_ylabel("Pooled RMSE")
    runtime_axis.set_title("Accuracy--cost trade-off")
    runtime_axis.grid(alpha=0.25)
    runtime_axis.legend(fontsize=8)

    figure.suptitle(
        "Stage 7: UKF and EnKF approximation comparison",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Execute healthy calibration and matched holdout filter comparison."""

    with BASE_CONFIG_PATH.open(encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    with MONITORING_CONFIG_PATH.open(encoding="utf-8") as stream:
        monitoring_config = yaml.safe_load(stream)
    with TRACKING_CONFIG_PATH.open(encoding="utf-8") as stream:
        tracking_config = yaml.safe_load(stream)
    with FILTER_CONFIG_PATH.open(encoding="utf-8") as stream:
        filter_config = yaml.safe_load(stream)

    sequence_config = tracking_config["monitoring_sequence"]
    parameter_config = tracking_config["parameter_model"]
    alarm_config = tracking_config["alarm"]
    replication_config = tracking_config["replication"]
    excitation_config = monitoring_config["excitation"]
    n_windows = int(sequence_config["number_of_windows"])
    time_days = (
        np.arange(n_windows, dtype=np.float64)
        * float(sequence_config["update_interval_days"])
    )
    frequencies_hz = np.linspace(
        float(sequence_config["minimum_frequency_hz"]),
        float(sequence_config["maximum_frequency_hz"]),
        int(sequence_config["number_of_frequencies"]),
    )
    force_distribution = np.asarray(
        excitation_config["spatial_distribution"],
        dtype=np.float64,
    )
    layout_name = str(sequence_config["sensor_layout"])
    layout = SensorLayout(
        layout_name,
        tuple(
            int(index) - 1
            for index in monitoring_config["sensor_layouts"][layout_name][
                "dofs_one_based"
            ]
        ),
    )
    parameter_names = tuple(str(name) for name in parameter_config["names"])
    zone_names = tuple(name.removeprefix("theta_") for name in parameter_names)
    truth_system = build_spectral_system(base_config)
    twin_system = build_spectral_system(base_config)
    zero_noise = np.zeros(layout.n_sensors, dtype=np.float64)
    healthy_clean_forward = make_forward_model(
        truth_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        zero_noise,
    )
    healthy_psd = healthy_clean_forward(np.ones(len(zone_names)))
    noise_fraction = float(sequence_config["noise_amplitude_fraction"])
    measurement_noise_psd = noise_fraction**2 * np.max(healthy_psd, axis=0)
    truth_forward = make_forward_model(
        truth_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        measurement_noise_psd,
    )
    twin_forward = make_forward_model(
        twin_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        measurement_noise_psd,
    )
    histories = build_truth_histories(tracking_config, time_days, zone_names)
    methods = list(filter_config["filters"])
    method_indices = {name: index for index, name in enumerate(methods)}
    n_averages = int(sequence_config["independent_periodogram_averages"])
    loss_threshold = float(alarm_config["meaningful_loss_fraction"])
    persistence = int(alarm_config["persistence_windows"])

    calibration_probabilities: dict[str, list[np.ndarray]] = {
        method: [] for method in methods
    }
    calibration_data_seeds = np.random.SeedSequence(
        int(replication_config["calibration_seed"])
    ).spawn(int(replication_config["healthy_calibration_replicates"]))
    healthy_truth = histories["healthy"].stiffness_multipliers
    for dataset_index, data_seed in enumerate(calibration_data_seeds):
        raw_observations = generate_spectral_windows(
            truth_forward,
            healthy_truth,
            n_averages,
            np.random.default_rng(data_seed),
        )
        corrected = corrected_log_periodogram(raw_observations, n_averages).reshape(
            n_windows,
            -1,
        )
        for method_name in methods:
            history = run_filter(
                method_name,
                raw_observations,
                corrected,
                twin_forward,
                tracking_config,
                filter_config,
                filter_rng(
                    int(filter_config["random_seeds"]["calibration_filter_seed"]),
                    dataset_index,
                    method_indices[method_name],
                ),
            )
            calibration_probabilities[method_name].append(
                history.loss_exceedance_probability(loss_threshold)
            )
    alarm_thresholds, calibration = calibrate_alarm_thresholds(
        calibration_probabilities,
        [float(value) for value in alarm_config["candidate_probability_thresholds"]],
        persistence,
        float(alarm_config["maximum_calibration_false_alarm_fraction"]),
    )

    evaluation_replicates = int(replication_config["evaluation_replicates"])
    data_seeds = np.random.SeedSequence(
        int(replication_config["evaluation_seed"])
    ).spawn(len(histories) * evaluation_replicates)
    data_seed_iterator = iter(data_seeds)
    run_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    dataset_index = 0
    for scenario_name, truth_history in histories.items():
        truth_parameters = truth_history.stiffness_multipliers
        for replicate in range(evaluation_replicates):
            data_seed = next(data_seed_iterator)
            raw_observations = generate_spectral_windows(
                truth_forward,
                truth_parameters,
                n_averages,
                np.random.default_rng(data_seed),
            )
            corrected = corrected_log_periodogram(
                raw_observations,
                n_averages,
            ).reshape(n_windows, -1)
            for method_name in methods:
                start_time = perf_counter()
                history = run_filter(
                    method_name,
                    raw_observations,
                    corrected,
                    twin_forward,
                    tracking_config,
                    filter_config,
                    filter_rng(
                        int(
                            filter_config["random_seeds"][
                                "evaluation_filter_seed"
                            ]
                        ),
                        dataset_index,
                        method_indices[method_name],
                    ),
                )
                elapsed = perf_counter() - start_time
                runtime_rows.append(
                    {
                        "scenario": scenario_name,
                        "replicate": replicate,
                        "method": method_name,
                        "runtime_s": elapsed,
                        "forward_model_evaluations": int(
                            np.sum(history.n_function_evaluations)
                        ),
                    }
                )
                probabilities = history.loss_exceedance_probability(loss_threshold)
                alarm_indices = first_persistent_alarm_indices(
                    probabilities,
                    alarm_thresholds[method_name],
                    persistence,
                )
                posterior_scale = history.posterior_standard_deviations
                covariance_eigenvalues = np.linalg.eigvalsh(
                    history.posterior_covariances
                )
                for zone_index, zone_name in enumerate(zone_names):
                    truth_zone = truth_parameters[:, zone_index]
                    crossing_candidates = np.flatnonzero(
                        1.0 - truth_zone > loss_threshold
                    )
                    crossing_index = (
                        int(crossing_candidates[0])
                        if crossing_candidates.size
                        else -1
                    )
                    alarm_index = int(alarm_indices[zone_index])
                    false_alarm = alarm_index >= 0 and (
                        crossing_index < 0 or alarm_index < crossing_index
                    )
                    detected = crossing_index >= 0 and alarm_index >= crossing_index
                    detection_rows.append(
                        {
                            "scenario": scenario_name,
                            "replicate": replicate,
                            "method": method_name,
                            "zone": zone_name,
                            "truth_crossing_window": crossing_index,
                            "alarm_window": alarm_index,
                            "detection_delay_windows": (
                                alarm_index - crossing_index
                                if detected
                                else np.nan
                            ),
                            "false_alarm": false_alarm,
                            "detected": detected,
                            "damage_zone": crossing_index >= 0,
                        }
                    )
                    for window_index in range(n_windows):
                        truth_value = float(truth_zone[window_index])
                        posterior_mean = float(
                            history.posterior_means[window_index, zone_index]
                        )
                        run_rows.append(
                            {
                                "scenario": scenario_name,
                                "replicate": replicate,
                                "method": method_name,
                                "window": window_index,
                                "time_day": time_days[window_index],
                                "zone": zone_name,
                                "true_parameter": truth_value,
                                "posterior_mean": posterior_mean,
                                "error": posterior_mean - truth_value,
                                "posterior_standard_deviation": (
                                    posterior_scale[window_index, zone_index]
                                ),
                                "credible_lower": history.credible_interval_lower[
                                    window_index,
                                    zone_index,
                                ],
                                "credible_upper": history.credible_interval_upper[
                                    window_index,
                                    zone_index,
                                ],
                                "covered": history.credible_interval_lower[
                                    window_index,
                                    zone_index,
                                ]
                                <= truth_value
                                <= history.credible_interval_upper[
                                    window_index,
                                    zone_index,
                                ],
                                "damage_probability": probabilities[
                                    window_index,
                                    zone_index,
                                ],
                                "covariance_minimum_eigenvalue": (
                                    covariance_eigenvalues[window_index, 0]
                                ),
                            }
                        )
            dataset_index += 1

    runs = pd.DataFrame(run_rows)
    detections = pd.DataFrame(detection_rows)
    runtime = pd.DataFrame(runtime_rows)
    summary_rows: list[dict[str, object]] = []
    for method_name in methods:
        method_runs = runs[runs["method"] == method_name]
        method_events = detections[detections["method"] == method_name]
        interior = method_runs[method_runs["true_parameter"] < 0.999]
        damage_events = method_events[method_events["damage_zone"]]
        nondamage_events = method_events[~method_events["damage_zone"]]
        abrupt_events = damage_events[
            damage_events["scenario"] == "abrupt_middle"
        ]
        summary_rows.append(
            {
                "method": method_name,
                "alarm_probability_threshold": alarm_thresholds[method_name],
                "pooled_rmse": np.sqrt(np.mean(method_runs["error"] ** 2)),
                "pooled_bias": np.mean(method_runs["error"]),
                "interior_coverage": np.mean(interior["covered"]),
                "mean_posterior_standard_deviation": np.mean(
                    method_runs["posterior_standard_deviation"]
                ),
                "nondamage_false_alarm_fraction": np.mean(
                    nondamage_events["false_alarm"]
                ),
                "damage_detection_fraction": np.mean(damage_events["detected"]),
                "mean_detection_delay_windows": np.mean(
                    damage_events.loc[
                        damage_events["detected"],
                        "detection_delay_windows",
                    ]
                ),
                "abrupt_detection_delay_windows": np.mean(
                    abrupt_events.loc[
                        abrupt_events["detected"],
                        "detection_delay_windows",
                    ]
                ),
                "minimum_covariance_eigenvalue": np.min(
                    method_runs["covariance_minimum_eigenvalue"]
                ),
                "mean_runtime_s": np.mean(
                    runtime.loc[runtime["method"] == method_name, "runtime_s"]
                ),
                "mean_forward_model_evaluations": np.mean(
                    runtime.loc[
                        runtime["method"] == method_name,
                        "forward_model_evaluations",
                    ]
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    indexed = summary.set_index("method")
    summary["rmse_ratio_to_laplace"] = (
        summary["pooled_rmse"] / indexed.loc["laplace", "pooled_rmse"]
    )
    indexed = summary.set_index("method")
    verification = filter_config["verification"]
    comparison_methods = ("ukf", "enkf_200")
    checks = {
        "UKF tracking RMSE": indexed.loc["ukf", "pooled_rmse"]
        <= float(verification["maximum_ukf_pooled_rmse"]),
        "UKF interval coverage": indexed.loc["ukf", "interior_coverage"]
        >= float(verification["minimum_ukf_interior_coverage"]),
        "EnKF-200 tracking RMSE": indexed.loc["enkf_200", "pooled_rmse"]
        <= float(verification["maximum_enkf_200_pooled_rmse"]),
        "EnKF-200 interval coverage": indexed.loc[
            "enkf_200", "interior_coverage"
        ]
        >= float(verification["minimum_enkf_200_interior_coverage"]),
        "nondamage false alarms controlled": all(
            indexed.loc[method, "nondamage_false_alarm_fraction"]
            <= float(verification["maximum_nondamage_false_alarm_fraction"])
            for method in comparison_methods
        ),
        "damage detection retained": all(
            indexed.loc[method, "damage_detection_fraction"]
            >= float(verification["minimum_damage_detection_fraction"])
            for method in comparison_methods
        ),
        "abrupt detection timely": all(
            indexed.loc[method, "abrupt_detection_delay_windows"]
            <= float(verification["maximum_abrupt_detection_delay_windows"])
            for method in comparison_methods
        ),
        "EnKF ensemble-size convergence": abs(
            indexed.loc["enkf_50", "pooled_rmse"]
            - indexed.loc["enkf_200", "pooled_rmse"]
        )
        <= float(verification["maximum_enkf_size_rmse_difference"]),
        "EnKF-200 remains near exact-likelihood baseline": (
            indexed.loc["enkf_200", "rmse_ratio_to_laplace"]
            <= float(verification["maximum_enkf_200_rmse_ratio_to_laplace"])
        ),
        "UKF reference-ratio failure retained": (
            indexed.loc["ukf", "rmse_ratio_to_laplace"]
            >= float(verification["minimum_ukf_rmse_ratio_to_laplace"])
        ),
        "posterior covariances positive": float(
            np.min(summary["minimum_covariance_eigenvalue"])
        )
        > float(verification["minimum_covariance_eigenvalue"]),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RUNS_PATH, index=False, float_format="%.12g")
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    detections.to_csv(DETECTION_PATH, index=False, float_format="%.12g")
    calibration.to_csv(CALIBRATION_PATH, index=False, float_format="%.12g")
    runtime.to_csv(RUNTIME_PATH, index=False, float_format="%.12g")
    create_comparison_figure(runs, summary, runtime, time_days)

    print("STAGE 7 - UKF AND ENKF APPROXIMATION COMPARISON")
    print("Shared data: Stage 6 holdout sequences")
    print(
        f"Log-periodogram variance: {log_periodogram_noise_variance(n_averages):.6f}"
    )
    print("")
    print(
        "Method    | RMSE    | Coverage | False alarm | Detection | "
        "Abrupt delay | Runtime"
    )
    print(
        "----------+---------+----------+-------------+-----------+"
        "--------------+--------"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.method:<9} | {row.pooled_rmse:>7.4f} | "
            f"{row.interior_coverage:>8.3f} | "
            f"{row.nondamage_false_alarm_fraction:>11.3f} | "
            f"{row.damage_detection_fraction:>9.3f} | "
            f"{row.abrupt_detection_delay_windows:>12.2f} | "
            f"{row.mean_runtime_s:>6.2f}s"
        )
    print("")
    print(
        "Retained UKF limitation: RMSE ratio to Laplace = "
        f"{indexed.loc['ukf', 'rmse_ratio_to_laplace']:.3f}"
    )
    print(f"Runs: {RUNS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Detection events: {DETECTION_PATH}")
    print(f"Alarm calibration: {CALIBRATION_PATH}")
    print(f"Runtime: {RUNTIME_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    if failed_checks:
        raise RuntimeError(f"Stage 7 checks failed: {', '.join(failed_checks)}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
