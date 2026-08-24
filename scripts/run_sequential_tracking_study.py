"""Run the Stage 6 sequential Bayesian deterioration-tracking study."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from probdt.deterioration import (  # noqa: E402
    DeteriorationHistory,
    abrupt_loss,
    build_deterioration_history,
    combine_independent_losses,
    smooth_gradual_loss,
)
from probdt.diagnostics import acceleration_auto_psd  # noqa: E402
from probdt.inference import (  # noqa: E402
    SequentialHistory,
    fit_batch_spectral_map,
    run_recursive_spectral_map,
    simulate_averaged_periodogram,
)
from probdt.monitoring import SensorLayout  # noqa: E402
from probdt.structural import (  # noqa: E402
    apply_stiffness_multipliers,
    assemble_mass_matrix,
    assemble_rayleigh_damping,
    assemble_stiffness_matrix,
    assemble_zonal_stiffness_components,
    fit_rayleigh_coefficients,
    solve_modal_properties,
)

FloatArray = NDArray[np.float64]
SpectralForwardModel = Callable[[FloatArray], FloatArray]

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
TRACKING_CONFIG_PATH = PROJECT_ROOT / "config" / "sequential_tracking.yaml"
RUNS_PATH = PROJECT_ROOT / "results" / "metrics" / "stage6_tracking_runs.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "metrics" / "stage6_method_summary.csv"
DETECTION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage6_detection_events.csv"
)
CALIBRATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage6_alarm_calibration.csv"
)
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage6_sequential_tracking.png"


@dataclass(frozen=True)
class SpectralSystem:
    """Fresh structural matrices for either truth or inference."""

    mass_matrix_kg: FloatArray
    damping_matrix_n_s_per_m: FloatArray
    zone_components_n_per_m: FloatArray


def build_spectral_system(base_config: dict[str, object]) -> SpectralSystem:
    """Construct one independently allocated matched-model system."""

    structural = base_config["structural_properties"]
    damping_config = base_config["damping"]
    mass = assemble_mass_matrix(structural["masses_kg"])
    healthy_stiffness = assemble_stiffness_matrix(
        structural["story_stiffness_n_per_m"]
    )
    _, zone_components = assemble_zonal_stiffness_components(
        structural["story_stiffness_n_per_m"],
        structural["stiffness_zones"],
    )
    modes = solve_modal_properties(mass, healthy_stiffness)
    mode_indices = [int(value) - 1 for value in damping_config["target_modes"]]
    damping_ratios = [
        float(value) for value in damping_config["target_modal_damping_ratios"]
    ]
    alpha, beta = fit_rayleigh_coefficients(
        modes.angular_frequencies_rad_s[mode_indices[0]],
        modes.angular_frequencies_rad_s[mode_indices[1]],
        damping_ratios[0],
        damping_ratios[1],
    )
    damping = assemble_rayleigh_damping(
        mass,
        healthy_stiffness,
        alpha_per_s=alpha,
        beta_s=beta,
    ).damping_matrix_n_s_per_m
    return SpectralSystem(
        mass_matrix_kg=mass.copy(),
        damping_matrix_n_s_per_m=damping.copy(),
        zone_components_n_per_m=zone_components.copy(),
    )


def make_forward_model(
    system: SpectralSystem,
    frequencies_hz: FloatArray,
    force_distribution: FloatArray,
    layout: SensorLayout,
    excitation_config: dict[str, object],
    measurement_noise_psd: FloatArray,
) -> SpectralForwardModel:
    """Create a fixed-nuisance zonal acceleration-spectrum model."""

    def forward(parameters: FloatArray) -> FloatArray:
        stiffness = apply_stiffness_multipliers(
            system.zone_components_n_per_m,
            parameters,
        )
        return acceleration_auto_psd(
            system.mass_matrix_kg,
            system.damping_matrix_n_s_per_m,
            stiffness,
            frequencies_hz,
            force_distribution,
            layout.dof_indices,
            force_standard_deviation_n=float(
                excitation_config["stationary_standard_deviation_n"]
            ),
            force_correlation_time_s=float(
                excitation_config["correlation_time_s"]
            ),
            measurement_noise_psd=measurement_noise_psd,
        )

    return forward


def build_truth_histories(
    tracking_config: dict[str, object],
    time_days: FloatArray,
    zone_names: tuple[str, ...],
) -> dict[str, DeteriorationHistory]:
    """Construct configured non-healing histories on the slow monitoring time."""

    seconds_per_day = 86400.0
    time_s = seconds_per_day * time_days
    histories: dict[str, DeteriorationHistory] = {}
    for scenario_name, specification in tracking_config["truth_scenarios"].items():
        zone_losses: dict[str, FloatArray] = {}
        for zone_name, processes in specification["zone_processes"].items():
            components: list[FloatArray] = []
            for process in processes:
                if process["process"] == "smooth_gradual":
                    components.append(
                        smooth_gradual_loss(
                            time_s,
                            onset_s=float(process["onset_day"]) * seconds_per_day,
                            completion_s=(
                                float(process["completion_day"]) * seconds_per_day
                            ),
                            final_loss_fraction=float(
                                process["final_loss_fraction"]
                            ),
                        )
                    )
                elif process["process"] == "abrupt":
                    components.append(
                        abrupt_loss(
                            time_s,
                            event_time_s=(
                                float(process["event_day"]) * seconds_per_day
                            ),
                            loss_fraction=float(process["loss_fraction"]),
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported Stage 6 truth process {process['process']!r}."
                    )
            zone_losses[zone_name] = (
                components[0]
                if len(components) == 1
                else combine_independent_losses(components)
            )
        histories[str(scenario_name)] = build_deterioration_history(
            str(scenario_name),
            time_s,
            zone_names,
            zone_losses,
        )
    return histories


def generate_spectral_windows(
    truth_forward_model: SpectralForwardModel,
    truth_parameters: FloatArray,
    n_independent_averages: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate one independent averaged-periodogram observation per window."""

    return np.stack(
        [
            simulate_averaged_periodogram(
                truth_forward_model(parameters),
                n_independent_averages,
                rng,
            )
            for parameters in truth_parameters
        ]
    )


def run_independent_window_baseline(
    forward_model: SpectralForwardModel,
    observations: FloatArray,
    parameter_names: tuple[str, ...],
    bounds: list[tuple[float, float]],
    initial_prior_mean: FloatArray,
    initial_prior_covariance: FloatArray,
    n_independent_averages: int,
    credible_level: float,
    derivative_relative_step: float,
    optimizer_max_iterations: int,
) -> SequentialHistory:
    """Fit every monitoring window from the same weak prior."""

    n_windows = observations.shape[0]
    n_parameters = len(parameter_names)
    posterior_means = np.empty((n_windows, n_parameters), dtype=np.float64)
    posterior_covariances = np.empty(
        (n_windows, n_parameters, n_parameters),
        dtype=np.float64,
    )
    lower_intervals = np.empty_like(posterior_means)
    upper_intervals = np.empty_like(posterior_means)
    success = np.empty(n_windows, dtype=np.bool_)
    evaluations = np.empty(n_windows, dtype=np.int64)
    for index, observation in enumerate(observations):
        estimate = fit_batch_spectral_map(
            forward_model,
            observation,
            parameter_names,
            bounds,
            prior_mean=initial_prior_mean,
            prior_standard_deviation=None,
            prior_covariance=initial_prior_covariance,
            n_independent_averages=n_independent_averages,
            initial_points=[initial_prior_mean],
            credible_level=credible_level,
            derivative_relative_step=derivative_relative_step,
            optimizer_max_iterations=optimizer_max_iterations,
        )
        posterior_means[index] = estimate.map_parameters
        posterior_covariances[index] = estimate.posterior_covariance
        lower_intervals[index] = estimate.credible_interval_lower
        upper_intervals[index] = estimate.credible_interval_upper
        success[index] = estimate.optimizer_success
        evaluations[index] = estimate.n_function_evaluations
    return SequentialHistory(
        parameter_names=parameter_names,
        parameter_bounds=np.asarray(bounds, dtype=np.float64),
        prior_means=np.broadcast_to(initial_prior_mean, posterior_means.shape).copy(),
        prior_covariances=np.broadcast_to(
            initial_prior_covariance,
            posterior_covariances.shape,
        ).copy(),
        posterior_means=posterior_means,
        posterior_covariances=posterior_covariances,
        credible_interval_lower=lower_intervals,
        credible_interval_upper=upper_intervals,
        optimizer_success=success,
        n_function_evaluations=evaluations,
    )


def run_method(
    method_name: str,
    forward_model: SpectralForwardModel,
    observations: FloatArray,
    tracking_config: dict[str, object],
) -> SequentialHistory:
    """Run either the independent baseline or one recursive process model."""

    parameter_config = tracking_config["parameter_model"]
    monitoring_config = tracking_config["monitoring_sequence"]
    optimizer_config = tracking_config["optimizer"]
    names = tuple(str(name) for name in parameter_config["names"])
    bounds = list(
        zip(
            parameter_config["lower_bounds"],
            parameter_config["upper_bounds"],
            strict=True,
        )
    )
    prior_mean = np.asarray(
        parameter_config["initial_prior_mean"],
        dtype=np.float64,
    )
    prior_scale = np.asarray(
        parameter_config["initial_prior_standard_deviation"],
        dtype=np.float64,
    )
    prior_covariance = np.diag(prior_scale**2)
    common_arguments = {
        "forward_model": forward_model,
        "observations": observations,
        "parameter_names": names,
        "bounds": bounds,
        "initial_prior_mean": prior_mean,
        "initial_prior_covariance": prior_covariance,
        "n_independent_averages": int(
            monitoring_config["independent_periodogram_averages"]
        ),
        "credible_level": float(parameter_config["credible_level"]),
        "derivative_relative_step": float(
            optimizer_config["derivative_relative_step"]
        ),
        "optimizer_max_iterations": int(optimizer_config["maximum_iterations"]),
    }
    if method_name == "independent":
        return run_independent_window_baseline(**common_arguments)

    process_scale = np.asarray(
        tracking_config["process_models"][method_name][
            "standard_deviation_per_window"
        ],
        dtype=np.float64,
    )
    recursive_arguments = common_arguments.copy()
    recursive_arguments["observed_psd_windows"] = recursive_arguments.pop(
        "observations"
    )
    return run_recursive_spectral_map(
        **recursive_arguments,
        process_covariance=np.diag(process_scale**2),
    )


def first_persistent_alarm_indices(
    probabilities: FloatArray,
    probability_threshold: float,
    persistence_windows: int,
) -> NDArray[np.int64]:
    """Return the first confirmation window for each parameter, or -1."""

    n_windows, n_parameters = probabilities.shape
    indices = np.full(n_parameters, -1, dtype=np.int64)
    for parameter_index in range(n_parameters):
        run_length = 0
        for window_index in range(n_windows):
            if probabilities[window_index, parameter_index] >= probability_threshold:
                run_length += 1
            else:
                run_length = 0
            if run_length >= persistence_windows:
                indices[parameter_index] = window_index
                break
    return indices


def calibrate_alarm_thresholds(
    calibration_probabilities: dict[str, list[FloatArray]],
    candidate_thresholds: Sequence[float],
    persistence_windows: int,
    maximum_false_alarm_fraction: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Choose the least stringent healthy-calibrated threshold for each method."""

    selected: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for method_name, probability_histories in calibration_probabilities.items():
        chosen_threshold: float | None = None
        for threshold in candidate_thresholds:
            alarms = [
                first_persistent_alarm_indices(
                    probabilities,
                    float(threshold),
                    persistence_windows,
                )
                for probabilities in probability_histories
            ]
            false_alarm_fraction = float(np.mean(np.stack(alarms) >= 0))
            passed = false_alarm_fraction <= maximum_false_alarm_fraction
            rows.append(
                {
                    "method": method_name,
                    "candidate_probability_threshold": threshold,
                    "healthy_false_alarm_fraction": false_alarm_fraction,
                    "passes_calibration": passed,
                }
            )
            if chosen_threshold is None and passed:
                chosen_threshold = float(threshold)
        if chosen_threshold is None:
            raise RuntimeError(
                f"No candidate alarm threshold calibrated for {method_name}."
            )
        selected[method_name] = chosen_threshold
    calibration = pd.DataFrame(rows)
    calibration["selected"] = calibration.apply(
        lambda row: bool(
            np.isclose(
                row["candidate_probability_threshold"],
                selected[str(row["method"])],
            )
        ),
        axis=1,
    )
    return selected, calibration


def create_tracking_figure(
    runs: pd.DataFrame,
    detections: pd.DataFrame,
    alarm_thresholds: dict[str, float],
    time_days: FloatArray,
    credible_level: float,
    recommended_method: str,
) -> None:
    """Plot the representative gradual, abrupt, and multi-zone tracking runs."""

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), constrained_layout=True)
    gradual_axis, abrupt_axis, multi_axis, probability_axis = axes.ravel()
    colors = {"lower": "#0072B2", "middle": "#D55E00", "upper": "#009E73"}
    method_colors = {
        "conservative": "#0072B2",
        "balanced": "#D55E00",
        "responsive": "#009E73",
    }

    gradual = runs[
        (runs["scenario"] == "gradual_lower")
        & (runs["replicate"] == 0)
        & (runs["zone"] == "lower")
    ]
    recommended_gradual = gradual[gradual["method"] == recommended_method]
    independent_gradual = gradual[gradual["method"] == "independent"]
    gradual_axis.plot(
        time_days,
        recommended_gradual["true_parameter"],
        color="black",
        linewidth=2.0,
        label="Truth",
    )
    gradual_axis.plot(
        time_days,
        independent_gradual["posterior_mean"],
        color="0.60",
        marker=".",
        linewidth=0.9,
        label="Independent windows",
    )
    gradual_axis.plot(
        time_days,
        recommended_gradual["posterior_mean"],
        color=method_colors[recommended_method],
        linewidth=1.8,
        label=f"Sequential {recommended_method}",
    )
    gradual_axis.fill_between(
        time_days,
        recommended_gradual["credible_lower"],
        recommended_gradual["credible_upper"],
        color=method_colors[recommended_method],
        alpha=0.18,
        label=f"{100 * credible_level:.0f}% interval",
    )
    gradual_axis.axhline(0.95, color="black", linestyle=":", linewidth=1.0)
    gradual_axis.set_title("Gradual lower-zone deterioration")
    gradual_axis.set_ylabel("Stiffness multiplier")
    gradual_axis.legend(fontsize=8)
    gradual_axis.grid(alpha=0.25)

    abrupt = runs[
        (runs["scenario"] == "abrupt_middle")
        & (runs["replicate"] == 0)
        & (runs["zone"] == "middle")
    ]
    truth_abrupt = abrupt[abrupt["method"] == "balanced"]
    abrupt_axis.plot(
        time_days,
        truth_abrupt["true_parameter"],
        color="black",
        linewidth=2.2,
        label="Truth",
    )
    for method_name, color in method_colors.items():
        method_data = abrupt[abrupt["method"] == method_name]
        abrupt_axis.plot(
            time_days,
            method_data["posterior_mean"],
            color=color,
            linewidth=1.5,
            label=method_name,
        )
        detection = detections[
            (detections["scenario"] == "abrupt_middle")
            & (detections["replicate"] == 0)
            & (detections["zone"] == "middle")
            & (detections["method"] == method_name)
        ]
        if not detection.empty and int(detection.iloc[0]["alarm_window"]) >= 0:
            alarm_window = int(detection.iloc[0]["alarm_window"])
            abrupt_axis.scatter(
                time_days[alarm_window],
                method_data.iloc[alarm_window]["posterior_mean"],
                color=color,
                edgecolor="black",
                zorder=4,
                s=38,
            )
    abrupt_axis.set_title("Abrupt damage: process-noise trade-off")
    abrupt_axis.legend(fontsize=8)
    abrupt_axis.grid(alpha=0.25)

    multi = runs[
        (runs["scenario"] == "multi_zone")
        & (runs["replicate"] == 0)
        & (runs["method"] == recommended_method)
    ]
    for zone, color in colors.items():
        zone_data = multi[multi["zone"] == zone]
        multi_axis.plot(
            time_days,
            zone_data["true_parameter"],
            color=color,
            linewidth=2.0,
            label=f"{zone} truth",
        )
        multi_axis.plot(
            time_days,
            zone_data["posterior_mean"],
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=f"{zone} estimate",
        )
    multi_axis.axhline(0.95, color="black", linestyle=":", linewidth=1.0)
    multi_axis.set_title(
        f"Two-zone tracking with the {recommended_method} model"
    )
    multi_axis.set_xlabel("Monitoring time (days)")
    multi_axis.set_ylabel("Stiffness multiplier")
    multi_axis.legend(ncols=2, fontsize=8)
    multi_axis.grid(alpha=0.25)

    recommended_threshold = alarm_thresholds[recommended_method]
    for zone, color in colors.items():
        zone_data = multi[multi["zone"] == zone]
        probability_axis.plot(
            time_days,
            zone_data["damage_probability"],
            color=color,
            linewidth=1.8,
            label=zone,
        )
    probability_axis.axhline(
        recommended_threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Calibrated alarm = {recommended_threshold:.3g}",
    )
    probability_axis.set_ylim(-0.02, 1.02)
    probability_axis.set_title("Probability loss exceeds 5%: multi-zone case")
    probability_axis.set_xlabel("Monitoring time (days)")
    probability_axis.set_ylabel("Posterior probability")
    probability_axis.legend(fontsize=8)
    probability_axis.grid(alpha=0.25)

    for axis in (gradual_axis, abrupt_axis):
        axis.set_xlabel("Monitoring time (days)")
    figure.suptitle(
        "Stage 6: sequential Bayesian deterioration tracking",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Execute calibration, sequential recovery, detection, and verification."""

    with BASE_CONFIG_PATH.open(encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    with MONITORING_CONFIG_PATH.open(encoding="utf-8") as stream:
        monitoring_config = yaml.safe_load(stream)
    with TRACKING_CONFIG_PATH.open(encoding="utf-8") as stream:
        tracking_config = yaml.safe_load(stream)

    sequence_config = tracking_config["monitoring_sequence"]
    parameter_config = tracking_config["parameter_model"]
    alarm_config = tracking_config["alarm"]
    replication_config = tracking_config["replication"]
    verification = tracking_config["verification"]
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
    if any(
        np.shares_memory(truth_array, twin_array)
        for truth_array, twin_array in (
            (truth_system.mass_matrix_kg, twin_system.mass_matrix_kg),
            (
                truth_system.damping_matrix_n_s_per_m,
                twin_system.damping_matrix_n_s_per_m,
            ),
            (
                truth_system.zone_components_n_per_m,
                twin_system.zone_components_n_per_m,
            ),
        )
    ):
        raise RuntimeError("Truth and twin systems unexpectedly share memory.")

    zero_noise = np.zeros(layout.n_sensors, dtype=np.float64)
    healthy_clean_forward = make_forward_model(
        truth_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        zero_noise,
    )
    healthy_clean_psd = healthy_clean_forward(np.ones(len(zone_names)))
    noise_fraction = float(sequence_config["noise_amplitude_fraction"])
    measurement_noise_psd = noise_fraction**2 * np.max(
        healthy_clean_psd,
        axis=0,
    )
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
    histories = build_truth_histories(
        tracking_config,
        time_days,
        zone_names,
    )
    methods = ["independent", *tracking_config["process_models"]]
    n_averages = int(sequence_config["independent_periodogram_averages"])
    loss_threshold = float(alarm_config["meaningful_loss_fraction"])
    persistence = int(alarm_config["persistence_windows"])

    calibration_probabilities: dict[str, list[FloatArray]] = {
        method: [] for method in methods
    }
    calibration_seeds = np.random.SeedSequence(
        int(replication_config["calibration_seed"])
    ).spawn(int(replication_config["healthy_calibration_replicates"]))
    healthy_truth = histories["healthy"].stiffness_multipliers
    for seed_sequence in calibration_seeds:
        observations = generate_spectral_windows(
            truth_forward,
            healthy_truth,
            n_averages,
            np.random.default_rng(seed_sequence),
        )
        for method_name in methods:
            method_history = run_method(
                method_name,
                twin_forward,
                observations,
                tracking_config,
            )
            calibration_probabilities[method_name].append(
                method_history.loss_exceedance_probability(loss_threshold)
            )

    alarm_thresholds, calibration = calibrate_alarm_thresholds(
        calibration_probabilities,
        [float(value) for value in alarm_config["candidate_probability_thresholds"]],
        persistence,
        float(alarm_config["maximum_calibration_false_alarm_fraction"]),
    )

    n_evaluation_replicates = int(replication_config["evaluation_replicates"])
    evaluation_seeds = np.random.SeedSequence(
        int(replication_config["evaluation_seed"])
    ).spawn(len(histories) * n_evaluation_replicates)
    seed_iterator = iter(evaluation_seeds)
    run_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    for scenario_name, truth_history in histories.items():
        truth_parameters = truth_history.stiffness_multipliers
        for replicate in range(n_evaluation_replicates):
            seed_sequence = next(seed_iterator)
            seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
            observations = generate_spectral_windows(
                truth_forward,
                truth_parameters,
                n_averages,
                np.random.default_rng(seed_sequence),
            )
            for method_name in methods:
                method_history = run_method(
                    method_name,
                    twin_forward,
                    observations,
                    tracking_config,
                )
                probabilities = method_history.loss_exceedance_probability(
                    loss_threshold
                )
                alarm_indices = first_persistent_alarm_indices(
                    probabilities,
                    alarm_thresholds[method_name],
                    persistence,
                )
                posterior_scale = method_history.posterior_standard_deviations
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
                    detection_delay = (
                        alarm_index - crossing_index if detected else np.nan
                    )
                    detection_rows.append(
                        {
                            "scenario": scenario_name,
                            "replicate": replicate,
                            "seed": seed,
                            "method": method_name,
                            "zone": zone_name,
                            "truth_crossing_window": crossing_index,
                            "alarm_window": alarm_index,
                            "truth_crossing_day": (
                                time_days[crossing_index]
                                if crossing_index >= 0
                                else np.nan
                            ),
                            "alarm_day": (
                                time_days[alarm_index]
                                if alarm_index >= 0
                                else np.nan
                            ),
                            "detection_delay_windows": detection_delay,
                            "false_alarm": false_alarm,
                            "detected": detected,
                            "damage_zone": crossing_index >= 0,
                        }
                    )
                    for window_index in range(n_windows):
                        truth_value = float(truth_zone[window_index])
                        posterior_mean = float(
                            method_history.posterior_means[
                                window_index,
                                zone_index,
                            ]
                        )
                        run_rows.append(
                            {
                                "scenario": scenario_name,
                                "replicate": replicate,
                                "seed": seed,
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
                                "credible_lower": (
                                    method_history.credible_interval_lower[
                                        window_index,
                                        zone_index,
                                    ]
                                ),
                                "credible_upper": (
                                    method_history.credible_interval_upper[
                                        window_index,
                                        zone_index,
                                    ]
                                ),
                                "covered": (
                                    method_history.credible_interval_lower[
                                        window_index,
                                        zone_index,
                                    ]
                                    <= truth_value
                                    <= method_history.credible_interval_upper[
                                        window_index,
                                        zone_index,
                                    ]
                                ),
                                "damage_probability": probabilities[
                                    window_index,
                                    zone_index,
                                ],
                                "optimizer_success": (
                                    method_history.optimizer_success[window_index]
                                ),
                            }
                        )

    runs = pd.DataFrame(run_rows)
    detections = pd.DataFrame(detection_rows)
    method_rows: list[dict[str, object]] = []
    for method_name in methods:
        method_runs = runs[runs["method"] == method_name]
        method_detections = detections[detections["method"] == method_name]
        interior = method_runs[method_runs["true_parameter"] < 0.999]
        damage_events = method_detections[method_detections["damage_zone"]]
        nondamage_events = method_detections[~method_detections["damage_zone"]]
        healthy_control_events = nondamage_events[
            nondamage_events["scenario"] == "healthy"
        ]
        gradual_events = damage_events[
            damage_events["scenario"] == "gradual_lower"
        ]
        abrupt_events = damage_events[
            damage_events["scenario"] == "abrupt_middle"
        ]
        method_rows.append(
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
                "healthy_control_false_alarm_fraction": np.mean(
                    healthy_control_events["false_alarm"]
                ),
                "damage_detection_fraction": np.mean(damage_events["detected"]),
                "mean_detection_delay_windows": np.mean(
                    damage_events.loc[
                        damage_events["detected"],
                        "detection_delay_windows",
                    ]
                ),
                "gradual_detection_delay_windows": np.mean(
                    gradual_events.loc[
                        gradual_events["detected"],
                        "detection_delay_windows",
                    ]
                ),
                "abrupt_detection_delay_windows": np.mean(
                    abrupt_events.loc[
                        abrupt_events["detected"],
                        "detection_delay_windows",
                    ]
                ),
                "optimizer_failure_fraction": 1.0
                - np.mean(method_runs["optimizer_success"]),
            }
        )
    method_summary = pd.DataFrame(method_rows)
    indexed_summary = method_summary.set_index("method")
    recommended_name = str(
        tracking_config["model_selection"]["recommended_process_model"]
    )
    recommended = indexed_summary.loc[recommended_name]
    conservative = indexed_summary.loc["conservative"]
    checks = {
        "recommended tracking RMSE": recommended["pooled_rmse"]
        <= float(verification["maximum_recommended_pooled_rmse"]),
        "recommended interior coverage": recommended["interior_coverage"]
        >= float(verification["minimum_recommended_interior_coverage"]),
        "nondamage false alarms controlled": recommended[
            "nondamage_false_alarm_fraction"
        ]
        <= float(verification["maximum_nondamage_false_alarm_fraction"]),
        "gradual detection delay": recommended["gradual_detection_delay_windows"]
        <= float(
            verification["maximum_recommended_gradual_detection_delay_windows"]
        ),
        "abrupt detection delay": recommended["abrupt_detection_delay_windows"]
        <= float(
            verification["maximum_recommended_abrupt_detection_delay_windows"]
        ),
        "damage detection fraction": recommended["damage_detection_fraction"]
        >= float(verification["minimum_recommended_detection_fraction"]),
        "optimizer convergence": float(
            np.max(method_summary["optimizer_failure_fraction"])
        )
        <= float(verification["maximum_optimizer_failure_fraction"]),
        "restrictive process model retained as failure case": (
            conservative["pooled_rmse"] / recommended["pooled_rmse"]
            >= float(verification["minimum_conservative_to_recommended_rmse_ratio"])
        ),
        "restrictive model delays abrupt detection": (
            conservative["abrupt_detection_delay_windows"]
            - recommended["abrupt_detection_delay_windows"]
            >= float(
                verification[
                    "minimum_conservative_abrupt_delay_penalty_windows"
                ]
            )
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RUNS_PATH, index=False, float_format="%.12g")
    method_summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    detections.to_csv(DETECTION_PATH, index=False, float_format="%.12g")
    calibration.to_csv(CALIBRATION_PATH, index=False, float_format="%.12g")
    create_tracking_figure(
        runs,
        detections,
        alarm_thresholds,
        time_days,
        float(parameter_config["credible_level"]),
        recommended_name,
    )

    print("STAGE 6 - SEQUENTIAL BAYESIAN DETERIORATION TRACKING")
    print(
        f"Monitoring windows: {n_windows} at "
        f"{sequence_config['update_interval_days']} days"
    )
    print(f"Sensor layout: {layout.name} at DOFs {[i + 1 for i in layout.dof_indices]}")
    print(f"Noise amplitude fraction: {noise_fraction:.2f}")
    print(
        "Evaluation: "
        f"{len(histories)} scenarios x {n_evaluation_replicates} replicates"
    )
    print(f"Pilot-selected process model: {recommended_name}")
    print("")
    print("Healthy-calibrated probability thresholds")
    for method_name in methods:
        print(f"  {method_name:<12}: {alarm_thresholds[method_name]:.3f}")
    print("")
    print(
        "Method       | RMSE    | Coverage | False alarm | Detection | "
        "Mean delay | Abrupt delay"
    )
    print(
        "-------------+---------+----------+-------------+-----------+"
        "------------+-------------"
    )
    for row in method_summary.itertuples(index=False):
        print(
            f"{row.method:<12} | {row.pooled_rmse:>7.4f} | "
            f"{row.interior_coverage:>8.3f} | "
            f"{row.nondamage_false_alarm_fraction:>11.3f} | "
            f"{row.damage_detection_fraction:>9.3f} | "
            f"{row.mean_detection_delay_windows:>10.2f} | "
            f"{row.abrupt_detection_delay_windows:>11.2f}"
        )
    print("")
    print(f"Runs: {RUNS_PATH}")
    print(f"Method summary: {SUMMARY_PATH}")
    print(f"Detection events: {DETECTION_PATH}")
    print(f"Alarm calibration: {CALIBRATION_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    if failed_checks:
        raise RuntimeError(f"Stage 6 checks failed: {', '.join(failed_checks)}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
