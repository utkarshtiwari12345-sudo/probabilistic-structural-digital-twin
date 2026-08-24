"""Run the Stage 8 controlled model-discrepancy credibility study."""

from __future__ import annotations

import os
from pathlib import Path

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
    make_forward_model,
    run_method,
)

from probdt.diagnostics import (  # noqa: E402
    SpectralModelDiscrepancy,
    coherent_log_spectral_residual_scores,
    discrepant_acceleration_auto_psd,
)
from probdt.inference import (  # noqa: E402
    SequentialHistory,
    corrected_log_periodogram,
    log_periodogram_noise_variance,
    simulate_averaged_periodogram,
)
from probdt.monitoring import SensorLayout  # noqa: E402

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
TRACKING_CONFIG_PATH = PROJECT_ROOT / "config" / "sequential_tracking.yaml"
DISCREPANCY_CONFIG_PATH = PROJECT_ROOT / "config" / "model_discrepancy.yaml"
RUNS_PATH = PROJECT_ROOT / "results" / "metrics" / "stage8_discrepancy_runs.csv"
SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage8_discrepancy_summary.csv"
)
MODEL_CHECK_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage8_model_check.csv"
)
DETECTION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage8_detection_events.csv"
)
CALIBRATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage8_calibration.csv"
)
CHECKS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage8_hypothesis_checks.csv"
)
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage8_model_discrepancy.png"


def parse_discrepancy(specification: dict[str, object]) -> SpectralModelDiscrepancy:
    """Build an explicit discrepancy object from one configured case."""

    sensor_gains = specification.get("sensor_gain_scales")
    return SpectralModelDiscrepancy(
        mass_scale=float(specification.get("mass_scale", 1.0)),
        damping_scale=float(specification.get("damping_scale", 1.0)),
        excitation_standard_deviation_scale=float(
            specification.get("excitation_standard_deviation_scale", 1.0)
        ),
        excitation_correlation_time_scale=float(
            specification.get("excitation_correlation_time_scale", 1.0)
        ),
        sensor_gain_scales=(
            tuple(float(value) for value in sensor_gains)
            if sensor_gains is not None
            else None
        ),
    )


def make_discrepant_truth_forward(
    truth_system,
    frequencies_hz: np.ndarray,
    force_distribution: np.ndarray,
    layout: SensorLayout,
    excitation_config: dict[str, object],
    measurement_noise_psd: np.ndarray,
    discrepancy: SpectralModelDiscrepancy,
):
    """Create a truth-only spectrum mapping with declared omitted physics."""

    def forward(parameters: np.ndarray) -> np.ndarray:
        return discrepant_acceleration_auto_psd(
            truth_system.mass_matrix_kg,
            truth_system.damping_matrix_n_s_per_m,
            truth_system.zone_components_n_per_m,
            parameters,
            frequencies_hz,
            force_distribution,
            layout.dof_indices,
            force_standard_deviation_n=float(
                excitation_config["stationary_standard_deviation_n"]
            ),
            force_correlation_time_s=float(
                excitation_config["correlation_time_s"]
            ),
            discrepancy=discrepancy,
            measurement_noise_psd=measurement_noise_psd,
        )

    return forward


def generate_windows(
    truth_forward,
    truth_parameters: np.ndarray,
    n_averages: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate averaged periodograms for one slow deterioration history."""

    return np.stack(
        [
            simulate_averaged_periodogram(
                truth_forward(parameters),
                n_averages,
                rng,
            )
            for parameters in truth_parameters
        ]
    )


def log_residual_diagnostics(
    raw_observations: np.ndarray,
    posterior_history: SequentialHistory,
    twin_forward,
    n_averages: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return instantaneous and persistent coherent log-residual scores."""

    n_windows = raw_observations.shape[0]
    corrected = corrected_log_periodogram(raw_observations, n_averages).reshape(
        n_windows,
        -1,
    )
    fitted = np.stack(
        [
            np.log(twin_forward(parameters)).reshape(-1)
            for parameters in posterior_history.posterior_means
        ]
    )
    variance = log_periodogram_noise_variance(n_averages)
    instantaneous = np.mean((corrected - fitted) ** 2 / variance, axis=1)
    coherent = coherent_log_spectral_residual_scores(
        corrected,
        fitted,
        variance,
    )
    return instantaneous, coherent


def create_discrepancy_figure(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    model_check: pd.DataFrame,
    time_days: np.ndarray,
) -> None:
    """Plot false damage, lack-of-fit, mass confounding, and damaged accuracy."""

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), constrained_layout=True)
    false_axis, residual_axis, mass_axis, damage_axis = axes.ravel()
    healthy_cases = summary[summary["truth_scenario"] == "healthy"].copy()
    healthy_cases["short_name"] = (
        healthy_cases["case"].str.replace("_healthy", "").str.replace("_", " ")
    )
    positions = np.arange(len(healthy_cases))
    false_axis.bar(
        positions - 0.18,
        healthy_cases["nondamage_false_alarm_fraction"],
        width=0.36,
        color="#D55E00",
        label="Persistent false-alarm fraction",
    )
    false_axis.bar(
        positions + 0.18,
        healthy_cases["late_apparent_loss"],
        width=0.36,
        color="#0072B2",
        label="Late apparent loss",
    )
    false_axis.set_xticks(
        positions,
        labels=healthy_cases["short_name"],
        rotation=25,
        ha="right",
    )
    false_axis.set_ylim(0.0, 1.05)
    false_axis.set_title("Healthy truth misinterpreted as deterioration")
    false_axis.legend(fontsize=8)
    false_axis.grid(axis="y", alpha=0.25)

    residual_axis.bar(
        positions,
        healthy_cases["residual_exceedance_fraction"],
        color="#009E73",
    )
    residual_axis.set_xticks(
        positions,
        labels=healthy_cases["short_name"],
        rotation=25,
        ha="right",
    )
    residual_axis.set_ylim(0.0, 1.05)
    residual_axis.set_ylabel("Eligible windows above calibrated limit")
    residual_axis.set_title("Can coherent spectral residuals expose mismatch?")
    residual_axis.grid(axis="y", alpha=0.25)

    mass = runs[
        (runs["case"] == "mass_plus_5")
        & (runs["replicate"] == 0)
    ]
    colors = {"lower": "#0072B2", "middle": "#D55E00", "upper": "#009E73"}
    for zone, color in colors.items():
        zone_data = mass[mass["zone"] == zone]
        mass_axis.plot(
            time_days,
            zone_data["posterior_mean"],
            color=color,
            linewidth=1.7,
            label=f"Inferred {zone}",
        )
    mass_axis.axhline(1.0, color="black", linewidth=2.0, label="Healthy truth")
    mass_axis.axhline(0.95, color="black", linestyle=":", linewidth=1.0)
    mass_axis.set_xlabel("Monitoring time (days)")
    mass_axis.set_ylabel("Stiffness multiplier")
    mass_axis.set_title("Dangerous mass--stiffness confounding: +5% truth mass")
    mass_axis.legend(fontsize=8)
    mass_axis.grid(alpha=0.25)

    damaged = summary[summary["truth_scenario"] != "healthy"]
    damage_positions = np.arange(len(damaged))
    damage_axis.bar(
        damage_positions - 0.18,
        damaged["pooled_rmse"],
        width=0.36,
        color="#CC79A7",
        label="Trajectory RMSE",
    )
    scaled_detection = 0.05 * damaged["damage_detection_fraction"]
    damage_axis.bar(
        damage_positions + 0.18,
        scaled_detection,
        width=0.36,
        color="0.7",
        edgecolor="black",
        label="Detection fraction x 0.05",
    )
    for position, detection in zip(
        damage_positions,
        damaged["damage_detection_fraction"],
        strict=True,
    ):
        damage_axis.text(
            position + 0.18,
            0.05 * detection + 0.001,
            f"{detection:.0%}",
            ha="center",
            fontsize=8,
        )
    damage_axis.set_xticks(
        damage_positions,
        labels=damaged["case"].str.replace("_", " "),
        rotation=20,
        ha="right",
    )
    damage_axis.set_title("Physical damage under simultaneous discrepancy")
    damage_axis.set_ylabel("RMSE (detection labels above grey bars)")
    damage_axis.legend(fontsize=8)
    damage_axis.grid(axis="y", alpha=0.25)

    residual_threshold = float(
        model_check["coherent_residual_threshold"].iloc[0]
    )
    figure.suptitle(
        "Stage 8: model discrepancy versus inferred deterioration\n"
        f"Healthy-calibrated residual threshold = {residual_threshold:.3f}",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Execute matched calibration and controlled discrepancy experiments."""

    with BASE_CONFIG_PATH.open(encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    with MONITORING_CONFIG_PATH.open(encoding="utf-8") as stream:
        monitoring_config = yaml.safe_load(stream)
    with TRACKING_CONFIG_PATH.open(encoding="utf-8") as stream:
        tracking_config = yaml.safe_load(stream)
    with DISCREPANCY_CONFIG_PATH.open(encoding="utf-8") as stream:
        discrepancy_config = yaml.safe_load(stream)

    sequence_config = tracking_config["monitoring_sequence"]
    parameter_config = tracking_config["parameter_model"]
    alarm_config = tracking_config["alarm"]
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
    healthy_clean_psd = healthy_clean_forward(np.ones(len(zone_names)))
    noise_fraction = float(sequence_config["noise_amplitude_fraction"])
    measurement_noise_psd = noise_fraction**2 * np.max(
        healthy_clean_psd,
        axis=0,
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
    n_averages = int(sequence_config["independent_periodogram_averages"])
    loss_threshold = float(alarm_config["meaningful_loss_fraction"])
    persistence = int(alarm_config["persistence_windows"])
    inference_process_name = str(
        discrepancy_config["data_model"]["inference_process_model"]
    )

    identity_truth_forward = make_discrepant_truth_forward(
        truth_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        measurement_noise_psd,
        SpectralModelDiscrepancy(),
    )
    calibration_config = discrepancy_config["calibration"]
    calibration_seeds = np.random.SeedSequence(
        int(calibration_config["random_seed"])
    ).spawn(int(calibration_config["healthy_replicates"]))
    calibration_probabilities: list[np.ndarray] = []
    calibration_residual_scores: list[np.ndarray] = []
    healthy_truth = histories["healthy"].stiffness_multipliers
    for seed_sequence in calibration_seeds:
        raw = generate_windows(
            identity_truth_forward,
            healthy_truth,
            n_averages,
            np.random.default_rng(seed_sequence),
        )
        history = run_method(
            inference_process_name,
            twin_forward,
            raw,
            tracking_config,
        )
        calibration_probabilities.append(
            history.loss_exceedance_probability(loss_threshold)
        )
        _, coherent_scores = log_residual_diagnostics(
            raw,
            history,
            twin_forward,
            n_averages,
        )
        calibration_residual_scores.append(coherent_scores)
    selected_thresholds, probability_calibration = calibrate_alarm_thresholds(
        {"naive_laplace": calibration_probabilities},
        [float(value) for value in alarm_config["candidate_probability_thresholds"]],
        persistence,
        float(alarm_config["maximum_calibration_false_alarm_fraction"]),
    )
    alarm_probability_threshold = selected_thresholds["naive_laplace"]
    minimum_accumulation_windows = int(
        calibration_config["minimum_accumulation_windows"]
    )
    if not 1 <= minimum_accumulation_windows <= n_windows:
        raise ValueError(
            "minimum_accumulation_windows must be between one and the sequence "
            "length."
        )
    eligible_slice = slice(minimum_accumulation_windows - 1, None)
    calibration_sequence_maxima = np.asarray(
        [np.max(scores[eligible_slice]) for scores in calibration_residual_scores]
    )
    sequence_false_alarm_probability = float(
        calibration_config[
            "coherent_residual_sequence_false_alarm_probability"
        ]
    )
    if not 0.0 < sequence_false_alarm_probability < 1.0:
        raise ValueError(
            "coherent_residual_sequence_false_alarm_probability must lie "
            "strictly between zero and one."
        )
    residual_threshold = float(
        np.quantile(
            calibration_sequence_maxima,
            1.0 - sequence_false_alarm_probability,
            method="higher",
        )
    )

    cases = discrepancy_config["cases"]
    evaluation_config = discrepancy_config["evaluation"]
    n_replicates = int(evaluation_config["replicates"])
    evaluation_seeds = np.random.SeedSequence(
        int(evaluation_config["random_seed"])
    ).spawn(len(cases) * n_replicates)
    seed_iterator = iter(evaluation_seeds)
    run_rows: list[dict[str, object]] = []
    model_check_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    for case_name, specification in cases.items():
        truth_scenario = str(specification["truth_scenario"])
        truth_parameters = histories[truth_scenario].stiffness_multipliers
        discrepancy = parse_discrepancy(specification)
        truth_forward = make_discrepant_truth_forward(
            truth_system,
            frequencies_hz,
            force_distribution,
            layout,
            excitation_config,
            measurement_noise_psd,
            discrepancy,
        )
        for replicate in range(n_replicates):
            seed_sequence = next(seed_iterator)
            seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
            raw = generate_windows(
                truth_forward,
                truth_parameters,
                n_averages,
                np.random.default_rng(seed_sequence),
            )
            history = run_method(
                inference_process_name,
                twin_forward,
                raw,
                tracking_config,
            )
            probabilities = history.loss_exceedance_probability(loss_threshold)
            alarm_indices = first_persistent_alarm_indices(
                probabilities,
                alarm_probability_threshold,
                persistence,
            )
            instantaneous_scores, coherent_scores = log_residual_diagnostics(
                raw,
                history,
                twin_forward,
                n_averages,
            )
            for window_index, (instantaneous_score, coherent_score) in enumerate(
                zip(instantaneous_scores, coherent_scores, strict=True)
            ):
                residual_eligible = (
                    window_index + 1 >= minimum_accumulation_windows
                )
                model_check_rows.append(
                    {
                        "case": case_name,
                        "truth_scenario": truth_scenario,
                        "replicate": replicate,
                        "seed": seed,
                        "window": window_index,
                        "time_day": time_days[window_index],
                        "instantaneous_log_residual": instantaneous_score,
                        "coherent_log_residual": coherent_score,
                        "coherent_residual_threshold": residual_threshold,
                        "residual_eligible": residual_eligible,
                        "coherent_residual_exceeds_threshold": (
                            residual_eligible
                            and coherent_score > residual_threshold
                        ),
                    }
                )
            posterior_scale = history.posterior_standard_deviations
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
                        "case": case_name,
                        "truth_scenario": truth_scenario,
                        "replicate": replicate,
                        "zone": zone_name,
                        "truth_crossing_window": crossing_index,
                        "alarm_window": alarm_index,
                        "detection_delay_windows": (
                            alarm_index - crossing_index if detected else np.nan
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
                            "case": case_name,
                            "truth_scenario": truth_scenario,
                            "replicate": replicate,
                            "seed": seed,
                            "window": window_index,
                            "time_day": time_days[window_index],
                            "zone": zone_name,
                            "true_parameter": truth_value,
                            "posterior_mean": posterior_mean,
                            "error": posterior_mean - truth_value,
                            "posterior_standard_deviation": posterior_scale[
                                window_index,
                                zone_index,
                            ],
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
                            "optimizer_success": history.optimizer_success[
                                window_index
                            ],
                        }
                    )

    runs = pd.DataFrame(run_rows)
    model_check = pd.DataFrame(model_check_rows)
    detections = pd.DataFrame(detection_rows)
    summary_rows: list[dict[str, object]] = []
    for case_name, specification in cases.items():
        truth_scenario = str(specification["truth_scenario"])
        case_runs = runs[runs["case"] == case_name]
        case_check = model_check[model_check["case"] == case_name]
        case_events = detections[detections["case"] == case_name]
        interior = case_runs[case_runs["true_parameter"] < 0.999]
        damage_events = case_events[case_events["damage_zone"]]
        nondamage_events = case_events[~case_events["damage_zone"]]
        late = case_runs[case_runs["window"] >= n_windows - 5]
        late_apparent_loss = float(
            (1.0 - late["posterior_mean"]).groupby(
                [late["replicate"], late["window"]]
            ).max().mean()
        )
        summary_rows.append(
            {
                "case": case_name,
                "truth_scenario": truth_scenario,
                "pooled_rmse": np.sqrt(np.mean(case_runs["error"] ** 2)),
                "pooled_bias": np.mean(case_runs["error"]),
                "interior_coverage": (
                    np.mean(interior["covered"]) if not interior.empty else np.nan
                ),
                "nondamage_false_alarm_fraction": np.mean(
                    nondamage_events["false_alarm"]
                ),
                "damage_detection_fraction": (
                    np.mean(damage_events["detected"])
                    if not damage_events.empty
                    else np.nan
                ),
                "mean_detection_delay_windows": (
                    np.mean(
                        damage_events.loc[
                            damage_events["detected"],
                            "detection_delay_windows",
                        ]
                    )
                    if not damage_events.empty
                    else np.nan
                ),
                "maximum_apparent_loss": np.max(
                    1.0 - case_runs["posterior_mean"]
                ),
                "late_apparent_loss": late_apparent_loss,
                "mean_instantaneous_log_residual": np.mean(
                    case_check["instantaneous_log_residual"]
                ),
                "mean_coherent_log_residual": np.mean(
                    case_check.loc[
                        case_check["residual_eligible"],
                        "coherent_log_residual",
                    ]
                ),
                "residual_exceedance_fraction": np.mean(
                    case_check.loc[
                        case_check["residual_eligible"],
                        "coherent_residual_exceeds_threshold",
                    ]
                ),
                "optimizer_failure_fraction": 1.0
                - np.mean(case_runs["optimizer_success"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    indexed = summary.set_index("case")
    healthy_discrepancy = summary[
        (summary["truth_scenario"] == "healthy")
        & (summary["case"] != "matched_healthy")
    ]
    damaged = summary[summary["truth_scenario"] != "healthy"]
    verification = discrepancy_config["verification"]
    hypotheses = discrepancy_config["hypotheses"]
    check_rows = [
        {
            "category": "verification",
            "check": "matched healthy false alarms controlled",
            "observed": indexed.loc[
                "matched_healthy", "nondamage_false_alarm_fraction"
            ],
            "comparator": "<=",
            "threshold": float(
                verification["maximum_matched_false_alarm_fraction"]
            ),
        },
        {
            "category": "verification",
            "check": "matched residual calibration retained",
            "observed": indexed.loc[
                "matched_healthy", "residual_exceedance_fraction"
            ],
            "comparator": "<=",
            "threshold": float(
                verification["maximum_matched_residual_exceedance_fraction"]
            ),
        },
        {
            "category": "verification",
            "check": "optimizer convergence",
            "observed": float(
                np.max(summary["optimizer_failure_fraction"])
            ),
            "comparator": "<=",
            "threshold": float(
                verification["maximum_optimizer_failure_fraction"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "healthy discrepancy creates false damage",
            "observed": float(
                np.max(
                    healthy_discrepancy["nondamage_false_alarm_fraction"]
                )
            ),
            "comparator": ">=",
            "threshold": float(
                hypotheses[
                    "minimum_any_healthy_discrepancy_false_alarm_fraction"
                ]
            ),
        },
        {
            "category": "hypothesis",
            "check": "mass discrepancy creates apparent loss",
            "observed": indexed.loc["mass_plus_5", "late_apparent_loss"],
            "comparator": ">=",
            "threshold": float(
                hypotheses["minimum_mass_apparent_loss_fraction"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "residual diagnostic detects at least one discrepancy",
            "observed": float(
                np.max(healthy_discrepancy["residual_exceedance_fraction"])
            ),
            "comparator": ">=",
            "threshold": float(
                hypotheses[
                    "minimum_any_discrepancy_residual_exceedance_fraction"
                ]
            ),
        },
        {
            "category": "hypothesis",
            "check": "damage detection survives every discrepancy",
            "observed": float(np.min(damaged["damage_detection_fraction"])),
            "comparator": ">=",
            "threshold": float(
                hypotheses["minimum_damage_case_detection_fraction"]
            ),
        },
    ]
    checks = pd.DataFrame(check_rows)
    checks["supported"] = np.where(
        checks["comparator"] == "<=",
        checks["observed"] <= checks["threshold"],
        checks["observed"] >= checks["threshold"],
    )
    failed_verification = checks[
        (checks["category"] == "verification") & ~checks["supported"]
    ]

    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RUNS_PATH, index=False, float_format="%.12g")
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    model_check.to_csv(MODEL_CHECK_PATH, index=False, float_format="%.12g")
    detections.to_csv(DETECTION_PATH, index=False, float_format="%.12g")
    calibration_output = probability_calibration.copy()
    calibration_output[
        "coherent_residual_sequence_false_alarm_probability"
    ] = sequence_false_alarm_probability
    calibration_output["maximum_calibration_sequence_residual"] = float(
        np.max(calibration_sequence_maxima)
    )
    calibration_output["minimum_accumulation_windows"] = (
        minimum_accumulation_windows
    )
    calibration_output["coherent_residual_threshold"] = residual_threshold
    calibration_output.to_csv(CALIBRATION_PATH, index=False, float_format="%.12g")
    checks.to_csv(CHECKS_PATH, index=False, float_format="%.12g")
    create_discrepancy_figure(runs, summary, model_check, time_days)

    print("STAGE 8 - CONTROLLED MODEL-DISCREPANCY STRESS TEST")
    print(f"Alarm probability threshold: {alarm_probability_threshold:.3f}")
    print(f"Coherent residual threshold: {residual_threshold:.4f}")
    print("")
    print(
        "Case                           | RMSE   | False alarm | Detection | "
        "Late loss | Residual flag"
    )
    print(
        "-------------------------------+--------+-------------+-----------+"
        "-----------+--------------"
    )
    for row in summary.itertuples(index=False):
        detection = (
            "   n/a" if np.isnan(row.damage_detection_fraction)
            else f"{row.damage_detection_fraction:>7.3f}"
        )
        print(
            f"{row.case:<30} | {row.pooled_rmse:>6.3f} | "
            f"{row.nondamage_false_alarm_fraction:>11.3f} | "
            f"{detection:>9} | {row.late_apparent_loss:>9.3f} | "
            f"{row.residual_exceedance_fraction:>12.3f}"
        )
    print("")
    print(f"Runs: {RUNS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Model check: {MODEL_CHECK_PATH}")
    print(f"Detection events: {DETECTION_PATH}")
    print(f"Calibration: {CALIBRATION_PATH}")
    print(f"Hypothesis checks: {CHECKS_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    hypothesis_checks = checks[checks["category"] == "hypothesis"]
    supported_count = int(hypothesis_checks["supported"].sum())
    print(
        "Predeclared hypotheses supported: "
        f"{supported_count}/{len(hypothesis_checks)}"
    )
    if not failed_verification.empty:
        failed_names = ", ".join(failed_verification["check"])
        raise RuntimeError(f"Stage 8 verification failed: {failed_names}")
    print("RESULT: PASS (experiment complete; rejected hypotheses retained)")


if __name__ == "__main__":
    main()
