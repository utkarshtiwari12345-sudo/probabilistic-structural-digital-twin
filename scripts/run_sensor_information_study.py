"""Run the Stage 10 paired sensor-information and placement study."""

from __future__ import annotations

import os
import time
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
from run_joint_discrepancy_inference_study import (  # noqa: E402
    build_joint_method_definition,
    make_joint_forward_model,
    nuisance_truth_vector,
    run_configured_method,
    run_identifiability_analysis,
    stiffness_loss_probabilities,
)
from run_model_discrepancy_study import (  # noqa: E402
    generate_windows,
    make_discrepant_truth_forward,
    parse_discrepancy,
)
from run_sequential_tracking_study import (  # noqa: E402
    build_spectral_system,
    build_truth_histories,
    calibrate_alarm_thresholds,
    first_persistent_alarm_indices,
    make_forward_model,
)

from probdt.inference import JOINT_NUISANCE_PARAMETER_NAMES  # noqa: E402
from probdt.monitoring import (  # noqa: E402
    SensorLayout,
    sensor_subset_columns,
    subset_sensor_spectra,
    top_k_localization_success,
)

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
TRACKING_CONFIG_PATH = PROJECT_ROOT / "config" / "sequential_tracking.yaml"
JOINT_CONFIG_PATH = PROJECT_ROOT / "config" / "joint_discrepancy_inference.yaml"
SENSOR_CONFIG_PATH = PROJECT_ROOT / "config" / "sensor_information.yaml"
RUNS_PATH = PROJECT_ROOT / "results" / "metrics" / "stage10_sensor_runs.csv"
CASE_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage10_case_summary.csv"
)
LAYOUT_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage10_layout_summary.csv"
)
DETECTION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage10_detection_events.csv"
)
LOCALIZATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage10_localization_events.csv"
)
NUISANCE_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage10_nuisance_final.csv"
)
CALIBRATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage10_alarm_calibration.csv"
)
IDENTIFIABILITY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage10_identifiability.csv"
)
CHECKS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage10_hypothesis_checks.csv"
)
RUNTIME_PATH = PROJECT_ROOT / "results" / "metrics" / "stage10_runtime.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage10_sensor_information.png"


def load_layouts(
    monitoring_config: dict[str, object],
    layout_names: list[str],
) -> dict[str, SensorLayout]:
    """Build configured sensor layouts with zero-based DOF indices."""

    configured = monitoring_config["sensor_layouts"]
    layouts = {
        name: SensorLayout(
            name,
            tuple(
                int(index) - 1
                for index in configured[name]["dofs_one_based"]
            ),
        )
        for name in layout_names
    }
    if len(layouts) != len(layout_names):
        raise ValueError("Sensor layout names must be unique.")
    return layouts


def summarize_cases(
    runs: pd.DataFrame,
    detections: pd.DataFrame,
    localization: pd.DataFrame,
    layout_names: list[str],
    cases: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Summarize estimation and decisions for every layout--case pair."""

    rows: list[dict[str, object]] = []
    for layout_name in layout_names:
        for case_name, specification in cases.items():
            selected = runs[
                (runs["layout"] == layout_name) & (runs["case"] == case_name)
            ]
            events = detections[
                (detections["layout"] == layout_name)
                & (detections["case"] == case_name)
            ]
            damage_events = events[events["damage_zone"]]
            nondamage_events = events[~events["damage_zone"]]
            interior = selected[selected["true_parameter"] < 0.999]
            selected_localization = localization[
                (localization["layout"] == layout_name)
                & (localization["case"] == case_name)
            ]
            rows.append(
                {
                    "layout": layout_name,
                    "case": case_name,
                    "truth_scenario": specification["truth_scenario"],
                    "pooled_rmse": np.sqrt(np.mean(selected["error"] ** 2)),
                    "pooled_bias": np.mean(selected["error"]),
                    "interior_coverage": (
                        np.mean(interior["covered"])
                        if not interior.empty
                        else np.nan
                    ),
                    "mean_credible_interval_width": np.mean(
                        selected["credible_upper"]
                        - selected["credible_lower"]
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
                        if np.any(damage_events["detected"])
                        else np.nan
                    ),
                    "localization_accuracy": (
                        np.mean(selected_localization["localized"])
                        if not selected_localization.empty
                        else np.nan
                    ),
                    "mean_probability_contrast": (
                        np.mean(
                            selected_localization["damage_probability_contrast"]
                        )
                        if not selected_localization.empty
                        else np.nan
                    ),
                    "optimizer_failure_fraction": 1.0
                    - np.mean(selected["optimizer_success"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_layouts(
    runs: pd.DataFrame,
    detections: pd.DataFrame,
    localization: pd.DataFrame,
    case_summary: pd.DataFrame,
    identifiability: pd.DataFrame,
    layouts: dict[str, SensorLayout],
) -> pd.DataFrame:
    """Aggregate damaged-case performance and layout information."""

    rows: list[dict[str, object]] = []
    for layout_name, layout in layouts.items():
        selected = runs[
            (runs["layout"] == layout_name)
            & (runs["truth_scenario"] != "healthy")
        ]
        interior = selected[selected["true_parameter"] < 0.999]
        events = detections[
            (detections["layout"] == layout_name)
            & detections["damage_zone"]
        ]
        selected_localization = localization[
            localization["layout"] == layout_name
        ]
        information = identifiability.set_index("layout").loc[layout_name]
        case_index = case_summary.set_index(["layout", "case"])
        rows.append(
            {
                "layout": layout_name,
                "n_sensors": layout.n_sensors,
                "damaged_pooled_rmse": np.sqrt(np.mean(selected["error"] ** 2)),
                "damaged_interior_coverage": np.mean(interior["covered"]),
                "damaged_mean_credible_interval_width": np.mean(
                    selected["credible_upper"] - selected["credible_lower"]
                ),
                "damage_detection_fraction": np.mean(events["detected"]),
                "mean_detection_delay_windows": np.mean(
                    events.loc[events["detected"], "detection_delay_windows"]
                ),
                "localization_accuracy": np.mean(
                    selected_localization["localized"]
                ),
                "mean_probability_contrast": np.mean(
                    selected_localization["damage_probability_contrast"]
                ),
                "matched_false_alarm_fraction": case_index.loc[
                    (layout_name, "matched_healthy"),
                    "nondamage_false_alarm_fraction",
                ],
                "mass_discrepancy_false_alarm_fraction": case_index.loc[
                    (layout_name, "mass_plus_5"),
                    "nondamage_false_alarm_fraction",
                ],
                "data_only_effective_rank": information[
                    "data_only_effective_rank"
                ],
                "data_only_rank_deficiency": information[
                    "data_only_rank_deficiency"
                ],
                "data_only_normalized_condition_number": information[
                    "data_only_normalized_condition_number"
                ],
                "data_only_maximum_theta_mass_correlation": information[
                    "data_only_maximum_theta_mass_correlation"
                ],
            }
        )
    return pd.DataFrame(rows)


def create_stage10_figure(layout_summary: pd.DataFrame) -> None:
    """Plot information, uncertainty, detection, and false-alarm comparisons."""

    order = layout_summary["layout"].tolist()
    labels = [name.replace("_", " ") for name in order]
    positions = np.arange(len(order), dtype=np.float64)
    colors = ["#999999", "#56B4E9", "#0072B2", "#E69F00", "#009E73"]
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), constrained_layout=True)
    information_axis, accuracy_axis, decision_axis, alarm_axis = axes.ravel()

    condition = layout_summary["data_only_normalized_condition_number"]
    information_axis.bar(positions, condition, color=colors)
    information_axis.set_yscale("log")
    information_axis.set_xticks(positions, labels=labels, rotation=20, ha="right")
    information_axis.set_ylabel("Normalized condition number (log scale)")
    information_axis.set_title("Joint spectral identifiability")
    for position, rank in zip(
        positions,
        layout_summary["data_only_effective_rank"],
        strict=True,
    ):
        information_axis.text(
            position,
            condition.iloc[int(position)] * 1.08,
            f"rank {int(rank)}/7",
            ha="center",
            fontsize=8,
        )
    information_axis.grid(axis="y", alpha=0.25, which="both")

    width = 0.36
    accuracy_axis.bar(
        positions - width / 2.0,
        layout_summary["damaged_pooled_rmse"],
        width=width,
        color="#CC79A7",
        label="Trajectory RMSE",
    )
    accuracy_axis.bar(
        positions + width / 2.0,
        layout_summary["damaged_mean_credible_interval_width"],
        width=width,
        color="#0072B2",
        label="Mean 95% interval width",
    )
    accuracy_axis.set_xticks(positions, labels=labels, rotation=20, ha="right")
    accuracy_axis.set_title("Damaged-case accuracy and uncertainty")
    accuracy_axis.legend(fontsize=8)
    accuracy_axis.grid(axis="y", alpha=0.25)

    decision_axis.bar(
        positions - width / 2.0,
        layout_summary["damage_detection_fraction"],
        width=width,
        color="#009E73",
        label="Detection fraction",
    )
    decision_axis.bar(
        positions + width / 2.0,
        layout_summary["localization_accuracy"],
        width=width,
        color="#E69F00",
        label="Top-k localisation accuracy",
    )
    decision_axis.set_xticks(positions, labels=labels, rotation=20, ha="right")
    decision_axis.set_ylim(0.0, 1.05)
    decision_axis.set_title("Damage decisions")
    decision_axis.legend(fontsize=8)
    decision_axis.grid(axis="y", alpha=0.25)

    alarm_axis.bar(
        positions - width / 2.0,
        layout_summary["matched_false_alarm_fraction"],
        width=width,
        color="0.65",
        label="Matched healthy",
    )
    alarm_axis.bar(
        positions + width / 2.0,
        layout_summary["mass_discrepancy_false_alarm_fraction"],
        width=width,
        color="#D55E00",
        label="Healthy +5% mass",
    )
    alarm_axis.axhline(0.05, color="black", linestyle="--", linewidth=1.0)
    alarm_axis.set_xticks(positions, labels=labels, rotation=20, ha="right")
    alarm_axis.set_ylim(0.0, 1.05)
    alarm_axis.set_title("Persistent false alarms")
    alarm_axis.legend(fontsize=8)
    alarm_axis.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Stage 10: sensor information, placement, and diminishing returns",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main() -> None:
    """Execute paired layout calibration, identifiability, and holdout tests."""

    with BASE_CONFIG_PATH.open(encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    with MONITORING_CONFIG_PATH.open(encoding="utf-8") as stream:
        monitoring_config = yaml.safe_load(stream)
    with TRACKING_CONFIG_PATH.open(encoding="utf-8") as stream:
        tracking_config = yaml.safe_load(stream)
    with JOINT_CONFIG_PATH.open(encoding="utf-8") as stream:
        joint_config = yaml.safe_load(stream)
    with SENSOR_CONFIG_PATH.open(encoding="utf-8") as stream:
        sensor_config = yaml.safe_load(stream)

    layout_names = [str(name) for name in sensor_config["layouts"]]
    layouts = load_layouts(monitoring_config, layout_names)
    reference_layout_name = str(sensor_config["data_model"]["reference_layout"])
    if reference_layout_name not in layouts:
        raise ValueError("The reference sensor layout must be evaluated.")
    reference_layout = layouts[reference_layout_name]
    subset_columns = {
        name: sensor_subset_columns(reference_layout, layout)
        for name, layout in layouts.items()
    }
    joint_method_name = str(sensor_config["data_model"]["joint_method"])
    method_specification = joint_config["methods"][joint_method_name]
    if method_specification["model"] != "joint":
        raise ValueError("Stage 10 requires a joint Stage 9 method.")
    joint_definition = build_joint_method_definition(
        method_specification,
        tracking_config,
    )
    joint_definitions = {joint_method_name: joint_definition}

    sequence_config = tracking_config["monitoring_sequence"]
    parameter_config = tracking_config["parameter_model"]
    alarm_config = tracking_config["alarm"]
    excitation_config = monitoring_config["excitation"]
    stiffness_names = tuple(str(name) for name in parameter_config["names"])
    zone_names = tuple(name.removeprefix("theta_") for name in stiffness_names)
    n_stiffness = len(stiffness_names)
    n_windows = int(sequence_config["number_of_windows"])
    time_days = np.arange(n_windows) * float(
        sequence_config["update_interval_days"]
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
    truth_system = build_spectral_system(base_config)
    twin_system = build_spectral_system(base_config)
    clean_reference_forward = make_forward_model(
        truth_system,
        frequencies_hz,
        force_distribution,
        reference_layout,
        excitation_config,
        np.zeros(reference_layout.n_sensors),
    )
    clean_reference_psd = clean_reference_forward(np.ones(n_stiffness))
    noise_fraction = float(sequence_config["noise_amplitude_fraction"])
    reference_noise_psd = noise_fraction**2 * np.max(
        clean_reference_psd,
        axis=0,
    )
    joint_forwards = {
        name: make_joint_forward_model(
            twin_system,
            frequencies_hz,
            force_distribution,
            layout,
            excitation_config,
            reference_noise_psd[list(subset_columns[name])],
        )
        for name, layout in layouts.items()
    }
    nominal_forwards = {
        name: make_forward_model(
            twin_system,
            frequencies_hz,
            force_distribution,
            layout,
            excitation_config,
            reference_noise_psd[list(subset_columns[name])],
        )
        for name, layout in layouts.items()
    }
    histories = build_truth_histories(tracking_config, time_days, zone_names)
    n_averages = int(sequence_config["independent_periodogram_averages"])
    loss_threshold = float(alarm_config["meaningful_loss_fraction"])
    persistence = int(alarm_config["persistence_windows"])

    identifiability_rows: list[pd.DataFrame] = []
    for layout_name in layout_names:
        information, _ = run_identifiability_analysis(
            joint_forwards[layout_name],
            joint_definition,
            joint_config["identifiability"],
            n_averages,
        )
        information.insert(0, "layout", layout_name)
        information.insert(1, "n_sensors", layouts[layout_name].n_sensors)
        identifiability_rows.append(information)
    identifiability = pd.concat(identifiability_rows, ignore_index=True)

    identity_truth_forward = make_discrepant_truth_forward(
        truth_system,
        frequencies_hz,
        force_distribution,
        reference_layout,
        excitation_config,
        reference_noise_psd,
        parse_discrepancy({}),
    )
    healthy_truth = histories["healthy"].stiffness_multipliers
    calibration_config = sensor_config["calibration"]
    calibration_seeds = np.random.SeedSequence(
        int(calibration_config["random_seed"])
    ).spawn(int(calibration_config["healthy_replicates"]))
    calibration_probabilities = {name: [] for name in layout_names}
    for seed_sequence in calibration_seeds:
        reference_raw = generate_windows(
            identity_truth_forward,
            healthy_truth,
            n_averages,
            np.random.default_rng(seed_sequence),
        )
        for layout_name in layout_names:
            observations = subset_sensor_spectra(
                reference_raw,
                subset_columns[layout_name],
            )
            history = run_configured_method(
                joint_method_name,
                method_specification,
                observations,
                nominal_forwards[layout_name],
                joint_forwards[layout_name],
                tracking_config,
                joint_definitions,
            )
            calibration_probabilities[layout_name].append(
                stiffness_loss_probabilities(
                    history,
                    loss_threshold,
                    n_stiffness,
                )
            )
    alarm_thresholds, alarm_calibration = calibrate_alarm_thresholds(
        calibration_probabilities,
        [float(value) for value in alarm_config["candidate_probability_thresholds"]],
        persistence,
        float(alarm_config["maximum_calibration_false_alarm_fraction"]),
    )

    all_cases = joint_config["cases"]
    cases = {name: all_cases[name] for name in sensor_config["case_names"]}
    evaluation_config = sensor_config["evaluation"]
    n_replicates = int(evaluation_config["replicates"])
    evaluation_seeds = np.random.SeedSequence(
        int(evaluation_config["random_seed"])
    ).spawn(len(cases) * n_replicates)
    seed_iterator = iter(evaluation_seeds)
    run_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    localization_rows: list[dict[str, object]] = []
    nuisance_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for case_name, specification in cases.items():
        truth_scenario = str(specification["truth_scenario"])
        truth_parameters = histories[truth_scenario].stiffness_multipliers
        truth_forward = make_discrepant_truth_forward(
            truth_system,
            frequencies_hz,
            force_distribution,
            reference_layout,
            excitation_config,
            reference_noise_psd,
            parse_discrepancy(specification),
        )
        true_nuisance = nuisance_truth_vector(specification)
        for replicate in range(n_replicates):
            seed_sequence = next(seed_iterator)
            seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
            reference_raw = generate_windows(
                truth_forward,
                truth_parameters,
                n_averages,
                np.random.default_rng(seed_sequence),
            )
            for layout_name in layout_names:
                observations = subset_sensor_spectra(
                    reference_raw,
                    subset_columns[layout_name],
                )
                start_time = time.perf_counter()
                history = run_configured_method(
                    joint_method_name,
                    method_specification,
                    observations,
                    nominal_forwards[layout_name],
                    joint_forwards[layout_name],
                    tracking_config,
                    joint_definitions,
                )
                runtime_rows.append(
                    {
                        "layout": layout_name,
                        "case": case_name,
                        "truth_scenario": truth_scenario,
                        "replicate": replicate,
                        "runtime_seconds": time.perf_counter() - start_time,
                    }
                )
                probabilities = stiffness_loss_probabilities(
                    history,
                    loss_threshold,
                    n_stiffness,
                )
                alarm_indices = first_persistent_alarm_indices(
                    probabilities,
                    alarm_thresholds[layout_name],
                    persistence,
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
                            "layout": layout_name,
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
                        truth_value = truth_zone[window_index]
                        posterior_mean = history.posterior_means[
                            window_index,
                            zone_index,
                        ]
                        run_rows.append(
                            {
                                "layout": layout_name,
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
                true_damage_mask = (
                    1.0 - truth_parameters[-1] > loss_threshold
                )
                if np.any(true_damage_mask) and not np.all(true_damage_mask):
                    final_probabilities = probabilities[-1]
                    localization_rows.append(
                        {
                            "layout": layout_name,
                            "case": case_name,
                            "truth_scenario": truth_scenario,
                            "replicate": replicate,
                            "localized": top_k_localization_success(
                                final_probabilities,
                                true_damage_mask,
                            ),
                            "damage_probability_contrast": (
                                np.mean(final_probabilities[true_damage_mask])
                                - np.mean(final_probabilities[~true_damage_mask])
                            ),
                        }
                    )
                for nuisance_index, nuisance_name in enumerate(
                    JOINT_NUISANCE_PARAMETER_NAMES
                ):
                    parameter_index = n_stiffness + nuisance_index
                    posterior_mean = history.posterior_means[-1, parameter_index]
                    nuisance_rows.append(
                        {
                            "layout": layout_name,
                            "case": case_name,
                            "truth_scenario": truth_scenario,
                            "replicate": replicate,
                            "parameter": nuisance_name,
                            "true_parameter": true_nuisance[nuisance_index],
                            "final_posterior_mean": posterior_mean,
                            "final_error": (
                                posterior_mean - true_nuisance[nuisance_index]
                            ),
                            "final_standard_deviation": posterior_scale[
                                -1,
                                parameter_index,
                            ],
                        }
                    )

    runs = pd.DataFrame(run_rows)
    detections = pd.DataFrame(detection_rows)
    localization = pd.DataFrame(localization_rows)
    nuisance = pd.DataFrame(nuisance_rows)
    runtimes = pd.DataFrame(runtime_rows)
    case_summary = summarize_cases(
        runs,
        detections,
        localization,
        layout_names,
        cases,
    )
    layout_summary = summarize_layouts(
        runs,
        detections,
        localization,
        case_summary,
        identifiability,
        layouts,
    )
    indexed = layout_summary.set_index("layout")
    verification = sensor_config["verification"]
    hypotheses = sensor_config["hypotheses"]
    operational_layout = str(verification["operational_layout"])
    if operational_layout not in layouts:
        raise ValueError("The operational verification layout is not evaluated.")
    three_to_top_rmse = indexed.loc[
        "three_spaced", "damaged_pooled_rmse"
    ] / indexed.loc["top_only", "damaged_pooled_rmse"]
    three_to_top_width = indexed.loc[
        "three_spaced", "damaged_mean_credible_interval_width"
    ] / indexed.loc["top_only", "damaged_mean_credible_interval_width"]
    localization_improvement = indexed.loc[
        "three_spaced", "localization_accuracy"
    ] - indexed.loc["top_only", "localization_accuracy"]
    three_to_clustered_rmse = indexed.loc[
        "three_spaced", "damaged_pooled_rmse"
    ] / indexed.loc["three_clustered", "damaged_pooled_rmse"]
    clustered_to_spaced_condition = indexed.loc[
        "three_clustered", "data_only_normalized_condition_number"
    ] / indexed.loc[
        "three_spaced", "data_only_normalized_condition_number"
    ]
    full_improvement = 1.0 - indexed.loc[
        "full_reference", "damaged_pooled_rmse"
    ] / indexed.loc["three_spaced", "damaged_pooled_rmse"]
    check_rows = [
        {
            "category": "verification",
            "check": "matched false alarms controlled for operational layout",
            "observed": indexed.loc[
                operational_layout,
                "matched_false_alarm_fraction",
            ],
            "comparator": "<=",
            "threshold": float(
                verification["maximum_matched_false_alarm_fraction"]
            ),
        },
        {
            "category": "verification",
            "check": "optimizer convergence",
            "observed": np.max(case_summary["optimizer_failure_fraction"]),
            "comparator": "<=",
            "threshold": float(
                verification["maximum_optimizer_failure_fraction"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "three spaced sensors improve RMSE over top only",
            "observed": three_to_top_rmse,
            "comparator": "<=",
            "threshold": float(
                hypotheses["maximum_three_spaced_to_top_only_rmse_ratio"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "three spaced sensors contract uncertainty",
            "observed": three_to_top_width,
            "comparator": "<=",
            "threshold": float(
                hypotheses[
                    "maximum_three_spaced_to_top_only_interval_width_ratio"
                ]
            ),
        },
        {
            "category": "hypothesis",
            "check": "three spaced sensors improve localisation",
            "observed": localization_improvement,
            "comparator": ">=",
            "threshold": float(
                hypotheses["minimum_three_spaced_localization_improvement"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "spaced placement improves RMSE over clustered placement",
            "observed": three_to_clustered_rmse,
            "comparator": "<=",
            "threshold": float(
                hypotheses["maximum_three_spaced_to_clustered_rmse_ratio"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "three spaced sensors retain damage detection",
            "observed": indexed.loc[
                "three_spaced", "damage_detection_fraction"
            ],
            "comparator": ">=",
            "threshold": float(
                hypotheses["minimum_three_spaced_detection_fraction"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "three spaced sensors control mass-discrepancy alarms",
            "observed": indexed.loc[
                "three_spaced", "mass_discrepancy_false_alarm_fraction"
            ],
            "comparator": "<=",
            "threshold": float(
                hypotheses[
                    "maximum_three_spaced_mass_false_alarm_fraction"
                ]
            ),
        },
        {
            "category": "hypothesis",
            "check": "spaced placement improves information conditioning",
            "observed": clustered_to_spaced_condition,
            "comparator": ">=",
            "threshold": float(
                hypotheses["minimum_clustered_to_spaced_condition_ratio"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "six sensors show diminishing RMSE returns",
            "observed": full_improvement,
            "comparator": "<=",
            "threshold": float(
                hypotheses["maximum_full_reference_rmse_improvement_fraction"]
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
    case_summary.to_csv(CASE_SUMMARY_PATH, index=False, float_format="%.12g")
    layout_summary.to_csv(LAYOUT_SUMMARY_PATH, index=False, float_format="%.12g")
    detections.to_csv(DETECTION_PATH, index=False, float_format="%.12g")
    localization.to_csv(LOCALIZATION_PATH, index=False, float_format="%.12g")
    nuisance.to_csv(NUISANCE_PATH, index=False, float_format="%.12g")
    alarm_calibration.to_csv(
        CALIBRATION_PATH,
        index=False,
        float_format="%.12g",
    )
    identifiability.to_csv(
        IDENTIFIABILITY_PATH,
        index=False,
        float_format="%.12g",
    )
    checks.to_csv(CHECKS_PATH, index=False, float_format="%.12g")
    runtimes.to_csv(RUNTIME_PATH, index=False, float_format="%.12g")
    create_stage10_figure(layout_summary)

    print("STAGE 10 - PAIRED SENSOR-INFORMATION STUDY")
    print(
        "Layout          | Sensors | Rank | RMSE  | Width | Detection | "
        "Localisation | Mass false"
    )
    print(
        "----------------+---------+------+-------+-------+-----------+"
        "--------------+-----------"
    )
    for row in layout_summary.itertuples(index=False):
        print(
            f"{row.layout:<15} | {row.n_sensors:>7} | "
            f"{int(row.data_only_effective_rank):>4}/7 | "
            f"{row.damaged_pooled_rmse:>5.3f} | "
            f"{row.damaged_mean_credible_interval_width:>5.3f} | "
            f"{row.damage_detection_fraction:>9.3f} | "
            f"{row.localization_accuracy:>12.3f} | "
            f"{row.mass_discrepancy_false_alarm_fraction:>9.3f}"
        )
    hypothesis_checks = checks[checks["category"] == "hypothesis"]
    print("")
    print(
        "Predeclared hypotheses supported: "
        f"{int(hypothesis_checks['supported'].sum())}/{len(hypothesis_checks)}"
    )
    print(f"Layout summary: {LAYOUT_SUMMARY_PATH}")
    print(f"Case summary: {CASE_SUMMARY_PATH}")
    print(f"Hypothesis checks: {CHECKS_PATH}")
    print(f"Runtime: {RUNTIME_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    if not failed_verification.empty:
        failed_names = ", ".join(failed_verification["check"])
        raise RuntimeError(f"Stage 10 verification failed: {failed_names}")
    print("RESULT: PASS (experiment complete; weak layouts retained)")


if __name__ == "__main__":
    main()
