"""Run the Stage 9 discrepancy-aware joint inference study."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
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
    run_method,
)

from probdt.diagnostics import analyze_identifiability  # noqa: E402
from probdt.inference import (  # noqa: E402
    JOINT_NUISANCE_PARAMETER_NAMES,
    SequentialHistory,
    joint_discrepancy_acceleration_auto_psd,
    joint_spectral_parameter_names,
    log_spectral_jacobian,
    run_recursive_spectral_map,
)
from probdt.monitoring import SensorLayout  # noqa: E402

FloatArray = NDArray[np.float64]
SpectralForwardModel = Callable[[FloatArray], FloatArray]

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
TRACKING_CONFIG_PATH = PROJECT_ROOT / "config" / "sequential_tracking.yaml"
JOINT_CONFIG_PATH = PROJECT_ROOT / "config" / "joint_discrepancy_inference.yaml"
RUNS_PATH = PROJECT_ROOT / "results" / "metrics" / "stage9_joint_runs.csv"
NUISANCE_RUNS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage9_nuisance_runs.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage9_method_summary.csv"
)
NUISANCE_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage9_nuisance_summary.csv"
)
DETECTION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage9_detection_events.csv"
)
CALIBRATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage9_alarm_calibration.csv"
)
IDENTIFIABILITY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage9_identifiability_summary.csv"
)
CORRELATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage9_identifiability_correlation.csv"
)
CHECKS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage9_hypothesis_checks.csv"
)
RUNTIME_PATH = PROJECT_ROOT / "results" / "metrics" / "stage9_runtime.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage9_joint_discrepancy_inference.png"


@dataclass(frozen=True)
class JointMethodDefinition:
    """Numerical definition of one seven-parameter recursive twin."""

    parameter_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    initial_prior_mean: FloatArray
    initial_prior_covariance: FloatArray
    process_covariance: FloatArray


def make_joint_forward_model(
    system,
    frequencies_hz: FloatArray,
    force_distribution: FloatArray,
    layout: SensorLayout,
    excitation_config: dict[str, object],
    measurement_noise_psd: FloatArray,
) -> SpectralForwardModel:
    """Create a seven-parameter stiffness--nuisance spectrum mapping."""

    def forward(parameters: FloatArray) -> FloatArray:
        return joint_discrepancy_acceleration_auto_psd(
            system.mass_matrix_kg,
            system.damping_matrix_n_s_per_m,
            system.zone_components_n_per_m,
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
            measurement_noise_psd=measurement_noise_psd,
        )

    return forward


def build_joint_method_definition(
    specification: dict[str, object],
    tracking_config: dict[str, object],
) -> JointMethodDefinition:
    """Combine Stage 6 stiffness settings with configured nuisance settings."""

    parameter_config = tracking_config["parameter_model"]
    process_name = tracking_config["model_selection"][
        "recommended_process_model"
    ]
    stiffness_names = tuple(str(name) for name in parameter_config["names"])
    parameter_names = joint_spectral_parameter_names(stiffness_names)
    stiffness_bounds = tuple(
        (float(lower), float(upper))
        for lower, upper in zip(
            parameter_config["lower_bounds"],
            parameter_config["upper_bounds"],
            strict=True,
        )
    )
    nuisance_bounds = tuple(
        (float(lower), float(upper))
        for lower, upper in zip(
            specification["nuisance_lower_bounds"],
            specification["nuisance_upper_bounds"],
            strict=True,
        )
    )
    prior_mean = np.asarray(
        [
            *specification["stiffness_initial_prior_mean"],
            *specification["nuisance_initial_prior_mean"],
        ],
        dtype=np.float64,
    )
    prior_scale = np.asarray(
        [
            *specification["stiffness_initial_prior_standard_deviation"],
            *specification["nuisance_initial_prior_standard_deviation"],
        ],
        dtype=np.float64,
    )
    stiffness_process_scale = tracking_config["process_models"][process_name][
        "standard_deviation_per_window"
    ]
    process_scale = np.asarray(
        [
            *stiffness_process_scale,
            *specification["nuisance_process_standard_deviation"],
        ],
        dtype=np.float64,
    )
    if prior_mean.shape != (len(parameter_names),):
        raise ValueError("A joint prior does not match the parameter vector.")
    if prior_scale.shape != prior_mean.shape or np.any(prior_scale <= 0.0):
        raise ValueError("Joint prior scales must be positive and complete.")
    if process_scale.shape != prior_mean.shape or np.any(process_scale < 0.0):
        raise ValueError("Joint process scales must be nonnegative and complete.")
    bounds = (*stiffness_bounds, *nuisance_bounds)
    bound_array = np.asarray(bounds, dtype=np.float64)
    if np.any(prior_mean < bound_array[:, 0]) or np.any(
        prior_mean > bound_array[:, 1]
    ):
        raise ValueError("A joint prior mean lies outside its bounds.")
    return JointMethodDefinition(
        parameter_names=parameter_names,
        bounds=bounds,
        initial_prior_mean=prior_mean,
        initial_prior_covariance=np.diag(prior_scale**2),
        process_covariance=np.diag(process_scale**2),
    )


def run_configured_method(
    method_name: str,
    specification: dict[str, object],
    observations: FloatArray,
    nominal_forward: SpectralForwardModel,
    joint_forward: SpectralForwardModel,
    tracking_config: dict[str, object],
    joint_definitions: dict[str, JointMethodDefinition],
) -> SequentialHistory:
    """Run one stiffness-only or joint recursive Laplace method."""

    if specification["model"] == "stiffness_only":
        process_name = tracking_config["model_selection"][
            "recommended_process_model"
        ]
        return run_method(
            str(process_name),
            nominal_forward,
            observations,
            tracking_config,
        )
    if specification["model"] != "joint":
        raise ValueError(f"Unknown Stage 9 method model for {method_name!r}.")
    definition = joint_definitions[method_name]
    parameter_config = tracking_config["parameter_model"]
    monitoring_config = tracking_config["monitoring_sequence"]
    optimizer_config = tracking_config["optimizer"]
    return run_recursive_spectral_map(
        joint_forward,
        observations,
        definition.parameter_names,
        definition.bounds,
        initial_prior_mean=definition.initial_prior_mean,
        initial_prior_covariance=definition.initial_prior_covariance,
        process_covariance=definition.process_covariance,
        n_independent_averages=int(
            monitoring_config["independent_periodogram_averages"]
        ),
        credible_level=float(parameter_config["credible_level"]),
        derivative_relative_step=float(
            optimizer_config["derivative_relative_step"]
        ),
        optimizer_max_iterations=int(optimizer_config["maximum_iterations"]),
    )


def stiffness_loss_probabilities(
    history: SequentialHistory,
    loss_threshold: float,
    n_stiffness_parameters: int,
) -> FloatArray:
    """Return bounded deterioration probabilities for stiffness entries only."""

    return history.loss_exceedance_probability(loss_threshold)[
        :, :n_stiffness_parameters
    ]


def nuisance_truth_vector(specification: dict[str, object]) -> FloatArray:
    """Return truth scales in the declared joint nuisance order."""

    discrepancy = parse_discrepancy(specification)
    return np.asarray(
        [
            discrepancy.mass_scale,
            discrepancy.damping_scale,
            discrepancy.excitation_standard_deviation_scale,
            discrepancy.excitation_correlation_time_scale,
        ],
        dtype=np.float64,
    )


def parameter_correlation(covariance: FloatArray) -> FloatArray:
    """Convert a positive covariance matrix to correlation form."""

    scale = np.sqrt(np.diag(covariance))
    return covariance / np.outer(scale, scale)


def run_identifiability_analysis(
    joint_forward: SpectralForwardModel,
    definition: JointMethodDefinition,
    configuration: dict[str, object],
    n_averages: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify data-only rank loss and prior-regularized conditioning."""

    reference = np.asarray(
        [
            *configuration["reference_stiffness_multipliers"],
            *configuration["reference_nuisance_scales"],
        ],
        dtype=np.float64,
    )
    jacobian = log_spectral_jacobian(
        joint_forward,
        reference,
        definition.bounds,
        relative_step=float(configuration["derivative_relative_step"]),
    )
    report = analyze_identifiability(
        jacobian,
        definition.parameter_names,
        effective_rank_relative_threshold=float(
            configuration["effective_rank_relative_threshold"]
        ),
    )
    prior_precision = np.linalg.inv(definition.initial_prior_covariance)
    regularized_precision = n_averages * (jacobian.T @ jacobian) + prior_precision
    regularized_covariance = np.linalg.inv(regularized_precision)
    regularized_correlation = parameter_correlation(regularized_covariance)
    theta_mass_correlation = np.max(
        np.abs(
            report.normalized_parameter_correlation[:3, 3]
        )
    )
    summary = pd.DataFrame(
        [
            {
                "n_parameters": len(definition.parameter_names),
                "data_only_effective_rank": report.effective_rank,
                "data_only_rank_deficiency": (
                    len(definition.parameter_names) - report.effective_rank
                ),
                "data_only_normalized_condition_number": (
                    report.normalized_condition_number
                ),
                "data_only_maximum_absolute_correlation": (
                    report.maximum_absolute_parameter_correlation
                ),
                "data_only_maximum_theta_mass_correlation": (
                    theta_mass_correlation
                ),
                "prior_regularized_precision_condition_number": np.linalg.cond(
                    regularized_precision
                ),
                "prior_regularized_maximum_absolute_correlation": np.max(
                    np.abs(
                        regularized_correlation
                        - np.eye(regularized_correlation.shape[0])
                    )
                ),
            }
        ]
    )
    correlation = pd.DataFrame(
        report.normalized_parameter_correlation,
        index=definition.parameter_names,
        columns=definition.parameter_names,
    )
    return summary, correlation


def summarize_stiffness_runs(
    runs: pd.DataFrame,
    detections: pd.DataFrame,
    cases: dict[str, dict[str, object]],
    methods: dict[str, dict[str, object]],
    n_windows: int,
) -> pd.DataFrame:
    """Summarize estimation, uncertainty, and decision performance."""

    rows: list[dict[str, object]] = []
    for method_name in methods:
        for case_name, specification in cases.items():
            selected = runs[
                (runs["method"] == method_name) & (runs["case"] == case_name)
            ]
            events = detections[
                (detections["method"] == method_name)
                & (detections["case"] == case_name)
            ]
            damage_events = events[events["damage_zone"]]
            nondamage_events = events[~events["damage_zone"]]
            interior = selected[selected["true_parameter"] < 0.999]
            late = selected[selected["window"] >= n_windows - 5]
            rows.append(
                {
                    "method": method_name,
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
                    "late_apparent_loss": (
                        (1.0 - late["posterior_mean"])
                        .groupby([late["replicate"], late["window"]])
                        .max()
                        .mean()
                    ),
                    "optimizer_failure_fraction": 1.0
                    - np.mean(selected["optimizer_success"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_nuisance_runs(nuisance_runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize nuisance recovery across time and at the final window."""

    rows: list[dict[str, object]] = []
    grouped = nuisance_runs.groupby(["method", "case", "parameter"], sort=False)
    for (method, case, parameter), selected in grouped:
        final = selected[selected["window"] == selected["window"].max()]
        rows.append(
            {
                "method": method,
                "case": case,
                "parameter": parameter,
                "true_parameter": selected["true_parameter"].iloc[0],
                "trajectory_rmse": np.sqrt(np.mean(selected["error"] ** 2)),
                "final_mean": np.mean(final["posterior_mean"]),
                "final_bias": np.mean(final["error"]),
                "final_standard_deviation": np.mean(
                    final["posterior_standard_deviation"]
                ),
                "final_coverage": np.mean(final["covered"]),
            }
        )
    return pd.DataFrame(rows)


def create_stage9_figure(
    runs: pd.DataFrame,
    nuisance_runs: pd.DataFrame,
    summary: pd.DataFrame,
    correlation: pd.DataFrame,
    time_days: FloatArray,
) -> None:
    """Plot false alarms, mass allocation, damage accuracy, and confounding."""

    colors = {
        "nominal": "#D55E00",
        "joint_commissioning": "#0072B2",
        "joint_weak": "#009E73",
    }
    labels = {
        "nominal": "Nominal",
        "joint_commissioning": "Joint, commissioning prior",
        "joint_weak": "Joint, weak prior",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), constrained_layout=True)
    false_axis, mass_axis, damage_axis, correlation_axis = axes.ravel()

    healthy_cases = [
        "mass_plus_5",
        "damping_plus_30",
        "excitation_plus_15",
        "correlation_time_plus_30",
        "combined_healthy",
    ]
    positions = np.arange(len(healthy_cases), dtype=np.float64)
    width = 0.24
    for method_index, method in enumerate(colors):
        values = [
            summary.loc[
                (summary["method"] == method) & (summary["case"] == case),
                "nondamage_false_alarm_fraction",
            ].iloc[0]
            for case in healthy_cases
        ]
        false_axis.bar(
            positions + (method_index - 1) * width,
            values,
            width=width,
            color=colors[method],
            label=labels[method],
        )
    false_axis.set_xticks(
        positions,
        labels=[case.replace("_", " ") for case in healthy_cases],
        rotation=22,
        ha="right",
    )
    false_axis.set_ylim(0.0, 1.05)
    false_axis.set_ylabel("Persistent false-alarm fraction")
    false_axis.set_title("Healthy discrepancy misclassified as damage")
    false_axis.legend(fontsize=8)
    false_axis.grid(axis="y", alpha=0.25)

    mass_case = runs[
        (runs["case"] == "mass_plus_5") & (runs["replicate"] == 0)
    ]
    for method in colors:
        selected = mass_case[mass_case["method"] == method]
        trajectory = selected.groupby("window")["posterior_mean"].mean()
        mass_axis.plot(
            time_days,
            trajectory,
            color=colors[method],
            linewidth=1.8,
            label=f"{labels[method]} stiffness",
        )
    mass_axis.axhline(1.0, color="black", linewidth=1.8, label="Healthy stiffness")
    mass_axis.set_xlabel("Monitoring time (days)")
    mass_axis.set_ylabel("Mean stiffness multiplier")
    mass_axis.set_ylim(0.85, 1.01)
    mass_axis.grid(alpha=0.25)
    nuisance_axis = mass_axis.twinx()
    for method, linestyle in (
        ("joint_commissioning", "--"),
        ("joint_weak", ":"),
    ):
        selected = nuisance_runs[
            (nuisance_runs["case"] == "mass_plus_5")
            & (nuisance_runs["replicate"] == 0)
            & (nuisance_runs["method"] == method)
            & (nuisance_runs["parameter"] == "mass_scale")
        ]
        nuisance_axis.plot(
            time_days,
            selected["posterior_mean"],
            color=colors[method],
            linestyle=linestyle,
            linewidth=1.5,
            label=f"{labels[method]} mass",
        )
    nuisance_axis.axhline(1.05, color="0.35", linestyle="--", linewidth=1.2)
    nuisance_axis.set_ylabel("Inferred mass scale")
    nuisance_axis.set_ylim(0.90, 1.06)
    handles, legend_labels = mass_axis.get_legend_handles_labels()
    other_handles, other_labels = nuisance_axis.get_legend_handles_labels()
    mass_axis.legend(
        [*handles, *other_handles],
        [*legend_labels, *other_labels],
        fontsize=7,
        loc="lower right",
    )
    mass_axis.set_title("Allocation of a persistent +5% mass discrepancy")

    damaged_cases = [
        "gradual_lower_mass_plus_5",
        "abrupt_middle_damping_plus_30",
        "multi_zone_combined",
    ]
    damage_positions = np.arange(len(damaged_cases), dtype=np.float64)
    for method_index, method in enumerate(colors):
        values = [
            summary.loc[
                (summary["method"] == method) & (summary["case"] == case),
                "pooled_rmse",
            ].iloc[0]
            for case in damaged_cases
        ]
        bars = damage_axis.bar(
            damage_positions + (method_index - 1) * width,
            values,
            width=width,
            color=colors[method],
            label=labels[method],
        )
        for bar, case in zip(bars, damaged_cases, strict=True):
            detection = summary.loc[
                (summary["method"] == method) & (summary["case"] == case),
                "damage_detection_fraction",
            ].iloc[0]
            damage_axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.001,
                f"{detection:.0%}",
                ha="center",
                fontsize=7,
            )
    damage_axis.set_xticks(
        damage_positions,
        labels=[case.replace("_", " ") for case in damaged_cases],
        rotation=20,
        ha="right",
    )
    damage_axis.set_ylabel("Stiffness trajectory RMSE")
    damage_axis.set_title("Damage accuracy (labels give detection fraction)")
    damage_axis.legend(fontsize=8)
    damage_axis.grid(axis="y", alpha=0.25)

    correlation_image = correlation_axis.imshow(
        correlation.to_numpy(),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    name_map = {
        "theta_lower": "lower",
        "theta_middle": "middle",
        "theta_upper": "upper",
        "mass_scale": "mass",
        "damping_scale": "damping",
        "excitation_standard_deviation_scale": "force amplitude",
        "excitation_correlation_time_scale": "force colour",
    }
    short_names = [name_map[name] for name in correlation.columns]
    correlation_axis.set_xticks(
        np.arange(len(short_names)),
        labels=short_names,
        rotation=35,
        ha="right",
    )
    correlation_axis.set_yticks(np.arange(len(short_names)), labels=short_names)
    correlation_axis.set_title("Data-only normalized sensitivity correlation")
    for row in range(len(short_names)):
        for column in range(len(short_names)):
            value = correlation.iloc[row, column]
            correlation_axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else "black",
                fontsize=6,
            )
    figure.colorbar(correlation_image, ax=correlation_axis, label="Correlation")
    figure.suptitle(
        "Stage 9: discrepancy-aware joint inference and its identifiability cost",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main() -> None:
    """Execute calibration, identifiability, and joint-inference experiments."""

    with BASE_CONFIG_PATH.open(encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    with MONITORING_CONFIG_PATH.open(encoding="utf-8") as stream:
        monitoring_config = yaml.safe_load(stream)
    with TRACKING_CONFIG_PATH.open(encoding="utf-8") as stream:
        tracking_config = yaml.safe_load(stream)
    with JOINT_CONFIG_PATH.open(encoding="utf-8") as stream:
        joint_config = yaml.safe_load(stream)

    sequence_config = tracking_config["monitoring_sequence"]
    parameter_config = tracking_config["parameter_model"]
    alarm_config = tracking_config["alarm"]
    excitation_config = monitoring_config["excitation"]
    methods = joint_config["methods"]
    cases = joint_config["cases"]
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
    truth_system = build_spectral_system(base_config)
    twin_system = build_spectral_system(base_config)
    zero_noise = np.zeros(layout.n_sensors)
    clean_forward = make_forward_model(
        truth_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        zero_noise,
    )
    clean_psd = clean_forward(np.ones(n_stiffness))
    noise_fraction = float(sequence_config["noise_amplitude_fraction"])
    measurement_noise_psd = noise_fraction**2 * np.max(clean_psd, axis=0)
    nominal_forward = make_forward_model(
        twin_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        measurement_noise_psd,
    )
    joint_forward = make_joint_forward_model(
        twin_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        measurement_noise_psd,
    )
    joint_definitions = {
        method_name: build_joint_method_definition(specification, tracking_config)
        for method_name, specification in methods.items()
        if specification["model"] == "joint"
    }
    histories = build_truth_histories(tracking_config, time_days, zone_names)
    n_averages = int(sequence_config["independent_periodogram_averages"])
    loss_threshold = float(alarm_config["meaningful_loss_fraction"])
    persistence = int(alarm_config["persistence_windows"])

    identifiability, correlation = run_identifiability_analysis(
        joint_forward,
        joint_definitions["joint_commissioning"],
        joint_config["identifiability"],
        n_averages,
    )

    identity_truth_forward = make_discrepant_truth_forward(
        truth_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        measurement_noise_psd,
        parse_discrepancy({}),
    )
    healthy_truth = histories["healthy"].stiffness_multipliers
    calibration_config = joint_config["calibration"]
    calibration_seeds = np.random.SeedSequence(
        int(calibration_config["random_seed"])
    ).spawn(int(calibration_config["healthy_replicates"]))
    calibration_probabilities = {name: [] for name in methods}
    for seed_sequence in calibration_seeds:
        raw = generate_windows(
            identity_truth_forward,
            healthy_truth,
            n_averages,
            np.random.default_rng(seed_sequence),
        )
        for method_name, specification in methods.items():
            history = run_configured_method(
                method_name,
                specification,
                raw,
                nominal_forward,
                joint_forward,
                tracking_config,
                joint_definitions,
            )
            calibration_probabilities[method_name].append(
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

    evaluation_config = joint_config["evaluation"]
    n_replicates = int(evaluation_config["replicates"])
    evaluation_seeds = np.random.SeedSequence(
        int(evaluation_config["random_seed"])
    ).spawn(len(cases) * n_replicates)
    seed_iterator = iter(evaluation_seeds)
    run_rows: list[dict[str, object]] = []
    nuisance_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
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
        true_nuisance = nuisance_truth_vector(specification)
        for replicate in range(n_replicates):
            seed_sequence = next(seed_iterator)
            seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
            raw = generate_windows(
                truth_forward,
                truth_parameters,
                n_averages,
                np.random.default_rng(seed_sequence),
            )
            for method_name, method_specification in methods.items():
                start_time = time.perf_counter()
                history = run_configured_method(
                    method_name,
                    method_specification,
                    raw,
                    nominal_forward,
                    joint_forward,
                    tracking_config,
                    joint_definitions,
                )
                runtime_rows.append(
                    {
                        "method": method_name,
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
                    alarm_thresholds[method_name],
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
                            "method": method_name,
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
                                "method": method_name,
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
                                    window_index, zone_index
                                ],
                                "credible_lower": history.credible_interval_lower[
                                    window_index, zone_index
                                ],
                                "credible_upper": history.credible_interval_upper[
                                    window_index, zone_index
                                ],
                                "covered": history.credible_interval_lower[
                                    window_index, zone_index
                                ]
                                <= truth_value
                                <= history.credible_interval_upper[
                                    window_index, zone_index
                                ],
                                "damage_probability": probabilities[
                                    window_index, zone_index
                                ],
                                "optimizer_success": history.optimizer_success[
                                    window_index
                                ],
                            }
                        )
                if method_specification["model"] == "joint":
                    for nuisance_index, nuisance_name in enumerate(
                        JOINT_NUISANCE_PARAMETER_NAMES
                    ):
                        parameter_index = n_stiffness + nuisance_index
                        truth_value = true_nuisance[nuisance_index]
                        for window_index in range(n_windows):
                            posterior_mean = history.posterior_means[
                                window_index, parameter_index
                            ]
                            nuisance_rows.append(
                                {
                                    "method": method_name,
                                    "case": case_name,
                                    "truth_scenario": truth_scenario,
                                    "replicate": replicate,
                                    "seed": seed,
                                    "window": window_index,
                                    "time_day": time_days[window_index],
                                    "parameter": nuisance_name,
                                    "true_parameter": truth_value,
                                    "posterior_mean": posterior_mean,
                                    "error": posterior_mean - truth_value,
                                    "posterior_standard_deviation": posterior_scale[
                                        window_index, parameter_index
                                    ],
                                    "credible_lower": (
                                        history.credible_interval_lower[
                                            window_index, parameter_index
                                        ]
                                    ),
                                    "credible_upper": (
                                        history.credible_interval_upper[
                                            window_index, parameter_index
                                        ]
                                    ),
                                    "covered": (
                                        history.credible_interval_lower[
                                            window_index, parameter_index
                                        ]
                                        <= truth_value
                                        <= history.credible_interval_upper[
                                            window_index, parameter_index
                                        ]
                                    ),
                                }
                            )

    runs = pd.DataFrame(run_rows)
    nuisance_runs = pd.DataFrame(nuisance_rows)
    detections = pd.DataFrame(detection_rows)
    runtimes = pd.DataFrame(runtime_rows)
    summary = summarize_stiffness_runs(
        runs,
        detections,
        cases,
        methods,
        n_windows,
    )
    nuisance_summary = summarize_nuisance_runs(nuisance_runs)

    indexed = summary.set_index(["method", "case"])
    damaged_runs = runs[runs["truth_scenario"] != "healthy"]
    damaged_events = detections[detections["damage_zone"]]
    aggregate_rows: list[dict[str, object]] = []
    for method_name in methods:
        selected_runs = damaged_runs[damaged_runs["method"] == method_name]
        selected_events = damaged_events[
            damaged_events["method"] == method_name
        ]
        aggregate_rows.append(
            {
                "method": method_name,
                "damaged_rmse": np.sqrt(np.mean(selected_runs["error"] ** 2)),
                "damage_detection_fraction": np.mean(selected_events["detected"]),
                "damaged_interior_coverage": np.mean(
                    selected_runs.loc[
                        selected_runs["true_parameter"] < 0.999,
                        "covered",
                    ]
                ),
                "damaged_interval_width": np.mean(
                    selected_runs["credible_upper"]
                    - selected_runs["credible_lower"]
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).set_index("method")
    verification = joint_config["verification"]
    hypotheses = joint_config["hypotheses"]
    operational_methods = tuple(
        str(method) for method in verification["operational_methods"]
    )
    if any(method not in methods for method in operational_methods):
        raise ValueError("An operational verification method is not configured.")
    mass_false_alarm_reduction = indexed.loc[
        ("nominal", "mass_plus_5"), "nondamage_false_alarm_fraction"
    ] - indexed.loc[
        ("joint_commissioning", "mass_plus_5"),
        "nondamage_false_alarm_fraction",
    ]
    damaged_rmse_ratio = aggregate.loc[
        "joint_commissioning", "damaged_rmse"
    ] / aggregate.loc["nominal", "damaged_rmse"]
    interval_width_ratio = aggregate.loc[
        "joint_weak", "damaged_interval_width"
    ] / aggregate.loc["joint_commissioning", "damaged_interval_width"]
    check_rows = [
        {
            "category": "verification",
            "check": "matched false alarms controlled for operational methods",
            "observed": np.max(
                [
                    indexed.loc[
                        (method, "matched_healthy"),
                        "nondamage_false_alarm_fraction",
                    ]
                    for method in operational_methods
                ]
            ),
            "comparator": "<=",
            "threshold": float(
                verification["maximum_matched_false_alarm_fraction"]
            ),
        },
        {
            "category": "verification",
            "check": "optimizer convergence",
            "observed": np.max(summary["optimizer_failure_fraction"]),
            "comparator": "<=",
            "threshold": float(
                verification["maximum_optimizer_failure_fraction"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "joint inference reduces mass-discrepancy false alarms",
            "observed": mass_false_alarm_reduction,
            "comparator": ">=",
            "threshold": float(
                hypotheses["minimum_mass_false_alarm_reduction"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "combined healthy discrepancy remains controlled",
            "observed": indexed.loc[
                ("joint_commissioning", "combined_healthy"),
                "nondamage_false_alarm_fraction",
            ],
            "comparator": "<=",
            "threshold": float(
                hypotheses[
                    "maximum_joint_combined_healthy_false_alarm_fraction"
                ]
            ),
        },
        {
            "category": "hypothesis",
            "check": "joint inference improves damaged-case RMSE",
            "observed": damaged_rmse_ratio,
            "comparator": "<=",
            "threshold": float(
                hypotheses["maximum_joint_to_nominal_damaged_rmse_ratio"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "joint inference retains matched-model accuracy",
            "observed": indexed.loc[
                ("joint_commissioning", "matched_healthy"),
                "pooled_rmse",
            ],
            "comparator": "<=",
            "threshold": float(
                hypotheses["maximum_joint_matched_healthy_rmse"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "joint inference retains damage detection",
            "observed": aggregate.loc[
                "joint_commissioning", "damage_detection_fraction"
            ],
            "comparator": ">=",
            "threshold": float(
                hypotheses["minimum_joint_damage_detection_fraction"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "joint inference retains damaged-state coverage",
            "observed": aggregate.loc[
                "joint_commissioning", "damaged_interior_coverage"
            ],
            "comparator": ">=",
            "threshold": float(
                hypotheses["minimum_joint_damaged_interior_coverage"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "joint data-only model is rank deficient",
            "observed": identifiability.loc[
                0, "data_only_rank_deficiency"
            ],
            "comparator": ">=",
            "threshold": float(
                hypotheses["minimum_data_only_rank_deficiency"]
            ),
        },
        {
            "category": "hypothesis",
            "check": "weak prior inflates stiffness uncertainty",
            "observed": interval_width_ratio,
            "comparator": ">=",
            "threshold": float(
                hypotheses[
                    "minimum_weak_to_commissioning_interval_width_ratio"
                ]
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
    nuisance_runs.to_csv(NUISANCE_RUNS_PATH, index=False, float_format="%.12g")
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    nuisance_summary.to_csv(
        NUISANCE_SUMMARY_PATH,
        index=False,
        float_format="%.12g",
    )
    detections.to_csv(DETECTION_PATH, index=False, float_format="%.12g")
    runtimes.to_csv(RUNTIME_PATH, index=False, float_format="%.12g")
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
    correlation.to_csv(CORRELATION_PATH, float_format="%.12g")
    checks.to_csv(CHECKS_PATH, index=False, float_format="%.12g")
    create_stage9_figure(runs, nuisance_runs, summary, correlation, time_days)

    print("STAGE 9 - DISCREPANCY-AWARE JOINT INFERENCE")
    print(
        "Data-only joint rank: "
        f"{int(identifiability.loc[0, 'data_only_effective_rank'])}/"
        f"{int(identifiability.loc[0, 'n_parameters'])}"
    )
    print("")
    print(
        "Method                  | Mass false alarm | Damaged RMSE | "
        "Detection | Interval width"
    )
    print(
        "------------------------+------------------+--------------+"
        "-----------+---------------"
    )
    for method_name in methods:
        mass_false_alarm = indexed.loc[
            (method_name, "mass_plus_5"),
            "nondamage_false_alarm_fraction",
        ]
        print(
            f"{method_name:<23} | "
            f"{mass_false_alarm:>16.3f} | "
            f"{aggregate.loc[method_name, 'damaged_rmse']:>12.3f} | "
            f"{aggregate.loc[method_name, 'damage_detection_fraction']:>9.3f} | "
            f"{aggregate.loc[method_name, 'damaged_interval_width']:>13.3f}"
        )
    hypothesis_checks = checks[checks["category"] == "hypothesis"]
    print("")
    print(
        "Predeclared hypotheses supported: "
        f"{int(hypothesis_checks['supported'].sum())}/{len(hypothesis_checks)}"
    )
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Nuisance recovery: {NUISANCE_SUMMARY_PATH}")
    print(f"Identifiability: {IDENTIFIABILITY_PATH}")
    print(f"Hypothesis checks: {CHECKS_PATH}")
    print(f"Runtime: {RUNTIME_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    if not failed_verification.empty:
        failed_names = ", ".join(failed_verification["check"])
        raise RuntimeError(f"Stage 9 verification failed: {failed_names}")
    print("RESULT: PASS (experiment complete; stress-case failures retained)")


if __name__ == "__main__":
    main()
