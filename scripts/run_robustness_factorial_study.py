"""Run the Stage 11 systematic robustness factorial study."""

from __future__ import annotations

import itertools
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as student_t

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from run_joint_discrepancy_inference_study import (  # noqa: E402
    JointMethodDefinition,
    build_joint_method_definition,
    make_joint_forward_model,
    nuisance_truth_vector,
    stiffness_loss_probabilities,
)
from run_model_discrepancy_study import (  # noqa: E402
    generate_windows,
    make_discrepant_truth_forward,
    parse_discrepancy,
)
from run_sequential_tracking_study import (  # noqa: E402
    build_spectral_system,
    calibrate_alarm_thresholds,
    first_persistent_alarm_indices,
    make_forward_model,
)

from probdt.deterioration import smooth_gradual_loss  # noqa: E402
from probdt.diagnostics import (  # noqa: E402
    estimate_two_level_factorial_effects,
)
from probdt.inference import (  # noqa: E402
    SequentialHistory,
    run_recursive_spectral_map_varying_observations,
)
from probdt.monitoring import SensorLayout, top_k_localization_success  # noqa: E402

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
TRACKING_CONFIG_PATH = PROJECT_ROOT / "config" / "sequential_tracking.yaml"
JOINT_CONFIG_PATH = PROJECT_ROOT / "config" / "joint_discrepancy_inference.yaml"
ROBUSTNESS_CONFIG_PATH = PROJECT_ROOT / "config" / "robustness_factorial.yaml"

RUNS_PATH = PROJECT_ROOT / "results" / "metrics" / "stage11_factorial_runs.csv"
CELLS_PATH = PROJECT_ROOT / "results" / "metrics" / "stage11_factorial_cells.csv"
EFFECTS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage11_factorial_effects.csv"
)
HEALTHY_PATH = PROJECT_ROOT / "results" / "metrics" / "stage11_healthy_holdout.csv"
CALIBRATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage11_alarm_calibration.csv"
)
TRAJECTORY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage11_challenging_trajectory.csv"
)
CHECKS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage11_hypothesis_checks.csv"
)
RUNTIME_PATH = PROJECT_ROOT / "results" / "metrics" / "stage11_runtime.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage11_robustness_factorial.png"

FACTOR_NAMES = (
    "noise",
    "severity",
    "data_support",
    "dropout",
    "mismatch",
)
RESPONSE_NAMES = (
    "damage_zone_rmse",
    "pooled_rmse",
    "damaged_interval_width",
    "damaged_interval_coverage",
    "detected",
    "localized",
    "unaffected_false_alarm_fraction",
    "nuisance_trajectory_rmse",
)


def _load_yaml(path: Path) -> dict[str, object]:
    """Load one required YAML mapping."""

    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} must contain a mapping.")
    return loaded


def _regime_name(noise: float, n_averages: int, dropout: float) -> str:
    """Create a stable key for one observation-quality regime."""

    return f"noise={noise:.2f}|averages={n_averages}|dropout={dropout:.2f}"


def _dropout_mask(
    n_windows: int,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw window-level loss of the designated top sensor channel."""

    if probability == 0.0:
        return np.zeros(n_windows, dtype=np.bool_)
    return rng.random(n_windows) < probability


def _run_varying_joint_history(
    observations: np.ndarray,
    dropout_mask: np.ndarray,
    full_forward,
    reduced_forward,
    definition: JointMethodDefinition,
    tracking_config: dict[str, object],
    n_averages: int,
    retained_columns: tuple[int, ...],
) -> SequentialHistory:
    """Run the commissioned joint twin with honest sensor omission."""

    if observations.ndim != 3 or observations.shape[0] != dropout_mask.size:
        raise ValueError("Observations and dropout mask must share their windows.")
    window_observations = [
        (
            observations[index][:, list(retained_columns)].copy()
            if dropped
            else observations[index].copy()
        )
        for index, dropped in enumerate(dropout_mask)
    ]
    forward_models = [
        reduced_forward if dropped else full_forward for dropped in dropout_mask
    ]
    parameter_config = tracking_config["parameter_model"]
    optimizer_config = tracking_config["optimizer"]
    return run_recursive_spectral_map_varying_observations(
        forward_models,
        window_observations,
        definition.parameter_names,
        definition.bounds,
        initial_prior_mean=definition.initial_prior_mean,
        initial_prior_covariance=definition.initial_prior_covariance,
        process_covariance=definition.process_covariance,
        n_independent_averages=n_averages,
        credible_level=float(parameter_config["credible_level"]),
        derivative_relative_step=float(
            optimizer_config["derivative_relative_step"]
        ),
        optimizer_max_iterations=int(optimizer_config["maximum_iterations"]),
    )


def _history_metrics(
    history: SequentialHistory,
    truth: np.ndarray,
    nuisance_truth: np.ndarray,
    alarm_threshold: float,
    loss_threshold: float,
    persistence: int,
) -> dict[str, object]:
    """Reduce one damaged trajectory to predeclared estimation metrics."""

    n_stiffness = truth.shape[1]
    estimates = history.posterior_means[:, :n_stiffness]
    probabilities = stiffness_loss_probabilities(
        history,
        loss_threshold,
        n_stiffness,
    )
    damaged = truth[:, 0] < 0.999
    crossing_candidates = np.flatnonzero(1.0 - truth[:, 0] >= loss_threshold)
    if crossing_candidates.size == 0:
        raise ValueError("The damaged truth must cross the alarm loss threshold.")
    crossing = int(crossing_candidates[0])
    relative_alarm = first_persistent_alarm_indices(
        probabilities[crossing:, [0]],
        alarm_threshold,
        persistence,
    )[0]
    alarm_window = int(crossing + relative_alarm) if relative_alarm >= 0 else -1
    unaffected_alarms = first_persistent_alarm_indices(
        probabilities[:, 1:],
        alarm_threshold,
        persistence,
    )
    covered = (
        (history.credible_interval_lower[:, 0] <= truth[:, 0])
        & (truth[:, 0] <= history.credible_interval_upper[:, 0])
    )
    localized = top_k_localization_success(
        probabilities[-1],
        np.asarray([True, False, False]),
    )
    nuisance_error = history.posterior_means[:, n_stiffness:] - nuisance_truth
    all_error = estimates - truth
    return {
        "damage_zone_rmse": float(np.sqrt(np.mean(all_error[:, 0] ** 2))),
        "pooled_rmse": float(np.sqrt(np.mean(all_error**2))),
        "damage_zone_bias": float(np.mean(all_error[:, 0])),
        "damaged_interval_width": float(
            np.mean(
                history.credible_interval_upper[damaged, 0]
                - history.credible_interval_lower[damaged, 0]
            )
        ),
        "damaged_interval_coverage": float(np.mean(covered[damaged])),
        "detected": bool(alarm_window >= 0),
        "truth_crossing_window": crossing,
        "alarm_window": alarm_window,
        "detection_delay_windows": (
            float(alarm_window - crossing) if alarm_window >= 0 else np.nan
        ),
        "unaffected_false_alarm_fraction": float(
            np.mean(unaffected_alarms >= 0)
        ),
        "localized": bool(localized),
        "final_damage_probability": float(probabilities[-1, 0]),
        "final_probability_contrast": float(
            probabilities[-1, 0] - np.max(probabilities[-1, 1:])
        ),
        "nuisance_trajectory_rmse": float(np.sqrt(np.mean(nuisance_error**2))),
        "nuisance_final_rmse": float(
            np.sqrt(np.mean(nuisance_error[-1] ** 2))
        ),
        "optimizer_failure_fraction": float(
            1.0 - np.mean(history.optimizer_success)
        ),
        "mean_function_evaluations": float(
            np.mean(history.n_function_evaluations)
        ),
    }


def _healthy_metrics(
    history: SequentialHistory,
    nuisance_truth: np.ndarray,
    alarm_threshold: float,
    loss_threshold: float,
    persistence: int,
) -> dict[str, object]:
    """Reduce one healthy holdout history to false-alarm metrics."""

    probabilities = stiffness_loss_probabilities(history, loss_threshold, 3)
    alarms = first_persistent_alarm_indices(
        probabilities,
        alarm_threshold,
        persistence,
    )
    nuisance_error = history.posterior_means[:, 3:] - nuisance_truth
    return {
        "healthy_false_alarm_fraction": float(np.mean(alarms >= 0)),
        "any_healthy_false_alarm": bool(np.any(alarms >= 0)),
        "maximum_apparent_loss": float(
            np.max(1.0 - history.posterior_means[:, :3])
        ),
        "nuisance_trajectory_rmse": float(np.sqrt(np.mean(nuisance_error**2))),
        "optimizer_failure_fraction": float(
            1.0 - np.mean(history.optimizer_success)
        ),
    }


def _factorial_effect_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Estimate main and two-factor interaction effects for every response."""

    coded = runs[[f"{name}_code" for name in FACTOR_NAMES]].to_numpy(
        dtype=np.float64
    )
    rows: list[dict[str, object]] = []
    for response_name in RESPONSE_NAMES:
        response = runs[response_name].to_numpy(dtype=np.float64)
        analysis = estimate_two_level_factorial_effects(
            coded,
            response,
            FACTOR_NAMES,
        )
        residual_degrees = len(response) - analysis.design_rank
        residual_variance = float(
            np.sum(analysis.residuals**2) / residual_degrees
        )
        effect_standard_error = 2.0 * np.sqrt(
            residual_variance / len(response)
        )
        critical = float(student_t.ppf(0.975, residual_degrees))
        for term_index, term_name in enumerate(analysis.term_names[1:], start=1):
            effect = float(analysis.effects[term_index])
            rows.append(
                {
                    "response": response_name,
                    "term": term_name,
                    "term_order": 2 if ":" in term_name else 1,
                    "mean_response": float(analysis.coefficients[0]),
                    "coefficient": float(analysis.coefficients[term_index]),
                    "high_minus_low_effect": effect,
                    "effect_standard_error": effect_standard_error,
                    "effect_ci_lower": effect
                    - critical * effect_standard_error,
                    "effect_ci_upper": effect
                    + critical * effect_standard_error,
                    "residual_degrees_of_freedom": residual_degrees,
                    "design_rank": analysis.design_rank,
                }
            )
    return pd.DataFrame(rows)


def _representative_rows(
    history: SequentialHistory,
    truth: np.ndarray,
    time_days: np.ndarray,
    dropout_mask: np.ndarray,
    loss_threshold: float,
) -> pd.DataFrame:
    """Make one auditable challenging-corner trajectory table."""

    rows: list[dict[str, object]] = []
    probabilities = stiffness_loss_probabilities(history, loss_threshold, 3)
    for window in range(len(time_days)):
        row: dict[str, object] = {
            "window": window,
            "time_day": time_days[window],
            "top_sensor_dropped": bool(dropout_mask[window]),
        }
        for parameter_index, name in enumerate(history.parameter_names):
            row[f"posterior_mean_{name}"] = history.posterior_means[
                window, parameter_index
            ]
            row[f"credible_lower_{name}"] = history.credible_interval_lower[
                window, parameter_index
            ]
            row[f"credible_upper_{name}"] = history.credible_interval_upper[
                window, parameter_index
            ]
        for zone_index, zone in enumerate(("lower", "middle", "upper")):
            row[f"true_theta_{zone}"] = truth[window, zone_index]
            row[f"loss_probability_{zone}"] = probabilities[window, zone_index]
        rows.append(row)
    return pd.DataFrame(rows)


def _annotated_heatmap(
    axis,
    table: pd.DataFrame,
    *,
    title: str,
    color_map: str,
    value_format: str,
    color_minimum: float | None = None,
    color_maximum: float | None = None,
) -> None:
    """Draw a compact annotated two-by-two robustness heatmap."""

    image = axis.imshow(
        table.to_numpy(),
        cmap=color_map,
        aspect="auto",
        vmin=color_minimum,
        vmax=color_maximum,
    )
    axis.set_xticks(np.arange(table.shape[1]), labels=table.columns)
    axis.set_yticks(np.arange(table.shape[0]), labels=table.index)
    axis.set_xlabel(table.columns.name)
    axis.set_ylabel(table.index.name)
    axis.set_title(title)
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            axis.text(
                column,
                row,
                format(table.iloc[row, column], value_format),
                ha="center",
                va="center",
                color="black",
                bbox={"facecolor": "white", "alpha": 0.62, "edgecolor": "none"},
            )
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _create_figure(
    runs: pd.DataFrame,
    effects: pd.DataFrame,
    healthy: pd.DataFrame,
) -> None:
    """Plot paired degradation of estimation, decisions, and specificity."""

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), constrained_layout=True)
    rmse_axis, detection_axis, effect_axis, healthy_axis = axes.ravel()

    rmse = runs.pivot_table(
        index="measurement_noise_fraction",
        columns="independent_periodogram_averages",
        values="damage_zone_rmse",
        aggfunc="mean",
    )
    rmse.index.name = "Noise amplitude fraction"
    rmse.columns.name = "Independent averages"
    _annotated_heatmap(
        rmse_axis,
        rmse,
        title="Damage-zone trajectory RMSE",
        color_map="magma_r",
        value_format=".3f",
    )

    detection = runs.pivot_table(
        index="final_loss_fraction",
        columns="independent_periodogram_averages",
        values="detected",
        aggfunc="mean",
    )
    detection.index.name = "Final stiffness loss"
    detection.columns.name = "Independent averages"
    _annotated_heatmap(
        detection_axis,
        detection,
        title="Persistent damage-detection fraction",
        color_map="viridis",
        value_format=".2f",
    )

    main_rmse = effects[
        (effects["response"] == "damage_zone_rmse")
        & (effects["term_order"] == 1)
    ]
    labels = [name.replace("_", " ") for name in main_rmse["term"]]
    colors = [
        "#D55E00" if value > 0.0 else "#0072B2"
        for value in main_rmse["high_minus_low_effect"]
    ]
    positions = np.arange(len(main_rmse))
    effect_axis.bar(
        positions,
        main_rmse["high_minus_low_effect"],
        color=colors,
    )
    effect_axis.axhline(0.0, color="black", linewidth=0.9)
    effect_axis.set_xticks(positions, labels=labels, rotation=20, ha="right")
    effect_axis.set_ylabel("High-minus-low effect on RMSE")
    effect_axis.set_title("Adjusted factorial main effects")
    effect_axis.grid(axis="y", alpha=0.25)

    healthy_table = healthy.pivot_table(
        index="model_mismatch",
        columns="top_sensor_dropout_probability",
        values="healthy_false_alarm_fraction",
        aggfunc="mean",
    )
    healthy_table.index.name = "Truth model"
    healthy_table.columns.name = "Dropout probability"
    _annotated_heatmap(
        healthy_axis,
        healthy_table,
        title="Healthy-holdout false-alarm fraction",
        color_map="Reds",
        value_format=".2f",
        color_minimum=0.0,
        color_maximum=0.10,
    )

    figure.suptitle(
        "Stage 11: systematic robustness of the commissioned digital twin",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main() -> None:
    """Execute calibration, the frozen factorial, and healthy holdout checks."""

    started = time.perf_counter()
    base_config = _load_yaml(BASE_CONFIG_PATH)
    monitoring_config = _load_yaml(MONITORING_CONFIG_PATH)
    tracking_config = _load_yaml(TRACKING_CONFIG_PATH)
    joint_config = _load_yaml(JOINT_CONFIG_PATH)
    robustness_config = _load_yaml(ROBUSTNESS_CONFIG_PATH)

    study_config = robustness_config["study"]
    factor_config = robustness_config["factors"]
    layout_specification = monitoring_config["sensor_layouts"][
        study_config["sensor_layout"]
    ]
    full_layout = SensorLayout(
        str(study_config["sensor_layout"]),
        tuple(
            int(index) - 1
            for index in layout_specification["dofs_one_based"]
        ),
    )
    dropout_index = (
        int(
            robustness_config["dropout"][
                "sensor_one_based_within_layout"
            ]
        )
        - 1
    )
    retained_columns = tuple(
        index for index in range(full_layout.n_sensors) if index != dropout_index
    )
    reduced_layout = SensorLayout(
        "three_spaced_without_top",
        tuple(full_layout.dof_indices[index] for index in retained_columns),
    )

    sequence_config = tracking_config["monitoring_sequence"]
    excitation_config = monitoring_config["excitation"]
    alarm_config = tracking_config["alarm"]
    n_windows = int(sequence_config["number_of_windows"])
    time_days = np.arange(n_windows, dtype=np.float64) * float(
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
    zero_noise = np.zeros(full_layout.n_sensors, dtype=np.float64)
    clean_forward = make_forward_model(
        truth_system,
        frequencies_hz,
        force_distribution,
        full_layout,
        excitation_config,
        zero_noise,
    )
    clean_psd = clean_forward(np.ones(3, dtype=np.float64))

    joint_method_name = str(study_config["inference_method"])
    method_specification = joint_config["methods"][joint_method_name]
    if method_specification["model"] != "joint":
        raise ValueError("Stage 11 requires the commissioned joint method.")
    definition = build_joint_method_definition(
        method_specification,
        tracking_config,
    )

    noise_levels = [
        float(value)
        for value in factor_config["measurement_noise_fraction"]
    ]
    severity_levels = [float(value) for value in factor_config["final_loss_fraction"]]
    data_levels = [
        int(value) for value in factor_config["independent_periodogram_averages"]
    ]
    dropout_levels = [
        float(value) for value in factor_config["top_sensor_dropout_probability"]
    ]
    mismatch_levels = [str(value) for value in factor_config["model_mismatch"]]
    for levels in (
        noise_levels,
        severity_levels,
        data_levels,
        dropout_levels,
        mismatch_levels,
    ):
        if len(levels) != 2:
            raise ValueError("Every Stage 11 factor must contain exactly two levels.")

    measurement_noise = {
        noise: noise**2 * np.max(clean_psd, axis=0) for noise in noise_levels
    }
    full_joint_forwards = {
        noise: make_joint_forward_model(
            twin_system,
            frequencies_hz,
            force_distribution,
            full_layout,
            excitation_config,
            measurement_noise[noise],
        )
        for noise in noise_levels
    }
    reduced_joint_forwards = {
        noise: make_joint_forward_model(
            twin_system,
            frequencies_hz,
            force_distribution,
            reduced_layout,
            excitation_config,
            measurement_noise[noise][list(retained_columns)],
        )
        for noise in noise_levels
    }
    truth_forwards = {
        (noise, mismatch): make_discrepant_truth_forward(
            truth_system,
            frequencies_hz,
            force_distribution,
            full_layout,
            excitation_config,
            measurement_noise[noise],
            parse_discrepancy(
                robustness_config["mismatch_cases"][mismatch]
            ),
        )
        for noise in noise_levels
        for mismatch in mismatch_levels
    }
    nuisance_truths = {
        mismatch: nuisance_truth_vector(
            robustness_config["mismatch_cases"][mismatch]
        )
        for mismatch in mismatch_levels
    }

    loss_threshold = float(alarm_config["meaningful_loss_fraction"])
    persistence = int(alarm_config["persistence_windows"])
    healthy_truth = np.ones((n_windows, 3), dtype=np.float64)

    calibration_started = time.perf_counter()
    calibration_config = robustness_config["calibration"]
    calibration_replicates = int(
        calibration_config["healthy_replicates_per_observation_regime"]
    )
    regimes = list(itertools.product(noise_levels, data_levels, dropout_levels))
    calibration_seeds = np.random.SeedSequence(
        int(calibration_config["random_seed"])
    ).spawn(len(regimes) * calibration_replicates)
    seed_index = 0
    calibration_probabilities: dict[str, list[np.ndarray]] = {}
    calibration_dropout_rows: list[dict[str, object]] = []
    for noise, n_averages, dropout_probability in regimes:
        regime = _regime_name(noise, n_averages, dropout_probability)
        calibration_probabilities[regime] = []
        for replicate in range(calibration_replicates):
            spectral_seed, dropout_seed = calibration_seeds[seed_index].spawn(2)
            seed_index += 1
            observations = generate_windows(
                truth_forwards[(noise, "matched")],
                healthy_truth,
                n_averages,
                np.random.default_rng(spectral_seed),
            )
            dropout_mask = _dropout_mask(
                n_windows,
                dropout_probability,
                np.random.default_rng(dropout_seed),
            )
            history = _run_varying_joint_history(
                observations,
                dropout_mask,
                full_joint_forwards[noise],
                reduced_joint_forwards[noise],
                definition,
                tracking_config,
                n_averages,
                retained_columns,
            )
            calibration_probabilities[regime].append(
                stiffness_loss_probabilities(history, loss_threshold, 3)
            )
            calibration_dropout_rows.append(
                {
                    "regime": regime,
                    "replicate": replicate,
                    "realized_dropout_fraction": np.mean(dropout_mask),
                    "optimizer_failure_fraction": 1.0
                    - np.mean(history.optimizer_success),
                }
            )
    thresholds, calibration = calibrate_alarm_thresholds(
        calibration_probabilities,
        alarm_config["candidate_probability_thresholds"],
        persistence,
        float(alarm_config["maximum_calibration_false_alarm_fraction"]),
    )
    calibration = calibration.rename(columns={"method": "regime"}).merge(
        pd.DataFrame(calibration_dropout_rows).groupby("regime", as_index=False).agg(
            realized_dropout_fraction=("realized_dropout_fraction", "mean"),
            optimizer_failure_fraction=("optimizer_failure_fraction", "mean"),
        ),
        on="regime",
        how="left",
    )
    calibration_seconds = time.perf_counter() - calibration_started

    evaluation_started = time.perf_counter()
    deterioration_config = robustness_config["deterioration"]
    seconds_per_day = 86400.0
    truth_by_severity = {}
    for severity in severity_levels:
        loss = smooth_gradual_loss(
            time_days * seconds_per_day,
            onset_s=float(deterioration_config["onset_day"]) * seconds_per_day,
            completion_s=(
                float(deterioration_config["completion_day"]) * seconds_per_day
            ),
            final_loss_fraction=severity,
        )
        truth = np.ones((n_windows, 3), dtype=np.float64)
        truth[:, 0] = 1.0 - loss
        truth_by_severity[severity] = truth

    evaluation_config = robustness_config["evaluation"]
    evaluation_replicates = int(evaluation_config["replicates"])
    evaluation_seed = int(evaluation_config["random_seed"])
    run_rows: list[dict[str, object]] = []
    representative = pd.DataFrame()
    coded_cells = list(itertools.product((-1, 1), repeat=len(FACTOR_NAMES)))
    for replicate in range(evaluation_replicates):
        for codes in coded_cells:
            noise = noise_levels[(codes[0] + 1) // 2]
            severity = severity_levels[(codes[1] + 1) // 2]
            n_averages = data_levels[(codes[2] + 1) // 2]
            dropout_probability = dropout_levels[(codes[3] + 1) // 2]
            mismatch = mismatch_levels[(codes[4] + 1) // 2]
            observations = generate_windows(
                truth_forwards[(noise, mismatch)],
                truth_by_severity[severity],
                n_averages,
                np.random.default_rng(evaluation_seed + replicate),
            )
            dropout_mask = _dropout_mask(
                n_windows,
                dropout_probability,
                np.random.default_rng(evaluation_seed + 1000 + replicate),
            )
            history = _run_varying_joint_history(
                observations,
                dropout_mask,
                full_joint_forwards[noise],
                reduced_joint_forwards[noise],
                definition,
                tracking_config,
                n_averages,
                retained_columns,
            )
            regime = _regime_name(noise, n_averages, dropout_probability)
            row = {
                "replicate": replicate,
                "measurement_noise_fraction": noise,
                "final_loss_fraction": severity,
                "independent_periodogram_averages": n_averages,
                "top_sensor_dropout_probability": dropout_probability,
                "model_mismatch": mismatch,
                "alarm_probability_threshold": thresholds[regime],
                "realized_dropout_fraction": float(np.mean(dropout_mask)),
                **{
                    f"{name}_code": code
                    for name, code in zip(FACTOR_NAMES, codes, strict=True)
                },
                **_history_metrics(
                    history,
                    truth_by_severity[severity],
                    nuisance_truths[mismatch],
                    thresholds[regime],
                    loss_threshold,
                    persistence,
                ),
            }
            run_rows.append(row)
            if replicate == 0 and codes == (1, -1, -1, 1, 1):
                representative = _representative_rows(
                    history,
                    truth_by_severity[severity],
                    time_days,
                    dropout_mask,
                    loss_threshold,
                )
    runs = pd.DataFrame(run_rows)
    evaluation_seconds = time.perf_counter() - evaluation_started

    holdout_started = time.perf_counter()
    holdout_config = robustness_config["healthy_holdout"]
    holdout_replicates = int(holdout_config["replicates"])
    holdout_seed = int(holdout_config["random_seed"])
    healthy_rows: list[dict[str, object]] = []
    for replicate in range(holdout_replicates):
        for noise, n_averages, dropout_probability, mismatch in itertools.product(
            noise_levels,
            data_levels,
            dropout_levels,
            mismatch_levels,
        ):
            observations = generate_windows(
                truth_forwards[(noise, mismatch)],
                healthy_truth,
                n_averages,
                np.random.default_rng(holdout_seed + replicate),
            )
            dropout_mask = _dropout_mask(
                n_windows,
                dropout_probability,
                np.random.default_rng(holdout_seed + 1000 + replicate),
            )
            history = _run_varying_joint_history(
                observations,
                dropout_mask,
                full_joint_forwards[noise],
                reduced_joint_forwards[noise],
                definition,
                tracking_config,
                n_averages,
                retained_columns,
            )
            regime = _regime_name(noise, n_averages, dropout_probability)
            healthy_rows.append(
                {
                    "replicate": replicate,
                    "measurement_noise_fraction": noise,
                    "independent_periodogram_averages": n_averages,
                    "top_sensor_dropout_probability": dropout_probability,
                    "model_mismatch": mismatch,
                    "alarm_probability_threshold": thresholds[regime],
                    "realized_dropout_fraction": float(np.mean(dropout_mask)),
                    **_healthy_metrics(
                        history,
                        nuisance_truths[mismatch],
                        thresholds[regime],
                        loss_threshold,
                        persistence,
                    ),
                }
            )
    healthy = pd.DataFrame(healthy_rows)
    holdout_seconds = time.perf_counter() - holdout_started

    effects = _factorial_effect_table(runs)
    factor_columns = [
        "measurement_noise_fraction",
        "final_loss_fraction",
        "independent_periodogram_averages",
        "top_sensor_dropout_probability",
        "model_mismatch",
    ]
    cell_metrics = [
        *RESPONSE_NAMES,
        "detection_delay_windows",
        "nuisance_final_rmse",
        "optimizer_failure_fraction",
        "realized_dropout_fraction",
    ]
    cells = runs.groupby(factor_columns, as_index=False)[cell_metrics].mean()

    effects_index = effects.set_index(["response", "term"])
    hypotheses = robustness_config["hypotheses"]
    challenging = runs[
        (runs["noise_code"] == 1)
        & (runs["severity_code"] == -1)
        & (runs["data_support_code"] == -1)
        & (runs["dropout_code"] == 1)
        & (runs["mismatch_code"] == 1)
    ]
    observed_checks = {
        "high_noise_rmse_increase": float(
            effects_index.loc[
                ("damage_zone_rmse", "noise"), "high_minus_low_effect"
            ]
        ),
        "high_data_rmse_effect": float(
            effects_index.loc[
                ("damage_zone_rmse", "data_support"), "high_minus_low_effect"
            ]
        ),
        "high_severity_detection_improvement": float(
            effects_index.loc[
                ("detected", "severity"), "high_minus_low_effect"
            ]
        ),
        "dropout_detection_reduction": float(
            -effects_index.loc[
                ("detected", "dropout"), "high_minus_low_effect"
            ]
        ),
        "mismatch_rmse_increase": float(
            effects_index.loc[
                ("damage_zone_rmse", "mismatch"), "high_minus_low_effect"
            ]
        ),
        "challenging_corner_detection_fraction": float(
            challenging["detected"].mean()
        ),
        "damaged_interval_coverage": float(
            runs["damaged_interval_coverage"].mean()
        ),
        "healthy_holdout_false_alarm_fraction": float(
            healthy["healthy_false_alarm_fraction"].mean()
        ),
    }
    scientific_checks = [
        (
            "high_noise_rmse_increase",
            observed_checks["high_noise_rmse_increase"],
            ">=",
            float(hypotheses["minimum_high_noise_rmse_increase"]),
        ),
        (
            "high_data_rmse_effect",
            observed_checks["high_data_rmse_effect"],
            "<=",
            float(hypotheses["maximum_high_data_rmse_effect"]),
        ),
        (
            "high_severity_detection_improvement",
            observed_checks["high_severity_detection_improvement"],
            ">=",
            float(
                hypotheses["minimum_high_severity_detection_improvement"]
            ),
        ),
        (
            "dropout_detection_reduction",
            observed_checks["dropout_detection_reduction"],
            "<=",
            float(hypotheses["maximum_dropout_detection_reduction"]),
        ),
        (
            "mismatch_rmse_increase",
            observed_checks["mismatch_rmse_increase"],
            "<=",
            float(hypotheses["maximum_mismatch_rmse_increase"]),
        ),
        (
            "challenging_corner_detection_fraction",
            observed_checks["challenging_corner_detection_fraction"],
            ">=",
            float(
                hypotheses["minimum_challenging_corner_detection_fraction"]
            ),
        ),
        (
            "damaged_interval_coverage",
            observed_checks["damaged_interval_coverage"],
            ">=",
            float(hypotheses["minimum_damaged_interval_coverage"]),
        ),
        (
            "healthy_holdout_false_alarm_fraction",
            observed_checks["healthy_holdout_false_alarm_fraction"],
            "<=",
            float(hypotheses["maximum_healthy_holdout_false_alarm_fraction"]),
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
    verification = robustness_config["verification"]
    verification_checks = [
        (
            "factorial_cell_count",
            float(len(cells)),
            "==",
            float(verification["expected_factorial_cells"]),
        ),
        (
            "evaluation_history_count",
            float(len(runs)),
            "==",
            float(verification["expected_evaluation_histories"]),
        ),
        (
            "optimizer_failure_fraction",
            float(
                pd.concat(
                    [
                        runs["optimizer_failure_fraction"],
                        healthy["optimizer_failure_fraction"],
                        calibration["optimizer_failure_fraction"],
                    ],
                    ignore_index=True,
                ).mean()
            ),
            "<=",
            float(verification["maximum_optimizer_failure_fraction"]),
        ),
        (
            "all_calibration_regimes_selected",
            float(
                calibration.loc[
                    calibration["selected"],
                    "passes_calibration",
                ].mean()
            ),
            "==",
            1.0,
        ),
        (
            "finite_factorial_responses",
            float(np.all(np.isfinite(runs[list(RESPONSE_NAMES)].astype(float)))),
            "==",
            1.0,
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
                    else observed <= criterion
                ),
            }
        )
    checks = pd.DataFrame(check_rows)

    total_seconds = time.perf_counter() - started
    runtime = pd.DataFrame(
        [
            {"phase": "calibration", "seconds": calibration_seconds},
            {"phase": "factorial_evaluation", "seconds": evaluation_seconds},
            {"phase": "healthy_holdout", "seconds": holdout_seconds},
            {"phase": "total", "seconds": total_seconds},
        ]
    )

    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RUNS_PATH, index=False, float_format="%.12g")
    cells.to_csv(CELLS_PATH, index=False, float_format="%.12g")
    effects.to_csv(EFFECTS_PATH, index=False, float_format="%.12g")
    healthy.to_csv(HEALTHY_PATH, index=False, float_format="%.12g")
    calibration.to_csv(CALIBRATION_PATH, index=False, float_format="%.12g")
    representative.to_csv(TRAJECTORY_PATH, index=False, float_format="%.12g")
    checks.to_csv(CHECKS_PATH, index=False, float_format="%.12g")
    runtime.to_csv(RUNTIME_PATH, index=False, float_format="%.12g")
    _create_figure(runs, effects, healthy)

    main_effects = effects[
        (effects["response"] == "damage_zone_rmse")
        & (effects["term_order"] == 1)
    ]
    print("STAGE 11 - SYSTEMATIC ROBUSTNESS FACTORIAL")
    print(
        f"Design: {len(cells)} cells x {evaluation_replicates} paired replicates "
        f"= {len(runs)} histories"
    )
    print("Adjusted high-minus-low main effects on damage-zone RMSE:")
    for row in main_effects.itertuples(index=False):
        print(f"  {row.term:<13}: {row.high_minus_low_effect:+.5f}")
    print(
        "Detection / localisation / healthy false alarm: "
        f"{runs['detected'].mean():.3f} / {runs['localized'].mean():.3f} / "
        f"{healthy['healthy_false_alarm_fraction'].mean():.3f}"
    )
    hypothesis_checks = checks[checks["category"] == "hypothesis"]
    print(
        "Predeclared hypotheses supported: "
        f"{int(hypothesis_checks['supported'].sum())}/{len(hypothesis_checks)}"
    )
    print(f"Factorial effects: {EFFECTS_PATH}")
    print(f"Checks: {CHECKS_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    failed_verification = checks[
        (checks["category"] == "verification") & ~checks["supported"]
    ]
    if not failed_verification.empty:
        failed_names = ", ".join(failed_verification["check"])
        raise RuntimeError(f"Stage 11 verification failed: {failed_names}")
    print("RESULT: PASS (experiment complete; unsupported hypotheses retained)")


if __name__ == "__main__":
    main()
