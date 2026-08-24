"""Run the Stage 12 Monte Carlo confirmation and power study."""

from __future__ import annotations

import itertools
import os
import time
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
from run_joint_discrepancy_inference_study import (  # noqa: E402
    build_joint_method_definition,
    make_joint_forward_model,
    nuisance_truth_vector,
)
from run_model_discrepancy_study import (  # noqa: E402
    generate_windows,
    make_discrepant_truth_forward,
    parse_discrepancy,
)
from run_robustness_factorial_study import (  # noqa: E402
    _dropout_mask,
    _healthy_metrics,
    _history_metrics,
    _load_yaml,
    _regime_name,
    _run_varying_joint_history,
)
from run_sequential_tracking_study import (  # noqa: E402
    build_spectral_system,
    make_forward_model,
)

from probdt.deterioration import smooth_gradual_loss  # noqa: E402
from probdt.diagnostics import (  # noqa: E402
    MonteCarloMeanInterval,
    monte_carlo_mean_interval,
)
from probdt.monitoring import SensorLayout  # noqa: E402

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
TRACKING_CONFIG_PATH = PROJECT_ROOT / "config" / "sequential_tracking.yaml"
JOINT_CONFIG_PATH = PROJECT_ROOT / "config" / "joint_discrepancy_inference.yaml"
ROBUSTNESS_CONFIG_PATH = PROJECT_ROOT / "config" / "robustness_factorial.yaml"
CONFIRMATION_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "monte_carlo_confirmation.yaml"
)
STAGE11_CALIBRATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage11_alarm_calibration.csv"
)
STAGE11_EFFECTS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage11_factorial_effects.csv"
)

RUNS_PATH = PROJECT_ROOT / "results" / "metrics" / "stage12_confirmation_runs.csv"
CELL_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage12_cell_summary.csv"
)
PAIRED_SAMPLES_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage12_paired_differences.csv"
)
PAIRED_EFFECTS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage12_paired_noise_effects.csv"
)
CONVERGENCE_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage12_effect_convergence.csv"
)
HEALTHY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage12_healthy_holdout.csv"
)
CHECKS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage12_hypothesis_checks.csv"
)
RUNTIME_PATH = PROJECT_ROOT / "results" / "metrics" / "stage12_runtime.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage12_monte_carlo_confirmation.png"

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


def _summary_columns(
    prefix: str,
    summary: MonteCarloMeanInterval,
) -> dict[str, float | int]:
    """Flatten one Monte Carlo interval into stable table columns."""

    return {
        f"{prefix}_n": summary.n_replicates,
        f"{prefix}_mean": summary.mean,
        f"{prefix}_standard_deviation": summary.standard_deviation,
        f"{prefix}_standard_error": summary.standard_error,
        f"{prefix}_ci_lower": summary.confidence_interval_lower,
        f"{prefix}_ci_upper": summary.confidence_interval_upper,
    }


def _summarize_cells(
    runs: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    """Return replication uncertainty for each frozen context--noise cell."""

    rows: list[dict[str, object]] = []
    for (context, noise), selected in runs.groupby(
        ["context", "measurement_noise_fraction"],
        sort=False,
    ):
        first = selected.iloc[0]
        row: dict[str, object] = {
            "context": context,
            "measurement_noise_fraction": noise,
            "final_loss_fraction": first["final_loss_fraction"],
            "independent_periodogram_averages": first[
                "independent_periodogram_averages"
            ],
            "top_sensor_dropout_probability": first[
                "top_sensor_dropout_probability"
            ],
            "model_mismatch": first["model_mismatch"],
        }
        for response in RESPONSE_NAMES:
            summary = monte_carlo_mean_interval(
                selected[response].to_numpy(dtype=np.float64),
                confidence_level=confidence_level,
            )
            row.update(_summary_columns(response, summary))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_effect_tables(
    runs: pd.DataFrame,
    noise_levels: tuple[float, float],
    confidence_level: float,
    stage11_effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return paired samples, summaries, and cumulative convergence."""

    low_noise, high_noise = noise_levels
    sample_rows: list[dict[str, object]] = []
    for context, selected in runs.groupby("context", sort=False):
        low = selected[selected["measurement_noise_fraction"] == low_noise].set_index(
            "replicate"
        )
        high = selected[
            selected["measurement_noise_fraction"] == high_noise
        ].set_index("replicate")
        if not low.index.equals(high.index):
            raise RuntimeError(f"Unpaired Stage 12 samples for {context}.")
        for replicate in low.index:
            for response in RESPONSE_NAMES:
                sample_rows.append(
                    {
                        "context": context,
                        "replicate": replicate,
                        "response": response,
                        "low_noise_response": float(low.loc[replicate, response]),
                        "high_noise_response": float(high.loc[replicate, response]),
                        "high_minus_low_difference": (
                            float(high.loc[replicate, response])
                            - float(low.loc[replicate, response])
                        ),
                    }
                )
    samples = pd.DataFrame(sample_rows)

    stage11_noise = stage11_effects[
        (stage11_effects["term"] == "noise")
        & (stage11_effects["term_order"] == 1)
    ].set_index("response")["high_minus_low_effect"]
    summary_rows: list[dict[str, object]] = []
    for response in RESPONSE_NAMES:
        for context in (*tuple(runs["context"].unique()), "overall"):
            selected = samples[samples["response"] == response]
            if context == "overall":
                differences = (
                    selected.groupby("replicate")["high_minus_low_difference"]
                    .mean()
                    .to_numpy(dtype=np.float64)
                )
            else:
                differences = selected[
                    selected["context"] == context
                ]["high_minus_low_difference"].to_numpy(dtype=np.float64)
            summary = monte_carlo_mean_interval(
                differences,
                confidence_level=confidence_level,
            )
            source_effect = (
                float(stage11_noise.loc[response])
                if response in stage11_noise.index
                else np.nan
            )
            summary_rows.append(
                {
                    "context": context,
                    "response": response,
                    "effect_definition": "high_noise_minus_low_noise",
                    **_summary_columns("effect", summary),
                    "stage11_main_effect": source_effect,
                    "stage12_minus_stage11_effect": summary.mean - source_effect,
                }
            )
    effects = pd.DataFrame(summary_rows)

    convergence_rows: list[dict[str, object]] = []
    for response in RESPONSE_NAMES:
        selected = samples[samples["response"] == response]
        replicate_effects = (
            selected.groupby("replicate")["high_minus_low_difference"]
            .mean()
            .sort_index()
            .to_numpy(dtype=np.float64)
        )
        for n_replicates in range(2, replicate_effects.size + 1):
            summary = monte_carlo_mean_interval(
                replicate_effects[:n_replicates],
                confidence_level=confidence_level,
            )
            convergence_rows.append(
                {
                    "response": response,
                    "n_replicates": n_replicates,
                    "cumulative_effect_mean": summary.mean,
                    "cumulative_effect_standard_error": summary.standard_error,
                    "cumulative_effect_ci_lower": (
                        summary.confidence_interval_lower
                    ),
                    "cumulative_effect_ci_upper": (
                        summary.confidence_interval_upper
                    ),
                }
            )
    return samples, effects, pd.DataFrame(convergence_rows)


def _create_figure(
    cell_summary: pd.DataFrame,
    effects: pd.DataFrame,
    convergence: pd.DataFrame,
    healthy: pd.DataFrame,
) -> None:
    """Plot replication intervals, paired effects, and convergence evidence."""

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), constrained_layout=True)
    cell_axis, effect_axis, convergence_axis, decision_axis = axes.ravel()
    contexts = list(cell_summary["context"].unique())
    context_labels = [name.replace("_", " ") for name in contexts]
    noise_levels = sorted(cell_summary["measurement_noise_fraction"].unique())
    positions = np.arange(len(contexts), dtype=np.float64)
    colors = ("#0072B2", "#D55E00")
    offsets = (-0.12, 0.12)
    for noise, color, offset in zip(
        noise_levels,
        colors,
        offsets,
        strict=True,
    ):
        selected = cell_summary[
            cell_summary["measurement_noise_fraction"] == noise
        ].set_index("context").loc[contexts]
        mean = selected["damage_zone_rmse_mean"].to_numpy(dtype=np.float64)
        lower = selected["damage_zone_rmse_ci_lower"].to_numpy(dtype=np.float64)
        upper = selected["damage_zone_rmse_ci_upper"].to_numpy(dtype=np.float64)
        cell_axis.errorbar(
            positions + offset,
            mean,
            yerr=np.vstack((mean - lower, upper - mean)),
            color=color,
            marker="o",
            capsize=4,
            linestyle="none",
            label=f"Noise {noise:.2f}",
        )
    cell_axis.set_xticks(positions, labels=context_labels, rotation=18, ha="right")
    cell_axis.set_ylabel("Lower-zone trajectory RMSE")
    cell_axis.set_title("Twenty-replicate cell estimates with 95% intervals")
    cell_axis.legend(fontsize=8)
    cell_axis.grid(axis="y", alpha=0.25)

    effect_responses = (
        "damage_zone_rmse",
        "pooled_rmse",
        "damaged_interval_width",
    )
    effect_labels = ("Lower-zone RMSE", "Pooled RMSE", "Interval width")
    overall = effects[
        (effects["context"] == "overall")
        & effects["response"].isin(effect_responses)
    ].set_index("response").loc[list(effect_responses)]
    effect_mean = overall["effect_mean"].to_numpy(dtype=np.float64)
    effect_lower = overall["effect_ci_lower"].to_numpy(dtype=np.float64)
    effect_upper = overall["effect_ci_upper"].to_numpy(dtype=np.float64)
    effect_positions = np.arange(len(effect_responses), dtype=np.float64)
    effect_axis.errorbar(
        effect_positions,
        effect_mean,
        yerr=np.vstack((effect_mean - effect_lower, effect_upper - effect_mean)),
        color="black",
        marker="o",
        capsize=5,
        linestyle="none",
        label="Stage 12 paired effect",
    )
    effect_axis.scatter(
        effect_positions,
        overall["stage11_main_effect"],
        color="#CC79A7",
        marker="D",
        label="Stage 11 screening effect",
        zorder=3,
    )
    effect_axis.axhline(0.0, color="0.35", linewidth=0.9)
    effect_axis.set_xticks(effect_positions, labels=effect_labels, rotation=12)
    effect_axis.set_ylabel("High-noise minus low-noise effect")
    effect_axis.set_title("Paired replication of noise effects")
    effect_axis.legend(fontsize=8)
    effect_axis.grid(axis="y", alpha=0.25)

    lower_convergence = convergence[
        convergence["response"] == "damage_zone_rmse"
    ]
    convergence_axis.plot(
        lower_convergence["n_replicates"],
        lower_convergence["cumulative_effect_mean"],
        color="#0072B2",
        linewidth=1.8,
        label="Cumulative Stage 12 effect",
    )
    convergence_axis.fill_between(
        lower_convergence["n_replicates"],
        lower_convergence["cumulative_effect_ci_lower"],
        lower_convergence["cumulative_effect_ci_upper"],
        color="#0072B2",
        alpha=0.18,
        label="95% interval",
    )
    stage11_effect = float(
        effects.loc[
            (effects["context"] == "overall")
            & (effects["response"] == "damage_zone_rmse"),
            "stage11_main_effect",
        ].iloc[0]
    )
    convergence_axis.axhline(
        stage11_effect,
        color="#CC79A7",
        linestyle="--",
        label="Stage 11 effect",
    )
    convergence_axis.axhline(0.0, color="black", linewidth=0.8)
    convergence_axis.set_xlabel("Paired replicates included")
    convergence_axis.set_ylabel("Cumulative lower-zone RMSE effect")
    convergence_axis.set_title("Monte Carlo convergence of the retained result")
    convergence_axis.legend(fontsize=8)
    convergence_axis.grid(alpha=0.25)

    decision_rows = []
    for noise in noise_levels:
        selected = cell_summary[
            cell_summary["measurement_noise_fraction"] == noise
        ]
        decision_rows.append(
            {
                "noise": noise,
                "detection": selected["detected_mean"].mean(),
                "coverage": selected["damaged_interval_coverage_mean"].mean(),
                "localization": selected["localized_mean"].mean(),
            }
        )
    decisions = pd.DataFrame(decision_rows)
    decision_positions = np.arange(len(noise_levels), dtype=np.float64)
    width = 0.24
    for offset, metric, color in (
        (-width, "detection", "#009E73"),
        (0.0, "coverage", "#56B4E9"),
        (width, "localization", "#E69F00"),
    ):
        decision_axis.bar(
            decision_positions + offset,
            decisions[metric],
            width=width,
            color=color,
            label=metric.capitalize(),
        )
    false_alarm = float(healthy["healthy_false_alarm_fraction"].mean())
    decision_axis.text(
        0.02,
        0.05,
        f"Independent healthy false-alarm fraction: {false_alarm:.3f}",
        transform=decision_axis.transAxes,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    decision_axis.set_xticks(
        decision_positions,
        labels=[f"Noise {noise:.2f}" for noise in noise_levels],
    )
    decision_axis.set_ylim(0.0, 1.05)
    decision_axis.set_ylabel("Fraction")
    decision_axis.set_title("Decision performance and interval calibration")
    decision_axis.legend(fontsize=8, loc="lower right")
    decision_axis.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Stage 12: Monte Carlo confirmation of Stage 11 robustness findings",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def main() -> None:
    """Execute the frozen paired confirmation and independent healthy holdout."""

    started = time.perf_counter()
    base_config = _load_yaml(BASE_CONFIG_PATH)
    monitoring_config = _load_yaml(MONITORING_CONFIG_PATH)
    tracking_config = _load_yaml(TRACKING_CONFIG_PATH)
    joint_config = _load_yaml(JOINT_CONFIG_PATH)
    robustness_config = _load_yaml(ROBUSTNESS_CONFIG_PATH)
    confirmation_config = _load_yaml(CONFIRMATION_CONFIG_PATH)

    study_config = confirmation_config["study"]
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
            confirmation_config["dropout"][
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
    clean_forward = make_forward_model(
        truth_system,
        frequencies_hz,
        force_distribution,
        full_layout,
        excitation_config,
        np.zeros(full_layout.n_sensors, dtype=np.float64),
    )
    clean_psd = clean_forward(np.ones(3, dtype=np.float64))

    method_name = str(study_config["inference_method"])
    method_specification = joint_config["methods"][method_name]
    if method_specification["model"] != "joint":
        raise ValueError("Stage 12 requires the commissioned joint method.")
    definition = build_joint_method_definition(
        method_specification,
        tracking_config,
    )

    noise_levels = tuple(
        float(value) for value in confirmation_config["noise_levels"]
    )
    if len(noise_levels) != 2 or noise_levels[0] >= noise_levels[1]:
        raise ValueError("Stage 12 requires ordered low and high noise levels.")
    contexts = {
        str(name): specification
        for name, specification in confirmation_config["contexts"].items()
    }
    if len(contexts) != 4:
        raise ValueError("Stage 12 requires exactly four frozen contexts.")
    severities = sorted(
        {
            float(specification["final_loss_fraction"])
            for specification in contexts.values()
        }
    )
    mismatch_levels = sorted(
        {str(specification["model_mismatch"]) for specification in contexts.values()}
    )
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

    seconds_per_day = 86400.0
    deterioration_config = confirmation_config["deterioration"]
    truth_by_severity: dict[float, np.ndarray] = {}
    for severity in severities:
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

    calibration = pd.read_csv(STAGE11_CALIBRATION_PATH)
    selected_calibration = calibration[calibration["selected"]]
    thresholds = dict(
        zip(
            selected_calibration["regime"],
            selected_calibration["candidate_probability_threshold"],
            strict=True,
        )
    )
    required_regimes = {
        _regime_name(
            noise,
            int(specification["independent_periodogram_averages"]),
            float(specification["top_sensor_dropout_probability"]),
        )
        for noise in noise_levels
        for specification in contexts.values()
    }
    if not required_regimes.issubset(thresholds):
        raise RuntimeError("Stage 11 calibration lacks a Stage 12 regime.")

    loss_threshold = float(alarm_config["meaningful_loss_fraction"])
    persistence = int(alarm_config["persistence_windows"])
    evaluation_config = confirmation_config["evaluation"]
    n_replicates = int(evaluation_config["paired_replicates"])
    evaluation_seed = int(evaluation_config["random_seed"])
    confidence_level = float(evaluation_config["confidence_level"])

    evaluation_started = time.perf_counter()
    run_rows: list[dict[str, object]] = []
    for replicate in range(n_replicates):
        for context_name, specification in contexts.items():
            severity = float(specification["final_loss_fraction"])
            n_averages = int(specification["independent_periodogram_averages"])
            dropout_probability = float(
                specification["top_sensor_dropout_probability"]
            )
            mismatch = str(specification["model_mismatch"])
            for noise in noise_levels:
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
                run_rows.append(
                    {
                        "replicate": replicate,
                        "context": context_name,
                        "measurement_noise_fraction": noise,
                        "final_loss_fraction": severity,
                        "independent_periodogram_averages": n_averages,
                        "top_sensor_dropout_probability": dropout_probability,
                        "model_mismatch": mismatch,
                        "alarm_probability_threshold": thresholds[regime],
                        "realized_dropout_fraction": float(np.mean(dropout_mask)),
                        **_history_metrics(
                            history,
                            truth_by_severity[severity],
                            nuisance_truths[mismatch],
                            thresholds[regime],
                            loss_threshold,
                            persistence,
                        ),
                    }
                )
    runs = pd.DataFrame(run_rows)
    evaluation_seconds = time.perf_counter() - evaluation_started

    stage11_effects = pd.read_csv(STAGE11_EFFECTS_PATH)
    cell_summary = _summarize_cells(runs, confidence_level)
    paired_samples, paired_effects, convergence = _paired_effect_tables(
        runs,
        noise_levels,
        confidence_level,
        stage11_effects,
    )

    holdout_started = time.perf_counter()
    healthy_config = confirmation_config["healthy_holdout"]
    healthy_replicates = int(
        healthy_config["replicates_per_regime_and_mismatch"]
    )
    healthy_seed = int(healthy_config["random_seed"])
    healthy_truth = np.ones((n_windows, 3), dtype=np.float64)
    observation_regimes = sorted(
        {
            (
                noise,
                int(specification["independent_periodogram_averages"]),
                float(specification["top_sensor_dropout_probability"]),
            )
            for noise in noise_levels
            for specification in contexts.values()
        }
    )
    healthy_rows: list[dict[str, object]] = []
    for replicate in range(healthy_replicates):
        for (noise, n_averages, dropout_probability), mismatch in itertools.product(
            observation_regimes,
            mismatch_levels,
        ):
            observations = generate_windows(
                truth_forwards[(noise, mismatch)],
                healthy_truth,
                n_averages,
                np.random.default_rng(healthy_seed + replicate),
            )
            dropout_mask = _dropout_mask(
                n_windows,
                dropout_probability,
                np.random.default_rng(healthy_seed + 1000 + replicate),
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

    overall_effects = paired_effects[
        paired_effects["context"] == "overall"
    ].set_index("response")
    lower_effect = float(
        overall_effects.loc["damage_zone_rmse", "effect_mean"]
    )
    stage11_lower_effect = float(
        overall_effects.loc["damage_zone_rmse", "stage11_main_effect"]
    )
    hypotheses = confirmation_config["hypotheses"]
    scientific_checks = [
        (
            "lower_zone_noise_effect_direction_reproduced",
            lower_effect,
            "<=",
            float(hypotheses["maximum_lower_zone_noise_effect"]),
        ),
        (
            "lower_zone_effect_agrees_with_stage11_margin",
            abs(lower_effect - stage11_lower_effect),
            "<=",
            float(
                hypotheses["maximum_absolute_stage11_effect_difference"]
            ),
        ),
        (
            "pooled_rmse_noise_increase",
            float(overall_effects.loc["pooled_rmse", "effect_mean"]),
            ">=",
            float(hypotheses["minimum_pooled_rmse_noise_effect"]),
        ),
        (
            "interval_width_noise_increase",
            float(
                overall_effects.loc["damaged_interval_width", "effect_mean"]
            ),
            ">=",
            float(hypotheses["minimum_interval_width_noise_effect"]),
        ),
        (
            "detection_noise_effect_nonpositive",
            float(overall_effects.loc["detected", "effect_mean"]),
            "<=",
            float(hypotheses["maximum_detection_noise_effect"]),
        ),
        (
            "overall_detection_fraction",
            float(runs["detected"].mean()),
            ">=",
            float(hypotheses["minimum_overall_detection_fraction"]),
        ),
        (
            "damaged_interval_coverage",
            float(runs["damaged_interval_coverage"].mean()),
            ">=",
            float(hypotheses["minimum_damaged_interval_coverage"]),
        ),
        (
            "healthy_false_alarm_fraction",
            float(healthy["healthy_false_alarm_fraction"].mean()),
            "<=",
            float(hypotheses["maximum_healthy_false_alarm_fraction"]),
        ),
        (
            "lower_zone_effect_monte_carlo_standard_error",
            float(
                overall_effects.loc[
                    "damage_zone_rmse",
                    "effect_standard_error",
                ]
            ),
            "<=",
            float(
                hypotheses["maximum_lower_zone_effect_standard_error"]
            ),
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

    verification = confirmation_config["verification"]
    optimizer_failure = float(
        pd.concat(
            [
                runs["optimizer_failure_fraction"],
                healthy["optimizer_failure_fraction"],
            ],
            ignore_index=True,
        ).mean()
    )
    verification_checks = [
        (
            "cell_count",
            float(len(cell_summary)),
            float(verification["expected_cells"]),
        ),
        (
            "damaged_history_count",
            float(len(runs)),
            float(verification["expected_damaged_histories"]),
        ),
        (
            "context_noise_pair_count",
            float(
                len(
                    paired_samples[
                        paired_samples["response"] == RESPONSE_NAMES[0]
                    ]
                )
            ),
            float(verification["expected_context_noise_pairs"]),
        ),
        (
            "healthy_history_count",
            float(len(healthy)),
            float(verification["expected_healthy_histories"]),
        ),
        (
            "optimizer_failure_fraction",
            optimizer_failure,
            float(verification["maximum_optimizer_failure_fraction"]),
        ),
        (
            "finite_responses",
            float(np.all(np.isfinite(runs[list(RESPONSE_NAMES)].astype(float)))),
            1.0,
        ),
    ]
    for name, observed, criterion in verification_checks:
        operator = "<=" if name == "optimizer_failure_fraction" else "=="
        check_rows.append(
            {
                "category": "verification",
                "check": name,
                "observed": observed,
                "operator": operator,
                "criterion": criterion,
                "supported": (
                    observed <= criterion
                    if operator == "<="
                    else np.isclose(observed, criterion)
                ),
            }
        )
    checks = pd.DataFrame(check_rows)

    total_seconds = time.perf_counter() - started
    runtime = pd.DataFrame(
        [
            {"phase": "damaged_confirmation", "seconds": evaluation_seconds},
            {"phase": "healthy_holdout", "seconds": holdout_seconds},
            {"phase": "total", "seconds": total_seconds},
        ]
    )
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RUNS_PATH, index=False, float_format="%.12g")
    cell_summary.to_csv(CELL_SUMMARY_PATH, index=False, float_format="%.12g")
    paired_samples.to_csv(PAIRED_SAMPLES_PATH, index=False, float_format="%.12g")
    paired_effects.to_csv(PAIRED_EFFECTS_PATH, index=False, float_format="%.12g")
    convergence.to_csv(CONVERGENCE_PATH, index=False, float_format="%.12g")
    healthy.to_csv(HEALTHY_PATH, index=False, float_format="%.12g")
    checks.to_csv(CHECKS_PATH, index=False, float_format="%.12g")
    runtime.to_csv(RUNTIME_PATH, index=False, float_format="%.12g")
    _create_figure(cell_summary, paired_effects, convergence, healthy)

    print("STAGE 12 - MONTE CARLO CONFIRMATION")
    print(
        f"Design: {len(cell_summary)} cells x {n_replicates} paired replicates "
        f"= {len(runs)} damaged histories"
    )
    print("Overall high-noise minus low-noise effects (95% intervals):")
    for response in (
        "damage_zone_rmse",
        "pooled_rmse",
        "damaged_interval_width",
        "detected",
    ):
        row = overall_effects.loc[response]
        print(
            f"  {response:<24}: {row.effect_mean:+.5f} "
            f"[{row.effect_ci_lower:+.5f}, {row.effect_ci_upper:+.5f}]"
        )
    print(
        "Detection / coverage / healthy false alarm: "
        f"{runs['detected'].mean():.3f} / "
        f"{runs['damaged_interval_coverage'].mean():.3f} / "
        f"{healthy['healthy_false_alarm_fraction'].mean():.3f}"
    )
    hypothesis_checks = checks[checks["category"] == "hypothesis"]
    print(
        "Predeclared confirmations supported: "
        f"{int(hypothesis_checks['supported'].sum())}/{len(hypothesis_checks)}"
    )
    print(f"Paired effects: {PAIRED_EFFECTS_PATH}")
    print(f"Checks: {CHECKS_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    failed_verification = checks[
        (checks["category"] == "verification") & ~checks["supported"]
    ]
    if not failed_verification.empty:
        failed_names = ", ".join(failed_verification["check"])
        raise RuntimeError(f"Stage 12 verification failed: {failed_names}")
    print("RESULT: PASS (confirmation complete; adverse findings retained)")


if __name__ == "__main__":
    main()
