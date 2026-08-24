"""Run Stage 22 hybrid feature-information remedy on severe OpenFAST holdouts."""

from __future__ import annotations

import hashlib
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

from probdt.openfast import (  # noqa: E402
    cosine_similarity,
    fit_shrunk_linear_damage_templates,
    monotonic_damage_transition,
    run_discrete_damage_filter,
    summarize_damage_posterior,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "phase2_openfast_information_remedy.yaml"
CALIBRATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage22_template_calibration.csv"
)
STATE_LIBRARY_PATH = PROJECT_ROOT / "results" / "metrics" / "stage22_state_library.csv"
POSTERIOR_PATH = PROJECT_ROOT / "results" / "metrics" / "stage22_posterior_history.csv"
STATE_PROBABILITY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage22_state_probabilities.csv"
)
SUMMARY_PATH = PROJECT_ROOT / "results" / "metrics" / "stage22_remedy_summary.csv"
DETECTION_PATH = PROJECT_ROOT / "results" / "metrics" / "stage22_detection_events.csv"
HYPOTHESIS_PATH = PROJECT_ROOT / "results" / "metrics" / "stage22_hypothesis_checks.csv"
VALIDATION_PATH = PROJECT_ROOT / "results" / "metrics" / "stage22_validation.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage22_openfast_information_remedy.png"


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(
    category: str,
    name: str,
    observed: object,
    operator: str,
    criterion: object,
) -> dict[str, object]:
    if operator == "==":
        supported = observed == criterion
    elif operator == "<=":
        supported = float(observed) <= float(criterion)
    elif operator == ">=":
        supported = float(observed) >= float(criterion)
    else:
        raise ValueError(f"Unsupported validation operator: {operator}")
    return {
        "category": category,
        "check": name,
        "observed": observed,
        "operator": operator,
        "criterion": criterion,
        "supported": bool(supported),
    }


def _case_id(zone: str, loss: float) -> str:
    if np.isclose(loss, 0.0):
        return "control_no_loss"
    return f"{zone}_loss_{round(100.0 * loss)}pct"


def _feature_vector(
    rows: pd.DataFrame,
    column: str,
    channels: tuple[str, ...],
) -> np.ndarray:
    selected = rows.copy()
    order = {channel: index for index, channel in enumerate(channels)}
    selected["_channel_order"] = selected["channel"].map(order)
    selected = selected.sort_values(["frequency_hz", "_channel_order"])
    values = selected[column].to_numpy(dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("A Stage 22 feature vector is empty or nonfinite.")
    return values


def _persistent_alarm(
    probabilities: np.ndarray,
    threshold: float,
    windows: int,
) -> np.ndarray:
    exceedance = probabilities >= threshold
    alarms = np.zeros_like(exceedance)
    for index in range(windows - 1, exceedance.size):
        alarms[index] = bool(np.all(exceedance[index - windows + 1 : index + 1]))
    return alarms


def _create_figure(
    calibration: pd.DataFrame,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    threshold: float,
    zone_order: tuple[str, ...],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), constrained_layout=True)
    methods = ("physics_template_control", "hybrid_local_linear")
    regimes = ("commissioning_reference", "paired_excitation_oracle")
    method_colors = {
        "physics_template_control": "#999999",
        "hybrid_local_linear": "#009E73",
    }
    regime_colors = {
        "commissioning_reference": "#CC79A7",
        "paired_excitation_oracle": "#56B4E9",
    }

    calibration_axis = axes[0, 0]
    positions = np.arange(len(regimes), dtype=np.float64)
    width = 0.34
    for offset, method in zip((-width / 2.0, width / 2.0), methods, strict=True):
        values = []
        for regime in regimes:
            selected = calibration[
                (calibration["regime"] == regime) & (calibration["method"] == method)
            ]
            values.append(float(selected["development_rmse"].mean()))
        calibration_axis.bar(
            positions + offset,
            values,
            width=width,
            color=method_colors[method],
            label=method.replace("_", " "),
        )
    calibration_axis.set_xticks(positions, ["operational", "oracle"])
    calibration_axis.set_ylabel("Development log-feature RMSE")
    calibration_axis.set_title("A. Development-template fit")
    calibration_axis.legend(fontsize=8)

    probability_axis = axes[0, 1]
    for method in methods:
        for regime in regimes:
            selected = history[
                (history["method"] == method) & (history["regime"] == regime)
            ]
            grouped = selected.groupby("monitoring_window", as_index=False)[
                "damage_probability"
            ].mean()
            probability_axis.plot(
                grouped["monitoring_window"],
                grouped["damage_probability"],
                color=regime_colors[regime],
                linestyle="-" if method == "hybrid_local_linear" else ":",
                marker="o" if method == "hybrid_local_linear" else None,
                linewidth=1.7,
                label=(
                    f"{'hybrid' if method == 'hybrid_local_linear' else 'physics'} / "
                    + (
                        "operational"
                        if regime == "commissioning_reference"
                        else "oracle"
                    )
                ),
            )
    probability_axis.axhline(
        threshold,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label="alarm threshold",
    )
    probability_axis.axvline(5.5, color="#777777", linestyle=":", linewidth=1.0)
    probability_axis.set_ylim(-0.02, 1.02)
    probability_axis.set_xlabel("Severe-holdout monitoring window")
    probability_axis.set_ylabel("Mean damage probability")
    probability_axis.set_title("B. Severe-holdout evidence accumulation")
    probability_axis.legend(fontsize=7, ncol=2)

    detection_axis = axes[1, 0]
    summary_indexed = summary.set_index(["method", "regime"])
    bar_positions = np.arange(len(regimes), dtype=np.float64)
    for offset, method in zip((-width / 2.0, width / 2.0), methods, strict=True):
        values = [
            float(summary_indexed.loc[(method, regime), "severe_detection_fraction"])
            for regime in regimes
        ]
        detection_axis.bar(
            bar_positions + offset,
            values,
            width=width,
            color=method_colors[method],
            label=method.replace("_", " "),
        )
    detection_axis.set_xticks(bar_positions, ["operational", "oracle"])
    detection_axis.set_ylim(0.0, 1.0)
    detection_axis.set_ylabel("Detected severe-window fraction")
    detection_axis.set_title("C. Holdout detection recovery")
    detection_axis.legend(fontsize=8)

    zone_axis = axes[1, 1]
    final = history[
        (history["method"] == "hybrid_local_linear")
        & (history["monitoring_window"] == history["monitoring_window"].max())
    ]
    zone_positions = np.arange(len(zone_order), dtype=np.float64)
    for offset, regime in zip((-width / 2.0, width / 2.0), regimes, strict=True):
        selected = final[final["regime"] == regime].set_index("trajectory_zone")
        values = [
            float(selected.loc[zone, "true_zone_probability"]) for zone in zone_order
        ]
        zone_axis.bar(
            zone_positions + offset,
            values,
            width=width,
            color=regime_colors[regime],
            label=("operational" if regime == "commissioning_reference" else "oracle"),
        )
    zone_axis.axhline(2.0 / 3.0, color="#333333", linestyle="--", linewidth=1.2)
    zone_axis.set_xticks(
        zone_positions,
        [zone.replace("_monopile", "") for zone in zone_order],
    )
    zone_axis.set_ylim(0.0, 1.0)
    zone_axis.set_xlabel("True severe-holdout zone")
    zone_axis.set_ylabel("Final conditional true-zone probability")
    zone_axis.set_title("D. Hybrid localisation evidence")
    zone_axis.legend(fontsize=8)

    hybrid_operational = summary_indexed.loc[
        ("hybrid_local_linear", "commissioning_reference")
    ]
    hybrid_oracle = summary_indexed.loc[
        ("hybrid_local_linear", "paired_excitation_oracle")
    ]
    figure.suptitle(
        "Stage 22: physics-shrunk OpenFAST information remedy\n"
        "hybrid severe detection="
        f"{hybrid_operational['severe_detection_fraction']:.2f} "
        f"(operational), {hybrid_oracle['severe_detection_fraction']:.2f} (oracle)",
        fontsize=15,
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> int:
    configuration = dict(_load_yaml(CONFIG_PATH)["stage22"])
    upstream = dict(configuration["upstream"])
    split = dict(configuration["split"])
    methods_config = dict(configuration["methods"])
    state_config = dict(configuration["state_library"])
    likelihood_config = dict(configuration["likelihood"])
    transition_config = dict(configuration["transition_model"])
    replay = dict(configuration["validation_replay"])
    alarm_config = dict(configuration["alarm"])
    expected = dict(configuration["expected_dimensions"])
    hypothesis_config = dict(configuration["hypotheses"])
    regimes = tuple(str(value) for value in configuration["observation_regimes"])

    pinned_paths = {
        "monitoring_windows": (
            upstream["monitoring_windows_path"],
            upstream["monitoring_windows_sha256"],
        ),
        "stage21_state_library": (
            upstream["stage21_state_library_path"],
            upstream["stage21_state_library_sha256"],
        ),
        "stage21_summary": (
            upstream["stage21_summary_path"],
            upstream["stage21_summary_sha256"],
        ),
        "stage21_validation": (
            upstream["stage21_validation_path"],
            upstream["stage21_validation_sha256"],
        ),
        "transfer_spectra": (
            upstream["transfer_spectra_path"],
            upstream["transfer_spectra_sha256"],
        ),
    }
    hash_results = {
        name: _sha256(PROJECT_ROOT / str(path)) == str(expected_hash)
        for name, (path, expected_hash) in pinned_paths.items()
    }
    if not all(hash_results.values()):
        raise RuntimeError("A pinned Stage 22 upstream artifact changed unexpectedly.")

    windows = pd.read_csv(PROJECT_ROOT / str(upstream["monitoring_windows_path"]))
    stage21_library = pd.read_csv(
        PROJECT_ROOT / str(upstream["stage21_state_library_path"])
    )
    stage21_validation = pd.read_csv(
        PROJECT_ROOT / str(upstream["stage21_validation_path"])
    )
    if not bool(stage21_validation["supported"].all()):
        raise RuntimeError("Stage 21 core validation is incomplete.")

    channels = tuple(str(value) for value in windows["channel"].drop_duplicates())
    feature_keys = windows[["frequency_hz", "channel"]].drop_duplicates(
        ignore_index=True
    )
    state_metadata = (
        stage21_library[stage21_library["regime"] == "commissioning_reference"][
            [
                "state_index",
                "state_id",
                "state_zone",
                "state_loss_fraction",
            ]
        ]
        .drop_duplicates()
        .sort_values("state_index")
    )
    state_ids = tuple(str(value) for value in state_metadata["state_id"])
    state_zones = tuple(str(value) for value in state_metadata["state_zone"])
    state_losses = state_metadata["state_loss_fraction"].to_numpy(dtype=np.float64)
    physics_templates = []
    for state_id in state_ids:
        selected = stage21_library[
            (stage21_library["regime"] == "commissioning_reference")
            & (stage21_library["state_id"] == state_id)
        ]
        physics_templates.append(
            _feature_vector(
                selected,
                "predicted_differential_log_psd",
                channels,
            )
        )
    physics_templates_array = np.stack(physics_templates)

    zone_order = tuple(str(value) for value in state_config["zone_order"])
    development_losses = tuple(
        float(value) for value in split["development_loss_levels"]
    )
    validation_loss = float(split["severe_validation_loss_fraction"])
    regime_columns = {
        "commissioning_reference": "commissioning_reference_log_ratio",
        "paired_excitation_oracle": "paired_excitation_log_ratio",
    }
    method_names = ("physics_template_control", "hybrid_local_linear")
    templates_by_regime_method: dict[tuple[str, str], np.ndarray] = {}
    variances_by_regime_method: dict[tuple[str, str], np.ndarray] = {}
    calibration_rows: list[dict[str, object]] = []

    for regime in regimes:
        feature_rows = []
        feature_zones = []
        feature_losses = []
        for zone in zone_order:
            for loss in development_losses:
                case_id = _case_id(zone, loss)
                selected_case = windows[windows["case_id"] == case_id]
                for source_window in sorted(selected_case["source_window"].unique()):
                    selected = selected_case[
                        selected_case["source_window"] == source_window
                    ]
                    feature_rows.append(
                        _feature_vector(selected, regime_columns[regime], channels)
                    )
                    feature_zones.append(zone)
                    feature_losses.append(loss)
        development_features = np.stack(feature_rows)
        development_losses_array = np.asarray(feature_losses, dtype=np.float64)
        hybrid = fit_shrunk_linear_damage_templates(
            physics_templates_array,
            state_zones,
            state_losses,
            development_features,
            feature_zones,
            development_losses_array,
            empirical_blend_weight=float(
                dict(methods_config["hybrid_local_linear"])["empirical_blend_weight"]
            ),
        )
        control_residuals = []
        for feature, zone, loss in zip(
            development_features,
            feature_zones,
            development_losses_array,
            strict=True,
        ):
            state_index = next(
                index
                for index, (state_zone, state_loss) in enumerate(
                    zip(state_zones, state_losses, strict=True)
                )
                if state_zone == zone and np.isclose(state_loss, loss)
            )
            control_residuals.append(feature - physics_templates_array[state_index])
        residuals_by_method = {
            "physics_template_control": np.stack(control_residuals),
            "hybrid_local_linear": hybrid.development_residuals,
        }
        templates_by_method = {
            "physics_template_control": physics_templates_array,
            "hybrid_local_linear": hybrid.corrected_state_templates,
        }
        minimum_variance = float(
            likelihood_config[
                "minimum_operational_log_variance"
                if regime == "commissioning_reference"
                else "minimum_oracle_log_variance"
            ]
        )
        extrapolation_variance = (
            float(likelihood_config["severe_extrapolation_standard_deviation"]) ** 2
        )
        for method in method_names:
            residuals = residuals_by_method[method]
            variance = (
                np.maximum(
                    np.var(residuals, axis=0, ddof=1),
                    minimum_variance,
                )
                + extrapolation_variance
            )
            templates_by_regime_method[(regime, method)] = templates_by_method[method]
            variances_by_regime_method[(regime, method)] = variance
            for zone_index, zone in enumerate(hybrid.zone_names):
                zone_mask = np.asarray(feature_zones) == zone
                calibration_rows.append(
                    {
                        "regime": regime,
                        "method": method,
                        "zone": zone,
                        "development_case_count": len(development_losses),
                        "development_window_count": int(np.sum(zone_mask)),
                        "development_rmse": float(
                            np.sqrt(np.mean(residuals[zone_mask] ** 2))
                        ),
                        "physics_slope_norm": float(
                            np.linalg.norm(hybrid.physics_slopes[zone_index])
                        ),
                        "empirical_slope_norm": float(
                            np.linalg.norm(hybrid.empirical_slopes[zone_index])
                        ),
                        "blended_slope_norm": float(
                            np.linalg.norm(hybrid.blended_slopes[zone_index])
                        ),
                        "physics_empirical_slope_cosine_similarity": cosine_similarity(
                            hybrid.physics_slopes[zone_index],
                            hybrid.empirical_slopes[zone_index],
                        ),
                        "mean_feature_variance": float(np.mean(variance)),
                        "maximum_feature_variance": float(np.max(variance)),
                        "empirical_blend_weight": dict(
                            methods_config["hybrid_local_linear"]
                        )["empirical_blend_weight"],
                    }
                )
    calibration = pd.DataFrame(calibration_rows)

    transition = monotonic_damage_transition(
        state_zones,
        state_losses,
        healthy_persistence_probability=float(
            transition_config["healthy_persistence_probability"]
        ),
        damage_progression_probability=float(
            transition_config["damage_progression_probability"]
        ),
        zone_switch_probability=float(transition_config["zone_switch_probability"]),
        healthy_entry_severity_weights=transition_config[
            "healthy_entry_severity_weights"
        ],
    )
    initial = np.zeros(len(state_ids), dtype=np.float64)
    initial[0] = float(transition_config["initial_healthy_probability"])
    initial /= np.sum(initial)

    state_rows: list[dict[str, object]] = []
    feature_grid = feature_keys.copy()
    order = {channel: index for index, channel in enumerate(channels)}
    feature_grid["_channel_order"] = feature_grid["channel"].map(order)
    feature_grid = feature_grid.sort_values(["frequency_hz", "_channel_order"])
    for regime in regimes:
        for method in method_names:
            templates = templates_by_regime_method[(regime, method)]
            variances = variances_by_regime_method[(regime, method)]
            for state_index, state_id in enumerate(state_ids):
                for feature_index, feature in enumerate(
                    feature_grid.itertuples(index=False)
                ):
                    state_rows.append(
                        {
                            "regime": regime,
                            "method": method,
                            "state_index": state_index,
                            "state_id": state_id,
                            "state_zone": state_zones[state_index],
                            "state_loss_fraction": state_losses[state_index],
                            "channel": feature.channel,
                            "frequency_hz": feature.frequency_hz,
                            "differential_log_psd_template": templates[
                                state_index, feature_index
                            ],
                            "feature_variance": variances[feature_index],
                            "is_severe_validation_state": bool(
                                np.isclose(state_losses[state_index], validation_loss)
                            ),
                        }
                    )
    state_library = pd.DataFrame(state_rows)

    history_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    healthy_case = "control_no_loss"
    healthy_window_count = int(replay["healthy_windows"])
    severe_window_count = int(replay["severe_windows"])
    for regime in regimes:
        for method in method_names:
            templates = templates_by_regime_method[(regime, method)]
            variances = variances_by_regime_method[(regime, method)]
            for trajectory_zone in zone_order:
                severe_case = _case_id(trajectory_zone, validation_loss)
                features = []
                source_cases = []
                source_windows = []
                truth_losses = []
                for case_id, count, truth_loss in (
                    (healthy_case, healthy_window_count, 0.0),
                    (severe_case, severe_window_count, validation_loss),
                ):
                    selected_case = windows[windows["case_id"] == case_id]
                    for source_window in range(1, count + 1):
                        selected = selected_case[
                            selected_case["source_window"] == source_window
                        ]
                        features.append(
                            _feature_vector(
                                selected,
                                regime_columns[regime],
                                channels,
                            )
                        )
                        source_cases.append(case_id)
                        source_windows.append(source_window)
                        truth_losses.append(truth_loss)
                feature_matrix = np.stack(features)
                tracking = run_discrete_damage_filter(
                    feature_matrix,
                    templates,
                    variances,
                    transition,
                    initial,
                    degrees_of_freedom=float(likelihood_config["degrees_of_freedom"]),
                    likelihood_power=float(likelihood_config["likelihood_power"]),
                )
                summaries = [
                    summarize_damage_posterior(
                        posterior,
                        state_zones,
                        state_losses,
                        damaged_zone_order=zone_order,
                        meaningful_loss_fraction=float(
                            state_config["meaningful_loss_fraction"]
                        ),
                        credible_level=float(state_config["credible_level"]),
                    )
                    for posterior in tracking.posterior_state_probabilities
                ]
                damage_probabilities = np.asarray(
                    [summary.damage_probability for summary in summaries]
                )
                alarms = _persistent_alarm(
                    damage_probabilities,
                    float(alarm_config["probability_threshold"]),
                    int(alarm_config["persistence_windows"]),
                )
                onset = healthy_window_count
                pre_onset_false_alarm = bool(np.any(alarms[:onset]))
                candidates = np.flatnonzero(alarms & (np.arange(alarms.size) >= onset))
                detected = bool(candidates.size) and not pre_onset_false_alarm
                first_detection = int(candidates[0]) if detected else -1
                detection_rows.append(
                    {
                        "regime": regime,
                        "method": method,
                        "trajectory_zone": trajectory_zone,
                        "severe_onset_window": onset + 1,
                        "pre_onset_false_alarm": pre_onset_false_alarm,
                        "detected": detected,
                        "first_persistent_alarm_window": (
                            first_detection + 1 if detected else np.nan
                        ),
                        "detection_delay_windows": (
                            first_detection - onset if detected else np.nan
                        ),
                        "probability_threshold": alarm_config["probability_threshold"],
                        "persistence_windows": alarm_config["persistence_windows"],
                    }
                )
                for window_index, summary in enumerate(summaries):
                    true_loss = float(truth_losses[window_index])
                    true_state_id = _case_id(trajectory_zone, true_loss)
                    true_state_index = state_ids.index(true_state_id)
                    selected_zone = zone_order[
                        int(np.argmax(summary.conditional_zone_probabilities))
                    ]
                    true_log_likelihood_advantage = (
                        tracking.log_likelihoods[window_index, true_state_index]
                        - tracking.log_likelihoods[window_index, 0]
                        if true_loss > 0.0
                        else np.nan
                    )
                    row = {
                        "regime": regime,
                        "method": method,
                        "trajectory_zone": trajectory_zone,
                        "monitoring_window": window_index + 1,
                        "source_case_id": source_cases[window_index],
                        "source_window": source_windows[window_index],
                        "true_loss_fraction": true_loss,
                        "damage_probability": summary.damage_probability,
                        "expected_loss_fraction": summary.expected_loss_fraction,
                        "credible_interval_lower": summary.credible_interval_lower,
                        "credible_interval_upper": summary.credible_interval_upper,
                        "severity_interval_contains_truth": bool(
                            summary.credible_interval_lower - 1.0e-12
                            <= true_loss
                            <= summary.credible_interval_upper + 1.0e-12
                        ),
                        "selected_zone": selected_zone,
                        "true_zone_probability": (
                            summary.conditional_zone_probabilities[
                                zone_order.index(trajectory_zone)
                            ]
                            if true_loss > 0.0
                            else np.nan
                        ),
                        "map_state_id": state_ids[summary.map_state_index],
                        "map_loss_fraction": state_losses[summary.map_state_index],
                        "true_state_probability": (
                            tracking.posterior_state_probabilities[
                                window_index, true_state_index
                            ]
                        ),
                        "true_state_log_likelihood_advantage_over_healthy": (
                            true_log_likelihood_advantage
                        ),
                        "posterior_entropy": summary.posterior_entropy,
                        "persistent_alarm": bool(alarms[window_index]),
                        "valid_detection_alarm": bool(
                            alarms[window_index]
                            and window_index >= onset
                            and not pre_onset_false_alarm
                        ),
                    }
                    for zone_index, zone in enumerate(zone_order):
                        row[f"conditional_probability_{zone}"] = (
                            summary.conditional_zone_probabilities[zone_index]
                        )
                    history_rows.append(row)
                    for state_index, state_id in enumerate(state_ids):
                        probability_rows.append(
                            {
                                "regime": regime,
                                "method": method,
                                "trajectory_zone": trajectory_zone,
                                "monitoring_window": window_index + 1,
                                "state_index": state_index,
                                "state_id": state_id,
                                "predicted_probability": (
                                    tracking.predicted_state_probabilities[
                                        window_index, state_index
                                    ]
                                ),
                                "posterior_probability": (
                                    tracking.posterior_state_probabilities[
                                        window_index, state_index
                                    ]
                                ),
                                "log_likelihood": tracking.log_likelihoods[
                                    window_index, state_index
                                ],
                            }
                        )

    history = pd.DataFrame(history_rows)
    state_probabilities = pd.DataFrame(probability_rows)
    detections = pd.DataFrame(detection_rows)
    summary_rows: list[dict[str, object]] = []
    for (method, regime), selected in history.groupby(["method", "regime"], sort=False):
        healthy = selected[np.isclose(selected["true_loss_fraction"], 0.0)]
        severe = selected[np.isclose(selected["true_loss_fraction"], validation_loss)]
        final = severe[severe["monitoring_window"] == severe["monitoring_window"].max()]
        selected_detections = detections[
            (detections["method"] == method) & (detections["regime"] == regime)
        ]
        summary_rows.append(
            {
                "method": method,
                "regime": regime,
                "commissioning_false_alarm_fraction": float(
                    healthy["persistent_alarm"].mean()
                ),
                "severe_detection_fraction": float(
                    severe["valid_detection_alarm"].mean()
                ),
                "severe_alarm_fraction_including_false_alarm_carryover": float(
                    severe["persistent_alarm"].mean()
                ),
                "trajectory_detection_fraction": float(
                    selected_detections["detected"].mean()
                ),
                "mean_detection_delay_windows": float(
                    selected_detections["detection_delay_windows"].mean()
                ),
                "severe_severity_mae": float(
                    np.mean(
                        np.abs(
                            severe["expected_loss_fraction"]
                            - severe["true_loss_fraction"]
                        )
                    )
                ),
                "severe_severity_interval_coverage": float(
                    severe["severity_interval_contains_truth"].mean()
                ),
                "final_zone_selection_fraction": float(
                    (final["selected_zone"] == final["trajectory_zone"]).mean()
                ),
                "mean_final_true_zone_probability": float(
                    final["true_zone_probability"].mean()
                ),
                "mean_severe_true_state_probability": float(
                    severe["true_state_probability"].mean()
                ),
                "mean_true_state_log_likelihood_advantage_over_healthy": float(
                    severe["true_state_log_likelihood_advantage_over_healthy"].mean()
                ),
                "maximum_damage_probability": float(severe["damage_probability"].max()),
                "damage_probability_brier_score": float(
                    np.mean(
                        (
                            selected["damage_probability"]
                            - (selected["true_loss_fraction"] > 0.0).astype(float)
                        )
                        ** 2
                    )
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    indexed = summary.set_index(["method", "regime"])
    control_operational = indexed.loc[
        ("physics_template_control", "commissioning_reference")
    ]
    control_oracle = indexed.loc[
        ("physics_template_control", "paired_excitation_oracle")
    ]
    hybrid_operational = indexed.loc[("hybrid_local_linear", "commissioning_reference")]
    hybrid_oracle = indexed.loc[("hybrid_local_linear", "paired_excitation_oracle")]

    hypothesis_rows = [
        _check(
            "scientific_hypothesis",
            "H22_1_remedied_operational_false_alarms_are_bounded",
            hybrid_operational["commissioning_false_alarm_fraction"],
            "<=",
            hypothesis_config["maximum_remedied_operational_false_alarm_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H22_2_remedied_oracle_false_alarms_are_bounded",
            hybrid_oracle["commissioning_false_alarm_fraction"],
            "<=",
            hypothesis_config["maximum_remedied_oracle_false_alarm_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H22_3_remedied_operational_severe_windows_are_detected",
            hybrid_operational["severe_detection_fraction"],
            ">=",
            hypothesis_config["minimum_remedied_operational_severe_detection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H22_4_remedied_oracle_severe_windows_are_detected",
            hybrid_oracle["severe_detection_fraction"],
            ">=",
            hypothesis_config["minimum_remedied_oracle_severe_detection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H22_5_remedied_operational_trajectories_are_detected",
            hybrid_operational["trajectory_detection_fraction"],
            ">=",
            hypothesis_config[
                "minimum_remedied_operational_trajectory_detection_fraction"
            ],
        ),
        _check(
            "scientific_hypothesis",
            "H22_6_remedied_oracle_trajectories_are_detected",
            hybrid_oracle["trajectory_detection_fraction"],
            ">=",
            hypothesis_config["minimum_remedied_oracle_trajectory_detection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H22_7_remedied_operational_severity_error_is_bounded",
            hybrid_operational["severe_severity_mae"],
            "<=",
            hypothesis_config["maximum_remedied_operational_severity_mae"],
        ),
        _check(
            "scientific_hypothesis",
            "H22_8_remedied_oracle_severity_error_is_bounded",
            hybrid_oracle["severe_severity_mae"],
            "<=",
            hypothesis_config["maximum_remedied_oracle_severity_mae"],
        ),
        _check(
            "scientific_hypothesis",
            "H22_9_remedied_operational_final_zone_is_selected",
            hybrid_operational["final_zone_selection_fraction"],
            ">=",
            hypothesis_config[
                "minimum_remedied_operational_final_zone_selection_fraction"
            ],
        ),
        _check(
            "scientific_hypothesis",
            "H22_10_remedied_oracle_final_zone_is_selected",
            hybrid_oracle["final_zone_selection_fraction"],
            ">=",
            hypothesis_config["minimum_remedied_oracle_final_zone_selection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H22_11_remedied_operational_intervals_cover_severe_truth",
            hybrid_operational["severe_severity_interval_coverage"],
            ">=",
            hypothesis_config[
                "minimum_remedied_operational_severity_interval_coverage"
            ],
        ),
        _check(
            "scientific_hypothesis",
            "H22_12_remedied_oracle_intervals_cover_severe_truth",
            hybrid_oracle["severe_severity_interval_coverage"],
            ">=",
            hypothesis_config["minimum_remedied_oracle_severity_interval_coverage"],
        ),
        _check(
            "scientific_hypothesis",
            "H22_13_operational_detection_improves_over_physics_control",
            hybrid_operational["severe_detection_fraction"]
            - control_operational["severe_detection_fraction"],
            ">=",
            hypothesis_config["minimum_operational_detection_improvement_over_physics"],
        ),
        _check(
            "scientific_hypothesis",
            "H22_14_oracle_detection_improves_over_physics_control",
            hybrid_oracle["severe_detection_fraction"]
            - control_oracle["severe_detection_fraction"],
            ">=",
            hypothesis_config["minimum_oracle_detection_improvement_over_physics"],
        ),
        _check(
            "scientific_hypothesis",
            "H22_15_remedied_operational_true_state_evidence_is_positive",
            hybrid_operational["mean_true_state_log_likelihood_advantage_over_healthy"],
            ">=",
            hypothesis_config[
                "minimum_remedied_operational_true_state_log_likelihood_advantage"
            ],
        ),
        _check(
            "scientific_hypothesis",
            "H22_16_remedied_oracle_true_state_evidence_is_positive",
            hybrid_oracle["mean_true_state_log_likelihood_advantage_over_healthy"],
            ">=",
            hypothesis_config[
                "minimum_remedied_oracle_true_state_log_likelihood_advantage"
            ],
        ),
    ]
    hypotheses = pd.DataFrame(hypothesis_rows)

    probability_sums = state_probabilities.groupby(
        ["method", "regime", "trajectory_zone", "monitoring_window"]
    )["posterior_probability"].sum()
    maximum_probability_sum_error = float(np.max(np.abs(probability_sums - 1.0)))
    healing_probability = 0.0
    for source_index, source_loss in enumerate(state_losses):
        for target_index, target_loss in enumerate(state_losses):
            if target_loss < source_loss:
                healing_probability = max(
                    healing_probability,
                    transition[source_index, target_index],
                )
    physics_copy_error = float(
        np.max(
            np.abs(
                templates_by_regime_method[
                    ("commissioning_reference", "physics_template_control")
                ]
                - physics_templates_array
            )
        )
    )
    hybrid_healthy_error = max(
        float(
            np.max(
                np.abs(templates_by_regime_method[(regime, "hybrid_local_linear")][0])
            )
        )
        for regime in regimes
    )
    development_cases = windows[windows["loss_fraction"].isin(development_losses)][
        "case_id"
    ].nunique()
    validation_cases = windows[np.isclose(windows["loss_fraction"], validation_loss)][
        "case_id"
    ].nunique()
    validation_rows = [
        *[
            _check("provenance", f"{name}_sha256_matches", value, "==", True)
            for name, value in hash_results.items()
        ],
        _check(
            "upstream",
            "stage21_core_validation_complete",
            bool(stage21_validation["supported"].all()),
            "==",
            True,
        ),
        _check(
            "data_split",
            "development_case_count",
            development_cases,
            "==",
            expected["development_case_count"],
        ),
        _check(
            "data_split",
            "severe_validation_case_count",
            validation_cases,
            "==",
            expected["validation_case_count"],
        ),
        _check(
            "data_split",
            "severe_validation_observations_used_in_remedy_fit",
            int(np.any(np.isclose(development_losses, validation_loss))),
            "==",
            0,
        ),
        _check(
            "dimensions",
            "state_count",
            len(state_ids),
            "==",
            expected["state_count"],
        ),
        _check(
            "dimensions",
            "trajectory_count",
            len(detections),
            "==",
            expected["trajectory_count"],
        ),
        _check(
            "dimensions",
            "posterior_history_rows",
            len(history),
            "==",
            int(expected["trajectory_count"]) * int(expected["windows_per_trajectory"]),
        ),
        _check(
            "probability",
            "maximum_posterior_sum_error",
            maximum_probability_sum_error,
            "<=",
            1.0e-12,
        ),
        _check(
            "transition",
            "maximum_probability_of_healing",
            healing_probability,
            "==",
            0.0,
        ),
        _check(
            "control",
            "maximum_physics_template_copy_error",
            physics_copy_error,
            "<=",
            1.0e-12,
        ),
        _check(
            "remedy",
            "maximum_hybrid_healthy_template_error",
            hybrid_healthy_error,
            "<=",
            1.0e-12,
        ),
        _check(
            "alarm",
            "probability_threshold_retained_from_stage21",
            alarm_config["probability_threshold"],
            "==",
            0.80,
        ),
        _check(
            "alarm",
            "persistence_retained_from_stage21",
            alarm_config["persistence_windows"],
            "==",
            2,
        ),
    ]
    validation = pd.DataFrame(validation_rows)
    if not bool(validation["supported"].all()):
        failed = validation.loc[~validation["supported"], "check"].tolist()
        raise RuntimeError(f"Stage 22 core validation failed: {failed}")

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(CALIBRATION_PATH, index=False, float_format="%.12g")
    state_library.to_csv(STATE_LIBRARY_PATH, index=False, float_format="%.12g")
    history.to_csv(POSTERIOR_PATH, index=False, float_format="%.12g")
    state_probabilities.to_csv(
        STATE_PROBABILITY_PATH,
        index=False,
        float_format="%.12g",
    )
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    detections.to_csv(DETECTION_PATH, index=False, float_format="%.12g")
    hypotheses.to_csv(HYPOTHESIS_PATH, index=False, float_format="%.12g")
    validation.to_csv(VALIDATION_PATH, index=False, float_format="%.12g")
    _create_figure(
        calibration,
        history,
        summary,
        float(alarm_config["probability_threshold"]),
        zone_order,
    )

    scientific_passes = int(hypotheses["supported"].sum())
    print("STAGE 22 - OPENFAST MODEL AND FEATURE-INFORMATION REMEDY")
    print("Split: 6 development cases (10%, 20%); 3 severe validation cases (30%)")
    print(
        "Operational severe detection physics/hybrid: "
        f"{control_operational['severe_detection_fraction']:.3f}/"
        f"{hybrid_operational['severe_detection_fraction']:.3f}"
    )
    print(
        "Oracle severe detection physics/hybrid: "
        f"{control_oracle['severe_detection_fraction']:.3f}/"
        f"{hybrid_oracle['severe_detection_fraction']:.3f}"
    )
    print(
        "Hybrid final zone selection operational/oracle: "
        f"{hybrid_operational['final_zone_selection_fraction']:.3f}/"
        f"{hybrid_oracle['final_zone_selection_fraction']:.3f}"
    )
    print(f"Core validation: {int(validation['supported'].sum())}/{len(validation)}")
    print(f"Scientific hypotheses: {scientific_passes}/{len(hypotheses)}")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS (predeclared remedy experiment complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
