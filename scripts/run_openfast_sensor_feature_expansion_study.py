"""Run Stage 24 high-fidelity OpenFAST sensor/feature expansion study."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from probdt.openfast import (  # noqa: E402
    band_log_energy_ratio,
    fit_linear_damage_state_library,
    log_spectral_ratio,
    monotonic_damage_transition,
    paired_log_transmissibility,
    read_openfast_output,
    run_discrete_damage_filter,
    summarize_damage_posterior,
)

CONFIG_PATH = (
    PROJECT_ROOT / "config" / "phase2_openfast_sensor_feature_expansion.yaml"
)
SPECTRA_PATH = PROJECT_ROOT / "data" / "processed" / "stage24_sensor_spectra.csv"
FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stage24_candidate_features.csv"
)
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
CV_FOLDS_PATH = METRICS_DIR / "stage24_candidate_cv_folds.csv"
CV_SUMMARY_PATH = METRICS_DIR / "stage24_candidate_cv_summary.csv"
SELECTION_PATH = METRICS_DIR / "stage24_feature_selection.csv"
STATE_LIBRARY_PATH = METRICS_DIR / "stage24_state_library.csv"
POSTERIOR_PATH = METRICS_DIR / "stage24_posterior_history.csv"
STATE_PROBABILITY_PATH = METRICS_DIR / "stage24_state_probabilities.csv"
DETECTION_PATH = METRICS_DIR / "stage24_detection_events.csv"
SUMMARY_PATH = METRICS_DIR / "stage24_validation_summary.csv"
HYPOTHESIS_PATH = METRICS_DIR / "stage24_hypothesis_checks.csv"
VALIDATION_PATH = METRICS_DIR / "stage24_validation.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage24_openfast_sensor_information.png"


def _load_configuration() -> dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream)["stage24"])


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


def _window_spectra(
    time_s: np.ndarray,
    values: np.ndarray,
    monitoring: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    increments = np.diff(time_s)
    time_step_s = float(np.median(increments))
    if np.max(np.abs(increments - time_step_s)) > 1.0e-10 * time_step_s:
        raise RuntimeError("Stage 24 requires uniform OpenFAST sampling.")
    duration_s = float(monitoring["window_duration_s"])
    window_count = int(monitoring["windows_per_state"])
    samples_per_window = round(duration_s / time_step_s)
    starts = float(monitoring["analysis_start_time_s"]) + duration_s * np.arange(
        window_count
    )
    density_windows: list[np.ndarray] = []
    frequencies: np.ndarray | None = None
    for start in starts:
        selected = (time_s >= start) & (time_s < start + duration_s)
        if int(np.sum(selected)) != samples_per_window:
            raise RuntimeError("A Stage 24 monitoring window has the wrong size.")
        channel_density = []
        for channel_index in range(values.shape[1]):
            current_frequency, density = welch(
                values[selected, channel_index],
                fs=1.0 / time_step_s,
                window="hann",
                nperseg=samples_per_window,
                noverlap=0,
                detrend="constant",
                scaling="density",
            )
            band = (
                current_frequency
                >= float(monitoring["minimum_frequency_hz"]) - 1.0e-12
            ) & (
                current_frequency
                <= float(monitoring["maximum_frequency_hz"]) + 1.0e-12
            )
            if frequencies is None:
                frequencies = current_frequency[band]
            elif not np.array_equal(frequencies, current_frequency[band]):
                raise RuntimeError("Stage 24 frequency grids do not align.")
            channel_density.append(density[band])
        density_windows.append(np.column_stack(channel_density))
    if frequencies is None:
        raise RuntimeError("Stage 24 generated no spectra.")
    densities = np.stack(density_windows)
    if np.any(densities <= 0.0) or not np.all(np.isfinite(densities)):
        raise RuntimeError("Stage 24 spectra must be positive and finite.")
    return starts, frequencies, densities


def _persistent_alarm(
    probabilities: np.ndarray,
    threshold: float,
    persistence: int,
) -> np.ndarray:
    exceedance = probabilities >= threshold
    alarms = np.zeros_like(exceedance)
    for index in range(persistence - 1, exceedance.size):
        alarms[index] = bool(
            np.all(exceedance[index - persistence + 1 : index + 1])
        )
    return alarms


def _state_metadata(
    zones: tuple[str, ...],
    losses: tuple[float, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    state_ids = ["control_no_loss"]
    state_zones = ["healthy"]
    state_losses = [0.0]
    for zone in zones:
        for loss in losses:
            state_ids.append(_case_id(zone, loss))
            state_zones.append(zone)
            state_losses.append(loss)
    return (
        tuple(state_ids),
        tuple(state_zones),
        np.asarray(state_losses, dtype=np.float64),
    )


def _feature_names(
    family: str,
    frequencies: np.ndarray,
    bands: tuple[tuple[float, float], ...],
    acceleration_labels: tuple[str, ...],
    moment_labels: tuple[str, ...],
    pair_labels: tuple[str, ...],
) -> tuple[str, ...]:
    acceleration_psd = tuple(
        f"accel_psd:{label}:{frequency:.2f}Hz"
        for frequency in frequencies
        for label in acceleration_labels
    )
    moment_psd = tuple(
        f"moment_psd:{label}:{frequency:.2f}Hz"
        for frequency in frequencies
        for label in moment_labels
    )
    moment_bands = tuple(
        f"moment_band:{label}:{lower:.2f}-{upper:.2f}Hz"
        for lower, upper in bands
        for label in moment_labels
    )
    transmissibility = tuple(
        f"transmissibility:{label}:{lower:.2f}-{upper:.2f}Hz"
        for lower, upper in bands
        for label in pair_labels
    )
    names = {
        "nacelle_psd": acceleration_psd,
        "moment_psd": moment_psd,
        "moment_bands": moment_bands,
        "transmissibility_bands": transmissibility,
        "fused": acceleration_psd + moment_bands + transmissibility,
    }
    try:
        return names[family]
    except KeyError as error:
        raise RuntimeError(f"Unknown Stage 24 feature family: {family}") from error


def _candidate_features(
    family: str,
    acceleration_log_psd: np.ndarray,
    moment_log_psd: np.ndarray,
    moment_log_bands: np.ndarray,
    transmissibility: np.ndarray,
) -> np.ndarray:
    windows = acceleration_log_psd.shape[0]
    values = {
        "nacelle_psd": acceleration_log_psd.reshape(windows, -1),
        "moment_psd": moment_log_psd.reshape(windows, -1),
        "moment_bands": moment_log_bands.reshape(windows, -1),
        "transmissibility_bands": transmissibility.reshape(windows, -1),
        "fused": np.concatenate(
            [
                acceleration_log_psd.reshape(windows, -1),
                moment_log_bands.reshape(windows, -1),
                transmissibility.reshape(windows, -1),
            ],
            axis=1,
        ),
    }
    try:
        return values[family]
    except KeyError as error:
        raise RuntimeError(f"Unknown Stage 24 feature family: {family}") from error


def _fit_library(
    feature_store: dict[tuple[str, str, str, str], np.ndarray],
    candidate_id: str,
    regime: str,
    training_realizations: tuple[str, ...],
    zones: tuple[str, ...],
    state_zones: tuple[str, ...],
    state_losses: np.ndarray,
    severe_loss: float,
    minimum_variance: float,
) -> object:
    healthy = np.concatenate(
        [
            feature_store[(candidate_id, regime, realization, "control_no_loss")]
            for realization in training_realizations
        ],
        axis=0,
    )
    damaged_rows = []
    damaged_zones = []
    for realization in training_realizations:
        for zone in zones:
            values = feature_store[
                (candidate_id, regime, realization, _case_id(zone, severe_loss))
            ]
            damaged_rows.append(values)
            damaged_zones.extend([zone] * values.shape[0])
    return fit_linear_damage_state_library(
        healthy,
        np.concatenate(damaged_rows, axis=0),
        damaged_zones,
        state_zones,
        state_losses,
        training_loss_fraction=severe_loss,
        minimum_feature_variance=minimum_variance,
    )


def _evaluate(
    feature_store: dict[tuple[str, str, str, str], np.ndarray],
    candidate_id: str,
    regime: str,
    evaluation_realizations: tuple[str, ...],
    library: object,
    transition: np.ndarray,
    initial: np.ndarray,
    state_ids: tuple[str, ...],
    state_zones: tuple[str, ...],
    state_losses: np.ndarray,
    zones: tuple[str, ...],
    severe_loss: float,
    tracker: dict[str, object],
    alarm: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    healthy_count = feature_store[
        (candidate_id, regime, evaluation_realizations[0], "control_no_loss")
    ].shape[0]
    for realization in evaluation_realizations:
        for trajectory_zone in zones:
            severe_case = _case_id(trajectory_zone, severe_loss)
            features = np.concatenate(
                [
                    feature_store[
                        (candidate_id, regime, realization, "control_no_loss")
                    ],
                    feature_store[(candidate_id, regime, realization, severe_case)],
                ],
                axis=0,
            )
            result = run_discrete_damage_filter(
                features,
                library.templates,
                library.feature_variance,
                transition,
                initial,
                degrees_of_freedom=float(tracker["degrees_of_freedom"]),
                likelihood_power=float(tracker["likelihood_power"]),
            )
            summaries = [
                summarize_damage_posterior(
                    posterior,
                    state_zones,
                    state_losses,
                    damaged_zone_order=zones,
                    meaningful_loss_fraction=float(
                        tracker["meaningful_loss_fraction"]
                    ),
                    credible_level=float(tracker["credible_level"]),
                )
                for posterior in result.posterior_state_probabilities
            ]
            damage_probabilities = np.asarray(
                [summary.damage_probability for summary in summaries]
            )
            alarms = _persistent_alarm(
                damage_probabilities,
                float(alarm["probability_threshold"]),
                int(alarm["persistence_windows"]),
            )
            pre_onset_false_alarm = bool(np.any(alarms[:healthy_count]))
            candidates = np.flatnonzero(
                alarms & (np.arange(alarms.size) >= healthy_count)
            )
            detected = bool(candidates.size) and not pre_onset_false_alarm
            first_detection = int(candidates[0]) if detected else -1
            detection_rows.append(
                {
                    "candidate_id": candidate_id,
                    "regime": regime,
                    "realization_id": realization,
                    "trajectory_zone": trajectory_zone,
                    "pre_onset_false_alarm": pre_onset_false_alarm,
                    "detected": detected,
                    "first_persistent_alarm_window": (
                        first_detection + 1 if detected else np.nan
                    ),
                    "detection_delay_windows": (
                        first_detection - healthy_count if detected else np.nan
                    ),
                }
            )
            for window_index, summary in enumerate(summaries):
                true_loss = 0.0 if window_index < healthy_count else severe_loss
                true_state_id = _case_id(trajectory_zone, true_loss)
                true_state_index = state_ids.index(true_state_id)
                selected_zone = zones[
                    int(np.argmax(summary.conditional_zone_probabilities))
                ]
                history_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "regime": regime,
                        "realization_id": realization,
                        "trajectory_zone": trajectory_zone,
                        "monitoring_window": window_index + 1,
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
                                zones.index(trajectory_zone)
                            ]
                            if true_loss > 0.0
                            else np.nan
                        ),
                        "map_state_id": state_ids[summary.map_state_index],
                        "map_loss_fraction": state_losses[summary.map_state_index],
                        "true_state_probability": (
                            result.posterior_state_probabilities[
                                window_index, true_state_index
                            ]
                        ),
                        "true_state_log_likelihood_advantage_over_healthy": (
                            result.log_likelihoods[window_index, true_state_index]
                            - result.log_likelihoods[window_index, 0]
                            if true_loss > 0.0
                            else np.nan
                        ),
                        "posterior_entropy": summary.posterior_entropy,
                        "persistent_alarm": bool(alarms[window_index]),
                        "valid_detection_alarm": bool(
                            alarms[window_index]
                            and window_index >= healthy_count
                            and not pre_onset_false_alarm
                        ),
                    }
                )
                for state_index, state_id in enumerate(state_ids):
                    probability_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "regime": regime,
                            "realization_id": realization,
                            "trajectory_zone": trajectory_zone,
                            "monitoring_window": window_index + 1,
                            "state_index": state_index,
                            "state_id": state_id,
                            "predicted_probability": (
                                result.predicted_state_probabilities[
                                    window_index, state_index
                                ]
                            ),
                            "log_likelihood": result.log_likelihoods[
                                window_index, state_index
                            ],
                            "posterior_probability": (
                                result.posterior_state_probabilities[
                                    window_index, state_index
                                ]
                            ),
                        }
                    )
    return (
        pd.DataFrame(history_rows),
        pd.DataFrame(detection_rows),
        pd.DataFrame(probability_rows),
    )


def _summary_metrics(
    history: pd.DataFrame,
    detections: pd.DataFrame,
    severe_loss: float,
) -> dict[str, float]:
    healthy = history[np.isclose(history["true_loss_fraction"], 0.0)]
    severe = history[np.isclose(history["true_loss_fraction"], severe_loss)]
    final = severe[
        severe["monitoring_window"] == severe["monitoring_window"].max()
    ]
    return {
        "false_alarm_fraction": float(healthy["persistent_alarm"].mean()),
        "pre_onset_false_alarm_trajectory_fraction": float(
            detections["pre_onset_false_alarm"].mean()
        ),
        "raw_severe_alarm_fraction": float(severe["persistent_alarm"].mean()),
        "valid_severe_detection_fraction": float(
            severe["valid_detection_alarm"].mean()
        ),
        "trajectory_detection_fraction": float(detections["detected"].mean()),
        "severity_mae": float(
            np.mean(np.abs(severe["expected_loss_fraction"] - severe_loss))
        ),
        "severity_interval_coverage": float(
            severe["severity_interval_contains_truth"].mean()
        ),
        "final_zone_selection_fraction": float(
            (final["selected_zone"] == final["trajectory_zone"]).mean()
        ),
        "mean_final_true_zone_probability": float(
            final["true_zone_probability"].mean()
        ),
        "mean_final_damage_probability": float(final["damage_probability"].mean()),
        "mean_true_state_log_likelihood_advantage_over_healthy": float(
            severe["true_state_log_likelihood_advantage_over_healthy"].mean()
        ),
    }


def _stage23_control_metrics(
    stage23_history: pd.DataFrame,
    stage23_detections: pd.DataFrame,
    regime: str,
    validation_realization: str,
    severe_loss: float,
) -> dict[str, float]:
    history = stage23_history[
        (stage23_history["method"] == "hybrid_local_linear")
        & (stage23_history["regime"] == regime)
        & (stage23_history["realization_id"] == validation_realization)
    ]
    detections = stage23_detections[
        (stage23_detections["method"] == "hybrid_local_linear")
        & (stage23_detections["regime"] == regime)
        & (stage23_detections["realization_id"] == validation_realization)
    ]
    return _summary_metrics(history, detections, severe_loss)


def _create_figure(
    cv_summary: pd.DataFrame,
    history: pd.DataFrame,
    stage23_history: pd.DataFrame,
    validation_summary: pd.DataFrame,
    selected_candidate: str,
    validation_realization: str,
    alarm_threshold: float,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), constrained_layout=True)
    regime_colors = {
        "commissioning_reference": "#CC79A7",
        "paired_excitation_oracle": "#56B4E9",
    }

    cv_axis = axes[0, 0]
    operational = cv_summary[
        cv_summary["regime"] == "commissioning_reference"
    ]
    for row in operational.itertuples(index=False):
        selected = row.candidate_id == selected_candidate
        cv_axis.scatter(
            row.pre_onset_false_alarm_trajectory_fraction,
            row.trajectory_detection_fraction,
            s=110 if selected else 55,
            color="#009E73" if selected else "#777777",
            marker="*" if selected else "o",
            zorder=3,
        )
    score_columns = [
        "pre_onset_false_alarm_trajectory_fraction",
        "trajectory_detection_fraction",
    ]
    for score, coincident in operational.groupby(score_columns, sort=False):
        candidate_labels = [
            str(value).replace("_", " ") for value in coincident["candidate_id"]
        ]
        label = (
            candidate_labels[0]
            if len(candidate_labels) == 1
            else f"{len(candidate_labels)} candidates overlap:\n"
            + "\n".join(candidate_labels)
        )
        cv_axis.annotate(
            label,
            score,
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    cv_axis.set_xlim(-0.03, 1.03)
    cv_axis.set_ylim(-0.03, 1.03)
    cv_axis.set_xlabel("Development false-alarm trajectory fraction")
    cv_axis.set_ylabel("Development detected-trajectory fraction")
    cv_axis.set_title("A. Leakage-safe candidate selection")

    probability_axis = axes[0, 1]
    for regime, color in regime_colors.items():
        selected = history[history["regime"] == regime]
        selected_mean = selected.groupby("monitoring_window", as_index=False)[
            "damage_probability"
        ].mean()
        control = stage23_history[
            (stage23_history["method"] == "hybrid_local_linear")
            & (stage23_history["regime"] == regime)
            & (stage23_history["realization_id"] == validation_realization)
        ]
        control_mean = control.groupby("monitoring_window", as_index=False)[
            "damage_probability"
        ].mean()
        probability_axis.plot(
            control_mean["monitoring_window"],
            control_mean["damage_probability"],
            color=color,
            linestyle=":",
            linewidth=1.5,
            label="Stage 23 / "
            + ("operational" if regime == "commissioning_reference" else "oracle"),
        )
        probability_axis.plot(
            selected_mean["monitoring_window"],
            selected_mean["damage_probability"],
            color=color,
            marker="o",
            linewidth=1.8,
            label="Stage 24 / "
            + ("operational" if regime == "commissioning_reference" else "oracle"),
        )
    probability_axis.axhline(alarm_threshold, color="#333333", linestyle="--")
    probability_axis.axvline(5.5, color="#777777", linestyle=":")
    probability_axis.set_ylim(-0.02, 1.02)
    probability_axis.set_xlabel("Validation monitoring window")
    probability_axis.set_ylabel("Mean damage probability")
    probability_axis.set_title("B. Held-out environment evidence")
    probability_axis.legend(fontsize=7, ncol=2)

    indexed = validation_summary.set_index(["method", "regime"])
    positions = np.arange(2, dtype=np.float64)
    width = 0.34
    detection_axis = axes[1, 0]
    for offset, method in zip(
        (-width / 2.0, width / 2.0),
        ("stage23_frozen_hybrid", "stage24_selected_sensor_model"),
        strict=True,
    ):
        values = [
            indexed.loc[(method, regime), "trajectory_detection_fraction"]
            for regime in regime_colors
        ]
        detection_axis.bar(
            positions + offset,
            values,
            width=width,
            color="#999999" if method.startswith("stage23") else "#009E73",
            label=method.replace("_", " "),
        )
    detection_axis.set_xticks(positions, ["operational", "oracle"])
    detection_axis.set_ylim(0.0, 1.0)
    detection_axis.set_ylabel("Detected-trajectory fraction")
    detection_axis.set_title("C. Held-out severe detection")
    detection_axis.legend(fontsize=8)

    zone_axis = axes[1, 1]
    selected_rows = validation_summary[
        validation_summary["method"] == "stage24_selected_sensor_model"
    ].set_index("regime")
    x = np.arange(2, dtype=np.float64)
    zone_values = [
        selected_rows.loc[regime, "final_zone_selection_fraction"]
        for regime in regime_colors
    ]
    severity_values = [
        1.0 - selected_rows.loc[regime, "severity_mae"] / 0.30
        for regime in regime_colors
    ]
    zone_axis.bar(x - width / 2.0, zone_values, width, label="zone selection")
    zone_axis.bar(
        x + width / 2.0,
        severity_values,
        width,
        color="#E69F00",
        label="1 - normalized severity MAE",
    )
    zone_axis.axhline(2.0 / 3.0, color="#333333", linestyle="--")
    zone_axis.set_xticks(x, ["operational", "oracle"])
    zone_axis.set_ylim(0.0, 1.0)
    zone_axis.set_title("D. Held-out localisation and severity")
    zone_axis.legend(fontsize=8)

    figure.suptitle(
        "Stage 24: high-fidelity sensor and feature-information expansion\n"
        f"development-selected layout: {selected_candidate.replace('_', ' ')}",
        fontsize=15,
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> int:
    configuration = _load_configuration()
    upstream = dict(configuration["upstream"])
    split = dict(configuration["split"])
    sensors = dict(configuration["sensors"])
    monitoring = dict(configuration["spectral_estimation"])
    state_config = dict(configuration["state_library"])
    likelihood = dict(configuration["likelihood"])
    transition_config = dict(configuration["transition_model"])
    alarm = dict(configuration["alarm"])
    expected = dict(configuration["expected_dimensions"])
    hypothesis_config = dict(configuration["hypotheses"])
    candidates = [dict(value) for value in configuration["candidates"]]

    pinned_paths = {
        "stage23_case_manifest": (
            upstream["stage23_case_manifest_path"],
            upstream["stage23_case_manifest_sha256"],
        ),
        "stage23_validation": (
            upstream["stage23_validation_path"],
            upstream["stage23_validation_sha256"],
        ),
        "stage23_posterior_history": (
            upstream["stage23_posterior_history_path"],
            upstream["stage23_posterior_history_sha256"],
        ),
        "stage23_detection_events": (
            upstream["stage23_detection_events_path"],
            upstream["stage23_detection_events_sha256"],
        ),
        "stage23_summary": (
            upstream["stage23_summary_path"],
            upstream["stage23_summary_sha256"],
        ),
    }
    hash_results = {
        name: _sha256(PROJECT_ROOT / str(path)) == str(expected_hash)
        for name, (path, expected_hash) in pinned_paths.items()
    }
    if not all(hash_results.values()):
        raise RuntimeError("A pinned Stage 24 upstream artifact changed.")
    stage23_validation = pd.read_csv(
        PROJECT_ROOT / str(upstream["stage23_validation_path"])
    )
    if not bool(stage23_validation["supported"].all()):
        raise RuntimeError("Stage 23 core validation is incomplete.")
    manifest = pd.read_csv(
        PROJECT_ROOT / str(upstream["stage23_case_manifest_path"])
    )
    stage23_history = pd.read_csv(
        PROJECT_ROOT / str(upstream["stage23_posterior_history_path"])
    )
    stage23_detections = pd.read_csv(
        PROJECT_ROOT / str(upstream["stage23_detection_events_path"])
    )

    acceleration_rows = [dict(value) for value in sensors["nacelle_acceleration"]]
    moment_rows = [dict(value) for value in sensors["distributed_bending_moment"]]
    sensor_rows = acceleration_rows + moment_rows
    channels = tuple(str(value["channel"]) for value in sensor_rows)
    labels = tuple(str(value["label"]) for value in sensor_rows)
    acceleration_labels = tuple(str(value["label"]) for value in acceleration_rows)
    moment_labels = tuple(str(value["label"]) for value in moment_rows)
    pairs_config = [dict(value) for value in sensors["transmissibility_pairs"]]
    pair_indices = tuple(
        (
            moment_labels.index(str(value["numerator"])),
            moment_labels.index(str(value["denominator"])),
        )
        for value in pairs_config
    )
    pair_labels = tuple(
        f"{value['numerator']}_over_{value['denominator']}"
        for value in pairs_config
    )
    bands = tuple(
        (float(value[0]), float(value[1]))
        for value in monitoring["frequency_bands_hz"]
    )

    psd_store: dict[tuple[str, str], np.ndarray] = {}
    spectra_rows: list[dict[str, object]] = []
    output_hashes_valid = True
    sensor_contract_valid = True
    frequency_grid: np.ndarray | None = None
    window_starts: np.ndarray | None = None
    for case in manifest.itertuples(index=False):
        output_path = PROJECT_ROOT / str(case.output_path)
        output_hashes_valid = output_hashes_valid and (
            output_path.is_file() and _sha256(output_path) == str(case.output_sha256)
        )
        output = read_openfast_output(output_path)
        for sensor in sensor_rows:
            channel = str(sensor["channel"])
            sensor_contract_valid = sensor_contract_valid and (
                channel in output.channel_names
                and output.unit(channel) == str(sensor["unit"])
            )
        values = np.column_stack([output.channel(channel) for channel in channels])
        starts, frequencies, densities = _window_spectra(
            output.time_s,
            values,
            monitoring,
        )
        if frequency_grid is None:
            frequency_grid = frequencies
            window_starts = starts
        elif not np.array_equal(frequency_grid, frequencies) or not np.array_equal(
            window_starts, starts
        ):
            raise RuntimeError("Stage 24 OpenFAST cases do not share a PSD grid.")
        key = (str(case.realization_id), str(case.case_id))
        psd_store[key] = densities
        for window_index, start in enumerate(starts):
            for frequency_index, frequency in enumerate(frequencies):
                for channel_index, channel in enumerate(channels):
                    spectra_rows.append(
                        {
                            "realization_id": case.realization_id,
                            "case_id": case.case_id,
                            "zone_id": case.zone_id,
                            "loss_fraction": case.loss_fraction,
                            "source_window": window_index + 1,
                            "window_start_s": start,
                            "channel": channel,
                            "sensor_label": labels[channel_index],
                            "frequency_hz": frequency,
                            "psd": densities[
                                window_index, frequency_index, channel_index
                            ],
                        }
                    )
    if frequency_grid is None or window_starts is None:
        raise RuntimeError("Stage 24 produced no sensor spectra.")
    spectra = pd.DataFrame(spectra_rows)

    feature_store: dict[tuple[str, str, str, str], np.ndarray] = {}
    feature_names: dict[str, tuple[str, ...]] = {}
    feature_rows: list[dict[str, object]] = []
    realizations = tuple(str(value) for value in manifest["realization_id"].unique())
    case_metadata = manifest[
        ["case_id", "zone_id", "loss_fraction"]
    ].drop_duplicates()
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        family = str(candidate["family"])
        names = _feature_names(
            family,
            frequency_grid,
            bands,
            acceleration_labels,
            moment_labels,
            pair_labels,
        )
        feature_names[candidate_id] = names
        for realization in realizations:
            healthy_psd = psd_store[(realization, "control_no_loss")]
            for case in case_metadata.itertuples(index=False):
                case_id = str(case.case_id)
                case_psd = psd_store[(realization, case_id)]
                for regime in (
                    "commissioning_reference",
                    "paired_excitation_oracle",
                ):
                    reference = (
                        np.mean(healthy_psd, axis=0)
                        if regime == "commissioning_reference"
                        else healthy_psd
                    )
                    acceleration_log_psd = log_spectral_ratio(
                        case_psd[:, :, : len(acceleration_rows)],
                        reference[:, : len(acceleration_rows)]
                        if reference.ndim == 2
                        else reference[:, :, : len(acceleration_rows)],
                    )
                    moment_case = case_psd[:, :, len(acceleration_rows) :]
                    moment_reference = (
                        reference[:, len(acceleration_rows) :]
                        if reference.ndim == 2
                        else reference[:, :, len(acceleration_rows) :]
                    )
                    moment_log_psd = log_spectral_ratio(
                        moment_case,
                        moment_reference,
                    )
                    moment_log_bands = band_log_energy_ratio(
                        frequency_grid,
                        moment_case,
                        moment_reference,
                        bands,
                    )
                    transmissibility = paired_log_transmissibility(
                        moment_log_bands,
                        pair_indices,
                    )
                    values = _candidate_features(
                        family,
                        acceleration_log_psd,
                        moment_log_psd,
                        moment_log_bands,
                        transmissibility,
                    )
                    feature_store[(candidate_id, regime, realization, case_id)] = (
                        values
                    )
                    for window_index in range(values.shape[0]):
                        for feature_index, feature_name in enumerate(names):
                            feature_rows.append(
                                {
                                    "candidate_id": candidate_id,
                                    "feature_family": family,
                                    "regime": regime,
                                    "realization_id": realization,
                                    "case_id": case_id,
                                    "zone_id": case.zone_id,
                                    "loss_fraction": case.loss_fraction,
                                    "source_window": window_index + 1,
                                    "feature_index": feature_index,
                                    "feature_name": feature_name,
                                    "feature_value": values[
                                        window_index, feature_index
                                    ],
                                }
                            )
    features = pd.DataFrame(feature_rows)

    zones = tuple(str(value) for value in state_config["zone_order"])
    loss_levels = tuple(float(value) for value in state_config["loss_levels"])
    severe_loss = float(state_config["training_loss_fraction"])
    state_ids, state_zones, state_losses = _state_metadata(zones, loss_levels)
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

    development_realizations = tuple(
        str(value) for value in split["development_realizations"]
    )
    validation_realization = str(split["validation_realization"])
    cv_rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        for regime in (
            "commissioning_reference",
            "paired_excitation_oracle",
        ):
            minimum_variance = float(
                likelihood[
                    "minimum_operational_feature_variance"
                    if regime == "commissioning_reference"
                    else "minimum_oracle_feature_variance"
                ]
            )
            for held_out in development_realizations:
                training = tuple(
                    value for value in development_realizations if value != held_out
                )
                library = _fit_library(
                    feature_store,
                    candidate_id,
                    regime,
                    training,
                    zones,
                    state_zones,
                    state_losses,
                    severe_loss,
                    minimum_variance,
                )
                history, detections, _ = _evaluate(
                    feature_store,
                    candidate_id,
                    regime,
                    (held_out,),
                    library,
                    transition,
                    initial,
                    state_ids,
                    state_zones,
                    state_losses,
                    zones,
                    severe_loss,
                    {**state_config, **likelihood},
                    alarm,
                )
                cv_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "regime": regime,
                        "held_out_realization": held_out,
                        "training_realizations": ";".join(training),
                        "feature_count": len(feature_names[candidate_id]),
                        **_summary_metrics(history, detections, severe_loss),
                    }
                )
    cv_folds = pd.DataFrame(cv_rows)
    metric_columns = [
        "false_alarm_fraction",
        "pre_onset_false_alarm_trajectory_fraction",
        "raw_severe_alarm_fraction",
        "valid_severe_detection_fraction",
        "trajectory_detection_fraction",
        "severity_mae",
        "severity_interval_coverage",
        "final_zone_selection_fraction",
        "mean_final_true_zone_probability",
        "mean_final_damage_probability",
        "mean_true_state_log_likelihood_advantage_over_healthy",
    ]
    cv_summary = (
        cv_folds.groupby(["candidate_id", "regime"], as_index=False)
        .agg(
            feature_count=("feature_count", "first"),
            **{column: (column, "mean") for column in metric_columns},
        )
        .sort_values(["candidate_id", "regime"])
    )
    primary_regime = str(dict(configuration["selection_rule"])["primary_regime"])
    ranked = cv_summary[cv_summary["regime"] == primary_regime].sort_values(
        [
            "pre_onset_false_alarm_trajectory_fraction",
            "trajectory_detection_fraction",
            "final_zone_selection_fraction",
            "severity_mae",
            "feature_count",
            "candidate_id",
        ],
        ascending=[True, False, False, True, True, True],
        kind="stable",
    )
    selected_candidate = str(ranked.iloc[0]["candidate_id"])
    selection = ranked.copy()
    selection.insert(0, "selection_rank", np.arange(1, len(selection) + 1))
    selection.insert(1, "selected", selection["candidate_id"] == selected_candidate)

    validation_histories = []
    validation_detections = []
    validation_probabilities = []
    state_rows: list[dict[str, object]] = []
    selected_libraries: dict[str, object] = {}
    for regime in (
        "commissioning_reference",
        "paired_excitation_oracle",
    ):
        minimum_variance = float(
            likelihood[
                "minimum_operational_feature_variance"
                if regime == "commissioning_reference"
                else "minimum_oracle_feature_variance"
            ]
        )
        library = _fit_library(
            feature_store,
            selected_candidate,
            regime,
            development_realizations,
            zones,
            state_zones,
            state_losses,
            severe_loss,
            minimum_variance,
        )
        selected_libraries[regime] = library
        history, detections, probabilities = _evaluate(
            feature_store,
            selected_candidate,
            regime,
            (validation_realization,),
            library,
            transition,
            initial,
            state_ids,
            state_zones,
            state_losses,
            zones,
            severe_loss,
            {**state_config, **likelihood},
            alarm,
        )
        validation_histories.append(history)
        validation_detections.append(detections)
        validation_probabilities.append(probabilities)
        for state_index, state_id in enumerate(state_ids):
            for feature_index, feature_name in enumerate(
                feature_names[selected_candidate]
            ):
                state_rows.append(
                    {
                        "candidate_id": selected_candidate,
                        "regime": regime,
                        "state_index": state_index,
                        "state_id": state_id,
                        "state_zone": state_zones[state_index],
                        "state_loss_fraction": state_losses[state_index],
                        "feature_index": feature_index,
                        "feature_name": feature_name,
                        "feature_template": library.templates[
                            state_index, feature_index
                        ],
                        "feature_variance": library.feature_variance[feature_index],
                    }
                )
    history = pd.concat(validation_histories, ignore_index=True)
    detections = pd.concat(validation_detections, ignore_index=True)
    state_probabilities = pd.concat(validation_probabilities, ignore_index=True)
    state_library = pd.DataFrame(state_rows)

    summary_rows: list[dict[str, object]] = []
    for regime in (
        "commissioning_reference",
        "paired_excitation_oracle",
    ):
        selected_history = history[history["regime"] == regime]
        selected_detections = detections[detections["regime"] == regime]
        summary_rows.append(
            {
                "method": "stage24_selected_sensor_model",
                "candidate_id": selected_candidate,
                "regime": regime,
                "validation_realization": validation_realization,
                "trajectory_count": len(selected_detections),
                **_summary_metrics(
                    selected_history,
                    selected_detections,
                    severe_loss,
                ),
            }
        )
        summary_rows.append(
            {
                "method": "stage23_frozen_hybrid",
                "candidate_id": "baseline_nacelle_psd",
                "regime": regime,
                "validation_realization": validation_realization,
                "trajectory_count": len(selected_detections),
                **_stage23_control_metrics(
                    stage23_history,
                    stage23_detections,
                    regime,
                    validation_realization,
                    severe_loss,
                ),
            }
        )
    validation_summary = pd.DataFrame(summary_rows)
    indexed_summary = validation_summary.set_index(["method", "regime"])
    operational = indexed_summary.loc[
        ("stage24_selected_sensor_model", "commissioning_reference")
    ]
    oracle = indexed_summary.loc[
        ("stage24_selected_sensor_model", "paired_excitation_oracle")
    ]
    control_operational = indexed_summary.loc[
        ("stage23_frozen_hybrid", "commissioning_reference")
    ]
    control_oracle = indexed_summary.loc[
        ("stage23_frozen_hybrid", "paired_excitation_oracle")
    ]
    selected_cv = cv_summary[
        (cv_summary["candidate_id"] == selected_candidate)
        & (cv_summary["regime"] == primary_regime)
    ].iloc[0]

    hypothesis_rows = [
        _check(
            "scientific_hypothesis",
            "H24_1_selected_development_false_alarm_control",
            selected_cv["pre_onset_false_alarm_trajectory_fraction"],
            "<=",
            hypothesis_config[
                "maximum_selected_cv_false_alarm_trajectory_fraction"
            ],
        ),
        _check(
            "scientific_hypothesis",
            "H24_2_selected_development_detection",
            selected_cv["trajectory_detection_fraction"],
            ">=",
            hypothesis_config["minimum_selected_cv_detection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H24_3_selected_development_zone_selection",
            selected_cv["final_zone_selection_fraction"],
            ">=",
            hypothesis_config["minimum_selected_cv_zone_selection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H24_4_validation_operational_false_alarm_control",
            operational["false_alarm_fraction"],
            "<=",
            hypothesis_config[
                "maximum_validation_operational_false_alarm_fraction"
            ],
        ),
        _check(
            "scientific_hypothesis",
            "H24_5_validation_oracle_false_alarm_control",
            oracle["false_alarm_fraction"],
            "<=",
            hypothesis_config["maximum_validation_oracle_false_alarm_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H24_6_validation_operational_detection",
            operational["trajectory_detection_fraction"],
            ">=",
            hypothesis_config["minimum_validation_operational_detection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H24_7_validation_oracle_detection",
            oracle["trajectory_detection_fraction"],
            ">=",
            hypothesis_config["minimum_validation_oracle_detection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H24_8_validation_operational_severity_mae",
            operational["severity_mae"],
            "<=",
            hypothesis_config["maximum_validation_operational_severity_mae"],
        ),
        _check(
            "scientific_hypothesis",
            "H24_9_validation_oracle_severity_mae",
            oracle["severity_mae"],
            "<=",
            hypothesis_config["maximum_validation_oracle_severity_mae"],
        ),
        _check(
            "scientific_hypothesis",
            "H24_10_validation_operational_zone_selection",
            operational["final_zone_selection_fraction"],
            ">=",
            hypothesis_config[
                "minimum_validation_operational_zone_selection_fraction"
            ],
        ),
        _check(
            "scientific_hypothesis",
            "H24_11_validation_oracle_zone_selection",
            oracle["final_zone_selection_fraction"],
            ">=",
            hypothesis_config["minimum_validation_oracle_zone_selection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H24_12_validation_operational_interval_coverage",
            operational["severity_interval_coverage"],
            ">=",
            hypothesis_config[
                "minimum_validation_operational_interval_coverage"
            ],
        ),
        _check(
            "scientific_hypothesis",
            "H24_13_validation_oracle_interval_coverage",
            oracle["severity_interval_coverage"],
            ">=",
            hypothesis_config["minimum_validation_oracle_interval_coverage"],
        ),
        _check(
            "scientific_hypothesis",
            "H24_14_operational_detection_improvement_over_stage23",
            operational["trajectory_detection_fraction"]
            - control_operational["trajectory_detection_fraction"],
            ">=",
            hypothesis_config[
                "minimum_operational_detection_improvement_over_stage23"
            ],
        ),
        _check(
            "scientific_hypothesis",
            "H24_15_oracle_detection_improvement_over_stage23",
            oracle["trajectory_detection_fraction"]
            - control_oracle["trajectory_detection_fraction"],
            ">=",
            hypothesis_config["minimum_oracle_detection_improvement_over_stage23"],
        ),
    ]
    hypotheses = pd.DataFrame(hypothesis_rows)

    feature_counts_valid = all(
        len(feature_names[candidate_id]) == int(expected_count)
        for candidate_id, expected_count in dict(
            expected["candidate_feature_counts"]
        ).items()
    )
    probability_sums = state_probabilities.groupby(
        [
            "candidate_id",
            "regime",
            "realization_id",
            "trajectory_zone",
            "monitoring_window",
        ]
    )["posterior_probability"].sum()
    probability_sum_error = float(np.max(np.abs(probability_sums - 1.0)))
    healing_probability = max(
        transition[source_index, target_index]
        for source_index, source_loss in enumerate(state_losses)
        for target_index, target_loss in enumerate(state_losses)
        if target_loss < source_loss
    )
    healthy_template_error = max(
        float(np.max(np.abs(library.templates[0])))
        for library in selected_libraries.values()
    )
    validation_rows = [
        *[
            _check("provenance", f"{name}_sha256_matches", value, "==", True)
            for name, value in hash_results.items()
        ],
        _check(
            "upstream",
            "stage23_core_validation_complete",
            bool(stage23_validation["supported"].all()),
            "==",
            True,
        ),
        _check(
            "provenance",
            "all_stage23_output_hashes_match",
            output_hashes_valid,
            "==",
            True,
        ),
        _check(
            "sensors",
            "all_requested_channels_and_units_available",
            sensor_contract_valid,
            "==",
            True,
        ),
        _check(
            "dimensions",
            "source_case_count",
            len(manifest),
            "==",
            expected["source_case_count"],
        ),
        _check(
            "dimensions",
            "source_channel_count",
            len(channels),
            "==",
            expected["source_channel_count"],
        ),
        _check(
            "dimensions",
            "frequency_count",
            len(frequency_grid),
            "==",
            expected["frequency_count"],
        ),
        _check(
            "dimensions",
            "candidate_count",
            len(candidates),
            "==",
            expected["candidate_count"],
        ),
        _check(
            "dimensions",
            "candidate_feature_counts",
            feature_counts_valid,
            "==",
            True,
        ),
        _check(
            "features",
            "all_candidate_features_finite",
            bool(np.isfinite(features["feature_value"]).all()),
            "==",
            True,
        ),
        _check(
            "data_split",
            "development_and_validation_disjoint",
            validation_realization not in development_realizations,
            "==",
            True,
        ),
        _check(
            "data_split",
            "validation_realization_absent_from_selection",
            validation_realization
            not in set(cv_folds["held_out_realization"])
            | set(cv_folds["training_realizations"]),
            "==",
            True,
        ),
        _check(
            "cross_validation",
            "cross_validation_fold_rows",
            len(cv_folds),
            "==",
            int(expected["candidate_count"])
            * int(expected["cross_validation_fold_count"])
            * 2,
        ),
        _check(
            "selection",
            "exactly_one_candidate_selected",
            int(selection["selected"].sum()),
            "==",
            1,
        ),
        _check(
            "tracker",
            "state_count",
            len(state_ids),
            "==",
            10,
        ),
        _check(
            "tracker",
            "validation_trajectory_count",
            len(detections),
            "==",
            expected["validation_trajectory_count"],
        ),
        _check(
            "tracker",
            "posterior_history_rows",
            len(history),
            "==",
            int(expected["validation_trajectory_count"])
            * int(expected["validation_windows_per_trajectory"]),
        ),
        _check(
            "probability",
            "maximum_posterior_sum_error",
            probability_sum_error,
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
            "state_library",
            "maximum_healthy_template_error",
            healthy_template_error,
            "<=",
            1.0e-12,
        ),
        _check(
            "alarm",
            "stage23_alarm_threshold_retained",
            alarm["probability_threshold"],
            "==",
            0.80,
        ),
        _check(
            "alarm",
            "stage23_alarm_persistence_retained",
            alarm["persistence_windows"],
            "==",
            2,
        ),
    ]
    validation = pd.DataFrame(validation_rows)
    if not bool(validation["supported"].all()):
        failed = validation.loc[~validation["supported"], "check"].tolist()
        raise RuntimeError(f"Stage 24 core validation failed: {failed}")

    SPECTRA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    spectra.to_csv(SPECTRA_PATH, index=False, float_format="%.12g")
    features.to_csv(FEATURES_PATH, index=False, float_format="%.12g")
    cv_folds.to_csv(CV_FOLDS_PATH, index=False, float_format="%.12g")
    cv_summary.to_csv(CV_SUMMARY_PATH, index=False, float_format="%.12g")
    selection.to_csv(SELECTION_PATH, index=False, float_format="%.12g")
    state_library.to_csv(STATE_LIBRARY_PATH, index=False, float_format="%.12g")
    history.to_csv(POSTERIOR_PATH, index=False, float_format="%.12g")
    state_probabilities.to_csv(
        STATE_PROBABILITY_PATH,
        index=False,
        float_format="%.12g",
    )
    detections.to_csv(DETECTION_PATH, index=False, float_format="%.12g")
    validation_summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    hypotheses.to_csv(HYPOTHESIS_PATH, index=False, float_format="%.12g")
    validation.to_csv(VALIDATION_PATH, index=False, float_format="%.12g")
    _create_figure(
        cv_summary,
        history,
        stage23_history,
        validation_summary,
        selected_candidate,
        validation_realization,
        float(alarm["probability_threshold"]),
    )

    print("STAGE 24 - HIGH-FIDELITY SENSOR AND FEATURE-INFORMATION EXPANSION")
    print(
        f"Candidate selection: {selected_candidate} "
        f"({len(feature_names[selected_candidate])} features)"
    )
    print(
        "Development operational false alarms/detections/zones: "
        f"{selected_cv['pre_onset_false_alarm_trajectory_fraction']:.3f}/"
        f"{selected_cv['trajectory_detection_fraction']:.3f}/"
        f"{selected_cv['final_zone_selection_fraction']:.3f}"
    )
    print(
        "Validation operational false alarms/detections/zones: "
        f"{operational['false_alarm_fraction']:.3f}/"
        f"{operational['trajectory_detection_fraction']:.3f}/"
        f"{operational['final_zone_selection_fraction']:.3f}"
    )
    print(
        "Validation oracle false alarms/detections/zones: "
        f"{oracle['false_alarm_fraction']:.3f}/"
        f"{oracle['trajectory_detection_fraction']:.3f}/"
        f"{oracle['final_zone_selection_fraction']:.3f}"
    )
    print(f"Core validation: {int(validation['supported'].sum())}/{len(validation)}")
    print(
        "Scientific hypotheses: "
        f"{int(hypotheses['supported'].sum())}/{len(hypotheses)}"
    )
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS (predeclared Stage 24 experiment complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
