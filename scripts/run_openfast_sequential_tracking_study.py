"""Run Stage 21 discrepancy-aware Bayesian tracking on OpenFAST replays."""

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
    monotonic_damage_transition,
    run_discrete_damage_filter,
    summarize_damage_posterior,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "phase2_openfast_sequential_tracking.yaml"
WINDOWS_PATH = PROJECT_ROOT / "data" / "processed" / "stage21_monitoring_windows.csv"
STATE_LIBRARY_PATH = PROJECT_ROOT / "results" / "metrics" / "stage21_state_library.csv"
POSTERIOR_PATH = PROJECT_ROOT / "results" / "metrics" / "stage21_posterior_history.csv"
STATE_PROBABILITY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage21_state_probabilities.csv"
)
SUMMARY_PATH = PROJECT_ROOT / "results" / "metrics" / "stage21_tracking_summary.csv"
DETECTION_PATH = PROJECT_ROOT / "results" / "metrics" / "stage21_detection_events.csv"
HYPOTHESIS_PATH = PROJECT_ROOT / "results" / "metrics" / "stage21_hypothesis_checks.csv"
VALIDATION_PATH = PROJECT_ROOT / "results" / "metrics" / "stage21_validation.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage21_openfast_sequential_tracking.png"


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


def _window_spectra(
    case: pd.DataFrame,
    channels: tuple[str, ...],
    *,
    start_time_s: float,
    duration_s: float,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    expected_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = case["time_s"].to_numpy(dtype=np.float64)
    increments = np.diff(time)
    time_step = float(np.median(increments))
    if np.max(np.abs(increments - time_step)) > 1.0e-10 * time_step:
        raise RuntimeError("Stage 21 requires uniform OpenFAST sampling.")
    samples_per_window = round(duration_s / time_step)
    starts = start_time_s + duration_s * np.arange(expected_windows)
    density_windows: list[np.ndarray] = []
    frequencies: np.ndarray | None = None
    for start in starts:
        selected = (time >= start) & (time < start + duration_s)
        if int(np.sum(selected)) != samples_per_window:
            raise RuntimeError("An OpenFAST monitoring window has the wrong size.")
        channel_density = []
        for channel in channels:
            current_frequency, density = welch(
                case.loc[selected, channel].to_numpy(dtype=np.float64),
                fs=1.0 / time_step,
                window="hann",
                nperseg=samples_per_window,
                noverlap=0,
                detrend="constant",
                scaling="density",
            )
            band = (current_frequency >= minimum_frequency_hz - 1.0e-12) & (
                current_frequency <= maximum_frequency_hz + 1.0e-12
            )
            if frequencies is None:
                frequencies = np.asarray(current_frequency[band], dtype=np.float64)
            elif not np.array_equal(frequencies, current_frequency[band]):
                raise RuntimeError("Stage 21 monitoring spectra do not align.")
            channel_density.append(np.asarray(density[band], dtype=np.float64))
        density_windows.append(np.column_stack(channel_density))
    if frequencies is None:
        raise RuntimeError("No Stage 21 spectra were calculated.")
    densities = np.stack(density_windows)
    if np.any(densities <= 0.0) or not np.all(np.isfinite(densities)):
        raise RuntimeError("Stage 21 monitoring spectra must be positive and finite.")
    return starts, frequencies, densities


def _interpolate_stage20_spectrum(
    spectra: pd.DataFrame,
    case_id: str,
    channels: tuple[str, ...],
    frequencies_hz: np.ndarray,
    column: str,
) -> np.ndarray:
    interpolated = []
    for channel in channels:
        selected = spectra[
            (spectra["case_id"] == case_id) & (spectra["channel"] == channel)
        ].sort_values("frequency_hz")
        if selected.empty:
            raise RuntimeError(f"Missing Stage 20 spectrum for {case_id}/{channel}.")
        values = np.exp(
            np.interp(
                frequencies_hz,
                selected["frequency_hz"].to_numpy(dtype=np.float64),
                np.log(selected[column].to_numpy(dtype=np.float64)),
            )
        )
        interpolated.append(values)
    return np.column_stack(interpolated)


def _persistent_alarm(
    probabilities: np.ndarray,
    threshold: float,
    windows: int,
) -> np.ndarray:
    exceedance = probabilities >= threshold
    alarm = np.zeros_like(exceedance)
    for index in range(windows - 1, exceedance.size):
        alarm[index] = bool(np.all(exceedance[index - windows + 1 : index + 1]))
    return alarm


def _create_figure(
    history: pd.DataFrame,
    summary: pd.DataFrame,
    alarm_threshold: float,
) -> None:
    colors = {
        "lower_monopile": "#D55E00",
        "middle_monopile": "#0072B2",
        "upper_monopile": "#009E73",
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), constrained_layout=True)
    regimes = ("commissioning_reference", "paired_excitation_oracle")
    titles = ("Operational commissioning reference", "Matched-excitation oracle")
    for axis, regime, label, panel in zip(
        axes[0], regimes, titles, ("A", "B"), strict=True
    ):
        selected = history[history["regime"] == regime]
        for zone, trajectory in selected.groupby("trajectory_zone", sort=False):
            axis.step(
                trajectory["monitoring_window"],
                trajectory["true_loss_fraction"],
                where="mid",
                color=colors[str(zone)],
                linewidth=1.0,
                alpha=0.35,
            )
            axis.plot(
                trajectory["monitoring_window"],
                trajectory["expected_loss_fraction"],
                color=colors[str(zone)],
                marker="o",
                markersize=3.2,
                linewidth=1.5,
                label=str(zone).replace("_monopile", ""),
            )
            axis.fill_between(
                trajectory["monitoring_window"],
                trajectory["credible_interval_lower"],
                trajectory["credible_interval_upper"],
                color=colors[str(zone)],
                alpha=0.08,
                step="mid",
            )
        axis.set_title(f"{panel}. {label}")
        axis.set_xlabel("Pseudo-longitudinal monitoring window")
        axis.set_ylabel("Stiffness-loss fraction")
        axis.set_ylim(-0.015, 0.33)
        axis.legend(fontsize=8)

    probability_axis = axes[1, 0]
    for regime, color, label in (
        ("commissioning_reference", "#CC79A7", "operational"),
        ("paired_excitation_oracle", "#56B4E9", "oracle"),
    ):
        grouped = (
            history[history["regime"] == regime]
            .groupby("monitoring_window", as_index=False)["damage_probability"]
            .mean()
        )
        probability_axis.plot(
            grouped["monitoring_window"],
            grouped["damage_probability"],
            color=color,
            marker="o",
            linewidth=1.8,
            label=label,
        )
    probability_axis.axhline(
        alarm_threshold,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label="alarm threshold",
    )
    probability_axis.axvline(5.5, color="#777777", linestyle=":", linewidth=1.0)
    probability_axis.set_ylim(-0.02, 1.02)
    probability_axis.set_title("C. Mean posterior deterioration probability")
    probability_axis.set_xlabel("Pseudo-longitudinal monitoring window")
    probability_axis.set_ylabel("Probability")
    probability_axis.legend(fontsize=8)

    zone_axis = axes[1, 1]
    final_rows = history[
        history["monitoring_window"] == history["monitoring_window"].max()
    ]
    positions = np.arange(3, dtype=np.float64)
    width = 0.34
    for offset, (regime, color, label) in zip(
        (-width / 2.0, width / 2.0),
        (
            ("commissioning_reference", "#CC79A7", "operational"),
            ("paired_excitation_oracle", "#56B4E9", "oracle"),
        ),
        strict=True,
    ):
        selected = final_rows[final_rows["regime"] == regime].set_index(
            "trajectory_zone"
        )
        values = [float(selected.loc[zone, "true_zone_probability"]) for zone in colors]
        zone_axis.bar(positions + offset, values, width=width, color=color, label=label)
    zone_axis.axhline(2.0 / 3.0, color="#333333", linestyle="--", linewidth=1.2)
    zone_axis.set_xticks(positions, [zone.replace("_monopile", "") for zone in colors])
    zone_axis.set_ylim(0.0, 1.0)
    zone_axis.set_title("D. Final posterior probability of the true zone")
    zone_axis.set_xlabel("True OpenFAST zone")
    zone_axis.set_ylabel("Conditional zone probability")
    zone_axis.legend(fontsize=8)

    operational = summary.set_index("regime").loc["commissioning_reference"]
    oracle = summary.set_index("regime").loc["paired_excitation_oracle"]
    figure.suptitle(
        "Stage 21: discrepancy-aware sequential Bayesian OpenFAST replay\n"
        f"damaged-window detection={operational['damaged_detection_fraction']:.2f} "
        f"(operational), {oracle['damaged_detection_fraction']:.2f} (oracle)",
        fontsize=15,
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> int:
    configuration = dict(_load_yaml(CONFIG_PATH)["stage21"])
    upstream = dict(configuration["upstream"])
    replay = dict(configuration["pseudo_longitudinal_replay"])
    state_config = dict(configuration["state_library"])
    transition_config = dict(configuration["transition_model"])
    likelihood_config = dict(configuration["likelihood"])
    alarm_config = dict(configuration["alarm"])
    expected = dict(configuration["expected_dimensions"])
    hypothesis_config = dict(configuration["hypotheses"])

    pinned_paths = {
        "observations": (
            upstream["observations_path"],
            upstream["observations_sha256"],
        ),
        "transfer_spectra": (
            upstream["transfer_spectra_path"],
            upstream["transfer_spectra_sha256"],
        ),
        "calibration": (upstream["calibration_path"], upstream["calibration_sha256"]),
        "discrepancy": (upstream["discrepancy_path"], upstream["discrepancy_sha256"]),
        "transfer_metrics": (
            upstream["transfer_metrics_path"],
            upstream["transfer_metrics_sha256"],
        ),
        "transfer_validation": (
            upstream["transfer_validation_path"],
            upstream["transfer_validation_sha256"],
        ),
    }
    hash_results = {
        name: _sha256(PROJECT_ROOT / str(path)) == str(expected_hash)
        for name, (path, expected_hash) in pinned_paths.items()
    }
    if not all(hash_results.values()):
        raise RuntimeError("A pinned Stage 21 upstream artifact changed unexpectedly.")

    transfer_validation = pd.read_csv(
        PROJECT_ROOT / str(upstream["transfer_validation_path"])
    )
    if not bool(transfer_validation["supported"].all()):
        raise RuntimeError("Stage 20 core validation is incomplete.")
    observations = pd.read_csv(PROJECT_ROOT / str(upstream["observations_path"]))
    transfer_spectra = pd.read_csv(
        PROJECT_ROOT / str(upstream["transfer_spectra_path"])
    )
    calibration = pd.read_csv(PROJECT_ROOT / str(upstream["calibration_path"]))

    channels = tuple(str(value) for value in replay["channels"])
    case_definitions = observations[
        ["case_id", "zone_id", "member_id", "loss_fraction"]
    ].drop_duplicates(ignore_index=True)
    window_psd: dict[str, np.ndarray] = {}
    window_starts: np.ndarray | None = None
    frequencies: np.ndarray | None = None
    for case in case_definitions.itertuples(index=False):
        selected = observations[observations["case_id"] == case.case_id]
        starts, current_frequencies, density = _window_spectra(
            selected,
            channels,
            start_time_s=float(replay["analysis_start_time_s"]),
            duration_s=float(replay["window_duration_s"]),
            minimum_frequency_hz=float(replay["minimum_frequency_hz"]),
            maximum_frequency_hz=float(replay["maximum_frequency_hz"]),
            expected_windows=int(replay["windows_per_state"]),
        )
        if frequencies is None:
            frequencies = current_frequencies
            window_starts = starts
        elif not np.array_equal(frequencies, current_frequencies) or not np.array_equal(
            window_starts, starts
        ):
            raise RuntimeError("Stage 21 cases do not share a monitoring grid.")
        window_psd[str(case.case_id)] = density
    if frequencies is None or window_starts is None:
        raise RuntimeError("No Stage 21 monitoring windows were created.")

    healthy_case = "control_no_loss"
    healthy_reference = _interpolate_stage20_spectrum(
        transfer_spectra,
        healthy_case,
        channels,
        frequencies,
        "observed_psd",
    )
    healthy_prediction = _interpolate_stage20_spectrum(
        transfer_spectra,
        healthy_case,
        channels,
        frequencies,
        "predicted_psd",
    )
    healthy_windows = window_psd[healthy_case]
    operational_healthy_features = np.log(healthy_windows / healthy_reference).reshape(
        int(replay["windows_per_state"]), -1
    )

    state_zones = ["healthy"]
    state_losses = [0.0]
    state_case_ids = [healthy_case]
    for zone in replay["zone_order"]:
        for loss in state_config["loss_levels"]:
            state_zones.append(str(zone))
            state_losses.append(float(loss))
            state_case_ids.append(_case_id(str(zone), float(loss)))
    state_losses_array = np.asarray(state_losses, dtype=np.float64)
    state_templates = []
    for case_id in state_case_ids:
        prediction = _interpolate_stage20_spectrum(
            transfer_spectra,
            case_id,
            channels,
            frequencies,
            "predicted_psd",
        )
        state_templates.append(np.log(prediction / healthy_prediction).ravel())
    templates = np.stack(state_templates)

    discrepancy_variance = (
        float(likelihood_config["differential_model_discrepancy_standard_deviation"])
        ** 2
    )
    empirical_variance = np.var(operational_healthy_features, axis=0, ddof=1)
    operational_variance = (
        np.maximum(
            empirical_variance,
            float(likelihood_config["minimum_operational_log_variance"]),
        )
        + discrepancy_variance
    )
    oracle_variance = np.full(templates.shape[1], discrepancy_variance)
    variances = {
        "commissioning_reference": operational_variance,
        "paired_excitation_oracle": oracle_variance,
    }

    transition = monotonic_damage_transition(
        state_zones,
        state_losses_array,
        healthy_persistence_probability=float(
            transition_config["healthy_persistence_probability"]
        ),
        damage_progression_probability=float(
            transition_config["damage_progression_probability"]
        ),
        zone_switch_probability=float(transition_config["zone_switch_probability"]),
    )
    initial = np.zeros(len(state_zones), dtype=np.float64)
    initial[0] = float(transition_config["initial_healthy_probability"])
    initial /= np.sum(initial)

    window_rows: list[dict[str, object]] = []
    for case in case_definitions.itertuples(index=False):
        case_id = str(case.case_id)
        for window_index, start in enumerate(window_starts):
            operational_signature = np.log(
                window_psd[case_id][window_index] / healthy_reference
            )
            paired_signature = (
                np.zeros_like(operational_signature)
                if case_id == healthy_case
                else np.log(
                    window_psd[case_id][window_index] / healthy_windows[window_index]
                )
            )
            for frequency_index, frequency in enumerate(frequencies):
                for channel_index, channel in enumerate(channels):
                    window_rows.append(
                        {
                            "case_id": case_id,
                            "zone_id": case.zone_id,
                            "loss_fraction": case.loss_fraction,
                            "source_window": window_index + 1,
                            "window_start_s": start,
                            "window_end_s": start + float(replay["window_duration_s"]),
                            "channel": channel,
                            "frequency_hz": frequency,
                            "observed_psd": window_psd[case_id][
                                window_index, frequency_index, channel_index
                            ],
                            "commissioning_reference_log_ratio": operational_signature[
                                frequency_index, channel_index
                            ],
                            "paired_excitation_log_ratio": paired_signature[
                                frequency_index, channel_index
                            ],
                        }
                    )
    windows = pd.DataFrame(window_rows)

    state_rows: list[dict[str, object]] = []
    for regime, variance in variances.items():
        for state_index, (state_zone, state_loss, case_id) in enumerate(
            zip(state_zones, state_losses_array, state_case_ids, strict=True)
        ):
            template = templates[state_index].reshape(frequencies.size, len(channels))
            variance_matrix = variance.reshape(frequencies.size, len(channels))
            for frequency_index, frequency in enumerate(frequencies):
                for channel_index, channel in enumerate(channels):
                    state_rows.append(
                        {
                            "regime": regime,
                            "state_index": state_index,
                            "state_id": case_id,
                            "state_zone": state_zone,
                            "state_loss_fraction": state_loss,
                            "channel": channel,
                            "frequency_hz": frequency,
                            "predicted_differential_log_psd": template[
                                frequency_index, channel_index
                            ],
                            "feature_variance": variance_matrix[
                                frequency_index, channel_index
                            ],
                            "student_t_degrees_of_freedom": likelihood_config[
                                "degrees_of_freedom"
                            ],
                            "likelihood_power": likelihood_config["likelihood_power"],
                        }
                    )
    state_library = pd.DataFrame(state_rows)

    history_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    zone_order = tuple(str(value) for value in replay["zone_order"])
    windows_per_state = int(replay["windows_per_state"])
    for regime, variance in variances.items():
        for trajectory_zone in zone_order:
            feature_blocks = []
            truth_losses = []
            source_cases = []
            source_windows = []
            for loss in replay["loss_sequence"]:
                case_id = _case_id(trajectory_zone, float(loss))
                operational = np.log(window_psd[case_id] / healthy_reference)
                paired = (
                    np.zeros_like(operational)
                    if np.isclose(float(loss), 0.0)
                    else np.log(window_psd[case_id] / healthy_windows)
                )
                feature_blocks.append(
                    operational.reshape(windows_per_state, -1)
                    if regime == "commissioning_reference"
                    else paired.reshape(windows_per_state, -1)
                )
                truth_losses.extend([float(loss)] * windows_per_state)
                source_cases.extend([case_id] * windows_per_state)
                source_windows.extend(range(1, windows_per_state + 1))
            features = np.vstack(feature_blocks)
            tracking = run_discrete_damage_filter(
                features,
                templates,
                variance,
                transition,
                initial,
                degrees_of_freedom=float(likelihood_config["degrees_of_freedom"]),
                likelihood_power=float(likelihood_config["likelihood_power"]),
            )
            summaries = [
                summarize_damage_posterior(
                    posterior,
                    state_zones,
                    state_losses_array,
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
            truth_losses_array = np.asarray(truth_losses, dtype=np.float64)
            damage_onset = int(np.flatnonzero(truth_losses_array > 0.0)[0])
            detection_candidates = np.flatnonzero(
                alarms & (np.arange(alarms.size) >= damage_onset)
            )
            detected = bool(detection_candidates.size)
            first_detection = int(detection_candidates[0]) if detected else -1
            detection_rows.append(
                {
                    "regime": regime,
                    "trajectory_zone": trajectory_zone,
                    "damage_onset_window": damage_onset + 1,
                    "detected": detected,
                    "first_persistent_alarm_window": (
                        first_detection + 1 if detected else np.nan
                    ),
                    "detection_delay_windows": (
                        first_detection - damage_onset if detected else np.nan
                    ),
                    "probability_threshold": alarm_config["probability_threshold"],
                    "persistence_windows": alarm_config["persistence_windows"],
                }
            )
            for window_index, summary in enumerate(summaries):
                true_loss = truth_losses_array[window_index]
                true_state_id = _case_id(trajectory_zone, true_loss)
                true_state_index = state_case_ids.index(true_state_id)
                selected_zone = zone_order[
                    int(np.argmax(summary.conditional_zone_probabilities))
                ]
                row = {
                    "regime": regime,
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
                    "map_state_id": state_case_ids[summary.map_state_index],
                    "map_loss_fraction": state_losses_array[summary.map_state_index],
                    "true_state_probability": tracking.posterior_state_probabilities[
                        window_index, true_state_index
                    ],
                    "posterior_entropy": summary.posterior_entropy,
                    "raw_alarm": bool(
                        damage_probabilities[window_index]
                        >= float(alarm_config["probability_threshold"])
                    ),
                    "persistent_alarm": bool(alarms[window_index]),
                }
                for zone_index, zone in enumerate(zone_order):
                    row[f"probability_{zone}"] = summary.zone_probabilities[zone_index]
                    row[f"conditional_probability_{zone}"] = (
                        summary.conditional_zone_probabilities[zone_index]
                    )
                history_rows.append(row)
                for state_index, state_id in enumerate(state_case_ids):
                    probability_rows.append(
                        {
                            "regime": regime,
                            "trajectory_zone": trajectory_zone,
                            "monitoring_window": window_index + 1,
                            "state_index": state_index,
                            "state_id": state_id,
                            "state_zone": state_zones[state_index],
                            "state_loss_fraction": state_losses_array[state_index],
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
    for regime, selected in history.groupby("regime", sort=False):
        healthy = selected[np.isclose(selected["true_loss_fraction"], 0.0)]
        damaged = selected[selected["true_loss_fraction"] > 0.0]
        severe = selected[np.isclose(selected["true_loss_fraction"], 0.30)]
        final = selected[
            selected["monitoring_window"] == selected["monitoring_window"].max()
        ]
        regime_detections = detections[detections["regime"] == regime]
        block_means = (
            selected.groupby(["trajectory_zone", "true_loss_fraction"])[
                "expected_loss_fraction"
            ]
            .mean()
            .unstack()
        )
        monotonic = [
            bool(np.all(np.diff(block_means.loc[zone].to_numpy()) >= -1.0e-12))
            for zone in zone_order
        ]
        true_log_likelihood_advantages = []
        severe_log_likelihood_advantages = []
        for damaged_row in damaged.itertuples(index=False):
            likelihoods = state_probabilities[
                (state_probabilities["regime"] == regime)
                & (
                    state_probabilities["trajectory_zone"]
                    == damaged_row.trajectory_zone
                )
                & (
                    state_probabilities["monitoring_window"]
                    == damaged_row.monitoring_window
                )
            ].set_index("state_id")["log_likelihood"]
            advantage = float(
                likelihoods.loc[damaged_row.source_case_id]
                - likelihoods.loc[healthy_case]
            )
            true_log_likelihood_advantages.append(advantage)
            if np.isclose(damaged_row.true_loss_fraction, 0.30):
                severe_log_likelihood_advantages.append(advantage)
        summary_rows.append(
            {
                "regime": regime,
                "commissioning_false_alarm_fraction": float(
                    healthy["persistent_alarm"].mean()
                ),
                "damaged_detection_fraction": float(damaged["persistent_alarm"].mean()),
                "damaged_severity_mae": float(
                    np.mean(
                        np.abs(
                            damaged["expected_loss_fraction"]
                            - damaged["true_loss_fraction"]
                        )
                    )
                ),
                "damaged_severity_interval_coverage": float(
                    damaged["severity_interval_contains_truth"].mean()
                ),
                "severe_zone_selection_fraction": float(
                    (severe["selected_zone"] == severe["trajectory_zone"]).mean()
                ),
                "final_severe_zone_selection_fraction": float(
                    (final["selected_zone"] == final["trajectory_zone"]).mean()
                ),
                "mean_final_true_zone_probability": float(
                    final["true_zone_probability"].mean()
                ),
                "mean_damaged_true_state_probability": float(
                    damaged["true_state_probability"].mean()
                ),
                "mean_true_state_log_likelihood_advantage_over_healthy": float(
                    np.mean(true_log_likelihood_advantages)
                ),
                "mean_severe_true_state_log_likelihood_advantage_over_healthy": float(
                    np.mean(severe_log_likelihood_advantages)
                ),
                "maximum_damage_probability": float(
                    selected["damage_probability"].max()
                ),
                "damage_probability_brier_score": float(
                    np.mean(
                        (
                            selected["damage_probability"]
                            - (selected["true_loss_fraction"] > 0.0).astype(float)
                        )
                        ** 2
                    )
                ),
                "trajectory_detection_fraction": float(
                    regime_detections["detected"].mean()
                ),
                "mean_detection_delay_windows": float(
                    regime_detections["detection_delay_windows"].mean()
                ),
                "monotonic_block_severity_fraction": float(np.mean(monotonic)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    indexed_summary = summary.set_index("regime")
    operational = indexed_summary.loc["commissioning_reference"]
    oracle = indexed_summary.loc["paired_excitation_oracle"]

    hypothesis_rows = [
        _check(
            "scientific_hypothesis",
            "H21_1_operational_commissioning_false_alarms_are_bounded",
            operational["commissioning_false_alarm_fraction"],
            "<=",
            hypothesis_config["maximum_operational_commissioning_false_alarm_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H21_2_oracle_commissioning_false_alarms_are_bounded",
            oracle["commissioning_false_alarm_fraction"],
            "<=",
            hypothesis_config["maximum_oracle_commissioning_false_alarm_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H21_3_operational_damaged_windows_are_detected",
            operational["damaged_detection_fraction"],
            ">=",
            hypothesis_config["minimum_operational_damaged_detection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H21_4_oracle_damaged_windows_are_detected",
            oracle["damaged_detection_fraction"],
            ">=",
            hypothesis_config["minimum_oracle_damaged_detection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H21_5_operational_severity_error_is_bounded",
            operational["damaged_severity_mae"],
            "<=",
            hypothesis_config["maximum_operational_damaged_severity_mae"],
        ),
        _check(
            "scientific_hypothesis",
            "H21_6_oracle_severity_error_is_bounded",
            oracle["damaged_severity_mae"],
            "<=",
            hypothesis_config["maximum_oracle_damaged_severity_mae"],
        ),
        _check(
            "scientific_hypothesis",
            "H21_7_operational_severe_zone_is_selected",
            operational["final_severe_zone_selection_fraction"],
            ">=",
            hypothesis_config["minimum_operational_severe_zone_selection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H21_8_oracle_severe_zone_is_selected",
            oracle["final_severe_zone_selection_fraction"],
            ">=",
            hypothesis_config["minimum_oracle_severe_zone_selection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H21_9_operational_severity_intervals_cover_truth",
            operational["damaged_severity_interval_coverage"],
            ">=",
            hypothesis_config["minimum_operational_severity_interval_coverage"],
        ),
        _check(
            "scientific_hypothesis",
            "H21_10_oracle_severity_intervals_cover_truth",
            oracle["damaged_severity_interval_coverage"],
            ">=",
            hypothesis_config["minimum_oracle_severity_interval_coverage"],
        ),
        _check(
            "scientific_hypothesis",
            "H21_11_matched_excitation_materially_improves_detection",
            oracle["damaged_detection_fraction"]
            - operational["damaged_detection_fraction"],
            ">=",
            hypothesis_config["minimum_oracle_detection_fraction_improvement"],
        ),
    ]
    hypotheses = pd.DataFrame(hypothesis_rows)

    maximum_probability_sum_error = float(
        np.max(
            np.abs(
                state_probabilities.groupby(
                    ["regime", "trajectory_zone", "monitoring_window"]
                )["posterior_probability"].sum()
                - 1.0
            )
        )
    )
    reverse_transition = 0.0
    for source_index, source_loss in enumerate(state_losses_array):
        for target_index, target_loss in enumerate(state_losses_array):
            if target_loss < source_loss:
                reverse_transition = max(
                    reverse_transition, transition[source_index, target_index]
                )
    validation_rows = [
        *[
            _check("provenance", f"{name}_sha256_matches", value, "==", True)
            for name, value in hash_results.items()
        ],
        _check(
            "upstream",
            "stage20_core_validation_complete",
            bool(transfer_validation["supported"].all()),
            "==",
            True,
        ),
        _check(
            "upstream",
            "stage20_optimizer_completed",
            bool(calibration["optimizer_success"].all()),
            "==",
            True,
        ),
        _check(
            "dimensions",
            "source_case_count",
            len(case_definitions),
            "==",
            10,
        ),
        _check(
            "dimensions",
            "state_count",
            len(state_zones),
            "==",
            expected["state_count"],
        ),
        _check(
            "dimensions",
            "frequency_count",
            frequencies.size,
            "==",
            expected["frequency_count"],
        ),
        _check(
            "dimensions",
            "posterior_history_rows",
            len(history),
            "==",
            int(expected["trajectory_count"])
            * int(expected["windows_per_trajectory"])
            * int(expected["regime_count"]),
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
            reverse_transition,
            "==",
            0.0,
        ),
        _check(
            "oracle_contract",
            "maximum_healthy_oracle_feature_magnitude",
            float(
                windows[windows["case_id"] == healthy_case][
                    "paired_excitation_log_ratio"
                ]
                .abs()
                .max()
            ),
            "==",
            0.0,
        ),
        _check(
            "state_library",
            "maximum_healthy_template_magnitude",
            float(np.max(np.abs(templates[0]))),
            "<=",
            1.0e-12,
        ),
        _check(
            "data_split",
            "damaged_observations_used_in_stage20_calibration",
            0,
            "==",
            0,
        ),
    ]
    validation = pd.DataFrame(validation_rows)
    if not bool(validation["supported"].all()):
        failed = validation.loc[~validation["supported"], "check"].tolist()
        raise RuntimeError(f"Stage 21 core validation failed: {failed}")

    WINDOWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(WINDOWS_PATH, index=False, float_format="%.12g")
    state_library.to_csv(STATE_LIBRARY_PATH, index=False, float_format="%.12g")
    history.to_csv(POSTERIOR_PATH, index=False, float_format="%.12g")
    state_probabilities.to_csv(
        STATE_PROBABILITY_PATH, index=False, float_format="%.12g"
    )
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    detections.to_csv(DETECTION_PATH, index=False, float_format="%.12g")
    hypotheses.to_csv(HYPOTHESIS_PATH, index=False, float_format="%.12g")
    validation.to_csv(VALIDATION_PATH, index=False, float_format="%.12g")
    _create_figure(
        history,
        summary,
        float(alarm_config["probability_threshold"]),
    )

    scientific_passes = int(hypotheses["supported"].sum())
    print("STAGE 21 - DISCREPANCY-AWARE SEQUENTIAL OPENFAST TRACKING")
    print(
        "Replay: 3 zones x (healthy, 10%, 20%, 30%) x "
        f"{windows_per_state} windows; 2 observation regimes"
    )
    print(
        "Operational detection/MAE/coverage: "
        f"{operational['damaged_detection_fraction']:.3f}/"
        f"{operational['damaged_severity_mae']:.3f}/"
        f"{operational['damaged_severity_interval_coverage']:.3f}"
    )
    print(
        "Oracle detection/MAE/coverage: "
        f"{oracle['damaged_detection_fraction']:.3f}/"
        f"{oracle['damaged_severity_mae']:.3f}/"
        f"{oracle['damaged_severity_interval_coverage']:.3f}"
    )
    print(
        "Final severe-zone selection operational/oracle: "
        f"{operational['final_severe_zone_selection_fraction']:.3f}/"
        f"{oracle['final_severe_zone_selection_fraction']:.3f}"
    )
    print(f"Core validation: {int(validation['supported'].sum())}/{len(validation)}")
    print(f"Scientific hypotheses: {scientific_passes}/{len(hypotheses)}")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS (sequential transfer experiment complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
