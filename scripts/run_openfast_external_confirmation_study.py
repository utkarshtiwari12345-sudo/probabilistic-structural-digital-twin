"""Run Stage 25 external confirmation of the frozen Stage 24 tracker."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
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
    log_spectral_ratio,
    monotonic_damage_transition,
    patch_openfast_integer_seed,
    patch_subdyn_member_stiffness,
    read_openfast_output,
    resolve_openfast_dependencies,
    run_discrete_damage_filter,
    summarize_damage_posterior,
    uniform_time_grid_error,
)

CONFIG_PATH = (
    PROJECT_ROOT / "config" / "phase2_openfast_external_confirmation.yaml"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CASE_OUTPUT_DIR = PROCESSED_DIR / "stage25_openfast_cases"
SPECTRA_PATH = PROCESSED_DIR / "stage25_distributed_moment_spectra.csv"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
ENVIRONMENT_PATH = METRICS_DIR / "stage25_environment_manifest.csv"
CASE_MANIFEST_PATH = METRICS_DIR / "stage25_openfast_case_manifest.csv"
PATCH_PATH = METRICS_DIR / "stage25_openfast_stiffness_patches.csv"
POSTERIOR_PATH = METRICS_DIR / "stage25_posterior_history.csv"
STATE_PROBABILITY_PATH = METRICS_DIR / "stage25_state_probabilities.csv"
SUMMARY_PATH = METRICS_DIR / "stage25_external_confirmation_summary.csv"
DETECTION_PATH = METRICS_DIR / "stage25_detection_events.csv"
HYPOTHESIS_PATH = METRICS_DIR / "stage25_hypothesis_checks.csv"
VALIDATION_PATH = METRICS_DIR / "stage25_validation.csv"
LOG_DIR = PROJECT_ROOT / "results" / "logs" / "stage25"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage25_openfast_external_confirmation.png"


def _load_configuration() -> dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream)["stage25"])


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


def _read_integer_field(path: Path, field: str) -> int:
    matches = [
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if len(line.split()) >= 2 and line.split()[1] == field
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {field} field in {path}.")
    return int(matches[0])


def _case_definitions(configuration: dict[str, object]) -> list[dict[str, object]]:
    parameters = dict(configuration["parameterization"])
    severe_loss = float(parameters["severe_loss_fraction"])
    cases: list[dict[str, object]] = [
        {
            "case_id": str(parameters["control_case_id"]),
            "zone_id": "control",
            "member_id": 0,
            "loss_fraction": 0.0,
            "stiffness_scale": 1.0,
        }
    ]
    for zone in parameters["zones"]:
        values = dict(zone)
        cases.append(
            {
                "case_id": _case_id(str(values["zone_id"]), severe_loss),
                "zone_id": str(values["zone_id"]),
                "member_id": int(values["member_id"]),
                "loss_fraction": severe_loss,
                "stiffness_scale": 1.0 - severe_loss,
            }
        )
    return cases


def _run_program(
    executable: Path,
    argument: str,
    directory: Path,
    timeout_s: float,
) -> tuple[str, int | None, float, str, str]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(executable), argument],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except OSError as error:
        return "launch_failed", None, time.perf_counter() - started, "", str(error)
    except subprocess.TimeoutExpired as error:
        return "execution_timeout", None, time.perf_counter() - started, "", str(
            error
        )
    status = "completed" if completed.returncode == 0 else "solver_failed"
    return (
        status,
        completed.returncode,
        time.perf_counter() - started,
        completed.stdout,
        completed.stderr,
    )


def _window_spectra(
    time_s: np.ndarray,
    values: np.ndarray,
    monitoring: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    increments = np.diff(time_s)
    time_step_s = float(np.median(increments))
    if np.max(np.abs(increments - time_step_s)) > 1.0e-10 * time_step_s:
        raise RuntimeError("Stage 25 requires uniform OpenFAST sampling.")
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
            raise RuntimeError("A Stage 25 monitoring window has the wrong size.")
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
                raise RuntimeError("Stage 25 frequency grids do not align.")
            channel_density.append(density[band])
        density_windows.append(np.column_stack(channel_density))
    if frequencies is None:
        raise RuntimeError("Stage 25 generated no spectra.")
    densities = np.stack(density_windows)
    if np.any(densities <= 0.0) or not np.all(np.isfinite(densities)):
        raise RuntimeError("Stage 25 spectra must be positive and finite.")
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
        "trajectory_count": len(detections),
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


def _create_figure(
    spectra: pd.DataFrame,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    stage24_summary: pd.DataFrame,
    alarm_threshold: float,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), constrained_layout=True)
    regime_colors = {
        "commissioning_reference": "#CC79A7",
        "paired_excitation_oracle": "#56B4E9",
    }

    spectra_axis = axes[0, 0]
    healthy = spectra[
        (spectra["case_id"] == "control_no_loss")
        & (spectra["sensor_label"] == "z0_moment_x")
    ]
    for realization, rows in healthy.groupby("realization_id", sort=False):
        mean_spectrum = rows.groupby("frequency_hz", as_index=False)["psd"].mean()
        spectra_axis.semilogy(
            mean_spectrum["frequency_hz"],
            mean_spectrum["psd"],
            linewidth=1.4,
            label=str(realization),
        )
    spectra_axis.set_xlabel("Frequency (Hz)")
    spectra_axis.set_ylabel("Healthy z=0 m moment PSD")
    spectra_axis.set_title("A. Newly generated environments")
    spectra_axis.legend(fontsize=7, ncol=2)

    probability_axis = axes[0, 1]
    for regime, color in regime_colors.items():
        selected = history[history["regime"] == regime]
        grouped = selected.groupby("monitoring_window", as_index=False)[
            "damage_probability"
        ].mean()
        probability_axis.plot(
            grouped["monitoring_window"],
            grouped["damage_probability"],
            color=color,
            marker="o",
            linewidth=1.8,
            label=(
                "operational"
                if regime == "commissioning_reference"
                else "matched-excitation oracle"
            ),
        )
    probability_axis.axhline(alarm_threshold, color="#333333", linestyle="--")
    probability_axis.axvline(5.5, color="#777777", linestyle=":")
    probability_axis.set_ylim(-0.02, 1.02)
    probability_axis.set_xlabel("External monitoring window")
    probability_axis.set_ylabel("Mean damage probability")
    probability_axis.set_title("B. Frozen-tracker evidence")
    probability_axis.legend(fontsize=8)

    stage24_selected = stage24_summary[
        stage24_summary["method"] == "stage24_selected_sensor_model"
    ].set_index("regime")
    stage25_indexed = summary.set_index("regime")
    positions = np.arange(2, dtype=np.float64)
    width = 0.34
    detection_axis = axes[1, 0]
    stage24_values = [
        stage24_selected.loc[regime, "trajectory_detection_fraction"]
        for regime in regime_colors
    ]
    stage25_values = [
        stage25_indexed.loc[regime, "trajectory_detection_fraction"]
        for regime in regime_colors
    ]
    detection_axis.bar(
        positions - width / 2.0,
        stage24_values,
        width,
        color="#999999",
        label="Stage 24 e03 feature holdout",
    )
    detection_axis.bar(
        positions + width / 2.0,
        stage25_values,
        width,
        color="#009E73",
        label="Stage 25 new environments",
    )
    detection_axis.set_xticks(positions, ["operational", "oracle"])
    detection_axis.set_ylim(0.0, 1.0)
    detection_axis.set_ylabel("Detected-trajectory fraction")
    detection_axis.set_title("C. External detection confirmation")
    detection_axis.legend(fontsize=8)

    zone_axis = axes[1, 1]
    zone_values = [
        stage25_indexed.loc[regime, "final_zone_selection_fraction"]
        for regime in regime_colors
    ]
    severity_values = [
        1.0 - stage25_indexed.loc[regime, "severity_mae"] / 0.30
        for regime in regime_colors
    ]
    zone_axis.bar(
        positions - width / 2.0,
        zone_values,
        width,
        color="#0072B2",
        label="zone selection",
    )
    zone_axis.bar(
        positions + width / 2.0,
        severity_values,
        width,
        color="#E69F00",
        label="1 - normalized severity MAE",
    )
    zone_axis.axhline(2.0 / 3.0, color="#333333", linestyle="--")
    zone_axis.set_xticks(positions, ["operational", "oracle"])
    zone_axis.set_ylim(0.0, 1.0)
    zone_axis.set_title("D. External localisation and severity")
    zone_axis.legend(fontsize=8)

    operational = stage25_indexed.loc["commissioning_reference"]
    figure.suptitle(
        "Stage 25: external OpenFAST confirmation of the frozen Stage 24 model\n"
        f"operational false alarms={operational['false_alarm_fraction']:.2f}; "
        f"valid detections={operational['trajectory_detection_fraction']:.2f}",
        fontsize=15,
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> int:
    configuration = _load_configuration()
    freeze = dict(configuration["freeze"])
    upstream = dict(configuration["upstream"])
    executables = dict(configuration["executables"])
    parameters = dict(configuration["parameterization"])
    contract = dict(configuration["output_contract"])
    monitoring = dict(configuration["monitoring"])
    tracker = dict(configuration["frozen_tracker"])
    expected = dict(configuration["expected_dimensions"])
    hypotheses_config = dict(configuration["hypotheses"])
    sensors = [dict(value) for value in configuration["sensors"]]
    realizations = [dict(value) for value in configuration["external_realizations"]]

    curated_root = PROJECT_ROOT / str(upstream["curated_deck_root"])
    primary_relative = Path(str(upstream["primary_file"]))
    subdyn_relative = Path(str(upstream["subdyn_file"]))
    seastate_relative = Path(str(upstream["seastate_file"]))
    wind_relative = Path(str(upstream["wind_directory"]))
    turbsim_source = PROJECT_ROOT / str(upstream["turbsim_input_source"])
    openfast_executable = PROJECT_ROOT / str(executables["openfast_path"])
    turbsim_executable = PROJECT_ROOT / str(executables["turbsim_path"])
    run_root = PROJECT_ROOT / str(executables["run_root"])

    pinned_paths = {
        "stage24_config": (
            freeze["stage24_config_path"],
            freeze["stage24_config_sha256"],
        ),
        "stage24_feature_selection": (
            freeze["stage24_feature_selection_path"],
            freeze["stage24_feature_selection_sha256"],
        ),
        "stage24_state_library": (
            freeze["stage24_state_library_path"],
            freeze["stage24_state_library_sha256"],
        ),
        "stage24_validation": (
            freeze["stage24_validation_path"],
            freeze["stage24_validation_sha256"],
        ),
        "stage24_validation_summary": (
            freeze["stage24_validation_summary_path"],
            freeze["stage24_validation_summary_sha256"],
        ),
        "stage23_environment_manifest": (
            freeze["stage23_environment_manifest_path"],
            freeze["stage23_environment_manifest_sha256"],
        ),
        "turbsim_input": (
            upstream["turbsim_input_source"],
            upstream["turbsim_input_sha256"],
        ),
        "openfast_executable": (
            executables["openfast_path"],
            executables["openfast_sha256"],
        ),
        "turbsim_executable": (
            executables["turbsim_path"],
            executables["turbsim_sha256"],
        ),
    }
    hash_results = {
        name: (PROJECT_ROOT / str(path)).is_file()
        and _sha256(PROJECT_ROOT / str(path)) == str(digest)
        for name, (path, digest) in pinned_paths.items()
    }
    if not all(hash_results.values()):
        raise RuntimeError("A pinned Stage 25 input or executable changed.")

    stage24_config = dict(
        yaml.safe_load(
            (PROJECT_ROOT / str(freeze["stage24_config_path"])).read_text(
                encoding="utf-8"
            )
        )["stage24"]
    )
    stage24_selection = pd.read_csv(
        PROJECT_ROOT / str(freeze["stage24_feature_selection_path"])
    )
    stage24_library = pd.read_csv(
        PROJECT_ROOT / str(freeze["stage24_state_library_path"])
    )
    stage24_validation = pd.read_csv(
        PROJECT_ROOT / str(freeze["stage24_validation_path"])
    )
    stage24_summary = pd.read_csv(
        PROJECT_ROOT / str(freeze["stage24_validation_summary_path"])
    )
    stage23_environments = pd.read_csv(
        PROJECT_ROOT / str(freeze["stage23_environment_manifest_path"])
    )
    selected_rows = stage24_selection[stage24_selection["selected"]]
    selected_candidate = str(freeze["selected_candidate_id"])
    selection_valid = bool(
        len(selected_rows) == 1
        and str(selected_rows.iloc[0]["candidate_id"]) == selected_candidate
        and set(stage24_library["candidate_id"]) == {selected_candidate}
    )
    if not selection_valid:
        raise RuntimeError("The frozen Stage 24 candidate contract is invalid.")
    if not bool(stage24_validation["supported"].all()):
        raise RuntimeError("Stage 24 core validation is incomplete.")

    stage24_sensors = [
        dict(value)
        for value in dict(stage24_config["sensors"])[
            "distributed_bending_moment"
        ]
    ]
    stage24_monitoring = dict(stage24_config["spectral_estimation"])
    stage24_likelihood = dict(stage24_config["likelihood"])
    stage24_transition = dict(stage24_config["transition_model"])
    stage24_alarm = dict(stage24_config["alarm"])
    stage24_state = dict(stage24_config["state_library"])
    frozen_parameter_contract = bool(
        sensors == stage24_sensors
        and all(
            monitoring[name] == stage24_monitoring[name]
            for name in (
                "windows_per_state",
                "window_duration_s",
                "analysis_start_time_s",
                "minimum_frequency_hz",
                "maximum_frequency_hz",
            )
        )
        and tracker["degrees_of_freedom"]
        == stage24_likelihood["degrees_of_freedom"]
        and tracker["likelihood_power"] == stage24_likelihood["likelihood_power"]
        and tracker["initial_healthy_probability"]
        == stage24_transition["initial_healthy_probability"]
        and tracker["healthy_persistence_probability"]
        == stage24_transition["healthy_persistence_probability"]
        and tracker["healthy_entry_severity_weights"]
        == stage24_transition["healthy_entry_severity_weights"]
        and tracker["damage_progression_probability"]
        == stage24_transition["damage_progression_probability"]
        and tracker["zone_switch_probability"]
        == stage24_transition["zone_switch_probability"]
        and tracker["alarm_probability_threshold"]
        == stage24_alarm["probability_threshold"]
        and tracker["alarm_persistence_windows"]
        == stage24_alarm["persistence_windows"]
        and tracker["zone_order"] == stage24_state["zone_order"]
        and tracker["meaningful_loss_fraction"]
        == stage24_state["meaningful_loss_fraction"]
        and tracker["credible_level"] == stage24_state["credible_level"]
    )
    if not frozen_parameter_contract:
        raise RuntimeError("A frozen Stage 24 tracking parameter changed.")

    sensor_channels = tuple(str(value["channel"]) for value in sensors)
    sensor_labels = tuple(str(value["label"]) for value in sensors)
    cases = _case_definitions(configuration)
    clone_ids = {
        int(member_id): int(property_id)
        for member_id, property_id in dict(parameters["clone_property_ids"]).items()
    }
    all_member_ids = sorted(clone_ids)

    CASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    environment_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    patch_rows: list[dict[str, object]] = []
    output_time_grids: dict[str, np.ndarray] = {}
    psd_by_case: dict[tuple[str, str], np.ndarray] = {}
    spectra_rows: list[dict[str, object]] = []
    frequency_grid: np.ndarray | None = None
    window_starts: np.ndarray | None = None

    total_cases = len(realizations) * len(cases)
    print("STAGE 25 - EXTERNAL OPENFAST CONFIRMATION", flush=True)
    print(
        f"Design: {len(realizations)} new environments x {len(cases)} cases",
        flush=True,
    )
    completed_cases = 0
    original_wind_seed = _read_integer_field(turbsim_source, "RandSeed1")
    original_wave_seed = _read_integer_field(
        curated_root / seastate_relative,
        "WaveSeed(1)",
    )
    for realization in realizations:
        realization_id = str(realization["realization_id"])
        wind_seed = int(realization["wind_seed"])
        wave_seed = int(realization["wave_seed"])
        environment_root = run_root / realization_id / "base"
        wind_directory = environment_root / wind_relative
        turbsim_input = wind_directory / "90m_12mps_twr.inp"
        wind_output = wind_directory / "90m_12mps_twr.bts"
        seastate_path = environment_root / seastate_relative
        reuse_environment = bool(
            turbsim_input.is_file()
            and wind_output.is_file()
            and seastate_path.is_file()
            and _read_integer_field(turbsim_input, "RandSeed1") == wind_seed
            and _read_integer_field(seastate_path, "WaveSeed(1)") == wave_seed
        )
        if reuse_environment:
            status, return_code, runtime_s = "completed", 0, 0.0
            stdout = "Reused a seed-verified Stage 25 TurbSim field.\n"
            stderr = ""
            print(
                f"[{realization_id}] reusing verified wind seed {wind_seed}",
                flush=True,
            )
        else:
            shutil.copytree(curated_root, environment_root, dirs_exist_ok=True)
            shutil.copyfile(turbsim_source, turbsim_input)
            patch_openfast_integer_seed(turbsim_input, "RandSeed1", wind_seed)
            patch_openfast_integer_seed(seastate_path, "WaveSeed(1)", wave_seed)
            if wind_output.is_file():
                wind_output.unlink()
            print(f"[{realization_id}] generating wind seed {wind_seed}", flush=True)
            status, return_code, runtime_s, stdout, stderr = _run_program(
                turbsim_executable,
                turbsim_input.name,
                wind_directory,
                float(executables["turbsim_timeout_s"]),
            )
        (LOG_DIR / f"{realization_id}_turbsim.stdout.txt").write_text(
            stdout, encoding="utf-8"
        )
        (LOG_DIR / f"{realization_id}_turbsim.stderr.txt").write_text(
            stderr, encoding="utf-8"
        )
        if status != "completed" or not wind_output.is_file():
            raise RuntimeError(f"TurbSim failed for {realization_id}: {status}")
        wind_hash = _sha256(wind_output)
        wave_hash = _sha256(seastate_path)
        environment_rows.append(
            {
                **realization,
                "wind_seed_before": original_wind_seed,
                "wind_seed_after": wind_seed,
                "wave_seed_before": original_wave_seed,
                "wave_seed_after": wave_seed,
                "execution_reused": reuse_environment,
                "turbsim_status": status,
                "turbsim_return_code": return_code,
                "turbsim_runtime_s": runtime_s,
                "turbsim_input_sha256": _sha256(turbsim_input),
                "wind_field_sha256": wind_hash,
                "wind_field_size_bytes": wind_output.stat().st_size,
                "seastate_sha256": wave_hash,
            }
        )

        for case in cases:
            case_id = str(case["case_id"])
            completed_cases += 1
            case_root = run_root / realization_id / case_id
            destination = CASE_OUTPUT_DIR / f"{realization_id}__{case_id}.outb"
            cached_signature = None
            cached_paths = (
                case_root / subdyn_relative,
                case_root / wind_relative / wind_output.name,
                case_root / seastate_relative,
            )
            if destination.is_file() and all(path.is_file() for path in cached_paths):
                cached_signature = tuple(_sha256(path) for path in cached_paths)
            shutil.copytree(environment_root, case_root, dirs_exist_ok=True)
            target_member = int(case["member_id"])
            scales = {member_id: 1.0 for member_id in all_member_ids}
            if target_member > 0:
                scales[target_member] = float(case["stiffness_scale"])
            patches = patch_subdyn_member_stiffness(
                case_root / subdyn_relative,
                scales,
                clone_ids,
            )
            closure = resolve_openfast_dependencies(case_root, primary_relative)
            case_directory = case_root / primary_relative.parent
            run_output = case_directory / primary_relative.with_suffix(".outb").name
            current_paths = (
                case_root / subdyn_relative,
                case_root / wind_relative / wind_output.name,
                case_root / seastate_relative,
            )
            current_signature = tuple(_sha256(path) for path in current_paths)
            reuse_case = bool(
                destination.is_file()
                and cached_signature is not None
                and cached_signature == current_signature
            )
            if reuse_case:
                status, return_code, runtime_s = "completed", 0, 0.0
                stdout = "Reused a verified Stage 25 OpenFAST output.\n"
                stderr = ""
                print(
                    f"[{completed_cases:02d}/{total_cases:02d}] "
                    f"{realization_id}/{case_id}: reusing verified output",
                    flush=True,
                )
            else:
                for suffix in (".out", ".outb", ".sum", ".ech"):
                    candidate = case_directory / primary_relative.with_suffix(
                        suffix
                    ).name
                    if candidate.is_file():
                        candidate.unlink()
                print(
                    f"[{completed_cases:02d}/{total_cases:02d}] "
                    f"{realization_id}/{case_id}: running OpenFAST",
                    flush=True,
                )
                status, return_code, runtime_s, stdout, stderr = _run_program(
                    openfast_executable,
                    primary_relative.name,
                    case_directory,
                    float(executables["openfast_timeout_s"]),
                )
            log_stem = f"{realization_id}__{case_id}"
            (LOG_DIR / f"{log_stem}.stdout.txt").write_text(
                stdout, encoding="utf-8"
            )
            (LOG_DIR / f"{log_stem}.stderr.txt").write_text(
                stderr, encoding="utf-8"
            )
            if status != "completed" or not (reuse_case or run_output.is_file()):
                raise RuntimeError(
                    f"OpenFAST failed for {realization_id}/{case_id}: {status}"
                )
            if not reuse_case:
                shutil.copyfile(run_output, destination)
            output = read_openfast_output(destination)
            sensor_contract_passed = all(
                str(sensor["channel"]) in output.channel_names
                and output.unit(str(sensor["channel"])) == str(sensor["unit"])
                for sensor in sensors
            )
            output_contract_passed = bool(
                output.data.shape[0] == int(contract["expected_sample_count"])
                and len(output.channel_names) == int(contract["expected_channel_count"])
                and abs(
                    output.time_step_s - float(contract["expected_time_step_s"])
                )
                <= 1.0e-12
                and uniform_time_grid_error(output.time_s) <= 1.0e-10
                and sensor_contract_passed
            )
            if realization_id not in output_time_grids:
                output_time_grids[realization_id] = output.time_s
            else:
                output_contract_passed = output_contract_passed and np.array_equal(
                    output_time_grids[realization_id], output.time_s
                )
            if not output_contract_passed:
                raise RuntimeError(
                    f"Output contract failed for {realization_id}/{case_id}."
                )
            values = np.column_stack(
                [output.channel(channel) for channel in sensor_channels]
            )
            starts, frequencies, densities = _window_spectra(
                output.time_s,
                values,
                monitoring,
            )
            if frequency_grid is None:
                frequency_grid = frequencies
                window_starts = starts
            elif not np.array_equal(
                frequency_grid, frequencies
            ) or not np.array_equal(window_starts, starts):
                raise RuntimeError("Stage 25 OpenFAST PSD grids do not align.")
            psd_by_case[(realization_id, case_id)] = densities
            for window_index, start in enumerate(starts):
                for frequency_index, frequency in enumerate(frequencies):
                    for channel_index, channel in enumerate(sensor_channels):
                        spectra_rows.append(
                            {
                                "realization_id": realization_id,
                                "wind_seed": wind_seed,
                                "wave_seed": wave_seed,
                                "case_id": case_id,
                                "zone_id": case["zone_id"],
                                "loss_fraction": case["loss_fraction"],
                                "source_window": window_index + 1,
                                "window_start_s": start,
                                "channel": channel,
                                "sensor_label": sensor_labels[channel_index],
                                "frequency_hz": frequency,
                                "psd": densities[
                                    window_index, frequency_index, channel_index
                                ],
                            }
                        )
            case_wind_hash = _sha256(case_root / wind_relative / wind_output.name)
            case_wave_hash = _sha256(case_root / seastate_relative)
            manifest_rows.append(
                {
                    **realization,
                    **case,
                    "dependency_file_count": len(closure.files),
                    "unresolved_dependency_count": len(closure.unresolved),
                    "wind_field_sha256": case_wind_hash,
                    "seastate_sha256": case_wave_hash,
                    "environment_hashes_match": bool(
                        case_wind_hash == wind_hash and case_wave_hash == wave_hash
                    ),
                    "execution_status": status,
                    "execution_reused": reuse_case,
                    "return_code": return_code,
                    "runtime_s": runtime_s,
                    "output_path": destination.relative_to(PROJECT_ROOT).as_posix(),
                    "output_sha256": _sha256(destination),
                    "sample_count": output.data.shape[0],
                    "channel_count": len(output.channel_names),
                    "time_step_s": output.time_step_s,
                    "sensor_contract_passed": sensor_contract_passed,
                    "output_contract_passed": output_contract_passed,
                }
            )
            for item in patches:
                targeted = item.member_id == target_member
                patch_rows.append(
                    {
                        "realization_id": realization_id,
                        "case_id": case_id,
                        "zone_id": case["zone_id"],
                        "declared_target_member_id": target_member,
                        "member_id": item.member_id,
                        "targeted": targeted,
                        "expected_stiffness_scale": (
                            float(case["stiffness_scale"]) if targeted else 1.0
                        ),
                        "observed_stiffness_scale": item.stiffness_scale,
                        "young_e_before_pa": item.young_e_before_pa,
                        "young_e_after_pa": item.young_e_after_pa,
                        "shear_g_before_pa": item.shear_g_before_pa,
                        "shear_g_after_pa": item.shear_g_after_pa,
                        "material_density_kg_per_m3": item.material_density_kg_per_m3,
                        "outer_diameter_m": item.outer_diameter_m,
                        "wall_thickness_m": item.wall_thickness_m,
                    }
                )
            print(
                f"[{completed_cases:02d}/{total_cases:02d}] "
                f"{realization_id}/{case_id}: PASS ({runtime_s:.1f} s)",
                flush=True,
            )

    environments = pd.DataFrame(environment_rows)
    manifest = pd.DataFrame(manifest_rows)
    patches = pd.DataFrame(patch_rows)
    spectra = pd.DataFrame(spectra_rows)
    if frequency_grid is None or window_starts is None:
        raise RuntimeError("Stage 25 produced no distributed-moment spectra.")

    feature_store: dict[tuple[str, str, str], np.ndarray] = {}
    control_id = str(parameters["control_case_id"])
    regimes = tuple(str(value) for value in tracker["regimes"])
    for realization in realizations:
        realization_id = str(realization["realization_id"])
        healthy_psd = psd_by_case[(realization_id, control_id)]
        commissioning_reference = np.mean(healthy_psd, axis=0)
        for case in cases:
            case_id = str(case["case_id"])
            case_psd = psd_by_case[(realization_id, case_id)]
            operational = log_spectral_ratio(case_psd, commissioning_reference)
            oracle = (
                np.zeros_like(case_psd)
                if case_id == control_id
                else log_spectral_ratio(case_psd, healthy_psd)
            )
            feature_store[
                ("commissioning_reference", realization_id, case_id)
            ] = operational.reshape(operational.shape[0], -1)
            feature_store[
                ("paired_excitation_oracle", realization_id, case_id)
            ] = oracle.reshape(oracle.shape[0], -1)

    state_metadata = (
        stage24_library[
            ["state_index", "state_id", "state_zone", "state_loss_fraction"]
        ]
        .drop_duplicates()
        .sort_values("state_index")
    )
    state_ids = tuple(str(value) for value in state_metadata["state_id"])
    state_zones = tuple(str(value) for value in state_metadata["state_zone"])
    state_losses = state_metadata["state_loss_fraction"].to_numpy(dtype=np.float64)
    zones = tuple(str(value) for value in tracker["zone_order"])
    feature_names = tuple(
        f"moment_psd:{label}:{frequency:.2f}Hz"
        for frequency in frequency_grid
        for label in sensor_labels
    )
    templates: dict[str, np.ndarray] = {}
    variances: dict[str, np.ndarray] = {}
    library_feature_contract = True
    for regime in regimes:
        selected_library = stage24_library[
            stage24_library["regime"] == regime
        ].sort_values(["state_index", "feature_index"])
        templates[regime] = np.stack(
            [
                selected_library[selected_library["state_id"] == state_id]
                .sort_values("feature_index")["feature_template"]
                .to_numpy(dtype=np.float64)
                for state_id in state_ids
            ]
        )
        healthy_library = selected_library[
            selected_library["state_id"] == state_ids[0]
        ].sort_values("feature_index")
        variances[regime] = healthy_library["feature_variance"].to_numpy(
            dtype=np.float64
        )
        library_feature_contract = library_feature_contract and bool(
            tuple(healthy_library["feature_name"]) == feature_names
            and templates[regime].shape
            == (int(expected["state_count"]), int(expected["feature_count"]))
            and variances[regime].shape == (int(expected["feature_count"]),)
            and np.all(variances[regime] > 0.0)
        )
    if not library_feature_contract:
        raise RuntimeError("Stage 24 library does not align with Stage 25 features.")

    transition = monotonic_damage_transition(
        state_zones,
        state_losses,
        healthy_persistence_probability=float(
            tracker["healthy_persistence_probability"]
        ),
        damage_progression_probability=float(
            tracker["damage_progression_probability"]
        ),
        zone_switch_probability=float(tracker["zone_switch_probability"]),
        healthy_entry_severity_weights=tracker["healthy_entry_severity_weights"],
    )
    initial = np.zeros(len(state_ids), dtype=np.float64)
    initial[0] = float(tracker["initial_healthy_probability"])
    initial /= np.sum(initial)

    history_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    healthy_count = int(monitoring["windows_per_state"])
    severe_loss = float(parameters["severe_loss_fraction"])
    for realization in realizations:
        realization_id = str(realization["realization_id"])
        for regime in regimes:
            for trajectory_zone in zones:
                severe_case = _case_id(trajectory_zone, severe_loss)
                features = np.concatenate(
                    [
                        feature_store[(regime, realization_id, control_id)],
                        feature_store[(regime, realization_id, severe_case)],
                    ],
                    axis=0,
                )
                result = run_discrete_damage_filter(
                    features,
                    templates[regime],
                    variances[regime],
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
                    float(tracker["alarm_probability_threshold"]),
                    int(tracker["alarm_persistence_windows"]),
                )
                pre_onset_false_alarm = bool(np.any(alarms[:healthy_count]))
                alarm_candidates = np.flatnonzero(
                    alarms & (np.arange(alarms.size) >= healthy_count)
                )
                detected = bool(alarm_candidates.size) and not pre_onset_false_alarm
                first_detection = int(alarm_candidates[0]) if detected else -1
                detection_rows.append(
                    {
                        "candidate_id": selected_candidate,
                        "regime": regime,
                        "realization_id": realization_id,
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
                            "candidate_id": selected_candidate,
                            "regime": regime,
                            "realization_id": realization_id,
                            "trajectory_zone": trajectory_zone,
                            "monitoring_window": window_index + 1,
                            "true_loss_fraction": true_loss,
                            "damage_probability": summary.damage_probability,
                            "expected_loss_fraction": summary.expected_loss_fraction,
                            "credible_interval_lower": (
                                summary.credible_interval_lower
                            ),
                            "credible_interval_upper": (
                                summary.credible_interval_upper
                            ),
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
                            "map_loss_fraction": state_losses[
                                summary.map_state_index
                            ],
                            "true_state_probability": (
                                result.posterior_state_probabilities[
                                    window_index, true_state_index
                                ]
                            ),
                            "true_state_log_likelihood_advantage_over_healthy": (
                                result.log_likelihoods[
                                    window_index, true_state_index
                                ]
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
                                "candidate_id": selected_candidate,
                                "regime": regime,
                                "realization_id": realization_id,
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

    history = pd.DataFrame(history_rows)
    state_probabilities = pd.DataFrame(probability_rows)
    detections = pd.DataFrame(detection_rows)
    summary_rows = []
    for regime in regimes:
        selected_history = history[history["regime"] == regime]
        selected_detections = detections[detections["regime"] == regime]
        summary_rows.append(
            {
                "method": "stage24_frozen_distributed_moment_psd",
                "candidate_id": selected_candidate,
                "regime": regime,
                "external_realization_count": len(realizations),
                **_summary_metrics(
                    selected_history,
                    selected_detections,
                    severe_loss,
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    indexed = summary.set_index("regime")
    operational = indexed.loc["commissioning_reference"]
    oracle = indexed.loc["paired_excitation_oracle"]
    hypothesis_rows = [
        _check(
            "scientific_hypothesis",
            "H25_1_operational_false_alarm_control",
            operational["false_alarm_fraction"],
            "<=",
            hypotheses_config["maximum_operational_false_alarm_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H25_2_oracle_false_alarm_control",
            oracle["false_alarm_fraction"],
            "<=",
            hypotheses_config["maximum_oracle_false_alarm_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H25_3_operational_trajectory_detection",
            operational["trajectory_detection_fraction"],
            ">=",
            hypotheses_config["minimum_operational_trajectory_detection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H25_4_oracle_trajectory_detection",
            oracle["trajectory_detection_fraction"],
            ">=",
            hypotheses_config["minimum_oracle_trajectory_detection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H25_5_operational_severity_mae",
            operational["severity_mae"],
            "<=",
            hypotheses_config["maximum_operational_severity_mae"],
        ),
        _check(
            "scientific_hypothesis",
            "H25_6_oracle_severity_mae",
            oracle["severity_mae"],
            "<=",
            hypotheses_config["maximum_oracle_severity_mae"],
        ),
        _check(
            "scientific_hypothesis",
            "H25_7_operational_final_zone_selection",
            operational["final_zone_selection_fraction"],
            ">=",
            hypotheses_config["minimum_operational_final_zone_selection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H25_8_oracle_final_zone_selection",
            oracle["final_zone_selection_fraction"],
            ">=",
            hypotheses_config["minimum_oracle_final_zone_selection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H25_9_operational_interval_coverage",
            operational["severity_interval_coverage"],
            ">=",
            hypotheses_config["minimum_operational_interval_coverage"],
        ),
        _check(
            "scientific_hypothesis",
            "H25_10_oracle_interval_coverage",
            oracle["severity_interval_coverage"],
            ">=",
            hypotheses_config["minimum_oracle_interval_coverage"],
        ),
    ]
    hypotheses = pd.DataFrame(hypothesis_rows)

    previous_wind_seeds = set(stage23_environments["wind_seed"].astype(int))
    previous_wave_seeds = set(stage23_environments["wave_seed"].astype(int))
    current_wind_seeds = set(environments["wind_seed"].astype(int))
    current_wave_seeds = set(environments["wave_seed"].astype(int))
    stage23_wind_hashes = set(stage23_environments["wind_field_sha256"])
    stage23_wave_hashes = set(stage23_environments["seastate_sha256"])
    probability_sums = state_probabilities.groupby(
        ["regime", "realization_id", "trajectory_zone", "monitoring_window"]
    )["posterior_probability"].sum()
    probability_sum_error = float(np.max(np.abs(probability_sums - 1.0)))
    healing_probability = max(
        transition[source_index, target_index]
        for source_index, source_loss in enumerate(state_losses)
        for target_index, target_loss in enumerate(state_losses)
        if target_loss < source_loss
    )
    target_scale_correct = bool(
        np.allclose(
            patches["observed_stiffness_scale"],
            patches["expected_stiffness_scale"],
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    validation_rows = [
        *[
            _check("provenance", f"{name}_sha256_matches", value, "==", True)
            for name, value in hash_results.items()
        ],
        _check(
            "freeze",
            "stage24_core_validation_complete",
            bool(stage24_validation["supported"].all()),
            "==",
            True,
        ),
        _check(
            "freeze",
            "selected_candidate_contract",
            selection_valid,
            "==",
            True,
        ),
        _check(
            "freeze",
            "tracking_parameter_contract",
            frozen_parameter_contract,
            "==",
            True,
        ),
        _check(
            "environment",
            "external_realization_count",
            len(environments),
            "==",
            expected["realization_count"],
        ),
        _check(
            "environment",
            "external_seed_pairs_unique",
            bool(
                environments["wind_seed"].nunique() == len(realizations)
                and environments["wave_seed"].nunique() == len(realizations)
            ),
            "==",
            True,
        ),
        _check(
            "environment",
            "external_seeds_disjoint_from_stage23",
            not (current_wind_seeds & previous_wind_seeds)
            and not (current_wave_seeds & previous_wave_seeds),
            "==",
            True,
        ),
        _check(
            "environment",
            "generated_environment_hashes_unique",
            bool(
                environments["wind_field_sha256"].nunique() == len(realizations)
                and environments["seastate_sha256"].nunique() == len(realizations)
            ),
            "==",
            True,
        ),
        _check(
            "environment",
            "generated_hashes_disjoint_from_stage23",
            not (set(environments["wind_field_sha256"]) & stage23_wind_hashes)
            and not (set(environments["seastate_sha256"]) & stage23_wave_hashes),
            "==",
            True,
        ),
        _check(
            "environment",
            "all_turbsim_executions_completed",
            bool((environments["turbsim_status"] == "completed").all()),
            "==",
            True,
        ),
        _check(
            "experiment",
            "openfast_case_count",
            len(manifest),
            "==",
            expected["openfast_case_count"],
        ),
        _check(
            "experiment",
            "all_openfast_executions_completed",
            bool((manifest["execution_status"] == "completed").all()),
            "==",
            True,
        ),
        _check(
            "experiment",
            "all_output_and_sensor_contracts_passed",
            bool(manifest["output_contract_passed"].all()),
            "==",
            True,
        ),
        _check(
            "environment",
            "case_environment_hashes_match_realization",
            bool(manifest["environment_hashes_match"].all()),
            "==",
            True,
        ),
        _check(
            "deck",
            "minimum_dependency_count",
            int(manifest["dependency_file_count"].min()),
            "==",
            upstream["expected_dependency_count"],
        ),
        _check(
            "deck",
            "unresolved_dependency_count",
            int(manifest["unresolved_dependency_count"].sum()),
            "==",
            0,
        ),
        _check(
            "parameterization",
            "stiffness_scales_exact",
            target_scale_correct,
            "==",
            True,
        ),
        _check(
            "features",
            "source_channel_count",
            len(sensor_channels),
            "==",
            expected["source_channel_count"],
        ),
        _check(
            "features",
            "frequency_count",
            len(frequency_grid),
            "==",
            expected["frequency_count"],
        ),
        _check(
            "features",
            "feature_count",
            len(feature_names),
            "==",
            expected["feature_count"],
        ),
        _check(
            "features",
            "all_spectra_finite_and_positive",
            bool(np.isfinite(spectra["psd"]).all() and (spectra["psd"] > 0).all()),
            "==",
            True,
        ),
        _check(
            "features",
            "stage24_library_feature_contract",
            library_feature_contract,
            "==",
            True,
        ),
        _check(
            "tracker",
            "state_count",
            len(state_ids),
            "==",
            expected["state_count"],
        ),
        _check(
            "tracker",
            "trajectory_count",
            len(detections),
            "==",
            expected["trajectory_count"],
        ),
        _check(
            "tracker",
            "posterior_history_rows",
            len(history),
            "==",
            int(expected["trajectory_count"])
            * int(expected["windows_per_trajectory"]),
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
            "alarm",
            "stage24_alarm_threshold_retained",
            tracker["alarm_probability_threshold"],
            "==",
            0.80,
        ),
        _check(
            "alarm",
            "stage24_alarm_persistence_retained",
            tracker["alarm_persistence_windows"],
            "==",
            2,
        ),
    ]
    validation = pd.DataFrame(validation_rows)
    if not bool(validation["supported"].all()):
        failed = validation.loc[~validation["supported"], "check"].tolist()
        raise RuntimeError(f"Stage 25 core validation failed: {failed}")

    spectra.to_csv(SPECTRA_PATH, index=False, float_format="%.12g")
    environments.to_csv(ENVIRONMENT_PATH, index=False, float_format="%.12g")
    manifest.to_csv(CASE_MANIFEST_PATH, index=False, float_format="%.12g")
    patches.to_csv(PATCH_PATH, index=False, float_format="%.12g")
    history.to_csv(POSTERIOR_PATH, index=False, float_format="%.12g")
    state_probabilities.to_csv(
        STATE_PROBABILITY_PATH, index=False, float_format="%.12g"
    )
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    detections.to_csv(DETECTION_PATH, index=False, float_format="%.12g")
    hypotheses.to_csv(HYPOTHESIS_PATH, index=False, float_format="%.12g")
    validation.to_csv(VALIDATION_PATH, index=False, float_format="%.12g")
    _create_figure(
        spectra,
        history,
        summary,
        stage24_summary,
        float(tracker["alarm_probability_threshold"]),
    )

    print(
        "External operational false alarms/detections/zones: "
        f"{operational['false_alarm_fraction']:.3f}/"
        f"{operational['trajectory_detection_fraction']:.3f}/"
        f"{operational['final_zone_selection_fraction']:.3f}"
    )
    print(
        "External oracle false alarms/detections/zones: "
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
    print("RESULT: PASS (external confirmation experiment complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
