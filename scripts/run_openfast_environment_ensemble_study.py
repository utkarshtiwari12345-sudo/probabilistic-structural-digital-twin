"""Run Stage 23 independent wind/wave OpenFAST ensemble validation."""

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
    ChannelMapping,
    map_openfast_channels,
    monotonic_damage_transition,
    patch_openfast_integer_seed,
    patch_subdyn_member_stiffness,
    read_openfast_output,
    resolve_openfast_dependencies,
    run_discrete_damage_filter,
    summarize_damage_posterior,
    uniform_time_grid_error,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "phase2_openfast_environment_ensemble.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CASE_OUTPUT_DIR = PROCESSED_DIR / "stage23_openfast_cases"
OBSERVATIONS_PATH = PROCESSED_DIR / "stage23_openfast_environment_ensemble.csv"
WINDOWS_PATH = PROCESSED_DIR / "stage23_monitoring_windows.csv"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
ENVIRONMENT_PATH = METRICS_DIR / "stage23_environment_manifest.csv"
CASE_MANIFEST_PATH = METRICS_DIR / "stage23_openfast_case_manifest.csv"
PATCH_PATH = METRICS_DIR / "stage23_openfast_stiffness_patches.csv"
POSTERIOR_PATH = METRICS_DIR / "stage23_posterior_history.csv"
STATE_PROBABILITY_PATH = METRICS_DIR / "stage23_state_probabilities.csv"
SUMMARY_PATH = METRICS_DIR / "stage23_ensemble_summary.csv"
DETECTION_PATH = METRICS_DIR / "stage23_detection_events.csv"
HYPOTHESIS_PATH = METRICS_DIR / "stage23_hypothesis_checks.csv"
VALIDATION_PATH = METRICS_DIR / "stage23_validation.csv"
LOG_DIR = PROJECT_ROOT / "results" / "logs" / "stage23"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage23_openfast_environment_ensemble.png"


def _load_configuration() -> dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream)["stage23"])


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


def _mappings(configuration: dict[str, object]) -> list[ChannelMapping]:
    contract = dict(configuration["output_contract"])
    return [
        ChannelMapping(
            source_channel=str(row["source_channel"]),
            source_unit=str(row["source_unit"]),
            target_name=str(row["target_name"]),
            target_unit=str(row["target_unit"]),
            scale_to_target=float(row["scale_to_target"]),
        )
        for row in contract["required_channels"]
    ]


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
        return "execution_timeout", None, time.perf_counter() - started, "", str(error)
    status = "completed" if completed.returncode == 0 else "solver_failed"
    return (
        status,
        completed.returncode,
        time.perf_counter() - started,
        completed.stdout,
        completed.stderr,
    )


def _mapped_output(
    output_path: Path,
    mappings: list[ChannelMapping],
) -> tuple[object, pd.DataFrame]:
    output = read_openfast_output(output_path)
    batch = map_openfast_channels(output, mappings)
    values = pd.DataFrame(batch.values, columns=batch.channel_names)
    values.insert(0, "time_s", batch.time_s)
    return output, values


def _window_spectra(
    case: pd.DataFrame,
    channels: tuple[str, ...],
    monitoring: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = case["time_s"].to_numpy(dtype=np.float64)
    increments = np.diff(time_s)
    time_step_s = float(np.median(increments))
    if np.max(np.abs(increments - time_step_s)) > 1.0e-10 * time_step_s:
        raise RuntimeError("Stage 23 requires a uniform OpenFAST time grid.")
    duration_s = float(monitoring["window_duration_s"])
    windows = int(monitoring["windows_per_state"])
    samples_per_window = round(duration_s / time_step_s)
    starts = float(monitoring["analysis_start_time_s"]) + duration_s * np.arange(
        windows
    )
    densities: list[np.ndarray] = []
    frequencies: np.ndarray | None = None
    for start in starts:
        selected = (time_s >= start) & (time_s < start + duration_s)
        if int(np.sum(selected)) != samples_per_window:
            raise RuntimeError("A Stage 23 monitoring window has the wrong size.")
        channel_densities = []
        for channel in channels:
            current_frequency, density = welch(
                case.loc[selected, channel].to_numpy(dtype=np.float64),
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
                raise RuntimeError("Stage 23 frequency grids do not align.")
            channel_densities.append(density[band])
        densities.append(np.column_stack(channel_densities))
    if frequencies is None:
        raise RuntimeError("Stage 23 produced no spectra.")
    density_array = np.stack(densities)
    if np.any(density_array <= 0.0) or not np.all(np.isfinite(density_array)):
        raise RuntimeError("Stage 23 spectra must be positive and finite.")
    return starts, frequencies, density_array


def _feature_vector(
    rows: pd.DataFrame,
    column: str,
    channels: tuple[str, ...],
) -> np.ndarray:
    order = {channel: index for index, channel in enumerate(channels)}
    selected = rows.assign(_channel_order=rows["channel"].map(order)).sort_values(
        ["frequency_hz", "_channel_order"]
    )
    values = selected[column].to_numpy(dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("A Stage 23 feature vector is empty or nonfinite.")
    return values


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


def _create_figure(
    windows: pd.DataFrame,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    alarm_threshold: float,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), constrained_layout=True)
    method_colors = {
        "physics_template_control": "#999999",
        "hybrid_local_linear": "#009E73",
    }
    regime_colors = {
        "commissioning_reference": "#CC79A7",
        "paired_excitation_oracle": "#56B4E9",
    }

    spectra_axis = axes[0, 0]
    channel = "nacelle_imu_fore_aft_acceleration_m_per_s2"
    healthy = windows[
        (windows["case_id"] == "control_no_loss") & (windows["channel"] == channel)
    ]
    for realization, rows in healthy.groupby("realization_id", sort=False):
        mean_spectrum = rows.groupby("frequency_hz", as_index=False)[
            "observed_psd"
        ].mean()
        spectra_axis.semilogy(
            mean_spectrum["frequency_hz"],
            mean_spectrum["observed_psd"],
            linewidth=1.6,
            label=str(realization),
        )
    spectra_axis.set_xlabel("Frequency (Hz)")
    spectra_axis.set_ylabel("Healthy fore-aft PSD")
    spectra_axis.set_title("A. Independent healthy environments")
    spectra_axis.legend(fontsize=8)

    probability_axis = axes[0, 1]
    for method in method_colors:
        for regime, regime_color in regime_colors.items():
            selected = history[
                (history["method"] == method) & (history["regime"] == regime)
            ]
            grouped = selected.groupby("monitoring_window", as_index=False)[
                "damage_probability"
            ].mean()
            probability_axis.plot(
                grouped["monitoring_window"],
                grouped["damage_probability"],
                color=regime_color,
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
    probability_axis.axhline(alarm_threshold, color="#333333", linestyle="--")
    probability_axis.axvline(5.5, color="#777777", linestyle=":")
    probability_axis.set_ylim(-0.02, 1.02)
    probability_axis.set_xlabel("Monitoring window")
    probability_axis.set_ylabel("Mean damage probability")
    probability_axis.set_title("B. Frozen-tracker evidence")
    probability_axis.legend(fontsize=7, ncol=2)

    summary_indexed = summary.set_index(["method", "regime"])
    positions = np.arange(2, dtype=np.float64)
    width = 0.34
    detection_axis = axes[1, 0]
    for offset, method in zip((-width / 2.0, width / 2.0), method_colors, strict=True):
        values = [
            summary_indexed.loc[(method, regime), "trajectory_detection_fraction"]
            for regime in regime_colors
        ]
        detection_axis.bar(
            positions + offset,
            values,
            width=width,
            color=method_colors[method],
            label=method.replace("_", " "),
        )
    detection_axis.set_xticks(positions, ["operational", "oracle"])
    detection_axis.set_ylim(0.0, 1.0)
    detection_axis.set_ylabel("Valid detected-trajectory fraction")
    detection_axis.set_title("C. Independent severe detection")
    detection_axis.legend(fontsize=8)

    zone_axis = axes[1, 1]
    for offset, regime in zip((-width / 2.0, width / 2.0), regime_colors, strict=True):
        values = [
            summary_indexed.loc[(method, regime), "final_zone_selection_fraction"]
            for method in method_colors
        ]
        zone_axis.bar(
            positions + offset,
            values,
            width=width,
            color=regime_colors[regime],
            label=("operational" if regime == "commissioning_reference" else "oracle"),
        )
    zone_axis.axhline(2.0 / 3.0, color="#333333", linestyle="--")
    zone_axis.set_xticks(positions, ["physics", "hybrid"])
    zone_axis.set_ylim(0.0, 1.0)
    zone_axis.set_ylabel("Final true-zone selection fraction")
    zone_axis.set_title("D. Independent localisation")
    zone_axis.legend(fontsize=8)

    hybrid = summary_indexed.loc[
        ("hybrid_local_linear", "commissioning_reference")
    ]
    figure.suptitle(
        "Stage 23: independent wind/wave OpenFAST ensemble\n"
        f"hybrid operational false alarms={hybrid['false_alarm_fraction']:.2f}; "
        f"valid detections={hybrid['trajectory_detection_fraction']:.2f}",
        fontsize=15,
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> int:
    configuration = _load_configuration()
    upstream = dict(configuration["upstream"])
    executables = dict(configuration["executables"])
    parameters = dict(configuration["parameterization"])
    contract = dict(configuration["output_contract"])
    monitoring = dict(configuration["monitoring"])
    tracker = dict(configuration["frozen_tracker"])
    expected = dict(configuration["expected_dimensions"])
    hypotheses_config = dict(configuration["hypotheses"])

    curated_root = PROJECT_ROOT / str(upstream["curated_deck_root"])
    primary_relative = Path(str(upstream["primary_file"]))
    subdyn_relative = Path(str(upstream["subdyn_file"]))
    seastate_relative = Path(str(upstream["seastate_file"]))
    wind_relative = Path(str(upstream["wind_directory"]))
    turbsim_source = PROJECT_ROOT / str(upstream["turbsim_input_source"])
    state_library_path = PROJECT_ROOT / str(upstream["stage22_state_library_path"])
    stage22_validation_path = PROJECT_ROOT / str(upstream["stage22_validation_path"])
    openfast_executable = PROJECT_ROOT / str(executables["openfast_path"])
    turbsim_executable = PROJECT_ROOT / str(executables["turbsim_path"])
    run_root = PROJECT_ROOT / str(executables["run_root"])

    pinned_hashes = {
        "turbsim_input": (turbsim_source, upstream["turbsim_input_sha256"]),
        "stage22_state_library": (
            state_library_path,
            upstream["stage22_state_library_sha256"],
        ),
        "stage22_validation": (
            stage22_validation_path,
            upstream["stage22_validation_sha256"],
        ),
        "openfast_executable": (
            openfast_executable,
            executables["openfast_sha256"],
        ),
        "turbsim_executable": (
            turbsim_executable,
            executables["turbsim_sha256"],
        ),
    }
    hash_results = {
        name: path.is_file() and _sha256(path) == str(digest)
        for name, (path, digest) in pinned_hashes.items()
    }
    if not all(hash_results.values()):
        raise RuntimeError("A pinned Stage 23 input or executable changed.")
    stage22_validation = pd.read_csv(stage22_validation_path)
    if not bool(stage22_validation["supported"].all()):
        raise RuntimeError("Stage 22 core validation is incomplete.")

    mappings = _mappings(configuration)
    channels = tuple(mapping.target_name for mapping in mappings)
    cases = _case_definitions(configuration)
    clone_ids = {
        int(member_id): int(property_id)
        for member_id, property_id in dict(parameters["clone_property_ids"]).items()
    }
    all_member_ids = sorted(clone_ids)
    realizations = [
        dict(values) for values in configuration["environment_realizations"]
    ]

    CASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    environment_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    patch_rows: list[dict[str, object]] = []
    observation_frames: list[pd.DataFrame] = []
    case_values: dict[tuple[str, str], pd.DataFrame] = {}
    output_time_grids: dict[str, np.ndarray] = {}

    total_cases = len(realizations) * len(cases)
    print("STAGE 23 - INDEPENDENT OPENFAST ENVIRONMENT ENSEMBLE", flush=True)
    print(
        f"Design: {len(realizations)} wind/wave realizations x {len(cases)} cases",
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
            stdout = "Reused a seed-verified Stage 23 TurbSim field.\n"
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
        wave_hash = _sha256(environment_root / seastate_relative)
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
                stdout = "Reused a verified Stage 23 OpenFAST output.\n"
                stderr = ""
                print(
                    f"[{completed_cases:02d}/{total_cases:02d}] "
                    f"{realization_id}/{case_id}: reusing verified output",
                    flush=True,
                )
            else:
                for suffix in (".out", ".outb", ".sum", ".ech"):
                    candidate = (
                        case_directory / primary_relative.with_suffix(suffix).name
                    )
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
            (LOG_DIR / f"{log_stem}.stdout.txt").write_text(stdout, encoding="utf-8")
            (LOG_DIR / f"{log_stem}.stderr.txt").write_text(stderr, encoding="utf-8")
            if status != "completed" or not (reuse_case or run_output.is_file()):
                raise RuntimeError(
                    f"OpenFAST failed for {realization_id}/{case_id}: {status}"
                )
            if not reuse_case:
                shutil.copyfile(run_output, destination)
            output, values = _mapped_output(destination, mappings)
            contract_passed = bool(
                output.data.shape[0] == int(contract["expected_sample_count"])
                and len(output.channel_names) == int(contract["expected_channel_count"])
                and abs(output.time_step_s - float(contract["expected_time_step_s"]))
                <= 1.0e-12
                and uniform_time_grid_error(output.time_s) <= 1.0e-10
            )
            if realization_id not in output_time_grids:
                output_time_grids[realization_id] = output.time_s
            else:
                contract_passed = contract_passed and np.array_equal(
                    output_time_grids[realization_id], output.time_s
                )
            if not contract_passed:
                raise RuntimeError(
                    f"Output contract failed for {realization_id}/{case_id}."
                )
            values.insert(0, "loss_fraction", float(case["loss_fraction"]))
            values.insert(0, "member_id", target_member)
            values.insert(0, "zone_id", str(case["zone_id"]))
            values.insert(0, "case_id", case_id)
            values.insert(0, "wave_seed", wave_seed)
            values.insert(0, "wind_seed", wind_seed)
            values.insert(0, "realization_id", realization_id)
            case_values[(realization_id, case_id)] = values
            observation_frames.append(values)
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
                    "output_contract_passed": contract_passed,
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
    observations = pd.concat(observation_frames, ignore_index=True)

    psd_by_case: dict[tuple[str, str], np.ndarray] = {}
    frequency_grid: np.ndarray | None = None
    window_starts: np.ndarray | None = None
    for key, values in case_values.items():
        starts, frequencies, density = _window_spectra(values, channels, monitoring)
        if frequency_grid is None:
            frequency_grid = frequencies
            window_starts = starts
        elif not np.array_equal(frequency_grid, frequencies) or not np.array_equal(
            window_starts, starts
        ):
            raise RuntimeError("Stage 23 cases do not share a spectral grid.")
        psd_by_case[key] = density
    if frequency_grid is None or window_starts is None:
        raise RuntimeError("Stage 23 produced no monitoring windows.")

    window_rows: list[dict[str, object]] = []
    control_id = str(parameters["control_case_id"])
    for realization in realizations:
        realization_id = str(realization["realization_id"])
        healthy_windows = psd_by_case[(realization_id, control_id)]
        commissioning_reference = np.mean(healthy_windows, axis=0)
        for case in cases:
            case_id = str(case["case_id"])
            density = psd_by_case[(realization_id, case_id)]
            operational = np.log(density / commissioning_reference)
            oracle = (
                np.zeros_like(density)
                if case_id == control_id
                else np.log(density / healthy_windows)
            )
            for window_index, start in enumerate(window_starts):
                for frequency_index, frequency in enumerate(frequency_grid):
                    for channel_index, channel in enumerate(channels):
                        window_rows.append(
                            {
                                "realization_id": realization_id,
                                "wind_seed": realization["wind_seed"],
                                "wave_seed": realization["wave_seed"],
                                "case_id": case_id,
                                "zone_id": case["zone_id"],
                                "loss_fraction": case["loss_fraction"],
                                "source_window": window_index + 1,
                                "window_start_s": start,
                                "window_end_s": start
                                + float(monitoring["window_duration_s"]),
                                "channel": channel,
                                "frequency_hz": frequency,
                                "observed_psd": density[
                                    window_index, frequency_index, channel_index
                                ],
                                "commissioning_reference_log_ratio": operational[
                                    window_index, frequency_index, channel_index
                                ],
                                "paired_excitation_log_ratio": oracle[
                                    window_index, frequency_index, channel_index
                                ],
                            }
                        )
    windows = pd.DataFrame(window_rows)

    state_library = pd.read_csv(state_library_path)
    state_metadata = (
        state_library[
            ["state_index", "state_id", "state_zone", "state_loss_fraction"]
        ]
        .drop_duplicates()
        .sort_values("state_index")
    )
    state_ids = tuple(str(value) for value in state_metadata["state_id"])
    state_zones = tuple(str(value) for value in state_metadata["state_zone"])
    state_losses = state_metadata["state_loss_fraction"].to_numpy(dtype=np.float64)
    methods = tuple(str(value) for value in tracker["methods"])
    regimes = tuple(str(value) for value in tracker["regimes"])
    zone_order = tuple(str(value) for value in tracker["zone_order"])
    regime_columns = {
        "commissioning_reference": "commissioning_reference_log_ratio",
        "paired_excitation_oracle": "paired_excitation_log_ratio",
    }
    templates: dict[tuple[str, str], np.ndarray] = {}
    variances: dict[tuple[str, str], np.ndarray] = {}
    library_grid_valid = True
    for regime in regimes:
        for method in methods:
            selected_library = state_library[
                (state_library["regime"] == regime)
                & (state_library["method"] == method)
            ]
            templates[(regime, method)] = np.stack(
                [
                    _feature_vector(
                        selected_library[selected_library["state_id"] == state_id],
                        "differential_log_psd_template",
                        channels,
                    )
                    for state_id in state_ids
                ]
            )
            healthy_library = selected_library[
                selected_library["state_id"] == state_ids[0]
            ]
            variances[(regime, method)] = _feature_vector(
                healthy_library,
                "feature_variance",
                channels,
            )
            library_frequencies = np.sort(
                selected_library["frequency_hz"].unique().astype(np.float64)
            )
            library_grid_valid = library_grid_valid and np.allclose(
                library_frequencies,
                frequency_grid,
                rtol=0.0,
                atol=1.0e-12,
            )
    if not library_grid_valid:
        raise RuntimeError("Stage 22 and Stage 23 feature grids do not align.")

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
    windows_per_state = int(monitoring["windows_per_state"])
    severe_loss = float(parameters["severe_loss_fraction"])
    for realization in realizations:
        realization_id = str(realization["realization_id"])
        for regime in regimes:
            for method in methods:
                for trajectory_zone in zone_order:
                    severe_case = _case_id(trajectory_zone, severe_loss)
                    features = []
                    source_cases = []
                    source_windows = []
                    truth_losses = []
                    for case_id, truth_loss in (
                        (control_id, 0.0),
                        (severe_case, severe_loss),
                    ):
                        selected_case = windows[
                            (windows["realization_id"] == realization_id)
                            & (windows["case_id"] == case_id)
                        ]
                        for source_window in range(1, windows_per_state + 1):
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
                    tracking_result = run_discrete_damage_filter(
                        np.stack(features),
                        templates[(regime, method)],
                        variances[(regime, method)],
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
                            damaged_zone_order=zone_order,
                            meaningful_loss_fraction=float(
                                tracker["meaningful_loss_fraction"]
                            ),
                            credible_level=float(tracker["credible_level"]),
                        )
                        for posterior in tracking_result.posterior_state_probabilities
                    ]
                    damage_probabilities = np.asarray(
                        [summary.damage_probability for summary in summaries]
                    )
                    alarms = _persistent_alarm(
                        damage_probabilities,
                        float(tracker["alarm_probability_threshold"]),
                        int(tracker["alarm_persistence_windows"]),
                    )
                    onset = windows_per_state
                    pre_onset_false_alarm = bool(np.any(alarms[:onset]))
                    candidates = np.flatnonzero(
                        alarms & (np.arange(alarms.size) >= onset)
                    )
                    detected = bool(candidates.size) and not pre_onset_false_alarm
                    first_detection = int(candidates[0]) if detected else -1
                    detection_rows.append(
                        {
                            "realization_id": realization_id,
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
                        }
                    )
                    for window_index, summary in enumerate(summaries):
                        true_loss = float(truth_losses[window_index])
                        true_state_id = _case_id(trajectory_zone, true_loss)
                        true_state_index = state_ids.index(true_state_id)
                        selected_zone = zone_order[
                            int(np.argmax(summary.conditional_zone_probabilities))
                        ]
                        row = {
                            "realization_id": realization_id,
                            "regime": regime,
                            "method": method,
                            "trajectory_zone": trajectory_zone,
                            "monitoring_window": window_index + 1,
                            "source_case_id": source_cases[window_index],
                            "source_window": source_windows[window_index],
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
                                    zone_order.index(trajectory_zone)
                                ]
                                if true_loss > 0.0
                                else np.nan
                            ),
                            "map_state_id": state_ids[summary.map_state_index],
                            "map_loss_fraction": state_losses[summary.map_state_index],
                            "true_state_probability": (
                                tracking_result.posterior_state_probabilities[
                                    window_index, true_state_index
                                ]
                            ),
                            "true_state_log_likelihood_advantage_over_healthy": (
                                tracking_result.log_likelihoods[
                                    window_index, true_state_index
                                ]
                                - tracking_result.log_likelihoods[window_index, 0]
                                if true_loss > 0.0
                                else np.nan
                            ),
                            "posterior_entropy": summary.posterior_entropy,
                            "persistent_alarm": bool(alarms[window_index]),
                            "valid_detection_alarm": bool(
                                alarms[window_index]
                                and window_index >= onset
                                and not pre_onset_false_alarm
                            ),
                        }
                        history_rows.append(row)
                        for state_index, state_id in enumerate(state_ids):
                            probability_rows.append(
                                {
                                    "realization_id": realization_id,
                                    "regime": regime,
                                    "method": method,
                                    "trajectory_zone": trajectory_zone,
                                    "monitoring_window": window_index + 1,
                                    "state_index": state_index,
                                    "state_id": state_id,
                                    "predicted_probability": (
                                        tracking_result.predicted_state_probabilities[
                                            window_index, state_index
                                        ]
                                    ),
                                    "log_likelihood": (
                                        tracking_result.log_likelihoods[
                                            window_index, state_index
                                        ]
                                    ),
                                    "posterior_probability": (
                                        tracking_result.posterior_state_probabilities[
                                            window_index, state_index
                                        ]
                                    ),
                                }
                            )

    history = pd.DataFrame(history_rows)
    state_probabilities = pd.DataFrame(probability_rows)
    detections = pd.DataFrame(detection_rows)
    summary_rows: list[dict[str, object]] = []
    for (method, regime), selected in history.groupby(["method", "regime"]):
        healthy = selected[np.isclose(selected["true_loss_fraction"], 0.0)]
        severe = selected[np.isclose(selected["true_loss_fraction"], severe_loss)]
        final = severe[
            severe["monitoring_window"] == severe["monitoring_window"].max()
        ]
        selected_detections = detections[
            (detections["method"] == method) & (detections["regime"] == regime)
        ]
        summary_rows.append(
            {
                "method": method,
                "regime": regime,
                "trajectory_count": len(selected_detections),
                "false_alarm_fraction": float(healthy["persistent_alarm"].mean()),
                "pre_onset_false_alarm_trajectory_fraction": float(
                    selected_detections["pre_onset_false_alarm"].mean()
                ),
                "raw_severe_alarm_fraction": float(
                    severe["persistent_alarm"].mean()
                ),
                "valid_severe_detection_fraction": float(
                    severe["valid_detection_alarm"].mean()
                ),
                "trajectory_detection_fraction": float(
                    selected_detections["detected"].mean()
                ),
                "severity_mae": float(
                    np.mean(
                        np.abs(severe["expected_loss_fraction"] - severe_loss)
                    )
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
                "mean_true_state_log_likelihood_advantage_over_healthy": float(
                    severe[
                        "true_state_log_likelihood_advantage_over_healthy"
                    ].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    indexed_summary = summary.set_index(["method", "regime"])
    hybrid_operational = indexed_summary.loc[
        ("hybrid_local_linear", "commissioning_reference")
    ]
    hybrid_oracle = indexed_summary.loc[
        ("hybrid_local_linear", "paired_excitation_oracle")
    ]
    physics_operational = indexed_summary.loc[
        ("physics_template_control", "commissioning_reference")
    ]
    physics_oracle = indexed_summary.loc[
        ("physics_template_control", "paired_excitation_oracle")
    ]
    hypothesis_rows = [
        _check(
            "scientific_hypothesis",
            "H23_1_hybrid_operational_false_alarm_control",
            hybrid_operational["false_alarm_fraction"],
            "<=",
            hypotheses_config["maximum_hybrid_operational_false_alarm_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H23_2_hybrid_oracle_false_alarm_control",
            hybrid_oracle["false_alarm_fraction"],
            "<=",
            hypotheses_config["maximum_hybrid_oracle_false_alarm_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H23_3_hybrid_operational_trajectory_detection",
            hybrid_operational["trajectory_detection_fraction"],
            ">=",
            hypotheses_config[
                "minimum_hybrid_operational_trajectory_detection_fraction"
            ],
        ),
        _check(
            "scientific_hypothesis",
            "H23_4_hybrid_oracle_trajectory_detection",
            hybrid_oracle["trajectory_detection_fraction"],
            ">=",
            hypotheses_config["minimum_hybrid_oracle_trajectory_detection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H23_5_hybrid_operational_severity_mae",
            hybrid_operational["severity_mae"],
            "<=",
            hypotheses_config["maximum_hybrid_operational_severity_mae"],
        ),
        _check(
            "scientific_hypothesis",
            "H23_6_hybrid_oracle_severity_mae",
            hybrid_oracle["severity_mae"],
            "<=",
            hypotheses_config["maximum_hybrid_oracle_severity_mae"],
        ),
        _check(
            "scientific_hypothesis",
            "H23_7_hybrid_operational_final_zone_selection",
            hybrid_operational["final_zone_selection_fraction"],
            ">=",
            hypotheses_config[
                "minimum_hybrid_operational_final_zone_selection_fraction"
            ],
        ),
        _check(
            "scientific_hypothesis",
            "H23_8_hybrid_oracle_final_zone_selection",
            hybrid_oracle["final_zone_selection_fraction"],
            ">=",
            hypotheses_config["minimum_hybrid_oracle_final_zone_selection_fraction"],
        ),
        _check(
            "scientific_hypothesis",
            "H23_9_hybrid_operational_interval_coverage",
            hybrid_operational["severity_interval_coverage"],
            ">=",
            hypotheses_config["minimum_hybrid_operational_interval_coverage"],
        ),
        _check(
            "scientific_hypothesis",
            "H23_10_hybrid_oracle_interval_coverage",
            hybrid_oracle["severity_interval_coverage"],
            ">=",
            hypotheses_config["minimum_hybrid_oracle_interval_coverage"],
        ),
        _check(
            "scientific_hypothesis",
            "H23_11_operational_detection_improves_over_physics",
            hybrid_operational["trajectory_detection_fraction"]
            - physics_operational["trajectory_detection_fraction"],
            ">=",
            hypotheses_config["minimum_operational_detection_improvement_over_physics"],
        ),
        _check(
            "scientific_hypothesis",
            "H23_12_oracle_detection_improves_over_physics",
            hybrid_oracle["trajectory_detection_fraction"]
            - physics_oracle["trajectory_detection_fraction"],
            ">=",
            hypotheses_config["minimum_oracle_detection_improvement_over_physics"],
        ),
    ]
    hypotheses = pd.DataFrame(hypothesis_rows)

    probability_sums = state_probabilities.groupby(
        ["realization_id", "method", "regime", "trajectory_zone", "monitoring_window"]
    )["posterior_probability"].sum()
    maximum_probability_sum_error = float(np.max(np.abs(probability_sums - 1.0)))
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
    environment_unique = bool(
        environments["wind_field_sha256"].nunique() == len(realizations)
        and environments["seastate_sha256"].nunique() == len(realizations)
    )
    validation_rows = [
        *[
            _check("provenance", f"{name}_sha256_matches", value, "==", True)
            for name, value in hash_results.items()
        ],
        _check(
            "upstream",
            "stage22_core_validation_complete",
            bool(stage22_validation["supported"].all()),
            "==",
            True,
        ),
        _check(
            "environment",
            "realization_count",
            len(environments),
            "==",
            expected["realization_count"],
        ),
        _check(
            "environment",
            "wind_and_wave_seeds_unique",
            bool(
                environments["wind_seed"].nunique() == len(realizations)
                and environments["wave_seed"].nunique() == len(realizations)
            ),
            "==",
            True,
        ),
        _check(
            "environment",
            "generated_wind_and_seastate_hashes_unique",
            environment_unique,
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
            "all_output_contracts_passed",
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
            "data",
            "observation_row_count",
            len(observations),
            "==",
            int(expected["openfast_case_count"])
            * int(contract["expected_sample_count"]),
        ),
        _check(
            "data",
            "all_observations_finite",
            bool(np.isfinite(observations[list(channels)].to_numpy()).all()),
            "==",
            True,
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
            "stage22_stage23_grid_alignment",
            library_grid_valid,
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
            "alarm",
            "stage22_alarm_threshold_retained",
            tracker["alarm_probability_threshold"],
            "==",
            0.80,
        ),
        _check(
            "alarm",
            "stage22_alarm_persistence_retained",
            tracker["alarm_persistence_windows"],
            "==",
            2,
        ),
    ]
    validation = pd.DataFrame(validation_rows)
    if not bool(validation["supported"].all()):
        failed = validation.loc[~validation["supported"], "check"].tolist()
        raise RuntimeError(f"Stage 23 core validation failed: {failed}")

    observations.to_csv(OBSERVATIONS_PATH, index=False, float_format="%.12g")
    windows.to_csv(WINDOWS_PATH, index=False, float_format="%.12g")
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
        windows,
        history,
        summary,
        float(tracker["alarm_probability_threshold"]),
    )

    print(
        "Hybrid operational false alarms/detections: "
        f"{hybrid_operational['false_alarm_fraction']:.3f}/"
        f"{hybrid_operational['trajectory_detection_fraction']:.3f}"
    )
    print(
        "Hybrid oracle false alarms/detections: "
        f"{hybrid_oracle['false_alarm_fraction']:.3f}/"
        f"{hybrid_oracle['trajectory_detection_fraction']:.3f}"
    )
    print(
        "Hybrid final zone selection operational/oracle: "
        f"{hybrid_operational['final_zone_selection_fraction']:.3f}/"
        f"{hybrid_oracle['final_zone_selection_fraction']:.3f}"
    )
    print(f"Core validation: {int(validation['supported'].sum())}/{len(validation)}")
    print(
        "Scientific hypotheses: "
        f"{int(hypotheses['supported'].sum())}/{len(hypotheses)}"
    )
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS (independent environment experiment complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
