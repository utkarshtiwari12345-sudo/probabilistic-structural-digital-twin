"""Run Stage 19 controlled OpenFAST monopile-stiffness deterioration cases."""

from __future__ import annotations

import hashlib
import itertools
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
    patch_subdyn_member_stiffness,
    read_openfast_output,
    resolve_openfast_dependencies,
    uniform_time_grid_error,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "phase2_openfast_deterioration.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CASE_OUTPUT_DIR = PROCESSED_DIR / "stage19_openfast_cases"
OBSERVATIONS_PATH = PROCESSED_DIR / "stage19_openfast_deterioration.csv"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
CASE_MANIFEST_PATH = METRICS_DIR / "stage19_openfast_case_manifest.csv"
PATCH_PATH = METRICS_DIR / "stage19_openfast_stiffness_patches.csv"
RESPONSE_PATH = METRICS_DIR / "stage19_openfast_response_metrics.csv"
WINDOW_PATH = METRICS_DIR / "stage19_openfast_window_metrics.csv"
HYPOTHESIS_PATH = METRICS_DIR / "stage19_openfast_hypothesis_checks.csv"
VALIDATION_PATH = METRICS_DIR / "stage19_openfast_validation.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage19_openfast_deterioration.png"
LOG_DIR = PROJECT_ROOT / "results" / "logs" / "stage19_openfast"


def _load_configuration() -> dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream)["stage19"])


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


def _case_definitions(configuration: dict[str, object]) -> list[dict[str, object]]:
    parameterization = dict(configuration["parameterization"])
    cases: list[dict[str, object]] = [
        {
            "case_id": str(parameterization["control_case_id"]),
            "zone_id": "control",
            "member_id": 0,
            "loss_fraction": 0.0,
            "stiffness_scale": 1.0,
        }
    ]
    for zone in parameterization["zones"]:
        zone_data = dict(zone)
        for loss_fraction in parameterization["loss_fractions"]:
            loss = float(loss_fraction)
            loss_percent = round(100.0 * loss)
            cases.append(
                {
                    "case_id": f"{zone_data['zone_id']}_loss_{loss_percent:02d}pct",
                    "zone_id": str(zone_data["zone_id"]),
                    "member_id": int(zone_data["member_id"]),
                    "loss_fraction": loss,
                    "stiffness_scale": 1.0 - loss,
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


def _mapped_output(
    output_path: Path,
    mappings: list[ChannelMapping],
) -> tuple[object, pd.DataFrame]:
    output = read_openfast_output(output_path)
    batch = map_openfast_channels(output, mappings)
    observations = pd.DataFrame(batch.values, columns=batch.channel_names)
    observations.insert(0, "time_s", batch.time_s)
    return output, observations


def _execute_solver(
    executable: Path,
    case_directory: Path,
    primary_name: str,
    timeout_s: float,
) -> tuple[str, int | None, Path, float, str, str]:
    output = case_directory / Path(primary_name).with_suffix(".outb").name
    for suffix in (".out", ".outb", ".sum", ".ech"):
        candidate = case_directory / Path(primary_name).with_suffix(suffix).name
        if candidate.is_file():
            candidate.unlink()
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(executable), primary_name],
            cwd=case_directory,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except OSError as error:
        elapsed = time.perf_counter() - started
        return "launch_failed", None, output, elapsed, "", str(error)
    except subprocess.TimeoutExpired as error:
        elapsed = time.perf_counter() - started
        return "execution_timeout", None, output, elapsed, "", str(error)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        return (
            "solver_failed",
            completed.returncode,
            output,
            elapsed,
            completed.stdout,
            completed.stderr,
        )
    if not output.is_file():
        return (
            "output_missing",
            completed.returncode,
            output,
            elapsed,
            completed.stdout,
            completed.stderr,
        )
    return (
        "completed",
        completed.returncode,
        output,
        elapsed,
        completed.stdout,
        completed.stderr,
    )


def _spectral_features(
    values: np.ndarray,
    time_step_s: float,
    segment_samples: int,
) -> tuple[float, float, float]:
    frequencies, density = welch(
        values,
        fs=1.0 / time_step_s,
        window="hann",
        nperseg=min(segment_samples, values.size),
        noverlap=min(segment_samples // 2, values.size // 2),
        detrend="constant",
        scaling="density",
    )
    band = (frequencies > 0.0) & (frequencies <= 2.0)
    band_frequency = frequencies[band]
    band_density = density[band]
    dominant_frequency = float(band_frequency[np.argmax(band_density)])
    band_energy = float(np.trapezoid(band_density, band_frequency))
    spectral_centroid = float(
        np.sum(band_frequency * band_density) / np.sum(band_density)
    )
    return dominant_frequency, band_energy, spectral_centroid


def _response_metrics(
    cases: list[dict[str, object]],
    case_values: dict[str, pd.DataFrame],
    control_id: str,
    channel_names: tuple[str, ...],
    analysis: dict[str, object],
) -> pd.DataFrame:
    control = case_values[control_id]
    start_time = float(analysis["start_time_s"])
    mask = control["time_s"].to_numpy() >= start_time
    time_step = float(np.median(np.diff(control["time_s"].to_numpy())))
    rows: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        values_frame = case_values[case_id]
        for channel in channel_names:
            reference = control.loc[mask, channel].to_numpy(dtype=float)
            values = values_frame.loc[mask, channel].to_numpy(dtype=float)
            difference = values - reference
            reference_rms = float(np.sqrt(np.mean(reference**2)))
            rms = float(np.sqrt(np.mean(values**2)))
            difference_rms = float(np.sqrt(np.mean(difference**2)))
            correlation = 1.0 if case_id == control_id else float(
                np.corrcoef(reference, values)[0, 1]
            )
            dominant, band_energy, centroid = _spectral_features(
                values,
                time_step,
                int(analysis["welch_segment_samples"]),
            )
            ref_dominant, ref_energy, ref_centroid = _spectral_features(
                reference,
                time_step,
                int(analysis["welch_segment_samples"]),
            )
            rows.append(
                {
                    **case,
                    "channel": channel,
                    "analysis_start_time_s": start_time,
                    "sample_count": int(values.size),
                    "mean_m_per_s2": float(np.mean(values)),
                    "standard_deviation_m_per_s2": float(np.std(values, ddof=1)),
                    "rms_m_per_s2": rms,
                    "reference_rms_m_per_s2": reference_rms,
                    "rms_ratio": rms / reference_rms,
                    "difference_rms_m_per_s2": difference_rms,
                    "nrmse": difference_rms / reference_rms,
                    "maximum_absolute_difference_m_per_s2": float(
                        np.max(np.abs(difference))
                    ),
                    "correlation_with_control": correlation,
                    "dominant_frequency_hz": dominant,
                    "dominant_frequency_shift_hz": dominant - ref_dominant,
                    "band_energy_0_to_2hz": band_energy,
                    "band_energy_ratio": band_energy / ref_energy,
                    "spectral_centroid_hz": centroid,
                    "spectral_centroid_shift_hz": centroid - ref_centroid,
                }
            )
    return pd.DataFrame(rows)


def _window_metrics(
    cases: list[dict[str, object]],
    case_values: dict[str, pd.DataFrame],
    control_id: str,
    channel_names: tuple[str, ...],
    analysis: dict[str, object],
) -> pd.DataFrame:
    control = case_values[control_id]
    start_time = float(analysis["start_time_s"])
    duration = float(analysis["window_duration_s"])
    final_time = float(control["time_s"].iloc[-1])
    window_starts = np.arange(start_time, final_time, duration)
    rows: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        values_frame = case_values[case_id]
        for window_index, window_start in enumerate(window_starts, start=1):
            window_end = window_start + duration
            mask = (control["time_s"] >= window_start) & (
                control["time_s"] < window_end
            )
            for channel in channel_names:
                reference = control.loc[mask, channel].to_numpy(dtype=float)
                values = values_frame.loc[mask, channel].to_numpy(dtype=float)
                difference_rms = float(
                    np.sqrt(np.mean((values - reference) ** 2))
                )
                reference_rms = float(np.sqrt(np.mean(reference**2)))
                rows.append(
                    {
                        **case,
                        "window_index": window_index,
                        "window_start_s": float(window_start),
                        "window_end_s": float(window_end),
                        "channel": channel,
                        "sample_count": int(values.size),
                        "rms_m_per_s2": float(np.sqrt(np.mean(values**2))),
                        "reference_rms_m_per_s2": reference_rms,
                        "nrmse": difference_rms / reference_rms,
                    }
                )
    return pd.DataFrame(rows)


def _hypothesis_checks(
    cases: list[dict[str, object]],
    case_values: dict[str, pd.DataFrame],
    healthy_values: pd.DataFrame,
    manifest: pd.DataFrame,
    response: pd.DataFrame,
    channel_names: tuple[str, ...],
    configuration: dict[str, object],
) -> pd.DataFrame:
    parameterization = dict(configuration["parameterization"])
    analysis = dict(configuration["analysis"])
    control_id = str(parameterization["control_case_id"])
    control = case_values[control_id]
    healthy_error = float(
        np.max(
            np.abs(
                control[list(channel_names)].to_numpy()
                - healthy_values[list(channel_names)].to_numpy()
            )
        )
    )
    rows = [
        _check(
            "scientific_hypothesis",
            "H19_1_all_cases_execute_and_match_contract",
            bool(manifest["output_contract_passed"].all()),
            "==",
            True,
        ),
        _check(
            "scientific_hypothesis",
            "H19_2_zero_loss_patch_reproduces_stage18",
            healthy_error,
            "<=",
            float(analysis["control_max_abs_error_m_per_s2"]),
        ),
    ]
    damaged = response[response["loss_fraction"] > 0.0]
    rows.append(
        _check(
            "scientific_hypothesis",
            "H19_3_each_damage_case_reaches_both_sensors",
            float(damaged["nrmse"].min()),
            ">=",
            float(analysis["transmitted_change_min_nrmse"]),
        )
    )
    for zone in parameterization["zones"]:
        zone_id = str(zone["zone_id"])
        for channel in channel_names:
            selected = response[
                (response["zone_id"] == zone_id)
                & (response["channel"] == channel)
            ].sort_values("loss_fraction")
            increasing = bool(np.all(np.diff(selected["nrmse"].to_numpy()) > 0.0))
            rows.append(
                _check(
                    "scientific_hypothesis",
                    f"H19_4_monotonic_severity_{zone_id}_{channel}",
                    increasing,
                    "==",
                    True,
                )
            )

    start_time = float(analysis["start_time_s"])
    mask = control["time_s"].to_numpy() >= start_time
    severe_cases = [
        case
        for case in cases
        if np.isclose(float(case["loss_fraction"]), 0.30)
    ]
    signatures: dict[str, np.ndarray] = {}
    control_values = control.loc[mask, list(channel_names)].to_numpy()
    for case in severe_cases:
        case_id = str(case["case_id"])
        values = case_values[case_id].loc[mask, list(channel_names)].to_numpy()
        signatures[str(case["zone_id"])] = (values - control_values).ravel()
    distances = []
    for first, second in itertools.combinations(signatures, 2):
        difference = np.linalg.norm(signatures[first] - signatures[second])
        denominator = max(
            np.linalg.norm(signatures[first]),
            np.linalg.norm(signatures[second]),
        )
        distances.append(float(difference / denominator))
    rows.append(
        _check(
            "scientific_hypothesis",
            "H19_5_severe_zone_signatures_are_distinct",
            min(distances),
            ">=",
            float(analysis["severe_signature_min_relative_distance"]),
        )
    )
    return pd.DataFrame(rows)


def _create_figure(
    observations: pd.DataFrame,
    response: pd.DataFrame,
    configuration: dict[str, object],
) -> None:
    parameterization = dict(configuration["parameterization"])
    analysis = dict(configuration["analysis"])
    control_id = str(parameterization["control_case_id"])
    channel = str(
        dict(configuration["output_contract"])["required_channels"][0][
            "target_name"
        ]
    )
    severe = response[np.isclose(response["loss_fraction"], 0.30)]
    severe_ids = severe["case_id"].drop_duplicates().tolist()
    colors = {
        "lower_monopile": "#D55E00",
        "middle_monopile": "#0072B2",
        "upper_monopile": "#009E73",
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    trace_axis, spectral_axis, trend_axis, heat_axis = axes.ravel()
    start = float(analysis["start_time_s"])
    trace_mask = (observations["time_s"] >= start) & (
        observations["time_s"] <= start + 10.0
    )
    control_trace = observations[
        (observations["case_id"] == control_id) & trace_mask
    ]
    trace_axis.plot(
        control_trace["time_s"],
        control_trace[channel],
        color="black",
        linewidth=1.4,
        label="control",
    )
    for case_id in severe_ids:
        selected = observations[(observations["case_id"] == case_id) & trace_mask]
        zone_id = str(selected["zone_id"].iloc[0])
        trace_axis.plot(
            selected["time_s"],
            selected[channel],
            color=colors[zone_id],
            linewidth=0.9,
            alpha=0.85,
            label=f"{zone_id.replace('_', ' ')}: 30%",
        )
    trace_axis.set_title("A. Direct nacelle fore-aft acceleration")
    trace_axis.set_xlabel("Time (s)")
    trace_axis.set_ylabel("Acceleration (m/s²)")
    trace_axis.legend(fontsize=8, ncol=2)

    spectral_mask = observations["time_s"] >= start
    for case_id in [control_id, *severe_ids]:
        selected = observations[
            (observations["case_id"] == case_id) & spectral_mask
        ]
        zone_id = str(selected["zone_id"].iloc[0])
        frequencies, density = welch(
            selected[channel].to_numpy(),
            fs=20.0,
            nperseg=int(analysis["welch_segment_samples"]),
            noverlap=int(analysis["welch_segment_samples"]) // 2,
            detrend="constant",
        )
        keep = (frequencies > 0.0) & (frequencies <= 2.0)
        color = "black" if case_id == control_id else colors[zone_id]
        label = "control" if case_id == control_id else zone_id.replace("_", " ")
        spectral_axis.semilogy(
            frequencies[keep], density[keep], color=color, linewidth=1.5, label=label
        )
    spectral_axis.set_title("B. Post-transient acceleration spectra")
    spectral_axis.set_xlabel("Frequency (Hz)")
    spectral_axis.set_ylabel("PSD ((m/s²)²/Hz)")
    spectral_axis.legend(fontsize=8)

    for zone in parameterization["zones"]:
        zone_id = str(zone["zone_id"])
        for channel_index, sensor_channel in enumerate(
            response["channel"].drop_duplicates()
        ):
            selected = response[
                (response["zone_id"] == zone_id)
                & (response["channel"] == sensor_channel)
            ].sort_values("loss_fraction")
            axis_label = "FA" if channel_index == 0 else "SS"
            trend_axis.plot(
                100.0 * selected["loss_fraction"],
                selected["nrmse"],
                color=colors[zone_id],
                linestyle="-" if channel_index == 0 else "--",
                marker="o" if channel_index == 0 else "s",
                label=f"{zone_id.replace('_', ' ')} / {axis_label}",
            )
    trend_axis.set_title("C. Damage signature versus severity")
    trend_axis.set_xlabel("Local stiffness loss (%)")
    trend_axis.set_ylabel("Acceleration NRMSE vs control")
    trend_axis.legend(fontsize=7, ncol=2)

    channel_names = response["channel"].drop_duplicates().tolist()
    zone_ids = [str(zone["zone_id"]) for zone in parameterization["zones"]]
    matrix = np.zeros((len(zone_ids), len(channel_names)))
    for row_index, zone_id in enumerate(zone_ids):
        for column_index, sensor_channel in enumerate(channel_names):
            row = severe[
                (severe["zone_id"] == zone_id)
                & (severe["channel"] == sensor_channel)
            ]
            matrix[row_index, column_index] = float(row["nrmse"].iloc[0])
    image = heat_axis.imshow(matrix, cmap="viridis", aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            text_color = (
                "white"
                if matrix[row_index, column_index] > matrix.max() / 2
                else "black"
            )
            heat_axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )
    heat_axis.set_xticks(range(len(channel_names)), ["Fore-aft", "Side-side"])
    heat_axis.set_yticks(
        range(len(zone_ids)), [zone.replace("_", " ") for zone in zone_ids]
    )
    heat_axis.set_title("D. NRMSE signature at 30% stiffness loss")
    figure.colorbar(image, ax=heat_axis, label="NRMSE", shrink=0.82)
    figure.suptitle(
        "Stage 19: controlled OpenFAST monopile stiffness deterioration",
        fontsize=16,
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> int:
    configuration = _load_configuration()
    upstream = dict(configuration["upstream"])
    solver = dict(configuration["solver"])
    parameterization = dict(configuration["parameterization"])
    contract = dict(configuration["output_contract"])
    analysis = dict(configuration["analysis"])
    curated_root = PROJECT_ROOT / str(upstream["curated_deck_root"])
    primary_relative = Path(str(upstream["primary_file"]))
    subdyn_relative = Path(str(upstream["subdyn_file"]))
    healthy_output_path = PROJECT_ROOT / str(upstream["healthy_output"])
    healthy_execution_path = PROJECT_ROOT / str(
        upstream["healthy_execution_record"]
    )
    executable = PROJECT_ROOT / str(solver["executable_path"])
    run_root = PROJECT_ROOT / str(solver["run_root"])
    mappings = _mappings(configuration)
    channel_names = tuple(mapping.target_name for mapping in mappings)

    healthy_execution = pd.read_csv(healthy_execution_path).iloc[0]
    healthy_gate_passed = bool(
        healthy_execution["execution_status"] == "completed"
        and healthy_execution["generated_output_verified"]
    )
    executable_verified = bool(
        executable.is_file()
        and _sha256(executable) == str(solver["executable_sha256"])
    )
    if not healthy_gate_passed:
        raise RuntimeError("Stage 18 healthy execution gate is not complete.")
    if not executable_verified:
        raise RuntimeError("The pinned OpenFAST executable is unavailable or changed.")

    healthy_output, healthy_values = _mapped_output(healthy_output_path, mappings)
    cases = _case_definitions(configuration)
    clone_ids = {
        int(member_id): int(property_id)
        for member_id, property_id in dict(
            parameterization["clone_property_ids"]
        ).items()
    }
    all_member_ids = sorted(clone_ids)
    case_values: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, object]] = []
    patch_rows: list[dict[str, object]] = []
    observation_frames: list[pd.DataFrame] = []
    CASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("STAGE 19 - CONTROLLED OPENFAST STIFFNESS DETERIORATION", flush=True)
    print(
        f"Cases: {len(cases)} (control + 3 zones x 3 severities)",
        flush=True,
    )
    for case_index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        target_member = int(case["member_id"])
        scales = {member_id: 1.0 for member_id in all_member_ids}
        if target_member > 0:
            scales[target_member] = float(case["stiffness_scale"])
        case_root = run_root / case_id
        shutil.copytree(curated_root, case_root, dirs_exist_ok=True)
        subdyn_path = case_root / subdyn_relative
        patches = patch_subdyn_member_stiffness(subdyn_path, scales, clone_ids)
        closure = resolve_openfast_dependencies(case_root, primary_relative)
        case_directory = case_root / primary_relative.parent
        print(
            f"[{case_index:02d}/{len(cases):02d}] {case_id}: running OpenFAST",
            flush=True,
        )
        status, return_code, run_output, runtime_s, stdout, stderr = _execute_solver(
            executable,
            case_directory,
            primary_relative.name,
            float(solver["timeout_s"]),
        )
        (LOG_DIR / f"{case_id}.stdout.txt").write_text(stdout, encoding="utf-8")
        (LOG_DIR / f"{case_id}.stderr.txt").write_text(stderr, encoding="utf-8")
        if status != "completed":
            raise RuntimeError(f"OpenFAST case {case_id} failed: {status}")

        destination = CASE_OUTPUT_DIR / f"{case_id}.outb"
        shutil.copyfile(run_output, destination)
        output, values = _mapped_output(destination, mappings)
        contract_passed = bool(
            output.data.shape[0] == int(contract["expected_sample_count"])
            and len(output.channel_names) == int(contract["expected_channel_count"])
            and abs(output.time_step_s - float(contract["expected_time_step_s"]))
            <= 1.0e-12
            and uniform_time_grid_error(output.time_s) <= 1.0e-10
            and np.array_equal(output.time_s, healthy_output.time_s)
        )
        if not contract_passed:
            raise RuntimeError(f"OpenFAST case {case_id} violated the output contract.")

        values.insert(0, "loss_fraction", float(case["loss_fraction"]))
        values.insert(0, "member_id", target_member)
        values.insert(0, "zone_id", str(case["zone_id"]))
        values.insert(0, "case_id", case_id)
        case_values[case_id] = values
        observation_frames.append(values)
        manifest_rows.append(
            {
                **case,
                "dependency_file_count": len(closure.files),
                "unresolved_dependency_count": len(closure.unresolved),
                "subdyn_sha256": _sha256(subdyn_path),
                "execution_status": status,
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
            expected_scale = (
                float(case["stiffness_scale"]) if targeted else 1.0
            )
            patch_rows.append(
                {
                    "case_id": case_id,
                    "zone_id": case["zone_id"],
                    "declared_target_member_id": target_member,
                    "member_id": item.member_id,
                    "targeted": targeted,
                    "source_property_id": item.source_property_id,
                    "clone_property_id": item.clone_property_id,
                    "expected_stiffness_scale": expected_scale,
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
            f"[{case_index:02d}/{len(cases):02d}] {case_id}: PASS "
            f"({runtime_s:.1f} s, {output.data.shape[0]} samples)",
            flush=True,
        )

    observations = pd.concat(observation_frames, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)
    patches = pd.DataFrame(patch_rows)
    control_id = str(parameterization["control_case_id"])
    response = _response_metrics(
        cases,
        case_values,
        control_id,
        channel_names,
        analysis,
    )
    windows = _window_metrics(
        cases,
        case_values,
        control_id,
        channel_names,
        analysis,
    )
    hypotheses = _hypothesis_checks(
        cases,
        case_values,
        healthy_values,
        manifest,
        response,
        channel_names,
        configuration,
    )

    control_values = case_values[control_id][list(channel_names)].to_numpy()
    healthy_array = healthy_values[list(channel_names)].to_numpy()
    control_error = float(np.max(np.abs(control_values - healthy_array)))
    target_scale_correct = bool(
        np.allclose(
            patches["observed_stiffness_scale"],
            patches["expected_stiffness_scale"],
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    invariants_preserved = bool(
        (patches["material_density_kg_per_m3"] == 8500.0).all()
        and (patches["outer_diameter_m"] == 6.0).all()
        and (patches["wall_thickness_m"] == 0.06).all()
    )
    validation = pd.DataFrame(
        [
            _check(
                "prerequisite",
                "stage18_healthy_gate",
                healthy_gate_passed,
                "==",
                True,
            ),
            _check(
                "provenance",
                "solver_executable_hash",
                executable_verified,
                "==",
                True,
            ),
            _check("experiment", "case_count", len(manifest), "==", len(cases)),
            _check(
                "experiment",
                "all_solver_executions_completed",
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
                "deck",
                "dependency_count_each_case",
                int(manifest["dependency_file_count"].min()),
                "==",
                int(upstream["expected_dependency_count"]),
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
                "patch_row_count",
                len(patches),
                "==",
                len(cases) * len(all_member_ids),
            ),
            _check(
                "parameterization",
                "target_and_nontarget_scales_exact",
                target_scale_correct,
                "==",
                True,
            ),
            _check(
                "parameterization",
                "mass_and_geometry_preserved",
                invariants_preserved,
                "==",
                True,
            ),
            _check(
                "output",
                "combined_observation_rows",
                len(observations),
                "==",
                len(cases) * int(contract["expected_sample_count"]),
            ),
            _check(
                "output",
                "all_observations_finite",
                bool(
                    np.isfinite(observations[list(channel_names)].to_numpy()).all()
                ),
                "==",
                True,
            ),
            _check(
                "control",
                "zero_loss_max_abs_error_m_per_s2",
                control_error,
                "<=",
                float(analysis["control_max_abs_error_m_per_s2"]),
            ),
        ]
    )

    observations.to_csv(OBSERVATIONS_PATH, index=False, float_format="%.12g")
    manifest.to_csv(CASE_MANIFEST_PATH, index=False, float_format="%.12g")
    patches.to_csv(PATCH_PATH, index=False, float_format="%.12g")
    response.to_csv(RESPONSE_PATH, index=False, float_format="%.12g")
    windows.to_csv(WINDOW_PATH, index=False, float_format="%.12g")
    hypotheses.to_csv(HYPOTHESIS_PATH, index=False, float_format="%.12g")
    validation.to_csv(VALIDATION_PATH, index=False, float_format="%.12g")
    _create_figure(observations, response, configuration)

    print(f"Core validation: {int(validation['supported'].sum())}/{len(validation)}")
    print(
        f"Scientific hypotheses: "
        f"{int(hypotheses['supported'].sum())}/{len(hypotheses)}"
    )
    print(f"Control reproduction max error: {control_error:.3e} m/s^2")
    print(f"Observations: {OBSERVATIONS_PATH}")
    print(f"Metrics: {RESPONSE_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    if bool(validation["supported"].all()):
        print("RESULT: PASS (controlled OpenFAST deterioration study complete)")
        return 0
    print("RESULT: FAIL (Stage 19 core validation failed)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
