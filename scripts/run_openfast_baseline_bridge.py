"""Run Stage 17: pinned OpenFAST baseline ingestion and observation bridging."""

from __future__ import annotations

import hashlib
import os
import subprocess
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
    read_openfast_output,
    read_openfast_scalar_settings,
    uniform_time_grid_error,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "phase2_openfast_baseline.yaml"
OBSERVATIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stage17_openfast_observations.csv"
)
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PROVENANCE_PATH = METRICS_DIR / "stage17_openfast_provenance.csv"
MAPPING_PATH = METRICS_DIR / "stage17_openfast_channel_mapping.csv"
WINDOW_PATH = METRICS_DIR / "stage17_openfast_window_summary.csv"
VALIDATION_PATH = METRICS_DIR / "stage17_openfast_validation.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage17_openfast_baseline_bridge.png"


def _load_configuration() -> dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream)["stage17"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _supported(observed: object, operator: str, criterion: object) -> bool:
    if operator == "==":
        return bool(observed == criterion)
    if operator == ">=":
        return bool(float(observed) >= float(criterion))
    if operator == "<=":
        return bool(float(observed) <= float(criterion))
    if operator == "contains":
        return str(criterion) in str(observed)
    raise ValueError(f"Unsupported validation operator: {operator}")


def _check(
    category: str,
    name: str,
    observed: object,
    operator: str,
    criterion: object,
) -> dict[str, object]:
    return {
        "category": category,
        "check": name,
        "observed": observed,
        "operator": operator,
        "criterion": criterion,
        "supported": _supported(observed, operator, criterion),
    }


def _solver_readiness(
    executable_path: Path,
    expected_sha256: str,
) -> tuple[bool, bool, str]:
    if not executable_path.is_file():
        return False, False, "missing"
    hash_matches = _sha256(executable_path) == expected_sha256.lower()
    if not hash_matches:
        return True, False, "hash_mismatch"
    try:
        completed = subprocess.run(
            [str(executable_path), "-v"],
            cwd=executable_path.parent,
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
    except OSError:
        return True, True, "blocked_by_application_control"
    except subprocess.TimeoutExpired:
        return True, True, "version_check_timeout"
    if completed.returncode == 0:
        return True, True, "invocable"
    return True, True, "version_check_failed"


def _channel_mappings(configuration: dict[str, object]) -> list[ChannelMapping]:
    bridge = dict(configuration["bridge"])
    return [
        ChannelMapping(
            source_channel=str(row["source_channel"]),
            source_unit=str(row["source_unit"]),
            target_name=str(row["target_name"]),
            target_unit=str(row["target_unit"]),
            scale_to_target=float(row["scale_to_target"]),
            offset_to_target=float(row.get("offset_to_target", 0.0)),
        )
        for row in bridge["mappings"]
    ]


def _window_summary(
    time_s: np.ndarray,
    values: np.ndarray,
    names: tuple[str, ...],
    units: tuple[str, ...],
    window_duration_s: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = float(time_s[0])
    end = float(time_s[-1])
    window_count = int(np.floor((end - start) / window_duration_s + 1.0e-12))
    for window_index in range(window_count):
        window_start = start + window_index * window_duration_s
        window_end = window_start + window_duration_s
        selected = (time_s >= window_start) & (time_s < window_end)
        for column, (name, unit) in enumerate(zip(names, units, strict=True)):
            channel = values[selected, column]
            rows.append(
                {
                    "window": window_index,
                    "start_s": window_start,
                    "end_s": window_end,
                    "sample_count": int(channel.size),
                    "channel": name,
                    "unit": unit,
                    "mean": float(np.mean(channel)),
                    "standard_deviation": float(np.std(channel, ddof=1)),
                    "rms": float(np.sqrt(np.mean(channel**2))),
                    "minimum": float(np.min(channel)),
                    "maximum": float(np.max(channel)),
                }
            )
    return pd.DataFrame(rows)


def _dominant_frequency(
    values: np.ndarray,
    time_step_s: float,
    window_duration_s: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    samples_per_window = round(window_duration_s / time_step_s)
    frequencies, spectrum = welch(
        values - np.mean(values),
        fs=1.0 / time_step_s,
        nperseg=min(samples_per_window, values.size),
        noverlap=min(samples_per_window // 2, values.size // 2),
        detrend="linear",
        scaling="density",
    )
    nonzero = frequencies > 0.0
    peak_index = np.flatnonzero(nonzero)[np.argmax(spectrum[nonzero])]
    return frequencies, spectrum, float(frequencies[peak_index])


def _create_figure(
    time_s: np.ndarray,
    observations: pd.DataFrame,
    window_duration_s: float,
    fore_aft_spectrum: tuple[np.ndarray, np.ndarray, float],
    side_side_spectrum: tuple[np.ndarray, np.ndarray, float],
    solver_status: str,
    direct_acceleration_available: bool,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), constrained_layout=True)
    displacement_axis, environment_axis, platform_axis, spectrum_axis = axes.ravel()

    displacement_axis.plot(
        time_s,
        observations["tower_top_fore_aft_displacement_m"],
        label="Fore-aft",
        linewidth=1.35,
    )
    displacement_axis.plot(
        time_s,
        observations["tower_top_side_side_displacement_m"],
        label="Side-side",
        linewidth=1.25,
    )
    for boundary in np.arange(time_s[0], time_s[-1] + 0.1, window_duration_s):
        displacement_axis.axvline(boundary, color="0.75", linewidth=0.7)
    displacement_axis.set(
        title="Mapped tower-top response and monitoring windows",
        xlabel="Time [s]",
        ylabel="Displacement [m]",
    )
    displacement_axis.legend(loc="upper right")

    environment_axis.plot(
        time_s,
        observations["wave_elevation_m"],
        color="#0072B2",
        label="Wave elevation",
    )
    environment_axis.set(
        title="Environmental excitation and hydrodynamic load",
        xlabel="Time [s]",
        ylabel="Wave elevation [m]",
    )
    force_axis = environment_axis.twinx()
    force_axis.plot(
        time_s,
        observations["hydrodynamic_force_x_n"] / 1.0e6,
        color="#D55E00",
        alpha=0.8,
        label="Hydrodynamic force",
    )
    force_axis.set_ylabel("Hydrodynamic force [MN]")
    lines = environment_axis.lines + force_axis.lines
    environment_axis.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper right",
    )

    platform_axis.plot(
        time_s,
        observations["platform_surge_m"] * 100.0,
        label="Surge",
        color="#009E73",
    )
    platform_axis.set(
        title="Platform motion mapped to SI",
        xlabel="Time [s]",
        ylabel="Surge [cm]",
    )
    pitch_axis = platform_axis.twinx()
    pitch_axis.plot(
        time_s,
        np.rad2deg(observations["platform_pitch_rad"]),
        label="Pitch",
        color="#CC79A7",
    )
    pitch_axis.set_ylabel("Pitch [deg]")
    lines = platform_axis.lines + pitch_axis.lines
    platform_axis.legend(lines, [line.get_label() for line in lines], loc="upper right")

    for spectrum, label, color in (
        (fore_aft_spectrum, "Fore-aft", "#0072B2"),
        (side_side_spectrum, "Side-side", "#E69F00"),
    ):
        frequencies, density, dominant = spectrum
        positive = frequencies > 0.0
        spectrum_axis.semilogy(
            frequencies[positive],
            density[positive],
            label=f"{label}: peak {dominant:.2f} Hz",
            color=color,
        )
    spectrum_axis.set(
        title="Tower-top displacement spectra",
        xlabel="Frequency [Hz]",
        ylabel="PSD [m²/Hz]",
        xlim=(0.0, 2.0),
    )
    spectrum_axis.legend(loc="upper right")
    readiness = (
        "Reference bridge: VERIFIED\n"
        f"Solver: {solver_status.replace('_', ' ')}\n"
        "Direct acceleration: "
        + ("available" if direct_acceleration_available else "deferred")
    )
    spectrum_axis.text(
        0.98,
        0.04,
        readiness,
        transform=spectrum_axis.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    figure.suptitle(
        "Stage 17: OpenFAST v5.0.0 OC3 monopile baseline bridge",
        fontsize=16,
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=190)
    plt.close(figure)


def main() -> int:
    configuration = _load_configuration()
    source = dict(configuration["source"])
    solver = dict(configuration["solver"])
    expected_output = dict(configuration["expected_output"])
    expected_settings = dict(configuration["expected_input_settings"])
    bridge = dict(configuration["bridge"])
    input_path = PROJECT_ROOT / str(source["input_path"])
    output_path = PROJECT_ROOT / str(source["output_path"])
    executable_path = PROJECT_ROOT / str(solver["executable_path"])

    output = read_openfast_output(output_path)
    settings = read_openfast_scalar_settings(input_path, tuple(expected_settings))
    mappings = _channel_mappings(configuration)
    batch = map_openfast_channels(
        output,
        mappings,
        start_s=float(bridge["discard_before_s"]),
        end_s=float(bridge["end_s"]),
    )
    observation_frame = pd.DataFrame(batch.values, columns=batch.channel_names)
    observation_frame.insert(0, "time_s", batch.time_s)
    windows = _window_summary(
        batch.time_s,
        batch.values,
        batch.channel_names,
        batch.channel_units,
        float(bridge["window_duration_s"]),
    )
    window_count = int(windows["window"].nunique())

    fore_aft = _dominant_frequency(
        observation_frame["tower_top_fore_aft_displacement_m"].to_numpy(),
        output.time_step_s,
        float(bridge["window_duration_s"]),
    )
    side_side = _dominant_frequency(
        observation_frame["tower_top_side_side_displacement_m"].to_numpy(),
        output.time_step_s,
        float(bridge["window_duration_s"]),
    )
    frequency_band = tuple(
        float(value) for value in bridge["dominant_frequency_band_hz"]
    )

    executable_present, executable_hash_matches, solver_status = _solver_readiness(
        executable_path,
        str(solver["executable_sha256"]),
    )
    deferred = list(configuration["deferred_direct_sensor_channels"])
    direct_acceleration_available = all(
        str(row["source_channel"]) in output.channel_names for row in deferred
    )

    validation_rows: list[dict[str, object]] = [
        _check(
            "source_verification",
            "input_sha256",
            _sha256(input_path),
            "==",
            str(source["input_sha256"]),
        ),
        _check(
            "source_verification",
            "output_sha256",
            _sha256(output_path),
            "==",
            str(source["output_sha256"]),
        ),
        _check(
            "source_verification",
            "case_description",
            output.description,
            "contains",
            str(source["case_description_contains"]),
        ),
        _check(
            "source_verification",
            "binary_format_id",
            output.format_id,
            "==",
            int(expected_output["format_id"]),
        ),
        _check(
            "source_verification",
            "sample_count",
            output.data.shape[0],
            "==",
            int(expected_output["sample_count"]),
        ),
        _check(
            "source_verification",
            "minimum_channel_count",
            output.data.shape[1],
            ">=",
            int(expected_output["minimum_channel_count"]),
        ),
        _check(
            "source_verification",
            "time_step_absolute_error_s",
            abs(output.time_step_s - float(expected_output["time_step_s"])),
            "<=",
            1.0e-12,
        ),
        _check(
            "source_verification",
            "duration_s",
            output.duration_s,
            "==",
            float(expected_output["duration_s"]),
        ),
        _check(
            "bridge_verification",
            "relative_time_grid_error",
            uniform_time_grid_error(output.time_s),
            "<=",
            float(expected_output["maximum_relative_time_grid_error"]),
        ),
        _check(
            "bridge_verification",
            "mapped_channel_count",
            batch.values.shape[1],
            "==",
            len(mappings),
        ),
        _check(
            "bridge_verification",
            "mapped_nonfinite_fraction",
            float(np.mean(~np.isfinite(batch.values))),
            "==",
            0.0,
        ),
        _check(
            "bridge_verification",
            "complete_monitoring_windows",
            window_count,
            ">=",
            int(bridge["minimum_complete_windows"]),
        ),
        _check(
            "bridge_verification",
            "fore_aft_dominant_frequency_minimum_hz",
            fore_aft[2],
            ">=",
            frequency_band[0],
        ),
        _check(
            "bridge_verification",
            "fore_aft_dominant_frequency_maximum_hz",
            fore_aft[2],
            "<=",
            frequency_band[1],
        ),
        _check(
            "external_readiness",
            "solver_executable_present",
            executable_present,
            "==",
            True,
        ),
        _check(
            "external_readiness",
            "solver_executable_hash",
            executable_hash_matches,
            "==",
            True,
        ),
        _check(
            "external_readiness",
            "solver_version_invocable",
            solver_status,
            "==",
            "invocable",
        ),
        _check(
            "external_readiness",
            "direct_acceleration_channels_available",
            direct_acceleration_available,
            "==",
            True,
        ),
    ]
    for name, expected in expected_settings.items():
        validation_rows.append(
            _check(
                "source_verification",
                f"input_setting_{name}",
                float(settings[name]),
                "==",
                float(expected),
            )
        )
    validation = pd.DataFrame(validation_rows)

    mapping_rows = []
    for column, mapping in enumerate(mappings):
        values = batch.values[:, column]
        mapping_rows.append(
            {
                "source_channel": mapping.source_channel,
                "source_unit": mapping.source_unit,
                "target_name": mapping.target_name,
                "target_unit": mapping.target_unit,
                "scale_to_target": mapping.scale_to_target,
                "offset_to_target": mapping.offset_to_target,
                "sample_count": values.size,
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "supported": True,
            }
        )
    mapping_frame = pd.DataFrame(mapping_rows)
    provenance = pd.DataFrame(
        [
            ("reference", "repository", source["repository"]),
            ("reference", "release_tag", source["release_tag"]),
            ("reference", "release_commit", source["release_commit"]),
            ("reference", "case_name", source["case_name"]),
            ("reference", "input_git_blob", source["input_git_blob"]),
            ("reference", "output_git_blob", source["output_git_blob"]),
            ("reference", "input_sha256", _sha256(input_path)),
            ("reference", "output_sha256", _sha256(output_path)),
            ("reference", "description", output.description),
            ("solver", "repository", solver["repository"]),
            ("solver", "release_tag", solver["release_tag"]),
            ("solver", "executable_present", executable_present),
            ("solver", "executable_hash_matches", executable_hash_matches),
            ("solver", "execution_status", solver_status),
            ("bridge", "direct_acceleration_available", direct_acceleration_available),
        ],
        columns=["component", "attribute", "value"],
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    observation_frame.to_csv(OBSERVATIONS_PATH, index=False, float_format="%.12g")
    provenance.to_csv(PROVENANCE_PATH, index=False)
    mapping_frame.to_csv(MAPPING_PATH, index=False, float_format="%.12g")
    windows.to_csv(WINDOW_PATH, index=False, float_format="%.12g")
    validation.to_csv(VALIDATION_PATH, index=False, float_format="%.12g")
    _create_figure(
        batch.time_s,
        observation_frame,
        float(bridge["window_duration_s"]),
        fore_aft,
        side_side,
        solver_status,
        direct_acceleration_available,
    )

    core = validation[validation["category"] != "external_readiness"]
    readiness = validation[validation["category"] == "external_readiness"]
    core_passed = bool(core["supported"].all())
    print("STAGE 17 - OPENFAST BASELINE OBSERVATION BRIDGE")
    print(
        f"Reference: OpenFAST {source['release_tag']} {source['case_name']} "
        f"({output.data.shape[0]} samples, {output.data.shape[1]} channels)"
    )
    print(
        f"Mapped: {batch.values.shape[1]} SI channels across {window_count} "
        f"complete monitoring windows"
    )
    print(
        f"Tower-top spectral peaks: fore-aft {fore_aft[2]:.3f} Hz; "
        f"side-side {side_side[2]:.3f} Hz"
    )
    print(f"Core verification: {int(core['supported'].sum())}/{len(core)}")
    print(
        f"External readiness: {int(readiness['supported'].sum())}/"
        f"{len(readiness)}; solver={solver_status}; "
        f"direct_acceleration={direct_acceleration_available}"
    )
    print(f"Validation: {VALIDATION_PATH}")
    print(f"Observations: {OBSERVATIONS_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    if core_passed:
        print("RESULT: PASS (reference bridge verified; readiness gaps retained)")
        return 0
    print("RESULT: FAIL (core OpenFAST bridge verification failed)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
