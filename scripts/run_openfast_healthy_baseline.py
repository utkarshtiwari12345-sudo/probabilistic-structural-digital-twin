"""Run Stage 18: prepare and attempt a sensor-matched healthy OpenFAST deck."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
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
    ChannelMapping,
    copy_openfast_dependency_closure,
    ensure_openfast_outlist_channels,
    map_openfast_channels,
    read_openfast_outlist_channels,
    read_openfast_output,
    read_openfast_scalar_settings,
    resolve_openfast_dependencies,
    uniform_time_grid_error,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "phase2_openfast_healthy.yaml"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
MANIFEST_PATH = METRICS_DIR / "stage18_openfast_deck_manifest.csv"
CONTRACT_PATH = METRICS_DIR / "stage18_openfast_sensor_contract.csv"
EXECUTION_PATH = METRICS_DIR / "stage18_openfast_execution.csv"
VALIDATION_PATH = METRICS_DIR / "stage18_openfast_validation.csv"
OBSERVATIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stage18_openfast_healthy_observations.csv"
)
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage18_openfast_healthy_readiness.png"


def _load_configuration() -> dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream)["stage18"])


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


def _prepare_curated_deck(
    configuration: dict[str, object],
) -> tuple[Path, Path, tuple[Path, ...], int]:
    source = dict(configuration["source"])
    controller = dict(configuration["controller"])
    deck = dict(configuration["deck"])
    source_root = PROJECT_ROOT / str(source["source_root"])
    curated_root = PROJECT_ROOT / str(deck["curated_root"])
    primary_relative = Path(str(source["primary_file"]))
    elastodyn_relative = Path(str(source["elastodyn_file"]))
    controller_source = PROJECT_ROOT / str(controller["release_asset"])
    controller_relative = Path(str(controller["target_file"]))

    modified_upstream_files = -1
    if source_root.is_dir():
        source_closure = resolve_openfast_dependencies(
            source_root,
            primary_relative,
            allowed_missing_names=source["allowed_external_dependencies"],
        )
        if len(source_closure.files) != int(
            source["expected_dependency_count_without_controller"]
        ):
            raise ValueError("Official OpenFAST source dependency count changed.")
        if _sha256(source_root / primary_relative) != str(source["primary_sha256"]):
            raise ValueError("Official OpenFAST primary input hash changed.")
        if _sha256(source_root / elastodyn_relative) != str(
            source["elastodyn_sha256"]
        ):
            raise ValueError("Official ElastoDyn input hash changed.")
        copy_openfast_dependency_closure(source_closure, curated_root)
        if not controller_source.is_file():
            raise ValueError("Pinned OpenFAST controller asset is unavailable.")
        controller_target = curated_root / controller_relative
        controller_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(controller_source, controller_target)
        ensure_openfast_outlist_channels(
            curated_root / elastodyn_relative,
            tuple(deck["requested_elastodyn_channels"]),
        )
        modified_upstream_files = sum(
            _sha256(source_root / relative) != _sha256(curated_root / relative)
            for relative in source_closure.files
        )
    elif not (curated_root / primary_relative).is_file():
        raise ValueError(
            "Neither the official source checkout nor the curated Stage 18 deck exists."
        )
    else:
        modified_upstream_files = int(
            _sha256(curated_root / elastodyn_relative)
            != str(source["elastodyn_sha256"])
        )

    curated_closure = resolve_openfast_dependencies(curated_root, primary_relative)
    return (
        curated_root,
        primary_relative,
        curated_closure.files,
        modified_upstream_files,
    )


def _execute_solver(
    executable: Path,
    run_root: Path,
    primary_relative: Path,
    timeout_s: float,
) -> tuple[str, int | None, Path]:
    case_directory = run_root / primary_relative.parent
    generated_output = case_directory / primary_relative.with_suffix(".outb").name
    for suffix in (".out", ".outb", ".sum", ".ech"):
        candidate = case_directory / primary_relative.with_suffix(suffix).name
        if candidate.is_file():
            candidate.unlink()
    try:
        completed = subprocess.run(
            [str(executable), primary_relative.name],
            cwd=case_directory,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except OSError:
        return "blocked_by_application_control", None, generated_output
    except subprocess.TimeoutExpired:
        return "execution_timeout", None, generated_output
    if completed.returncode != 0:
        return "solver_failed", completed.returncode, generated_output
    if not generated_output.is_file():
        return "output_missing", completed.returncode, generated_output
    return "completed", completed.returncode, generated_output


def _manifest(
    root: Path,
    files: tuple[Path, ...],
    configuration: dict[str, object],
) -> pd.DataFrame:
    source = dict(configuration["source"])
    controller = dict(configuration["controller"])
    rows = []
    for relative in files:
        path = root / relative
        if relative == Path(str(source["primary_file"])):
            role = "primary_input"
        elif relative == Path(str(source["elastodyn_file"])):
            role = "sensor_patched_input"
        elif relative == Path(str(controller["target_file"])):
            role = "release_controller"
        else:
            role = "dependency"
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "role": role,
                "suffix": relative.suffix.lower() or "none",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows)


def _create_figure(
    manifest: pd.DataFrame,
    validation: pd.DataFrame,
    contract: pd.DataFrame,
    execution_status: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)
    count_axis, size_axis, gate_axis, contract_axis = axes.ravel()

    suffix_summary = manifest.groupby("suffix", sort=True).agg(
        files=("relative_path", "count"),
        size_bytes=("size_bytes", "sum"),
    )
    count_axis.bar(
        suffix_summary.index,
        suffix_summary["files"],
        color="#0072B2",
    )
    count_axis.set(
        title="Hermetic deck dependency composition",
        xlabel="File type",
        ylabel="Files",
    )

    size_order = suffix_summary.sort_values("size_bytes", ascending=True)
    size_axis.barh(
        size_order.index,
        size_order["size_bytes"] / 1.0e6,
        color="#009E73",
    )
    size_axis.set(
        title="Curated deck size by file type",
        xlabel="Size [MB]",
        ylabel="File type",
    )

    grouped = validation.groupby("category")["supported"].agg(["sum", "count"])
    fractions = grouped["sum"] / grouped["count"]
    colors = ["#009E73" if value == 1.0 else "#D55E00" for value in fractions]
    gate_axis.barh(fractions.index, fractions, color=colors)
    gate_axis.set(
        title="Verification and execution gates",
        xlabel="Supported fraction",
        xlim=(0.0, 1.05),
    )
    for index, (supported, total) in enumerate(
        zip(grouped["sum"], grouped["count"], strict=True)
    ):
        gate_axis.text(
            min(float(fractions.iloc[index]) + 0.02, 0.96),
            index,
            f"{int(supported)}/{int(total)}",
            va="center",
        )

    contract_axis.axis("off")
    contract_lines = [
        "Direct sensor contract",
        "",
        *[
            f"OK  {row.source_channel}: requested in ElastoDyn OutList"
            for row in contract.itertuples()
        ],
        "",
        f"Solver status: {execution_status.replace('_', ' ')}",
        "Generated channel verification: "
        + ("complete" if contract["generated_output_verified"].all() else "pending"),
        "",
        "No deterioration or Phase I covariance calibration applied.",
    ]
    contract_axis.text(
        0.04,
        0.96,
        "\n".join(contract_lines),
        va="top",
        fontsize=11.5,
        linespacing=1.45,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    contract_axis.set_title("Sensor-matched execution contract")
    figure.suptitle(
        "Stage 18: healthy OpenFAST deck and execution readiness",
        fontsize=16,
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=190)
    plt.close(figure)


def main() -> int:
    configuration = _load_configuration()
    source = dict(configuration["source"])
    controller = dict(configuration["controller"])
    deck = dict(configuration["deck"])
    solver = dict(configuration["solver"])
    output_contract = dict(configuration["output_contract"])
    curated_root, primary_relative, curated_files, modified_count = (
        _prepare_curated_deck(configuration)
    )
    manifest = _manifest(curated_root, curated_files, configuration)
    primary_path = curated_root / primary_relative
    elastodyn_path = curated_root / str(source["elastodyn_file"])
    controller_path = curated_root / str(controller["target_file"])
    settings = read_openfast_scalar_settings(
        primary_path,
        tuple(dict(deck["expected_input_settings"])),
    )
    outlist_channels = read_openfast_outlist_channels(elastodyn_path)
    requested_channels = tuple(deck["requested_elastodyn_channels"])

    run_root = PROJECT_ROOT / str(deck["run_root"])
    run_closure = resolve_openfast_dependencies(curated_root, primary_relative)
    copy_openfast_dependency_closure(run_closure, run_root)
    executable = PROJECT_ROOT / str(solver["executable_path"])
    executable_present = executable.is_file()
    executable_hash_matches = executable_present and (
        _sha256(executable) == str(solver["executable_sha256"])
    )
    if executable_hash_matches:
        execution_status, return_code, generated_run_output = _execute_solver(
            executable,
            run_root,
            primary_relative,
            float(solver["timeout_s"]),
        )
    else:
        execution_status = "executable_missing_or_unverified"
        return_code = None
        generated_run_output = run_root / primary_relative.with_suffix(".outb")

    generated_output_verified = False
    generated_sample_count = 0
    generated_time_step_s = np.nan
    generated_channel_names: set[str] = set()
    mapped_observations = pd.DataFrame()
    if execution_status == "completed" and generated_run_output.is_file():
        destination = PROJECT_ROOT / str(output_contract["generated_output_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated_run_output, destination)
        generated = read_openfast_output(destination)
        generated_sample_count = generated.data.shape[0]
        generated_time_step_s = generated.time_step_s
        generated_channel_names = set(generated.channel_names)
        mappings = [
            ChannelMapping(
                source_channel=str(row["source_channel"]),
                source_unit=str(row["source_unit"]),
                target_name=str(row["target_name"]),
                target_unit=str(row["target_unit"]),
                scale_to_target=float(row["scale_to_target"]),
            )
            for row in output_contract["required_channels"]
        ]
        batch = map_openfast_channels(generated, mappings)
        mapped_observations = pd.DataFrame(batch.values, columns=batch.channel_names)
        mapped_observations.insert(0, "time_s", batch.time_s)
        generated_output_verified = (
            generated_sample_count == int(output_contract["expected_sample_count"])
            and abs(
                generated_time_step_s
                - float(output_contract["expected_time_step_s"])
            )
            <= 1.0e-12
            and uniform_time_grid_error(generated.time_s) <= 1.0e-10
        )
        mapped_observations.to_csv(
            OBSERVATIONS_PATH,
            index=False,
            float_format="%.12g",
        )

    contract_rows = []
    for row in output_contract["required_channels"]:
        channel = str(row["source_channel"])
        contract_rows.append(
            {
                "source_channel": channel,
                "expected_unit": row["source_unit"],
                "target_name": row["target_name"],
                "target_unit": row["target_unit"],
                "requested_in_outlist": channel in outlist_channels,
                "present_in_generated_output": channel in generated_channel_names,
                "generated_output_verified": generated_output_verified,
            }
        )
    contract = pd.DataFrame(contract_rows)

    core_rows: list[dict[str, object]] = [
        _check(
            "deck_verification",
            "primary_input_sha256_unchanged",
            _sha256(primary_path),
            "==",
            str(source["primary_sha256"]),
        ),
        _check(
            "deck_verification",
            "controller_sha256",
            _sha256(controller_path),
            "==",
            str(controller["sha256"]),
        ),
        _check(
            "deck_verification",
            "dependency_file_count",
            len(curated_files),
            "==",
            int(deck["expected_total_file_count"]),
        ),
        _check(
            "deck_verification",
            "unresolved_dependency_count",
            len(run_closure.unresolved),
            "==",
            0,
        ),
        _check(
            "deck_verification",
            "sensor_patched_input_count",
            int((manifest["role"] == "sensor_patched_input").sum()),
            "==",
            1,
        ),
        _check(
            "deck_verification",
            "modified_upstream_file_count",
            modified_count,
            "==",
            1,
        ),
        _check(
            "sensor_contract",
            "requested_channel_count",
            int(contract["requested_in_outlist"].sum()),
            "==",
            len(requested_channels),
        ),
    ]
    for name, expected in dict(deck["expected_input_settings"]).items():
        core_rows.append(
            _check(
                "deck_verification",
                f"input_setting_{name}",
                float(settings[name]),
                "==",
                float(expected),
            )
        )
    external_rows = [
        _check(
            "external_execution",
            "solver_executable_present",
            executable_present,
            "==",
            True,
        ),
        _check(
            "external_execution",
            "solver_executable_hash",
            executable_hash_matches,
            "==",
            True,
        ),
        _check(
            "external_execution",
            "solver_execution_status",
            execution_status,
            "==",
            "completed",
        ),
        _check(
            "external_execution",
            "solver_return_code",
            -1 if return_code is None else return_code,
            "==",
            0,
        ),
        _check(
            "external_execution",
            "generated_output_exists",
            generated_run_output.is_file(),
            "==",
            True,
        ),
        _check(
            "external_execution",
            "generated_required_channel_count",
            int(contract["present_in_generated_output"].sum()),
            "==",
            len(requested_channels),
        ),
        _check(
            "external_execution",
            "generated_output_contract",
            generated_output_verified,
            "==",
            True,
        ),
    ]
    validation = pd.DataFrame(core_rows + external_rows)
    execution = pd.DataFrame(
        [
            {
                "release_tag": solver["release_tag"],
                "executable_present": executable_present,
                "executable_hash_matches": executable_hash_matches,
                "execution_status": execution_status,
                "return_code": return_code,
                "generated_output_exists": generated_run_output.is_file(),
                "generated_sample_count": generated_sample_count,
                "generated_time_step_s": generated_time_step_s,
                "generated_output_verified": generated_output_verified,
            }
        ]
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    contract.to_csv(CONTRACT_PATH, index=False)
    execution.to_csv(EXECUTION_PATH, index=False, float_format="%.12g")
    validation.to_csv(VALIDATION_PATH, index=False, float_format="%.12g")
    _create_figure(manifest, validation, contract, execution_status)

    core = validation[validation["category"] != "external_execution"]
    external = validation[validation["category"] == "external_execution"]
    core_passed = bool(core["supported"].all())
    external_passed = bool(external["supported"].all())
    print("STAGE 18 - HEALTHY OPENFAST DECK AND EXECUTION READINESS")
    print(
        f"Curated deck: {len(curated_files)} files, "
        f"{manifest['size_bytes'].sum() / 1.0e6:.2f} MB, "
        f"unresolved dependencies={len(run_closure.unresolved)}"
    )
    print(
        f"Direct acceleration channels requested: "
        f"{int(contract['requested_in_outlist'].sum())}/{len(contract)}"
    )
    print(f"Core deck verification: {int(core['supported'].sum())}/{len(core)}")
    print(
        f"External execution: {int(external['supported'].sum())}/{len(external)}; "
        f"status={execution_status}"
    )
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Validation: {VALIDATION_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    if core_passed and external_passed:
        print("RESULT: PASS (healthy OpenFAST baseline executed and verified)")
        return 0
    if core_passed:
        print("RESULT: PASS (healthy deck verified; external execution incomplete)")
        return 0
    print("RESULT: FAIL (healthy OpenFAST deck verification failed)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
