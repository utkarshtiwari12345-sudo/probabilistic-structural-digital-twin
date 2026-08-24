"""Build and verify the Stage 26 compact research release candidate."""

from __future__ import annotations

import os
import subprocess
import tomllib
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

from probdt.diagnostics import audit_pipeline_artifacts  # noqa: E402
from probdt.release import (  # noqa: E402
    build_release_files,
    forbidden_release_paths,
    metadata_placeholders,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "release.yaml"
PIPELINE_PATH = PROJECT_ROOT / "config" / "phase1_pipeline.yaml"
RELEASE_DIR = PROJECT_ROOT / "results" / "release"
READINESS_PATH = RELEASE_DIR / "stage26_release_readiness.csv"
SUMMARY_PATH = RELEASE_DIR / "stage26_release_summary.csv"
MANIFEST_PATH = RELEASE_DIR / "stage26_release_manifest.csv"
CHECKSUM_PATH = RELEASE_DIR / "stage26_sha256sums.txt"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage26_release_readiness.png"
OUTPUT_EXCLUSIONS = {
    READINESS_PATH.relative_to(PROJECT_ROOT).as_posix(),
    SUMMARY_PATH.relative_to(PROJECT_ROOT).as_posix(),
    MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
    CHECKSUM_PATH.relative_to(PROJECT_ROOT).as_posix(),
    FIGURE_PATH.relative_to(PROJECT_ROOT).as_posix(),
}


def _load_configuration() -> dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream)["stage26"])


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
        raise ValueError(f"Unsupported release operator: {operator}")
    return {
        "category": category,
        "check": name,
        "observed": observed,
        "operator": operator,
        "criterion": criterion,
        "supported": bool(supported),
    }


def _git_release_candidates() -> tuple[str, ...]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    paths = tuple(
        value.decode("utf-8")
        for value in completed.stdout.split(b"\0")
        if value
    )
    return tuple(sorted(path for path in paths if path not in OUTPUT_EXCLUSIONS))


def _dependency_names(values: list[str]) -> set[str]:
    names = set()
    for value in values:
        name = value.split("[", maxsplit=1)[0]
        for marker in (">=", "<=", "==", "~=", ">", "<"):
            name = name.split(marker, maxsplit=1)[0]
        names.add(name.strip().casefold())
    return names


def _create_figure(
    manifest: pd.DataFrame,
    readiness: pd.DataFrame,
    summary_values: dict[str, object],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), constrained_layout=True)

    category = (
        manifest.groupby("category", as_index=False)["size_bytes"]
        .sum()
        .sort_values("size_bytes")
    )
    axes[0, 0].barh(
        category["category"].str.replace("_", " "),
        category["size_bytes"] / (1024.0**2),
        color="#0072B2",
    )
    axes[0, 0].set_xlabel("Release payload (MiB)")
    axes[0, 0].set_title("A. Compact repository composition")

    largest = manifest.nlargest(8, "size_bytes").sort_values("size_bytes")
    axes[0, 1].barh(
        largest["relative_path"].map(lambda value: Path(value).name),
        largest["size_bytes"] / (1024.0**2),
        color="#56B4E9",
    )
    axes[0, 1].axvline(50.0, color="#D55E00", linestyle="--")
    axes[0, 1].set_xlabel("File size (MiB)")
    axes[0, 1].set_title("B. Largest candidate files")

    gate_counts = readiness.groupby("category", as_index=False).agg(
        checks=("check", "count"),
        passed=("supported", "sum"),
    )
    positions = np.arange(len(gate_counts), dtype=np.float64)
    axes[1, 0].bar(
        positions,
        gate_counts["checks"],
        color="#D9D9D9",
        label="declared checks",
    )
    axes[1, 0].bar(
        positions,
        gate_counts["passed"],
        color="#009E73",
        label="passed",
    )
    axes[1, 0].set_xticks(
        positions,
        gate_counts["category"].str.replace("_", " "),
        rotation=25,
        ha="right",
    )
    axes[1, 0].set_ylabel("Check count")
    axes[1, 0].set_title("C. Release-readiness gates")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axis("off")
    lines = [
        "LOCAL RELEASE CANDIDATE",
        "",
        f"Version: {summary_values['version']}",
        f"Payload files: {summary_values['candidate_file_count']}",
        f"Payload size: {summary_values['candidate_total_mib']:.1f} MiB",
        f"Research figures: {summary_values['figure_count']}",
        f"Compact evidence tables: {summary_values['evidence_table_count']}",
        f"Verified upstream stages: {summary_values['upstream_stage_count']}",
        f"Readiness checks: {summary_values['passed_check_count']}/"
        f"{summary_values['check_count']}",
        "",
        "External publication remains manual:",
        "initial commit and GitHub remote;",
        "release tag and optional DOI.",
    ]
    axes[1, 1].text(
        0.04,
        0.96,
        "\n".join(lines),
        va="top",
        fontsize=10.5,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "#999999", "pad": 12},
    )
    axes[1, 1].set_title("D. Release boundary")

    figure.suptitle(
        "Stage 26: GitHub and research-release readiness",
        fontsize=16,
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> int:
    configuration = _load_configuration()
    limits = dict(configuration["github_limits"])
    required_paths = tuple(str(value) for value in configuration["required_paths"])
    required_evidence = tuple(
        str(value) for value in configuration["required_evidence"]
    )
    candidates = _git_release_candidates()
    candidate_set = set(candidates)
    release_files = build_release_files(PROJECT_ROOT, candidates)
    manifest = pd.DataFrame(
        [
            {
                "relative_path": value.relative_path,
                "category": value.category,
                "size_bytes": value.size_bytes,
                "sha256": value.sha256,
            }
            for value in release_files
        ]
    )

    missing_required = tuple(
        path for path in required_paths if not (PROJECT_ROOT / path).is_file()
    )
    missing_evidence = tuple(
        path for path in required_evidence if not (PROJECT_ROOT / path).is_file()
    )
    forbidden = forbidden_release_paths(
        candidates,
        forbidden_prefixes=tuple(
            str(value) for value in configuration["forbidden_candidate_prefixes"]
        ),
        forbidden_names=tuple(
            str(value) for value in configuration["forbidden_candidate_names"]
        ),
    )
    placeholder_paths = (
        "CITATION.cff",
        "LICENSE",
        "pyproject.toml",
        "README.md",
    )
    placeholders = metadata_placeholders(
        PROJECT_ROOT,
        placeholder_paths,
        tuple(str(value) for value in configuration["metadata_placeholder_tokens"]),
    )

    cff = dict(yaml.safe_load((PROJECT_ROOT / "CITATION.cff").read_text()))
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    project = dict(pyproject["project"])
    author = str(configuration["author"])
    cff_authors = tuple(
        " ".join(
            filter(
                None,
                (str(row.get("given-names", "")), str(row.get("family-names", ""))),
            )
        )
        for row in cff["authors"]
    )
    project_authors = tuple(str(row["name"]) for row in project["authors"])
    lock_lines = [
        line.strip()
        for line in (PROJECT_ROOT / "requirements-lock.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    lock_names = _dependency_names(lock_lines)
    runtime_names = _dependency_names(list(project["dependencies"]))
    development_names = _dependency_names(list(project["optional-dependencies"]["dev"]))
    ci_text = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    communication_text = (
        PROJECT_ROOT / "docs" / "research_communication.md"
    ).read_text(encoding="utf-8")

    with PIPELINE_PATH.open("r", encoding="utf-8") as stream:
        stages = list(yaml.safe_load(stream)["pipeline"]["stages"])
    upstream_stages = [stage for stage in stages if str(stage["id"]) != "26"]
    upstream_audit = audit_pipeline_artifacts(PROJECT_ROOT, upstream_stages)

    maximum_file_size = int(manifest["size_bytes"].max())
    total_size = int(manifest["size_bytes"].sum())
    warning_file_count = int(
        (manifest["size_bytes"] > int(limits["warning_file_size_bytes"])).sum()
    )
    figure_count = sum(path.startswith("figures/") for path in candidates)
    evidence_table_count = sum(
        path.startswith("results/metrics/") for path in candidates
    )
    processed_candidate_count = sum(
        path.startswith("data/processed/") for path in candidates
    )
    readiness_rows = [
        _check("metadata", "all_required_paths_exist", len(missing_required), "==", 0),
        _check(
            "metadata",
            "required_paths_in_release_candidate",
            all(path in candidate_set for path in required_paths),
            "==",
            True,
        ),
        _check(
            "evidence",
            "all_required_evidence_exists",
            len(missing_evidence),
            "==",
            0,
        ),
        _check(
            "evidence",
            "required_evidence_in_release_candidate",
            all(path in candidate_set for path in required_evidence),
            "==",
            True,
        ),
        _check("metadata", "metadata_placeholders", len(placeholders), "==", 0),
        _check(
            "metadata",
            "citation_version_matches_package",
            str(cff["version"]),
            "==",
            str(project["version"]),
        ),
        _check(
            "metadata",
            "citation_type_is_software",
            str(cff["type"]),
            "==",
            "software",
        ),
        _check(
            "metadata",
            "author_metadata_matches",
            author in cff_authors and author in project_authors,
            "==",
            True,
        ),
        _check(
            "metadata",
            "license_attribution_matches",
            author in (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8"),
            "==",
            True,
        ),
        _check(
            "environment",
            "lock_covers_runtime_dependencies",
            runtime_names <= lock_names,
            "==",
            True,
        ),
        _check(
            "environment",
            "lock_covers_development_dependencies",
            development_names <= lock_names,
            "==",
            True,
        ),
        _check(
            "automation",
            "ci_runs_lint_and_tests",
            "python -m ruff check ." in ci_text and "python -m pytest -q" in ci_text,
            "==",
            True,
        ),
        _check("privacy", "forbidden_candidate_paths", len(forbidden), "==", 0),
        _check(
            "privacy",
            "processed_data_excluded",
            processed_candidate_count,
            "==",
            0,
        ),
        _check(
            "size",
            "maximum_candidate_file_size_bytes",
            maximum_file_size,
            "<=",
            limits["maximum_file_size_bytes"],
        ),
        _check(
            "size",
            "files_above_warning_size",
            warning_file_count,
            "==",
            0,
        ),
        _check(
            "size",
            "candidate_total_size_bytes",
            total_size,
            "<=",
            limits["target_repository_size_bytes"],
        ),
        _check("evidence", "research_figure_count", figure_count, ">=", 26),
        _check(
            "evidence",
            "compact_evidence_table_count",
            evidence_table_count,
            ">=",
            50,
        ),
        _check(
            "provenance",
            "openfast_reference_provenance_included",
            "data/raw/openfast/v5.0.0/PROVENANCE.md" in candidate_set
            and "data/raw/openfast/v5.0.0/stage18_healthy_deck/PROVENANCE.md"
            in candidate_set,
            "==",
            True,
        ),
        _check(
            "pipeline",
            "verified_upstream_stage_count",
            len(upstream_stages),
            "==",
            configuration["expected_upstream_stage_count"],
        ),
        _check(
            "pipeline",
            "all_upstream_pipeline_artifacts_pass",
            all(row.passed for row in upstream_audit),
            "==",
            True,
        ),
        _check(
            "communication",
            "interview_cv_and_application_material_present",
            all(
                heading in communication_text
                for heading in (
                    "## Two-to-three-minute interview explanation",
                    "## CV bullet",
                    "## PhD-application paragraph",
                    "## What is established and what this project contributes",
                )
            ),
            "==",
            True,
        ),
        _check(
            "repository",
            "release_candidate_file_count",
            len(candidates),
            ">=",
            200,
        ),
    ]
    readiness = pd.DataFrame(readiness_rows)
    release_ready = bool(readiness["supported"].all())
    summary_values = {
        "release_name": configuration["release_name"],
        "version": configuration["version"],
        "publication_state": configuration["publication_state"],
        "repository_url": configuration["repository_url"],
        "candidate_file_count": len(candidates),
        "candidate_total_bytes": total_size,
        "candidate_total_mib": total_size / (1024.0**2),
        "largest_candidate_path": str(
            manifest.loc[manifest["size_bytes"].idxmax(), "relative_path"]
        ),
        "largest_candidate_size_bytes": maximum_file_size,
        "figure_count": figure_count,
        "evidence_table_count": evidence_table_count,
        "upstream_stage_count": len(upstream_stages),
        "upstream_artifact_count": len(upstream_audit),
        "check_count": len(readiness),
        "passed_check_count": int(readiness["supported"].sum()),
        "release_ready": release_ready,
        "external_publication_performed": (
            configuration["publication_state"] != "local_release_candidate"
        ),
    }
    summary = pd.DataFrame(
        {"metric": list(summary_values), "value": list(summary_values.values())}
    )

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    readiness.to_csv(READINESS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    manifest.to_csv(MANIFEST_PATH, index=False)
    CHECKSUM_PATH.write_text(
        "".join(
            f"{row.sha256}  {row.relative_path}\n"
            for row in manifest.itertuples(index=False)
        ),
        encoding="utf-8",
    )
    _create_figure(manifest, readiness, summary_values)

    print("STAGE 26 - GITHUB AND RESEARCH-RELEASE READINESS")
    print(
        f"Release payload: {len(candidates)} files, "
        f"{total_size / (1024.0**2):.2f} MiB"
    )
    print(
        f"Evidence: {figure_count} figures, "
        f"{evidence_table_count} compact tables"
    )
    print(
        f"Readiness checks: {int(readiness['supported'].sum())}/{len(readiness)}"
    )
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    if not release_ready:
        failed = readiness.loc[~readiness["supported"], "check"].tolist()
        print(f"RESULT: FAIL ({'; '.join(failed)})")
        return 1
    print("RESULT: PASS (release payload ready; verifier performs no publication)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
