from __future__ import annotations

import hashlib

import pytest

from probdt.release import (
    ReleaseAuditError,
    build_release_files,
    classify_release_path,
    forbidden_release_paths,
    metadata_placeholders,
    normalize_release_path,
)


def test_release_paths_are_normalized_and_classified() -> None:
    assert normalize_release_path(r"figures\result.png") == "figures/result.png"
    assert classify_release_path("figures/result.png") == "research_figures"
    assert classify_release_path("src/probdt/model.py") == "code_and_tests"
    assert classify_release_path("config/model.yaml") == "configuration"
    with pytest.raises(ReleaseAuditError):
        normalize_release_path("../secret.txt")


def test_release_records_are_sorted_and_hashed(tmp_path) -> None:
    (tmp_path / "README.md").write_text("release\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "model.yaml").write_text("value: 1\n", encoding="utf-8")
    records = build_release_files(
        tmp_path,
        ("config/model.yaml", "README.md"),
    )
    assert [record.relative_path for record in records] == [
        "README.md",
        "config/model.yaml",
    ]
    expected = hashlib.sha256((tmp_path / "README.md").read_bytes()).hexdigest()
    assert records[0].sha256 == expected


def test_forbidden_release_paths_detect_private_and_runtime_files() -> None:
    invalid = forbidden_release_paths(
        ("src/model.py", "work/run/output.outb", "secrets/.env"),
        forbidden_prefixes=("work/",),
        forbidden_names=(".env",),
    )
    assert invalid == ("secrets/.env", "work/run/output.outb")


def test_metadata_placeholder_scan(tmp_path) -> None:
    (tmp_path / "CITATION.cff").write_text(
        "repository-code: https://github.com/OWNER/repository\n",
        encoding="utf-8",
    )
    assert metadata_placeholders(
        tmp_path,
        ("CITATION.cff",),
        ("OWNER", "YOUR_NAME"),
    ) == ("CITATION.cff:OWNER",)
