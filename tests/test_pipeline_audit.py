"""Tests for Phase I pipeline selection and artifact auditing."""

from pathlib import Path

import pytest

from probdt.diagnostics import (
    PipelineAuditError,
    audit_pipeline_artifacts,
    select_pipeline_stages,
)


def stages() -> list[dict[str, object]]:
    return [
        {"id": "1a", "name": "first", "required_artifacts": []},
        {"id": "1b", "name": "second", "required_artifacts": []},
        {"id": "2", "name": "third", "required_artifacts": []},
    ]


def test_pipeline_stage_range_preserves_manifest_order() -> None:
    selected = select_pipeline_stages(
        stages(),
        from_stage="1b",
        through_stage="2",
    )
    assert [stage["id"] for stage in selected] == ["1b", "2"]


def test_exact_and_range_selection_cannot_be_combined() -> None:
    with pytest.raises(PipelineAuditError, match="cannot be combined"):
        select_pipeline_stages(
            stages(),
            selected_ids=["1a"],
            through_stage="2",
        )


def test_artifact_audit_hashes_present_files_and_reports_missing(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "result.csv"
    existing.write_text("a,b\n1,2\n", encoding="utf-8")
    definitions = [
        {
            "id": "1",
            "name": "test",
            "required_artifacts": [
                {"path": "result.csv", "minimum_size_bytes": 4},
                {"path": "missing.csv", "minimum_size_bytes": 1},
            ],
        }
    ]

    audit = audit_pipeline_artifacts(tmp_path, definitions)

    assert audit[0].passed
    assert len(audit[0].sha256) == 64
    assert not audit[1].passed
    assert audit[1].sha256 == ""


def test_artifact_audit_rejects_parent_traversal(tmp_path: Path) -> None:
    definitions = [
        {
            "id": "1",
            "required_artifacts": [{"path": "../outside.txt"}],
        }
    ]
    with pytest.raises(PipelineAuditError, match="escapes"):
        audit_pipeline_artifacts(tmp_path, definitions)
