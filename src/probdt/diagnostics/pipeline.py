"""Artifact auditing and ordered selection for the reproducible pipeline."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class PipelineAuditError(ValueError):
    """Raised when a pipeline manifest or stage selection is invalid."""


@dataclass(frozen=True)
class ArtifactAudit:
    """One immutable required-artifact audit result."""

    stage_id: str
    stage_name: str
    relative_path: str
    exists: bool
    size_bytes: int
    sha256: str
    passed: bool


def _stage_ids(stages: Sequence[Mapping[str, object]]) -> list[str]:
    identifiers = [str(stage.get("id", "")) for stage in stages]
    if not identifiers or any(not identifier for identifier in identifiers):
        raise PipelineAuditError("Every pipeline stage requires a nonempty id.")
    if len(set(identifiers)) != len(identifiers):
        raise PipelineAuditError("Pipeline stage ids must be unique.")
    return identifiers


def select_pipeline_stages(
    stages: Sequence[Mapping[str, object]],
    *,
    selected_ids: Sequence[str] = (),
    from_stage: str | None = None,
    through_stage: str | None = None,
) -> list[Mapping[str, object]]:
    """Select stages in manifest order using exact ids or one contiguous range."""

    identifiers = _stage_ids(stages)
    if selected_ids and (from_stage is not None or through_stage is not None):
        raise PipelineAuditError(
            "Exact stage selection cannot be combined with a stage range."
        )
    if selected_ids:
        unknown = set(selected_ids) - set(identifiers)
        if unknown:
            raise PipelineAuditError(f"Unknown pipeline stage ids: {sorted(unknown)}")
        requested = set(selected_ids)
        return [stage for stage in stages if str(stage["id"]) in requested]
    first = 0 if from_stage is None else identifiers.index(str(from_stage))
    last = len(stages) - 1 if through_stage is None else identifiers.index(
        str(through_stage)
    )
    if first > last:
        raise PipelineAuditError("from_stage must not follow through_stage.")
    return list(stages[first : last + 1])


def audit_pipeline_artifacts(
    project_root: Path,
    stages: Sequence[Mapping[str, object]],
) -> list[ArtifactAudit]:
    """Hash every required artifact and reject paths outside the project root."""

    root = Path(project_root).resolve()
    _stage_ids(stages)
    rows: list[ArtifactAudit] = []
    for stage in stages:
        stage_id = str(stage["id"])
        stage_name = str(stage.get("name", stage_id))
        artifacts = stage.get("required_artifacts", ())
        if not isinstance(artifacts, Sequence) or isinstance(artifacts, str):
            raise PipelineAuditError(
                f"Stage {stage_id} required_artifacts must be a sequence."
            )
        for specification in artifacts:
            if not isinstance(specification, Mapping) or "path" not in specification:
                raise PipelineAuditError(
                    f"Stage {stage_id} contains an invalid artifact specification."
                )
            relative = str(specification["path"])
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise PipelineAuditError(
                    f"Artifact path escapes the project root: {relative}"
                ) from error
            minimum_size = int(specification.get("minimum_size_bytes", 1))
            if minimum_size < 0:
                raise PipelineAuditError("minimum_size_bytes must be nonnegative.")
            exists = candidate.is_file()
            size = candidate.stat().st_size if exists else 0
            digest = ""
            if exists:
                hasher = hashlib.sha256()
                with candidate.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
            rows.append(
                ArtifactAudit(
                    stage_id=stage_id,
                    stage_name=stage_name,
                    relative_path=relative,
                    exists=exists,
                    size_bytes=size,
                    sha256=digest,
                    passed=exists and size >= minimum_size,
                )
            )
    return rows
