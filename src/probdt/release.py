"""Utilities for building and validating a compact research release."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class ReleaseAuditError(ValueError):
    """Raised when a release candidate path violates the local contract."""


@dataclass(frozen=True)
class ReleaseFile:
    """Immutable metadata for one release-candidate file."""

    relative_path: str
    category: str
    size_bytes: int
    sha256: str


def normalize_release_path(value: str) -> str:
    """Return a safe repository-relative POSIX path."""

    normalized = str(value).replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ReleaseAuditError("Release paths must remain inside the repository.")
    return path.as_posix()


def classify_release_path(relative_path: str) -> str:
    """Classify a release path for size and content summaries."""

    path = normalize_release_path(relative_path)
    if path.startswith("data/raw/openfast/"):
        return "pinned_openfast_reference"
    if path.startswith("figures/"):
        return "research_figures"
    if path.startswith("results/metrics/"):
        return "compact_metrics"
    if path.startswith(("src/", "scripts/", "tests/")) or path.endswith(".py"):
        return "code_and_tests"
    if path.startswith("config/"):
        return "configuration"
    if path.startswith("docs/") or path.endswith(".md"):
        return "documentation"
    return "repository_metadata"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""

    if not path.is_file():
        raise ReleaseAuditError(f"Release candidate is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_release_files(
    repository_root: Path,
    relative_paths: Iterable[str],
) -> tuple[ReleaseFile, ...]:
    """Build sorted size and checksum records for candidate paths."""

    records = []
    for value in sorted({normalize_release_path(path) for path in relative_paths}):
        absolute = repository_root / value
        if not absolute.is_file():
            raise ReleaseAuditError(f"Candidate path is missing: {value}")
        records.append(
            ReleaseFile(
                relative_path=value,
                category=classify_release_path(value),
                size_bytes=absolute.stat().st_size,
                sha256=sha256_file(absolute),
            )
        )
    return tuple(records)


def forbidden_release_paths(
    relative_paths: Iterable[str],
    *,
    forbidden_prefixes: Sequence[str],
    forbidden_names: Sequence[str],
) -> tuple[str, ...]:
    """Return candidate paths matching forbidden prefixes or file names."""

    prefixes = tuple(
        normalize_release_path(value).rstrip("/") + "/"
        for value in forbidden_prefixes
    )
    names = {str(value).casefold() for value in forbidden_names}
    invalid = []
    for value in relative_paths:
        path = normalize_release_path(value)
        if path.startswith(prefixes) or PurePosixPath(path).name.casefold() in names:
            invalid.append(path)
    return tuple(sorted(set(invalid)))


def metadata_placeholders(
    repository_root: Path,
    relative_paths: Iterable[str],
    tokens: Sequence[str],
) -> tuple[str, ...]:
    """Return path/token pairs for unresolved release metadata placeholders."""

    matches = []
    for value in relative_paths:
        path = normalize_release_path(value)
        text = (repository_root / path).read_text(encoding="utf-8")
        for token in tokens:
            if str(token) in text:
                matches.append(f"{path}:{token}")
    return tuple(sorted(matches))
