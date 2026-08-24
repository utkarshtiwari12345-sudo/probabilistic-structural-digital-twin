"""Dependency closure and deterministic patching for OpenFAST input decks."""

from __future__ import annotations

import re
import shutil
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TEXT_INPUT_SUFFIXES = {".fst", ".dat", ".inp", ".txt"}
REFERENCED_FILE_SUFFIXES = TEXT_INPUT_SUFFIXES | {
    ".af",
    ".bts",
    ".csv",
    ".dll",
    ".dylib",
    ".so",
    ".wnd",
}
QUOTED_VALUE = re.compile(r'"([^"]+)"')


class OpenFASTDeckError(ValueError):
    """Raised when an OpenFAST input deck is incomplete or unsafe."""


@dataclass(frozen=True)
class DeckDependencyClosure:
    """Files reached from a primary input and allowed unresolved references."""

    source_root: Path
    primary_file: Path
    files: tuple[Path, ...]
    unresolved: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.files or self.primary_file not in self.files:
            raise OpenFASTDeckError("Dependency closure must contain its primary file.")
        if len(set(self.files)) != len(self.files):
            raise OpenFASTDeckError("Dependency closure files must be unique.")


def openfast_file_references(path: str | Path) -> tuple[str, ...]:
    """Extract quoted file references from one text-based OpenFAST input."""

    source = Path(path)
    if source.suffix.lower() not in TEXT_INPUT_SUFFIXES:
        return ()
    try:
        content = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise OpenFASTDeckError(f"Could not read OpenFAST input: {source}") from error
    references: list[str] = []
    for value in QUOTED_VALUE.findall(content):
        cleaned = value.strip()
        normalized = cleaned.replace("\\", "/")
        suffix = Path(normalized).suffix.lower()
        if suffix in REFERENCED_FILE_SUFFIXES:
            references.append(normalized)
    return tuple(dict.fromkeys(references))


def resolve_openfast_dependencies(
    source_root: str | Path,
    primary_file: str | Path,
    *,
    allowed_missing_names: Sequence[str] = (),
) -> DeckDependencyClosure:
    """Resolve the transitive, root-confined dependency set for a FAST deck."""

    root = Path(source_root).resolve()
    primary = Path(primary_file)
    primary_relative = (
        primary if not primary.is_absolute() else primary.relative_to(root)
    )
    allowed = {name.lower() for name in allowed_missing_names}
    queue: deque[Path] = deque([primary_relative])
    visited: set[Path] = set()
    unresolved: set[str] = set()
    while queue:
        relative = queue.popleft()
        candidate = (root / relative).resolve()
        try:
            normalized = candidate.relative_to(root)
        except ValueError as error:
            raise OpenFASTDeckError(
                f"OpenFAST dependency escapes the source root: {relative}"
            ) from error
        if normalized in visited:
            continue
        if not candidate.is_file():
            if candidate.name.lower() in allowed:
                unresolved.add(candidate.name)
                continue
            raise OpenFASTDeckError(f"Missing OpenFAST dependency: {normalized}")
        visited.add(normalized)
        for reference in openfast_file_references(candidate):
            referenced = (root / normalized.parent / Path(reference)).resolve()
            try:
                referenced_relative = referenced.relative_to(root)
            except ValueError as error:
                raise OpenFASTDeckError(
                    f"OpenFAST dependency escapes the source root: {reference}"
                ) from error
            queue.append(referenced_relative)
    return DeckDependencyClosure(
        source_root=root,
        primary_file=primary_relative,
        files=tuple(sorted(visited, key=lambda item: item.as_posix().lower())),
        unresolved=tuple(sorted(unresolved)),
    )


def copy_openfast_dependency_closure(
    closure: DeckDependencyClosure,
    target_root: str | Path,
) -> tuple[Path, ...]:
    """Copy a resolved dependency closure while preserving relative paths."""

    destination_root = Path(target_root).resolve()
    copied: list[Path] = []
    for relative in closure.files:
        source = closure.source_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(relative)
    return tuple(copied)


def read_openfast_outlist_channels(path: str | Path) -> tuple[str, ...]:
    """Read the first standard OutList block from an OpenFAST module input."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start: int | None = None
    end: int | None = None
    for index, line in enumerate(lines):
        if start is None and "OutList" in line.split():
            start = index + 1
            continue
        if start is not None and line.lstrip().upper().startswith("END"):
            end = index
            break
    if start is None or end is None:
        raise OpenFASTDeckError("Could not locate a complete OpenFAST OutList block.")
    channels: list[str] = []
    for line in lines[start:end]:
        quoted = QUOTED_VALUE.search(line)
        if not quoted:
            continue
        channels.extend(
            channel
            for channel in re.split(r"[,;\s]+", quoted.group(1).strip())
            if channel
        )
    return tuple(channels)


def ensure_openfast_outlist_channels(
    path: str | Path,
    channels: Sequence[str],
) -> tuple[str, ...]:
    """Add missing channels before the first OutList END line, idempotently."""

    target = Path(path)
    requested = tuple(channels)
    if not requested or any(not channel.strip() for channel in requested):
        raise OpenFASTDeckError("At least one nonempty OutList channel is required.")
    if len(set(requested)) != len(requested):
        raise OpenFASTDeckError("Requested OutList channels must be unique.")
    content = target.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    start: int | None = None
    end: int | None = None
    for index, line in enumerate(lines):
        if start is None and "OutList" in line.split():
            start = index + 1
            continue
        if start is not None and line.lstrip().upper().startswith("END"):
            end = index
            break
    if start is None or end is None:
        raise OpenFASTDeckError("Could not locate a complete OpenFAST OutList block.")
    existing = set(read_openfast_outlist_channels(target))
    missing = tuple(channel for channel in requested if channel not in existing)
    if missing:
        newline = "\r\n" if "\r\n" in content else "\n"
        lines.insert(end, f'"{", ".join(missing)}"{newline}')
        target.write_text("".join(lines), encoding="utf-8", newline="")
    return missing
