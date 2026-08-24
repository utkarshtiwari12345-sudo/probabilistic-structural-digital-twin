"""Auditable random-seed patches for OpenFAST environmental inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from probdt.openfast.deck import OpenFASTDeckError


@dataclass(frozen=True)
class OpenFASTSeedPatch:
    """One integer seed replacement in a labelled OpenFAST-style input file."""

    field: str
    before: int
    after: int
    line_number: int


def patch_openfast_integer_seed(
    path: str | Path,
    field: str,
    seed: int,
) -> OpenFASTSeedPatch:
    """Replace exactly one labelled integer seed while preserving its line."""

    target = Path(path)
    if not field or any(character.isspace() for character in field):
        raise OpenFASTDeckError("The seed field must be one nonempty token.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise OpenFASTDeckError("The random seed must be an integer.")
    if not -(2**31) <= seed <= 2**31 - 1:
        raise OpenFASTDeckError("The random seed must fit in a signed 32-bit integer.")

    lines = target.read_text(encoding="utf-8").splitlines()
    matches: list[tuple[int, list[str]]] = []
    for index, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) >= 2 and tokens[1] == field:
            matches.append((index, tokens))
    if len(matches) != 1:
        raise OpenFASTDeckError(
            f"Expected exactly one {field} seed field; found {len(matches)}."
        )

    line_index, tokens = matches[0]
    try:
        before = int(tokens[0])
    except ValueError as error:
        raise OpenFASTDeckError(f"The {field} seed value is not an integer.") from error
    source_line = lines[line_index]
    value_start = len(source_line) - len(source_line.lstrip())
    value_end = value_start + len(tokens[0])
    lines[line_index] = source_line[:value_start] + str(seed) + source_line[value_end:]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return OpenFASTSeedPatch(
        field=field,
        before=before,
        after=seed,
        line_number=line_index + 1,
    )
