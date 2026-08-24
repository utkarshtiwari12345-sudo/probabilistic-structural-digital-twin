"""Tests for auditable OpenFAST and TurbSim environmental seed changes."""

from pathlib import Path

import pytest

from probdt.openfast import OpenFASTDeckError, patch_openfast_integer_seed


def test_integer_seed_patch_changes_only_the_value(tmp_path: Path) -> None:
    target = tmp_path / "environment.dat"
    target.write_text(
        "header\n      13428   RandSeed1       - first seed\nfooter\n",
        encoding="utf-8",
    )

    patch = patch_openfast_integer_seed(target, "RandSeed1", 27531)

    assert patch.before == 13428
    assert patch.after == 27531
    assert patch.line_number == 2
    assert target.read_text(encoding="utf-8") == (
        "header\n      27531   RandSeed1       - first seed\nfooter\n"
    )


@pytest.mark.parametrize("seed", [-(2**31) - 1, 2**31, 1.5, True])
def test_integer_seed_patch_rejects_invalid_seed(tmp_path: Path, seed: object) -> None:
    target = tmp_path / "environment.dat"
    target.write_text("123 WaveSeed(1) - seed\n", encoding="utf-8")

    with pytest.raises(OpenFASTDeckError, match="seed"):
        patch_openfast_integer_seed(target, "WaveSeed(1)", seed)  # type: ignore[arg-type]


def test_integer_seed_patch_requires_exactly_one_field(tmp_path: Path) -> None:
    target = tmp_path / "environment.dat"
    target.write_text("1 OtherSeed - seed\n", encoding="utf-8")

    with pytest.raises(OpenFASTDeckError, match="exactly one"):
        patch_openfast_integer_seed(target, "RandSeed1", 2)
