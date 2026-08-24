"""Tests for deterministic, root-confined OpenFAST deck preparation."""

from pathlib import Path

import pytest

from probdt.openfast import (
    OpenFASTDeckError,
    copy_openfast_dependency_closure,
    ensure_openfast_outlist_channels,
    openfast_file_references,
    read_openfast_outlist_channels,
    resolve_openfast_dependencies,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_DECK = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "openfast"
    / "v5.0.0"
    / "stage18_healthy_deck"
)
CURATED_PRIMARY = Path(
    "5MW_OC3Mnpl_DLL_WTurb_WavesIrr/5MW_OC3Mnpl_DLL_WTurb_WavesIrr.fst"
)


def test_dependency_closure_follows_text_and_binary_references(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.fst").write_text(
        '"module/module.dat" EDFile - input\n', encoding="utf-8"
    )
    module = source / "module"
    module.mkdir()
    (module / "module.dat").write_text(
        '"../wind.bts" WindFile - binary\n', encoding="utf-8"
    )
    (source / "wind.bts").write_bytes(b"wind")

    closure = resolve_openfast_dependencies(source, "main.fst")

    assert closure.files == (
        Path("main.fst"),
        Path("module/module.dat"),
        Path("wind.bts"),
    )
    copied = copy_openfast_dependency_closure(closure, tmp_path / "target")
    assert copied == closure.files
    assert (tmp_path / "target" / "wind.bts").read_bytes() == b"wind"


def test_dependency_closure_rejects_escape_and_missing_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "outside.dat").write_text("outside", encoding="utf-8")
    primary = source / "main.fst"
    primary.write_text('"../outside.dat" EDFile\n', encoding="utf-8")

    with pytest.raises(OpenFASTDeckError, match="escapes"):
        resolve_openfast_dependencies(source, "main.fst")

    primary.write_text('"missing.dat" EDFile\n', encoding="utf-8")
    with pytest.raises(OpenFASTDeckError, match="Missing"):
        resolve_openfast_dependencies(source, "main.fst")


def test_allowed_missing_dependency_is_recorded(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.fst").write_text(
        '"DISCON.dll" DLL_FileName\n', encoding="utf-8"
    )

    closure = resolve_openfast_dependencies(
        source,
        "main.fst",
        allowed_missing_names=["DISCON.dll"],
    )

    assert closure.files == (Path("main.fst"),)
    assert closure.unresolved == ("DISCON.dll",)


def test_outlist_patching_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "ElastoDyn.dat"
    path.write_text(
        "OutList - channels\n"
        '"PtfmSurge, TwHt1TPxi"\n'
        "END of OutList\n"
        "OutList - optional nodal channels\n"
        "END\n",
        encoding="utf-8",
    )

    added = ensure_openfast_outlist_channels(
        path,
        ["NcIMUTAxs", "NcIMUTAys"],
    )
    added_again = ensure_openfast_outlist_channels(
        path,
        ["NcIMUTAxs", "NcIMUTAys"],
    )

    assert added == ("NcIMUTAxs", "NcIMUTAys")
    assert added_again == ()
    assert read_openfast_outlist_channels(path) == (
        "PtfmSurge",
        "TwHt1TPxi",
        "NcIMUTAxs",
        "NcIMUTAys",
    )


def test_file_reference_extraction_ignores_nonfiles(tmp_path: Path) -> None:
    path = tmp_path / "module.dat"
    path.write_text(
        '"default" DT\n"unused" File\n"Airfoils/foil.dat" AFName\n',
        encoding="utf-8",
    )

    assert openfast_file_references(path) == ("Airfoils/foil.dat",)


def test_curated_stage18_deck_is_closed_and_sensor_matched() -> None:
    closure = resolve_openfast_dependencies(CURATED_DECK, CURATED_PRIMARY)
    elastodyn = (
        CURATED_DECK
        / "5MW_OC3Mnpl_DLL_WTurb_WavesIrr"
        / "NRELOffshrBsline5MW_OC3Monopile_ElastoDyn.dat"
    )

    assert len(closure.files) == 31
    assert closure.unresolved == ()
    assert {"NcIMUTAxs", "NcIMUTAys"} <= set(
        read_openfast_outlist_channels(elastodyn)
    )
