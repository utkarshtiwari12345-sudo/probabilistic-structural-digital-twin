"""Tests for controlled Stage 19 SubDyn stiffness perturbations."""

from pathlib import Path
from shutil import copyfile

import pytest

from probdt.openfast import OpenFASTDeckError, patch_subdyn_member_stiffness

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUBDYN = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "openfast"
    / "v5.0.0"
    / "stage18_healthy_deck"
    / "5MW_OC3Mnpl_DLL_WTurb_WavesIrr"
    / "NRELOffshrBsline5MW_OC3Monopile_SubDyn.dat"
)


def test_member_stiffness_patch_is_local_and_mass_preserving(tmp_path: Path) -> None:
    target = tmp_path / "SubDyn.dat"
    copyfile(SOURCE_SUBDYN, target)

    patches = patch_subdyn_member_stiffness(
        target,
        member_scales={1: 1.0, 2: 0.8, 3: 1.0},
        clone_property_ids={1: 5, 2: 6, 3: 7},
    )

    assert [patch.member_id for patch in patches] == [1, 2, 3]
    assert [patch.clone_property_id for patch in patches] == [5, 6, 7]
    assert patches[1].young_e_after_pa == pytest.approx(1.68e11)
    assert patches[1].shear_g_after_pa == pytest.approx(6.464e10)
    assert {patch.material_density_kg_per_m3 for patch in patches} == {8500.0}
    assert {patch.outer_diameter_m for patch in patches} == {6.0}
    assert {patch.wall_thickness_m for patch in patches} == {0.06}

    text = target.read_text(encoding="utf-8")
    assert "7   NPropSetsCyl" in text
    assert "1.68000e+11" in text
    assert "6.46400e+10" in text
    for member_id, property_id in ((1, 5), (2, 6), (3, 7)):
        member_line = next(
            line
            for line in text.splitlines()
            if line.split()[:1] == [str(member_id)]
            and len(line.split()) == 7
            and line.split()[5] == "1c"
        )
        assert member_line.split()[3:5] == [str(property_id), str(property_id)]


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.01])
def test_member_stiffness_patch_rejects_invalid_scale(
    tmp_path: Path,
    scale: float,
) -> None:
    target = tmp_path / "SubDyn.dat"
    copyfile(SOURCE_SUBDYN, target)

    with pytest.raises(OpenFASTDeckError, match="scales"):
        patch_subdyn_member_stiffness(
            target,
            member_scales={1: scale},
            clone_property_ids={1: 5},
        )


def test_member_stiffness_patch_rejects_reused_clone_id(tmp_path: Path) -> None:
    target = tmp_path / "SubDyn.dat"
    copyfile(SOURCE_SUBDYN, target)

    with pytest.raises(OpenFASTDeckError, match="already exists"):
        patch_subdyn_member_stiffness(
            target,
            member_scales={1: 0.9},
            clone_property_ids={1: 1},
        )
