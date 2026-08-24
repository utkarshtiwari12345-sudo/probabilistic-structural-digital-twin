"""Controlled stiffness perturbations for OpenFAST SubDyn members."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from probdt.openfast.deck import OpenFASTDeckError


@dataclass(frozen=True)
class SubDynStiffnessPatch:
    """Auditable material-stiffness change applied to one SubDyn member."""

    member_id: int
    source_property_id: int
    clone_property_id: int
    stiffness_scale: float
    young_e_before_pa: float
    young_e_after_pa: float
    shear_g_before_pa: float
    shear_g_after_pa: float
    material_density_kg_per_m3: float
    outer_diameter_m: float
    wall_thickness_m: float


def _named_count(lines: list[str], name: str) -> tuple[int, int]:
    for index, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) >= 2 and tokens[1] == name:
            try:
                return index, int(tokens[0])
            except ValueError as error:
                message = f"Invalid {name} value in SubDyn file."
                raise OpenFASTDeckError(message) from error
    raise OpenFASTDeckError(f"SubDyn field {name} was not found.")


def _table_start(lines: list[str], first: str, second: str) -> int:
    for index, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0] == first and tokens[1] == second:
            return index + 2
    raise OpenFASTDeckError(f"SubDyn table {first}/{second} was not found.")


def patch_subdyn_member_stiffness(
    path: str | Path,
    member_scales: Mapping[int, float],
    clone_property_ids: Mapping[int, int],
) -> tuple[SubDynStiffnessPatch, ...]:
    """Clone member properties and apply local isotropic stiffness scales.

    Each selected member receives a unique circular property row. Young's and
    shear moduli are scaled together; density and cross-section geometry are
    copied unchanged. The original property rows remain untouched.
    """

    target = Path(path)
    lines = target.read_text(encoding="utf-8").splitlines()
    if not member_scales:
        raise OpenFASTDeckError("At least one SubDyn member scale is required.")
    if set(member_scales) != set(clone_property_ids):
        raise OpenFASTDeckError("Member scales and clone-property IDs must align.")
    if len(set(clone_property_ids.values())) != len(clone_property_ids):
        raise OpenFASTDeckError("Clone-property IDs must be unique.")
    for member_id, scale in member_scales.items():
        if member_id <= 0:
            raise OpenFASTDeckError("SubDyn member IDs must be positive.")
        if not 0.0 < float(scale) <= 1.0:
            raise OpenFASTDeckError("SubDyn stiffness scales must lie in (0, 1].")

    member_count_index, member_count = _named_count(lines, "NMembers")
    del member_count_index
    member_start = _table_start(lines, "MemberID", "MJointID1")
    member_rows: dict[int, tuple[int, list[str]]] = {}
    for offset in range(member_count):
        line_index = member_start + offset
        fields = lines[line_index].split()
        if len(fields) < 7:
            raise OpenFASTDeckError("Malformed SubDyn member row.")
        member_rows[int(fields[0])] = (line_index, fields)

    property_count_index, property_count = _named_count(lines, "NPropSetsCyl")
    property_start = _table_start(lines, "PropSetID", "YoungE")
    properties: dict[int, tuple[float, float, float, float, float]] = {}
    existing_property_ids: set[int] = set()
    for offset in range(property_count):
        fields = lines[property_start + offset].split()
        if len(fields) < 6:
            raise OpenFASTDeckError("Malformed SubDyn circular-property row.")
        property_id = int(fields[0])
        existing_property_ids.add(property_id)
        properties[property_id] = tuple(float(value) for value in fields[1:6])

    requested_clone_ids = set(clone_property_ids.values())
    if existing_property_ids & requested_clone_ids:
        raise OpenFASTDeckError("A requested clone-property ID already exists.")

    patches: list[SubDynStiffnessPatch] = []
    clone_rows: list[str] = []
    for member_id in sorted(member_scales):
        if member_id not in member_rows:
            raise OpenFASTDeckError(f"SubDyn member {member_id} was not found.")
        line_index, fields = member_rows[member_id]
        source_start = int(fields[3])
        source_end = int(fields[4])
        if source_start != source_end:
            raise OpenFASTDeckError(
                "Stage 19 requires members with one uniform source property."
            )
        if source_start not in properties:
            raise OpenFASTDeckError("A member references an unknown property row.")

        young_e, shear_g, density, diameter, thickness = properties[source_start]
        scale = float(member_scales[member_id])
        clone_id = int(clone_property_ids[member_id])
        fields[3] = str(clone_id)
        fields[4] = str(clone_id)
        lines[line_index] = (
            f"{int(fields[0]):4d} {int(fields[1]):11d} {int(fields[2]):11d} "
            f"{clone_id:12d} {clone_id:13d} {fields[5]:>11s} {fields[6]:>8s}"
        )
        clone_rows.append(
            f"{clone_id:4d}        {young_e * scale:.5e}     "
            f"{shear_g * scale:.5e}       {density:.2f}         "
            f"{diameter:.6f}        {thickness:.6f}"
        )
        patches.append(
            SubDynStiffnessPatch(
                member_id=member_id,
                source_property_id=source_start,
                clone_property_id=clone_id,
                stiffness_scale=scale,
                young_e_before_pa=young_e,
                young_e_after_pa=young_e * scale,
                shear_g_before_pa=shear_g,
                shear_g_after_pa=shear_g * scale,
                material_density_kg_per_m3=density,
                outer_diameter_m=diameter,
                wall_thickness_m=thickness,
            )
        )

    new_property_count = property_count + len(clone_rows)
    count_tokens = lines[property_count_index].split(maxsplit=2)
    if len(count_tokens) < 3:
        raise OpenFASTDeckError("Malformed NPropSetsCyl line.")
    lines[property_count_index] = (
        f"{new_property_count:14d}   {count_tokens[1]} {count_tokens[2]}"
    )
    insert_at = property_start + property_count
    lines[insert_at:insert_at] = clone_rows
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tuple(patches)
