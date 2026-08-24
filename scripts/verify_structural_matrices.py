"""Run the Stage 1A structural-matrix verification gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from probdt.structural import (
    apply_stiffness_multipliers,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    assemble_zonal_stiffness_components,
    validate_structural_matrices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"


def main() -> None:
    """Load the reference structure, verify it, and print auditable evidence."""

    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model = config["model"]
    properties = config["structural_properties"]
    verification = config["verification"]

    masses = np.asarray(properties["masses_kg"], dtype=np.float64)
    story_stiffnesses = np.asarray(
        properties["story_stiffness_n_per_m"],
        dtype=np.float64,
    )
    stiffness_zones = properties["stiffness_zones"]

    if masses.size != model["n_dof"] or story_stiffnesses.size != model["n_dof"]:
        raise ValueError("Configured masses and story stiffnesses must match n_dof.")

    mass = assemble_mass_matrix(masses)
    stiffness = assemble_stiffness_matrix(story_stiffnesses)
    zone_names, zone_components = assemble_zonal_stiffness_components(
        story_stiffnesses,
        stiffness_zones,
    )
    report = validate_structural_matrices(
        mass,
        stiffness,
        symmetry_atol=float(verification["symmetry_atol"]),
    )

    reconstructed_stiffness = apply_stiffness_multipliers(
        zone_components,
        np.ones(len(zone_names)),
    )
    reconstruction_error = float(
        np.linalg.norm(reconstructed_stiffness - stiffness, ord="fro")
    )
    tolerance = float(verification["reconstruction_atol"])
    if reconstruction_error > tolerance:
        raise RuntimeError(
            "Zonal matrices do not reconstruct the healthy stiffness matrix."
        )

    uniformly_reduced = apply_stiffness_multipliers(
        zone_components,
        0.9 * np.ones(len(zone_names)),
    )
    uniform_scaling_error = float(
        np.linalg.norm(uniformly_reduced - 0.9 * stiffness, ord="fro")
    )
    if uniform_scaling_error > tolerance:
        raise RuntimeError("Uniform zonal multipliers do not scale K consistently.")

    print("STAGE 1A - STRUCTURAL MATRIX VERIFICATION")
    print(f"Model: {model['name']}")
    print(f"DOFs: {report.n_dof}")
    print(f"Mass matrix shape: {mass.shape}")
    print(f"Stiffness matrix shape: {stiffness.shape}")
    print(f"Zones: {', '.join(zone_names)}")
    print(f"M symmetric: {np.allclose(mass, mass.T)}")
    print(f"K symmetric: {np.allclose(stiffness, stiffness.T)}")
    print(f"Minimum eigenvalue(M): {report.minimum_mass_eigenvalue_kg:.6e} kg")
    print(
        "Minimum eigenvalue(K): "
        f"{report.minimum_stiffness_eigenvalue_n_per_m:.6e} N/m"
    )
    print(f"Condition number(M): {report.mass_condition_number:.6f}")
    print(f"Condition number(K): {report.stiffness_condition_number:.6f}")
    print(f"Zonal reconstruction error: {reconstruction_error:.6e} N/m")
    print(f"Uniform 10% loss scaling error: {uniform_scaling_error:.6e} N/m")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
