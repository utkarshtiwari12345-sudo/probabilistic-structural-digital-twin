"""Verification tests for fixed-base shear-model matrix assembly."""

import numpy as np
import pytest

from probdt.structural import (
    StructuralMatrixError,
    apply_stiffness_multipliers,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    assemble_story_stiffness_components,
    assemble_zonal_stiffness_components,
    validate_structural_matrices,
)


def test_mass_matrix_is_diagonal_in_si_units() -> None:
    masses = np.array([10.0, 20.0, 30.0])
    mass = assemble_mass_matrix(masses)

    np.testing.assert_array_equal(mass, np.diag(masses))


def test_three_dof_stiffness_matches_hand_assembled_matrix() -> None:
    story_stiffnesses = np.array([100.0, 80.0, 60.0])
    expected = np.array(
        [
            [180.0, -80.0, 0.0],
            [-80.0, 140.0, -60.0],
            [0.0, -60.0, 60.0],
        ]
    )

    stiffness = assemble_stiffness_matrix(story_stiffnesses)

    np.testing.assert_array_equal(stiffness, expected)


def test_story_components_reconstruct_global_stiffness() -> None:
    story_stiffnesses = np.array([100.0, 80.0, 60.0])
    components = assemble_story_stiffness_components(story_stiffnesses)

    np.testing.assert_array_equal(
        components.sum(axis=0),
        assemble_stiffness_matrix(story_stiffnesses),
    )


def test_zones_reconstruct_healthy_stiffness_exactly() -> None:
    story_stiffnesses = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0])
    zones = {"lower": [0, 1], "middle": [2, 3], "upper": [4, 5]}
    names, components = assemble_zonal_stiffness_components(
        story_stiffnesses,
        zones,
    )

    reconstructed = apply_stiffness_multipliers(components, np.ones(3))

    assert names == ("lower", "middle", "upper")
    np.testing.assert_array_equal(
        reconstructed,
        assemble_stiffness_matrix(story_stiffnesses),
    )


def test_zonal_multiplier_affects_only_requested_component() -> None:
    story_stiffnesses = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0])
    zones = {"lower": [0, 1], "middle": [2, 3], "upper": [4, 5]}
    _, components = assemble_zonal_stiffness_components(story_stiffnesses, zones)

    multipliers = np.array([0.8, 1.0, 1.0])
    reduced = apply_stiffness_multipliers(components, multipliers)
    expected = 0.8 * components[0] + components[1] + components[2]

    np.testing.assert_allclose(reduced, expected, rtol=0.0, atol=1.0e-12)


def test_validation_reports_positive_definite_reference_matrices() -> None:
    mass = assemble_mass_matrix([10.0, 20.0, 30.0])
    stiffness = assemble_stiffness_matrix([100.0, 80.0, 60.0])

    report = validate_structural_matrices(mass, stiffness)

    assert report.n_dof == 3
    assert report.minimum_mass_eigenvalue_kg > 0.0
    assert report.minimum_stiffness_eigenvalue_n_per_m > 0.0


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([10.0, 0.0], "strictly positive"),
        ([10.0, -1.0], "strictly positive"),
        ([10.0, np.nan], "finite"),
    ],
)
def test_invalid_mass_values_are_rejected(values: list[float], message: str) -> None:
    with pytest.raises(StructuralMatrixError, match=message):
        assemble_mass_matrix(values)


@pytest.mark.parametrize(
    "zones",
    [
        {"lower": [0, 1], "upper": [1, 2]},
        {"lower": [0], "upper": [2]},
        {"all": [0, 1, 3]},
    ],
)
def test_zones_must_partition_every_story_once(
    zones: dict[str, list[int]],
) -> None:
    with pytest.raises(StructuralMatrixError):
        assemble_zonal_stiffness_components([100.0, 80.0, 60.0], zones)


@pytest.mark.parametrize("multiplier", [0.0, -0.1, 1.1, np.nan])
def test_nonphysical_stiffness_multipliers_are_rejected(multiplier: float) -> None:
    _, components = assemble_zonal_stiffness_components(
        [100.0, 80.0],
        {"lower": [0], "upper": [1]},
    )

    with pytest.raises(StructuralMatrixError):
        apply_stiffness_multipliers(components, [multiplier, 1.0])


def test_validation_rejects_nonsymmetric_stiffness() -> None:
    mass = np.eye(2)
    nonsymmetric_stiffness = np.array([[2.0, -1.0], [0.0, 1.0]])

    with pytest.raises(StructuralMatrixError, match="symmetric"):
        validate_structural_matrices(mass, nonsymmetric_stiffness)
