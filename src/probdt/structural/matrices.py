"""Assembly and validation of reduced-order structural matrices.

The structure is a fixed-base shear model with one lateral translational degree
of freedom per lumped mass. Story spring ``i`` connects DOF ``i`` to DOF
``i - 1``; story zero connects the first DOF directly to the fixed base.

All public functions use SI units:

- mass: kg;
- stiffness: N/m;
- displacement: m.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class StructuralMatrixError(ValueError):
    """Raised when structural inputs cannot define a physical matrix model."""


@dataclass(frozen=True)
class MatrixValidationReport:
    """Numerical evidence that the structural matrices satisfy core checks."""

    n_dof: int
    minimum_mass_eigenvalue_kg: float
    minimum_stiffness_eigenvalue_n_per_m: float
    mass_condition_number: float
    stiffness_condition_number: float


def _positive_vector(values: ArrayLike, *, name: str) -> FloatArray:
    """Convert an input to a finite, strictly positive one-dimensional array."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise StructuralMatrixError(f"{name} must be a one-dimensional sequence.")
    if vector.size == 0:
        raise StructuralMatrixError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(vector)):
        raise StructuralMatrixError(f"{name} must contain only finite values.")
    if np.any(vector <= 0.0):
        raise StructuralMatrixError(f"{name} must be strictly positive.")
    return vector.copy()


def assemble_mass_matrix(masses_kg: ArrayLike) -> FloatArray:
    """Return the diagonal lumped-mass matrix in kilograms."""

    masses = _positive_vector(masses_kg, name="masses_kg")
    return np.diag(masses)


def assemble_story_stiffness_components(
    story_stiffness_n_per_m: ArrayLike,
) -> FloatArray:
    """Return the contribution of every story spring to the global stiffness.

    The returned array has shape ``(n_story, n_dof, n_dof)``. For this shear
    model, the number of stories equals the number of DOFs.
    """

    stiffnesses = _positive_vector(
        story_stiffness_n_per_m,
        name="story_stiffness_n_per_m",
    )
    n_dof = stiffnesses.size
    components = np.zeros((n_dof, n_dof, n_dof), dtype=np.float64)

    for story_index, stiffness in enumerate(stiffnesses):
        component = components[story_index]
        component[story_index, story_index] += stiffness

        if story_index > 0:
            lower_dof = story_index - 1
            component[lower_dof, lower_dof] += stiffness
            component[lower_dof, story_index] -= stiffness
            component[story_index, lower_dof] -= stiffness

    return components


def assemble_stiffness_matrix(story_stiffness_n_per_m: ArrayLike) -> FloatArray:
    """Assemble the healthy fixed-base shear-model stiffness matrix in N/m."""

    return assemble_story_stiffness_components(
        story_stiffness_n_per_m
    ).sum(axis=0)


def assemble_zonal_stiffness_components(
    story_stiffness_n_per_m: ArrayLike,
    stiffness_zones: Mapping[str, Sequence[int]],
) -> tuple[tuple[str, ...], FloatArray]:
    """Aggregate story contributions into named deterioration zones.

    Every story must be assigned exactly once. This explicit partition prevents
    stiffness from being silently omitted or counted twice during deterioration
    inference.

    Returns
    -------
    zone_names:
        Zone names in insertion order.
    zone_components:
        Array with shape ``(n_zone, n_dof, n_dof)``.
    """

    story_components = assemble_story_stiffness_components(
        story_stiffness_n_per_m
    )
    n_story = story_components.shape[0]

    if not stiffness_zones:
        raise StructuralMatrixError("stiffness_zones must define at least one zone.")

    zone_names = tuple(stiffness_zones)
    zone_components = np.zeros(
        (len(zone_names), n_story, n_story),
        dtype=np.float64,
    )
    assigned_stories: list[int] = []

    for zone_index, zone_name in enumerate(zone_names):
        story_indices = list(stiffness_zones[zone_name])
        if not story_indices:
            raise StructuralMatrixError(
                f"Stiffness zone {zone_name!r} must contain at least one story."
            )

        for story_index in story_indices:
            if isinstance(story_index, bool) or not isinstance(story_index, int):
                raise StructuralMatrixError(
                    f"Story indices must be integers; received {story_index!r}."
                )
            if not 0 <= story_index < n_story:
                raise StructuralMatrixError(
                    f"Story index {story_index} is outside [0, {n_story - 1}]."
                )
            assigned_stories.append(story_index)
            zone_components[zone_index] += story_components[story_index]

    expected = list(range(n_story))
    if sorted(assigned_stories) != expected:
        raise StructuralMatrixError(
            "stiffness_zones must assign every story exactly once."
        )

    return zone_names, zone_components


def apply_stiffness_multipliers(
    zone_components: ArrayLike,
    stiffness_multipliers: ArrayLike,
) -> FloatArray:
    """Combine zonal matrices using physical multipliers ``0 < theta <= 1``."""

    components = np.asarray(zone_components, dtype=np.float64)
    multipliers = np.asarray(stiffness_multipliers, dtype=np.float64)

    if components.ndim != 3 or components.shape[1] != components.shape[2]:
        raise StructuralMatrixError(
            "zone_components must have shape (n_zone, n_dof, n_dof)."
        )
    if multipliers.ndim != 1 or multipliers.size != components.shape[0]:
        raise StructuralMatrixError(
            "stiffness_multipliers must contain one value per zone."
        )
    if not np.all(np.isfinite(components)) or not np.all(np.isfinite(multipliers)):
        raise StructuralMatrixError(
            "Stiffness components and multipliers must be finite."
        )
    if np.any(multipliers <= 0.0) or np.any(multipliers > 1.0):
        raise StructuralMatrixError(
            "Each stiffness multiplier must satisfy 0 < theta <= 1."
        )

    return np.einsum("z,zij->ij", multipliers, components)


def validate_structural_matrices(
    mass_matrix_kg: ArrayLike,
    stiffness_matrix_n_per_m: ArrayLike,
    *,
    symmetry_atol: float = 1.0e-10,
) -> MatrixValidationReport:
    """Validate dimensions, symmetry, finiteness, and positive definiteness."""

    mass = np.asarray(mass_matrix_kg, dtype=np.float64)
    stiffness = np.asarray(stiffness_matrix_n_per_m, dtype=np.float64)

    for matrix, name in ((mass, "mass"), (stiffness, "stiffness")):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise StructuralMatrixError(f"The {name} matrix must be square.")
        if not np.all(np.isfinite(matrix)):
            raise StructuralMatrixError(f"The {name} matrix must be finite.")
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=symmetry_atol):
            raise StructuralMatrixError(f"The {name} matrix must be symmetric.")

    if mass.shape != stiffness.shape:
        raise StructuralMatrixError(
            "Mass and stiffness matrices must have identical dimensions."
        )

    mass_eigenvalues = np.linalg.eigvalsh(mass)
    stiffness_eigenvalues = np.linalg.eigvalsh(stiffness)

    if mass_eigenvalues[0] <= 0.0:
        raise StructuralMatrixError("The mass matrix must be positive definite.")
    if stiffness_eigenvalues[0] <= 0.0:
        raise StructuralMatrixError("The stiffness matrix must be positive definite.")

    return MatrixValidationReport(
        n_dof=mass.shape[0],
        minimum_mass_eigenvalue_kg=float(mass_eigenvalues[0]),
        minimum_stiffness_eigenvalue_n_per_m=float(stiffness_eigenvalues[0]),
        mass_condition_number=float(np.linalg.cond(mass)),
        stiffness_condition_number=float(np.linalg.cond(stiffness)),
    )
