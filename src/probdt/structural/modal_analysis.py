"""Modal analysis and proportional-damping utilities.

The undamped modes solve the generalized symmetric eigenproblem

``K phi_i = omega_i**2 M phi_i``.

Modes are explicitly mass-normalized, giving ``Phi.T M Phi = I``. Rayleigh
damping is constructed as ``C = alpha M + beta K`` and is therefore diagonal in
the same modal basis. All calculations use SI units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh

from probdt.structural.matrices import (
    StructuralMatrixError,
    validate_structural_matrices,
)

FloatArray = NDArray[np.float64]


class ModalAnalysisError(ValueError):
    """Raised when modal or damping inputs are mathematically invalid."""


@dataclass(frozen=True)
class ModalProperties:
    """Mass-normalized modal properties ordered by increasing frequency."""

    angular_frequencies_rad_s: FloatArray
    frequencies_hz: FloatArray
    mode_shapes: FloatArray
    modal_mass_matrix: FloatArray
    modal_stiffness_matrix: FloatArray
    participation_factors: FloatArray
    effective_modal_masses_kg: FloatArray
    effective_modal_mass_ratios: FloatArray
    relative_eigen_residuals: FloatArray

    @property
    def cumulative_effective_modal_mass_ratio(self) -> FloatArray:
        """Return cumulative effective mass in ascending modal order."""

        return np.cumsum(self.effective_modal_mass_ratios)


@dataclass(frozen=True)
class RayleighDampingModel:
    """Rayleigh coefficients and the assembled physical damping matrix."""

    alpha_per_s: float
    beta_s: float
    damping_matrix_n_s_per_m: FloatArray


def _matrix_pair(
    mass_matrix_kg: ArrayLike,
    stiffness_matrix_n_per_m: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return validated float copies of the mass and stiffness matrices."""

    mass = np.asarray(mass_matrix_kg, dtype=np.float64)
    stiffness = np.asarray(stiffness_matrix_n_per_m, dtype=np.float64)
    try:
        validate_structural_matrices(mass, stiffness)
    except StructuralMatrixError as error:
        raise ModalAnalysisError(str(error)) from error
    return mass.copy(), stiffness.copy()


def solve_modal_properties(
    mass_matrix_kg: ArrayLike,
    stiffness_matrix_n_per_m: ArrayLike,
    *,
    n_modes: int | None = None,
    influence_vector: ArrayLike | None = None,
) -> ModalProperties:
    """Solve and verify the mass-normalized generalized eigenproblem.

    The default influence vector is a vector of ones, corresponding to uniform
    support acceleration in the single lateral direction of the shear model.
    """

    mass, stiffness = _matrix_pair(mass_matrix_kg, stiffness_matrix_n_per_m)
    n_dof = mass.shape[0]

    if n_modes is None:
        n_modes = n_dof
    if isinstance(n_modes, bool) or not isinstance(n_modes, int):
        raise ModalAnalysisError("n_modes must be an integer or None.")
    if not 1 <= n_modes <= n_dof:
        raise ModalAnalysisError(f"n_modes must be within [1, {n_dof}].")

    eigenvalues, mode_shapes = eigh(
        stiffness,
        mass,
        subset_by_index=(0, n_modes - 1),
        check_finite=True,
    )
    if np.any(eigenvalues <= 0.0):
        raise ModalAnalysisError("All retained eigenvalues must be positive.")

    # Re-normalize explicitly rather than relying on a library convention. The
    # sign rule makes repeated runs and visual comparisons deterministic.
    for mode_index in range(n_modes):
        mode = mode_shapes[:, mode_index]
        modal_mass = float(mode.T @ mass @ mode)
        mode /= np.sqrt(modal_mass)
        reference_dof = int(np.argmax(np.abs(mode)))
        if mode[reference_dof] < 0.0:
            mode *= -1.0

    angular_frequencies = np.sqrt(eigenvalues)
    frequencies_hz = angular_frequencies / (2.0 * np.pi)
    modal_mass_matrix = mode_shapes.T @ mass @ mode_shapes
    modal_stiffness_matrix = mode_shapes.T @ stiffness @ mode_shapes

    if influence_vector is None:
        influence = np.ones(n_dof, dtype=np.float64)
    else:
        influence = np.asarray(influence_vector, dtype=np.float64)
        if influence.ndim != 1 or influence.size != n_dof:
            raise ModalAnalysisError(
                f"influence_vector must have shape ({n_dof},)."
            )
        if not np.all(np.isfinite(influence)):
            raise ModalAnalysisError("influence_vector must contain finite values.")

    total_generalized_mass = float(influence.T @ mass @ influence)
    if total_generalized_mass <= 0.0:
        raise ModalAnalysisError("influence_vector must excite nonzero mass.")

    modal_masses = np.diag(modal_mass_matrix)
    generalized_forces = mode_shapes.T @ mass @ influence
    participation_factors = generalized_forces / modal_masses
    effective_modal_masses = generalized_forces**2 / modal_masses
    effective_modal_mass_ratios = effective_modal_masses / total_generalized_mass

    relative_residuals = np.empty(n_modes, dtype=np.float64)
    for mode_index, eigenvalue in enumerate(eigenvalues):
        mode = mode_shapes[:, mode_index]
        stiffness_term = stiffness @ mode
        inertia_term = eigenvalue * (mass @ mode)
        denominator = np.linalg.norm(stiffness_term) + np.linalg.norm(inertia_term)
        relative_residuals[mode_index] = (
            np.linalg.norm(stiffness_term - inertia_term) / denominator
        )

    return ModalProperties(
        angular_frequencies_rad_s=angular_frequencies,
        frequencies_hz=frequencies_hz,
        mode_shapes=mode_shapes,
        modal_mass_matrix=modal_mass_matrix,
        modal_stiffness_matrix=modal_stiffness_matrix,
        participation_factors=participation_factors,
        effective_modal_masses_kg=effective_modal_masses,
        effective_modal_mass_ratios=effective_modal_mass_ratios,
        relative_eigen_residuals=relative_residuals,
    )


def modal_orthogonality_errors(properties: ModalProperties) -> tuple[float, float]:
    """Return mass and relative stiffness diagonalization errors."""

    n_modes = properties.mode_shapes.shape[1]
    mass_target = np.eye(n_modes)
    stiffness_target = np.diag(properties.angular_frequencies_rad_s**2)

    mass_error = float(
        np.linalg.norm(properties.modal_mass_matrix - mass_target, ord="fro")
    )
    stiffness_error = float(
        np.linalg.norm(
            properties.modal_stiffness_matrix - stiffness_target,
            ord="fro",
        )
        / np.linalg.norm(stiffness_target, ord="fro")
    )
    return mass_error, stiffness_error


def fit_rayleigh_coefficients(
    first_angular_frequency_rad_s: float,
    second_angular_frequency_rad_s: float,
    first_damping_ratio: float,
    second_damping_ratio: float,
) -> tuple[float, float]:
    """Fit nonnegative Rayleigh coefficients to two modal damping targets.

    Rayleigh damping gives

    ``zeta_i = alpha / (2 omega_i) + beta omega_i / 2``.
    """

    frequencies = np.asarray(
        [first_angular_frequency_rad_s, second_angular_frequency_rad_s],
        dtype=np.float64,
    )
    targets = np.asarray(
        [first_damping_ratio, second_damping_ratio],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ModalAnalysisError(
            "Target angular frequencies must be finite and positive."
        )
    if np.isclose(frequencies[0], frequencies[1], rtol=1.0e-12, atol=0.0):
        raise ModalAnalysisError("Rayleigh target frequencies must be distinct.")
    if not np.all(np.isfinite(targets)) or np.any(targets < 0.0):
        raise ModalAnalysisError(
            "Target damping ratios must be finite and nonnegative."
        )

    system = np.column_stack((1.0 / (2.0 * frequencies), frequencies / 2.0))
    alpha, beta = np.linalg.solve(system, targets)
    numerical_tolerance = 1.0e-14
    if alpha < -numerical_tolerance or beta < -numerical_tolerance:
        raise ModalAnalysisError(
            "The requested targets require a negative Rayleigh coefficient."
        )
    return max(float(alpha), 0.0), max(float(beta), 0.0)


def assemble_rayleigh_damping(
    mass_matrix_kg: ArrayLike,
    stiffness_matrix_n_per_m: ArrayLike,
    *,
    alpha_per_s: float,
    beta_s: float,
) -> RayleighDampingModel:
    """Assemble ``C = alpha M + beta K`` in N s/m."""

    mass, stiffness = _matrix_pair(mass_matrix_kg, stiffness_matrix_n_per_m)
    coefficients = np.asarray([alpha_per_s, beta_s], dtype=np.float64)
    if not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0.0):
        raise ModalAnalysisError(
            "Rayleigh coefficients must be finite and nonnegative."
        )
    if np.all(coefficients == 0.0):
        raise ModalAnalysisError("At least one Rayleigh coefficient must be positive.")

    damping = float(alpha_per_s) * mass + float(beta_s) * stiffness
    if not np.allclose(damping, damping.T, rtol=0.0, atol=1.0e-10):
        raise ModalAnalysisError("The Rayleigh damping matrix is not symmetric.")
    if np.linalg.eigvalsh(damping)[0] <= 0.0:
        raise ModalAnalysisError(
            "The Rayleigh damping matrix is not positive definite."
        )

    return RayleighDampingModel(
        alpha_per_s=float(alpha_per_s),
        beta_s=float(beta_s),
        damping_matrix_n_s_per_m=damping,
    )


def calculate_modal_damping_ratios(
    damping_matrix_n_s_per_m: ArrayLike,
    mass_matrix_kg: ArrayLike,
    properties: ModalProperties,
) -> FloatArray:
    """Project physical damping onto each retained undamped mode."""

    damping = np.asarray(damping_matrix_n_s_per_m, dtype=np.float64)
    mass = np.asarray(mass_matrix_kg, dtype=np.float64)
    n_dof = properties.mode_shapes.shape[0]
    if damping.shape != (n_dof, n_dof) or mass.shape != (n_dof, n_dof):
        raise ModalAnalysisError("Mass and damping matrices must match the mode size.")
    if not np.all(np.isfinite(damping)) or not np.all(np.isfinite(mass)):
        raise ModalAnalysisError("Mass and damping matrices must be finite.")

    modal_damping = properties.mode_shapes.T @ damping @ properties.mode_shapes
    modal_mass = properties.mode_shapes.T @ mass @ properties.mode_shapes
    return np.diag(modal_damping) / (
        2.0
        * properties.angular_frequencies_rad_s
        * np.diag(modal_mass)
    )
