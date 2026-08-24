"""Verification tests for modal analysis and Rayleigh damping."""

import numpy as np
import pytest

from probdt.structural import (
    ModalAnalysisError,
    assemble_mass_matrix,
    assemble_rayleigh_damping,
    assemble_stiffness_matrix,
    calculate_modal_damping_ratios,
    fit_rayleigh_coefficients,
    modal_orthogonality_errors,
    solve_modal_properties,
)


@pytest.fixture
def six_dof_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed Stage 1 reference matrices."""

    mass = assemble_mass_matrix(
        [420000.0, 400000.0, 380000.0, 360000.0, 340000.0, 520000.0]
    )
    stiffness = assemble_stiffness_matrix(
        [30000000.0, 28000000.0, 26000000.0, 24000000.0, 22000000.0, 20000000.0]
    )
    return mass, stiffness


def test_single_dof_frequency_matches_closed_form_solution() -> None:
    mass = np.array([[4.0]])
    stiffness = np.array([[100.0]])

    properties = solve_modal_properties(mass, stiffness)

    expected_angular_frequency = np.sqrt(100.0 / 4.0)
    np.testing.assert_allclose(
        properties.angular_frequencies_rad_s,
        [expected_angular_frequency],
        rtol=1.0e-14,
    )
    modal_mass = properties.mode_shapes.T @ mass @ properties.mode_shapes
    np.testing.assert_allclose(modal_mass, 1.0)


def test_reference_frequencies_are_stable_regression_values(
    six_dof_matrices: tuple[np.ndarray, np.ndarray],
) -> None:
    mass, stiffness = six_dof_matrices
    properties = solve_modal_properties(mass, stiffness)
    expected_hz = np.array(
        [
            0.308914965325132,
            0.852955990294804,
            1.393888531359764,
            1.874433708950105,
            2.250115288005738,
            2.499607236655644,
        ]
    )

    np.testing.assert_allclose(properties.frequencies_hz, expected_hz, rtol=1.0e-12)


def test_modes_are_mass_orthonormal_and_stiffness_orthogonal(
    six_dof_matrices: tuple[np.ndarray, np.ndarray],
) -> None:
    mass, stiffness = six_dof_matrices
    properties = solve_modal_properties(mass, stiffness)
    mass_error, stiffness_error = modal_orthogonality_errors(properties)

    assert mass_error < 1.0e-12
    assert stiffness_error < 1.0e-12


def test_generalized_eigen_residuals_are_small(
    six_dof_matrices: tuple[np.ndarray, np.ndarray],
) -> None:
    mass, stiffness = six_dof_matrices
    properties = solve_modal_properties(mass, stiffness)

    assert np.max(properties.relative_eigen_residuals) < 1.0e-12


def test_complete_modes_recover_total_effective_mass(
    six_dof_matrices: tuple[np.ndarray, np.ndarray],
) -> None:
    mass, stiffness = six_dof_matrices
    properties = solve_modal_properties(mass, stiffness)

    assert np.sum(properties.effective_modal_mass_ratios) == pytest.approx(
        1.0,
        abs=1.0e-12,
    )
    assert np.all(properties.effective_modal_mass_ratios >= 0.0)


def test_mode_sign_convention_is_deterministic(
    six_dof_matrices: tuple[np.ndarray, np.ndarray],
) -> None:
    mass, stiffness = six_dof_matrices
    properties = solve_modal_properties(mass, stiffness)

    for mode in properties.mode_shapes.T:
        largest_component = mode[np.argmax(np.abs(mode))]
        assert largest_component > 0.0


def test_rayleigh_damping_hits_modes_one_and_three(
    six_dof_matrices: tuple[np.ndarray, np.ndarray],
) -> None:
    mass, stiffness = six_dof_matrices
    properties = solve_modal_properties(mass, stiffness)
    target_modes = np.array([0, 2])
    target_ratio = 0.015
    alpha, beta = fit_rayleigh_coefficients(
        properties.angular_frequencies_rad_s[target_modes[0]],
        properties.angular_frequencies_rad_s[target_modes[1]],
        target_ratio,
        target_ratio,
    )
    damping = assemble_rayleigh_damping(
        mass,
        stiffness,
        alpha_per_s=alpha,
        beta_s=beta,
    )
    ratios = calculate_modal_damping_ratios(
        damping.damping_matrix_n_s_per_m,
        mass,
        properties,
    )

    np.testing.assert_allclose(ratios[target_modes], target_ratio, atol=1.0e-14)
    assert np.all(ratios > 0.0)
    np.testing.assert_allclose(
        damping.damping_matrix_n_s_per_m,
        damping.damping_matrix_n_s_per_m.T,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.linalg.eigvalsh(damping.damping_matrix_n_s_per_m)[0] > 0.0


@pytest.mark.parametrize("n_modes", [0, 7, 1.5, True])
def test_invalid_mode_count_is_rejected(
    six_dof_matrices: tuple[np.ndarray, np.ndarray],
    n_modes: object,
) -> None:
    mass, stiffness = six_dof_matrices

    with pytest.raises(ModalAnalysisError, match="n_modes"):
        solve_modal_properties(mass, stiffness, n_modes=n_modes)  # type: ignore[arg-type]


def test_zero_influence_vector_is_rejected(
    six_dof_matrices: tuple[np.ndarray, np.ndarray],
) -> None:
    mass, stiffness = six_dof_matrices

    with pytest.raises(ModalAnalysisError, match="nonzero mass"):
        solve_modal_properties(mass, stiffness, influence_vector=np.zeros(6))


@pytest.mark.parametrize(
    ("frequencies", "ratios", "message"),
    [
        ((0.0, 2.0), (0.01, 0.01), "positive"),
        ((2.0, 2.0), (0.01, 0.01), "distinct"),
        ((1.0, 2.0), (-0.01, 0.01), "nonnegative"),
    ],
)
def test_invalid_rayleigh_targets_are_rejected(
    frequencies: tuple[float, float],
    ratios: tuple[float, float],
    message: str,
) -> None:
    with pytest.raises(ModalAnalysisError, match=message):
        fit_rayleigh_coefficients(*frequencies, *ratios)
