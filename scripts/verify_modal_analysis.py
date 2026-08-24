"""Run the Stage 1B modal-analysis and Rayleigh-damping verification gate."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from probdt.structural import (  # noqa: E402
    assemble_mass_matrix,
    assemble_rayleigh_damping,
    assemble_stiffness_matrix,
    calculate_modal_damping_ratios,
    fit_rayleigh_coefficients,
    modal_orthogonality_errors,
    solve_modal_properties,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage1b_modal_properties.png"


def create_verification_figure(
    level_spacing_m: float,
    frequencies_hz: np.ndarray,
    mode_shapes: np.ndarray,
    damping_ratios: np.ndarray,
    target_modes: np.ndarray,
) -> None:
    """Save mode-shape and modal-damping evidence in one compact figure."""

    elevations = level_spacing_m * np.arange(1, mode_shapes.shape[0] + 1)
    elevations_with_base = np.concatenate(([0.0], elevations))

    figure, (shape_axis, damping_axis) = plt.subplots(
        1,
        2,
        figsize=(11.0, 5.0),
        constrained_layout=True,
    )

    for mode_index in range(min(3, mode_shapes.shape[1])):
        plotted_mode = mode_shapes[:, mode_index]
        plotted_mode = plotted_mode / np.max(np.abs(plotted_mode))
        plotted_mode = np.concatenate(([0.0], plotted_mode))
        shape_axis.plot(
            plotted_mode,
            elevations_with_base,
            marker="o",
            linewidth=1.8,
            label=f"Mode {mode_index + 1}: {frequencies_hz[mode_index]:.3f} Hz",
        )

    shape_axis.axvline(0.0, color="0.5", linewidth=0.8)
    shape_axis.set_xlabel("Mode amplitude (max-normalized for display)")
    shape_axis.set_ylabel("Elevation (m)")
    shape_axis.set_title("First three lateral mode shapes")
    shape_axis.grid(alpha=0.25)
    shape_axis.legend(frameon=False)

    mode_numbers = np.arange(1, frequencies_hz.size + 1)
    damping_axis.plot(
        mode_numbers,
        100.0 * damping_ratios,
        marker="o",
        color="#A23B72",
        linewidth=1.8,
        label="Rayleigh modal damping",
    )
    damping_axis.scatter(
        target_modes + 1,
        100.0 * damping_ratios[target_modes],
        s=90,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        zorder=3,
        label="Fitted target modes",
    )
    damping_axis.set_xticks(mode_numbers)
    damping_axis.set_xlabel("Mode number")
    damping_axis.set_ylabel("Modal damping ratio (%)")
    damping_axis.set_title("Damping implied across all modes")
    damping_axis.grid(alpha=0.25)
    damping_axis.legend(frameon=False)

    figure.suptitle("Stage 1B: verified modal properties", fontweight="bold")
    figure.savefig(FIGURE_PATH, dpi=200)
    plt.close(figure)


def main() -> None:
    """Verify modal solution, participation, and Rayleigh damping targets."""

    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model = config["model"]
    structural = config["structural_properties"]
    damping_config = config["damping"]
    verification = config["verification"]

    mass = assemble_mass_matrix(structural["masses_kg"])
    stiffness = assemble_stiffness_matrix(structural["story_stiffness_n_per_m"])
    properties = solve_modal_properties(mass, stiffness)

    target_modes = np.asarray(damping_config["target_modes"], dtype=int) - 1
    target_ratios = np.asarray(
        damping_config["target_modal_damping_ratios"],
        dtype=np.float64,
    )
    if target_modes.shape != (2,) or target_ratios.shape != (2,):
        raise ValueError("Rayleigh damping requires exactly two configured targets.")
    if np.any(target_modes < 0) or np.any(target_modes >= model["n_dof"]):
        raise ValueError("Configured damping target mode is outside the model.")

    alpha, beta = fit_rayleigh_coefficients(
        properties.angular_frequencies_rad_s[target_modes[0]],
        properties.angular_frequencies_rad_s[target_modes[1]],
        target_ratios[0],
        target_ratios[1],
    )
    damping = assemble_rayleigh_damping(
        mass,
        stiffness,
        alpha_per_s=alpha,
        beta_s=beta,
    )
    modal_damping = calculate_modal_damping_ratios(
        damping.damping_matrix_n_s_per_m,
        mass,
        properties,
    )

    mass_error, stiffness_error = modal_orthogonality_errors(properties)
    max_residual = float(np.max(properties.relative_eigen_residuals))
    effective_mass_sum_error = abs(
        float(np.sum(properties.effective_modal_mass_ratios)) - 1.0
    )
    target_error = float(np.max(np.abs(modal_damping[target_modes] - target_ratios)))

    checks = {
        "mass orthogonality": mass_error
        <= float(verification["mass_orthogonality_atol"]),
        "stiffness diagonalization": stiffness_error
        <= float(verification["stiffness_diagonalization_rtol"]),
        "generalized eigen residual": max_residual
        <= float(verification["eigen_residual_rtol"]),
        "effective modal mass completeness": effective_mass_sum_error
        <= float(verification["effective_mass_sum_atol"]),
        "Rayleigh target reproduction": target_error
        <= float(verification["damping_ratio_atol"]),
        "positive damping ratios": bool(np.all(modal_damping > 0.0)),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise RuntimeError(f"Stage 1B checks failed: {', '.join(failed_checks)}")

    create_verification_figure(
        float(model["level_spacing_m"]),
        properties.frequencies_hz,
        properties.mode_shapes,
        modal_damping,
        target_modes,
    )

    print("STAGE 1B - MODAL AND RAYLEIGH DAMPING VERIFICATION")
    print(f"Model: {model['name']}")
    print(f"Rayleigh alpha: {alpha:.8e} 1/s")
    print(f"Rayleigh beta:  {beta:.8e} s")
    print("")
    print("Mode | Frequency (Hz) | Damping (%) | Effective mass (%) | Cumulative (%)")
    print("-----+----------------+-------------+--------------------+---------------")
    for mode_index in range(model["n_dof"]):
        cumulative_ratio = properties.cumulative_effective_modal_mass_ratio[
            mode_index
        ]
        print(
            f"{mode_index + 1:>4d} | "
            f"{properties.frequencies_hz[mode_index]:>14.6f} | "
            f"{100.0 * modal_damping[mode_index]:>11.6f} | "
            f"{100.0 * properties.effective_modal_mass_ratios[mode_index]:>18.6f} | "
            f"{100.0 * cumulative_ratio:>13.6f}"
        )
    print("")
    print(f"Mass orthogonality error: {mass_error:.6e}")
    print(f"Relative stiffness diagonalization error: {stiffness_error:.6e}")
    print(f"Maximum relative eigen residual: {max_residual:.6e}")
    print(f"Effective modal mass sum error: {effective_mass_sum_error:.6e}")
    print(f"Maximum damping target error: {target_error:.6e}")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
