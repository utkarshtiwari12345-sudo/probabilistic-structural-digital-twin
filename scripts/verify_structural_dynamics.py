"""Run the Stage 1C structural-dynamics verification gate."""

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
    build_continuous_state_space,
    calculate_energy_balance,
    discretize_state_space_zoh,
    fit_rayleigh_coefficients,
    simulate_discrete_state_space,
    simulate_newmark_average_acceleration,
    solve_modal_properties,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage1c_dynamic_verification.png"


def uniform_time(duration_s: float, time_step_s: float) -> np.ndarray:
    """Create an endpoint-inclusive uniform grid without cumulative drift."""

    n_steps = round(duration_s / time_step_s)
    if not np.isclose(n_steps * time_step_s, duration_s):
        raise ValueError("duration_s must be an integer multiple of time_step_s.")
    return np.arange(n_steps + 1, dtype=np.float64) * time_step_s


def harmonic_force_history(
    time_s: np.ndarray,
    n_dof: int,
    force_dof: int,
    amplitude_n: float,
    angular_frequency_rad_s: float,
) -> np.ndarray:
    """Return a single-DOF harmonic physical-force history."""

    forces = np.zeros((time_s.size, n_dof), dtype=np.float64)
    forces[:, force_dof] = amplitude_n * np.sin(angular_frequency_rad_s * time_s)
    return forces


def relative_rms_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Return RMS error normalized by the reference RMS amplitude."""

    return float(
        np.sqrt(np.mean((estimate - reference) ** 2))
        / np.sqrt(np.mean(reference**2))
    )


def create_verification_figure(
    free_time: np.ndarray,
    free_top_displacement: np.ndarray,
    normalized_energy: np.ndarray,
    forced_time: np.ndarray,
    forced_top_displacement: np.ndarray,
    force_history: np.ndarray,
    time_steps: np.ndarray,
    convergence_errors: np.ndarray,
) -> None:
    """Save response, energy, equilibrium context, and convergence evidence."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.0),
        constrained_layout=True,
    )
    free_axis, energy_axis, forced_axis, convergence_axis = axes.ravel()

    free_axis.plot(free_time, 1000.0 * free_top_displacement, linewidth=1.3)
    free_axis.set_xlabel("Time (s)")
    free_axis.set_ylabel("Top displacement (mm)")
    free_axis.set_title("Free response from first-mode displacement")
    free_axis.grid(alpha=0.25)

    energy_axis.semilogy(free_time, normalized_energy, linewidth=1.5, color="#2A9D8F")
    energy_axis.set_xlabel("Time (s)")
    energy_axis.set_ylabel("Mechanical energy / initial energy")
    energy_axis.set_title("Monotonic damped-energy decay")
    energy_axis.grid(alpha=0.25, which="both")

    forced_axis.plot(
        forced_time,
        1000.0 * forced_top_displacement,
        linewidth=1.3,
        label="Top displacement",
    )
    forced_axis.set_xlabel("Time (s)")
    forced_axis.set_ylabel("Top displacement (mm)")
    forced_axis.set_title("Harmonic forced response")
    forced_axis.grid(alpha=0.25)
    force_axis = forced_axis.twinx()
    force_axis.plot(
        forced_time,
        force_history / 1000.0,
        color="0.45",
        alpha=0.35,
        linewidth=0.8,
        label="Applied force",
    )
    force_axis.set_ylabel("Applied force (kN)", color="0.35")

    convergence_axis.loglog(
        time_steps,
        convergence_errors,
        marker="o",
        linewidth=1.5,
        color="#A23B72",
        label="Observed error",
    )
    second_order_reference = convergence_errors[-1] * (
        time_steps / time_steps[-1]
    ) ** 2
    convergence_axis.loglog(
        time_steps,
        second_order_reference,
        linestyle="--",
        color="0.35",
        label=r"Reference $O(\Delta t^2)$",
    )
    convergence_axis.set_xlabel("Time step (s)")
    convergence_axis.set_ylabel("Relative RMS displacement error")
    convergence_axis.set_title("Time-step convergence")
    convergence_axis.grid(alpha=0.25, which="both")
    convergence_axis.legend(frameon=False)

    figure.suptitle("Stage 1C: structural dynamics verification", fontweight="bold")
    figure.savefig(FIGURE_PATH, dpi=200)
    plt.close(figure)


def main() -> None:
    """Run all Stage 1C numerical and physical consistency checks."""

    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model_config = config["model"]
    structural = config["structural_properties"]
    damping_config = config["damping"]
    dynamics_config = config["dynamics"]
    verification = config["verification"]

    mass = assemble_mass_matrix(structural["masses_kg"])
    stiffness = assemble_stiffness_matrix(structural["story_stiffness_n_per_m"])
    modes = solve_modal_properties(mass, stiffness)
    target_modes = np.asarray(damping_config["target_modes"], dtype=int) - 1
    target_ratios = np.asarray(
        damping_config["target_modal_damping_ratios"],
        dtype=np.float64,
    )
    alpha, beta = fit_rayleigh_coefficients(
        modes.angular_frequencies_rad_s[target_modes[0]],
        modes.angular_frequencies_rad_s[target_modes[1]],
        target_ratios[0],
        target_ratios[1],
    )
    damping = assemble_rayleigh_damping(
        mass,
        stiffness,
        alpha_per_s=alpha,
        beta_s=beta,
    ).damping_matrix_n_s_per_m
    n_dof = mass.shape[0]

    top_force_distribution = np.zeros(n_dof, dtype=np.float64)
    top_force_distribution[-1] = 1.0
    continuous = build_continuous_state_space(
        mass,
        damping,
        stiffness,
        force_distribution=top_force_distribution,
    )
    state_eigenvalues = np.linalg.eigvals(continuous.state_matrix_per_s)
    maximum_state_real_part = float(np.max(state_eigenvalues.real))

    free_config = dynamics_config["free_response"]
    free_time = uniform_time(
        float(free_config["duration_s"]),
        float(free_config["time_step_s"]),
    )
    free_force = np.zeros((free_time.size, n_dof), dtype=np.float64)
    initial_displacement = modes.mode_shapes[:, 0].copy()
    initial_displacement *= (
        float(free_config["initial_top_displacement_m"])
        / initial_displacement[-1]
    )
    free_response = simulate_newmark_average_acceleration(
        mass,
        damping,
        stiffness,
        free_time,
        free_force,
        initial_displacement_m=initial_displacement,
    )
    free_energy = calculate_energy_balance(
        free_response,
        mass,
        damping,
        stiffness,
        free_force,
    )
    initial_energy = float(free_energy.mechanical_energy_j[0])
    maximum_energy_increase_fraction = float(
        np.max(np.diff(free_energy.mechanical_energy_j)) / initial_energy
    )
    normalized_energy_balance_error = float(
        np.max(np.abs(free_energy.balance_residual_j)) / initial_energy
    )

    discrete = discretize_state_space_zoh(
        continuous,
        float(free_config["time_step_s"]),
    )
    initial_state = np.concatenate((initial_displacement, np.zeros(n_dof)))
    exact_states = simulate_discrete_state_space(
        discrete,
        np.zeros((free_time.size, 1)),
        initial_state,
    )
    state_space_agreement_error = relative_rms_error(
        free_response.displacement_m,
        exact_states[:, :n_dof],
    )

    forced_config = dynamics_config["forced_response"]
    forced_time = uniform_time(
        float(forced_config["duration_s"]),
        float(forced_config["time_step_s"]),
    )
    force_dof = int(forced_config["force_dof_one_based"]) - 1
    forcing_angular_frequency = (
        float(forced_config["forcing_frequency_ratio_to_first_mode"])
        * modes.angular_frequencies_rad_s[0]
    )
    forced_load = harmonic_force_history(
        forced_time,
        n_dof,
        force_dof,
        float(forced_config["force_amplitude_n"]),
        forcing_angular_frequency,
    )
    forced_response = simulate_newmark_average_acceleration(
        mass,
        damping,
        stiffness,
        forced_time,
        forced_load,
    )
    equilibrium_residual = (
        forced_response.acceleration_m_per_s2 @ mass.T
        + forced_response.velocity_m_per_s @ damping.T
        + forced_response.displacement_m @ stiffness.T
        - forced_load
    )
    equilibrium_scale = max(float(np.max(np.abs(forced_load))), 1.0)
    normalized_equilibrium_residual = float(
        np.max(np.abs(equilibrium_residual)) / equilibrium_scale
    )

    convergence_config = dynamics_config["convergence"]
    convergence_duration = float(convergence_config["duration_s"])
    reference_time_step = float(convergence_config["reference_time_step_s"])
    reference_time = uniform_time(convergence_duration, reference_time_step)
    reference_force = harmonic_force_history(
        reference_time,
        n_dof,
        force_dof,
        float(forced_config["force_amplitude_n"]),
        forcing_angular_frequency,
    )
    reference_response = simulate_newmark_average_acceleration(
        mass,
        damping,
        stiffness,
        reference_time,
        reference_force,
    )

    time_steps = np.asarray(convergence_config["time_steps_s"], dtype=np.float64)
    convergence_errors = np.empty(time_steps.size, dtype=np.float64)
    for index, time_step in enumerate(time_steps):
        time = uniform_time(convergence_duration, float(time_step))
        force = harmonic_force_history(
            time,
            n_dof,
            force_dof,
            float(forced_config["force_amplitude_n"]),
            forcing_angular_frequency,
        )
        response = simulate_newmark_average_acceleration(
            mass,
            damping,
            stiffness,
            time,
            force,
        )
        stride = round(float(time_step) / reference_time_step)
        reference_at_grid = reference_response.displacement_m[::stride]
        if reference_at_grid.shape != response.displacement_m.shape:
            raise RuntimeError("Reference and convergence grids are inconsistent.")
        convergence_errors[index] = relative_rms_error(
            response.displacement_m,
            reference_at_grid,
        )

    observed_orders = np.log(
        convergence_errors[:-1] / convergence_errors[1:]
    ) / np.log(time_steps[:-1] / time_steps[1:])
    minimum_observed_order = float(np.min(observed_orders))

    checks = {
        "asymptotically stable state matrix": maximum_state_real_part < 0.0,
        "non-increasing free-response energy": maximum_energy_increase_fraction
        <= float(verification["maximum_energy_increase_fraction"]),
        "energy balance": normalized_energy_balance_error
        <= float(verification["energy_balance_rtol"]),
        "state-space and Newmark agreement": state_space_agreement_error
        <= float(verification["state_space_agreement_rtol"]),
        "forced equilibrium": normalized_equilibrium_residual
        <= float(verification["equilibrium_residual_rtol"]),
        "second-order convergence": minimum_observed_order
        >= float(verification["minimum_convergence_order"]),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise RuntimeError(f"Stage 1C checks failed: {', '.join(failed_checks)}")

    create_verification_figure(
        free_time,
        free_response.displacement_m[:, -1],
        free_energy.mechanical_energy_j / initial_energy,
        forced_time,
        forced_response.displacement_m[:, -1],
        forced_load[:, force_dof],
        time_steps,
        convergence_errors,
    )

    print("STAGE 1C - STRUCTURAL DYNAMICS VERIFICATION")
    print(f"Model: {model_config['name']}")
    print(f"Continuous state dimension: {continuous.state_matrix_per_s.shape[0]}")
    print(f"Maximum Re[eigenvalue(A)]: {maximum_state_real_part:.6e} 1/s")
    print(
        "Maximum normalized energy increase: "
        f"{maximum_energy_increase_fraction:.6e}"
    )
    print(
        "Normalized energy-balance error: "
        f"{normalized_energy_balance_error:.6e}"
    )
    print(
        "Newmark/state-space relative RMS difference: "
        f"{state_space_agreement_error:.6e}"
    )
    print(
        "Normalized forced-equilibrium residual: "
        f"{normalized_equilibrium_residual:.6e}"
    )
    print("")
    print("dt (s) | Relative RMS error")
    print("-------+-------------------")
    for time_step, error in zip(time_steps, convergence_errors, strict=True):
        print(f"{time_step:>6.4f} | {error:.6e}")
    print(f"Observed orders: {', '.join(f'{order:.4f}' for order in observed_orders)}")
    print(f"Minimum observed order: {minimum_observed_order:.4f}")
    peak_forced_displacement = np.max(
        np.abs(forced_response.displacement_m[:, -1])
    )
    print(f"Peak forced top displacement: {peak_forced_displacement:.6e} m")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
