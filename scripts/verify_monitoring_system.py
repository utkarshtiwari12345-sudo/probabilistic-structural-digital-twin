"""Run the Stage 3 sparse synthetic-monitoring verification gate."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from probdt.monitoring import (  # noqa: E402
    AccelerationMeasurements,
    ContiguousOutage,
    SensorLayout,
    create_acceleration_measurements,
    distribute_scalar_force,
    extract_clean_acceleration,
    generate_availability_mask,
    generate_ornstein_uhlenbeck_force,
)
from probdt.structural import (  # noqa: E402
    TimeResponse,
    assemble_mass_matrix,
    assemble_rayleigh_damping,
    assemble_stiffness_matrix,
    fit_rayleigh_coefficients,
    simulate_newmark_average_acceleration,
    solve_modal_properties,
)

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "stage3_sparse_monitoring.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage3_sparse_monitoring.png"


def uniform_time(duration_s: float, time_step_s: float) -> np.ndarray:
    """Create an endpoint-inclusive uniform time grid."""

    n_steps = round(duration_s / time_step_s)
    if n_steps < 1 or not np.isclose(n_steps * time_step_s, duration_s):
        raise ValueError("duration_s must contain complete time steps.")
    return np.arange(n_steps + 1, dtype=np.float64) * time_step_s


def trim_response(
    response: TimeResponse,
    start_s: float,
) -> TimeResponse:
    """Discard the structural transient and reset retained within-window time."""

    retained = response.time_s >= start_s
    return TimeResponse(
        time_s=response.time_s[retained] - start_s,
        displacement_m=response.displacement_m[retained].copy(),
        velocity_m_per_s=response.velocity_m_per_s[retained].copy(),
        acceleration_m_per_s2=response.acceleration_m_per_s2[retained].copy(),
    )


def load_sensor_layouts(
    layout_config: dict[str, dict[str, list[int]]],
    n_dof: int,
) -> dict[str, SensorLayout]:
    """Convert one-based configuration locations to checked zero-based layouts."""

    layouts: dict[str, SensorLayout] = {}
    for name, specification in layout_config.items():
        indices = tuple(index - 1 for index in specification["dofs_one_based"])
        layout = SensorLayout(name, indices)
        if max(layout.dof_indices) >= n_dof:
            raise ValueError(f"Layout {name!r} references a nonexistent DOF.")
        layouts[name] = layout
    return layouts


def create_verification_figure(
    layouts: dict[str, SensorLayout],
    n_dof: int,
    time_s: np.ndarray,
    scalar_force_n: np.ndarray,
    correlation_time_s: float,
    clean_acceleration: np.ndarray,
    clean_layout: SensorLayout,
    noise_fractions: np.ndarray,
    empirical_noise_fractions: np.ndarray,
    imperfect: AccelerationMeasurements,
    outage: ContiguousOutage,
) -> None:
    """Save compact evidence for layouts, excitation, and imperfections."""

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.0, 8.0),
        constrained_layout=True,
    )
    (
        layout_axis,
        excitation_axis,
        acceleration_axis,
        noise_axis,
        bias_axis,
        availability_axis,
    ) = axes.ravel()

    displayed_layouts = [name for name in layouts if name != "full_reference"]
    layout_matrix = np.zeros((len(displayed_layouts), n_dof), dtype=float)
    for row, layout_name in enumerate(displayed_layouts):
        layout_matrix[row, layouts[layout_name].dof_indices] = 1.0
    layout_axis.imshow(
        layout_matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    layout_axis.set_xticks(np.arange(n_dof), labels=np.arange(1, n_dof + 1))
    layout_axis.set_yticks(
        np.arange(len(displayed_layouts)),
        labels=[name.replace("_", " ") for name in displayed_layouts],
    )
    layout_axis.set_xlabel("Structural DOF")
    layout_axis.set_title("Alternative acceleration layouts")

    display_interval = time_s <= 20.0
    excitation_axis.plot(
        time_s[display_interval],
        scalar_force_n[display_interval] / 1000.0,
        linewidth=1.0,
        color="#264653",
    )
    excitation_axis.set_xlabel("Within-window time (s)")
    excitation_axis.set_ylabel("Scalar load (kN)")
    excitation_axis.set_title("Colored stochastic excitation")
    excitation_axis.grid(alpha=0.25)
    sampling_frequency = 1.0 / (time_s[1] - time_s[0])
    frequencies, spectral_density = welch(
        scalar_force_n,
        fs=sampling_frequency,
        nperseg=min(1024, scalar_force_n.size),
    )
    inset = excitation_axis.inset_axes([0.53, 0.55, 0.43, 0.38])
    positive = frequencies > 0.0
    inset.loglog(frequencies[positive], spectral_density[positive], linewidth=0.9)
    corner_frequency = 1.0 / (2.0 * np.pi * correlation_time_s)
    inset.axvline(corner_frequency, color="#E76F51", linestyle="--", linewidth=0.8)
    inset.set_xlabel("Hz", fontsize=7)
    inset.set_ylabel("PSD", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.2, which="both")

    acceleration_interval = time_s <= 12.0
    for sensor_index, dof_index in enumerate(clean_layout.dof_indices):
        acceleration_axis.plot(
            time_s[acceleration_interval],
            clean_acceleration[acceleration_interval, sensor_index],
            linewidth=1.0,
            label=f"DOF {dof_index + 1}",
        )
    acceleration_axis.set_xlabel("Within-window time (s)")
    acceleration_axis.set_ylabel(r"Clean acceleration (m/s$^2$)")
    acceleration_axis.set_title("Sparse clean acceleration channels")
    acceleration_axis.grid(alpha=0.25)
    acceleration_axis.legend(frameon=False, fontsize=8)

    noise_axis.plot(
        100.0 * noise_fractions,
        100.0 * noise_fractions,
        color="0.35",
        linestyle="--",
        label="Target",
    )
    for sensor_index, dof_index in enumerate(clean_layout.dof_indices):
        noise_axis.plot(
            100.0 * noise_fractions,
            100.0 * empirical_noise_fractions[:, sensor_index],
            marker="o",
            linewidth=1.2,
            label=f"DOF {dof_index + 1}",
        )
    noise_axis.set_xlabel("Configured noise SD / signal SD (%)")
    noise_axis.set_ylabel("Empirical noise SD / signal SD (%)")
    noise_axis.set_title("Noise-level calibration")
    noise_axis.grid(alpha=0.25)
    noise_axis.legend(frameon=False, fontsize=8)

    displayed_sensor = imperfect.layout.n_sensors - 1
    observed = imperfect.observed_m_per_s2[:, displayed_sensor]
    clean = imperfect.clean_m_per_s2[:, displayed_sensor]
    residual = observed - clean
    bias_axis.plot(
        imperfect.time_s,
        residual,
        color="0.55",
        linewidth=0.7,
        label="Observed minus clean",
    )
    bias_axis.plot(
        imperfect.time_s,
        imperfect.bias_history_m_per_s2[:, displayed_sensor],
        color="#D62828",
        linewidth=1.8,
        label="Injected bias + drift",
    )
    bias_axis.axvspan(outage.start_s, outage.end_s, color="0.85", label="Outage")
    bias_axis.set_xlabel("Within-window time (s)")
    bias_axis.set_ylabel(r"Acceleration residual (m/s$^2$)")
    bias_axis.set_title("Bias, drift, noise, and outage")
    bias_axis.grid(alpha=0.25)
    bias_axis.legend(frameon=False, fontsize=8)

    availability_axis.imshow(
        imperfect.availability_mask.T,
        cmap="Greys_r",
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
        extent=[time_s[0], time_s[-1], imperfect.layout.n_sensors + 0.5, 0.5],
    )
    availability_axis.set_yticks(
        np.arange(1, imperfect.layout.n_sensors + 1),
        labels=[f"DOF {index + 1}" for index in imperfect.layout.dof_indices],
    )
    availability_axis.set_xlabel("Within-window time (s)")
    availability_axis.set_ylabel("Sensor")
    availability_axis.set_title("Availability mask: white = available")

    figure.suptitle("Stage 3: verified sparse synthetic monitoring", fontweight="bold")
    figure.savefig(FIGURE_PATH, dpi=200)
    plt.close(figure)


def main() -> None:
    """Generate and verify one canonical sparse monitoring window."""

    with BASE_CONFIG_PATH.open(encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    with MONITORING_CONFIG_PATH.open(encoding="utf-8") as stream:
        monitoring_config = yaml.safe_load(stream)

    structural = base_config["structural_properties"]
    damping_config = base_config["damping"]
    window_config = monitoring_config["window"]
    excitation_config = monitoring_config["excitation"]
    seed_config = monitoring_config["random_seeds"]
    verification = monitoring_config["verification"]

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

    total_duration = float(window_config["burn_in_s"]) + float(
        window_config["duration_s"]
    )
    time_step = float(window_config["time_step_s"])
    full_time = uniform_time(total_duration, time_step)
    excitation_rng = np.random.default_rng(int(seed_config["excitation"]))
    full_scalar_force = generate_ornstein_uhlenbeck_force(
        full_time,
        stationary_standard_deviation_n=float(
            excitation_config["stationary_standard_deviation_n"]
        ),
        correlation_time_s=float(excitation_config["correlation_time_s"]),
        mean_n=float(excitation_config["mean_n"]),
        rng=excitation_rng,
    )
    full_force = distribute_scalar_force(
        full_scalar_force,
        excitation_config["spatial_distribution"],
    )
    full_response = simulate_newmark_average_acceleration(
        mass,
        damping,
        stiffness,
        full_time,
        full_force,
    )
    burn_in = float(window_config["burn_in_s"])
    retained = full_time >= burn_in
    response = trim_response(full_response, burn_in)
    scalar_force = full_scalar_force[retained].copy()

    layouts = load_sensor_layouts(
        monitoring_config["sensor_layouts"],
        mass.shape[0],
    )
    selected_layout = layouts[monitoring_config["imperfect_case"]["layout"]]
    clean_acceleration = extract_clean_acceleration(response, selected_layout)

    noise_fractions = np.asarray(
        monitoring_config["noise"]["standard_deviation_fractions"],
        dtype=np.float64,
    )
    empirical_noise_fractions = np.zeros(
        (noise_fractions.size, selected_layout.n_sensors),
        dtype=np.float64,
    )
    noise_measurements: dict[float, AccelerationMeasurements] = {}
    clean_standard_deviation = np.std(clean_acceleration, axis=0, ddof=1)
    for index, noise_fraction in enumerate(noise_fractions):
        measurement = create_acceleration_measurements(
            response,
            selected_layout,
            rng=np.random.default_rng(int(seed_config["noise_base"]) + index),
            noise_standard_deviation_fraction=float(noise_fraction),
        )
        noise_measurements[float(noise_fraction)] = measurement
        residual = measurement.observed_m_per_s2 - measurement.clean_m_per_s2
        empirical_noise_fractions[index] = (
            np.std(residual, axis=0, ddof=1) / clean_standard_deviation
        )

    imperfect_config = monitoring_config["imperfect_case"]
    outages = tuple(
        ContiguousOutage(
            sensor_index=int(specification["sensor_one_based"]) - 1,
            start_s=float(specification["start_s"]),
            end_s=float(specification["end_s"]),
        )
        for specification in imperfect_config["contiguous_outages"]
    )
    availability = generate_availability_mask(
        response.time_s,
        selected_layout.n_sensors,
        random_dropout_probability=float(
            imperfect_config["random_dropout_probability"]
        ),
        rng=np.random.default_rng(int(seed_config["dropout"])),
        contiguous_outages=outages,
    )
    imperfect = create_acceleration_measurements(
        response,
        selected_layout,
        rng=np.random.default_rng(int(seed_config["imperfect_measurement"])),
        noise_standard_deviation_fraction=float(
            imperfect_config["noise_standard_deviation_fraction"]
        ),
        constant_bias_fraction=float(imperfect_config["constant_bias_fraction"]),
        end_of_window_drift_fraction=float(
            imperfect_config["end_of_window_drift_fraction"]
        ),
        availability_mask=availability,
    )

    repeated_availability = generate_availability_mask(
        response.time_s,
        selected_layout.n_sensors,
        random_dropout_probability=float(
            imperfect_config["random_dropout_probability"]
        ),
        rng=np.random.default_rng(int(seed_config["dropout"])),
        contiguous_outages=outages,
    )
    repeated_imperfect = create_acceleration_measurements(
        response,
        selected_layout,
        rng=np.random.default_rng(int(seed_config["imperfect_measurement"])),
        noise_standard_deviation_fraction=float(
            imperfect_config["noise_standard_deviation_fraction"]
        ),
        constant_bias_fraction=float(imperfect_config["constant_bias_fraction"]),
        end_of_window_drift_fraction=float(
            imperfect_config["end_of_window_drift_fraction"]
        ),
        availability_mask=repeated_availability,
    )
    imperfect_reproducible = np.array_equal(
        imperfect.observed_m_per_s2,
        repeated_imperfect.observed_m_per_s2,
        equal_nan=True,
    )

    configured_standard_deviation = float(
        excitation_config["stationary_standard_deviation_n"]
    )
    configured_mean = float(excitation_config["mean_n"])
    excitation_mean_fraction = abs(np.mean(scalar_force) - configured_mean) / (
        configured_standard_deviation
    )
    excitation_std_relative_error = abs(
        np.std(scalar_force, ddof=1) / configured_standard_deviation - 1.0
    )
    retained_duration = float(response.time_s[-1] - response.time_s[0])
    correlation_time = float(excitation_config["correlation_time_s"])
    mean_standard_error_fraction = np.sqrt(
        2.0 * correlation_time / retained_duration
    )
    standard_deviation_standard_error_fraction = np.sqrt(
        correlation_time / (2.0 * retained_duration)
    )
    excitation_mean_z_score = (
        excitation_mean_fraction / mean_standard_error_fraction
    )
    excitation_std_z_score = (
        excitation_std_relative_error
        / standard_deviation_standard_error_fraction
    )
    expected_lag_one = np.exp(
        -time_step / float(excitation_config["correlation_time_s"])
    )
    empirical_lag_one = float(np.corrcoef(scalar_force[:-1], scalar_force[1:])[0, 1])
    lag_one_error = abs(empirical_lag_one - expected_lag_one)

    nonzero_noise = noise_fractions > 0.0
    relative_noise_errors = np.abs(
        empirical_noise_fractions[nonzero_noise]
        / noise_fractions[nonzero_noise, np.newaxis]
        - 1.0
    )
    maximum_noise_relative_error = float(np.max(relative_noise_errors))
    zero_noise_exact = np.array_equal(
        noise_measurements[0.0].observed_m_per_s2,
        noise_measurements[0.0].clean_m_per_s2,
    )
    layout_dimensions_valid = all(
        extract_clean_acceleration(response, layout).shape
        == (response.time_s.size, layout.n_sensors)
        for layout in layouts.values()
    )
    outages_complete = all(
        not np.any(
            imperfect.availability_mask[
                (response.time_s >= outage.start_s)
                & (response.time_s <= outage.end_s),
                outage.sensor_index,
            ]
        )
        for outage in outages
    )
    missing_values_explicit = bool(
        np.all(np.isnan(imperfect.observed_m_per_s2[~imperfect.availability_mask]))
    )

    checks = {
        "finite-window excitation mean": excitation_mean_z_score
        <= float(verification["finite_window_statistical_z_limit"]),
        "finite-window excitation standard deviation": excitation_std_z_score
        <= float(verification["finite_window_statistical_z_limit"]),
        "excitation lag-one correlation": lag_one_error
        <= float(verification["lag_one_correlation_tolerance"]),
        "all layout dimensions": layout_dimensions_valid,
        "zero-noise recovery": zero_noise_exact,
        "noise calibration": maximum_noise_relative_error
        <= float(verification["empirical_noise_relative_tolerance"]),
        "minimum availability": bool(
            np.all(
                imperfect.available_fraction
                >= float(verification["minimum_available_fraction"])
            )
        ),
        "contiguous outages complete": outages_complete,
        "missing values explicit": missing_values_explicit,
        "imperfect measurement reproducible": imperfect_reproducible,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise RuntimeError(f"Stage 3 checks failed: {', '.join(failed_checks)}")

    table: dict[str, object] = {
        "time_s": response.time_s,
        "scalar_excitation_n": scalar_force,
    }
    for sensor_index, dof_index in enumerate(selected_layout.dof_indices):
        prefix = f"dof_{dof_index + 1}"
        table[f"{prefix}_clean_acceleration_m_per_s2"] = imperfect.clean_m_per_s2[
            :, sensor_index
        ]
        table[f"{prefix}_observed_acceleration_m_per_s2"] = (
            imperfect.observed_m_per_s2[:, sensor_index]
        )
        table[f"{prefix}_bias_m_per_s2"] = imperfect.bias_history_m_per_s2[
            :, sensor_index
        ]
        table[f"{prefix}_available"] = imperfect.availability_mask[:, sensor_index]
    monitoring_table = pd.DataFrame(table)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    monitoring_table.to_csv(DATA_PATH, index=False, float_format="%.12g")

    create_verification_figure(
        layouts,
        mass.shape[0],
        response.time_s,
        scalar_force,
        float(excitation_config["correlation_time_s"]),
        clean_acceleration,
        selected_layout,
        noise_fractions,
        empirical_noise_fractions,
        imperfect,
        outages[0],
    )

    print("STAGE 3 - SPARSE SYNTHETIC MONITORING VERIFICATION")
    print(f"Retained window samples: {response.time_s.size}")
    print(f"Sampling frequency: {1.0 / time_step:.3f} Hz")
    print(f"Layouts verified: {', '.join(layouts)}")
    print(f"Selected imperfect layout: {selected_layout.name}")
    print(f"Excitation mean / target SD: {excitation_mean_fraction:.6e}")
    print(f"Excitation SD relative error: {excitation_std_relative_error:.6e}")
    print(f"Finite-window mean z-score: {excitation_mean_z_score:.6f}")
    print(f"Finite-window SD z-score: {excitation_std_z_score:.6f}")
    print(f"Expected lag-one correlation: {expected_lag_one:.6f}")
    print(f"Empirical lag-one correlation: {empirical_lag_one:.6f}")
    print(
        "Maximum noise calibration relative error: "
        f"{maximum_noise_relative_error:.6e}"
    )
    print(
        "Available fraction by sensor: "
        + ", ".join(f"{fraction:.4f}" for fraction in imperfect.available_fraction)
    )
    print(f"Imperfect data reproducible: {imperfect_reproducible}")
    print(f"Exported rows: {len(monitoring_table)}")
    print(f"Data: {DATA_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
