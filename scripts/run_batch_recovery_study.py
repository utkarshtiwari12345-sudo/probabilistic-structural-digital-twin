"""Run the Stage 5 batch baseline and matched-model recovery experiments."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from probdt.diagnostics import acceleration_auto_psd  # noqa: E402
from probdt.inference import (  # noqa: E402
    BatchEstimate,
    fit_batch_spectral_map,
    simulate_averaged_periodogram,
)
from probdt.monitoring import SensorLayout  # noqa: E402
from probdt.structural import (  # noqa: E402
    apply_stiffness_multipliers,
    assemble_mass_matrix,
    assemble_rayleigh_damping,
    assemble_stiffness_matrix,
    assemble_zonal_stiffness_components,
    fit_rayleigh_coefficients,
    solve_modal_properties,
)

FloatArray = NDArray[np.float64]
SpectralForwardModel = Callable[[FloatArray], FloatArray]

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
INFERENCE_CONFIG_PATH = PROJECT_ROOT / "config" / "batch_inference.yaml"
RECOVERY_PATH = PROJECT_ROOT / "results" / "metrics" / "stage5_recovery_runs.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "metrics" / "stage5_recovery_summary.csv"
IDEAL_PATH = PROJECT_ROOT / "results" / "metrics" / "stage5_ideal_recovery.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage5_batch_recovery.png"


@dataclass(frozen=True)
class SpectralSystem:
    """Independently allocated structural inputs for truth or twin evaluation."""

    mass_matrix_kg: FloatArray
    damping_matrix_n_s_per_m: FloatArray
    zone_components_n_per_m: FloatArray


def build_spectral_system(base_config: dict[str, object]) -> SpectralSystem:
    """Build a fresh structural system from immutable configuration values."""

    structural = base_config["structural_properties"]
    damping_config = base_config["damping"]
    mass = assemble_mass_matrix(structural["masses_kg"])
    healthy_stiffness = assemble_stiffness_matrix(
        structural["story_stiffness_n_per_m"]
    )
    _, zone_components = assemble_zonal_stiffness_components(
        structural["story_stiffness_n_per_m"],
        structural["stiffness_zones"],
    )
    modes = solve_modal_properties(mass, healthy_stiffness)
    target_mode_indices = [
        int(mode_number) - 1 for mode_number in damping_config["target_modes"]
    ]
    target_ratios = [
        float(value) for value in damping_config["target_modal_damping_ratios"]
    ]
    alpha, beta = fit_rayleigh_coefficients(
        modes.angular_frequencies_rad_s[target_mode_indices[0]],
        modes.angular_frequencies_rad_s[target_mode_indices[1]],
        target_ratios[0],
        target_ratios[1],
    )
    damping = assemble_rayleigh_damping(
        mass,
        healthy_stiffness,
        alpha_per_s=alpha,
        beta_s=beta,
    ).damping_matrix_n_s_per_m
    return SpectralSystem(
        mass_matrix_kg=mass.copy(),
        damping_matrix_n_s_per_m=damping.copy(),
        zone_components_n_per_m=zone_components.copy(),
    )


def make_forward_model(
    system: SpectralSystem,
    frequencies_hz: FloatArray,
    force_distribution: FloatArray,
    sensor_layout: SensorLayout,
    excitation_config: dict[str, object],
    measurement_noise_psd: FloatArray,
) -> SpectralForwardModel:
    """Create a bounded zonal-stiffness-to-acceleration-spectrum mapping."""

    def forward(parameters: FloatArray) -> FloatArray:
        stiffness = apply_stiffness_multipliers(
            system.zone_components_n_per_m,
            parameters,
        )
        return acceleration_auto_psd(
            system.mass_matrix_kg,
            system.damping_matrix_n_s_per_m,
            stiffness,
            frequencies_hz,
            force_distribution,
            sensor_layout.dof_indices,
            force_standard_deviation_n=float(
                excitation_config["stationary_standard_deviation_n"]
            ),
            force_correlation_time_s=float(
                excitation_config["correlation_time_s"]
            ),
            measurement_noise_psd=measurement_noise_psd,
        )

    return forward


def fit_one_dataset(
    forward_model: SpectralForwardModel,
    observed_psd: FloatArray,
    inference_config: dict[str, object],
) -> BatchEstimate:
    """Fit one configured batch posterior approximation."""

    parameter_config = inference_config["parameter_model"]
    likelihood_config = inference_config["spectral_likelihood"]
    optimizer_config = inference_config["optimizer"]
    bounds = list(
        zip(
            parameter_config["lower_bounds"],
            parameter_config["upper_bounds"],
            strict=True,
        )
    )
    return fit_batch_spectral_map(
        forward_model,
        observed_psd,
        parameter_config["names"],
        bounds,
        prior_mean=parameter_config["prior_mean"],
        prior_standard_deviation=parameter_config["prior_standard_deviation"],
        n_independent_averages=int(
            likelihood_config["independent_periodogram_averages"]
        ),
        initial_points=optimizer_config["initial_points"],
        credible_level=float(parameter_config["credible_level"]),
        derivative_relative_step=float(
            optimizer_config["derivative_relative_step"]
        ),
        optimizer_max_iterations=int(optimizer_config["maximum_iterations"]),
    )


def create_recovery_figure(
    recovery: pd.DataFrame,
    summary: pd.DataFrame,
    case_order: list[str],
    noise_order: list[float],
    example: dict[str, object],
    credible_level: float,
) -> None:
    """Plot recovery accuracy, noise robustness, coverage, and one spectral fit."""

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    recovery_axis, rmse_axis, coverage_axis, spectrum_axis = axes.ravel()
    colors = {"lower": "#0072B2", "middle": "#D55E00", "upper": "#009E73"}

    selected_noise = min(noise_order, key=lambda value: abs(value - 0.05))
    selected_summary = summary[np.isclose(summary["noise_fraction"], selected_noise)]
    for zone in colors:
        zone_summary = selected_summary[selected_summary["zone"] == zone].set_index(
            "case"
        )
        truths = np.array(
            [zone_summary.loc[case, "true_parameter"] for case in case_order]
        )
        estimates = np.array(
            [zone_summary.loc[case, "mean_estimate"] for case in case_order]
        )
        errors = np.array(
            [
                zone_summary.loc[case, "estimate_standard_deviation"]
                for case in case_order
            ]
        )
        recovery_axis.errorbar(
            truths,
            estimates,
            yerr=errors,
            fmt="o",
            capsize=3,
            color=colors[zone],
            label=zone,
            alpha=0.9,
        )
    recovery_axis.plot([0.78, 1.01], [0.78, 1.01], "k--", linewidth=1.2)
    recovery_axis.set_xlim(0.78, 1.01)
    recovery_axis.set_ylim(0.78, 1.01)
    recovery_axis.set_xlabel("True stiffness multiplier")
    recovery_axis.set_ylabel("Mean batch MAP estimate")
    recovery_axis.set_title(f"Matched recovery at {100 * selected_noise:.0f}% noise")
    recovery_axis.legend(title="Zone")
    recovery_axis.grid(alpha=0.25)

    rmse_matrix = np.empty((len(case_order), len(noise_order)), dtype=np.float64)
    for row, case in enumerate(case_order):
        for column, noise in enumerate(noise_order):
            subset = recovery[
                (recovery["case"] == case)
                & np.isclose(recovery["noise_fraction"], noise)
            ]
            rmse_matrix[row, column] = np.sqrt(np.mean(subset["error"] ** 2))
    rmse_image = rmse_axis.imshow(rmse_matrix, cmap="viridis", aspect="auto")
    rmse_axis.set_xticks(
        np.arange(len(noise_order)),
        labels=[f"{100 * noise:.0f}%" for noise in noise_order],
    )
    rmse_axis.set_yticks(
        np.arange(len(case_order)),
        labels=[case.replace("_", " ") for case in case_order],
    )
    rmse_axis.set_xlabel("Sensor-noise amplitude fraction")
    rmse_axis.set_title("Pooled zonal RMSE")
    for row in range(len(case_order)):
        for column in range(len(noise_order)):
            rmse_axis.text(
                column,
                row,
                f"{rmse_matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if rmse_matrix[row, column] > 0.012 else "black",
                fontsize=8,
            )
    figure.colorbar(rmse_image, ax=rmse_axis, label="RMSE in multiplier")

    interior = recovery[recovery["true_parameter"] < 0.999]
    x_locations = np.arange(len(noise_order))
    width = 0.24
    for zone_index, zone in enumerate(colors):
        coverages = [
            interior[
                (interior["zone"] == zone)
                & np.isclose(interior["noise_fraction"], noise)
            ]["covered"].mean()
            for noise in noise_order
        ]
        coverage_axis.bar(
            x_locations + (zone_index - 1) * width,
            coverages,
            width,
            color=colors[zone],
            label=zone,
        )
    coverage_axis.axhline(
        credible_level,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Nominal {100 * credible_level:.0f}%",
    )
    coverage_axis.set_xticks(
        x_locations,
        labels=[f"{100 * noise:.0f}%" for noise in noise_order],
    )
    coverage_axis.set_ylim(0.0, 1.05)
    coverage_axis.set_xlabel("Sensor-noise amplitude fraction")
    coverage_axis.set_ylabel("Empirical interval coverage")
    coverage_axis.set_title("Interior-parameter coverage")
    coverage_axis.legend(ncols=2, fontsize=8)
    coverage_axis.grid(axis="y", alpha=0.25)

    frequencies = example["frequencies_hz"]
    observed = example["observed_psd"]
    truth_psd = example["truth_psd"]
    fitted_psd = example["fitted_psd"]
    sensor_column = -1
    spectrum_axis.semilogy(
        frequencies,
        observed[:, sensor_column],
        color="0.65",
        linewidth=0.8,
        label="Observed averaged periodogram",
    )
    spectrum_axis.semilogy(
        frequencies,
        truth_psd[:, sensor_column],
        color="#0072B2",
        linewidth=2.0,
        label="Truth expectation",
    )
    spectrum_axis.semilogy(
        frequencies,
        fitted_psd[:, sensor_column],
        color="#D55E00",
        linestyle="--",
        linewidth=1.6,
        label="Batch fitted spectrum",
    )
    spectrum_axis.set_xlabel("Frequency (Hz)")
    spectrum_axis.set_ylabel(r"Acceleration PSD $(m/s^2)^2/(rad/s)$")
    spectrum_axis.set_title("Representative mixed-loss fit: upper sensor")
    spectrum_axis.legend(fontsize=8)
    spectrum_axis.grid(alpha=0.25)

    figure.suptitle(
        "Stage 5: batch spectral inference and parameter recovery",
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Execute the controlled Stage 5 inference verification gate."""

    with BASE_CONFIG_PATH.open(encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    with MONITORING_CONFIG_PATH.open(encoding="utf-8") as stream:
        monitoring_config = yaml.safe_load(stream)
    with INFERENCE_CONFIG_PATH.open(encoding="utf-8") as stream:
        inference_config = yaml.safe_load(stream)

    likelihood_config = inference_config["spectral_likelihood"]
    parameter_config = inference_config["parameter_model"]
    experiment_config = inference_config["recovery_experiment"]
    verification = inference_config["verification"]
    excitation_config = monitoring_config["excitation"]

    frequencies_hz = np.linspace(
        float(likelihood_config["minimum_frequency_hz"]),
        float(likelihood_config["maximum_frequency_hz"]),
        int(likelihood_config["number_of_frequencies"]),
    )
    force_distribution = np.asarray(
        excitation_config["spatial_distribution"],
        dtype=np.float64,
    )
    layout_name = str(likelihood_config["sensor_layout"])
    layout_specification = monitoring_config["sensor_layouts"][layout_name]
    layout = SensorLayout(
        layout_name,
        tuple(int(index) - 1 for index in layout_specification["dofs_one_based"]),
    )
    truth_system = build_spectral_system(base_config)
    twin_system = build_spectral_system(base_config)
    if any(
        np.shares_memory(truth_array, twin_array)
        for truth_array, twin_array in (
            (truth_system.mass_matrix_kg, twin_system.mass_matrix_kg),
            (
                truth_system.damping_matrix_n_s_per_m,
                twin_system.damping_matrix_n_s_per_m,
            ),
            (
                truth_system.zone_components_n_per_m,
                twin_system.zone_components_n_per_m,
            ),
        )
    ):
        raise RuntimeError("Truth and twin systems unexpectedly share memory.")

    zero_noise = np.zeros(layout.n_sensors, dtype=np.float64)
    healthy_parameters = np.ones(3, dtype=np.float64)
    healthy_truth_forward = make_forward_model(
        truth_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        zero_noise,
    )
    healthy_clean_psd = healthy_truth_forward(healthy_parameters)
    noise_reference = np.max(healthy_clean_psd, axis=0)

    cases = {
        str(name): np.asarray(values, dtype=np.float64)
        for name, values in experiment_config["cases"].items()
    }
    noise_levels = [
        float(value) for value in experiment_config["noise_amplitude_fractions"]
    ]
    parameter_names = tuple(str(name) for name in parameter_config["names"])
    zone_names = tuple(name.removeprefix("theta_") for name in parameter_names)

    ideal_rows: list[dict[str, object]] = []
    ideal_truth_forward = make_forward_model(
        truth_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        zero_noise,
    )
    ideal_twin_forward = make_forward_model(
        twin_system,
        frequencies_hz,
        force_distribution,
        layout,
        excitation_config,
        zero_noise,
    )
    for case_name, truth_parameters in cases.items():
        estimate = fit_one_dataset(
            ideal_twin_forward,
            ideal_truth_forward(truth_parameters),
            inference_config,
        )
        for index, zone in enumerate(zone_names):
            ideal_rows.append(
                {
                    "case": case_name,
                    "zone": zone,
                    "true_parameter": truth_parameters[index],
                    "map_estimate": estimate.map_parameters[index],
                    "error": estimate.map_parameters[index] - truth_parameters[index],
                }
            )

    total_datasets = (
        len(cases)
        * len(noise_levels)
        * int(experiment_config["monte_carlo_replicates"])
    )
    seed_sequences = np.random.SeedSequence(
        int(experiment_config["random_seed"])
    ).spawn(total_datasets)
    seed_iterator = iter(seed_sequences)
    recovery_rows: list[dict[str, object]] = []
    example: dict[str, object] = {}

    for noise_fraction in noise_levels:
        measurement_noise_psd = noise_fraction**2 * noise_reference
        truth_forward = make_forward_model(
            truth_system,
            frequencies_hz,
            force_distribution,
            layout,
            excitation_config,
            measurement_noise_psd,
        )
        twin_forward = make_forward_model(
            twin_system,
            frequencies_hz,
            force_distribution,
            layout,
            excitation_config,
            measurement_noise_psd,
        )
        for case_name, truth_parameters in cases.items():
            truth_psd = truth_forward(truth_parameters)
            for replicate in range(int(experiment_config["monte_carlo_replicates"])):
                seed_sequence = next(seed_iterator)
                seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
                observed_psd = simulate_averaged_periodogram(
                    truth_psd,
                    int(likelihood_config["independent_periodogram_averages"]),
                    np.random.default_rng(seed_sequence),
                )
                estimate = fit_one_dataset(
                    twin_forward,
                    observed_psd,
                    inference_config,
                )
                posterior_correlation = estimate.posterior_correlation
                covariance_minimum_eigenvalue = float(
                    np.linalg.eigvalsh(estimate.posterior_covariance)[0]
                )
                for index, zone in enumerate(zone_names):
                    truth_value = float(truth_parameters[index])
                    estimate_value = float(estimate.map_parameters[index])
                    recovery_rows.append(
                        {
                            "case": case_name,
                            "noise_fraction": noise_fraction,
                            "replicate": replicate,
                            "seed": seed,
                            "zone": zone,
                            "true_parameter": truth_value,
                            "map_estimate": estimate_value,
                            "error": estimate_value - truth_value,
                            "posterior_standard_deviation": (
                                estimate.posterior_standard_deviations[index]
                            ),
                            "credible_interval_lower": (
                                estimate.credible_interval_lower[index]
                            ),
                            "credible_interval_upper": (
                                estimate.credible_interval_upper[index]
                            ),
                            "credible_interval_width": (
                                estimate.credible_interval_upper[index]
                                - estimate.credible_interval_lower[index]
                            ),
                            "covered": (
                                estimate.credible_interval_lower[index]
                                <= truth_value
                                <= estimate.credible_interval_upper[index]
                            ),
                            "active_lower_bound": estimate.active_lower_bounds[index],
                            "active_upper_bound": estimate.active_upper_bounds[index],
                            "optimizer_success": estimate.optimizer_success,
                            "n_function_evaluations": (
                                estimate.n_function_evaluations
                            ),
                            "fisher_condition_number": (
                                estimate.fisher_condition_number
                            ),
                            "correlation_lower_middle": posterior_correlation[0, 1],
                            "correlation_lower_upper": posterior_correlation[0, 2],
                            "correlation_middle_upper": posterior_correlation[1, 2],
                            "covariance_minimum_eigenvalue": (
                                covariance_minimum_eigenvalue
                            ),
                        }
                    )
                if (
                    case_name == "mixed_loss"
                    and np.isclose(noise_fraction, 0.05)
                    and replicate == 0
                ):
                    example = {
                        "frequencies_hz": frequencies_hz.copy(),
                        "observed_psd": observed_psd.copy(),
                        "truth_psd": truth_psd.copy(),
                        "fitted_psd": twin_forward(estimate.map_parameters),
                    }

    recovery = pd.DataFrame(recovery_rows)
    ideal = pd.DataFrame(ideal_rows)
    summary = (
        recovery.groupby(["case", "noise_fraction", "zone"], sort=False)
        .agg(
            true_parameter=("true_parameter", "first"),
            mean_estimate=("map_estimate", "mean"),
            estimate_standard_deviation=("map_estimate", "std"),
            bias=("error", "mean"),
            rmse=("error", lambda values: np.sqrt(np.mean(values**2))),
            mean_posterior_standard_deviation=(
                "posterior_standard_deviation",
                "mean",
            ),
            empirical_coverage=("covered", "mean"),
            mean_interval_width=("credible_interval_width", "mean"),
        )
        .reset_index()
    )

    optimizer_failure_fraction = 1.0 - float(recovery["optimizer_success"].mean())
    pooled_rmse = float(np.sqrt(np.mean(recovery["error"] ** 2)))
    pooled_zone_bias = recovery.groupby("zone")["error"].mean()
    maximum_pooled_absolute_bias = float(np.max(np.abs(pooled_zone_bias)))
    interior = recovery[recovery["true_parameter"] < 0.999]
    interior_coverage = float(interior["covered"].mean())
    prior_scale = np.asarray(
        parameter_config["prior_standard_deviation"],
        dtype=np.float64,
    )
    posterior_scale_by_zone = recovery.groupby("zone", sort=False)[
        "posterior_standard_deviation"
    ].mean()
    uncertainty_contraction = float(
        np.min(prior_scale / posterior_scale_by_zone.to_numpy())
    )
    ideal_maximum_error = float(np.max(np.abs(ideal["error"])))
    all_covariances_positive = bool(
        np.all(recovery["covariance_minimum_eigenvalue"] > 0.0)
    )

    checks = {
        "ideal expected-spectrum recovery": ideal_maximum_error
        <= float(verification["ideal_maximum_absolute_error"]),
        "pooled stochastic recovery RMSE": pooled_rmse
        <= float(verification["maximum_pooled_rmse"]),
        "pooled zonal bias": maximum_pooled_absolute_bias
        <= float(verification["maximum_pooled_absolute_bias"]),
        "interior interval coverage": interior_coverage
        >= float(verification["minimum_interior_interval_coverage"]),
        "optimizer convergence": optimizer_failure_fraction
        <= float(verification["maximum_optimizer_failure_fraction"]),
        "positive posterior covariance": all_covariances_positive,
        "posterior uncertainty contraction": uncertainty_contraction
        >= float(verification["minimum_uncertainty_contraction_ratio"]),
        "representative spectrum retained": bool(example),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    RECOVERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    recovery.to_csv(RECOVERY_PATH, index=False, float_format="%.12g")
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    ideal.to_csv(IDEAL_PATH, index=False, float_format="%.12g")
    create_recovery_figure(
        recovery,
        summary,
        list(cases),
        noise_levels,
        example,
        float(parameter_config["credible_level"]),
    )

    print("STAGE 5 - BATCH BASELINE INFERENCE AND PARAMETER RECOVERY")
    print(f"Sensor layout: {layout.name} at DOFs {[i + 1 for i in layout.dof_indices]}")
    print(
        "Datasets: "
        f"{len(cases)} cases x {len(noise_levels)} noise levels x "
        f"{experiment_config['monte_carlo_replicates']} replicates = {total_datasets}"
    )
    print(
        "Independent spectral averages: "
        f"{likelihood_config['independent_periodogram_averages']}"
    )
    print(f"Ideal maximum absolute error: {ideal_maximum_error:.6f}")
    print("")
    print("Noise | Pooled RMSE | Mean bias | Interior coverage | Mean posterior SD")
    print("------+-------------+-----------+-------------------+------------------")
    for noise_fraction in noise_levels:
        subset = recovery[np.isclose(recovery["noise_fraction"], noise_fraction)]
        subset_interior = subset[subset["true_parameter"] < 0.999]
        print(
            f"{100 * noise_fraction:>4.0f}% | "
            f"{np.sqrt(np.mean(subset['error'] ** 2)):>11.5f} | "
            f"{np.mean(subset['error']):>+9.5f} | "
            f"{np.mean(subset_interior['covered']):>17.3f} | "
            f"{np.mean(subset['posterior_standard_deviation']):>16.5f}"
        )
    print("")
    selected = summary[np.isclose(summary["noise_fraction"], 0.05)]
    print("Mean recovery at 5% noise")
    for case_name, truth_parameters in cases.items():
        case_summary = selected[selected["case"] == case_name].set_index("zone")
        estimates = [case_summary.loc[zone, "mean_estimate"] for zone in zone_names]
        print(
            f"{case_name:<16}: truth {truth_parameters.round(3).tolist()} -> "
            f"estimate {np.round(estimates, 3).tolist()}"
        )
    print("")
    print(f"Pooled RMSE: {pooled_rmse:.6f}")
    print(
        "Maximum pooled zonal absolute bias: "
        f"{maximum_pooled_absolute_bias:.6f}"
    )
    print(
        f"Interior {100 * parameter_config['credible_level']:.0f}% coverage: "
        f"{interior_coverage:.3f}"
    )
    print(
        "Minimum prior/posterior SD contraction: "
        f"{uncertainty_contraction:.1f}x"
    )
    print(f"Optimizer failure fraction: {optimizer_failure_fraction:.6f}")
    print(f"Runs: {RECOVERY_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Ideal recovery: {IDEAL_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    if failed_checks:
        raise RuntimeError(f"Stage 5 checks failed: {', '.join(failed_checks)}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
