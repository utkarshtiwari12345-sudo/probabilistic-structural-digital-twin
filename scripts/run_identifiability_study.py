"""Run the Stage 4 local practical-identifiability study."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "results" / "logs" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from probdt.diagnostics import (  # noqa: E402
    IdentifiabilityReport,
    acceleration_auto_psd,
    analyze_identifiability,
    central_difference_sensitivity,
    log_spectral_features,
)
from probdt.monitoring import SensorLayout  # noqa: E402
from probdt.structural import (  # noqa: E402
    apply_stiffness_multipliers,
    assemble_mass_matrix,
    assemble_rayleigh_damping,
    assemble_story_stiffness_components,
    assemble_zonal_stiffness_components,
    fit_rayleigh_coefficients,
    solve_modal_properties,
)

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
MONITORING_CONFIG_PATH = PROJECT_ROOT / "config" / "monitoring.yaml"
IDENTIFIABILITY_CONFIG_PATH = PROJECT_ROOT / "config" / "identifiability.yaml"
SUMMARY_PATH = PROJECT_ROOT / "results" / "metrics" / "stage4_identifiability.csv"
CORRELATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage4_selected_correlation.csv"
)
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage4_identifiability.png"


@dataclass(frozen=True)
class ParameterDefinition:
    """One scalar multiplier in a configured identifiability model."""

    name: str
    kind: str
    index: int | None = None


def parse_parameter_sets(
    configured_sets: dict[str, list[dict[str, object]]],
) -> dict[str, tuple[ParameterDefinition, ...]]:
    """Convert YAML mappings into checked immutable parameter definitions."""

    supported_kinds = {
        "zone_stiffness",
        "story_stiffness",
        "damping_scale",
        "mass_scale",
        "excitation_amplitude_scale",
    }
    parameter_sets: dict[str, tuple[ParameterDefinition, ...]] = {}
    for set_name, specifications in configured_sets.items():
        definitions = tuple(
            ParameterDefinition(
                name=str(specification["name"]),
                kind=str(specification["kind"]),
                index=(
                    int(specification["index"])
                    if "index" in specification
                    else None
                ),
            )
            for specification in specifications
        )
        if any(definition.kind not in supported_kinds for definition in definitions):
            raise ValueError(f"Parameter set {set_name!r} uses an unsupported kind.")
        if len({definition.name for definition in definitions}) != len(definitions):
            raise ValueError(f"Parameter set {set_name!r} has duplicate names.")
        has_zone = any(
            definition.kind == "zone_stiffness" for definition in definitions
        )
        has_story = any(
            definition.kind == "story_stiffness" for definition in definitions
        )
        if has_zone == has_story:
            raise ValueError(
                f"Parameter set {set_name!r} must use either zone or story stiffness."
            )
        parameter_sets[set_name] = definitions
    return parameter_sets


def load_layouts(
    layout_config: dict[str, dict[str, list[int]]],
    n_dof: int,
) -> dict[str, SensorLayout]:
    """Load checked zero-based sensor layouts from one-based configuration."""

    layouts = {
        name: SensorLayout(
            name,
            tuple(index - 1 for index in specification["dofs_one_based"]),
        )
        for name, specification in layout_config.items()
    }
    if any(max(layout.dof_indices) >= n_dof for layout in layouts.values()):
        raise ValueError("A configured sensor layout lies outside the model.")
    return layouts


def create_verification_figure(
    reports: dict[tuple[str, str], IdentifiabilityReport],
    layout_names: list[str],
    parameter_set_names: list[str],
) -> None:
    """Plot conditioning, effective rank, confounding, and singular spectra."""

    condition_numbers = np.array(
        [
            [
                reports[(layout, parameter_set)].normalized_condition_number
                for parameter_set in parameter_set_names
            ]
            for layout in layout_names
        ],
        dtype=np.float64,
    )
    rank_fractions = np.array(
        [
            [
                reports[(layout, parameter_set)].effective_rank
                / len(reports[(layout, parameter_set)].parameter_names)
                for parameter_set in parameter_set_names
            ]
            for layout in layout_names
        ],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 9.0),
        constrained_layout=True,
    )
    condition_axis, rank_axis, correlation_axis, singular_axis = axes.ravel()

    display_condition = np.log10(np.minimum(condition_numbers, 1.0e6))
    condition_image = condition_axis.imshow(
        display_condition,
        cmap="magma",
        aspect="auto",
        vmin=0.0,
    )
    condition_axis.set_xticks(
        np.arange(len(parameter_set_names)),
        labels=[name.replace("_", "\n") for name in parameter_set_names],
        fontsize=8,
    )
    condition_axis.set_yticks(
        np.arange(len(layout_names)),
        labels=[name.replace("_", " ") for name in layout_names],
    )
    condition_axis.set_title("Normalized sensitivity condition number")
    for row in range(len(layout_names)):
        for column in range(len(parameter_set_names)):
            value = condition_numbers[row, column]
            label = "inf" if not np.isfinite(value) else f"{value:.1f}"
            condition_axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if display_condition[row, column] > 1.2 else "black",
                fontsize=8,
            )
    figure.colorbar(condition_image, ax=condition_axis, label=r"$\log_{10}\kappa$")

    rank_image = rank_axis.imshow(
        rank_fractions,
        cmap="Blues",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    rank_axis.set_xticks(
        np.arange(len(parameter_set_names)),
        labels=[name.replace("_", "\n") for name in parameter_set_names],
        fontsize=8,
    )
    rank_axis.set_yticks(
        np.arange(len(layout_names)),
        labels=[name.replace("_", " ") for name in layout_names],
    )
    rank_axis.set_title("Effective rank / parameter count")
    for row in range(len(layout_names)):
        for column in range(len(parameter_set_names)):
            report = reports[(layout_names[row], parameter_set_names[column])]
            rank_axis.text(
                column,
                row,
                f"{report.effective_rank}/{len(report.parameter_names)}",
                ha="center",
                va="center",
                color="white" if rank_fractions[row, column] > 0.55 else "black",
                fontsize=8,
            )
    figure.colorbar(rank_image, ax=rank_axis, label="Rank fraction")

    selected_report = reports[("three_spaced", "three_zone_plus_nuisance")]
    correlation_image = correlation_axis.imshow(
        selected_report.normalized_parameter_correlation,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    shortened_names = [
        name.replace("theta_", "").replace("_scale", "")
        for name in selected_report.parameter_names
    ]
    correlation_axis.set_xticks(
        np.arange(len(shortened_names)),
        labels=shortened_names,
        rotation=35,
        ha="right",
    )
    correlation_axis.set_yticks(
        np.arange(len(shortened_names)),
        labels=shortened_names,
    )
    correlation_axis.set_title("Three-spaced layout: nuisance confounding")
    for row in range(len(shortened_names)):
        for column in range(len(shortened_names)):
            value = selected_report.normalized_parameter_correlation[row, column]
            correlation_axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else "black",
                fontsize=7,
            )
    figure.colorbar(correlation_image, ax=correlation_axis, label="Correlation")

    for parameter_set_name in parameter_set_names:
        report = reports[("three_spaced", parameter_set_name)]
        singular_axis.semilogy(
            np.arange(1, report.normalized_singular_values.size + 1),
            report.normalized_singular_values,
            marker="o",
            linewidth=1.4,
            label=parameter_set_name.replace("_", " "),
        )
    singular_axis.axhline(
        0.01,
        color="0.4",
        linestyle="--",
        linewidth=1.0,
        label="Effective-rank threshold",
    )
    singular_axis.set_xlabel("Singular-value index")
    singular_axis.set_ylabel("Singular value / largest")
    singular_axis.set_title("Three-spaced layout: normalized singular spectra")
    singular_axis.grid(alpha=0.25, which="both")
    singular_axis.legend(frameon=False, fontsize=8)

    figure.suptitle("Stage 4: local practical identifiability", fontweight="bold")
    figure.savefig(FIGURE_PATH, dpi=200)
    plt.close(figure)


def main() -> None:
    """Run the configured sensor and parameterization identifiability matrix."""

    with BASE_CONFIG_PATH.open(encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    with MONITORING_CONFIG_PATH.open(encoding="utf-8") as stream:
        monitoring_config = yaml.safe_load(stream)
    with IDENTIFIABILITY_CONFIG_PATH.open(encoding="utf-8") as stream:
        identifiability_config = yaml.safe_load(stream)

    structural = base_config["structural_properties"]
    damping_config = base_config["damping"]
    excitation_config = monitoring_config["excitation"]
    spectral_config = identifiability_config["spectral_features"]
    reference_config = identifiability_config["reference_condition"]
    difference_config = identifiability_config["finite_difference"]
    verification = identifiability_config["verification"]

    base_mass = assemble_mass_matrix(structural["masses_kg"])
    story_components = assemble_story_stiffness_components(
        structural["story_stiffness_n_per_m"]
    )
    zone_names, zone_components = assemble_zonal_stiffness_components(
        structural["story_stiffness_n_per_m"],
        structural["stiffness_zones"],
    )
    healthy_stiffness = story_components.sum(axis=0)
    modes = solve_modal_properties(base_mass, healthy_stiffness)
    damping_target_modes = np.asarray(damping_config["target_modes"], dtype=int) - 1
    damping_targets = np.asarray(
        damping_config["target_modal_damping_ratios"],
        dtype=np.float64,
    )
    alpha, beta = fit_rayleigh_coefficients(
        modes.angular_frequencies_rad_s[damping_target_modes[0]],
        modes.angular_frequencies_rad_s[damping_target_modes[1]],
        damping_targets[0],
        damping_targets[1],
    )
    base_damping = assemble_rayleigh_damping(
        base_mass,
        healthy_stiffness,
        alpha_per_s=alpha,
        beta_s=beta,
    ).damping_matrix_n_s_per_m

    frequencies_hz = np.linspace(
        float(spectral_config["minimum_frequency_hz"]),
        float(spectral_config["maximum_frequency_hz"]),
        int(spectral_config["number_of_frequencies"]),
    )
    force_distribution = np.asarray(
        excitation_config["spatial_distribution"],
        dtype=np.float64,
    )
    layouts = load_layouts(monitoring_config["sensor_layouts"], base_mass.shape[0])
    parameter_sets = parse_parameter_sets(identifiability_config["parameter_sets"])

    reference_zone = np.asarray(
        reference_config["zone_stiffness_multipliers"],
        dtype=np.float64,
    )
    reference_story = np.asarray(
        reference_config["story_stiffness_multipliers"],
        dtype=np.float64,
    )
    if reference_zone.size != len(zone_names):
        raise ValueError("Reference zone multipliers do not match structural zones.")
    if reference_story.size != story_components.shape[0]:
        raise ValueError("Reference story multipliers do not match structural stories.")

    reports: dict[tuple[str, str], IdentifiabilityReport] = {}
    summary_rows: list[dict[str, object]] = []

    for layout_name, layout in layouts.items():
        for set_name, definitions in parameter_sets.items():
            reference_values: list[float] = []
            parameter_steps: list[float] = []
            for definition in definitions:
                if definition.kind == "zone_stiffness":
                    if definition.index is None:
                        raise ValueError("A zone stiffness parameter needs an index.")
                    reference_values.append(reference_zone[definition.index])
                    parameter_steps.append(float(difference_config["stiffness_step"]))
                elif definition.kind == "story_stiffness":
                    if definition.index is None:
                        raise ValueError("A story stiffness parameter needs an index.")
                    reference_values.append(reference_story[definition.index])
                    parameter_steps.append(float(difference_config["stiffness_step"]))
                else:
                    reference_values.append(float(reference_config[definition.kind]))
                    parameter_steps.append(float(difference_config["nuisance_step"]))

            reference_vector = np.asarray(reference_values, dtype=np.float64)
            step_vector = np.asarray(parameter_steps, dtype=np.float64)

            def system_from_parameters(
                parameters: np.ndarray,
                parameter_definitions: tuple[
                    ParameterDefinition, ...
                ] = definitions,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
                zone_multipliers = reference_zone.copy()
                story_multipliers = reference_story.copy()
                damping_scale = float(reference_config["damping_scale"])
                mass_scale = float(reference_config["mass_scale"])
                excitation_scale = float(
                    reference_config["excitation_amplitude_scale"]
                )
                for value, definition in zip(
                    parameters,
                    parameter_definitions,
                    strict=True,
                ):
                    if definition.kind == "zone_stiffness":
                        zone_multipliers[definition.index] = value
                    elif definition.kind == "story_stiffness":
                        story_multipliers[definition.index] = value
                    elif definition.kind == "damping_scale":
                        damping_scale = value
                    elif definition.kind == "mass_scale":
                        mass_scale = value
                    elif definition.kind == "excitation_amplitude_scale":
                        excitation_scale = value

                if any(
                    definition.kind == "zone_stiffness"
                    for definition in parameter_definitions
                ):
                    stiffness = apply_stiffness_multipliers(
                        zone_components,
                        zone_multipliers,
                    )
                else:
                    stiffness = np.einsum(
                        "s,sij->ij",
                        story_multipliers,
                        story_components,
                    )
                return (
                    mass_scale * base_mass,
                    damping_scale * base_damping,
                    stiffness,
                    excitation_scale,
                )

            baseline_mass, baseline_damping, baseline_stiffness, baseline_excitation = (
                system_from_parameters(reference_vector)
            )
            baseline_clean_psd = acceleration_auto_psd(
                baseline_mass,
                baseline_damping,
                baseline_stiffness,
                frequencies_hz,
                force_distribution,
                layout.dof_indices,
                force_standard_deviation_n=float(
                    excitation_config["stationary_standard_deviation_n"]
                ),
                force_correlation_time_s=float(
                    excitation_config["correlation_time_s"]
                ),
                force_amplitude_scale=baseline_excitation,
            )
            noise_amplitude_fraction = float(
                spectral_config["measurement_noise_amplitude_fraction"]
            )
            fixed_noise_psd = noise_amplitude_fraction**2 * np.max(
                baseline_clean_psd,
                axis=0,
            )

            def feature_function(
                parameters: np.ndarray,
                active_layout: SensorLayout = layout,
                active_noise_psd: np.ndarray = fixed_noise_psd,
                system_function=system_from_parameters,
            ) -> np.ndarray:
                mass, damping, stiffness, excitation_scale = system_function(parameters)
                spectral_density = acceleration_auto_psd(
                    mass,
                    damping,
                    stiffness,
                    frequencies_hz,
                    force_distribution,
                    active_layout.dof_indices,
                    force_standard_deviation_n=float(
                        excitation_config["stationary_standard_deviation_n"]
                    ),
                    force_correlation_time_s=float(
                        excitation_config["correlation_time_s"]
                    ),
                    force_amplitude_scale=excitation_scale,
                    measurement_noise_psd=active_noise_psd,
                )
                return log_spectral_features(spectral_density)

            sensitivity = central_difference_sensitivity(
                feature_function,
                reference_vector,
                step_vector,
            )
            report = analyze_identifiability(
                sensitivity,
                tuple(definition.name for definition in definitions),
                effective_rank_relative_threshold=float(
                    verification["effective_rank_relative_threshold"]
                ),
            )
            reports[(layout_name, set_name)] = report
            summary_rows.append(
                {
                    "layout": layout_name,
                    "parameter_set": set_name,
                    "n_sensors": layout.n_sensors,
                    "n_parameters": len(definitions),
                    "effective_rank": report.effective_rank,
                    "normalized_condition_number": (
                        report.normalized_condition_number
                    ),
                    "maximum_absolute_parameter_correlation": (
                        report.maximum_absolute_parameter_correlation
                    ),
                    "minimum_sensitivity_column_norm": float(
                        np.min(report.column_norms)
                    ),
                }
            )

    selected_layout = "three_spaced"
    three_zone_report = reports[(selected_layout, "three_zone")]
    six_story_report = reports[(selected_layout, "six_story")]
    nuisance_report = reports[(selected_layout, "three_zone_plus_nuisance")]
    full_reference_report = reports[("full_reference", "three_zone")]
    top_only_report = reports[("top_only", "three_zone")]
    six_story_condition_ratio = (
        six_story_report.normalized_condition_number
        / three_zone_report.normalized_condition_number
    )

    checks = {
        "three-zone rank retained": three_zone_report.effective_rank
        >= int(verification["minimum_three_zone_rank"]),
        "three-zone conditioning acceptable": (
            three_zone_report.normalized_condition_number
            <= float(verification["maximum_condition_number_for_three_zone_spaced"])
        ),
        "six-story parameterization is materially worse": (
            six_story_condition_ratio
            >= float(verification["minimum_six_story_condition_ratio"])
        ),
        "nuisance model exposes high confounding": (
            nuisance_report.maximum_absolute_parameter_correlation
            >= float(verification["high_parameter_correlation_threshold"])
        ),
        "full sensing adds sensitivity information": (
            np.min(full_reference_report.column_norms)
            > np.min(top_only_report.column_norms)
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise RuntimeError(f"Stage 4 checks failed: {', '.join(failed_checks)}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.12g")
    selected_correlation = pd.DataFrame(
        nuisance_report.normalized_parameter_correlation,
        index=nuisance_report.parameter_names,
        columns=nuisance_report.parameter_names,
    )
    selected_correlation.to_csv(CORRELATION_PATH, float_format="%.12g")

    layout_names = list(layouts)
    parameter_set_names = list(parameter_sets)
    create_verification_figure(
        reports,
        layout_names,
        parameter_set_names,
    )

    print("STAGE 4 - LOCAL PRACTICAL IDENTIFIABILITY")
    print(f"Spectral frequency bins: {frequencies_hz.size}")
    print(f"Reference zonal stiffness multipliers: {reference_zone.tolist()}")
    print("")
    print("Three-spaced layout")
    print("Parameterization           | Rank | Condition | Max |correlation|")
    print("---------------------------+------+-----------+------------------")
    for set_name in parameter_set_names:
        report = reports[(selected_layout, set_name)]
        print(
            f"{set_name:<26} | "
            f"{report.effective_rank:>2}/{len(report.parameter_names):<2} | "
            f"{report.normalized_condition_number:>9.3f} | "
            f"{report.maximum_absolute_parameter_correlation:>16.6f}"
        )
    print("")
    combination = ", ".join(
        f"{name}={coefficient:+.3f}"
        for name, coefficient in zip(
            nuisance_report.parameter_names,
            nuisance_report.least_identifiable_combination,
            strict=True,
        )
    )
    print(f"Six-story / three-zone condition ratio: {six_story_condition_ratio:.3f}")
    print(f"Least identifiable nuisance combination: {combination}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Selected correlation: {CORRELATION_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
