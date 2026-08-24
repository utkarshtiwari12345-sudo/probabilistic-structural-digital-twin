"""Run the Stage 2 deterioration-truth verification gate."""

from __future__ import annotations

import os
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

from probdt.deterioration import (  # noqa: E402
    ConditionThresholds,
    DeteriorationHistory,
    abrupt_loss,
    build_deterioration_history,
    classify_condition,
    combine_independent_losses,
    gamma_process_loss,
    smooth_gradual_loss,
    validate_deterioration_history,
)
from probdt.structural import (  # noqa: E402
    apply_stiffness_multipliers,
    assemble_mass_matrix,
    assemble_zonal_stiffness_components,
    solve_modal_properties,
)

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base_model.yaml"
SCENARIO_CONFIG_PATH = PROJECT_ROOT / "config" / "deterioration_scenarios.yaml"
GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stage2_deterioration_ground_truth.csv"
)
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage2_deterioration_scenarios.png"


def uniform_time(start_s: float, end_s: float, interval_s: float) -> np.ndarray:
    """Create an endpoint-inclusive monitoring grid."""

    n_intervals = round((end_s - start_s) / interval_s)
    if n_intervals < 1 or not np.isclose(
        start_s + n_intervals * interval_s,
        end_s,
    ):
        raise ValueError("Scenario time range must contain complete intervals.")
    return start_s + np.arange(n_intervals + 1) * interval_s


def build_scenario(
    scenario_name: str,
    specification: dict[str, object],
    time_s: np.ndarray,
    zone_names: tuple[str, ...],
) -> DeteriorationHistory:
    """Build one configured scenario from explicit, supported process types."""

    zone_processes = specification.get("zone_processes", {})
    if not isinstance(zone_processes, dict):
        raise ValueError("zone_processes must be a mapping.")
    rng = np.random.default_rng(specification.get("random_seed"))
    zone_losses: dict[str, np.ndarray] = {}

    for zone_name, process_specs in zone_processes.items():
        if not isinstance(process_specs, list):
            raise ValueError("Each zone must contain a list of process specifications.")
        components: list[np.ndarray] = []
        for process_spec in process_specs:
            process_type = process_spec["process"]
            if process_type == "smooth_gradual":
                component = smooth_gradual_loss(
                    time_s,
                    onset_s=process_spec["onset_s"],
                    completion_s=process_spec["completion_s"],
                    final_loss_fraction=process_spec["final_loss_fraction"],
                )
            elif process_type == "abrupt":
                component = abrupt_loss(
                    time_s,
                    event_time_s=process_spec["event_time_s"],
                    loss_fraction=process_spec["loss_fraction"],
                )
            elif process_type == "gamma":
                component = gamma_process_loss(
                    time_s,
                    onset_s=process_spec["onset_s"],
                    mean_final_loss_fraction=process_spec[
                        "mean_final_loss_fraction"
                    ],
                    coefficient_of_variation=process_spec[
                        "coefficient_of_variation"
                    ],
                    maximum_loss_fraction=process_spec["maximum_loss_fraction"],
                    rng=rng,
                )
            else:
                raise ValueError(f"Unsupported deterioration process: {process_type}")
            components.append(component)
        zone_losses[zone_name] = combine_independent_losses(components)

    return build_deterioration_history(
        scenario_name,
        time_s,
        zone_names,
        zone_losses,
    )


def create_verification_figure(
    histories: dict[str, DeteriorationHistory],
    first_frequencies_hz: dict[str, np.ndarray],
    thresholds: ConditionThresholds,
) -> None:
    """Plot scenario structure, severity stages, and modal consequences."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.0),
        constrained_layout=True,
    )
    overview_axis, zoning_axis, stage_axis, frequency_axis = axes.ravel()

    for scenario_name in (
        "healthy",
        "gradual_lower",
        "gradual_plus_abrupt_lower",
        "stochastic_fatigue_lower",
    ):
        history = histories[scenario_name]
        overview_axis.plot(
            history.time_s,
            100.0 * history.loss_fractions[:, 0],
            linewidth=1.6,
            label=scenario_name.replace("_", " "),
        )
    overview_axis.set_xlabel("Monitoring time (s; methodological scale)")
    overview_axis.set_ylabel("Lower-zone stiffness loss (%)")
    overview_axis.set_title("Controlled truth scenarios")
    overview_axis.grid(alpha=0.25)
    overview_axis.legend(frameon=False, fontsize=8)

    sequential = histories["sequential_two_zone"]
    for zone_index, zone_name in enumerate(sequential.zone_names):
        zoning_axis.plot(
            sequential.time_s,
            100.0 * sequential.loss_fractions[:, zone_index],
            linewidth=1.7,
            label=zone_name,
        )
    zoning_axis.set_xlabel("Monitoring time (s; methodological scale)")
    zoning_axis.set_ylabel("Stiffness loss (%)")
    zoning_axis.set_title("Sequential two-zone deterioration")
    zoning_axis.grid(alpha=0.25)
    zoning_axis.legend(frameon=False)

    demonstration = histories["gradual_plus_abrupt_lower"]
    demonstration_loss = 100.0 * demonstration.loss_fractions[:, 0]
    stage_axis.plot(
        demonstration.time_s,
        demonstration_loss,
        color="black",
        linewidth=1.8,
        label="Known lower-zone truth",
    )
    stage_boundaries = [
        100.0 * thresholds.healthy_max_loss_fraction,
        100.0 * thresholds.early_max_loss_fraction,
        100.0 * thresholds.moderate_max_loss_fraction,
    ]
    stage_colors = ["#E9F5DB", "#FFF3B0", "#FFD6A5", "#FFADAD"]
    stage_labels = ["healthy", "early", "moderate", "significant"]
    stage_edges = [0.0, *stage_boundaries, max(22.0, demonstration_loss.max() + 1.0)]
    for lower, upper, color, label in zip(
        stage_edges[:-1],
        stage_edges[1:],
        stage_colors,
        stage_labels,
        strict=True,
    ):
        stage_axis.axhspan(lower, upper, color=color, alpha=0.65, label=label)
    stage_axis.plot(
        demonstration.time_s,
        demonstration_loss,
        color="black",
        linewidth=1.8,
    )
    stage_axis.set_ylim(0.0, stage_edges[-1])
    stage_axis.set_xlabel("Monitoring time (s; methodological scale)")
    stage_axis.set_ylabel("Lower-zone stiffness loss (%)")
    stage_axis.set_title("Healthy-to-significant progression")
    stage_axis.grid(alpha=0.2)
    stage_axis.legend(frameon=False, fontsize=8, ncol=2)

    healthy_frequency = first_frequencies_hz["healthy"][0]
    for scenario_name, frequencies in first_frequencies_hz.items():
        if scenario_name == "healthy":
            continue
        frequency_shift = 100.0 * (frequencies / healthy_frequency - 1.0)
        frequency_axis.plot(
            histories[scenario_name].time_s,
            frequency_shift,
            linewidth=1.6,
            label=scenario_name.replace("_", " "),
        )
    frequency_axis.axhline(0.0, color="0.4", linewidth=0.8)
    frequency_axis.set_xlabel("Monitoring time (s; methodological scale)")
    frequency_axis.set_ylabel("First-frequency change (%)")
    frequency_axis.set_title("Physical modal consequence of deterioration")
    frequency_axis.grid(alpha=0.25)
    frequency_axis.legend(frameon=False, fontsize=8)

    figure.suptitle("Stage 2: verified deterioration ground truth", fontweight="bold")
    figure.savefig(FIGURE_PATH, dpi=200)
    plt.close(figure)


def main() -> None:
    """Generate, validate, export, and visualize all Stage 2 histories."""

    with BASE_CONFIG_PATH.open(encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    with SCENARIO_CONFIG_PATH.open(encoding="utf-8") as stream:
        scenario_config = yaml.safe_load(stream)

    time_config = scenario_config["time"]
    time_s = uniform_time(
        float(time_config["start_s"]),
        float(time_config["end_s"]),
        float(time_config["update_interval_s"]),
    )
    zone_names = tuple(scenario_config["zone_names"])
    threshold_config = scenario_config["condition_thresholds"]
    thresholds = ConditionThresholds(**threshold_config)
    verification = scenario_config["verification"]

    structural = base_config["structural_properties"]
    mass = assemble_mass_matrix(structural["masses_kg"])
    configured_zone_names, zone_components = assemble_zonal_stiffness_components(
        structural["story_stiffness_n_per_m"],
        structural["stiffness_zones"],
    )
    if configured_zone_names != zone_names:
        raise RuntimeError("Deterioration and structural zone ordering do not match.")

    histories: dict[str, DeteriorationHistory] = {}
    first_frequencies_hz: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    reports = {}

    for scenario_name, specification in scenario_config["scenarios"].items():
        history = build_scenario(scenario_name, specification, time_s, zone_names)
        report = validate_deterioration_history(
            history,
            bound_tolerance=float(verification["bound_tolerance"]),
            monotonicity_tolerance=float(verification["monotonicity_tolerance"]),
        )
        histories[scenario_name] = history
        reports[scenario_name] = report

        frequencies = np.empty(time_s.size, dtype=np.float64)
        condition_stages = classify_condition(history.loss_fractions, thresholds)
        for time_index, multipliers in enumerate(history.stiffness_multipliers):
            stiffness = apply_stiffness_multipliers(zone_components, multipliers)
            frequencies[time_index] = solve_modal_properties(
                mass,
                stiffness,
                n_modes=1,
            ).frequencies_hz[0]
            for zone_index, zone_name in enumerate(zone_names):
                rows.append(
                    {
                        "scenario": scenario_name,
                        "time_s": time_s[time_index],
                        "zone": zone_name,
                        "stiffness_multiplier": multipliers[zone_index],
                        "loss_fraction": history.loss_fractions[
                            time_index,
                            zone_index,
                        ],
                        "condition_stage": condition_stages[time_index, zone_index],
                        "first_frequency_hz": frequencies[time_index],
                    }
                )
        first_frequencies_hz[scenario_name] = frequencies

    repeated_stochastic = build_scenario(
        "stochastic_fatigue_lower",
        scenario_config["scenarios"]["stochastic_fatigue_lower"],
        time_s,
        zone_names,
    )
    stochastic_reproducible = np.array_equal(
        histories["stochastic_fatigue_lower"].stiffness_multipliers,
        repeated_stochastic.stiffness_multipliers,
    )
    stochastic_specification = scenario_config["scenarios"][
        "stochastic_fatigue_lower"
    ]["zone_processes"]["lower"][0]
    stochastic_cap = float(stochastic_specification["maximum_loss_fraction"])
    stochastic_maximum = float(
        np.max(histories["stochastic_fatigue_lower"].loss_fractions)
    )
    stochastic_cap_reached = bool(
        np.isclose(
            stochastic_maximum,
            stochastic_cap,
            rtol=0.0,
            atol=float(verification["bound_tolerance"]),
        )
    )

    required_stages = {"healthy", "early", "moderate", "significant"}
    demonstration_stages = set(
        classify_condition(
            histories["gradual_plus_abrupt_lower"].loss_fractions[:, 0],
            thresholds,
        )
    )
    healthy_frequency_constant = bool(
        np.allclose(
            first_frequencies_hz["healthy"],
            first_frequencies_hz["healthy"][0],
            rtol=0.0,
            atol=float(verification["frequency_monotonicity_tolerance_hz"]),
        )
    )
    frequency_monotone = all(
        np.max(np.diff(frequencies))
        <= float(verification["frequency_monotonicity_tolerance_hz"])
        for name, frequencies in first_frequencies_hz.items()
        if name != "healthy"
    )

    checks = {
        "all histories initially healthy": all(
            report.initially_healthy for report in reports.values()
        ),
        "all histories bounded": all(
            report.minimum_multiplier > 0.0 and report.maximum_multiplier <= 1.0
            for report in reports.values()
        ),
        "no history silently heals": all(
            report.monotone_nonincreasing for report in reports.values()
        ),
        "stochastic history reproducible": stochastic_reproducible,
        "stochastic safety cap inactive": not stochastic_cap_reached,
        "all severity stages represented": required_stages <= demonstration_stages,
        "healthy modal frequency constant": healthy_frequency_constant,
        "deterioration does not increase first frequency": frequency_monotone,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise RuntimeError(f"Stage 2 checks failed: {', '.join(failed_checks)}")

    ground_truth = pd.DataFrame(rows)
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    ground_truth.to_csv(GROUND_TRUTH_PATH, index=False, float_format="%.12g")
    expected_rows = len(histories) * time_s.size * len(zone_names)
    if len(ground_truth) != expected_rows:
        raise RuntimeError("Exported ground-truth table has an unexpected row count.")

    create_verification_figure(histories, first_frequencies_hz, thresholds)

    healthy_frequency = first_frequencies_hz["healthy"][0]
    print("STAGE 2 - DETERIORATION TRUTH VERIFICATION")
    print(f"Monitoring updates per scenario: {time_s.size}")
    print(f"Zones: {', '.join(zone_names)}")
    print("")
    print("Scenario                    | Maximum loss (%) | Final f1 shift (%)")
    print("----------------------------+------------------+-------------------")
    for scenario_name, history in histories.items():
        maximum_loss = 100.0 * np.max(history.loss_fractions)
        final_shift = 100.0 * (
            first_frequencies_hz[scenario_name][-1] / healthy_frequency - 1.0
        )
        print(f"{scenario_name:<27} | {maximum_loss:>16.6f} | {final_shift:>17.6f}")
    print("")
    print(f"Stochastic realization reproducible: {stochastic_reproducible}")
    print(f"Stochastic safety cap reached: {stochastic_cap_reached}")
    stages_represented = required_stages <= demonstration_stages
    print(f"Required severity stages represented: {stages_represented}")
    print(f"Exported ground-truth rows: {len(ground_truth)}")
    print(f"Ground truth: {GROUND_TRUTH_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
