"""Run Stage 20 healthy-only reduced-twin calibration and transfer checks."""

from __future__ import annotations

import hashlib
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

from probdt.openfast import (  # noqa: E402
    ReducedTwinCalibration,
    averaged_welch_auto_psd,
    calibrate_reduced_twin_nuisance,
    cosine_similarity,
    differential_log_spectral_signature,
    log_spectral_rmse,
    reduced_twin_directional_psd,
)
from probdt.structural import (  # noqa: E402
    assemble_mass_matrix,
    assemble_rayleigh_damping,
    assemble_stiffness_matrix,
    assemble_zonal_stiffness_components,
    fit_rayleigh_coefficients,
    solve_modal_properties,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "phase2_openfast_twin_transfer.yaml"
CALIBRATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage20_twin_calibration.csv"
)
SPECTRA_PATH = PROJECT_ROOT / "data" / "processed" / "stage20_transfer_spectra.csv"
CASE_METRICS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage20_transfer_case_metrics.csv"
)
CHANNEL_METRICS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage20_transfer_channel_metrics.csv"
)
DISCREPANCY_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage20_discrepancy_model.csv"
)
MATCHING_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage20_zone_signature_matching.csv"
)
HYPOTHESIS_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage20_hypothesis_checks.csv"
)
VALIDATION_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "stage20_transfer_validation.csv"
)
FIGURE_PATH = PROJECT_ROOT / "figures" / "stage20_openfast_twin_transfer.png"


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(
    category: str,
    name: str,
    observed: object,
    operator: str,
    criterion: object,
) -> dict[str, object]:
    if operator == "==":
        supported = observed == criterion
    elif operator == "<=":
        supported = float(observed) <= float(criterion)
    elif operator == ">=":
        supported = float(observed) >= float(criterion)
    else:
        raise ValueError(f"Unsupported validation operator: {operator}")
    return {
        "category": category,
        "check": name,
        "observed": observed,
        "operator": operator,
        "criterion": criterion,
        "supported": bool(supported),
    }


def _build_reduced_system(
    base_config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    structural = dict(base_config["structural_properties"])
    damping_config = dict(base_config["damping"])
    mass = assemble_mass_matrix(structural["masses_kg"])
    healthy_stiffness = assemble_stiffness_matrix(
        structural["story_stiffness_n_per_m"]
    )
    zone_names, components = assemble_zonal_stiffness_components(
        structural["story_stiffness_n_per_m"],
        structural["stiffness_zones"],
    )
    modes = solve_modal_properties(mass, healthy_stiffness)
    mode_indices = [int(value) - 1 for value in damping_config["target_modes"]]
    damping_targets = [
        float(value) for value in damping_config["target_modal_damping_ratios"]
    ]
    alpha, beta = fit_rayleigh_coefficients(
        modes.angular_frequencies_rad_s[mode_indices[0]],
        modes.angular_frequencies_rad_s[mode_indices[1]],
        damping_targets[0],
        damping_targets[1],
    )
    damping = assemble_rayleigh_damping(
        mass,
        healthy_stiffness,
        alpha_per_s=alpha,
        beta_s=beta,
    ).damping_matrix_n_s_per_m
    return mass, damping, zone_names, components


def _case_stiffness(
    case: pd.Series,
    zone_names: tuple[str, ...],
    zone_map: dict[str, str],
) -> np.ndarray:
    multipliers = np.ones(len(zone_names), dtype=np.float64)
    if str(case["zone_id"]) == "control":
        return multipliers
    reduced_zone = zone_map[str(case["zone_id"])]
    multipliers[zone_names.index(reduced_zone)] = 1.0 - float(
        case["loss_fraction"]
    )
    return multipliers


def _create_figure(
    spectra: pd.DataFrame,
    case_metrics: pd.DataFrame,
    matching: pd.DataFrame,
    calibration_case_id: str,
    calibration_rmse: float,
    inflation_factor: float,
) -> None:
    channel_names = spectra["channel"].drop_duplicates().tolist()
    channel_labels = ["Fore-aft", "Side-side"]
    colors = {
        "lower_monopile": "#D55E00",
        "middle_monopile": "#0072B2",
        "upper_monopile": "#009E73",
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), constrained_layout=True)
    healthy_axes = axes[0]
    similarity_axis = axes[1, 0]
    matching_axis = axes[1, 1]

    for index, (axis, channel, label) in enumerate(
        zip(healthy_axes, channel_names, channel_labels, strict=True)
    ):
        selected = spectra[
            (spectra["case_id"] == calibration_case_id)
            & (spectra["channel"] == channel)
        ]
        axis.semilogy(
            selected["frequency_hz"],
            selected["observed_psd"],
            color="black",
            linewidth=2.0,
            label="OpenFAST",
        )
        axis.semilogy(
            selected["frequency_hz"],
            selected["uncommissioned_psd"],
            color="#999999",
            linewidth=1.3,
            linestyle=":",
            label="uncalibrated twin",
        )
        axis.semilogy(
            selected["frequency_hz"],
            selected["predicted_psd"],
            color="#CC79A7" if index == 0 else "#56B4E9",
            linewidth=1.7,
            linestyle="--",
            label="healthy-calibrated twin",
        )
        axis.set_title(f"{'A' if index == 0 else 'B'}. Healthy {label} spectrum")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("PSD ((m/s²)²/Hz)")
        axis.legend(fontsize=8)

    holdout = case_metrics[case_metrics["loss_fraction"] > 0.0]
    for zone_id, selected in holdout.groupby("zone_id", sort=False):
        selected = selected.sort_values("loss_fraction")
        similarity_axis.plot(
            100.0 * selected["loss_fraction"],
            selected["differential_cosine_similarity"],
            color=colors[str(zone_id)],
            marker="o",
            linewidth=1.8,
            label=str(zone_id).replace("_", " "),
        )
    similarity_axis.axhline(0.0, color="#777777", linewidth=1.0)
    similarity_axis.set_ylim(-1.05, 1.05)
    similarity_axis.set_title("C. Holdout stiffness-change direction")
    similarity_axis.set_xlabel("Local stiffness loss (%)")
    similarity_axis.set_ylabel("Observed/predicted cosine similarity")
    similarity_axis.legend(fontsize=8)

    severe = matching[np.isclose(matching["loss_fraction"], 0.30)]
    observed_zones = severe["observed_zone_id"].drop_duplicates().tolist()
    predicted_zones = severe["predicted_zone_id"].drop_duplicates().tolist()
    matrix = np.zeros((len(observed_zones), len(predicted_zones)))
    for row_index, observed_zone in enumerate(observed_zones):
        for column_index, predicted_zone in enumerate(predicted_zones):
            row = severe[
                (severe["observed_zone_id"] == observed_zone)
                & (severe["predicted_zone_id"] == predicted_zone)
            ]
            matrix[row_index, column_index] = float(row["cosine_similarity"].iloc[0])
    image = matching_axis.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            text_color = (
                "white"
                if abs(matrix[row_index, column_index]) > 0.55
                else "black"
            )
            matching_axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color=text_color,
            )
    matching_axis.set_xticks(
        range(len(predicted_zones)),
        [zone.replace("_monopile", "") for zone in predicted_zones],
    )
    matching_axis.set_yticks(
        range(len(observed_zones)),
        [zone.replace("_monopile", "") for zone in observed_zones],
    )
    matching_axis.set_xlabel("Predicted reduced-twin zone")
    matching_axis.set_ylabel("Observed OpenFAST zone")
    matching_axis.set_title("D. Severe-case signature matching")
    figure.colorbar(image, ax=matching_axis, label="Cosine similarity", shrink=0.82)
    figure.suptitle(
        "Stage 20: healthy-calibrated OpenFAST-to-reduced-twin transfer\n"
        f"healthy log-RMSE={calibration_rmse:.3f}; "
        f"95% discrepancy inflation={inflation_factor:.2f}",
        fontsize=15,
        fontweight="bold",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> int:
    configuration = dict(_load_yaml(CONFIG_PATH)["stage20"])
    upstream = dict(configuration["upstream"])
    spectral_config = dict(configuration["spectral_estimation"])
    twin_config = dict(configuration["reduced_twin"])
    calibration_config = dict(configuration["healthy_calibration"])
    discrepancy_config = dict(configuration["discrepancy"])
    hypothesis_config = dict(configuration["hypotheses"])

    observations_path = PROJECT_ROOT / str(upstream["observations_path"])
    base_model_path = PROJECT_ROOT / str(upstream["base_model_path"])
    stage19_validation_path = PROJECT_ROOT / str(
        upstream["stage19_validation_path"]
    )
    observations_hash_matches = bool(
        _sha256(observations_path) == str(upstream["observations_sha256"])
    )
    base_model_hash_matches = bool(
        _sha256(base_model_path) == str(upstream["base_model_sha256"])
    )
    stage19_validation = pd.read_csv(stage19_validation_path)
    stage19_passed = bool(stage19_validation["supported"].all())
    if not observations_hash_matches or not base_model_hash_matches:
        raise RuntimeError("A pinned Stage 20 input changed unexpectedly.")
    if not stage19_passed:
        raise RuntimeError("Stage 19 validation is not complete.")

    observations = pd.read_csv(observations_path)
    channels = tuple(str(value) for value in spectral_config["channel_names"])
    case_definitions = observations[
        ["case_id", "zone_id", "member_id", "loss_fraction"]
    ].drop_duplicates(ignore_index=True)
    calibration_case_id = str(upstream["calibration_case_id"])
    calibration_case_count = int(
        (case_definitions["case_id"] == calibration_case_id).sum()
    )
    holdout_case_count = len(case_definitions) - calibration_case_count

    case_psd: dict[str, np.ndarray] = {}
    frequencies: np.ndarray | None = None
    for case in case_definitions.itertuples(index=False):
        selected = observations[observations["case_id"] == case.case_id]
        current_frequencies, density = averaged_welch_auto_psd(
            selected["time_s"].to_numpy(),
            selected[list(channels)].to_numpy(),
            start_time_s=float(spectral_config["start_time_s"]),
            segment_samples=int(spectral_config["segment_samples"]),
            minimum_frequency_hz=float(
                spectral_config["minimum_frequency_hz"]
            ),
            maximum_frequency_hz=float(
                spectral_config["maximum_frequency_hz"]
            ),
        )
        if frequencies is None:
            frequencies = current_frequencies
        elif not np.array_equal(frequencies, current_frequencies):
            raise RuntimeError("Stage 20 cases do not share one spectral grid.")
        case_psd[str(case.case_id)] = density
    if frequencies is None:
        raise RuntimeError("No Stage 20 spectra were generated.")

    base_config = _load_yaml(base_model_path)
    mass, damping, zone_names, components = _build_reduced_system(base_config)
    zone_map = {
        str(source): str(target)
        for source, target in dict(
            twin_config["openfast_to_twin_zone_map"]
        ).items()
    }
    zone_mapping_complete = bool(set(zone_map.values()) == set(zone_names))
    force_distribution = np.asarray(
        twin_config["force_distribution"],
        dtype=np.float64,
    )
    top_dof_index = int(
        dict(twin_config["sensor_mapping"])["reduced_model_dof_one_based"]
    ) - 1
    healthy_theta = np.ones(len(zone_names), dtype=np.float64)
    initial_calibration = ReducedTwinCalibration.from_parameter_vector(
        calibration_config["initial_parameters"]
    )

    def healthy_forward(calibration: ReducedTwinCalibration) -> np.ndarray:
        return reduced_twin_directional_psd(
            mass,
            damping,
            components,
            healthy_theta,
            frequencies,
            force_distribution,
            top_dof_index=top_dof_index,
            baseline_force_standard_deviation_n=float(
                twin_config["baseline_force_standard_deviation_n"]
            ),
            calibration=calibration,
        )

    healthy_observed = case_psd[calibration_case_id]
    calibration_result = calibrate_reduced_twin_nuisance(
        healthy_observed,
        healthy_forward,
        initial_parameters=calibration_config["initial_parameters"],
        lower_bounds=calibration_config["lower_bounds"],
        upper_bounds=calibration_config["upper_bounds"],
        maximum_function_evaluations=int(
            calibration_config["maximum_function_evaluations"]
        ),
        robust_loss=str(calibration_config["robust_loss"]),
        robust_scale=float(calibration_config["robust_scale"]),
    )
    calibrated = calibration_result.calibration
    initial_parameters = np.asarray(
        calibration_config["initial_parameters"], dtype=np.float64
    )
    calibrated_parameters = calibrated.as_parameter_vector()
    lower_bounds = np.asarray(calibration_config["lower_bounds"], dtype=float)
    upper_bounds = np.asarray(calibration_config["upper_bounds"], dtype=float)
    calibration_table = pd.DataFrame(
        {
            "parameter": calibration_config["parameter_names"],
            "initial_value": initial_parameters,
            "calibrated_value": calibrated_parameters,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds,
            "relative_lower_margin": (
                (calibrated_parameters - lower_bounds)
                / (upper_bounds - lower_bounds)
            ),
            "relative_upper_margin": (
                (upper_bounds - calibrated_parameters)
                / (upper_bounds - lower_bounds)
            ),
        }
    )
    calibration_table["optimizer_success"] = calibration_result.optimizer_success
    calibration_table["optimizer_status"] = calibration_result.optimizer_status
    calibration_table["function_evaluations"] = (
        calibration_result.function_evaluations
    )
    calibration_table["initial_log_spectral_rmse"] = (
        calibration_result.initial_log_spectral_rmse
    )
    calibration_table["calibrated_log_spectral_rmse"] = (
        calibration_result.calibrated_log_spectral_rmse
    )
    calibration_table["minimum_bound_margin_fraction"] = (
        calibration_result.minimum_bound_margin_fraction
    )
    calibration_table["optimizer_message"] = calibration_result.optimizer_message

    predicted_psd: dict[str, np.ndarray] = {}
    uncommissioned_psd: dict[str, np.ndarray] = {}
    spectra_rows: list[dict[str, object]] = []
    case_metric_rows: list[dict[str, object]] = []
    channel_metric_rows: list[dict[str, object]] = []
    healthy_predicted = healthy_forward(calibrated)
    healthy_observed_signature_reference = healthy_observed
    healthy_predicted_signature_reference = healthy_predicted

    for _, case in case_definitions.iterrows():
        case_id = str(case["case_id"])
        theta = _case_stiffness(case, zone_names, zone_map)
        predicted = reduced_twin_directional_psd(
            mass,
            damping,
            components,
            theta,
            frequencies,
            force_distribution,
            top_dof_index=top_dof_index,
            baseline_force_standard_deviation_n=float(
                twin_config["baseline_force_standard_deviation_n"]
            ),
            calibration=calibrated,
        )
        uncommissioned = reduced_twin_directional_psd(
            mass,
            damping,
            components,
            theta,
            frequencies,
            force_distribution,
            top_dof_index=top_dof_index,
            baseline_force_standard_deviation_n=float(
                twin_config["baseline_force_standard_deviation_n"]
            ),
            calibration=initial_calibration,
        )
        observed = case_psd[case_id]
        predicted_psd[case_id] = predicted
        uncommissioned_psd[case_id] = uncommissioned
        if case_id == calibration_case_id:
            differential_rmse = 0.0
            differential_similarity = 1.0
            observed_signature_norm = 0.0
            predicted_signature_norm = 0.0
        else:
            observed_signature = differential_log_spectral_signature(
                observed,
                healthy_observed_signature_reference,
            )
            predicted_signature = differential_log_spectral_signature(
                predicted,
                healthy_predicted_signature_reference,
            )
            differential_rmse = float(
                np.sqrt(np.mean((predicted_signature - observed_signature) ** 2))
            )
            differential_similarity = cosine_similarity(
                observed_signature,
                predicted_signature,
            )
            observed_signature_norm = float(np.linalg.norm(observed_signature))
            predicted_signature_norm = float(np.linalg.norm(predicted_signature))
        case_metric_rows.append(
            {
                **case.to_dict(),
                "stiffness_theta_lower": theta[0],
                "stiffness_theta_middle": theta[1],
                "stiffness_theta_upper": theta[2],
                "absolute_log_spectral_rmse": log_spectral_rmse(
                    observed, predicted
                ),
                "uncommissioned_log_spectral_rmse": log_spectral_rmse(
                    observed, uncommissioned
                ),
                "differential_log_spectral_rmse": differential_rmse,
                "differential_cosine_similarity": differential_similarity,
                "observed_differential_signature_norm": observed_signature_norm,
                "predicted_differential_signature_norm": predicted_signature_norm,
            }
        )
        for channel_index, channel in enumerate(channels):
            observed_channel = observed[:, channel_index]
            predicted_channel = predicted[:, channel_index]
            peak_observed = float(frequencies[np.argmax(observed_channel)])
            peak_predicted = float(frequencies[np.argmax(predicted_channel)])
            channel_metric_rows.append(
                {
                    **case.to_dict(),
                    "channel": channel,
                    "log_spectral_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    np.log(predicted_channel)
                                    - np.log(observed_channel)
                                )
                                ** 2
                            )
                        )
                    ),
                    "observed_band_energy": float(
                        np.trapezoid(observed_channel, frequencies)
                    ),
                    "predicted_band_energy": float(
                        np.trapezoid(predicted_channel, frequencies)
                    ),
                    "band_energy_ratio_predicted_to_observed": float(
                        np.trapezoid(predicted_channel, frequencies)
                        / np.trapezoid(observed_channel, frequencies)
                    ),
                    "observed_peak_frequency_hz": peak_observed,
                    "predicted_peak_frequency_hz": peak_predicted,
                    "peak_frequency_error_hz": peak_predicted - peak_observed,
                    "log_residual_mean": float(
                        np.mean(np.log(observed_channel / predicted_channel))
                    ),
                    "log_residual_standard_deviation": float(
                        np.std(
                            np.log(observed_channel / predicted_channel),
                            ddof=1,
                        )
                    ),
                }
            )
            for frequency_index, frequency in enumerate(frequencies):
                spectra_rows.append(
                    {
                        **case.to_dict(),
                        "channel": channel,
                        "frequency_hz": frequency,
                        "observed_psd": observed[frequency_index, channel_index],
                        "predicted_psd": predicted[
                            frequency_index, channel_index
                        ],
                        "uncommissioned_psd": uncommissioned[
                            frequency_index, channel_index
                        ],
                        "log_residual_observed_minus_predicted": float(
                            np.log(
                                observed[frequency_index, channel_index]
                                / predicted[frequency_index, channel_index]
                            )
                        ),
                    }
                )

    spectra = pd.DataFrame(spectra_rows)
    case_metrics = pd.DataFrame(case_metric_rows)
    channel_metrics = pd.DataFrame(channel_metric_rows)

    healthy_residual = np.log(healthy_observed / healthy_predicted)
    residual_bias = np.mean(healthy_residual, axis=0)
    residual_scale = np.std(healthy_residual, axis=0, ddof=1)
    holdout_residuals = []
    discrepancy_rows: list[dict[str, object]] = []
    standard_score = float(discrepancy_config["nominal_standard_score"])
    for channel_index, channel in enumerate(channels):
        channel_holdout = []
        for _, case in case_definitions.iterrows():
            case_id = str(case["case_id"])
            if case_id == calibration_case_id:
                continue
            residual = np.log(
                case_psd[case_id][:, channel_index]
                / predicted_psd[case_id][:, channel_index]
            )
            standardized = (
                residual - residual_bias[channel_index]
            ) / residual_scale[channel_index]
            channel_holdout.append(standardized)
            holdout_residuals.append(standardized)
        flattened = np.concatenate(channel_holdout)
        coverage = float(np.mean(np.abs(flattened) <= standard_score))
        empirical_scale_ratio = float(
            np.quantile(np.abs(flattened), 0.95) / standard_score
        )
        inflation = max(1.0, empirical_scale_ratio)
        discrepancy_rows.append(
            {
                "scope": "channel",
                "channel": channel,
                "healthy_log_residual_bias": residual_bias[channel_index],
                "healthy_log_residual_standard_deviation": residual_scale[
                    channel_index
                ],
                "holdout_standardized_residual_coverage": coverage,
                "target_coverage": discrepancy_config["target_coverage"],
                "empirical_95pct_scale_ratio": empirical_scale_ratio,
                "required_95pct_inflation_factor": inflation,
                "holdout_residual_count": flattened.size,
            }
        )
    pooled_standardized = np.concatenate(holdout_residuals)
    pooled_coverage = float(
        np.mean(np.abs(pooled_standardized) <= standard_score)
    )
    pooled_empirical_scale_ratio = float(
        np.quantile(np.abs(pooled_standardized), 0.95) / standard_score
    )
    pooled_inflation = max(1.0, pooled_empirical_scale_ratio)
    discrepancy_rows.append(
        {
            "scope": "pooled",
            "channel": "all",
            "healthy_log_residual_bias": float(np.mean(residual_bias)),
            "healthy_log_residual_standard_deviation": float(
                np.sqrt(np.mean(residual_scale**2))
            ),
            "holdout_standardized_residual_coverage": pooled_coverage,
            "target_coverage": discrepancy_config["target_coverage"],
            "empirical_95pct_scale_ratio": pooled_empirical_scale_ratio,
            "required_95pct_inflation_factor": pooled_inflation,
            "holdout_residual_count": pooled_standardized.size,
        }
    )
    discrepancy = pd.DataFrame(discrepancy_rows)

    severe_cases = case_definitions[np.isclose(case_definitions["loss_fraction"], 0.30)]
    predicted_severe_signatures: dict[str, np.ndarray] = {}
    for _, case in severe_cases.iterrows():
        predicted_severe_signatures[str(case["zone_id"])] = (
            differential_log_spectral_signature(
                predicted_psd[str(case["case_id"])],
                healthy_predicted,
            )
        )
    matching_rows: list[dict[str, object]] = []
    for _, case in severe_cases.iterrows():
        observed_zone = str(case["zone_id"])
        observed_signature = differential_log_spectral_signature(
            case_psd[str(case["case_id"])],
            healthy_observed,
        )
        similarities = {
            predicted_zone: cosine_similarity(observed_signature, signature)
            for predicted_zone, signature in predicted_severe_signatures.items()
        }
        selected_zone = max(similarities, key=similarities.get)
        for predicted_zone, similarity in similarities.items():
            matching_rows.append(
                {
                    "case_id": case["case_id"],
                    "loss_fraction": case["loss_fraction"],
                    "observed_zone_id": observed_zone,
                    "predicted_zone_id": predicted_zone,
                    "cosine_similarity": similarity,
                    "selected_as_best_match": predicted_zone == selected_zone,
                    "correct_zone_selected": selected_zone == observed_zone,
                }
            )
    matching = pd.DataFrame(matching_rows)
    match_outcomes = matching.drop_duplicates("observed_zone_id")[
        "correct_zone_selected"
    ]
    severe_match_fraction = float(match_outcomes.mean())

    holdout_metrics = case_metrics[case_metrics["loss_fraction"] > 0.0]
    improvement_fraction = float(
        1.0
        - calibration_result.calibrated_log_spectral_rmse
        / calibration_result.initial_log_spectral_rmse
    )
    monotonic_results = []
    for zone_id, selected in holdout_metrics.groupby("zone_id", sort=False):
        del zone_id
        selected = selected.sort_values("loss_fraction")
        monotonic_results.append(
            bool(
                np.all(
                    np.diff(
                        selected["predicted_differential_signature_norm"].to_numpy()
                    )
                    > 0.0
                )
            )
        )
    monotonic_fraction = float(np.mean(monotonic_results))
    positive_similarity_fraction = float(
        np.mean(holdout_metrics["differential_cosine_similarity"] > 0.0)
    )
    hypotheses = pd.DataFrame(
        [
            _check(
                "scientific_hypothesis",
                "H20_1_healthy_calibration_improves_log_rmse",
                improvement_fraction,
                ">=",
                hypothesis_config[
                    "minimum_healthy_log_rmse_improvement_fraction"
                ],
            ),
            _check(
                "scientific_hypothesis",
                "H20_2_healthy_calibrated_log_rmse_is_bounded",
                calibration_result.calibrated_log_spectral_rmse,
                "<=",
                hypothesis_config["maximum_healthy_log_spectral_rmse"],
            ),
            _check(
                "scientific_hypothesis",
                "H20_3_mean_holdout_change_direction_is_preserved",
                float(
                    holdout_metrics["differential_cosine_similarity"].mean()
                ),
                ">=",
                hypothesis_config[
                    "minimum_mean_holdout_differential_cosine_similarity"
                ],
            ),
            _check(
                "scientific_hypothesis",
                "H20_4_positive_change_similarity_fraction",
                positive_similarity_fraction,
                ">=",
                hypothesis_config["minimum_positive_similarity_case_fraction"],
            ),
            _check(
                "scientific_hypothesis",
                "H20_5_mean_holdout_differential_log_rmse",
                float(
                    holdout_metrics["differential_log_spectral_rmse"].mean()
                ),
                "<=",
                hypothesis_config[
                    "maximum_mean_holdout_differential_log_rmse"
                ],
            ),
            _check(
                "scientific_hypothesis",
                "H20_6_predicted_severity_is_monotonic_by_zone",
                monotonic_fraction,
                ">=",
                hypothesis_config["required_monotonic_zone_fraction"],
            ),
            _check(
                "scientific_hypothesis",
                "H20_7_severe_zone_signature_match_fraction",
                severe_match_fraction,
                ">=",
                hypothesis_config["minimum_severe_zone_match_fraction"],
            ),
            _check(
                "scientific_hypothesis",
                "H20_8_required_discrepancy_inflation_is_bounded",
                pooled_inflation,
                "<=",
                hypothesis_config["maximum_required_discrepancy_inflation"],
            ),
        ]
    )

    predictions_finite = bool(
        all(
            np.all(np.isfinite(values)) and np.all(values > 0.0)
            for values in predicted_psd.values()
        )
    )
    validation = pd.DataFrame(
        [
            _check("provenance", "stage19_validation", stage19_passed, "==", True),
            _check(
                "provenance",
                "stage19_observations_sha256",
                observations_hash_matches,
                "==",
                True,
            ),
            _check(
                "provenance",
                "base_model_sha256",
                base_model_hash_matches,
                "==",
                True,
            ),
            _check(
                "data_split",
                "healthy_calibration_case_count",
                calibration_case_count,
                "==",
                1,
            ),
            _check(
                "data_split",
                "damaged_holdout_case_count",
                holdout_case_count,
                "==",
                int(upstream["expected_holdout_case_count"]),
            ),
            _check(
                "data_split",
                "holdout_used_for_calibration",
                False,
                "==",
                False,
            ),
            _check(
                "mapping",
                "ordered_zone_mapping_complete",
                zone_mapping_complete,
                "==",
                True,
            ),
            _check(
                "spectral_contract",
                "frequency_bin_count",
                frequencies.size,
                "==",
                40,
            ),
            _check(
                "calibration",
                "optimizer_success",
                calibration_result.optimizer_success,
                "==",
                True,
            ),
            _check(
                "calibration",
                "calibrated_parameter_count",
                calibrated_parameters.size,
                "==",
                6,
            ),
            _check(
                "evaluation",
                "evaluated_case_count",
                len(case_metrics),
                "==",
                len(case_definitions),
            ),
            _check(
                "evaluation",
                "positive_finite_predictions",
                predictions_finite,
                "==",
                True,
            ),
            _check(
                "discrepancy",
                "finite_positive_inflation_factor",
                bool(np.isfinite(pooled_inflation) and pooled_inflation > 0.0),
                "==",
                True,
            ),
        ]
    )

    for path in (
        CALIBRATION_PATH,
        SPECTRA_PATH,
        CASE_METRICS_PATH,
        CHANNEL_METRICS_PATH,
        DISCREPANCY_PATH,
        MATCHING_PATH,
        HYPOTHESIS_PATH,
        VALIDATION_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    calibration_table.to_csv(CALIBRATION_PATH, index=False, float_format="%.12g")
    spectra.to_csv(SPECTRA_PATH, index=False, float_format="%.12g")
    case_metrics.to_csv(CASE_METRICS_PATH, index=False, float_format="%.12g")
    channel_metrics.to_csv(
        CHANNEL_METRICS_PATH, index=False, float_format="%.12g"
    )
    discrepancy.to_csv(DISCREPANCY_PATH, index=False, float_format="%.12g")
    matching.to_csv(MATCHING_PATH, index=False, float_format="%.12g")
    hypotheses.to_csv(HYPOTHESIS_PATH, index=False, float_format="%.12g")
    validation.to_csv(VALIDATION_PATH, index=False, float_format="%.12g")
    _create_figure(
        spectra,
        case_metrics,
        matching,
        calibration_case_id,
        calibration_result.calibrated_log_spectral_rmse,
        pooled_inflation,
    )

    print("STAGE 20 - OPENFAST TO REDUCED-TWIN TRANSFER")
    print(
        f"Data split: {calibration_case_count} healthy calibration case, "
        f"{holdout_case_count} damaged holdouts"
    )
    print(
        f"Healthy log-RMSE: {calibration_result.initial_log_spectral_rmse:.3f} "
        f"-> {calibration_result.calibrated_log_spectral_rmse:.3f}"
    )
    print(
        f"Mean holdout differential similarity: "
        f"{holdout_metrics['differential_cosine_similarity'].mean():.3f}"
    )
    print(f"Severe zone matching: {severe_match_fraction:.3f}")
    print(
        f"Pooled holdout coverage before inflation: {pooled_coverage:.3f}; "
        f"required 95% inflation={pooled_inflation:.3f}"
    )
    print(f"Core validation: {int(validation['supported'].sum())}/{len(validation)}")
    print(
        f"Scientific hypotheses: "
        f"{int(hypotheses['supported'].sum())}/{len(hypotheses)}"
    )
    print(f"Calibration: {CALIBRATION_PATH}")
    print(f"Discrepancy: {DISCREPANCY_PATH}")
    print(f"Figure: {FIGURE_PATH}")
    if bool(validation["supported"].all()):
        print("RESULT: PASS (fidelity-transfer experiment complete)")
        return 0
    print("RESULT: FAIL (Stage 20 core validation failed)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
