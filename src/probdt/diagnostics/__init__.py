"""Diagnostics for identifiability, calibration, and estimator credibility."""

from probdt.diagnostics.factorial import (
    FactorialAnalysisError,
    FactorialEffects,
    estimate_two_level_factorial_effects,
)
from probdt.diagnostics.identifiability import (
    IdentifiabilityError,
    IdentifiabilityReport,
    acceleration_auto_psd,
    analyze_identifiability,
    central_difference_sensitivity,
    log_spectral_features,
)
from probdt.diagnostics.model_discrepancy import (
    ModelDiscrepancyError,
    SpectralModelDiscrepancy,
    coherent_log_spectral_residual_scores,
    discrepant_acceleration_auto_psd,
)
from probdt.diagnostics.monte_carlo import (
    MonteCarloAnalysisError,
    MonteCarloMeanInterval,
    monte_carlo_mean_interval,
    paired_monte_carlo_difference,
)
from probdt.diagnostics.pipeline import (
    ArtifactAudit,
    PipelineAuditError,
    audit_pipeline_artifacts,
    select_pipeline_stages,
)

__all__ = [
    "ArtifactAudit",
    "FactorialAnalysisError",
    "FactorialEffects",
    "IdentifiabilityError",
    "IdentifiabilityReport",
    "ModelDiscrepancyError",
    "MonteCarloAnalysisError",
    "MonteCarloMeanInterval",
    "PipelineAuditError",
    "SpectralModelDiscrepancy",
    "acceleration_auto_psd",
    "analyze_identifiability",
    "audit_pipeline_artifacts",
    "central_difference_sensitivity",
    "coherent_log_spectral_residual_scores",
    "discrepant_acceleration_auto_psd",
    "estimate_two_level_factorial_effects",
    "log_spectral_features",
    "monte_carlo_mean_interval",
    "paired_monte_carlo_difference",
    "select_pipeline_stages",
]
