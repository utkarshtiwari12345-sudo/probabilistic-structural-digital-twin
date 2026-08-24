"""Time-evolving deterioration truth models."""

from probdt.deterioration.processes import (
    ConditionThresholds,
    DeteriorationError,
    DeteriorationHistory,
    DeteriorationValidationReport,
    abrupt_loss,
    build_deterioration_history,
    classify_condition,
    combine_independent_losses,
    gamma_process_loss,
    smooth_gradual_loss,
    validate_deterioration_history,
)

__all__ = [
    "ConditionThresholds",
    "DeteriorationError",
    "DeteriorationHistory",
    "DeteriorationValidationReport",
    "abrupt_loss",
    "build_deterioration_history",
    "classify_condition",
    "combine_independent_losses",
    "gamma_process_loss",
    "smooth_gradual_loss",
    "validate_deterioration_history",
]
