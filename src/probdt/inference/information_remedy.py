"""Mixed observation channels for augmented structural-state inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probdt.inference.joint_state import (
    AugmentedStructuralModel,
    JointStateInferenceError,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MixedObservationModel:
    """Filter-compatible structural model with heterogeneous measurements.

    Channels are ordered as acceleration, latent force, displacement,
    interstory strain, and modal coordinate.  The ``acceleration`` method is
    retained as the filter observation interface; it returns whichever mixed
    channels are requested by their zero-based channel indices.
    """

    structural_model: AugmentedStructuralModel
    acceleration_dof_indices: tuple[int, ...]
    displacement_dof_indices: tuple[int, ...] = ()
    strain_story_indices: tuple[int, ...] = ()
    level_spacing_m: float = 1.0
    modal_projection_vectors: FloatArray | None = None
    include_latent_force: bool = False

    def __post_init__(self) -> None:
        model = self.structural_model
        for name, indices in (
            ("acceleration_dof_indices", self.acceleration_dof_indices),
            ("displacement_dof_indices", self.displacement_dof_indices),
            ("strain_story_indices", self.strain_story_indices),
        ):
            if len(set(indices)) != len(indices):
                raise JointStateInferenceError(f"{name} must not contain duplicates.")
            if indices and (min(indices) < 0 or max(indices) >= model.n_dof):
                raise JointStateInferenceError(f"{name} contains an invalid DOF.")
        if not self.acceleration_dof_indices:
            raise JointStateInferenceError(
                "At least one acceleration channel is required."
            )
        spacing = float(self.level_spacing_m)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise JointStateInferenceError(
                "level_spacing_m must be finite and positive."
            )
        if self.modal_projection_vectors is None:
            modal = np.empty((0, model.n_dof), dtype=np.float64)
        else:
            modal = np.asarray(self.modal_projection_vectors, dtype=np.float64)
            if (
                modal.ndim != 2
                or modal.shape[1] != model.n_dof
                or not np.all(np.isfinite(modal))
            ):
                raise JointStateInferenceError(
                    "modal_projection_vectors must be finite with one column per DOF."
                )
        object.__setattr__(self, "modal_projection_vectors", modal.copy())

    @property
    def n_dof(self) -> int:
        return self.structural_model.n_dof

    @property
    def n_zones(self) -> int:
        return self.structural_model.n_zones

    @property
    def n_state(self) -> int:
        return self.structural_model.n_state

    @property
    def load_index(self) -> int:
        return self.structural_model.load_index

    @property
    def theta_slice(self) -> slice:
        return self.structural_model.theta_slice

    @property
    def force_correlation_time_s(self) -> float:
        return self.structural_model.force_correlation_time_s

    @property
    def n_channels(self) -> int:
        return len(self.channel_names())

    def state_names(self) -> tuple[str, ...]:
        return self.structural_model.state_names()

    def channel_names(self) -> tuple[str, ...]:
        names = [
            *(
                f"acceleration_dof_{index + 1}"
                for index in self.acceleration_dof_indices
            ),
        ]
        if self.include_latent_force:
            names.append("latent_force")
        names.extend(
            f"displacement_dof_{index + 1}"
            for index in self.displacement_dof_indices
        )
        names.extend(
            f"strain_story_{index + 1}" for index in self.strain_story_indices
        )
        names.extend(
            f"modal_coordinate_{index + 1}"
            for index in range(self.modal_projection_vectors.shape[0])
        )
        return tuple(names)

    def channel_types(self) -> tuple[str, ...]:
        types = ["acceleration"] * len(self.acceleration_dof_indices)
        if self.include_latent_force:
            types.append("force")
        types.extend(["displacement"] * len(self.displacement_dof_indices))
        types.extend(["strain"] * len(self.strain_story_indices))
        types.extend(["modal"] * self.modal_projection_vectors.shape[0])
        return tuple(types)

    def derivative(self, states: ArrayLike) -> FloatArray:
        return self.structural_model.derivative(states)

    def propagate_rk4(self, states: ArrayLike, time_step_s: float) -> FloatArray:
        return self.structural_model.propagate_rk4(states, time_step_s)

    def observe(self, states: ArrayLike) -> FloatArray:
        """Evaluate every configured observation channel."""

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1] != self.n_state or not np.all(np.isfinite(values)):
            raise JointStateInferenceError(
                "states must be finite with the augmented state trailing size."
            )
        q = values[..., : self.n_dof]
        channels: list[FloatArray] = [
            self.structural_model.acceleration(
                values,
                self.acceleration_dof_indices,
            )
        ]
        if self.include_latent_force:
            channels.append(values[..., self.load_index, None])
        if self.displacement_dof_indices:
            channels.append(q[..., list(self.displacement_dof_indices)])
        if self.strain_story_indices:
            strains = []
            for story in self.strain_story_indices:
                lower = 0.0 if story == 0 else q[..., story - 1]
                strains.append((q[..., story] - lower) / self.level_spacing_m)
            channels.append(np.stack(strains, axis=-1))
        if self.modal_projection_vectors.shape[0] > 0:
            channels.append(q @ self.modal_projection_vectors.T)
        return np.concatenate(channels, axis=-1)

    def acceleration(
        self,
        states: ArrayLike,
        channel_indices: tuple[int, ...],
    ) -> FloatArray:
        """Return selected heterogeneous channels through the filter interface."""

        if (
            not channel_indices
            or len(set(channel_indices)) != len(channel_indices)
            or min(channel_indices) < 0
            or max(channel_indices) >= self.n_channels
        ):
            raise JointStateInferenceError("Observation channel indices are invalid.")
        return self.observe(states)[..., list(channel_indices)]
