from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "cafe.generator_conditioning.v1"
INTENSITY_POLICY_ID = "dataset-local-relative-quantiles-v1"
REAL_BOUNDED_INTENSITY_POLICY_ID = (
    "dataset-local-real-bounded-generator-feasible-v1"
)
SUPPORTED_INTENSITY_POLICY_IDS = frozenset(
    {INTENSITY_POLICY_ID, REAL_BOUNDED_INTENSITY_POLICY_ID}
)


@dataclass(frozen=True)
class GeneratorConditioning:
    profile_id: str
    dataset_id: str
    capability_id: str
    context_length: int
    horizon: int
    target_dim: int
    season_length: int
    frequency: str
    parameters: dict[str, float]
    intensity_lambdas: tuple[float, ...]
    target_percentile_levels: tuple[float, ...]
    target_feature: str
    target_values: tuple[float, ...]
    calibrated_realized_strengths: tuple[float, ...]
    calibration_max_normalized_error: float
    intensity_policy_id: str
    artifact_schema_version: str
    artifact_created_at: str | None
    calibration_method: str
    artifact_generator_version: str | None = None

    def lambda_for(self, intensity: int) -> float:
        index = int(intensity) - 1
        if index < 0 or index >= len(self.intensity_lambdas):
            raise ValueError("intensity must be between 1 and 5")
        return float(self.intensity_lambdas[index])

    def metadata(self, intensity: int) -> dict[str, Any]:
        target_level = self.target_percentile_levels[int(intensity) - 1]
        real_bounded = (
            self.intensity_policy_id == REAL_BOUNDED_INTENSITY_POLICY_ID
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_schema_version": self.artifact_schema_version,
            "artifact_created_at": self.artifact_created_at,
            "artifact_generator_version": self.artifact_generator_version,
            "profile_id": self.profile_id,
            "dataset_id": self.dataset_id,
            "capability_id": self.capability_id,
            "context_length": self.context_length,
            "horizon": self.horizon,
            "target_dim": self.target_dim,
            "season_length": self.season_length,
            "frequency": self.frequency,
            "intensity": int(intensity),
            "base_lambda": (int(intensity) - 1) / 4,
            "profile_lambda": self.lambda_for(intensity),
            "intensity_semantics": (
                (
                    "dataset-local relative position inside the real-bounded "
                    "generator-feasible interval"
                )
                if real_bounded
                else "dataset-local relative strength quantile"
            ),
            "intensity_policy_id": self.intensity_policy_id,
            "target_relative_level": target_level,
            # Kept for existing API/import consumers. Under the real-bounded
            # policy this is a serialization alias, not an empirical quantile.
            "target_percentile_level": target_level,
            "target_level_semantics": (
                "relative_position"
                if real_bounded
                else "empirical_quantile"
            ),
            "target_feature": self.target_feature,
            "target_strength": self.target_values[int(intensity) - 1],
            "calibrated_expected_strength": self.calibrated_realized_strengths[int(intensity) - 1],
            "calibration_max_normalized_error": self.calibration_max_normalized_error,
            "parameters": dict(self.parameters),
            "calibration_method": self.calibration_method,
        }
