from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_v2_generator_conditioning_artifact.json"
SCHEMA_VERSION = "synthetic_generator_conditioning.v2"


@dataclass(frozen=True)
class GeneratorConditioning:
    profile_id: str
    capability_id: str
    context_length: int
    horizon: int
    target_dim: int
    season_length: int
    frequency: str
    parameters: dict[str, float]
    intensity_lambdas: tuple[float, ...]
    canonical_reference_percentile_levels: tuple[float, ...]
    canonical_target_feature: str
    canonical_target_values: tuple[float, ...]
    calibrated_realized_strengths: tuple[float, ...]
    local_real_percentiles: tuple[float, ...]
    local_real_target_quantiles: dict[str, tuple[float, ...]]
    calibration_max_normalized_error: float
    canonical_scale_id: str
    canonical_scale_fingerprint: str
    artifact_schema_version: str
    artifact_created_at: str | None
    calibration_method: str

    def lambda_for(self, intensity: int) -> float:
        index = int(intensity) - 1
        if index < 0 or index >= len(self.intensity_lambdas):
            raise ValueError("intensity must be between 1 and 5")
        return float(self.intensity_lambdas[index])

    def metadata(self, intensity: int) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_schema_version": self.artifact_schema_version,
            "artifact_created_at": self.artifact_created_at,
            "profile_id": self.profile_id,
            "capability_id": self.capability_id,
            "context_length": self.context_length,
            "horizon": self.horizon,
            "target_dim": self.target_dim,
            "season_length": self.season_length,
            "frequency": self.frequency,
            "intensity": int(intensity),
            "base_lambda": (int(intensity) - 1) / 4,
            "profile_lambda": self.lambda_for(intensity),
            "intensity_semantics": "capability-global canonical realized strength; not model difficulty",
            "canonical_scale_id": self.canonical_scale_id,
            "canonical_scale_fingerprint": self.canonical_scale_fingerprint,
            "canonical_reference_percentile_level": self.canonical_reference_percentile_levels[
                int(intensity) - 1
            ],
            "canonical_target_feature": self.canonical_target_feature,
            "canonical_target_strength": self.canonical_target_values[int(intensity) - 1],
            "calibrated_profile_expected_strength": self.calibrated_realized_strengths[
                int(intensity) - 1
            ],
            "local_real_percentile": self.local_real_percentiles[int(intensity) - 1],
            "local_real_target_quantiles": {
                name: list(values) for name, values in self.local_real_target_quantiles.items()
            },
            "calibration_max_normalized_error": self.calibration_max_normalized_error,
            "parameters": dict(self.parameters),
            "calibration_method": self.calibration_method,
        }


@lru_cache(maxsize=1)
def load_generator_conditioning_artifact() -> dict[str, Any] | None:
    if not ARTIFACT_PATH.exists():
        return None
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def matching_generator_profiles(
    *,
    capability_id: str,
    profile_ids: tuple[str, ...],
    context_length: int,
    horizon: int,
    target_dim: int,
    frequency: str | None = None,
    artifact: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source = artifact if artifact is not None else load_generator_conditioning_artifact()
    if not source:
        return []
    requested_frequency = _canonical_frequency(frequency) if frequency else None
    profiles = source.get("profiles", {})
    matches: list[dict[str, Any]] = []
    for profile_id in profile_ids:
        profile = profiles.get(profile_id)
        if profile is None or capability_id not in profile.get("capabilities", {}):
            continue
        capability = profile["capabilities"][capability_id]
        if capability.get("canonical_calibration", {}).get("status") != "supported":
            continue
        if int(profile.get("context_length", -1)) != int(context_length):
            continue
        if int(profile.get("horizon", -1)) != int(horizon):
            continue
        if int(profile.get("target_dim", -1)) != int(target_dim):
            continue
        if requested_frequency and _canonical_frequency(str(profile.get("frequency", ""))) != requested_frequency:
            continue
        matches.append(profile)
    return matches


def resolve_generator_conditioning(
    *,
    capability_id: str,
    profile_id: str,
    context_length: int,
    horizon: int,
    target_dim: int,
    artifact: dict[str, Any] | None = None,
) -> GeneratorConditioning | None:
    source = artifact if artifact is not None else load_generator_conditioning_artifact()
    if not source:
        return None
    matches = matching_generator_profiles(
        capability_id=capability_id,
        profile_ids=(profile_id,),
        context_length=context_length,
        horizon=horizon,
        target_dim=target_dim,
        artifact=source,
    )
    if not matches:
        return None
    profile = matches[0]
    capability = profile["capabilities"][capability_id]
    parameters = {
        str(name): float(value)
        for name, value in {
            **profile.get("nuisance_parameters", {}),
            **capability.get("parameters", {}),
        }.items()
    }
    intensity_lambdas = tuple(float(value) for value in capability.get("intensity_lambdas", ()))
    reference_percentile_levels = tuple(
        float(value) for value in capability.get("canonical_reference_percentile_levels", ())
    )
    canonical_target_values = tuple(
        float(value) for value in capability.get("canonical_target_values", ())
    )
    calibrated_realized_strengths = tuple(
        float(value) for value in capability.get("calibrated_realized_strengths", ())
    )
    local_real_percentiles = tuple(
        float(value) for value in capability.get("local_real_percentiles_at_canonical_targets", ())
    )
    if len(intensity_lambdas) != 5 or any(
        right < left for left, right in zip(intensity_lambdas, intensity_lambdas[1:])
    ):
        raise ValueError(f"invalid intensity_lambdas for {profile_id}/{capability_id}")
    five_level_fields = {
        "canonical_reference_percentile_levels": reference_percentile_levels,
        "canonical_target_values": canonical_target_values,
        "calibrated_realized_strengths": calibrated_realized_strengths,
        "local_real_percentiles_at_canonical_targets": local_real_percentiles,
    }
    for field_name, values in five_level_fields.items():
        if len(values) != 5:
            raise ValueError(f"invalid {field_name} for {profile_id}/{capability_id}")
    canonical_definition = source.get("canonical_intensity", {}).get("capabilities", {}).get(
        capability_id,
        {},
    )
    artifact_targets = tuple(
        float(value) for value in canonical_definition.get("target_values", ())
    )
    if artifact_targets != canonical_target_values:
        raise ValueError(f"profile canonical targets do not match artifact for {profile_id}/{capability_id}")
    canonical_intensity = source.get("canonical_intensity", {})
    canonical_scale_id = str(canonical_intensity.get("scale_id", ""))
    canonical_scale_fingerprint = str(canonical_intensity.get("scale_fingerprint", ""))
    if not canonical_scale_id or not canonical_scale_fingerprint:
        raise ValueError("generator conditioning artifact has no canonical scale identity")
    local_real_target_quantiles = {
        str(name): tuple(float(value) for value in values)
        for name, values in capability.get("local_real_target_quantiles", {}).items()
    }
    return GeneratorConditioning(
        profile_id=str(profile["profile_id"]),
        capability_id=capability_id,
        context_length=int(profile["context_length"]),
        horizon=int(profile["horizon"]),
        target_dim=int(profile["target_dim"]),
        season_length=int(profile["season_length"]),
        frequency=str(profile.get("frequency", "")),
        parameters=parameters,
        intensity_lambdas=intensity_lambdas,
        canonical_reference_percentile_levels=reference_percentile_levels,
        canonical_target_feature=str(capability["canonical_target_feature"]),
        canonical_target_values=canonical_target_values,
        calibrated_realized_strengths=calibrated_realized_strengths,
        local_real_percentiles=local_real_percentiles,
        local_real_target_quantiles=local_real_target_quantiles,
        calibration_max_normalized_error=float(
            capability.get("canonical_calibration", {}).get("max_normalized_error", float("inf"))
        ),
        canonical_scale_id=canonical_scale_id,
        canonical_scale_fingerprint=canonical_scale_fingerprint,
        artifact_schema_version=str(source.get("schema_version", "unknown")),
        artifact_created_at=source.get("created_at"),
        calibration_method=str(capability.get("calibration_method", "unknown")),
    )


def select_balanced_profile_id(
    profile_ids: tuple[str, ...],
    *,
    capability_id: str,
    seed: int,
    sample_index: int,
) -> str:
    if not profile_ids:
        raise ValueError("at least one profile is required")
    ordered = tuple(sorted(profile_ids))
    payload = f"{int(seed)}:{capability_id}:anchor-profile-order".encode("utf-8")
    offset = int(hashlib.blake2s(payload, digest_size=8).hexdigest(), 16) % len(ordered)
    return ordered[(offset + int(sample_index)) % len(ordered)]


def _canonical_frequency(frequency: str | None) -> str:
    value = (frequency or "").strip().lower()
    aliases = {
        "hour": "h",
        "hourly": "h",
        "day": "d",
        "daily": "d",
        "minute": "1min",
        "min": "1min",
    }
    return aliases.get(value, value)
