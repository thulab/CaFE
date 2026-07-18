from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_v2_generator_conditioning_artifact.json"
SCHEMA_VERSION = "synthetic_generator_conditioning.v4"
ARTIFACT_SCHEMA_VERSION = "synthetic_v2_generator_conditioning_artifact.v4"
INTENSITY_POLICY_ID = "dataset-local-relative-quantiles-v1"


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
                "dataset-local relative strength quantile; target strength is not comparable "
                "across datasets and is not model difficulty"
            ),
            "intensity_policy_id": self.intensity_policy_id,
            "target_percentile_level": self.target_percentile_levels[int(intensity) - 1],
            "target_feature": self.target_feature,
            "target_strength": self.target_values[int(intensity) - 1],
            "calibrated_expected_strength": self.calibrated_realized_strengths[int(intensity) - 1],
            "calibration_max_normalized_error": self.calibration_max_normalized_error,
            "parameters": dict(self.parameters),
            "calibration_method": self.calibration_method,
        }


@lru_cache(maxsize=1)
def load_generator_conditioning_artifact() -> dict[str, Any] | None:
    if not ARTIFACT_PATH.exists():
        return None
    try:
        artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return artifact if _is_compatible_artifact(artifact) else None


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
    if not source or not _is_compatible_artifact(source):
        return []
    requested_frequency = _canonical_frequency(frequency) if frequency else None
    profiles = source.get("profiles", {})
    matches: list[dict[str, Any]] = []
    for profile_id in profile_ids:
        profile = profiles.get(profile_id)
        if not isinstance(profile, dict):
            continue
        capabilities = profile.get("capabilities")
        if not isinstance(capabilities, dict) or capability_id not in capabilities:
            continue
        capability = capabilities[capability_id]
        if not _is_compatible_profile_capability(
            profile_id=profile_id,
            profile=profile,
            capability=capability,
            policy_percentile_levels=tuple(
                float(value) for value in source["intensity_policy"]["percentile_levels"]
            ),
        ):
            continue
        try:
            has_requested_shape = (
                int(profile.get("context_length", -1)) == int(context_length)
                and int(profile.get("horizon", -1)) == int(horizon)
                and int(profile.get("target_dim", -1)) == int(target_dim)
            )
        except (TypeError, ValueError):
            has_requested_shape = False
        if not has_requested_shape:
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
    if not source or not _is_compatible_artifact(source):
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
    target_percentile_levels = tuple(
        float(value) for value in capability.get("target_percentile_levels", ())
    )
    target_values = tuple(float(value) for value in capability.get("target_values", ()))
    calibrated_realized_strengths = tuple(
        float(value) for value in capability.get("calibrated_realized_strengths", ())
    )
    if not _valid_five_level_curve(intensity_lambdas, strict=False):
        raise ValueError(f"invalid intensity_lambdas for {profile_id}/{capability_id}")
    policy_percentile_levels = tuple(
        float(value) for value in source["intensity_policy"]["percentile_levels"]
    )
    five_level_fields = {
        "target_percentile_levels": target_percentile_levels,
        "target_values": target_values,
        "calibrated_realized_strengths": calibrated_realized_strengths,
    }
    for field_name, values in five_level_fields.items():
        if not _valid_five_level_curve(values, strict=field_name != "calibrated_realized_strengths"):
            raise ValueError(f"invalid {field_name} for {profile_id}/{capability_id}")
    if target_percentile_levels != policy_percentile_levels:
        raise ValueError(
            f"profile percentile levels do not match intensity policy for {profile_id}/{capability_id}"
        )
    target_feature = str(capability.get("target_feature", "")).strip()
    if not target_feature:
        raise ValueError(f"missing target_feature for {profile_id}/{capability_id}")
    return GeneratorConditioning(
        profile_id=str(profile["profile_id"]),
        dataset_id=str(profile["dataset_id"]),
        capability_id=capability_id,
        context_length=int(profile["context_length"]),
        horizon=int(profile["horizon"]),
        target_dim=int(profile["target_dim"]),
        season_length=int(profile["season_length"]),
        frequency=str(profile.get("frequency", "")),
        parameters=parameters,
        intensity_lambdas=intensity_lambdas,
        target_percentile_levels=target_percentile_levels,
        target_feature=target_feature,
        target_values=target_values,
        calibrated_realized_strengths=calibrated_realized_strengths,
        calibration_max_normalized_error=float(
            capability.get("calibration", {}).get("max_normalized_error", float("inf"))
        ),
        intensity_policy_id=str(source["intensity_policy"]["policy_id"]),
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


def _is_compatible_artifact(artifact: Any) -> bool:
    if not isinstance(artifact, dict):
        return False
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        return False
    policy = artifact.get("intensity_policy")
    if not isinstance(policy, dict) or policy.get("policy_id") != INTENSITY_POLICY_ID:
        return False
    if not str(policy.get("definition", "")).strip():
        return False
    try:
        percentile_levels = tuple(float(value) for value in policy.get("percentile_levels", ()))
    except (TypeError, ValueError):
        return False
    return (
        _valid_five_level_curve(percentile_levels, strict=True)
        and percentile_levels[0] >= 0.0
        and percentile_levels[-1] <= 1.0
        and isinstance(artifact.get("profiles"), dict)
    )


def _valid_five_level_curve(values: tuple[float, ...], *, strict: bool) -> bool:
    if len(values) != 5 or not all(math.isfinite(value) for value in values):
        return False
    comparisons = zip(values, values[1:])
    if strict:
        return all(right > left for left, right in comparisons)
    return all(right >= left for left, right in comparisons)


def _is_compatible_profile_capability(
    *,
    profile_id: str,
    profile: Any,
    capability: Any,
    policy_percentile_levels: tuple[float, ...],
) -> bool:
    if not isinstance(profile, dict) or not isinstance(capability, dict):
        return False
    if str(profile.get("profile_id", "")) != profile_id:
        return False
    if not str(profile.get("dataset_id", "")).strip():
        return False
    calibration = capability.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("status") != "supported":
        return False
    if not str(capability.get("target_feature", "")).strip():
        return False
    try:
        target_percentile_levels = tuple(
            float(value) for value in capability.get("target_percentile_levels", ())
        )
        target_values = tuple(float(value) for value in capability.get("target_values", ()))
        realized = tuple(
            float(value) for value in capability.get("calibrated_realized_strengths", ())
        )
        lambdas = tuple(float(value) for value in capability.get("intensity_lambdas", ()))
        calibration_error = float(calibration.get("max_normalized_error"))
        parameters = {
            str(name): float(value)
            for name, value in {
                **profile.get("nuisance_parameters", {}),
                **capability.get("parameters", {}),
            }.items()
        }
        positive_shape = all(
            int(profile.get(name, 0)) > 0
            for name in ("context_length", "horizon", "target_dim", "season_length")
        )
    except (TypeError, ValueError):
        return False
    return (
        positive_shape
        and target_percentile_levels == policy_percentile_levels
        and _valid_five_level_curve(target_percentile_levels, strict=True)
        and _valid_five_level_curve(target_values, strict=True)
        and _valid_five_level_curve(realized, strict=False)
        and _valid_five_level_curve(lambdas, strict=False)
        and lambdas[0] >= 0.0
        and lambdas[-1] <= 1.0
        and math.isfinite(calibration_error)
        and calibration_error >= 0.0
        and all(math.isfinite(value) for value in parameters.values())
    )
