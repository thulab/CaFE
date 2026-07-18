#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.synthetic_capability_contrast import (  # noqa: E402
    evaluate_capability_contrast,
    summarize_capability_contrasts,
)
from app.services.synthetic_generation_service import (  # noqa: E402
    _generate_sample_values,
    _standardize_by_context,
    _standardize_hierarchy_by_context,
)
from app.services.synthetic_generator_conditioning import (  # noqa: E402
    INTENSITY_POLICY_ID,
    load_generator_conditioning_artifact,
    resolve_generator_conditioning,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runtime/research/synthetic-capability-shortcut-audit/summary.json"
)
AUDIT_PROFILE_BY_CAPABILITY = {
    "trend": "traffic_hourly_daily_168ctx",
    "multi_seasonal": "m4_hourly_daily_168ctx",
    "time_varying_seasonality": "electricity_hourly_daily_168ctx",
    "regime_switching": "traffic_hourly_daily_168ctx",
    "nonlinear_persistence": "traffic_hourly_daily_168ctx",
    "predictable_intermittency": "m4_hourly_daily_168ctx",
    "common_factor": "traffic_hourly_panel_168ctx",
    "hierarchical_coherence": "m5_daily_hierarchy_365ctx_28h",
    "covariate_response": "gefcom2014_load_hourly_covariate_168ctx_24h",
}
NONSEASONAL_CAPABILITIES = {
    "trend",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
    "common_factor",
    "hierarchical_coherence",
    "covariate_response",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit fixed-seasonal-naive shortcuts and capability-aware headroom "
            "on profile-conditioned synthetic generators."
        )
    )
    parser.add_argument("--seed-count", type=int, default=128)
    parser.add_argument(
        "--intensities",
        nargs="+",
        type=int,
        default=[1, 3, 5],
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_audit(
        seed_count=args.seed_count,
        intensities=tuple(args.intensities),
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote shortcut audit: {args.output}")
    print(f"overall passed: {summary['overall_passed']}")
    return 0 if summary["overall_passed"] else 1


def run_audit(
    *,
    seed_count: int,
    intensities: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    if seed_count < 24:
        raise ValueError("seed_count must be at least 24 for aggregate qualification")
    if not intensities or any(value < 1 or value > 5 for value in intensities):
        raise ValueError("intensities must be between 1 and 5")
    artifact = load_generator_conditioning_artifact()
    if artifact is None:
        raise FileNotFoundError("generator conditioning artifact is missing")
    if artifact["intensity_policy"].get("policy_id") != INTENSITY_POLICY_ID:
        raise ValueError(f"shortcut audit requires intensity policy {INTENSITY_POLICY_ID}")
    rows: list[dict[str, Any]] = []
    unsupported_cells: list[dict[str, Any]] = []
    high_intensity = max(intensities)
    for capability_id, profile_id in AUDIT_PROFILE_BY_CAPABILITY.items():
        profile = artifact["profiles"].get(profile_id)
        if profile is None:
            unsupported_cells.append(
                {
                    "dataset_id": None,
                    "profile_id": profile_id,
                    "capability_id": capability_id,
                    "reason": "profile_missing",
                }
            )
            continue
        capability = profile.get("capabilities", {}).get(capability_id)
        if not _is_supported_capability(capability):
            calibration = (
                capability.get("calibration", {})
                if isinstance(capability, dict)
                else {}
            )
            unsupported_cells.append(
                {
                    "dataset_id": profile.get("dataset_id"),
                    "profile_id": profile_id,
                    "capability_id": capability_id,
                    "reason": (
                        "capability_missing"
                        if capability is None
                        else (
                            capability.get(
                                "unsupported_reason",
                                calibration.get("status", "unsupported"),
                            )
                            if isinstance(capability, dict)
                            else "invalid_capability_config"
                        )
                    ),
                }
            )
            continue
        context_length = int(profile["context_length"])
        horizon = int(profile["horizon"])
        target_dim = int(profile["target_dim"])
        season_length = int(profile["season_length"])
        conditioning = resolve_generator_conditioning(
            capability_id=capability_id,
            profile_id=profile_id,
            context_length=context_length,
            horizon=horizon,
            target_dim=target_dim,
            artifact=artifact,
        )
        if conditioning is None:
            raise ValueError(
                f"missing conditioning for {profile_id}/{capability_id}"
            )
        for intensity in intensities:
            contrast_rows: list[dict[str, Any]] = []
            seasonal_errors: list[float] = []
            last_errors: list[float] = []
            for seed_index in range(seed_count):
                sample_seed = _audit_seed(
                    seed,
                    capability_id,
                    intensity,
                    seed_index,
                )
                target, latent, covariates = _generate_sample_values(
                    capability_id,
                    context_length + horizon,
                    context_length,
                    target_dim,
                    season_length,
                    intensity,
                    np.random.default_rng(sample_seed),
                    generator_conditioning=conditioning,
                )
                contrast_rows.append(
                    evaluate_capability_contrast(
                        capability_id=capability_id,
                        target=target,
                        context_length=context_length,
                        season_length=season_length,
                        intensity=intensity,
                        latent_params=latent,
                        covariates=covariates,
                        evaluation_scale="generator_raw",
                    )
                )
                standardized = (
                    _standardize_hierarchy_by_context(
                        target,
                        context_length,
                    )
                    if capability_id == "hierarchical_coherence"
                    else _standardize_by_context(target, context_length)
                )
                history = standardized[:context_length]
                future = standardized[context_length:]
                pattern = history[-min(season_length, context_length) :]
                seasonal = np.vstack(
                    [
                        pattern[index % len(pattern)]
                        for index in range(horizon)
                    ]
                )
                last = np.repeat(history[-1:], horizon, axis=0)
                seasonal_errors.append(
                    float(np.mean(np.abs(future - seasonal)))
                )
                last_errors.append(float(np.mean(np.abs(future - last))))
            contrast_summary = summarize_capability_contrasts(contrast_rows)
            seasonal_ratio = float(
                np.mean(seasonal_errors)
                / max(np.mean(last_errors), 1e-9)
            )
            shortcut_passed = bool(
                capability_id not in NONSEASONAL_CAPABILITIES
                or seasonal_ratio > 0.85
            )
            rows.append(
                {
                    "dataset_id": conditioning.dataset_id,
                    "capability_id": capability_id,
                    "profile_id": profile_id,
                    "intensity": intensity,
                    "target_percentile_level": conditioning.target_percentile_levels[
                        intensity - 1
                    ],
                    "target_feature": conditioning.target_feature,
                    "target_strength": conditioning.target_values[intensity - 1],
                    "seasonal_naive_mae_mean": float(
                        np.mean(seasonal_errors)
                    ),
                    "last_value_mae_mean": float(np.mean(last_errors)),
                    "seasonal_naive_to_last_ratio": seasonal_ratio,
                    "fixed_seasonal_shortcut_passed": shortcut_passed,
                    "capability_contrast": contrast_summary,
                }
            )
    high_rows = [
        row for row in rows if int(row["intensity"]) == high_intensity
    ]
    overall_passed = bool(
        high_rows
        and all(row["fixed_seasonal_shortcut_passed"] for row in high_rows)
        and all(row["capability_contrast"]["passed"] for row in high_rows)
    )
    return {
        "schema_version": "synthetic_capability_shortcut_audit.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_artifact_schema_version": artifact["schema_version"],
        "intensity_policy": dict(artifact["intensity_policy"]),
        "supported_cell_count": len(AUDIT_PROFILE_BY_CAPABILITY) - len(unsupported_cells),
        "unsupported_cell_count": len(unsupported_cells),
        "unsupported_cells": unsupported_cells,
        "seed_count": int(seed_count),
        "intensities": list(intensities),
        "qualification_intensity": int(high_intensity),
        "fixed_seasonal_shortcut_threshold": {
            "scope": sorted(NONSEASONAL_CAPABILITIES),
            "minimum_seasonal_naive_to_last_ratio": 0.85,
        },
        "rows": rows,
        "overall_passed": overall_passed,
    }


def _is_supported_capability(capability: Any) -> bool:
    calibration = (
        capability.get("calibration")
        if isinstance(capability, dict)
        else None
    )
    return bool(
        isinstance(capability, dict)
        and capability.get("status", "supported") == "supported"
        and isinstance(calibration, dict)
        and calibration.get("status") == "supported"
    )


def _audit_seed(
    seed: int,
    capability_id: str,
    intensity: int,
    seed_index: int,
) -> int:
    payload = f"{seed}:{capability_id}:{intensity}:{seed_index}".encode(
        "utf-8"
    )
    value = 0
    for byte in payload:
        value = (value * 257 + byte) % (2**32 - 1)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
