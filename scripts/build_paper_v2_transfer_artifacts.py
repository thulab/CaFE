#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for import_path in (BACKEND_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.services.synthetic_generation_service import (  # noqa: E402
    TARGET_FEATURES_BY_CAPABILITY,
    _generate_accepted_sample_values,
    _realized_features,
    _seed_for,
)
from app.services.synthetic_generator_conditioning import (  # noqa: E402
    resolve_generator_conditioning,
)
from app.core.errors import ApiError  # noqa: E402
from build_synthetic_v2_feature_gate_artifact import (  # noqa: E402
    DEFAULT_CALIBRATION_FRACTION,
    DEFAULT_COVERAGE,
    DEFAULT_GATE_REFERENCE_FRACTION,
    calibrate_capability as calibrate_feature_gate_capability,
    split_real_rows,
    split_real_rows_three_way,
)
from build_synthetic_v2_generator_conditioning_artifact import (  # noqa: E402
    CANONICAL_MIN_ADJACENT_GAP_FRACTION,
    PRIMARY_TARGET_FEATURE,
    calibrate_capability_conditioning,
    derive_profile_nuisance,
    empirical_percentiles,
    enforce_target_resolution,
    finite_values,
    quantiles_for_levels,
    reference_percentile_levels,
    summarize_real_features,
)
from paper_v2_transfer_common import (  # noqa: E402
    CANONICAL_REFERENCE_SPECS,
    DEFAULT_GIFT_EVAL_DIR,
    PAPER_UNIVARIATE_CAPABILITY_IDS,
    CanonicalReferenceSpec,
    TransferProfileSpec,
    load_transfer_training_rows,
    primary_feature_intensity_coordinate,
    profile_spec_payload,
    transfer_profile_specs,
)
from run_synthetic_v2_near_distance_calibration import (  # noqa: E402
    DEFAULT_DATA_DIR,
    load_real_bucket,
    online_artifact_bucket,
    thresholds_from_split,
)
from synthetic_capability_qualification import regime_clock_features  # noqa: E402


SCHEMA_VERSION = "paper_v2_transfer_protocol_freeze.v1"
EXPERIMENT_VERSION = "v2"
EXPERIMENT_ID = "00_transfer_protocol_freeze"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/paper_exp" / EXPERIMENT_VERSION / EXPERIMENT_ID
PARENT_CANONICAL_ARTIFACT_PATH = (
    REPO_ROOT / "backend/app/data/synthetic_v2_generator_conditioning_artifact.json"
)
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/2026-07-17-paper-v2-synthetic-real-transfer-protocol.md"
)
DEFAULT_MAX_WINDOWS = 600
DEFAULT_CALIBRATION_SAMPLES = 16
DEFAULT_SEED = 2026071701
DEFAULT_PREFLIGHT_SAMPLES_PER_CELL = 1
CANONICAL_SCALE_ID = "synthetic-v2-paper-v2-504ctx-frozen-2026-07-17"
REGIME_RECURRING_CLOCK_TARGET_INTERVAL = (0.56, 0.94)
NONLINEAR_CONDITIONAL_TARGET_INTERVAL = (0.002, 0.025)
PAPER_V2_PRIMARY_TARGET_FEATURE = {
    **PRIMARY_TARGET_FEATURE,
    "nonlinear_persistence": "nonlinear_conditional_gain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the train-only GIFT transfer profiles and build experiment-local "
            "generator, feature-support, and near-distance artifacts."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--gift-eval-dir", type=Path, default=DEFAULT_GIFT_EVAL_DIR)
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
    )
    parser.add_argument(
        "--preflight-samples-per-cell",
        type=int,
        default=DEFAULT_PREFLIGHT_SAMPLES_PER_CELL,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--stage",
        choices=("all", "build", "preflight"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_windows < 60:
        raise ValueError("max-windows must be at least 60")
    if args.calibration_samples < 4:
        raise ValueError("calibration-samples must be at least 4")
    if args.preflight_samples_per_cell < 1:
        raise ValueError("preflight-samples-per-cell must be positive")
    output_dir = args.output_dir.resolve()
    specs = transfer_profile_specs(args.profiles)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"paper-v2 transfer freeze is already sealed: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"all", "build"}:
        build_and_write_artifacts(
            output_dir,
            specs=specs,
            data_dir=args.data_dir.resolve(),
            gift_eval_dir=args.gift_eval_dir.resolve(),
            max_windows=int(args.max_windows),
            calibration_samples=int(args.calibration_samples),
            seed=int(args.seed),
        )
    if args.stage in {"all", "preflight"}:
        require_artifacts(output_dir)
        preflight = run_preflight(
            output_dir,
            samples_per_cell=int(args.preflight_samples_per_cell),
            seed=int(args.seed),
        )
        write_json(output_dir / "preflight.json", preflight)
        (output_dir / "report.md").write_text(
            render_report(
                read_json(output_dir / "summary.json"),
                preflight,
            ),
            encoding="utf-8",
        )
        write_manifest(output_dir)
    print(f"paper-v2 transfer freeze: {output_dir}", flush=True)
    return 0


def build_v2_canonical_intensity(
    *,
    data_dir: Path,
    gift_eval_dir: Path,
    max_windows: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit one final-shape scale on development families only.

    Raw target curves remain equal-weight medians of real-data quantiles.  The
    regime curve is mapped into the observable support of the recurring-clock
    construction because ordinary one-off change points below that floor do not
    satisfy the capability's predictability contract.
    """

    curves: dict[str, list[list[float]]] = {
        capability_id: [] for capability_id in PAPER_UNIVARIATE_CAPABILITY_IDS
    }
    reference_summaries: list[dict[str, Any]] = []
    for reference_index, spec in enumerate(CANONICAL_REFERENCE_SPECS):
        print(
            f"[canonical {reference_index + 1}/{len(CANONICAL_REFERENCE_SPECS)}] "
            f"profiling {spec.profile_id}",
            flush=True,
        )
        path = canonical_reference_asset_path(
            spec,
            data_dir=data_dir,
            gift_eval_dir=gift_eval_dir,
        )
        # The one-series BizITObs reference needs enough temporal blocks for a
        # leakage-safe three-way split even when a transfer smoke requests only
        # a small held-out window sample.
        reference_max_windows = max(int(max_windows), 300)
        rows = load_real_bucket(
            spec,
            path,
            max_windows=reference_max_windows,
        )
        parameter_rows, scale_holdout_rows, split_summary = split_real_rows(
            rows,
            spec,
            calibration_fraction=DEFAULT_CALIBRATION_FRACTION,
            seed=_seed_for(seed, spec.profile_id, 0),
        )
        measurement_rows = canonical_measurement_rows(parameter_rows, spec)
        capability_curves: dict[str, list[float]] = {}
        for capability_id in PAPER_UNIVARIATE_CAPABILITY_IDS:
            primary_feature = PAPER_V2_PRIMARY_TARGET_FEATURE[capability_id]
            values = finite_values(measurement_rows, primary_feature)
            if not values.size:
                raise ValueError(
                    f"{spec.profile_id}/{capability_id} has no finite "
                    f"{primary_feature}"
                )
            curve = quantiles_for_levels(
                values,
                reference_percentile_levels(capability_id),
            )
            curves[capability_id].append(curve)
            capability_curves[capability_id] = curve
        reference_summaries.append(
            {
                "profile_id": spec.profile_id,
                "family_id": spec.family_id,
                "kind": spec.kind,
                "asset_name": spec.asset_name,
                "source_window_count": len(rows),
                "parameter_window_count": len(parameter_rows),
                "scale_holdout_window_count": len(scale_holdout_rows),
                "split": split_summary,
                "primary_feature_quantile_curves": capability_curves,
            }
        )
        del rows, parameter_rows, scale_holdout_rows, measurement_rows
        gc.collect()

    definitions: dict[str, dict[str, Any]] = {}
    for capability_id in PAPER_UNIVARIATE_CAPABILITY_IDS:
        matrix = np.asarray(curves[capability_id], dtype=float)
        raw_targets = np.maximum.accumulate(np.median(matrix, axis=0))
        resolved_targets = enforce_target_resolution(raw_targets)
        support_projection: dict[str, Any] = {
            "applied": False,
            "reason": None,
        }
        support_interval: tuple[float, float] | None = None
        support_reason: str | None = None
        if capability_id == "regime_switching":
            support_interval = REGIME_RECURRING_CLOCK_TARGET_INTERVAL
            support_reason = (
                "ordinary one-off real change points below the construction "
                "floor are not predictable recurring regimes"
            )
        elif capability_id == "nonlinear_persistence":
            support_interval = NONLINEAR_CONDITIONAL_TARGET_INTERVAL
            support_reason = (
                "the raw real p20-p90 curve lies near the finite-sample "
                "estimator floor; preserve its coordinates inside a modest "
                "construction-supported range so five doses remain separable"
            )
        if support_interval is not None:
            lower, upper = support_interval
            coordinates = (resolved_targets - resolved_targets[0]) / (
                resolved_targets[-1] - resolved_targets[0]
            )
            resolved_targets = lower + coordinates * (upper - lower)
            support_projection = {
                "applied": True,
                "reason": support_reason,
                "observable_interval": [lower, upper],
            }
        definitions[capability_id] = {
            "primary_feature": PAPER_V2_PRIMARY_TARGET_FEATURE[capability_id],
            "target_values": [
                round_float(value) for value in resolved_targets
            ],
            "raw_reference_quantile_values": [
                round_float(value) for value in raw_targets
            ],
            "reference_percentile_levels": list(
                reference_percentile_levels(capability_id)
            ),
            "contributing_profile_ids": [
                spec.profile_id for spec in CANONICAL_REFERENCE_SPECS
            ],
            "contributing_parameter_window_counts": {
                summary["profile_id"]: summary["parameter_window_count"]
                for summary in reference_summaries
            },
            "profile_weighting": "equal",
            "aggregation": (
                "coordinate-wise median of final-shape development-family "
                "quantile curves, minimum-gap projection, then any declared "
                "construction-support projection"
            ),
            "target_resolution": {
                "minimum_adjacent_gap_fraction_of_raw_range": (
                    CANONICAL_MIN_ADJACENT_GAP_FRACTION
                ),
                "applied": bool(
                    not np.allclose(
                        enforce_target_resolution(raw_targets),
                        raw_targets,
                        rtol=0.0,
                        atol=1e-12,
                    )
                ),
            },
            "construction_support_projection": support_projection,
        }

    identities = canonical_reference_input_identities(
        data_dir=data_dir,
        gift_eval_dir=gift_eval_dir,
    )
    fingerprint_payload = {
        "scale_id": CANONICAL_SCALE_ID,
        "reference_profile_ids": [
            spec.profile_id for spec in CANONICAL_REFERENCE_SPECS
        ],
        "asset_identities": identities,
        "capabilities": definitions,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    canonical_intensity = {
        "scale_id": CANONICAL_SCALE_ID,
        "scale_fingerprint": fingerprint,
        "reference_corpus_role": (
            "paper-v2 final-shape development calibration only; all E4 transfer "
            "families and official test windows are excluded"
        ),
        "asset_identities": identities,
        "default_reference_percentile_levels": [0.20, 0.35, 0.50, 0.70, 0.90],
        "policy": (
            "one absolute realized-feature target curve per capability at the "
            "frozen 504/48/24 shape; profiles only fit inverse maps"
        ),
        "reference_preprocessing": {
            "context_length": 504,
            "horizon": 48,
            "season_length": 24,
            "canonical_measurement_length": 528,
            "profile_weighting": "equal",
        },
        "reference_qualification": {
            "regime_switching": (
                "real quantile coordinates are preserved but mapped to the "
                "pre-registered recurring-clock observable interval"
            )
        },
        "held_out": {
            "role": "all profiles registered in TRANSFER_PROFILE_SPECS",
            "test_access": "none",
        },
        "parent_development_scale": {
            "scale_id": "synthetic-v2-paper-v1-frozen-2026-07-16",
            "role": "method-development provenance only; targets are not reused",
        },
        "capabilities": definitions,
    }
    return (
        canonical_intensity,
        {
            "reference_profile_count": len(reference_summaries),
            "profiles": reference_summaries,
            "capabilities": definitions,
        },
    )


def canonical_reference_asset_path(
    spec: CanonicalReferenceSpec,
    *,
    data_dir: Path,
    gift_eval_dir: Path,
) -> Path:
    if spec.kind == "gift_univariate":
        return gift_eval_dir / spec.asset_name
    return data_dir / spec.asset_name


def build_and_write_artifacts(
    output_dir: Path,
    *,
    specs: tuple[TransferProfileSpec, ...],
    data_dir: Path,
    gift_eval_dir: Path,
    max_windows: int,
    calibration_samples: int,
    seed: int,
) -> None:
    canonical_intensity, canonical_reference_summary = build_v2_canonical_intensity(
        data_dir=data_dir,
        gift_eval_dir=gift_eval_dir,
        max_windows=max_windows,
        seed=seed,
    )
    canonical_capabilities = canonical_intensity["capabilities"]
    created_at = datetime.now(timezone.utc).isoformat()
    generator_profiles: dict[str, dict[str, Any]] = {}
    feature_buckets: dict[str, dict[str, Any]] = {}
    near_buckets: dict[str, dict[str, Any]] = {}
    audit_profiles: list[dict[str, Any]] = []
    summary_profiles: list[dict[str, Any]] = []

    for profile_index, spec in enumerate(specs):
        print(
            f"[{profile_index + 1}/{len(specs)}] fitting {spec.profile_id}",
            flush=True,
        )
        rows, source_summary = load_transfer_training_rows(
            spec,
            gift_eval_dir=gift_eval_dir,
            max_windows=max_windows,
        )
        parameter_rows, gate_reference, gate_calibration, split_summary = (
            split_real_rows_three_way(
                rows,
                spec,
                calibration_fraction=DEFAULT_CALIBRATION_FRACTION,
                gate_reference_fraction=DEFAULT_GATE_REFERENCE_FRACTION,
                seed=_seed_for(seed, spec.profile_id, 0),
            )
        )
        measurement_parameter_rows = canonical_measurement_rows(parameter_rows, spec)
        real_feature_summary = summarize_real_features(measurement_parameter_rows)
        profile_nuisance = derive_profile_nuisance(
            real_feature_summary,
            spec.context_length,
            spec.season_length,
        )
        capability_configs: dict[str, dict[str, Any]] = {}
        capability_audit: dict[str, dict[str, Any]] = {}
        unsupported: list[str] = []
        for capability_index, capability_id in enumerate(spec.synthetic_capabilities):
            definition = canonical_capabilities[capability_id]
            target_values = [float(value) for value in definition["target_values"]]
            primary_feature = PAPER_V2_PRIMARY_TARGET_FEATURE[capability_id]
            capability_calibration_samples = (
                calibration_samples * 2
                if capability_id == "nonlinear_persistence"
                else calibration_samples
            )
            primary_values = finite_values(measurement_parameter_rows, primary_feature)
            if not primary_values.size:
                raise ValueError(
                    f"{spec.profile_id}/{capability_id} has no finite {primary_feature}"
                )
            local_quantiles = {
                name: quantiles_for_levels(
                    finite_values(measurement_parameter_rows, name),
                    reference_percentile_levels(capability_id),
                )
                for name in TARGET_FEATURES_BY_CAPABILITY[capability_id]
                if finite_values(measurement_parameter_rows, name).size
            }
            parameters, intensity_lambdas, calibration = (
                calibrate_capability_conditioning(
                    spec=spec,
                    capability_id=capability_id,
                    profile_nuisance=profile_nuisance,
                    real_feature_summary=real_feature_summary,
                    canonical_target_values=target_values,
                    sample_count=capability_calibration_samples,
                    seed=_seed_for(
                        seed,
                        spec.profile_id,
                        100 + capability_index,
                    ),
                    primary_feature=primary_feature,
                )
            )
            if calibration["status"] != "supported":
                unsupported.append(
                    f"{capability_id}(max_error={calibration['max_normalized_error']})"
                )
            capability_configs[capability_id] = {
                "parameters": parameters,
                "intensity_lambdas": intensity_lambdas,
                "canonical_reference_percentile_levels": definition[
                    "reference_percentile_levels"
                ],
                "canonical_target_feature": primary_feature,
                "canonical_target_values": target_values,
                "canonical_raw_reference_quantile_values": definition[
                    "raw_reference_quantile_values"
                ],
                "calibrated_realized_strengths": calibration["realized_values"],
                "local_real_percentiles_at_canonical_targets": empirical_percentiles(
                    primary_values,
                    target_values,
                ),
                "local_real_target_quantiles": local_quantiles,
                "canonical_calibration": {
                    key: value
                    for key, value in calibration.items()
                    if key
                    in {
                        "status",
                        "normalized_absolute_errors",
                        "max_normalized_error",
                        "tolerance",
                        "target_scale",
                        "fit_sample_count",
                        "fit_seed_bank_count",
                        "fit_samples_per_seed_bank",
                        "validation_sample_count",
                        "validation_seed_is_independent",
                    }
                },
                "calibration_method": (
                    "frozen canonical target with train-only profile inverse calibration"
                ),
            }
            capability_audit[capability_id] = audit_capability(
                capability_id,
                measurement_parameter_rows,
                primary_feature=primary_feature,
                canonical_target_values=target_values,
                context_length=spec.context_length,
                season_length=spec.season_length,
            )
        if unsupported:
            raise ValueError(
                f"{spec.profile_id} has unsupported canonical calibrations: "
                + ", ".join(unsupported)
            )

        feature_capabilities = {
            capability_id: calibrate_feature_gate_capability(
                capability_id,
                gate_reference,
                gate_calibration,
                coverage=DEFAULT_COVERAGE,
            )
            for capability_id in spec.synthetic_capabilities
        }
        thresholds, near_diagnostics = thresholds_from_split(
            gate_reference,
            gate_calibration,
        )
        near_bucket = online_artifact_bucket(
            spec,
            gate_reference[: min(192, len(gate_reference))],
            thresholds=thresholds,
            split_summary=split_summary,
        )
        generator_profiles[spec.profile_id] = {
            "profile_id": spec.profile_id,
            "conditioning_role": "paper_v2_held_out_train_only",
            "dataset_name": spec.dataset_name,
            "family_id": spec.family_id,
            "context_length": int(spec.context_length),
            "horizon": int(spec.horizon),
            "target_dim": int(spec.target_dim),
            "covariate_dim": int(spec.covariate_dim),
            "season_length": int(spec.season_length),
            "feature_measurement_horizon": int(spec.feature_measurement_horizon),
            "frequency": spec.frequency,
            "selection_weight": 1.0,
            "nuisance_parameters": profile_nuisance,
            "real_parameter_feature_summary": real_feature_summary,
            "split": split_summary,
            "capabilities": capability_configs,
        }
        feature_buckets[spec.profile_id] = {
            "profile_id": spec.profile_id,
            "context_length": int(spec.context_length),
            "horizon": int(spec.horizon),
            "season_length": int(spec.season_length),
            "target_dim": int(spec.target_dim),
            "covariate_dim": int(spec.covariate_dim),
            "split": split_summary,
            "capabilities": feature_capabilities,
        }
        near_buckets[spec.profile_id] = near_bucket
        audit_profiles.append(
            {
                **profile_spec_payload(spec),
                "source": source_summary,
                "split": split_summary,
                "capabilities": capability_audit,
            }
        )
        summary_profiles.append(
            {
                "profile_id": spec.profile_id,
                "dataset_name": spec.dataset_name,
                "family_id": spec.family_id,
                "source_window_count": len(rows),
                "parameter_window_count": len(parameter_rows),
                "gate_reference_count": len(gate_reference),
                "gate_calibration_count": len(gate_calibration),
                "near_distance_feature_names": list(near_diagnostics["feature_names"]),
                "capability_calibration_max_error": {
                    capability_id: capability_configs[capability_id][
                        "canonical_calibration"
                    ]["max_normalized_error"]
                    for capability_id in spec.synthetic_capabilities
                },
            }
        )
        del (
            rows,
            parameter_rows,
            measurement_parameter_rows,
            gate_reference,
            gate_calibration,
        )
        gc.collect()

    input_identities = transfer_input_identities(
        data_dir=data_dir,
        gift_eval_dir=gift_eval_dir,
    )
    profile_ids = [spec.profile_id for spec in specs]
    config = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at": created_at,
        "profile_ids": profile_ids,
        "capability_ids": list(PAPER_UNIVARIATE_CAPABILITY_IDS),
        "max_windows_per_profile": int(max_windows),
        "calibration_samples_per_grid_cell": int(calibration_samples),
        "nonlinear_calibration_samples_per_grid_cell": int(
            calibration_samples * 2
        ),
        "split_policy": (
            "official GIFT training prefix only, followed by leakage-safe three-way "
            "parameter/reference/calibration split"
        ),
        "test_access_policy": (
            "official validation horizon and all official test windows are excluded "
            "from every transfer artifact"
        ),
        "canonical_scale_id": canonical_intensity["scale_id"],
        "canonical_scale_fingerprint": canonical_intensity["scale_fingerprint"],
        "canonical_scale_policy": (
            "refrozen once on five paper-v2 development families at the final "
            "504/48/24 shape; held-out transfer profiles cannot refit targets"
        ),
        "input_identities": input_identities,
        "seed": int(seed),
    }
    generator_artifact = {
        "schema_version": "synthetic_v2_generator_conditioning_artifact.v2",
        "created_at": created_at,
        "config": {
            **config,
            "conditioning_profile_ids": profile_ids,
            "online_conditioning_profile_ids": profile_ids,
        },
        "canonical_intensity": canonical_intensity,
        "profiles": generator_profiles,
    }
    feature_artifact = {
        "schema_version": "synthetic_v2_feature_gate_online.v1",
        "created_at": created_at,
        "config": {
            **config,
            "coverage": DEFAULT_COVERAGE,
            "support_method": (
                "median_iqr_standardization + shrunk_robust_mahalanobis"
            ),
        },
        "buckets": feature_buckets,
    }
    near_artifact = {
        "schema_version": "synthetic_v2_near_distance_online.v2",
        "created_at": created_at,
        "source_summary_schema_version": SCHEMA_VERSION,
        "config": {
            **config,
            "artifact_reference_count": 192,
            "strict_rule": (
                "full-window OR context-only raw MAE/L2 DCR <= real calibration p01"
            ),
            "combined_rule": (
                "full-window combined rule OR context raw MAE/L2 <= p05 and "
                "context NNDR <= p01"
            ),
        },
        "buckets": near_buckets,
    }
    audit = {
        "schema_version": "paper_v2_train_only_capability_audit.v1",
        "created_at": created_at,
        "config": config,
        "profiles": audit_profiles,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "config": config,
        "profile_count": len(specs),
        "profile_capability_count": len(specs)
        * len(PAPER_UNIVARIATE_CAPABILITY_IDS),
        "canonical_reference_summary": canonical_reference_summary,
        "profiles": summary_profiles,
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "generator_conditioning_artifact.json", generator_artifact)
    write_json(output_dir / "feature_gate_artifact.json", feature_artifact)
    write_json(output_dir / "near_distance_artifact.json", near_artifact)
    write_json(output_dir / "capability_audit.json", audit)
    write_json(output_dir / "summary.json", summary)


def canonical_measurement_rows(
    rows: list[dict[str, Any]],
    spec: TransferProfileSpec | CanonicalReferenceSpec,
) -> list[dict[str, Any]]:
    measurement_end = int(spec.context_length + spec.feature_measurement_horizon)
    measured: list[dict[str, Any]] = []
    for row in rows:
        target = np.asarray(row["target"], dtype=float)[:measurement_end]
        covariates = row.get("covariates")
        covariate_prefix = (
            np.asarray(covariates, dtype=float)[:measurement_end]
            if covariates is not None
            else None
        )
        measured.append(
            {
                **row,
                "features": _realized_features(
                    target,
                    covariate_prefix,
                    int(spec.season_length),
                    int(spec.context_length),
                ),
            }
        )
    return measured


def audit_capability(
    capability_id: str,
    rows: list[dict[str, Any]],
    *,
    primary_feature: str,
    canonical_target_values: list[float],
    context_length: int,
    season_length: int,
) -> dict[str, Any]:
    values = finite_values(rows, primary_feature)
    quantiles = np.quantile(values, (0.25, 0.50, 0.75))
    result: dict[str, Any] = {
        "primary_feature": primary_feature,
        "primary_feature_q25": round_float(quantiles[0]),
        "primary_feature_q50": round_float(quantiles[1]),
        "primary_feature_q75": round_float(quantiles[2]),
        "median_canonical_intensity_coordinate": round_float(
            primary_feature_intensity_coordinate(
                float(quantiles[1]),
                canonical_target_values,
            )
        ),
        "canonical_target_values": list(canonical_target_values),
        "window_count": len(rows),
    }
    if capability_id == "regime_switching":
        sampled = sample_evenly(rows, min(120, len(rows)))
        audits = [
            regime_clock_features(
                np.asarray(row["target"], dtype=float),
                context_length=context_length,
                season_length=season_length,
            )
            for row in sampled
        ]
        qualified = [audit for audit in audits if audit["qualified"]]
        result["predictability_qualification"] = {
            "method": "history-selected recurring clock with untouched pseudo-future",
            "audited_window_count": len(audits),
            "qualified_window_count": len(qualified),
            "qualified_rate": round_float(len(qualified) / max(len(audits), 1)),
        }
    else:
        result["predictability_qualification"] = {
            "method": "capability-specific train-only headroom audit deferred to E4 selection",
            "status": "not_used_to_prune_E2_v2",
        }
    return result


def run_preflight(
    output_dir: Path,
    *,
    samples_per_cell: int,
    seed: int,
) -> dict[str, Any]:
    generator_artifact = read_json(output_dir / "generator_conditioning_artifact.json")
    feature_artifact = read_json(output_dir / "feature_gate_artifact.json")
    near_artifact = read_json(output_dir / "near_distance_artifact.json")
    rows: list[dict[str, Any]] = []
    for profile_index, profile_id in enumerate(
        generator_artifact["config"]["online_conditioning_profile_ids"]
    ):
        profile = generator_artifact["profiles"][profile_id]
        for capability_index, capability_id in enumerate(sorted(profile["capabilities"])):
            conditioning = resolve_generator_conditioning(
                capability_id=capability_id,
                profile_id=profile_id,
                context_length=int(profile["context_length"]),
                horizon=int(profile["horizon"]),
                target_dim=int(profile["target_dim"]),
                artifact=generator_artifact,
            )
            if conditioning is None:
                raise RuntimeError(f"missing transfer conditioning: {profile_id}/{capability_id}")
            for sample_index in range(samples_per_cell):
                sample_seed = _seed_for(
                    seed,
                    f"{profile_id}:{capability_id}",
                    sample_index,
                )
                try:
                    _target, latent, _covariates, features = (
                        _generate_accepted_sample_values(
                            capability_id,
                            int(profile["context_length"])
                            + int(profile["horizon"]),
                            int(profile["context_length"]),
                            int(profile["target_dim"]),
                            int(profile["season_length"]),
                            3,
                            sample_seed,
                            anchor_profile_id=profile_id,
                            generator_conditioning=conditioning,
                            generator_conditioning_artifact=generator_artifact,
                            feature_gate_artifact=feature_artifact,
                            near_distance_artifact=near_artifact,
                            acceptance_profile_ids=(profile_id,),
                        )
                    )
                except ApiError as exc:
                    raise RuntimeError(
                        "paper-v2 preflight failed: "
                        + json.dumps(
                            {
                                "profile_id": profile_id,
                                "capability_id": capability_id,
                                "sample_index": sample_index,
                                "sample_seed": sample_seed,
                                "error_code": exc.error_code,
                                "details": exc.details,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ) from exc
                validation = latent["acceptance"]["validation"]
                rows.append(
                    {
                        "profile_id": profile_id,
                        "capability_id": capability_id,
                        "sample_index": sample_index,
                        "accepted": bool(latent["acceptance"]["accepted"]),
                        "attempts": int(latent["acceptance"]["attempts"]),
                        "feature_gate_status": validation["feature_gate"]["status"],
                        "near_distance_status": validation["near_distance_gate"]["status"],
                        "primary_feature": conditioning.canonical_target_feature,
                        "primary_feature_value": float(
                            features[conditioning.canonical_target_feature]
                        ),
                    }
                )
        print(
            f"preflight {profile_index + 1}/"
            f"{len(generator_artifact['config']['online_conditioning_profile_ids'])}: "
            f"{profile_id}",
            flush=True,
        )
    accepted = sum(bool(row["accepted"]) for row in rows)
    return {
        "schema_version": "paper_v2_transfer_preflight.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "accepted_count": accepted,
        "acceptance_rate": accepted / max(len(rows), 1),
        "max_attempts": max(int(row["attempts"]) for row in rows),
        "all_passed": accepted == len(rows),
        "rows": rows,
    }


def canonical_reference_input_identities(
    *,
    data_dir: Path,
    gift_eval_dir: Path,
) -> dict[str, str]:
    from build_synthetic_v2_generator_conditioning_artifact import (
        gift_eval_arrow_manifest_sha256,
        git_head,
    )

    return {
        "m4_hourly_sha256": sha256_file(data_dir / "m4_hourly_dataset.zip"),
        "electricity_hourly_sha256": sha256_file(
            data_dir / "electricity_hourly_dataset.zip"
        ),
        "traffic_hourly_sha256": sha256_file(
            data_dir / "traffic_hourly_dataset.zip"
        ),
        "gift_eval_arrow_manifest_sha256": gift_eval_arrow_manifest_sha256(
            gift_eval_dir
        ),
        "gift_eval_protocol_git_commit": git_head(Path.home() / "xmy/gift-eval-code"),
    }


def transfer_input_identities(
    *,
    data_dir: Path,
    gift_eval_dir: Path,
) -> dict[str, str]:
    return {
        **canonical_reference_input_identities(
            data_dir=data_dir,
            gift_eval_dir=gift_eval_dir,
        ),
        "parent_canonical_generator_artifact_sha256": sha256_file(
            PARENT_CANONICAL_ARTIFACT_PATH
        ),
    }


def render_report(summary: dict[str, Any], preflight: dict[str, Any]) -> str:
    config = summary["config"]
    return "\n".join(
        [
            "# Paper v2：合成—真实迁移协议冻结",
            "",
            f"- Profiles：{summary['profile_count']}",
            f"- Profile × capability cells：{summary['profile_capability_count']}",
            (
                f"- Canonical scale：`{config['canonical_scale_id']}` / "
                f"`{config['canonical_scale_fingerprint']}`"
            ),
            (
                f"- Preflight：{preflight['accepted_count']} / "
                f"{preflight['sample_count']} accepted，"
                f"max attempts={preflight['max_attempts']}"
            ),
            "- 所有 conditioning 与 gates 只读取 GIFT 官方 training prefix。",
            "- 官方 validation horizon 与全部 test windows 未进入任何 artifact。",
            "- 本目录封存后由 E2-v2 只读消费。",
            "",
        ]
    )


def write_manifest(output_dir: Path) -> None:
    required = (
        "config.json",
        "summary.json",
        "capability_audit.json",
        "generator_conditioning_artifact.json",
        "feature_gate_artifact.json",
        "near_distance_artifact.json",
        "preflight.json",
        "report.md",
    )
    files: dict[str, dict[str, Any]] = {}
    for name in required:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing transfer freeze output: {path}")
        files[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    files[relative_path(PROTOCOL_PATH)] = {
        "size_bytes": PROTOCOL_PATH.stat().st_size,
        "sha256": sha256_file(PROTOCOL_PATH),
    }
    runner = Path(__file__).resolve()
    files[relative_path(runner)] = {
        "size_bytes": runner.stat().st_size,
        "sha256": sha256_file(runner),
    }
    common = SCRIPT_DIR / "paper_v2_transfer_common.py"
    files[relative_path(common)] = {
        "size_bytes": common.stat().st_size,
        "sha256": sha256_file(common),
    }
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "paper_v2_transfer_freeze_manifest.v1",
            "experiment_version": EXPERIMENT_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_head(REPO_ROOT),
            "files": files,
        },
    )


def require_artifacts(output_dir: Path) -> None:
    for name in (
        "generator_conditioning_artifact.json",
        "feature_gate_artifact.json",
        "near_distance_artifact.json",
        "summary.json",
    ):
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"run --stage build first: {path}")


def sample_evenly(values: list[Any], count: int) -> list[Any]:
    if count >= len(values):
        return list(values)
    indexes = np.linspace(0, len(values) - 1, num=count, dtype=int)
    return [values[int(index)] for index in indexes]


def round_float(value: float, digits: int = 8) -> float:
    return float(round(float(value), digits))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
