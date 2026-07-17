#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for import_path in (BACKEND_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import paper_v2_transfer_common as transfer  # noqa: E402
import run_paper_e2_dynamic_stability as inference  # noqa: E402
from synthetic_feature_profile import (  # noqa: E402
    gift_eval_short_term_test_holdout_steps,
)


SCHEMA_VERSION = "paper_e4_synthetic_real_transfer.v1"
EXPERIMENT_VERSION = "v2"
EXPERIMENT_ID = "E4_synthetic_real_transfer"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/paper_exp/v2/E4_synthetic_real_transfer"
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/"
    "2026-07-17-paper-v2-e4-synthetic-real-transfer-protocol.md"
)
SELECTION_RECEIPT_PATH = (
    REPO_ROOT
    / "docs/superpowers/baselines/"
    "2026-07-17-paper-v2-e4-selection-freeze.json"
)
FREEZE_DIR = REPO_ROOT / "runtime/paper_exp/v2/00_transfer_protocol_freeze"
CAPABILITY_AUDIT_PATH = FREEZE_DIR / "capability_audit.json"
TRANSFER_FREEZE_MANIFEST_PATH = FREEZE_DIR / "manifest.json"
E3_V2_DIR = REPO_ROOT / "runtime/paper_exp/v2/E3_model_capability_profiles"
E3_V1_DIR = REPO_ROOT / "runtime/paper_exp/v1/E3_model_capability_profiles"
E3_V2_SCORES_PATH = E3_V2_DIR / "profile_intensity_scores.csv"
E3_V1_SCORES_PATH = E3_V1_DIR / "profile_intensity_scores.csv"
E3_V2_CONTRASTS_PATH = E3_V2_DIR / "model_capability_contrasts.csv"

MODELS = inference.DEFAULT_MODELS
BASELINES = inference.BASELINE_MODELS
CAPABILITIES = transfer.PAPER_UNIVARIATE_CAPABILITY_IDS
CONTEXT_LENGTH = transfer.PAPER_V2_CONTEXT_LENGTH
HORIZON = transfer.PAPER_V2_HORIZON
SEASON_LENGTH = transfer.PAPER_V2_SEASON_LENGTH
MAX_TASKS_PER_PROFILE = 600
MIN_CONTEXT_OBSERVED_FRACTION = 0.50
MIN_FUTURE_OBSERVED_COUNT = 24
MASE_ABSOLUTE_FLOOR = 1e-8
MASE_RELATIVE_FLOOR = 1e-6
HIGH_LOADING_THRESHOLD = 3.0
MIN_HYPOTHESIS_FAMILIES = 3
PAIR_HYPOTHESES_PER_CAPABILITY = 2
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2026071741
TIME_COLUMN = inference.TIME_COLUMN

PREDICTOR_IDS = (
    "v2_dataset_local_capability",
    "v2_global_capability",
    "v1_development_global_capability",
    "v2_scalar_macro",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-v2 E4 synthetic-to-real capability transfer on the "
            "pre-registered controlled GIFT-Eval slice."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "infer", "analyze"),
        required=True,
        help=(
            "prepare is intentionally separate: commit the generated selection "
            "receipt before running infer"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=transfer.DEFAULT_GIFT_EVAL_DIR,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:10810")
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--devices", default=inference.DEFAULT_DEVICES)
    parser.add_argument(
        "--request-max-attempts",
        type=int,
        default=inference.DEFAULT_REQUEST_MAX_ATTEMPTS,
    )
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--model-load-timeout-seconds", type=int, default=1800)
    parser.add_argument("--keep-loaded", action="store_true")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.stage == "prepare":
        prepare_experiment(
            output_dir,
            gift_eval_dir=args.gift_eval_dir.resolve(),
            base_url=args.base_url,
            api_prefix=args.api_prefix,
            devices=args.devices,
            request_max_attempts=args.request_max_attempts,
            forecast_timeout_seconds=args.forecast_timeout_seconds,
            model_load_timeout_seconds=args.model_load_timeout_seconds,
        )
    elif args.stage == "infer":
        run_inference_stage(output_dir, args=args)
    else:
        analyze_experiment(
            output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    print(f"E4 output: {output_dir}", flush=True)
    return 0


def prepare_experiment(
    output_dir: Path,
    *,
    gift_eval_dir: Path,
    base_url: str,
    api_prefix: str,
    devices: str,
    request_max_attempts: int,
    forecast_timeout_seconds: int,
    model_load_timeout_seconds: int,
) -> None:
    require_file(PROTOCOL_PATH)
    require_file(CAPABILITY_AUDIT_PATH)
    require_file(TRANSFER_FREEZE_MANIFEST_PATH)
    require_file(E3_V2_DIR / "manifest.json")
    require_file(E3_V1_DIR / "manifest.json")
    require_file(E3_V2_SCORES_PATH)
    require_file(E3_V1_SCORES_PATH)
    require_file(E3_V2_CONTRASTS_PATH)
    reject_existing_model_outputs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "failures").mkdir(exist_ok=True)

    verify_source_manifest(E3_V2_DIR, ("profile_intensity_scores.csv",))
    verify_source_manifest(E3_V1_DIR, ("profile_intensity_scores.csv",))
    verify_source_manifest(FREEZE_DIR, ("capability_audit.json",))

    audit = read_json(CAPABILITY_AUDIT_PATH)
    coordinates = capability_coordinate_frame(audit)
    qualified = qualified_cell_frame(coordinates)
    predictors = synthetic_predictor_frame()
    hypotheses = pair_hypothesis_frame(qualified)

    write_csv(output_dir / "capability_coordinates.csv", coordinates)
    write_csv(output_dir / "qualified_cells.csv", qualified)
    write_csv(output_dir / "synthetic_predictors.csv", predictors)
    write_csv(output_dir / "pair_hypotheses.csv", hypotheses)

    tasks, profile_summaries = build_real_tasks(
        gift_eval_dir=gift_eval_dir,
        max_tasks_per_profile=MAX_TASKS_PER_PROFILE,
    )
    task_path = output_dir / "tasks.jsonl"
    write_jsonl(task_path, tasks)
    preflight = validate_real_tasks(tasks, profile_summaries=profile_summaries)
    write_json(output_dir / "preflight.json", preflight)

    task_manifest = {
        "schema_version": "paper_e4_real_task_manifest.v1",
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "task_file": relative_path(task_path),
        "task_file_sha256": sha256_file(task_path),
        "task_id_sequence_sha256": sha256_lines(
            str(task["sample_id"]) for task in tasks
        ),
        "task_count": len(tasks),
        "profile_count": len(profile_summaries),
        "profile_summaries": profile_summaries,
        "request_shape": {
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "season_length": SEASON_LENGTH,
            "target_dim": 1,
            "covariate_dim": 0,
        },
        "selection_policy": task_selection_policy(),
    }
    write_json(output_dir / "task_manifest.json", task_manifest)

    selection_manifest = selection_manifest_payload(
        output_dir=output_dir,
        gift_eval_dir=gift_eval_dir,
        task_manifest=task_manifest,
        coordinates=coordinates,
        qualified=qualified,
        hypotheses=hypotheses,
    )
    write_json(output_dir / "selection_manifest.json", selection_manifest)

    config = {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "selection_manifest_sha256": sha256_file(
            output_dir / "selection_manifest.json"
        ),
        "task_manifest_sha256": sha256_file(output_dir / "task_manifest.json"),
        "expected_task_count": len(tasks),
        "requested_models": list(MODELS),
        "baselines": list(BASELINES),
        "model_execution": {
            model_id: dict(inference.MODEL_EXECUTION_CONFIG[model_id])
            for model_id in MODELS
        },
        "devices": str(devices),
        "request_max_attempts": int(request_max_attempts),
        "forecast_timeout_seconds": int(forecast_timeout_seconds),
        "model_load_timeout_seconds": int(model_load_timeout_seconds),
        "service": {
            "base_url": str(base_url).rstrip("/"),
            "api_prefix": "/" + str(api_prefix).strip("/"),
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "held-out family",
            "confidence_level": 0.95,
        },
    }
    write_json(output_dir / "config.json", config)

    receipt = selection_receipt_payload(output_dir)
    write_json(output_dir / "selection_receipt_candidate.json", receipt)
    print(
        f"prepared {len(tasks)} tasks across {len(profile_summaries)} profiles; "
        f"qualified cells={len(qualified)}, pair hypotheses={len(hypotheses)}",
        flush=True,
    )
    print(
        "No inference was requested. Commit the candidate receipt at "
        f"{relative_path(SELECTION_RECEIPT_PATH)} before --stage infer.",
        flush=True,
    )


def task_selection_policy() -> dict[str, Any]:
    return {
        "source_origins": (
            "all official GIFT short-term rolling origins before deterministic cap"
        ),
        "max_tasks_per_profile": MAX_TASKS_PER_PROFILE,
        "stratification": "equal allocation over official rolling-origin index",
        "within_origin": (
            "deterministic evenly spaced selection in Arrow row/channel order"
        ),
        "context": {
            "length": CONTEXT_LENGTH,
            "minimum_observed_fraction": MIN_CONTEXT_OBSERVED_FRACTION,
            "minimum_finite_points": 2,
            "imputation": (
                "linear interpolation inside context and nearest-fill at edges"
            ),
        },
        "future": {
            "length": HORIZON,
            "minimum_observed_count": MIN_FUTURE_OBSERVED_COUNT,
            "imputation": "none; missing labels are metric-masked",
        },
        "mase": {
            "period": SEASON_LENGTH,
            "minimum_scale": (
                "max(1e-8, 1e-6 * mean(abs(imputed_context)))"
            ),
        },
        "timestamps": "Arrow start plus native frequency; no synthetic date",
    }


def capability_coordinate_frame(audit: dict[str, Any]) -> pd.DataFrame:
    profiles = {spec.profile_id: spec for spec in transfer.TRANSFER_PROFILE_SPECS}
    rows: list[dict[str, Any]] = []
    for profile in audit.get("profiles", []):
        profile_id = str(profile["profile_id"])
        if profile_id not in profiles:
            raise ValueError(f"unknown profile in capability audit: {profile_id}")
        spec = profiles[profile_id]
        capability_payload = profile.get("capabilities") or {}
        if set(capability_payload) != set(CAPABILITIES):
            raise ValueError(
                f"{profile_id} capability audit mismatch: "
                f"{sorted(capability_payload)}"
            )
        for capability_id in CAPABILITIES:
            payload = capability_payload[capability_id]
            rows.append(
                {
                    "profile_id": profile_id,
                    "dataset_name": spec.dataset_name,
                    "family_id": spec.family_id,
                    "capability_id": capability_id,
                    "primary_feature": str(payload["primary_feature"]),
                    "primary_feature_q25": float(payload["primary_feature_q25"]),
                    "primary_feature_q50": float(payload["primary_feature_q50"]),
                    "primary_feature_q75": float(payload["primary_feature_q75"]),
                    "canonical_intensity_coordinate": float(
                        payload["median_canonical_intensity_coordinate"]
                    ),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    if len(frame) != len(profiles) * len(CAPABILITIES):
        raise ValueError("capability audit does not cover the frozen E4 grid")
    return frame.sort_values(
        ["profile_id", "capability_id"], kind="stable"
    ).reset_index(drop=True)


def qualified_cell_frame(coordinates: pd.DataFrame) -> pd.DataFrame:
    result = coordinates[
        coordinates["canonical_intensity_coordinate"] >= HIGH_LOADING_THRESHOLD
    ].copy()
    result["qualification_threshold"] = HIGH_LOADING_THRESHOLD
    result["qualification"] = "train_only_loading_at_or_above_midpoint"
    result = result.sort_values(
        ["family_id", "profile_id", "capability_id"], kind="stable"
    ).reset_index(drop=True)
    if "nonlinear_persistence" in set(result["capability_id"]):
        raise ValueError(
            "nonlinear_persistence unexpectedly has real support in the frozen audit"
        )
    return result


def synthetic_predictor_frame() -> pd.DataFrame:
    v2 = read_synthetic_score_source(E3_V2_SCORES_PATH, source_version="v2")
    v1 = read_synthetic_score_source(E3_V1_SCORES_PATH, source_version="v1")
    expected_models = set(MODELS)
    for label, frame in (("E3-v2", v2), ("E3-v1", v1)):
        if set(frame["model_id"]) != expected_models:
            raise ValueError(f"{label} model set does not match E4")
        missing = set(CAPABILITIES) - set(frame["capability_id"])
        if missing:
            raise ValueError(f"{label} is missing capabilities: {sorted(missing)}")

    specs = transfer.TRANSFER_PROFILE_SPECS
    profile_ids = [spec.profile_id for spec in specs]
    v2 = v2[v2["capability_id"].isin(CAPABILITIES)].copy()
    v1 = v1[v1["capability_id"].isin(CAPABILITIES)].copy()
    local = (
        v2.groupby(["model_id", "profile_id", "capability_id"], sort=True)[
            "log_mase_ratio"
        ]
        .mean()
        .rename("synthetic_score")
        .reset_index()
    )
    expected_local = len(MODELS) * len(profile_ids) * len(CAPABILITIES)
    if len(local) != expected_local:
        raise ValueError(
            f"E3-v2 local predictor grid is incomplete: {len(local)}/{expected_local}"
        )

    v2_global = (
        local.groupby(["model_id", "capability_id"], sort=True)["synthetic_score"]
        .mean()
        .reset_index()
    )
    v1_profile = (
        v1.groupby(["model_id", "profile_id", "capability_id"], sort=True)[
            "log_mase_ratio"
        ]
        .mean()
        .reset_index()
    )
    v1_global = (
        v1_profile.groupby(["model_id", "capability_id"], sort=True)[
            "log_mase_ratio"
        ]
        .mean()
        .rename("synthetic_score")
        .reset_index()
    )
    scalar = (
        v2_global.groupby("model_id", sort=True)["synthetic_score"]
        .mean()
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    local_lookup = {
        (str(row.model_id), str(row.profile_id), str(row.capability_id)): float(
            row.synthetic_score
        )
        for row in local.itertuples(index=False)
    }
    v2_global_lookup = {
        (str(row.model_id), str(row.capability_id)): float(row.synthetic_score)
        for row in v2_global.itertuples(index=False)
    }
    v1_global_lookup = {
        (str(row.model_id), str(row.capability_id)): float(row.synthetic_score)
        for row in v1_global.itertuples(index=False)
    }
    for profile_id, capability_id, model_id in itertools.product(
        profile_ids, CAPABILITIES, MODELS
    ):
        scores = {
            "v2_dataset_local_capability": local_lookup[
                (model_id, profile_id, capability_id)
            ],
            "v2_global_capability": v2_global_lookup[(model_id, capability_id)],
            "v1_development_global_capability": v1_global_lookup[
                (model_id, capability_id)
            ],
            "v2_scalar_macro": float(scalar[model_id]),
        }
        for predictor_id, score in scores.items():
            rows.append(
                {
                    "predictor_id": predictor_id,
                    "profile_id": profile_id,
                    "capability_id": capability_id,
                    "model_id": model_id,
                    "synthetic_log_mase_ratio": score,
                    "lower_is_better": True,
                }
            )
    result = pd.DataFrame.from_records(rows)
    return result.sort_values(
        ["predictor_id", "profile_id", "capability_id", "model_id"],
        kind="stable",
    ).reset_index(drop=True)


def read_synthetic_score_source(path: Path, *, source_version: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "model_id",
        "profile_id",
        "capability_id",
        "intensity",
        "mase_mean",
        "seasonal_naive_mase_mean",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    if bool((frame["mase_mean"] <= 0).any()) or bool(
        (frame["seasonal_naive_mase_mean"] <= 0).any()
    ):
        raise ValueError(f"{path} contains non-positive MASE")
    frame["log_mase_ratio"] = np.log(
        frame["mase_mean"].to_numpy(dtype=float)
        / frame["seasonal_naive_mase_mean"].to_numpy(dtype=float)
    )
    frame["source_version"] = source_version
    return frame


def pair_hypothesis_frame(qualified: pd.DataFrame) -> pd.DataFrame:
    family_counts = (
        qualified.groupby("capability_id", sort=True)["family_id"]
        .nunique()
        .to_dict()
    )
    contrasts = pd.read_csv(E3_V2_CONTRASTS_PATH)
    required = {
        "capability_id",
        "model_id",
        "reference_best_model",
        "relative_mase_gap_vs_best",
        "paired_mase_gap_vs_best_ci_low",
        "paired_mase_gap_vs_best_ci_high",
    }
    missing = required - set(contrasts.columns)
    if missing:
        raise ValueError(f"E3-v2 contrasts missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for capability_id in CAPABILITIES:
        high_family_count = int(family_counts.get(capability_id, 0))
        if high_family_count < MIN_HYPOTHESIS_FAMILIES:
            continue
        candidates = contrasts[
            (contrasts["capability_id"] == capability_id)
            & (contrasts["model_id"] != contrasts["reference_best_model"])
            & (contrasts["paired_mase_gap_vs_best_ci_low"] > 0)
        ].sort_values(
            ["relative_mase_gap_vs_best", "model_id"],
            ascending=[False, True],
            kind="stable",
        )
        selected = candidates.head(PAIR_HYPOTHESES_PER_CAPABILITY)
        if len(selected) != PAIR_HYPOTHESES_PER_CAPABILITY:
            raise ValueError(
                f"{capability_id} does not have two eligible E3 pair hypotheses"
            )
        for order, row in enumerate(selected.itertuples(index=False), start=1):
            rows.append(
                {
                    "hypothesis_id": f"{capability_id}:top{order}",
                    "capability_id": capability_id,
                    "weaker_model": str(row.model_id),
                    "reference_model": str(row.reference_best_model),
                    "selection_order": order,
                    "high_loading_family_count": high_family_count,
                    "e3_relative_mase_gap": float(row.relative_mase_gap_vs_best),
                    "e3_paired_gap_ci_low": float(
                        row.paired_mase_gap_vs_best_ci_low
                    ),
                    "e3_paired_gap_ci_high": float(
                        row.paired_mase_gap_vs_best_ci_high
                    ),
                    "real_direction_hypothesis": (
                        "log(MASE_weaker/MASE_reference) > 0 on high-loading "
                        "family macro"
                    ),
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(
        ["capability_id", "selection_order"], kind="stable"
    ).reset_index(drop=True)


def build_real_tasks(
    *,
    gift_eval_dir: Path,
    max_tasks_per_profile: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in transfer.TRANSFER_PROFILE_SPECS:
        profile_tasks, summary = build_profile_tasks(
            spec,
            gift_eval_dir=gift_eval_dir,
            max_tasks=max_tasks_per_profile,
        )
        tasks.extend(profile_tasks)
        summaries.append(summary)
        print(
            f"{spec.profile_id}: selected {len(profile_tasks)}/"
            f"{summary['eligible_candidate_count']} eligible tasks "
            f"over {summary['official_test_window_count']} origins",
            flush=True,
        )
    if len({task["sample_id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate E4 sample_id across profiles")
    return tasks, summaries


def build_profile_tasks(
    spec: transfer.TransferProfileSpec,
    *,
    gift_eval_dir: Path,
    max_tasks: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_path = gift_eval_dir / spec.dataset_name
    frequency, records = read_gift_records(dataset_path)
    holdout_steps = gift_eval_short_term_test_holdout_steps(
        frequency,
        [
            (str(record["item_id"]), np.asarray(record["target"], dtype=float))
            for record in records
        ],
    )
    if holdout_steps % HORIZON:
        raise ValueError(f"{spec.profile_id} has non-integral official windows")
    official_windows = holdout_steps // HORIZON
    strata: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(official_windows)
    }
    rejection_counts: defaultdict[str, int] = defaultdict(int)
    raw_candidate_count = 0
    expanded_series_count = 0

    for row_index, record in enumerate(records):
        native = np.asarray(record["target"], dtype=float)
        channels = native if native.ndim == 2 else native[None, :]
        for channel_index, values in enumerate(channels):
            expanded_series_count += 1
            series_id = (
                str(record["item_id"])
                if native.ndim == 1
                else f"{record['item_id']}:dim:{channel_index}"
            )
            for origin_index in range(official_windows):
                origin = len(values) - holdout_steps + origin_index * HORIZON
                if origin < CONTEXT_LENGTH or origin + HORIZON > len(values):
                    rejection_counts["insufficient_length"] += 1
                    continue
                raw_candidate_count += 1
                raw_context = np.asarray(
                    values[origin - CONTEXT_LENGTH : origin], dtype=float
                )
                context, observed_fraction = transfer.impute_observed_window(
                    raw_context,
                    minimum_observed_fraction=MIN_CONTEXT_OBSERVED_FRACTION,
                )
                if context is None:
                    rejection_counts["context_missing"] += 1
                    continue
                future = np.asarray(values[origin : origin + HORIZON], dtype=float)
                future_observed_count = int(np.isfinite(future).sum())
                if future_observed_count < MIN_FUTURE_OBSERVED_COUNT:
                    rejection_counts["future_missing"] += 1
                    continue
                mase_scale = seasonal_mase_scale(context, SEASON_LENGTH)
                minimum_scale = max(
                    MASE_ABSOLUTE_FLOOR,
                    MASE_RELATIVE_FLOOR * float(np.mean(np.abs(context))),
                )
                if not np.isfinite(mase_scale) or mase_scale <= minimum_scale:
                    rejection_counts["unstable_mase_scale"] += 1
                    continue
                timestamps = real_history_timestamps(
                    record["start"],
                    frequency=frequency,
                    start_index=origin - CONTEXT_LENGTH,
                    periods=CONTEXT_LENGTH,
                )
                candidate = {
                    "schema_version": "paper_e4_real_task.v1",
                    "sample_id": real_sample_id(
                        spec.profile_id,
                        series_id,
                        origin_index,
                        origin,
                    ),
                    "profile_id": spec.profile_id,
                    "dataset_name": spec.dataset_name,
                    "family_id": spec.family_id,
                    "series_id": series_id,
                    "base_item_id": str(record["item_id"]),
                    "native_row_index": row_index,
                    "channel_index": channel_index,
                    "origin_index": origin_index,
                    "source_origin": int(origin),
                    "context_length": CONTEXT_LENGTH,
                    "horizon": HORIZON,
                    "season_length": SEASON_LENGTH,
                    "target_dim": 1,
                    "covariate_dim": 0,
                    "frequency": str(frequency).lower(),
                    "timestamps": timestamps,
                    "target": [
                        [float(value)] for value in context
                    ]
                    + [
                        [float(value) if np.isfinite(value) else None]
                        for value in future
                    ],
                    "context_observed_fraction": float(observed_fraction),
                    "future_observed_count": future_observed_count,
                    "mase_scale": float(mase_scale),
                    # Compatibility fields consumed only by the shared audited
                    # inference engine. E4's prediction writer preserves the
                    # real-task identity instead of interpreting these values.
                    "capability_id": "real_gift_eval",
                    "intensity": 0,
                    "round_index": origin_index,
                    "sample_index": row_index * max(1, native.shape[0] if native.ndim == 2 else 1)
                    + channel_index,
                }
                strata[origin_index].append(candidate)

    allocations = balanced_stratum_allocations(
        {key: len(value) for key, value in strata.items()},
        max_total=max_tasks,
    )
    selected: list[dict[str, Any]] = []
    for origin_index in sorted(strata):
        candidates = strata[origin_index]
        count = allocations[origin_index]
        selected.extend(select_evenly(candidates, count))
    selected.sort(
        key=lambda row: (
            int(row["origin_index"]),
            int(row["native_row_index"]),
            int(row["channel_index"]),
        )
    )
    for profile_index, task in enumerate(selected):
        task["profile_task_index"] = profile_index

    summary = {
        "profile_id": spec.profile_id,
        "dataset_name": spec.dataset_name,
        "family_id": spec.family_id,
        "source_frequency": frequency,
        "native_record_count": len(records),
        "expanded_series_count": expanded_series_count,
        "official_prediction_length": HORIZON,
        "official_test_window_count": official_windows,
        "official_test_tail_steps": holdout_steps,
        "raw_candidate_count": raw_candidate_count,
        "eligible_candidate_count": int(sum(len(value) for value in strata.values())),
        "selected_task_count": len(selected),
        "selected_per_origin": {
            str(key): int(value) for key, value in sorted(allocations.items())
        },
        "eligible_per_origin": {
            str(key): len(value) for key, value in sorted(strata.items())
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "minimum_context_observed_fraction": (
            min(
                (float(task["context_observed_fraction"]) for task in selected),
                default=None,
            )
        ),
        "minimum_future_observed_count": min(
            (int(task["future_observed_count"]) for task in selected),
            default=None,
        ),
    }
    return selected, summary


def read_gift_records(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if not path.is_dir():
        raise FileNotFoundError(f"GIFT-Eval config directory not found: {path}")
    arrow_files = sorted(path.glob("data-*.arrow"))
    if len(arrow_files) != 1:
        raise ValueError(f"expected one data-*.arrow in {path}")
    with pa.memory_map(str(arrow_files[0]), "r") as source:
        table = pa_ipc.open_stream(source).read_all()
    required = {"item_id", "start", "freq", "target"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"{path} missing Arrow columns: {sorted(missing)}")
    frequencies = {str(value) for value in table["freq"].to_pylist()}
    if len(frequencies) != 1:
        raise ValueError(f"{path} has multiple frequencies: {frequencies}")
    rows: list[dict[str, Any]] = []
    for item_id, start, target in zip(
        table["item_id"].to_pylist(),
        table["start"].to_pylist(),
        table["target"].to_pylist(),
        strict=True,
    ):
        values = np.asarray(target, dtype=float)
        if values.ndim not in (1, 2):
            raise ValueError(f"{path}/{item_id} target shape {values.shape}")
        rows.append({"item_id": str(item_id), "start": start, "target": values})
    return next(iter(frequencies)), rows


def balanced_stratum_allocations(
    capacities: dict[int, int],
    *,
    max_total: int,
) -> dict[int, int]:
    if max_total <= 0:
        raise ValueError("max_total must be positive")
    keys = sorted(capacities)
    allocations = {key: 0 for key in keys}
    remaining = min(max_total, sum(max(0, int(capacities[key])) for key in keys))
    while remaining:
        progressed = False
        for key in keys:
            if remaining == 0:
                break
            if allocations[key] < int(capacities[key]):
                allocations[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("balanced allocation could not place remaining tasks")
    return allocations


def select_evenly(values: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 0 or count > len(values):
        raise ValueError(f"invalid deterministic sample count {count}/{len(values)}")
    if count == len(values):
        return list(values)
    if count == 0:
        return []
    indexes = np.linspace(0, len(values) - 1, count).round().astype(int)
    unique = sorted(set(indexes.tolist()))
    if len(unique) != count:
        raise RuntimeError("even selection produced duplicate indexes")
    return [values[index] for index in unique]


def seasonal_mase_scale(context: np.ndarray, period: int) -> float:
    values = np.asarray(context, dtype=float).reshape(-1)
    if len(values) <= period:
        return float("nan")
    return float(np.mean(np.abs(values[period:] - values[:-period])))


def real_history_timestamps(
    start: datetime,
    *,
    frequency: str,
    start_index: int,
    periods: int,
) -> list[str]:
    normalized_frequency = {
        "H": "h",
        "T": "min",
        "S": "s",
        "M": "ME",
        "Q": "QE",
        "A": "YE",
        "Y": "YE",
    }.get(str(frequency), str(frequency))
    offset = pd.tseries.frequencies.to_offset(normalized_frequency)
    first = pd.Timestamp(start) + start_index * offset
    return [
        timestamp.isoformat()
        for timestamp in pd.date_range(first, periods=periods, freq=offset)
    ]


def real_sample_id(
    profile_id: str,
    series_id: str,
    origin_index: int,
    source_origin: int,
) -> str:
    raw = f"{profile_id}|{series_id}|{origin_index}|{source_origin}"
    return "e4-real-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def validate_real_tasks(
    tasks: list[dict[str, Any]],
    *,
    profile_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    profile_ids = {spec.profile_id for spec in transfer.TRANSFER_PROFILE_SPECS}
    if {str(task["profile_id"]) for task in tasks} != profile_ids:
        raise ValueError("real tasks do not cover every frozen E4 profile")
    if len({str(task["sample_id"]) for task in tasks}) != len(tasks):
        raise ValueError("real tasks contain duplicate sample IDs")
    failures: list[str] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for task in tasks:
        profile_id = str(task["profile_id"])
        counts[profile_id] += 1
        if (
            int(task["context_length"]) != CONTEXT_LENGTH
            or int(task["horizon"]) != HORIZON
            or int(task["season_length"]) != SEASON_LENGTH
            or int(task["target_dim"]) != 1
            or int(task["covariate_dim"]) != 0
        ):
            failures.append(f"{task['sample_id']}: shape")
        target = task["target"]
        if len(target) != CONTEXT_LENGTH + HORIZON:
            failures.append(f"{task['sample_id']}: target length")
        if len(task["timestamps"]) != CONTEXT_LENGTH:
            failures.append(f"{task['sample_id']}: timestamp length")
        context = np.asarray(target[:CONTEXT_LENGTH], dtype=float)
        future = np.asarray(
            [
                float(row[0]) if row[0] is not None else np.nan
                for row in target[CONTEXT_LENGTH:]
            ],
            dtype=float,
        )
        if not np.isfinite(context).all():
            failures.append(f"{task['sample_id']}: non-finite context")
        if int(np.isfinite(future).sum()) < MIN_FUTURE_OBSERVED_COUNT:
            failures.append(f"{task['sample_id']}: future coverage")
        observed_scale = seasonal_mase_scale(context, SEASON_LENGTH)
        if not math.isclose(
            observed_scale,
            float(task["mase_scale"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            failures.append(f"{task['sample_id']}: MASE scale")
    if failures:
        raise ValueError("E4 task preflight failed: " + "; ".join(failures[:10]))
    expected_counts = {
        str(row["profile_id"]): int(row["selected_task_count"])
        for row in profile_summaries
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"task counts do not match profile summaries: {dict(counts)}")
    return {
        "schema_version": "paper_e4_real_task_preflight.v1",
        "all_passed": True,
        "task_count": len(tasks),
        "unique_task_count": len({task["sample_id"] for task in tasks}),
        "profile_count": len(counts),
        "profile_task_counts": dict(sorted(counts.items())),
        "checks": [
            "fixed 504/48/24 univariate shape",
            "finite imputed context",
            "at least 24 observed future labels",
            "stable seasonal MASE denominator",
            "real history timestamps",
            "unique deterministic task IDs",
            "all official origin strata represented when eligible",
        ],
    }


def selection_manifest_payload(
    *,
    output_dir: Path,
    gift_eval_dir: Path,
    task_manifest: dict[str, Any],
    coordinates: pd.DataFrame,
    qualified: pd.DataFrame,
    hypotheses: pd.DataFrame,
) -> dict[str, Any]:
    capability_family_counts = (
        qualified.groupby("capability_id", sort=True)["family_id"]
        .nunique()
        .to_dict()
    )
    capability_cell_counts = (
        qualified.groupby("capability_id", sort=True).size().to_dict()
    )
    return {
        "schema_version": "paper_e4_selection_manifest.v1",
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "frozen_before_real_model_inference": True,
        "models": list(MODELS),
        "baselines": list(BASELINES),
        "profiles": [
            transfer.profile_spec_payload(spec)
            for spec in transfer.TRANSFER_PROFILE_SPECS
        ],
        "capabilities": list(CAPABILITIES),
        "source_identities": {
            "gift_eval_dir": str(gift_eval_dir),
            "gift_eval_arrow_manifest_sha256": (
                read_json(CAPABILITY_AUDIT_PATH)["config"]["input_identities"][
                    "gift_eval_arrow_manifest_sha256"
                ]
            ),
            "transfer_freeze_manifest_sha256": sha256_file(
                TRANSFER_FREEZE_MANIFEST_PATH
            ),
            "capability_audit_sha256": sha256_file(CAPABILITY_AUDIT_PATH),
            "e3_v2_manifest_sha256": sha256_file(E3_V2_DIR / "manifest.json"),
            "e3_v2_scores_sha256": sha256_file(E3_V2_SCORES_PATH),
            "e3_v2_contrasts_sha256": sha256_file(E3_V2_CONTRASTS_PATH),
            "e3_v1_manifest_sha256": sha256_file(E3_V1_DIR / "manifest.json"),
            "e3_v1_scores_sha256": sha256_file(E3_V1_SCORES_PATH),
        },
        "code_identities": {
            "runner": relative_path(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "protocol": relative_path(PROTOCOL_PATH),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "gift_eval_protocol_git_commit": git_commit(
                Path.home() / "xmy/gift-eval-code"
            ),
        },
        "task_manifest_sha256": sha256_file(output_dir / "task_manifest.json"),
        "task_file_sha256": str(task_manifest["task_file_sha256"]),
        "task_count": int(task_manifest["task_count"]),
        "task_selection_policy": task_selection_policy(),
        "capability_qualification": {
            "source": (
                "train-only median_canonical_intensity_coordinate from frozen audit"
            ),
            "threshold": HIGH_LOADING_THRESHOLD,
            "qualified_cell_count": len(qualified),
            "qualified_profile_count": int(qualified["profile_id"].nunique()),
            "qualified_family_count": int(qualified["family_id"].nunique()),
            "cell_count_by_capability": {
                capability: int(capability_cell_counts.get(capability, 0))
                for capability in CAPABILITIES
            },
            "family_count_by_capability": {
                capability: int(capability_family_counts.get(capability, 0))
                for capability in CAPABILITIES
            },
            "nonlinear_policy": (
                "not externally testable in this slice because all train-only "
                "coordinates equal 1"
            ),
            "coordinate_row_count": len(coordinates),
            "coordinate_file_sha256": sha256_file(
                output_dir / "capability_coordinates.csv"
            ),
            "qualified_cell_file_sha256": sha256_file(
                output_dir / "qualified_cells.csv"
            ),
        },
        "predictors": {
            "ids": list(PREDICTOR_IDS),
            "definition": (
                "mean over intensities of log(model MASE / seasonal-naive MASE)"
            ),
            "file_sha256": sha256_file(output_dir / "synthetic_predictors.csv"),
            "wrong_label_null": (
                "exact enumeration of all non-identity global permutations of "
                "six capability labels"
            ),
        },
        "pair_hypotheses": {
            "selection_rule": (
                "top two E3 relative gaps with paired CI lower > 0 for each "
                "capability supported by at least three high-loading families"
            ),
            "count": len(hypotheses),
            "file_sha256": sha256_file(output_dir / "pair_hypotheses.csv"),
        },
        "statistics": {
            "primary_endpoint": (
                "family-macro Kendall tau-b for v2_dataset_local_capability"
            ),
            "secondary": [
                "Spearman rho",
                "model-pair direction concordance",
                "within-cell z-score Pearson correlation",
                "within-cell z-score RMSE",
                "paired tau delta versus v2_scalar_macro",
            ],
            "macro_order": "capability within profile, profile within family, family",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_cluster": "held-out family",
            "leave_one_family_out": True,
            "alpha": 0.05,
        },
    }


def selection_receipt_payload(output_dir: Path) -> dict[str, Any]:
    selection_manifest = output_dir / "selection_manifest.json"
    task_manifest = output_dir / "task_manifest.json"
    task_file = output_dir / "tasks.jsonl"
    return {
        "schema_version": "paper_e4_selection_receipt.v1",
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "frozen_before_real_model_inference": True,
        "selection_manifest_path": relative_path(selection_manifest),
        "selection_manifest_sha256": sha256_file(selection_manifest),
        "task_manifest_path": relative_path(task_manifest),
        "task_manifest_sha256": sha256_file(task_manifest),
        "task_file_path": relative_path(task_file),
        "task_file_sha256": sha256_file(task_file),
        "synthetic_predictors_sha256": sha256_file(
            output_dir / "synthetic_predictors.csv"
        ),
        "qualified_cells_sha256": sha256_file(output_dir / "qualified_cells.csv"),
        "pair_hypotheses_sha256": sha256_file(output_dir / "pair_hypotheses.csv"),
        "runner_path": relative_path(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_path": relative_path(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "instruction": (
            "copy this exact JSON to the version-controlled selection receipt "
            "path and commit it before --stage infer"
        ),
    }


def run_inference_stage(output_dir: Path, *, args: argparse.Namespace) -> None:
    config = read_json(output_dir / "config.json")
    validate_committed_selection_receipt(output_dir)
    if int(config["expected_task_count"]) != count_jsonl(output_dir / "tasks.jsonl"):
        raise ValueError("E4 task count changed after selection freeze")

    # Reuse the already exercised E2 loading, topology validation, grouped HTTP
    # concurrency and append-safe retry engine. Only timestamps and metric
    # serialization are specialized for masked real targets.
    inference.sample_timestamps = e4_sample_timestamps
    inference.prediction_row = e4_prediction_row
    inference.run_inference(output_dir, config=config, args=args)
    write_json(
        output_dir / "inference_freeze_status.json",
        {
            "schema_version": "paper_e4_inference_freeze_status.v1",
            "selection_receipt_commit": last_commit_touching(
                SELECTION_RECEIPT_PATH
            ),
            "selection_manifest_sha256": sha256_file(
                output_dir / "selection_manifest.json"
            ),
            "task_file_sha256": sha256_file(output_dir / "tasks.jsonl"),
        },
    )


def e4_sample_timestamps(sample: dict[str, Any]) -> list[str]:
    timestamps = [str(value) for value in sample.get("timestamps") or []]
    if len(timestamps) != int(sample["context_length"]):
        raise ValueError(
            f"{sample.get('sample_id')} has {len(timestamps)} real timestamps"
        )
    return timestamps


def e4_prediction_row(
    model_id: str,
    model_group: str,
    sample: dict[str, Any],
    forecast: np.ndarray | list[list[float]],
) -> dict[str, Any]:
    values = np.asarray(forecast, dtype=float)
    if values.shape != (HORIZON, 1) or not np.isfinite(values).all():
        raise ValueError(
            f"{model_id}/{sample['sample_id']} invalid forecast shape/values "
            f"{values.shape}"
        )
    context_length = int(sample["context_length"])
    context = np.asarray(sample["target"][:context_length], dtype=float).reshape(-1)
    future = np.asarray(
        [
            float(row[0]) if row[0] is not None else np.nan
            for row in sample["target"][context_length:]
        ],
        dtype=float,
    )
    metrics = masked_real_metrics(
        future,
        values[:, 0],
        context=context,
        period=int(sample["season_length"]),
    )
    return {
        "schema_version": "paper_e4_real_prediction.v1",
        "model_id": model_id,
        "model_group": model_group,
        "sample_id": str(sample["sample_id"]),
        "profile_id": str(sample["profile_id"]),
        "dataset_name": str(sample["dataset_name"]),
        "family_id": str(sample["family_id"]),
        "series_id": str(sample["series_id"]),
        "base_item_id": str(sample["base_item_id"]),
        "origin_index": int(sample["origin_index"]),
        "profile_task_index": int(sample["profile_task_index"]),
        "metrics": metrics,
        "forecast": values.tolist(),
    }


def masked_real_metrics(
    target: np.ndarray,
    forecast: np.ndarray,
    *,
    context: np.ndarray,
    period: int,
) -> dict[str, float | int]:
    expected = np.asarray(target, dtype=float).reshape(-1)
    predicted = np.asarray(forecast, dtype=float).reshape(-1)
    if expected.shape != predicted.shape:
        raise ValueError("masked real target/forecast shape mismatch")
    observed = np.isfinite(expected)
    if int(observed.sum()) < MIN_FUTURE_OBSERVED_COUNT:
        raise ValueError("masked real metric has insufficient future observations")
    errors = predicted[observed] - expected[observed]
    absolute = np.abs(errors)
    scale = seasonal_mase_scale(np.asarray(context, dtype=float), period)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("masked real metric has invalid MASE scale")
    return {
        "mse": float(np.mean(errors**2)),
        "mae": float(np.mean(absolute)),
        "mase": float(np.mean(absolute) / scale),
        "mase_scale": float(scale),
        "abs_error_sum": float(np.sum(absolute)),
        "squared_error_sum": float(np.sum(errors**2)),
        "target_abs_sum": float(np.sum(np.abs(expected[observed]))),
        "observed_future_count": int(observed.sum()),
    }


def validate_committed_selection_receipt(output_dir: Path) -> None:
    require_file(SELECTION_RECEIPT_PATH)
    receipt = read_json(SELECTION_RECEIPT_PATH)
    expected = selection_receipt_payload(output_dir)
    ignored = {"instruction"}
    observed_comparable = {
        key: value for key, value in receipt.items() if key not in ignored
    }
    expected_comparable = {
        key: value for key, value in expected.items() if key not in ignored
    }
    if observed_comparable != expected_comparable:
        raise ValueError(
            "committed E4 selection receipt does not match runtime freeze"
        )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_path(SELECTION_RECEIPT_PATH)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if tracked.returncode:
        raise ValueError("E4 selection receipt is not tracked by git")
    relevant_paths = [
        relative_path(SELECTION_RECEIPT_PATH),
        relative_path(PROTOCOL_PATH),
        relative_path(Path(__file__).resolve()),
    ]
    for args in (
        ["git", "diff", "--quiet", "HEAD", "--", *relevant_paths],
        ["git", "diff", "--cached", "--quiet", "--", *relevant_paths],
    ):
        result = subprocess.run(args, cwd=REPO_ROOT, check=False)
        if result.returncode:
            raise ValueError(
                "E4 receipt/protocol/runner has uncommitted changes; inference "
                "is blocked"
            )


def analyze_experiment(output_dir: Path, *, bootstrap_replicates: int) -> None:
    validate_committed_selection_receipt(output_dir)
    config = read_json(output_dir / "config.json")
    tasks = {str(row["sample_id"]): row for row in iter_jsonl(output_dir / "tasks.jsonl")}
    if len(tasks) != int(config["expected_task_count"]):
        raise ValueError("E4 task map is incomplete")
    observations = load_real_observations(output_dir, tasks=tasks)
    profile_scores = real_profile_score_frame(observations)
    predictors = pd.read_csv(output_dir / "synthetic_predictors.csv")
    qualified = pd.read_csv(output_dir / "qualified_cells.csv")
    coordinates = pd.read_csv(output_dir / "capability_coordinates.csv")
    hypotheses = pd.read_csv(output_dir / "pair_hypotheses.csv")

    model_cell_scores, cell_concordance = concordance_frames(
        profile_scores=profile_scores,
        predictors=predictors,
        qualified=qualified,
    )
    predictor_summary, bootstrap_summary = predictor_summary_frames(
        cell_concordance,
        bootstrap_replicates=bootstrap_replicates,
    )
    capability_summary = capability_concordance_frame(cell_concordance)
    lodo = leave_one_family_out_frame(cell_concordance)
    permutation, permutation_summary = exact_label_permutation_frame(
        profile_scores=profile_scores,
        predictors=predictors,
        qualified=qualified,
        observed_summary=predictor_summary,
    )
    pair_profile_gaps, pair_results, pair_overall = pair_hypothesis_frames(
        profile_scores=profile_scores,
        coordinates=coordinates,
        hypotheses=hypotheses,
        bootstrap_replicates=bootstrap_replicates,
    )
    coverage = real_task_coverage_frame(tasks.values())

    write_csv(output_dir / "real_observations.csv", observations)
    write_csv(output_dir / "real_profile_scores.csv", profile_scores)
    write_csv(output_dir / "cell_model_scores.csv", model_cell_scores)
    write_csv(output_dir / "cell_concordance.csv", cell_concordance)
    write_csv(output_dir / "predictor_summary.csv", predictor_summary)
    write_csv(output_dir / "capability_concordance.csv", capability_summary)
    write_csv(output_dir / "bootstrap_summary.csv", bootstrap_summary)
    write_csv(output_dir / "leave_one_family_out.csv", lodo)
    write_csv(output_dir / "label_permutation_null.csv", permutation)
    write_csv(output_dir / "pair_profile_gaps.csv", pair_profile_gaps)
    write_csv(output_dir / "pair_hypothesis_results.csv", pair_results)
    write_csv(output_dir / "real_task_coverage.csv", coverage)

    create_figures(
        output_dir,
        profile_scores=profile_scores,
        model_cell_scores=model_cell_scores,
        cell_concordance=cell_concordance,
        predictor_summary=predictor_summary,
        lodo=lodo,
        pair_results=pair_results,
    )
    summary = analysis_summary_payload(
        output_dir=output_dir,
        config=config,
        observations=observations,
        profile_scores=profile_scores,
        predictor_summary=predictor_summary,
        capability_summary=capability_summary,
        permutation_summary=permutation_summary,
        lodo=lodo,
        pair_results=pair_results,
        pair_overall=pair_overall,
        coverage=coverage,
        bootstrap_replicates=bootstrap_replicates,
    )
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        render_analysis_report(summary, predictor_summary, capability_summary, pair_results),
        encoding="utf-8",
    )
    (output_dir / "paper_tables.md").write_text(
        render_paper_tables(predictor_summary, capability_summary, pair_results),
        encoding="utf-8",
    )
    write_final_manifest(output_dir, config=config)
    primary = summary["primary_endpoint"]
    print(
        "E4 primary family-macro Kendall tau="
        f"{primary['estimate']:.4f} "
        f"[{primary['ci_low']:.4f}, {primary['ci_high']:.4f}], "
        f"wrong-label exact p={summary['wrong_label_permutation']['p_value']:.4g}",
        flush=True,
    )


def load_real_observations(
    output_dir: Path,
    *,
    tasks: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_models = (*BASELINES, *MODELS)
    for model_id in expected_models:
        path = inference.prediction_path_for(output_dir, model_id)
        require_file(path)
        observed_ids: set[str] = set()
        for prediction in iter_jsonl(path):
            sample_id = str(prediction["sample_id"])
            if sample_id not in tasks:
                raise ValueError(f"{model_id} has unknown E4 sample {sample_id}")
            if sample_id in observed_ids:
                raise ValueError(f"{model_id} duplicates E4 sample {sample_id}")
            observed_ids.add(sample_id)
            task = tasks[sample_id]
            for field in (
                "profile_id",
                "family_id",
                "series_id",
                "base_item_id",
                "origin_index",
            ):
                if prediction.get(field) != task.get(field):
                    raise ValueError(
                        f"{model_id}/{sample_id} identity mismatch for {field}"
                    )
            metrics = prediction.get("metrics") or {}
            required_metrics = (
                "mase",
                "mae",
                "mse",
                "abs_error_sum",
                "squared_error_sum",
                "target_abs_sum",
                "observed_future_count",
                "mase_scale",
            )
            missing = [name for name in required_metrics if name not in metrics]
            if missing:
                raise ValueError(
                    f"{model_id}/{sample_id} missing metrics: {missing}"
                )
            numeric = {
                name: float(metrics[name])
                for name in required_metrics
                if name != "observed_future_count"
            }
            if not all(np.isfinite(value) for value in numeric.values()):
                raise ValueError(f"{model_id}/{sample_id} has non-finite metrics")
            rows.append(
                {
                    "model_id": model_id,
                    "model_group": str(prediction["model_group"]),
                    "sample_id": sample_id,
                    "profile_id": str(task["profile_id"]),
                    "dataset_name": str(task["dataset_name"]),
                    "family_id": str(task["family_id"]),
                    "series_id": str(task["series_id"]),
                    "base_item_id": str(task["base_item_id"]),
                    "origin_index": int(task["origin_index"]),
                    "mase": numeric["mase"],
                    "mae": numeric["mae"],
                    "mse": numeric["mse"],
                    "abs_error_sum": numeric["abs_error_sum"],
                    "squared_error_sum": numeric["squared_error_sum"],
                    "target_abs_sum": numeric["target_abs_sum"],
                    "observed_future_count": int(
                        metrics["observed_future_count"]
                    ),
                    "mase_scale": numeric["mase_scale"],
                }
            )
        if observed_ids != set(tasks):
            missing = sorted(set(tasks) - observed_ids)
            raise ValueError(
                f"{model_id} predictions incomplete: "
                f"{len(observed_ids)}/{len(tasks)}, first missing={missing[:3]}"
            )
    frame = pd.DataFrame.from_records(rows)
    expected_count = len(tasks) * len(expected_models)
    if len(frame) != expected_count:
        raise ValueError(f"E4 observation count {len(frame)}/{expected_count}")
    return frame.sort_values(
        ["model_id", "profile_id", "sample_id"], kind="stable"
    ).reset_index(drop=True)


def real_profile_score_frame(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model_id", "model_group", "profile_id", "dataset_name", "family_id"]
    for key, group in observations.groupby(keys, sort=True):
        model_id, model_group, profile_id, dataset_name, family_id = key
        target_abs_sum = float(group["target_abs_sum"].sum())
        rows.append(
            {
                "model_id": model_id,
                "model_group": model_group,
                "profile_id": profile_id,
                "dataset_name": dataset_name,
                "family_id": family_id,
                "sample_count": int(len(group)),
                "series_count": int(group["series_id"].nunique()),
                "origin_count": int(group["origin_index"].nunique()),
                "mase_mean": float(group["mase"].mean()),
                "mae_pooled": float(
                    group["abs_error_sum"].sum()
                    / group["observed_future_count"].sum()
                ),
                "nmae_abs": (
                    float(group["abs_error_sum"].sum() / target_abs_sum)
                    if target_abs_sum > 0
                    else np.nan
                ),
            }
        )
    scores = pd.DataFrame.from_records(rows)
    baseline = scores[scores["model_id"] == "seasonal_naive"][
        ["profile_id", "mase_mean", "nmae_abs"]
    ].rename(
        columns={
            "mase_mean": "seasonal_naive_mase_mean",
            "nmae_abs": "seasonal_naive_nmae_abs",
        }
    )
    foundation = scores[scores["model_id"].isin(MODELS)].copy()
    foundation = foundation.merge(
        baseline,
        on="profile_id",
        how="left",
        validate="many_to_one",
    )
    if foundation[
        ["mase_mean", "seasonal_naive_mase_mean"]
    ].isna().any().any():
        raise ValueError("real profile scores are missing MASE")
    if bool((foundation[["mase_mean", "seasonal_naive_mase_mean"]] <= 0).any().any()):
        raise ValueError("real profile MASE must be positive")
    foundation["real_log_mase_ratio"] = np.log(
        foundation["mase_mean"] / foundation["seasonal_naive_mase_mean"]
    )
    foundation["seasonal_naive_skill_mase"] = 1.0 - (
        foundation["mase_mean"] / foundation["seasonal_naive_mase_mean"]
    )
    foundation["seasonal_naive_skill_nmae"] = 1.0 - (
        foundation["nmae_abs"] / foundation["seasonal_naive_nmae_abs"]
    )
    foundation["real_mase_rank"] = foundation.groupby("profile_id", sort=False)[
        "mase_mean"
    ].rank(method="average", ascending=True)
    return foundation.sort_values(
        ["profile_id", "mase_mean", "model_id"], kind="stable"
    ).reset_index(drop=True)


def concordance_frames(
    *,
    profile_scores: pd.DataFrame,
    predictors: pd.DataFrame,
    qualified: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_columns = [
        "profile_id",
        "model_id",
        "real_log_mase_ratio",
        "mase_mean",
        "real_mase_rank",
    ]
    grid = qualified[
        [
            "profile_id",
            "dataset_name",
            "family_id",
            "capability_id",
            "canonical_intensity_coordinate",
        ]
    ].merge(
        predictors,
        on=["profile_id", "capability_id"],
        how="left",
        validate="one_to_many",
    )
    grid = grid.merge(
        profile_scores[real_columns],
        on=["profile_id", "model_id"],
        how="left",
        validate="many_to_one",
    )
    if grid[
        ["synthetic_log_mase_ratio", "real_log_mase_ratio"]
    ].isna().any().any():
        raise ValueError("qualified E4 concordance grid is incomplete")
    grid["synthetic_rank"] = grid.groupby(
        ["predictor_id", "profile_id", "capability_id"], sort=False
    )["synthetic_log_mase_ratio"].rank(method="average", ascending=True)
    grid["synthetic_z"] = grid.groupby(
        ["predictor_id", "profile_id", "capability_id"], sort=False
    )["synthetic_log_mase_ratio"].transform(zscore)
    grid["real_z"] = grid.groupby(
        ["predictor_id", "profile_id", "capability_id"], sort=False
    )["real_log_mase_ratio"].transform(zscore)

    cell_rows: list[dict[str, Any]] = []
    keys = [
        "predictor_id",
        "profile_id",
        "dataset_name",
        "family_id",
        "capability_id",
        "canonical_intensity_coordinate",
    ]
    for key, group in grid.groupby(keys, sort=True):
        (
            predictor_id,
            profile_id,
            dataset_name,
            family_id,
            capability_id,
            coordinate,
        ) = key
        group = group.sort_values("model_id", kind="stable")
        synthetic = group["synthetic_log_mase_ratio"].to_numpy(dtype=float)
        real = group["real_log_mase_ratio"].to_numpy(dtype=float)
        metrics = concordance_metrics(synthetic, real)
        cell_rows.append(
            {
                "predictor_id": predictor_id,
                "profile_id": profile_id,
                "dataset_name": dataset_name,
                "family_id": family_id,
                "capability_id": capability_id,
                "canonical_intensity_coordinate": float(coordinate),
                "model_count": len(group),
                **metrics,
            }
        )
    cell = pd.DataFrame.from_records(cell_rows)
    expected = len(qualified) * len(PREDICTOR_IDS)
    if len(cell) != expected:
        raise ValueError(f"E4 cell concordance count {len(cell)}/{expected}")
    return (
        grid.sort_values(
            ["predictor_id", "profile_id", "capability_id", "model_id"],
            kind="stable",
        ).reset_index(drop=True),
        cell.sort_values(
            ["predictor_id", "profile_id", "capability_id"], kind="stable"
        ).reset_index(drop=True),
    )


CONCORDANCE_METRICS = (
    "kendall_tau_b",
    "spearman_rho",
    "pair_direction_concordance",
    "pearson_centered",
    "zscore_rmse",
)


def concordance_metrics(synthetic: np.ndarray, real: np.ndarray) -> dict[str, float]:
    left = np.asarray(synthetic, dtype=float)
    right = np.asarray(real, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("concordance vectors must be aligned one-dimensional arrays")
    left_z = zscore(pd.Series(left)).to_numpy(dtype=float)
    right_z = zscore(pd.Series(right)).to_numpy(dtype=float)
    return {
        "kendall_tau_b": float(inference.kendall_tau_b(left, right)),
        "spearman_rho": float(spearman_correlation(left, right)),
        "pair_direction_concordance": float(
            pair_direction_concordance(left, right)
        ),
        "pearson_centered": float(pearson_correlation(left_z, right_z)),
        "zscore_rmse": float(np.sqrt(np.mean((left_z - right_z) ** 2))),
    }


def zscore(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    standard_deviation = float(np.std(array, ddof=0))
    if standard_deviation <= 0 or not np.isfinite(standard_deviation):
        return pd.Series(np.zeros(len(array), dtype=float), index=values.index)
    return (values - float(np.mean(array))) / standard_deviation


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    return pearson_correlation(left_rank, right_rank)


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2 or float(np.std(x)) <= 0 or float(np.std(y)) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pair_direction_concordance(left: np.ndarray, right: np.ndarray) -> float:
    matched = 0
    compared = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_delta = float(left[first] - left[second])
            right_delta = float(right[first] - right[second])
            if left_delta == 0 or right_delta == 0:
                continue
            matched += int(left_delta * right_delta > 0)
            compared += 1
    return float(matched / compared) if compared else float("nan")


def predictor_summary_frames(
    cell: pd.DataFrame,
    *,
    bootstrap_replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = family_macro_frame(cell, group_columns=["predictor_id"])
    estimates = (
        family.groupby("predictor_id", sort=True)[list(CONCORDANCE_METRICS)]
        .mean()
        .reset_index()
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    scalar_id = "v2_scalar_macro"
    scalar_family = family[family["predictor_id"] == scalar_id].set_index(
        "family_id"
    )
    for row in estimates.itertuples(index=False):
        predictor_id = str(row.predictor_id)
        predictor_family = family[family["predictor_id"] == predictor_id].set_index(
            "family_id"
        )
        families = sorted(predictor_family.index)
        if families != sorted(scalar_family.index):
            raise ValueError("predictor family support differs from scalar baseline")
        bootstrap_values = {
            metric: np.empty(bootstrap_replicates, dtype=float)
            for metric in CONCORDANCE_METRICS
        }
        delta_values = {
            metric: np.empty(bootstrap_replicates, dtype=float)
            for metric in CONCORDANCE_METRICS
        }
        for replicate in range(bootstrap_replicates):
            indexes = rng.integers(0, len(families), size=len(families))
            sampled = [families[index] for index in indexes]
            for metric in CONCORDANCE_METRICS:
                predictor_values = predictor_family.loc[sampled, metric].to_numpy(
                    dtype=float
                )
                scalar_values = scalar_family.loc[sampled, metric].to_numpy(
                    dtype=float
                )
                bootstrap_values[metric][replicate] = float(
                    np.nanmean(predictor_values)
                )
                delta_values[metric][replicate] = float(
                    np.nanmean(predictor_values - scalar_values)
                )
        output: dict[str, Any] = {
            "predictor_id": predictor_id,
            "family_count": len(families),
            "qualified_cell_count": int(
                cell[cell["predictor_id"] == predictor_id][
                    ["profile_id", "capability_id"]
                ].drop_duplicates().shape[0]
            ),
        }
        for metric in CONCORDANCE_METRICS:
            estimate = float(getattr(row, metric))
            ci_low, ci_high = percentile_ci(bootstrap_values[metric])
            delta_estimate = float(
                np.nanmean(
                    predictor_family[metric].to_numpy(dtype=float)
                    - scalar_family[metric].to_numpy(dtype=float)
                )
            )
            delta_low, delta_high = percentile_ci(delta_values[metric])
            output[metric] = estimate
            output[f"{metric}_ci_low"] = ci_low
            output[f"{metric}_ci_high"] = ci_high
            output[f"{metric}_delta_vs_scalar"] = delta_estimate
            output[f"{metric}_delta_vs_scalar_ci_low"] = delta_low
            output[f"{metric}_delta_vs_scalar_ci_high"] = delta_high
            bootstrap_rows.extend(
                [
                    {
                        "estimand_type": "predictor",
                        "predictor_id": predictor_id,
                        "reference_predictor": "",
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "bootstrap_replicates": bootstrap_replicates,
                        "cluster": "family",
                    },
                    {
                        "estimand_type": "delta_vs_scalar",
                        "predictor_id": predictor_id,
                        "reference_predictor": scalar_id,
                        "metric": metric,
                        "estimate": delta_estimate,
                        "ci_low": delta_low,
                        "ci_high": delta_high,
                        "bootstrap_replicates": bootstrap_replicates,
                        "cluster": "family_paired",
                    },
                ]
            )
        summary_rows.append(output)
    summary = pd.DataFrame.from_records(summary_rows)
    predictor_order = {name: index for index, name in enumerate(PREDICTOR_IDS)}
    summary["_order"] = summary["predictor_id"].map(predictor_order)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return summary, pd.DataFrame.from_records(bootstrap_rows)


def family_macro_frame(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
) -> pd.DataFrame:
    metrics = list(CONCORDANCE_METRICS)
    profile = (
        frame.groupby(
            [*group_columns, "family_id", "profile_id"], sort=True
        )[metrics]
        .mean()
        .reset_index()
    )
    family = (
        profile.groupby([*group_columns, "family_id"], sort=True)[metrics]
        .mean()
        .reset_index()
    )
    return family


def capability_concordance_frame(cell: pd.DataFrame) -> pd.DataFrame:
    family = family_macro_frame(
        cell,
        group_columns=["predictor_id", "capability_id"],
    )
    rows: list[dict[str, Any]] = []
    for key, group in family.groupby(
        ["predictor_id", "capability_id"], sort=True
    ):
        predictor_id, capability_id = key
        rows.append(
            {
                "predictor_id": predictor_id,
                "capability_id": capability_id,
                "family_count": int(group["family_id"].nunique()),
                **{
                    metric: float(group[metric].mean())
                    for metric in CONCORDANCE_METRICS
                },
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["predictor_id", "capability_id"], kind="stable"
    ).reset_index(drop=True)


def leave_one_family_out_frame(cell: pd.DataFrame) -> pd.DataFrame:
    families = sorted(cell["family_id"].unique())
    rows: list[dict[str, Any]] = []
    for excluded in families:
        retained = cell[cell["family_id"] != excluded]
        family = family_macro_frame(retained, group_columns=["predictor_id"])
        for predictor_id, group in family.groupby("predictor_id", sort=True):
            rows.append(
                {
                    "excluded_family": excluded,
                    "predictor_id": predictor_id,
                    "remaining_family_count": int(group["family_id"].nunique()),
                    **{
                        metric: float(group[metric].mean())
                        for metric in CONCORDANCE_METRICS
                    },
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(
        ["excluded_family", "predictor_id"], kind="stable"
    ).reset_index(drop=True)


def exact_label_permutation_frame(
    *,
    profile_scores: pd.DataFrame,
    predictors: pd.DataFrame,
    qualified: pd.DataFrame,
    observed_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    local = predictors[
        predictors["predictor_id"] == "v2_dataset_local_capability"
    ]
    lookup = {
        (str(row.profile_id), str(row.capability_id), str(row.model_id)): float(
            row.synthetic_log_mase_ratio
        )
        for row in local.itertuples(index=False)
    }
    real_lookup = {
        (str(row.profile_id), str(row.model_id)): float(row.real_log_mase_ratio)
        for row in profile_scores.itertuples(index=False)
    }
    observed = float(
        observed_summary.loc[
            observed_summary["predictor_id"]
            == "v2_dataset_local_capability",
            "kendall_tau_b",
        ].iloc[0]
    )
    identity = tuple(CAPABILITIES)
    rows: list[dict[str, Any]] = []
    for permutation_index, permutation in enumerate(
        itertools.permutations(CAPABILITIES)
    ):
        if permutation == identity:
            continue
        mapping = dict(zip(CAPABILITIES, permutation, strict=True))
        cell_rows: list[dict[str, Any]] = []
        for qualified_row in qualified.itertuples(index=False):
            profile_id = str(qualified_row.profile_id)
            capability_id = str(qualified_row.capability_id)
            source_capability = mapping[capability_id]
            synthetic = np.asarray(
                [
                    lookup[(profile_id, source_capability, model_id)]
                    for model_id in MODELS
                ],
                dtype=float,
            )
            real = np.asarray(
                [real_lookup[(profile_id, model_id)] for model_id in MODELS],
                dtype=float,
            )
            cell_rows.append(
                {
                    "predictor_id": "wrong_label",
                    "profile_id": profile_id,
                    "family_id": str(qualified_row.family_id),
                    "kendall_tau_b": float(
                        inference.kendall_tau_b(synthetic, real)
                    ),
                    "spearman_rho": spearman_correlation(synthetic, real),
                    "pair_direction_concordance": pair_direction_concordance(
                        synthetic, real
                    ),
                    "pearson_centered": pearson_correlation(
                        zscore(pd.Series(synthetic)).to_numpy(dtype=float),
                        zscore(pd.Series(real)).to_numpy(dtype=float),
                    ),
                    "zscore_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    zscore(pd.Series(synthetic)).to_numpy(
                                        dtype=float
                                    )
                                    - zscore(pd.Series(real)).to_numpy(dtype=float)
                                )
                                ** 2
                            )
                        )
                    ),
                }
            )
        family = family_macro_frame(
            pd.DataFrame.from_records(cell_rows),
            group_columns=["predictor_id"],
        )
        tau = float(family["kendall_tau_b"].mean())
        rows.append(
            {
                "permutation_index": permutation_index,
                "mapping": json.dumps(mapping, sort_keys=True),
                "family_macro_kendall_tau_b": tau,
            }
        )
    frame = pd.DataFrame.from_records(rows)
    null_values = frame["family_macro_kendall_tau_b"].to_numpy(dtype=float)
    p_value = float((1 + int(np.sum(null_values >= observed))) / (1 + len(null_values)))
    return frame, {
        "observed_family_macro_kendall_tau_b": observed,
        "non_identity_permutation_count": len(frame),
        "null_mean": float(np.mean(null_values)),
        "null_ci_low": float(np.quantile(null_values, 0.025)),
        "null_ci_high": float(np.quantile(null_values, 0.975)),
        "exact_one_sided_p_value": p_value,
        "tail": "wrong-label tau >= identity-label tau",
    }


def pair_hypothesis_frames(
    *,
    profile_scores: pd.DataFrame,
    coordinates: pd.DataFrame,
    hypotheses: pd.DataFrame,
    bootstrap_replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    score_lookup = {
        (str(row.profile_id), str(row.model_id)): float(row.real_log_mase_ratio)
        for row in profile_scores.itertuples(index=False)
    }
    profile_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 101)
    for hypothesis in hypotheses.itertuples(index=False):
        capability_id = str(hypothesis.capability_id)
        capability_coordinates = coordinates[
            coordinates["capability_id"] == capability_id
        ]
        for coordinate in capability_coordinates.itertuples(index=False):
            profile_id = str(coordinate.profile_id)
            gap = (
                score_lookup[(profile_id, str(hypothesis.weaker_model))]
                - score_lookup[(profile_id, str(hypothesis.reference_model))]
            )
            profile_rows.append(
                {
                    "hypothesis_id": str(hypothesis.hypothesis_id),
                    "capability_id": capability_id,
                    "weaker_model": str(hypothesis.weaker_model),
                    "reference_model": str(hypothesis.reference_model),
                    "profile_id": profile_id,
                    "family_id": str(coordinate.family_id),
                    "canonical_intensity_coordinate": float(
                        coordinate.canonical_intensity_coordinate
                    ),
                    "high_loading": bool(
                        float(coordinate.canonical_intensity_coordinate)
                        >= HIGH_LOADING_THRESHOLD
                    ),
                    "real_log_mase_gap": float(gap),
                }
            )
        current = pd.DataFrame.from_records(
            [
                row
                for row in profile_rows
                if row["hypothesis_id"] == str(hypothesis.hypothesis_id)
            ]
        )
        high = current[current["high_loading"]]
        high_family = (
            high.groupby("family_id", sort=True)["real_log_mase_gap"].mean()
        )
        if len(high_family) < MIN_HYPOTHESIS_FAMILIES:
            raise ValueError(
                f"{hypothesis.hypothesis_id} has insufficient high-loading families"
            )
        bootstrap = np.empty(bootstrap_replicates, dtype=float)
        values = high_family.to_numpy(dtype=float)
        for replicate in range(bootstrap_replicates):
            indexes = rng.integers(0, len(values), size=len(values))
            bootstrap[replicate] = float(np.mean(values[indexes]))
        ci_low, ci_high = percentile_ci(bootstrap)
        all_family = (
            current.groupby("family_id", sort=True)[
                ["canonical_intensity_coordinate", "real_log_mase_gap"]
            ]
            .mean()
            .reset_index()
        )
        loading_rho = spearman_correlation(
            all_family["canonical_intensity_coordinate"].to_numpy(dtype=float),
            all_family["real_log_mase_gap"].to_numpy(dtype=float),
        )
        positive_family_count = int((high_family > 0).sum())
        result_rows.append(
            {
                "hypothesis_id": str(hypothesis.hypothesis_id),
                "capability_id": capability_id,
                "weaker_model": str(hypothesis.weaker_model),
                "reference_model": str(hypothesis.reference_model),
                "e3_relative_mase_gap": float(hypothesis.e3_relative_mase_gap),
                "high_loading_profile_count": int(high["profile_id"].nunique()),
                "high_loading_family_count": len(high_family),
                "real_high_loading_log_mase_gap": float(high_family.mean()),
                "real_gap_ci_low": ci_low,
                "real_gap_ci_high": ci_high,
                "direction_supported": bool(high_family.mean() > 0),
                "ci_entirely_positive": bool(ci_low > 0),
                "positive_family_count": positive_family_count,
                "family_direction_binomial_p_one_sided": binomial_upper_tail(
                    positive_family_count, len(high_family)
                ),
                "loading_gap_spearman_rho_all_families": loading_rho,
            }
        )
    profile_frame = pd.DataFrame.from_records(profile_rows)
    result_frame = pd.DataFrame.from_records(result_rows)
    direction_hits = int(result_frame["direction_supported"].sum())
    return profile_frame, result_frame, {
        "hypothesis_count": len(result_frame),
        "direction_supported_count": direction_hits,
        "direction_supported_fraction": float(direction_hits / len(result_frame)),
        "ci_entirely_positive_count": int(
            result_frame["ci_entirely_positive"].sum()
        ),
        "dependence_guardrail": (
            "no binomial p-value across hypotheses because pairs and capabilities "
            "share models and held-out families"
        ),
    }


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def binomial_upper_tail(successes: int, trials: int) -> float:
    if not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    return float(
        sum(math.comb(trials, count) for count in range(successes, trials + 1))
        / (2**trials)
    )


def real_task_coverage_frame(tasks: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "profile_id": str(task["profile_id"]),
            "family_id": str(task["family_id"]),
            "sample_id": str(task["sample_id"]),
            "series_id": str(task["series_id"]),
            "origin_index": int(task["origin_index"]),
            "context_observed_fraction": float(
                task["context_observed_fraction"]
            ),
            "future_observed_fraction": float(
                int(task["future_observed_count"]) / HORIZON
            ),
            "mase_scale": float(task["mase_scale"]),
        }
        for task in tasks
    ]
    frame = pd.DataFrame.from_records(rows)
    result = (
        frame.groupby(["profile_id", "family_id"], sort=True)
        .agg(
            task_count=("sample_id", "size"),
            series_count=("series_id", "nunique"),
            origin_count=("origin_index", "nunique"),
            context_observed_fraction_mean=("context_observed_fraction", "mean"),
            context_observed_fraction_min=("context_observed_fraction", "min"),
            future_observed_fraction_mean=("future_observed_fraction", "mean"),
            future_observed_fraction_min=("future_observed_fraction", "min"),
            mase_scale_median=("mase_scale", "median"),
        )
        .reset_index()
    )
    return result


def create_figures(
    output_dir: Path,
    *,
    profile_scores: pd.DataFrame,
    model_cell_scores: pd.DataFrame,
    cell_concordance: pd.DataFrame,
    predictor_summary: pd.DataFrame,
    lodo: pd.DataFrame,
    pair_results: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_order = list(MODELS)
    profile_order = [
        spec.profile_id for spec in transfer.TRANSFER_PROFILE_SPECS
    ]
    short_profiles = {
        spec.profile_id: spec.dataset_name.replace("/H", "")
        for spec in transfer.TRANSFER_PROFILE_SPECS
    }
    predictor_labels = {
        "v2_dataset_local_capability": "v2 local capability",
        "v2_global_capability": "v2 global capability",
        "v1_development_global_capability": "v1 dev global",
        "v2_scalar_macro": "v2 scalar macro",
    }

    heatmap = profile_scores.pivot(
        index="model_id",
        columns="profile_id",
        values="seasonal_naive_skill_mase",
    ).reindex(index=model_order, columns=profile_order)
    fig, axis = plt.subplots(figsize=(12.0, 4.6))
    bound = max(0.1, float(np.nanmax(np.abs(heatmap.to_numpy(dtype=float)))))
    image = axis.imshow(
        heatmap.to_numpy(dtype=float),
        aspect="auto",
        cmap="RdYlGn",
        vmin=-bound,
        vmax=bound,
    )
    axis.set_xticks(range(len(profile_order)))
    axis.set_xticklabels(
        [short_profiles[value] for value in profile_order],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(model_order)))
    axis.set_yticklabels(model_order)
    axis.set_title("Real controlled GIFT slice: skill vs seasonal naive")
    for row in range(len(model_order)):
        for column in range(len(profile_order)):
            value = float(heatmap.iloc[row, column])
            axis.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    fig.colorbar(image, ax=axis, label="1 - model MASE / seasonal-naive MASE")
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_1_real_model_skill_heatmap")
    plt.close(fig)

    summary = predictor_summary.set_index("predictor_id").loc[list(PREDICTOR_IDS)]
    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    x = np.arange(len(summary))
    estimate = summary["kendall_tau_b"].to_numpy(dtype=float)
    lower = summary["kendall_tau_b_ci_low"].to_numpy(dtype=float)
    upper = summary["kendall_tau_b_ci_high"].to_numpy(dtype=float)
    axis.bar(x, estimate, color=["#2563EB", "#60A5FA", "#A78BFA", "#94A3B8"])
    axis.errorbar(
        x,
        estimate,
        yerr=np.vstack([estimate - lower, upper - estimate]),
        fmt="none",
        color="#111827",
        capsize=4,
        linewidth=1.2,
    )
    axis.axhline(0, color="#475569", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [predictor_labels[value] for value in summary.index],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Family-macro Kendall tau-b")
    axis.set_title("Synthetic predictors of real model ordering")
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_2_predictor_comparison")
    plt.close(fig)

    local_scores = model_cell_scores[
        model_cell_scores["predictor_id"] == "v2_dataset_local_capability"
    ]
    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    colors = {
        capability: plt.cm.tab10(index)
        for index, capability in enumerate(CAPABILITIES)
    }
    for capability, group in local_scores.groupby("capability_id", sort=True):
        axis.scatter(
            group["synthetic_z"],
            group["real_z"],
            s=23,
            alpha=0.68,
            label=capability.replace("_", " "),
            color=colors[capability],
            edgecolor="none",
        )
    limits = (-2.8, 2.8)
    axis.plot(limits, limits, linestyle="--", color="#64748B", linewidth=1)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Synthetic capability score (within-cell z)")
    axis.set_ylabel("Real log-MASE ratio (within-profile z)")
    axis.set_title("Dataset-local synthetic capability vs real effect")
    axis.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_3_synthetic_real_score_scatter")
    plt.close(fig)

    ordered_pairs = pair_results.sort_values(
        "real_high_loading_log_mase_gap", ascending=True
    ).reset_index(drop=True)
    fig_height = max(4.5, 0.42 * len(ordered_pairs) + 1.5)
    fig, axis = plt.subplots(figsize=(9.0, fig_height))
    y = np.arange(len(ordered_pairs))
    estimate = ordered_pairs["real_high_loading_log_mase_gap"].to_numpy(dtype=float)
    lower = ordered_pairs["real_gap_ci_low"].to_numpy(dtype=float)
    upper = ordered_pairs["real_gap_ci_high"].to_numpy(dtype=float)
    colors_pair = [
        "#16A34A" if value > 0 else "#DC2626" for value in estimate
    ]
    for index, color in enumerate(colors_pair):
        axis.errorbar(
            [estimate[index]],
            [y[index]],
            xerr=np.asarray(
                [
                    [estimate[index] - lower[index]],
                    [upper[index] - estimate[index]],
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=5,
            capsize=3,
            linewidth=1.4,
        )
    labels = [
        f"{row.weaker_model} vs {row.reference_model} | "
        f"{str(row.capability_id).replace('_', ' ')}"
        for row in ordered_pairs.itertuples(index=False)
    ]
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.axvline(0, color="#475569", linewidth=0.9)
    axis.set_xlabel("Real high-loading log(MASE weaker / MASE reference)")
    axis.set_title("E3-preregistered capability defect pairs on real data")
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_4_pair_hypothesis_forest")
    plt.close(fig)

    selected_lodo = lodo[
        lodo["predictor_id"].isin(
            ["v2_dataset_local_capability", "v2_scalar_macro"]
        )
    ].copy()
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    families = sorted(selected_lodo["excluded_family"].unique())
    for predictor_id, group in selected_lodo.groupby("predictor_id", sort=True):
        values = group.set_index("excluded_family").loc[families, "kendall_tau_b"]
        axis.plot(
            np.arange(len(families)),
            values,
            marker="o",
            linewidth=1.6,
            label=predictor_labels[predictor_id],
        )
    axis.axhline(0, color="#64748B", linewidth=0.8)
    axis.set_xticks(range(len(families)))
    axis.set_xticklabels(families, rotation=30, ha="right")
    axis.set_ylabel("Kendall tau-b after exclusion")
    axis.set_xlabel("Left-out family")
    axis.set_title("Leave-one-family-out robustness")
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_5_leave_one_family_out")
    plt.close(fig)

    local_cells = cell_concordance[
        cell_concordance["predictor_id"] == "v2_dataset_local_capability"
    ]
    matrix = local_cells.pivot(
        index="capability_id", columns="profile_id", values="kendall_tau_b"
    ).reindex(index=CAPABILITIES, columns=profile_order)
    fig, axis = plt.subplots(figsize=(11.8, 4.4))
    image = axis.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    axis.set_xticks(range(len(profile_order)))
    axis.set_xticklabels(
        [short_profiles[value] for value in profile_order],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(CAPABILITIES)))
    axis.set_yticklabels([value.replace("_", " ") for value in CAPABILITIES])
    axis.set_title("Qualified dataset-local capability transfer (Kendall tau-b)")
    for row in range(len(CAPABILITIES)):
        for column in range(len(profile_order)):
            value = matrix.iloc[row, column]
            if pd.notna(value):
                axis.text(
                    column,
                    row,
                    f"{float(value):+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    fig.colorbar(image, ax=axis, label="Kendall tau-b")
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_6_transfer_cell_heatmap")
    plt.close(fig)


def save_figure(figure: Any, path_without_suffix: Path) -> None:
    for suffix in (".png", ".svg", ".pdf"):
        figure.savefig(
            path_without_suffix.with_suffix(suffix),
            bbox_inches="tight",
            facecolor="white",
        )


def analysis_summary_payload(
    *,
    output_dir: Path,
    config: dict[str, Any],
    observations: pd.DataFrame,
    profile_scores: pd.DataFrame,
    predictor_summary: pd.DataFrame,
    capability_summary: pd.DataFrame,
    permutation_summary: dict[str, Any],
    lodo: pd.DataFrame,
    pair_results: pd.DataFrame,
    pair_overall: dict[str, Any],
    coverage: pd.DataFrame,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    primary_row = predictor_summary[
        predictor_summary["predictor_id"] == "v2_dataset_local_capability"
    ].iloc[0]
    scalar_row = predictor_summary[
        predictor_summary["predictor_id"] == "v2_scalar_macro"
    ].iloc[0]
    local_lodo = lodo[
        lodo["predictor_id"] == "v2_dataset_local_capability"
    ]["kendall_tau_b"]
    primary = {
        "predictor_id": "v2_dataset_local_capability",
        "metric": "family_macro_kendall_tau_b",
        "estimate": float(primary_row["kendall_tau_b"]),
        "ci_low": float(primary_row["kendall_tau_b_ci_low"]),
        "ci_high": float(primary_row["kendall_tau_b_ci_high"]),
        "positive_ci": bool(primary_row["kendall_tau_b_ci_low"] > 0),
    }
    delta = {
        "reference": "v2_scalar_macro",
        "estimate": float(primary_row["kendall_tau_b_delta_vs_scalar"]),
        "ci_low": float(
            primary_row["kendall_tau_b_delta_vs_scalar_ci_low"]
        ),
        "ci_high": float(
            primary_row["kendall_tau_b_delta_vs_scalar_ci_high"]
        ),
        "positive_ci": bool(
            primary_row["kendall_tau_b_delta_vs_scalar_ci_low"] > 0
        ),
        "scalar_estimate": float(scalar_row["kendall_tau_b"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "selection_manifest_sha256": sha256_file(
            output_dir / "selection_manifest.json"
        ),
        "selection_receipt_commit": last_commit_touching(
            SELECTION_RECEIPT_PATH
        ),
        "task_count": int(config["expected_task_count"]),
        "observation_count": len(observations),
        "foundation_model_count": len(MODELS),
        "profile_count": int(profile_scores["profile_id"].nunique()),
        "family_count": int(profile_scores["family_id"].nunique()),
        "qualified_cell_count": int(
            predictor_summary["qualified_cell_count"].max()
        ),
        "qualified_family_count": int(
            predictor_summary["family_count"].max()
        ),
        "primary_endpoint": primary,
        "incremental_vs_scalar": delta,
        "wrong_label_permutation": {
            "p_value": float(
                permutation_summary["exact_one_sided_p_value"]
            ),
            **{
                key: value
                for key, value in permutation_summary.items()
                if key != "exact_one_sided_p_value"
            },
        },
        "leave_one_family_out": {
            "minimum": float(local_lodo.min()),
            "maximum": float(local_lodo.max()),
            "all_positive": bool((local_lodo > 0).all()),
        },
        "pair_hypotheses": pair_overall,
        "pair_hypothesis_rows": clean_json(
            pair_results.to_dict(orient="records")
        ),
        "predictor_rows": clean_json(
            predictor_summary.to_dict(orient="records")
        ),
        "capability_rows": clean_json(
            capability_summary[
                capability_summary["predictor_id"]
                == "v2_dataset_local_capability"
            ].to_dict(orient="records")
        ),
        "task_coverage": clean_json(coverage.to_dict(orient="records")),
        "bootstrap": {
            "replicates": int(bootstrap_replicates),
            "seed": BOOTSTRAP_SEED,
            "cluster": "held-out family",
        },
        "interpretation_guardrails": [
            "hard ranks are accompanied by continuous score association",
            "nonlinear_persistence has no qualified real support in this slice",
            "pair hypotheses decompose the primary endpoint and are not independent discoveries",
            "no exploratory case is included in the confirmatory outputs",
        ],
    }


def render_analysis_report(
    summary: dict[str, Any],
    predictor_summary: pd.DataFrame,
    capability_summary: pd.DataFrame,
    pair_results: pd.DataFrame,
) -> str:
    primary = summary["primary_endpoint"]
    delta = summary["incremental_vs_scalar"]
    permutation = summary["wrong_label_permutation"]
    pair = summary["pair_hypotheses"]
    lines = [
        "# Paper v2 E4：合成能力画像到真实缺陷的迁移",
        "",
        "## 结论摘要",
        "",
        (
            f"- 主终点 family-macro Kendall tau-b = {primary['estimate']:.4f}，"
            f"family-cluster 95% CI [{primary['ci_low']:.4f}, "
            f"{primary['ci_high']:.4f}]。"
        ),
        (
            f"- 相对忽略 capability 的 scalar synthetic baseline，tau 增量为 "
            f"{delta['estimate']:+.4f}，95% CI [{delta['ci_low']:.4f}, "
            f"{delta['ci_high']:.4f}]；scalar 本身 tau={delta['scalar_estimate']:.4f}。"
        ),
        (
            f"- 六能力错标签的 719 个非 identity 精确置换：p="
            f"{permutation['p_value']:.4g}，null mean="
            f"{permutation['null_mean']:.4f}。"
        ),
        (
            f"- 预注册 E3 配对缺陷方向命中 {pair['direction_supported_count']}/"
            f"{pair['hypothesis_count']}；其中 "
            f"{pair['ci_entirely_positive_count']} 个 family-bootstrap CI 完全大于 0。"
        ),
        (
            f"- leave-one-family-out 的 local tau 范围为 "
            f"[{summary['leave_one_family_out']['minimum']:.4f}, "
            f"{summary['leave_one_family_out']['maximum']:.4f}]。"
        ),
        "",
        "这些统计只回答预先冻结的九个 hourly profiles、504/48 窗口和六个单变量能力。"
        "`nonlinear_persistence` 在 train-only audit 中没有真实 headroom，因此没有进入"
        "确认性 high-loading 单元。",
        "",
        "## Predictor 对比",
        "",
        "| Predictor | Kendall tau-b | 95% CI | Spearman | Pair direction | Pearson | z-RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in predictor_summary.itertuples(index=False):
        lines.append(
            f"| `{row.predictor_id}` | {row.kendall_tau_b:.4f} | "
            f"[{row.kendall_tau_b_ci_low:.4f}, {row.kendall_tau_b_ci_high:.4f}] | "
            f"{row.spearman_rho:.4f} | {row.pair_direction_concordance:.4f} | "
            f"{row.pearson_centered:.4f} | {row.zscore_rmse:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Dataset-local capability 分解",
            "",
            "| Capability | Families | Kendall | Spearman | Pair direction | Pearson |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    local_capability = capability_summary[
        capability_summary["predictor_id"]
        == "v2_dataset_local_capability"
    ]
    for row in local_capability.itertuples(index=False):
        lines.append(
            f"| `{row.capability_id}` | {row.family_count} | "
            f"{row.kendall_tau_b:.4f} | {row.spearman_rho:.4f} | "
            f"{row.pair_direction_concordance:.4f} | "
            f"{row.pearson_centered:.4f} |"
        )
    lines.extend(
        [
            "",
            "## E3 预注册配对缺陷",
            "",
            "| Capability | Weaker vs ref | E3 gap | Real log gap | 95% CI | Loading-gap rho |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in pair_results.itertuples(index=False):
        lines.append(
            f"| `{row.capability_id}` | `{row.weaker_model}` vs "
            f"`{row.reference_model}` | {row.e3_relative_mase_gap:.4f} | "
            f"{row.real_high_loading_log_mase_gap:+.4f} | "
            f"[{row.real_gap_ci_low:.4f}, {row.real_gap_ci_high:.4f}] | "
            f"{row.loading_gap_spearman_rho_all_families:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 真实 future 缺失值只在指标中 mask；context 的插值没有使用 future。",
            "- profile 与 family 等权，不能把 task 较多的数据集解释为证据更强。",
            "- E2 已显示 near-tie 下严格名次敏感，因此 Kendall 与 pair direction 必须和"
            "连续 Pearson/z-RMSE 联合阅读。",
            "- 本报告没有根据真实结果新增或删减 dataset、capability、模型 pair 或案例。",
            "",
        ]
    )
    return "\n".join(lines)


def render_paper_tables(
    predictor_summary: pd.DataFrame,
    capability_summary: pd.DataFrame,
    pair_results: pd.DataFrame,
) -> str:
    lines = [
        "# E4 paper-ready tables",
        "",
        "## Main synthetic-to-real transfer",
        "",
        "| Predictor | Kendall tau-b (95% CI) | Spearman | Direction concordance | Centered Pearson |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in predictor_summary.itertuples(index=False):
        lines.append(
            f"| {row.predictor_id} | {row.kendall_tau_b:.3f} "
            f"({row.kendall_tau_b_ci_low:.3f}, {row.kendall_tau_b_ci_high:.3f}) | "
            f"{row.spearman_rho:.3f} | {row.pair_direction_concordance:.3f} | "
            f"{row.pearson_centered:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Local transfer by capability",
            "",
            "| Capability | Family n | Kendall | Spearman | Direction concordance |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    local = capability_summary[
        capability_summary["predictor_id"]
        == "v2_dataset_local_capability"
    ]
    for row in local.itertuples(index=False):
        lines.append(
            f"| {row.capability_id} | {row.family_count} | "
            f"{row.kendall_tau_b:.3f} | {row.spearman_rho:.3f} | "
            f"{row.pair_direction_concordance:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered pair defects",
            "",
            "| Capability | Pair | Real high-loading log gap (95% CI) |",
            "|---|---|---:|",
        ]
    )
    for row in pair_results.itertuples(index=False):
        lines.append(
            f"| {row.capability_id} | {row.weaker_model} / "
            f"{row.reference_model} | {row.real_high_loading_log_mase_gap:+.3f} "
            f"({row.real_gap_ci_low:.3f}, {row.real_gap_ci_high:.3f}) |"
        )
    return "\n".join(lines) + "\n"


def write_final_manifest(output_dir: Path, *, config: dict[str, Any]) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = str(path.relative_to(output_dir))
        files[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "paper_e4_experiment_manifest.v1",
            "experiment_version": EXPERIMENT_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip(),
            "selection_manifest_sha256": config["selection_manifest_sha256"],
            "files": files,
        },
    )


def reject_existing_model_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    prediction_dir = output_dir / "predictions"
    existing = sorted(prediction_dir.glob("*.jsonl")) if prediction_dir.exists() else []
    if existing:
        raise ValueError(
            "prepare refuses to run after real model outputs exist: "
            + ", ".join(path.name for path in existing)
        )
    if (output_dir / "model_status.json").exists():
        raise ValueError("prepare refuses to overwrite an inference status")


def verify_source_manifest(directory: Path, required_files: Sequence[str]) -> None:
    manifest = read_json(directory / "manifest.json")
    files = manifest.get("files") or {}
    for name in required_files:
        path = directory / name
        require_file(path)
        payload = files.get(name)
        if payload is None:
            raise ValueError(f"{directory}/manifest.json does not seal {name}")
        expected = (
            payload.get("sha256") if isinstance(payload, dict) else str(payload)
        )
        if expected != sha256_file(path):
            raise ValueError(f"sealed source hash mismatch: {path}")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            clean_json(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".in_progress")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    clean_json(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def count_jsonl(path: Path) -> int:
    return sum(1 for _row in iter_jsonl(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def git_commit(directory: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=directory,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def last_commit_touching(path: Path) -> str:
    return subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative_path(path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
