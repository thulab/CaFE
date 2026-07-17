#!/usr/bin/env python3
"""Run paper E4-v3 on mechanism-gated, temporally held-out real windows."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for import_path in (BACKEND_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import paper_v2_transfer_common as transfer  # noqa: E402
import run_paper_e2_dynamic_stability as inference  # noqa: E402
import run_paper_v2_e4_synthetic_real_transfer as e4v2  # noqa: E402
from predictive_capability_gate import (  # noqa: E402
    CAPABILITY_IDS,
    evaluate_capability_fingerprint,
    gate_decision,
)
from synthetic_feature_profile import (  # noqa: E402
    gift_eval_short_term_test_holdout_steps,
)


SCHEMA_VERSION = "paper_e4_mechanism_gated_transfer.v1"
EXPERIMENT_VERSION = "v3"
EXPERIMENT_ID = "E4_mechanism_gated_transfer"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runtime/paper_exp/v3/E4_mechanism_gated_transfer"
)
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/"
    "2026-07-17-paper-v3-e4-mechanism-gated-transfer-protocol.md"
)
GATE_FREEZE_PATH = (
    REPO_ROOT
    / "docs/superpowers/baselines/"
    "2026-07-17-paper-v3-mechanism-gate-freeze.json"
)
SELECTION_RECEIPT_PATH = (
    REPO_ROOT
    / "docs/superpowers/baselines/"
    "2026-07-17-paper-v3-e4-selection-freeze.json"
)
E3_V2_DIR = REPO_ROOT / "runtime/paper_exp/v2/E3_model_capability_profiles"
E3_V1_DIR = REPO_ROOT / "runtime/paper_exp/v1/E3_model_capability_profiles"

MODELS = inference.DEFAULT_MODELS
BASELINES = inference.BASELINE_MODELS
PREDICTOR_IDS = e4v2.PREDICTOR_IDS
CONTEXT_LENGTH = transfer.PAPER_V2_CONTEXT_LENGTH
HORIZON = transfer.PAPER_V2_HORIZON
SEASON_LENGTH = transfer.PAPER_V2_SEASON_LENGTH
MIN_CONTEXT_OBSERVED_FRACTION = 0.50
MIN_FUTURE_OBSERVED_COUNT = 24
MASE_ABSOLUTE_FLOOR = 1e-8
MASE_RELATIVE_FLOOR = 1e-6
MIN_CELL_TASKS = 12
MIN_CELL_SERIES = 12
MAX_TASKS_PER_CELL = 160
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2026071771
TIME_COLUMN = inference.TIME_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run E4-v3 with frozen history-only mechanism gates on the "
            "previously untouched GIFT validation horizon."
        )
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "infer", "analyze"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=transfer.DEFAULT_GIFT_EVAL_DIR,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
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
            workers=max(1, int(args.workers)),
            args=args,
        )
    elif args.stage == "infer":
        run_inference_stage(output_dir, args=args)
    else:
        analyze_experiment(
            output_dir,
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
    print(f"E4-v3 output: {output_dir}", flush=True)
    return 0


def prepare_experiment(
    output_dir: Path,
    *,
    gift_eval_dir: Path,
    workers: int,
    args: argparse.Namespace,
) -> None:
    for path in (
        PROTOCOL_PATH,
        GATE_FREEZE_PATH,
        E3_V2_DIR / "manifest.json",
        E3_V1_DIR / "manifest.json",
    ):
        require_file(path)
    reject_existing_predictions(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "failures").mkdir(exist_ok=True)

    gate_artifact = read_json(GATE_FREEZE_PATH)
    thresholds = gate_artifact["thresholds"]
    validated_capabilities = validated_capability_ids(gate_artifact)
    candidates, profile_summaries = build_validation_candidates(
        gift_eval_dir=gift_eval_dir
    )
    print(
        f"mechanism-gating {len(candidates)} untouched validation contexts "
        f"with {workers} workers",
        flush=True,
    )
    nested_diagnostics = compute_gate_diagnostics(
        candidates,
        thresholds=thresholds,
        workers=workers,
    )
    write_jsonl(
        output_dir / "gate_diagnostics.jsonl",
        nested_diagnostics,
    )
    decisions = gate_decision_frame(nested_diagnostics)
    write_csv(output_dir / "gate_decisions.csv", decisions)
    overlap = gate_overlap_frame(
        decisions,
        validated_capabilities=validated_capabilities,
    )
    write_csv(output_dir / "gate_overlap.csv", overlap)

    selected_tasks, cell_map, qualified_cells = select_gated_tasks(
        candidates,
        decisions=decisions,
        validated_capabilities=validated_capabilities,
    )
    write_jsonl(output_dir / "tasks.jsonl", selected_tasks)
    ensure_shared_sample_alias(output_dir)
    write_csv(output_dir / "cell_task_map.csv", cell_map)
    write_csv(output_dir / "qualified_cells.csv", qualified_cells)
    predictors = e4v2.synthetic_predictor_frame()
    write_csv(output_dir / "synthetic_predictors.csv", predictors)
    preflight = validate_selected_tasks(
        selected_tasks,
        cell_map=cell_map,
        qualified_cells=qualified_cells,
    )
    write_json(output_dir / "preflight.json", preflight)

    task_manifest = task_manifest_payload(
        output_dir=output_dir,
        tasks=selected_tasks,
        profile_summaries=profile_summaries,
    )
    write_json(output_dir / "task_manifest.json", task_manifest)
    selection_manifest = selection_manifest_payload(
        output_dir=output_dir,
        gift_eval_dir=gift_eval_dir,
        gate_artifact=gate_artifact,
        validated_capabilities=validated_capabilities,
        profile_summaries=profile_summaries,
        qualified_cells=qualified_cells,
        cell_map=cell_map,
        task_manifest=task_manifest,
    )
    write_json(output_dir / "selection_manifest.json", selection_manifest)
    config = inference_config(
        output_dir=output_dir,
        task_count=len(selected_tasks),
        args=args,
    )
    write_json(output_dir / "config.json", config)
    receipt = selection_receipt_payload(output_dir)
    write_json(output_dir / "selection_receipt_candidate.json", receipt)

    counts = (
        qualified_cells.groupby("capability_id", sort=True)["profile_id"]
        .nunique()
        .to_dict()
    )
    print(
        f"prepared {len(selected_tasks)} unique tasks, "
        f"{len(qualified_cells)} inclusive cells; profiles by capability="
        f"{counts}",
        flush=True,
    )
    print(
        "No inference was requested. Copy the candidate receipt to "
        f"{relative_path(SELECTION_RECEIPT_PATH)} and commit it before infer.",
        flush=True,
    )


def validated_capability_ids(
    gate_artifact: dict[str, Any],
) -> tuple[str, ...]:
    metrics = {
        str(row["capability_id"]): bool(
            row["strict_constraints_satisfied"]
        )
        for row in gate_artifact["calibration_metrics"]
    }
    missing = set(CAPABILITY_IDS) - set(metrics)
    if missing:
        raise ValueError(f"gate freeze misses capabilities: {sorted(missing)}")
    result = tuple(
        capability_id
        for capability_id in CAPABILITY_IDS
        if metrics[capability_id]
    )
    if "nonlinear_persistence" in result:
        raise ValueError(
            "nonlinear persistence unexpectedly passed the frozen gate audit"
        )
    if len(result) != 5:
        raise ValueError(
            f"E4-v3 expected five validated gates, observed {result}"
        )
    return result


def build_validation_candidates(
    *,
    gift_eval_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in transfer.TRANSFER_PROFILE_SPECS:
        profile_tasks, summary = build_profile_validation_candidates(
            spec,
            gift_eval_dir=gift_eval_dir,
        )
        tasks.extend(profile_tasks)
        summaries.append(summary)
        print(
            f"{spec.profile_id}: validation candidates "
            f"{len(profile_tasks)}/{summary['raw_candidate_count']}",
            flush=True,
        )
    if len({str(task["sample_id"]) for task in tasks}) != len(tasks):
        raise ValueError("duplicate E4-v3 validation sample IDs")
    return tasks, summaries


def build_profile_validation_candidates(
    spec: transfer.TransferProfileSpec,
    *,
    gift_eval_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frequency, records = e4v2.read_gift_records(
        gift_eval_dir / spec.dataset_name
    )
    holdout_steps = gift_eval_short_term_test_holdout_steps(
        frequency,
        [
            (
                str(record["item_id"]),
                np.asarray(record["target"], dtype=float),
            )
            for record in records
        ],
    )
    rejection_counts: defaultdict[str, int] = defaultdict(int)
    tasks: list[dict[str, Any]] = []
    raw_count = 0
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
            origin = int(len(values) - holdout_steps - HORIZON)
            if origin < CONTEXT_LENGTH or origin + HORIZON > len(values):
                rejection_counts["insufficient_length"] += 1
                continue
            raw_count += 1
            raw_context = np.asarray(
                values[origin - CONTEXT_LENGTH : origin],
                dtype=float,
            )
            context, observed_fraction = transfer.impute_observed_window(
                raw_context,
                minimum_observed_fraction=MIN_CONTEXT_OBSERVED_FRACTION,
            )
            if context is None:
                rejection_counts["context_missing"] += 1
                continue
            future = np.asarray(
                values[origin : origin + HORIZON],
                dtype=float,
            )
            future_observed_count = int(np.isfinite(future).sum())
            if future_observed_count < MIN_FUTURE_OBSERVED_COUNT:
                rejection_counts["future_missing"] += 1
                continue
            mase_scale = e4v2.seasonal_mase_scale(
                context,
                SEASON_LENGTH,
            )
            minimum_scale = max(
                MASE_ABSOLUTE_FLOOR,
                MASE_RELATIVE_FLOOR * float(np.mean(np.abs(context))),
            )
            if not np.isfinite(mase_scale) or mase_scale <= minimum_scale:
                rejection_counts["unstable_mase_scale"] += 1
                continue
            timestamps = e4v2.real_history_timestamps(
                record["start"],
                frequency=frequency,
                start_index=origin - CONTEXT_LENGTH,
                periods=CONTEXT_LENGTH,
            )
            sample_id = validation_sample_id(
                spec.profile_id,
                series_id,
                source_origin=origin,
            )
            tasks.append(
                {
                    "schema_version": "paper_e4_v3_real_task.v1",
                    "sample_id": sample_id,
                    "profile_id": spec.profile_id,
                    "dataset_name": spec.dataset_name,
                    "family_id": spec.family_id,
                    "series_id": series_id,
                    "base_item_id": str(record["item_id"]),
                    "native_row_index": row_index,
                    "channel_index": channel_index,
                    "origin_index": -1,
                    "origin_role": "gift_validation_horizon",
                    "source_origin": origin,
                    "official_test_tail_steps": int(holdout_steps),
                    "context_length": CONTEXT_LENGTH,
                    "horizon": HORIZON,
                    "season_length": SEASON_LENGTH,
                    "target_dim": 1,
                    "covariate_dim": 0,
                    "frequency": str(frequency).lower(),
                    "timestamps": timestamps,
                    "target": (
                        [[float(value)] for value in context]
                        + [
                            [
                                float(value)
                                if np.isfinite(value)
                                else None
                            ]
                            for value in future
                        ]
                    ),
                    "context_observed_fraction": float(observed_fraction),
                    "future_observed_count": future_observed_count,
                    "mase_scale": float(mase_scale),
                    # Compatibility aliases for the audited E2 inference engine.
                    "capability_id": "real_gift_eval_validation",
                    "intensity": 0,
                    "round_index": 0,
                    "sample_index": row_index * max(
                        1,
                        native.shape[0] if native.ndim == 2 else 1,
                    )
                    + channel_index,
                    "covariates": None,
                }
            )
    tasks.sort(
        key=lambda row: (
            int(row["native_row_index"]),
            int(row["channel_index"]),
        )
    )
    return tasks, {
        "profile_id": spec.profile_id,
        "dataset_name": spec.dataset_name,
        "family_id": spec.family_id,
        "source_frequency": frequency,
        "native_record_count": len(records),
        "expanded_series_count": expanded_series_count,
        "official_test_tail_steps": int(holdout_steps),
        "validation_horizon": HORIZON,
        "validation_origin_policy": (
            "series_length - official_test_tail_steps - horizon"
        ),
        "raw_candidate_count": raw_count,
        "eligible_candidate_count": len(tasks),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def validation_sample_id(
    profile_id: str,
    series_id: str,
    *,
    source_origin: int,
) -> str:
    raw = f"e4-v3|{profile_id}|{series_id}|{source_origin}"
    return "e4v3-validation-" + hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:24]


def compute_gate_diagnostics(
    candidates: list[dict[str, Any]],
    *,
    thresholds: dict[str, dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [(task, thresholds) for task in candidates]
    if workers == 1:
        rows = [_diagnose_real_context(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
        ) as executor:
            rows = list(
                executor.map(
                    _diagnose_real_context,
                    jobs,
                    chunksize=max(1, len(jobs) // (workers * 8)),
                )
            )
    rows.sort(
        key=lambda row: (
            row["profile_id"],
            row["series_id"],
            row["sample_id"],
        )
    )
    return rows


def _diagnose_real_context(
    job: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    task, thresholds = job
    context_length = int(task["context_length"])
    history = np.asarray(
        task["target"][:context_length],
        dtype=float,
    )[:, 0]
    fingerprint = evaluate_capability_fingerprint(
        history,
        season_length=int(task["season_length"]),
        pseudo_horizon=int(task["horizon"]),
    )
    decisions = {
        capability_id: gate_decision(
            fingerprint[capability_id],
            thresholds[capability_id],
        )
        for capability_id in CAPABILITY_IDS
    }
    return {
        "schema_version": "paper_e4_v3_real_gate_diagnostic.v1",
        "sample_id": str(task["sample_id"]),
        "profile_id": str(task["profile_id"]),
        "dataset_name": str(task["dataset_name"]),
        "family_id": str(task["family_id"]),
        "series_id": str(task["series_id"]),
        "uses_benchmark_future": False,
        "context_length": context_length,
        "fingerprint": fingerprint,
        "decisions": decisions,
    }


def gate_decision_frame(
    rows: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    flat = []
    for row in rows:
        for capability_id in CAPABILITY_IDS:
            diagnostics = row["fingerprint"][capability_id]
            decision = row["decisions"][capability_id]
            flat.append(
                {
                    "sample_id": row["sample_id"],
                    "profile_id": row["profile_id"],
                    "dataset_name": row["dataset_name"],
                    "family_id": row["family_id"],
                    "series_id": row["series_id"],
                    "capability_id": capability_id,
                    "qualified": bool(decision["qualified"]),
                    "fingerprint_weight": float(
                        decision["fingerprint_weight"]
                    ),
                    "predictive_gain": float(
                        diagnostics[
                            decision["gain_statistic"]
                        ]
                    ),
                    "gain_mean": float(diagnostics["gain_mean"]),
                    "gain_lcb": float(diagnostics["gain_lcb"]),
                    "positive_fold_fraction": float(
                        diagnostics["positive_fold_fraction"]
                    ),
                    "support_median": float(
                        diagnostics["support_median"]
                    ),
                    "parameter_stability": float(
                        diagnostics["parameter_stability"]
                    ),
                    "phase_permutation_pvalue": float(
                        diagnostics["phase_permutation_pvalue"]
                    ),
                    "valid_fold_count": int(
                        diagnostics["valid_fold_count"]
                    ),
                    "uses_benchmark_future": False,
                }
            )
    return pd.DataFrame.from_records(flat).sort_values(
        ["profile_id", "sample_id", "capability_id"],
        kind="stable",
    ).reset_index(drop=True)


def gate_overlap_frame(
    decisions: pd.DataFrame,
    *,
    validated_capabilities: Sequence[str],
) -> pd.DataFrame:
    subset = decisions[
        decisions["capability_id"].isin(validated_capabilities)
    ]
    lookup = subset.pivot(
        index="sample_id",
        columns="capability_id",
        values="qualified",
    ).fillna(False)
    rows = []
    for left, right in itertools.product(
        validated_capabilities,
        validated_capabilities,
    ):
        left_values = lookup[left].to_numpy(dtype=bool)
        right_values = lookup[right].to_numpy(dtype=bool)
        union = int(np.sum(left_values | right_values))
        rows.append(
            {
                "left_capability": left,
                "right_capability": right,
                "both_count": int(np.sum(left_values & right_values)),
                "union_count": union,
                "jaccard": (
                    float(np.sum(left_values & right_values) / union)
                    if union
                    else 0.0
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def select_gated_tasks(
    candidates: list[dict[str, Any]],
    *,
    decisions: pd.DataFrame,
    validated_capabilities: Sequence[str],
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    task_lookup = {
        str(task["sample_id"]): task for task in candidates
    }
    validated = decisions[
        decisions["capability_id"].isin(validated_capabilities)
    ].copy()
    qualified_sets = (
        validated[validated["qualified"]]
        .groupby("sample_id", sort=False)["capability_id"]
        .agg(lambda values: tuple(sorted(str(value) for value in values)))
        .to_dict()
    )
    selected_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for profile_id, capability_id in itertools.product(
        [spec.profile_id for spec in transfer.TRANSFER_PROFILE_SPECS],
        validated_capabilities,
    ):
        subset = validated[
            (validated["profile_id"] == profile_id)
            & (validated["capability_id"] == capability_id)
            & validated["qualified"]
        ].copy()
        if subset.empty:
            continue
        subset["_order"] = subset["sample_id"].map(
            {
                str(task["sample_id"]): index
                for index, task in enumerate(candidates)
            }
        )
        subset = subset.sort_values("_order", kind="stable")
        inclusive_series = int(subset["series_id"].nunique())
        inclusive_eligible = len(subset)
        if (
            inclusive_eligible < MIN_CELL_TASKS
            or inclusive_series < MIN_CELL_SERIES
        ):
            continue
        inclusive_selected = e4v2.select_evenly(
            subset.to_dict(orient="records"),
            min(MAX_TASKS_PER_CELL, inclusive_eligible),
        )
        exclusive_subset = subset[
            subset["sample_id"].map(
                lambda sample_id: len(
                    qualified_sets.get(str(sample_id), ())
                )
                == 1
            )
        ]
        exclusive_series = int(exclusive_subset["series_id"].nunique())
        exclusive_eligible = len(exclusive_subset)
        exclusive_selected: list[dict[str, Any]] = []
        if (
            exclusive_eligible >= MIN_CELL_TASKS
            and exclusive_series >= MIN_CELL_SERIES
        ):
            exclusive_selected = e4v2.select_evenly(
                exclusive_subset.to_dict(orient="records"),
                min(MAX_TASKS_PER_CELL, exclusive_eligible),
            )
        first = inclusive_selected[0]
        cell_rows.append(
            {
                "profile_id": profile_id,
                "dataset_name": first["dataset_name"],
                "family_id": first["family_id"],
                "capability_id": capability_id,
                "inclusive_eligible_count": inclusive_eligible,
                "inclusive_eligible_series_count": inclusive_series,
                "inclusive_selected_count": len(inclusive_selected),
                "exclusive_eligible_count": exclusive_eligible,
                "exclusive_eligible_series_count": exclusive_series,
                "exclusive_selected_count": len(exclusive_selected),
                "minimum_cell_tasks": MIN_CELL_TASKS,
                "minimum_cell_series": MIN_CELL_SERIES,
                "maximum_tasks_per_cell": MAX_TASKS_PER_CELL,
            }
        )
        for scope, selected in (
            ("inclusive", inclusive_selected),
            ("exclusive", exclusive_selected),
        ):
            for row in selected:
                selected_rows.append(
                    {
                        "selection_scope": scope,
                        "profile_id": profile_id,
                        "dataset_name": row["dataset_name"],
                        "family_id": row["family_id"],
                        "capability_id": capability_id,
                        "sample_id": row["sample_id"],
                        "series_id": row["series_id"],
                        "fingerprint_weight": float(
                            row["fingerprint_weight"]
                        ),
                        "qualified_capability_count": len(
                            qualified_sets.get(str(row["sample_id"]), ())
                        ),
                    }
                )
    cell_map = pd.DataFrame.from_records(selected_rows)
    qualified_cells = pd.DataFrame.from_records(cell_rows)
    if qualified_cells.empty:
        raise ValueError("mechanism gates produced no confirmatory real cells")
    inclusive_map = cell_map[
        cell_map["selection_scope"] == "inclusive"
    ]
    required_ids = set(inclusive_map["sample_id"])
    required_ids.update(
        cell_map.loc[
            cell_map["selection_scope"] == "exclusive",
            "sample_id",
        ]
    )
    tasks = [task_lookup[sample_id] for sample_id in sorted(required_ids)]
    tasks.sort(
        key=lambda task: (
            str(task["profile_id"]),
            int(task["native_row_index"]),
            int(task["channel_index"]),
        )
    )
    for index, task in enumerate(tasks):
        task["profile_task_index"] = index
    return (
        tasks,
        cell_map.sort_values(
            [
                "selection_scope",
                "family_id",
                "profile_id",
                "capability_id",
                "sample_id",
            ],
            kind="stable",
        ).reset_index(drop=True),
        qualified_cells.sort_values(
            ["family_id", "profile_id", "capability_id"],
            kind="stable",
        ).reset_index(drop=True),
    )


def validate_selected_tasks(
    tasks: list[dict[str, Any]],
    *,
    cell_map: pd.DataFrame,
    qualified_cells: pd.DataFrame,
) -> dict[str, Any]:
    ids = [str(task["sample_id"]) for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("selected E4-v3 tasks are not unique")
    mapped = set(cell_map["sample_id"])
    if mapped != set(ids):
        raise ValueError("cell map and selected task union differ")
    failures = []
    for task in tasks:
        if (
            int(task["context_length"]) != CONTEXT_LENGTH
            or int(task["horizon"]) != HORIZON
            or int(task["season_length"]) != SEASON_LENGTH
        ):
            failures.append(f"{task['sample_id']}:shape")
        target = task["target"]
        if len(target) != CONTEXT_LENGTH + HORIZON:
            failures.append(f"{task['sample_id']}:target_length")
        context = np.asarray(target[:CONTEXT_LENGTH], dtype=float)
        if not np.isfinite(context).all():
            failures.append(f"{task['sample_id']}:context")
        if len(task["timestamps"]) != CONTEXT_LENGTH:
            failures.append(f"{task['sample_id']}:timestamps")
        if str(task["origin_role"]) != "gift_validation_horizon":
            failures.append(f"{task['sample_id']}:origin")
    if failures:
        raise ValueError(
            "E4-v3 task preflight failed: " + ";".join(failures[:10])
        )
    return {
        "schema_version": "paper_e4_v3_preflight.v1",
        "all_passed": True,
        "unique_task_count": len(tasks),
        "inclusive_cell_count": len(qualified_cells),
        "exclusive_cell_count": int(
            (qualified_cells["exclusive_selected_count"] > 0).sum()
        ),
        "mapped_row_count": len(cell_map),
        "checks": [
            "previously untouched GIFT validation origin",
            "fixed 504/48/24 univariate request shape",
            "history-only gate diagnostics",
            "finite context and masked future",
            "cell support and deterministic cap",
            "unique task union",
        ],
    }


def task_manifest_payload(
    *,
    output_dir: Path,
    tasks: list[dict[str, Any]],
    profile_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    task_path = output_dir / "tasks.jsonl"
    return {
        "schema_version": "paper_e4_v3_task_manifest.v1",
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "task_file": relative_path(task_path),
        "task_file_sha256": sha256_file(task_path),
        "task_id_sequence_sha256": sha256_lines(
            str(task["sample_id"]) for task in tasks
        ),
        "task_count": len(tasks),
        "profile_count": len({str(task["profile_id"]) for task in tasks}),
        "profile_summaries": profile_summaries,
        "request_shape": {
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "season_length": SEASON_LENGTH,
            "target_dim": 1,
            "covariate_dim": 0,
        },
        "temporal_holdout": (
            "single GIFT validation horizon immediately before official test tail"
        ),
    }


def selection_manifest_payload(
    *,
    output_dir: Path,
    gift_eval_dir: Path,
    gate_artifact: dict[str, Any],
    validated_capabilities: Sequence[str],
    profile_summaries: list[dict[str, Any]],
    qualified_cells: pd.DataFrame,
    cell_map: pd.DataFrame,
    task_manifest: dict[str, Any],
) -> dict[str, Any]:
    real_supported_capabilities = sorted(
        qualified_cells["capability_id"].unique()
    )
    return {
        "schema_version": "paper_e4_v3_selection_manifest.v1",
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "frozen_before_real_model_inference": True,
        "models": list(MODELS),
        "baselines": list(BASELINES),
        "profiles": [
            transfer.profile_spec_payload(spec)
            for spec in transfer.TRANSFER_PROFILE_SPECS
        ],
        "source_identities": {
            "gift_eval_dir": str(gift_eval_dir),
            "gate_freeze_path": relative_path(GATE_FREEZE_PATH),
            "gate_freeze_sha256": sha256_file(GATE_FREEZE_PATH),
            "gate_calibration_source_sha256": gate_artifact[
                "calibration_source"
            ]["sha256"],
            "e3_v2_manifest_sha256": sha256_file(
                E3_V2_DIR / "manifest.json"
            ),
            "e3_v1_manifest_sha256": sha256_file(
                E3_V1_DIR / "manifest.json"
            ),
        },
        "code_identities": {
            "runner": relative_path(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "gate_module": relative_path(
                SCRIPT_DIR / "predictive_capability_gate.py"
            ),
            "gate_module_sha256": sha256_file(
                SCRIPT_DIR / "predictive_capability_gate.py"
            ),
            "protocol": relative_path(PROTOCOL_PATH),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "gift_eval_protocol_git_commit": git_commit(
                Path.home() / "xmy/gift-eval-code"
            ),
        },
        "gate_validated_capabilities": list(validated_capabilities),
        "gate_unvalidated_capabilities": [
            capability_id
            for capability_id in CAPABILITY_IDS
            if capability_id not in validated_capabilities
        ],
        "real_supported_capabilities": real_supported_capabilities,
        "insufficient_real_support_capabilities": [
            capability_id
            for capability_id in validated_capabilities
            if capability_id not in real_supported_capabilities
        ],
        "profile_summaries": profile_summaries,
        "selection_policy": {
            "primary": "inclusive frozen mechanism gate",
            "sensitivity": [
                "exclusive single-gate windows",
                "continuous fingerprint-weighted inclusive windows",
            ],
            "minimum_tasks_per_cell": MIN_CELL_TASKS,
            "minimum_series_per_cell": MIN_CELL_SERIES,
            "maximum_tasks_per_cell": MAX_TASKS_PER_CELL,
            "within_cell_sampling": (
                "deterministic evenly spaced selection in Arrow row/channel order"
            ),
            "task_union": (
                "union over selected inclusive and exclusive capability cells"
            ),
        },
        "qualified_cell_count": len(qualified_cells),
        "exclusive_cell_count": int(
            (qualified_cells["exclusive_selected_count"] > 0).sum()
        ),
        "cell_task_map_row_count": len(cell_map),
        "qualified_cells_sha256": sha256_file(
            output_dir / "qualified_cells.csv"
        ),
        "cell_task_map_sha256": sha256_file(
            output_dir / "cell_task_map.csv"
        ),
        "gate_decisions_sha256": sha256_file(
            output_dir / "gate_decisions.csv"
        ),
        "gate_diagnostics_sha256": sha256_file(
            output_dir / "gate_diagnostics.jsonl"
        ),
        "synthetic_predictors_sha256": sha256_file(
            output_dir / "synthetic_predictors.csv"
        ),
        "task_manifest_sha256": sha256_file(
            output_dir / "task_manifest.json"
        ),
        "task_file_sha256": task_manifest["task_file_sha256"],
        "task_count": task_manifest["task_count"],
        "statistics": {
            "primary_predictor": "v2_dataset_local_capability",
            "primary_metric": "family-macro Kendall tau-b",
            "bootstrap_cluster": "family",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "wrong_label_null": (
                "exact nonidentity permutation over validated capability labels"
            ),
        },
    }


def inference_config(
    *,
    output_dir: Path,
    task_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "selection_manifest_sha256": sha256_file(
            output_dir / "selection_manifest.json"
        ),
        "task_manifest_sha256": sha256_file(
            output_dir / "task_manifest.json"
        ),
        "expected_generated_sample_count": int(task_count),
        "expected_task_count": int(task_count),
        "requested_models": list(MODELS),
        "baseline_models": list(BASELINES),
        "model_execution": {
            model_id: dict(inference.MODEL_EXECUTION_CONFIG[model_id])
            for model_id in MODELS
        },
        "devices": str(args.devices),
        "request_max_attempts": int(args.request_max_attempts),
        "forecast_timeout_seconds": int(args.forecast_timeout_seconds),
        "model_load_timeout_seconds": int(
            args.model_load_timeout_seconds
        ),
        "service": {
            "base_url": str(args.base_url).rstrip("/"),
            "api_prefix": "/" + str(args.api_prefix).strip("/"),
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "family",
        },
    }


def selection_receipt_payload(output_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "paper_e4_v3_selection_receipt.v1",
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "frozen_before_real_model_inference": True,
        "selection_manifest_path": relative_path(
            output_dir / "selection_manifest.json"
        ),
        "selection_manifest_sha256": sha256_file(
            output_dir / "selection_manifest.json"
        ),
        "task_manifest_path": relative_path(
            output_dir / "task_manifest.json"
        ),
        "task_manifest_sha256": sha256_file(
            output_dir / "task_manifest.json"
        ),
        "task_file_path": relative_path(output_dir / "tasks.jsonl"),
        "task_file_sha256": sha256_file(output_dir / "tasks.jsonl"),
        "gate_freeze_path": relative_path(GATE_FREEZE_PATH),
        "gate_freeze_sha256": sha256_file(GATE_FREEZE_PATH),
        "gate_decisions_sha256": sha256_file(
            output_dir / "gate_decisions.csv"
        ),
        "qualified_cells_sha256": sha256_file(
            output_dir / "qualified_cells.csv"
        ),
        "cell_task_map_sha256": sha256_file(
            output_dir / "cell_task_map.csv"
        ),
        "synthetic_predictors_sha256": sha256_file(
            output_dir / "synthetic_predictors.csv"
        ),
        "runner_path": relative_path(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "gate_module_path": relative_path(
            SCRIPT_DIR / "predictive_capability_gate.py"
        ),
        "gate_module_sha256": sha256_file(
            SCRIPT_DIR / "predictive_capability_gate.py"
        ),
        "protocol_path": relative_path(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "instruction": (
            "copy this exact JSON to the version-controlled receipt path and "
            "commit it before --stage infer"
        ),
    }


def run_inference_stage(
    output_dir: Path,
    *,
    args: argparse.Namespace,
) -> None:
    validate_committed_selection_receipt(output_dir)
    config = read_json(output_dir / "config.json")
    ensure_shared_sample_alias(output_dir)
    observed = count_jsonl(output_dir / "tasks.jsonl")
    if observed != int(config["expected_task_count"]):
        raise ValueError(
            f"E4-v3 task count changed after freeze: {observed}"
        )
    inference.sample_timestamps = e4v2.e4_sample_timestamps
    inference.prediction_row = e4v3_prediction_row
    inference.run_inference(output_dir, config=config, args=args)
    write_json(
        output_dir / "inference_freeze_status.json",
        {
            "schema_version": "paper_e4_v3_inference_freeze_status.v1",
            "selection_receipt_commit": last_commit_touching(
                SELECTION_RECEIPT_PATH
            ),
            "selection_manifest_sha256": sha256_file(
                output_dir / "selection_manifest.json"
            ),
            "task_file_sha256": sha256_file(
                output_dir / "tasks.jsonl"
            ),
        },
    )


def e4v3_prediction_row(
    model_id: str,
    model_group: str,
    sample: dict[str, Any],
    forecast: np.ndarray | list[list[float]],
) -> dict[str, Any]:
    row = e4v2.e4_prediction_row(
        model_id,
        model_group,
        sample,
        forecast,
    )
    row["schema_version"] = "paper_e4_v3_real_prediction.v1"
    row["origin_role"] = "gift_validation_horizon"
    return row


def ensure_shared_sample_alias(output_dir: Path) -> None:
    task_path = output_dir / "tasks.jsonl"
    alias_path = output_dir / "samples.jsonl"
    require_file(task_path)
    if alias_path.is_symlink():
        if alias_path.resolve() != task_path.resolve():
            raise ValueError("E4-v3 samples alias points outside task freeze")
        return
    if alias_path.exists():
        if sha256_file(alias_path) != sha256_file(task_path):
            raise ValueError("E4-v3 samples alias differs from frozen tasks")
        return
    alias_path.symlink_to(task_path.name)


def validate_committed_selection_receipt(output_dir: Path) -> None:
    require_file(SELECTION_RECEIPT_PATH)
    observed = read_json(SELECTION_RECEIPT_PATH)
    expected = selection_receipt_payload(output_dir)
    ignored = {"instruction"}
    if {
        key: value for key, value in observed.items() if key not in ignored
    } != {
        key: value for key, value in expected.items() if key not in ignored
    }:
        raise ValueError(
            "committed E4-v3 selection receipt differs from runtime freeze"
        )
    for path in (
        SELECTION_RECEIPT_PATH,
        GATE_FREEZE_PATH,
        PROTOCOL_PATH,
        Path(__file__).resolve(),
        SCRIPT_DIR / "predictive_capability_gate.py",
    ):
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative_path(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if tracked.returncode:
            raise ValueError(f"E4-v3 frozen source is untracked: {path}")
    relevant = [
        relative_path(SELECTION_RECEIPT_PATH),
        relative_path(GATE_FREEZE_PATH),
        relative_path(PROTOCOL_PATH),
        relative_path(Path(__file__).resolve()),
        relative_path(SCRIPT_DIR / "predictive_capability_gate.py"),
    ]
    for command in (
        ["git", "diff", "--quiet", "HEAD", "--", *relevant],
        ["git", "diff", "--cached", "--quiet", "--", *relevant],
    ):
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode:
            raise ValueError(
                "E4-v3 frozen receipt/protocol/gate/runner has uncommitted "
                "changes; inference is blocked"
            )


def analyze_experiment(
    output_dir: Path,
    *,
    bootstrap_replicates: int,
) -> None:
    validate_committed_selection_receipt(output_dir)
    config = read_json(output_dir / "config.json")
    tasks = {
        str(row["sample_id"]): row
        for row in iter_jsonl(output_dir / "tasks.jsonl")
    }
    observations = e4v2.load_real_observations(
        output_dir,
        tasks=tasks,
    )
    cell_map = pd.read_csv(output_dir / "cell_task_map.csv")
    predictors = pd.read_csv(output_dir / "synthetic_predictors.csv")
    qualified_cells = pd.read_csv(output_dir / "qualified_cells.csv")

    scope_outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for scope, weighted in (
        ("inclusive", False),
        ("exclusive", False),
        ("fingerprint_weighted", True),
    ):
        mapping_scope = "inclusive" if weighted else scope
        scores = real_cell_score_frame(
            observations,
            cell_map=cell_map,
            selection_scope=mapping_scope,
            weighted=weighted,
        )
        if scores.empty:
            continue
        model_grid, concordance = concordance_frames(
            scores=scores,
            predictors=predictors,
        )
        predictor_summary, bootstrap = e4v2.predictor_summary_frames(
            concordance,
            bootstrap_replicates=bootstrap_replicates,
        )
        capability = e4v2.capability_concordance_frame(concordance)
        lodo = e4v2.leave_one_family_out_frame(concordance)
        scope_outputs[scope] = {
            "scores": scores,
            "model_grid": model_grid,
            "concordance": concordance,
            "predictor_summary": predictor_summary,
            "bootstrap": bootstrap,
            "capability": capability,
            "lodo": lodo,
        }
        write_csv(output_dir / f"{scope}_real_cell_scores.csv", scores)
        write_csv(output_dir / f"{scope}_model_grid.csv", model_grid)
        write_csv(
            output_dir / f"{scope}_cell_concordance.csv",
            concordance,
        )
        write_csv(
            output_dir / f"{scope}_predictor_summary.csv",
            predictor_summary,
        )
        write_csv(
            output_dir / f"{scope}_bootstrap_summary.csv",
            bootstrap,
        )
        write_csv(
            output_dir / f"{scope}_capability_concordance.csv",
            capability,
        )
        write_csv(
            output_dir / f"{scope}_leave_one_family_out.csv",
            lodo,
        )
    if "inclusive" not in scope_outputs:
        raise ValueError("E4-v3 inclusive main analysis has no scores")
    permutation, permutation_summary = exact_label_permutation(
        scores=scope_outputs["inclusive"]["scores"],
        predictors=predictors,
        observed_summary=scope_outputs["inclusive"]["predictor_summary"],
        label_universe=validated_capability_ids(read_json(GATE_FREEZE_PATH)),
    )
    write_csv(output_dir / "label_permutation_null.csv", permutation)
    write_csv(output_dir / "real_observations.csv", observations)
    coverage = cell_coverage_frame(
        cell_map=cell_map,
        qualified_cells=qualified_cells,
    )
    write_csv(output_dir / "cell_coverage.csv", coverage)
    create_figures(
        output_dir,
        decisions=pd.read_csv(output_dir / "gate_decisions.csv"),
        qualified_cells=qualified_cells,
        scope_outputs=scope_outputs,
    )
    summary = analysis_summary(
        config=config,
        scope_outputs=scope_outputs,
        permutation_summary=permutation_summary,
        qualified_cells=qualified_cells,
        observations=observations,
        bootstrap_replicates=bootstrap_replicates,
    )
    write_json(output_dir / "summary.json", summary)
    write_text(
        output_dir / "report.md",
        render_analysis_report(summary, scope_outputs),
    )
    write_text(
        output_dir / "paper_tables.md",
        render_paper_tables(scope_outputs),
    )
    write_final_manifest(output_dir, config=config)
    primary = summary["primary_endpoint"]
    print(
        "E4-v3 primary family-macro Kendall tau="
        f"{primary['estimate']:.4f} "
        f"[{primary['ci_low']:.4f}, {primary['ci_high']:.4f}], "
        f"wrong-label p={summary['wrong_label_permutation']['p_value']:.4g}",
        flush=True,
    )


def real_cell_score_frame(
    observations: pd.DataFrame,
    *,
    cell_map: pd.DataFrame,
    selection_scope: str,
    weighted: bool,
) -> pd.DataFrame:
    mapping = cell_map[
        cell_map["selection_scope"] == selection_scope
    ][
        [
            "sample_id",
            "profile_id",
            "dataset_name",
            "family_id",
            "capability_id",
            "series_id",
            "fingerprint_weight",
        ]
    ].copy()
    if mapping.empty:
        return pd.DataFrame()
    joined = observations.merge(
        mapping,
        on=[
            "sample_id",
            "profile_id",
            "dataset_name",
            "family_id",
            "series_id",
        ],
        how="inner",
        validate="many_to_many",
    )
    rows = []
    keys = [
        "model_id",
        "model_group",
        "profile_id",
        "dataset_name",
        "family_id",
        "capability_id",
    ]
    for key, group in joined.groupby(keys, sort=True):
        weights = (
            group["fingerprint_weight"].to_numpy(dtype=float)
            if weighted
            else np.ones(len(group), dtype=float)
        )
        if float(np.sum(weights)) <= 0:
            continue
        mase = float(
            np.average(group["mase"].to_numpy(dtype=float), weights=weights)
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "selection_scope": (
                    "fingerprint_weighted" if weighted else selection_scope
                ),
                "sample_count": len(group),
                "series_count": int(group["series_id"].nunique()),
                "weight_sum": float(np.sum(weights)),
                "mase_mean": mase,
                "mae_mean": float(
                    np.average(
                        group["mae"].to_numpy(dtype=float),
                        weights=weights,
                    )
                ),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return frame
    baseline = frame[frame["model_id"] == "seasonal_naive"][
        ["profile_id", "capability_id", "mase_mean"]
    ].rename(columns={"mase_mean": "seasonal_naive_mase_mean"})
    foundation = frame[frame["model_id"].isin(MODELS)].merge(
        baseline,
        on=["profile_id", "capability_id"],
        how="left",
        validate="many_to_one",
    )
    if foundation["seasonal_naive_mase_mean"].isna().any():
        raise ValueError("E4-v3 gated score misses seasonal-naive baseline")
    foundation["real_log_mase_ratio"] = np.log(
        foundation["mase_mean"]
        / foundation["seasonal_naive_mase_mean"]
    )
    foundation["seasonal_naive_skill_mase"] = 1.0 - (
        foundation["mase_mean"]
        / foundation["seasonal_naive_mase_mean"]
    )
    foundation["real_mase_rank"] = foundation.groupby(
        ["profile_id", "capability_id"],
        sort=False,
    )["mase_mean"].rank(method="average", ascending=True)
    return foundation.sort_values(
        ["family_id", "profile_id", "capability_id", "mase_mean"],
        kind="stable",
    ).reset_index(drop=True)


def concordance_frames(
    *,
    scores: pd.DataFrame,
    predictors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = scores[
        [
            "profile_id",
            "dataset_name",
            "family_id",
            "capability_id",
        ]
    ].drop_duplicates()
    grid = cells.merge(
        predictors,
        on=["profile_id", "capability_id"],
        how="left",
        validate="one_to_many",
    ).merge(
        scores[
            [
                "profile_id",
                "capability_id",
                "model_id",
                "real_log_mase_ratio",
                "mase_mean",
                "real_mase_rank",
                "sample_count",
                "series_count",
            ]
        ],
        on=["profile_id", "capability_id", "model_id"],
        how="left",
        validate="many_to_one",
    )
    if grid[
        ["synthetic_log_mase_ratio", "real_log_mase_ratio"]
    ].isna().any().any():
        raise ValueError("E4-v3 concordance grid is incomplete")
    grid["synthetic_rank"] = grid.groupby(
        ["predictor_id", "profile_id", "capability_id"],
        sort=False,
    )["synthetic_log_mase_ratio"].rank(method="average", ascending=True)
    rows = []
    keys = [
        "predictor_id",
        "profile_id",
        "dataset_name",
        "family_id",
        "capability_id",
    ]
    for key, group in grid.groupby(keys, sort=True):
        group = group.sort_values("model_id", kind="stable")
        metrics = e4v2.concordance_metrics(
            group["synthetic_log_mase_ratio"].to_numpy(dtype=float),
            group["real_log_mase_ratio"].to_numpy(dtype=float),
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "model_count": len(group),
                "sample_count": int(group["sample_count"].iloc[0]),
                "series_count": int(group["series_count"].iloc[0]),
                **metrics,
            }
        )
    cell = pd.DataFrame.from_records(rows)
    return (
        grid.sort_values(
            ["predictor_id", "profile_id", "capability_id", "model_id"],
            kind="stable",
        ).reset_index(drop=True),
        cell.sort_values(
            ["predictor_id", "profile_id", "capability_id"],
            kind="stable",
        ).reset_index(drop=True),
    )


def exact_label_permutation(
    *,
    scores: pd.DataFrame,
    predictors: pd.DataFrame,
    observed_summary: pd.DataFrame,
    label_universe: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    local = predictors[
        predictors["predictor_id"] == "v2_dataset_local_capability"
    ]
    lookup = {
        (
            str(row.profile_id),
            str(row.capability_id),
            str(row.model_id),
        ): float(row.synthetic_log_mase_ratio)
        for row in local.itertuples(index=False)
    }
    observed = float(
        observed_summary.loc[
            observed_summary["predictor_id"]
            == "v2_dataset_local_capability",
            "kendall_tau_b",
        ].iloc[0]
    )
    capabilities = tuple(str(value) for value in label_universe)
    observed_capabilities = set(
        str(value) for value in scores["capability_id"].unique()
    )
    if not observed_capabilities.issubset(set(capabilities)):
        raise ValueError("real gated cells contain labels outside null universe")
    cells = scores[
        ["profile_id", "dataset_name", "family_id", "capability_id"]
    ].drop_duplicates()
    rows = []
    for index, permutation in enumerate(
        itertools.permutations(capabilities)
    ):
        if permutation == capabilities:
            continue
        mapping = dict(zip(capabilities, permutation, strict=True))
        cell_rows = []
        for cell in cells.itertuples(index=False):
            group = scores[
                (scores["profile_id"] == cell.profile_id)
                & (scores["capability_id"] == cell.capability_id)
            ].sort_values("model_id", kind="stable")
            synthetic = np.asarray(
                [
                    lookup[
                        (
                            str(cell.profile_id),
                            mapping[str(cell.capability_id)],
                            str(model_id),
                        )
                    ]
                    for model_id in group["model_id"]
                ],
                dtype=float,
            )
            metric = e4v2.concordance_metrics(
                synthetic,
                group["real_log_mase_ratio"].to_numpy(dtype=float),
            )
            cell_rows.append(
                {
                    "predictor_id": "permuted",
                    "profile_id": cell.profile_id,
                    "family_id": cell.family_id,
                    "capability_id": cell.capability_id,
                    **metric,
                }
            )
        macro = e4v2.family_macro_frame(
            pd.DataFrame.from_records(cell_rows),
            group_columns=["predictor_id"],
        )
        tau = float(macro["kendall_tau_b"].mean())
        rows.append(
            {
                "permutation_index": index,
                "mapping": ";".join(
                    f"{source}->{mapping[source]}"
                    for source in capabilities
                ),
                "family_macro_kendall_tau_b": tau,
            }
        )
    frame = pd.DataFrame.from_records(rows)
    null = frame["family_macro_kendall_tau_b"].to_numpy(dtype=float)
    pvalue = float((1 + np.sum(null >= observed)) / (1 + len(null)))
    return frame, {
        "observed": observed,
        "null_count": len(null),
        "null_mean": float(np.mean(null)) if len(null) else math.nan,
        "null_q95": float(np.quantile(null, 0.95)) if len(null) else math.nan,
        "p_value": pvalue,
        "alternative": "observed tau greater than wrong-label tau",
    }


def cell_coverage_frame(
    *,
    cell_map: pd.DataFrame,
    qualified_cells: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cell in qualified_cells.itertuples(index=False):
        for scope in ("inclusive", "exclusive"):
            subset = cell_map[
                (cell_map["selection_scope"] == scope)
                & (cell_map["profile_id"] == cell.profile_id)
                & (cell_map["capability_id"] == cell.capability_id)
            ]
            rows.append(
                {
                    "selection_scope": scope,
                    "profile_id": cell.profile_id,
                    "dataset_name": cell.dataset_name,
                    "family_id": cell.family_id,
                    "capability_id": cell.capability_id,
                    "selected_task_count": len(subset),
                    "selected_series_count": int(
                        subset["series_id"].nunique()
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def analysis_summary(
    *,
    config: dict[str, Any],
    scope_outputs: dict[str, dict[str, pd.DataFrame]],
    permutation_summary: dict[str, Any],
    qualified_cells: pd.DataFrame,
    observations: pd.DataFrame,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    scopes = {}
    for scope, outputs in scope_outputs.items():
        summary = outputs["predictor_summary"]
        primary = summary[
            summary["predictor_id"]
            == "v2_dataset_local_capability"
        ].iloc[0]
        scalar = summary[
            summary["predictor_id"] == "v2_scalar_macro"
        ].iloc[0]
        scopes[scope] = {
            "cell_count": int(
                outputs["concordance"][
                    ["profile_id", "capability_id"]
                ].drop_duplicates().shape[0]
            ),
            "family_count": int(
                outputs["concordance"]["family_id"].nunique()
            ),
            "primary_kendall_tau_b": float(primary["kendall_tau_b"]),
            "primary_kendall_ci_low": float(
                primary["kendall_tau_b_ci_low"]
            ),
            "primary_kendall_ci_high": float(
                primary["kendall_tau_b_ci_high"]
            ),
            "scalar_kendall_tau_b": float(scalar["kendall_tau_b"]),
            "primary_delta_vs_scalar": float(
                primary["kendall_tau_b_delta_vs_scalar"]
            ),
            "primary_delta_vs_scalar_ci_low": float(
                primary["kendall_tau_b_delta_vs_scalar_ci_low"]
            ),
            "primary_delta_vs_scalar_ci_high": float(
                primary["kendall_tau_b_delta_vs_scalar_ci_high"]
            ),
            "primary_spearman_rho": float(primary["spearman_rho"]),
            "primary_pair_direction_concordance": float(
                primary["pair_direction_concordance"]
            ),
            "primary_pearson_centered": float(
                primary["pearson_centered"]
            ),
        }
    main = scopes["inclusive"]
    capability_support = (
        qualified_cells.groupby("capability_id", sort=True)
        .agg(
            profile_count=("profile_id", "nunique"),
            family_count=("family_id", "nunique"),
            cell_count=("profile_id", "size"),
            selected_task_count=("inclusive_selected_count", "sum"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    gate_validated_capabilities = validated_capability_ids(
        read_json(GATE_FREEZE_PATH)
    )
    real_supported_capabilities = sorted(
        qualified_cells["capability_id"].unique()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "config": config,
        "primary_endpoint": {
            "selection_scope": "inclusive",
            "predictor_id": "v2_dataset_local_capability",
            "metric": "family-macro Kendall tau-b",
            "estimate": main["primary_kendall_tau_b"],
            "ci_low": main["primary_kendall_ci_low"],
            "ci_high": main["primary_kendall_ci_high"],
            "bootstrap_replicates": bootstrap_replicates,
            "cluster": "family",
        },
        "wrong_label_permutation": permutation_summary,
        "scope_results": scopes,
        "capability_support": capability_support,
        "real_task_count": int(
            observations["sample_id"].nunique()
        ),
        "real_prediction_count": len(observations),
        "gate_validated_capabilities": gate_validated_capabilities,
        "gate_unvalidated_capabilities": [
            capability_id
            for capability_id in CAPABILITY_IDS
            if capability_id not in gate_validated_capabilities
        ],
        "real_supported_capabilities": real_supported_capabilities,
        "insufficient_real_support_capabilities": [
            capability_id
            for capability_id in gate_validated_capabilities
            if capability_id not in real_supported_capabilities
        ],
        "interpretation_policy": (
            "Gate passing is mechanism-aligned predictive behavior, not a "
            "causal mechanism claim. Nonlinear persistence is excluded because "
            "its synthetic gate failed the frozen independent audit. "
            "Time-varying seasonality and regime switching remain gate-validated "
            "but lack sufficient real-window support in the frozen GIFT slices."
        ),
    }


def create_figures(
    output_dir: Path,
    *,
    decisions: pd.DataFrame,
    qualified_cells: pd.DataFrame,
    scope_outputs: dict[str, dict[str, pd.DataFrame]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    rates = (
        decisions.groupby(["profile_id", "capability_id"], sort=True)[
            "qualified"
        ]
        .mean()
        .unstack(fill_value=0.0)
    )
    fig, axis = plt.subplots(figsize=(10, 5.5))
    image = axis.imshow(rates.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
    axis.set_xticks(range(len(rates.columns)), labels=[
        short_capability(value) for value in rates.columns
    ], rotation=35, ha="right")
    axis.set_yticks(range(len(rates.index)), labels=[
        short_profile(value) for value in rates.index
    ])
    axis.set_title("History-only mechanism-gate qualification rate")
    fig.colorbar(image, ax=axis, label="qualification rate")
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_1_gate_support_heatmap")

    cell = scope_outputs["inclusive"]["concordance"]
    local = cell[
        cell["predictor_id"] == "v2_dataset_local_capability"
    ]
    heat = local.pivot(
        index="profile_id",
        columns="capability_id",
        values="kendall_tau_b",
    )
    fig, axis = plt.subplots(figsize=(10, 5.5))
    image = axis.imshow(
        heat.to_numpy(dtype=float),
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )
    axis.set_xticks(range(len(heat.columns)), labels=[
        short_capability(value) for value in heat.columns
    ], rotation=35, ha="right")
    axis.set_yticks(range(len(heat.index)), labels=[
        short_profile(value) for value in heat.index
    ])
    axis.set_title("Synthetic–real model-rank concordance by gated cell")
    fig.colorbar(image, ax=axis, label="Kendall tau-b")
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_2_cell_concordance_heatmap")

    scope_rows = []
    for scope, outputs in scope_outputs.items():
        summary = outputs["predictor_summary"]
        for row in summary.itertuples(index=False):
            scope_rows.append(
                (
                    scope,
                    str(row.predictor_id),
                    float(row.kendall_tau_b),
                    float(row.kendall_tau_b_ci_low),
                    float(row.kendall_tau_b_ci_high),
                )
            )
    labels = [f"{scope}\\n{predictor}" for scope, predictor, *_ in scope_rows]
    estimates = np.asarray([row[2] for row in scope_rows], dtype=float)
    lows = np.asarray([row[3] for row in scope_rows], dtype=float)
    highs = np.asarray([row[4] for row in scope_rows], dtype=float)
    fig, axis = plt.subplots(figsize=(12, 5.5))
    positions = np.arange(len(labels))
    axis.errorbar(
        positions,
        estimates,
        yerr=np.vstack([estimates - lows, highs - estimates]),
        fmt="o",
        capsize=3,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels=labels, rotation=55, ha="right")
    axis.set_ylabel("Family-macro Kendall tau-b")
    axis.set_title("Predictor comparison across frozen gate scopes")
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_3_predictor_comparison")

    support = qualified_cells.pivot(
        index="profile_id",
        columns="capability_id",
        values="inclusive_selected_count",
    )
    fig, axis = plt.subplots(figsize=(10, 5.5))
    image = axis.imshow(
        support.fillna(0).to_numpy(dtype=float),
        aspect="auto",
        cmap="viridis",
    )
    axis.set_xticks(range(len(support.columns)), labels=[
        short_capability(value) for value in support.columns
    ], rotation=35, ha="right")
    axis.set_yticks(range(len(support.index)), labels=[
        short_profile(value) for value in support.index
    ])
    axis.set_title("Selected validation tasks per confirmatory cell")
    fig.colorbar(image, ax=axis, label="task count")
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_4_cell_support")


def render_analysis_report(
    summary: dict[str, Any],
    scope_outputs: dict[str, dict[str, pd.DataFrame]],
) -> str:
    primary = summary["primary_endpoint"]
    lines = [
        "# Paper E4-v3：机制门控后的合成—真实迁移",
        "",
        "本轮使用此前未推理的 GIFT validation horizon；真实窗口能力资格只由"
        " 504 点 context 内四个伪未来决定。",
        "",
        "## 主结果",
        "",
        (
            "- Inclusive gate、dataset-local capability predictor 的 "
            f"family-macro Kendall τ-b = {primary['estimate']:.4f} "
            f"(95% CI [{primary['ci_low']:.4f}, "
            f"{primary['ci_high']:.4f}])。"
        ),
        (
            "- Wrong-label exact permutation p = "
            f"{summary['wrong_label_permutation']['p_value']:.6f}。"
        ),
        f"- 唯一真实推理 task：{summary['real_task_count']}。",
        (
            "- `nonlinear_persistence` 因合成独立审计未通过 gate 校准约束，"
            "不进入确认性真实迁移主表。"
        ),
        (
            "- `time_varying_seasonality` 与 `regime_switching` 的 gate 已通过"
            "合成独立审计，但当前冻结 GIFT slices 中没有达到 cell 最小支持量，"
            "因此本轮不对其真实迁移作结论。"
        ),
        "",
        "## 冻结稳健性范围",
        "",
        "| scope | cells | families | capability τ-b | scalar τ-b | Δ vs scalar | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, row in summary["scope_results"].items():
        lines.append(
            f"| `{scope}` | {row['cell_count']} | {row['family_count']} | "
            f"{row['primary_kendall_tau_b']:.4f} | "
            f"{row['scalar_kendall_tau_b']:.4f} | "
            f"{row['primary_delta_vs_scalar']:.4f} | "
            f"{row['primary_spearman_rho']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 分能力结果（inclusive）",
            "",
            "| capability | families | Kendall τ-b | Spearman ρ | pair direction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    capability = scope_outputs["inclusive"]["capability"]
    capability = capability[
        capability["predictor_id"] == "v2_dataset_local_capability"
    ]
    for row in capability.itertuples(index=False):
        lines.append(
            f"| `{row.capability_id}` | {int(row.family_count)} | "
            f"{float(row.kendall_tau_b):.4f} | "
            f"{float(row.spearman_rho):.4f} | "
            f"{float(row.pair_direction_concordance):.4f} |"
        )
    lines.extend(
        [
            "",
            "Gate 通过表示机制对齐的预测行为，不表示识别出真实系统的因果生成机制。",
            "",
        ]
    )
    return "\n".join(lines)


def render_paper_tables(
    scope_outputs: dict[str, dict[str, pd.DataFrame]],
) -> str:
    lines = ["# E4-v3 paper-ready tables", ""]
    for scope, outputs in scope_outputs.items():
        lines.extend(
            [
                f"## {scope}",
                "",
                "| predictor | Kendall τ-b | 95% CI | Spearman ρ | pair direction | Δτ vs scalar |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in outputs["predictor_summary"].itertuples(index=False):
            lines.append(
                f"| `{row.predictor_id}` | "
                f"{float(row.kendall_tau_b):.4f} | "
                f"[{float(row.kendall_tau_b_ci_low):.4f}, "
                f"{float(row.kendall_tau_b_ci_high):.4f}] | "
                f"{float(row.spearman_rho):.4f} | "
                f"{float(row.pair_direction_concordance):.4f} | "
                f"{float(row.kendall_tau_b_delta_vs_scalar):.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def save_figure(figure: Any, base_path: Path) -> None:
    for extension in ("png", "svg", "pdf"):
        figure.savefig(
            base_path.with_suffix(f".{extension}"),
            dpi=220 if extension == "png" else None,
            bbox_inches="tight",
        )
    import matplotlib.pyplot as plt

    plt.close(figure)


def short_profile(value: str) -> str:
    return (
        str(value)
        .replace("gift_", "")
        .replace("_h_504ctx_48h", "")
    )


def short_capability(value: str) -> str:
    return {
        "trend": "trend",
        "multi_seasonal": "multi-season",
        "time_varying_seasonality": "TV-season",
        "regime_switching": "regime",
        "nonlinear_persistence": "nonlinear",
        "predictable_intermittency": "intermittent",
    }.get(str(value), str(value))


def write_final_manifest(
    output_dir: Path,
    *,
    config: dict[str, Any],
) -> None:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if (
            not path.is_file()
            or path.name == "manifest.json"
            or path.is_symlink()
        ):
            continue
        files.append(
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "paper_experiment_manifest.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_id": EXPERIMENT_ID,
            "selection_manifest_sha256": config[
                "selection_manifest_sha256"
            ],
            "files": files,
        },
    )


def reject_existing_predictions(output_dir: Path) -> None:
    prediction_dir = output_dir / "predictions"
    if prediction_dir.exists() and any(prediction_dir.glob("*.jsonl")):
        raise ValueError(
            "E4-v3 prepare refuses to overwrite existing model predictions"
        )


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.10g")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def last_commit_touching(path: Path) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative_path(path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
