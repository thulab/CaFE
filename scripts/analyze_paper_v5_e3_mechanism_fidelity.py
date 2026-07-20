#!/usr/bin/env python3
"""Build formal Paper v7 E3 mechanism-aware profiles from sealed E2 output."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for import_path in (BACKEND_DIR,):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.services.synthetic_mechanism_fidelity import (  # noqa: E402
    SCHEMA_VERSION as METRIC_SCHEMA_VERSION,
    capability_score,
    evaluate_mechanism_fidelity,
)


SCHEMA_VERSION = "paper_v7_e3_mechanism_fidelity.v1"
EXPERIMENT_ID = "E3_mechanism_fidelity"
DEFAULT_E2_DIR = REPO_ROOT / "runtime/paper_exp/v7/E2_dynamic_stability"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runtime/paper_exp/v7/E3_mechanism_fidelity/formal_analysis"
)
DEFAULT_PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/"
    "2026-07-21-paper-v7-structured-dataset-expansion-protocol.md"
)
DEFAULT_MODELS = (
    "Timer-3.5",
    "Timer-3.0",
    "Chronos-2",
    "moirai2",
    "toto2.0",
    "timesfm2.5",
    "tirex2",
    "tabpfn-ts3",
)
BLIND_REFERENCE_MODEL = "naive"
MAX_CONTEXT_LENGTH = 504
HORIZON = 48
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20260720
DEFAULT_CI_LEVEL = 0.95
DEFAULT_EQUIVALENCE_MARGINS = (0.01, 0.02, 0.05)
DEFAULT_PRIMARY_EQUIVALENCE_MARGIN = 0.02
DEFAULT_SPLIT_SIZE = 160
V7_INPUT_ADAPTATION_POLICY_ID = "paper-v7-input-adaptation-v1"
PROFILE_KEYS = ["dataset_id", "task_id", "capability_id"]
METRIC_SPECS = {
    "mase": {
        "column": "mase_mean",
        "higher_is_better": False,
        "effect_scale": "relative",
    },
    "mechanism": {
        "column": "mechanism_fidelity_score",
        "higher_is_better": True,
        "effect_scale": "absolute",
    },
    "ability": {
        "column": "ability_score",
        "higher_is_better": True,
        "effect_scale": "absolute",
    },
}
SAMPLE_COLUMNS = (
    "model_id",
    "dataset_id",
    "task_id",
    "capability_id",
    "intensity",
    "paired_group_id",
    "master_sample_id",
    "round_index",
    "sample_index",
    "analysis_pool_index",
    "analysis_block_id",
    "context_length",
    "oracle_mase",
    "formal_score_eligible",
    "unsupported_reason",
    "mechanism_fidelity_score",
    "detection_score",
    "timing_score",
    "magnitude_score",
    "selectivity_score",
    "truth_mechanism_strength",
    "forecast_mechanism_strength",
    "point_mae",
    "input_execution_mode",
    "target_input_mode",
    "covariate_input_mode",
    "input_adaptation_policy_id",
    "counterfactual_mode",
    "counterfactual_effect_mae",
    "counterfactual_http_request_count",
    "component_scores_json",
    "diagnostics_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate mechanism-aligned forecast behavior for Paper v7 E3 "
            "without regenerating samples or rerunning ordinary predictions."
        )
    )
    parser.add_argument("--e2-dir", type=Path, default=DEFAULT_E2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--max-paired-groups-per-cell",
        type=int,
        default=0,
        help=(
            "Optional cap per dataset/capability. Each selected paired group "
            "retains all five intensity levels. Use 0 for all groups."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Optional dataset filter. By default every supported "
            "dataset/task/capability cell present in the sealed generation "
            "shards is analyzed."
        ),
    )
    parser.add_argument(
        "--covariate-ablation-predictions-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing one <safe model id>.jsonl file. "
            "Rows must provide master_sample_id, context_length, forecast and "
            "ablation=future_covariates_zero."
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=DEFAULT_CI_LEVEL,
    )
    parser.add_argument(
        "--equivalence-margins",
        type=float,
        nargs="+",
        default=list(DEFAULT_EQUIVALENCE_MARGINS),
    )
    parser.add_argument(
        "--primary-equivalence-margin",
        type=float,
        default=DEFAULT_PRIMARY_EQUIVALENCE_MARGIN,
    )
    parser.add_argument(
        "--split-size",
        type=int,
        default=DEFAULT_SPLIT_SIZE,
        help=(
            "Number of paired groups in each deterministic reliability half. "
            "Every selected cell must contain exactly twice this count; the "
            "v7 default therefore validates and splits all 320 groups into "
            "two mutually exclusive blocks of 160. Use 0 to skip the audit."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_filename(model_id: str) -> str:
    return (
        str(model_id)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def prediction_path(e2_dir: Path, model_id: str) -> Path:
    path = e2_dir / "predictions" / f"{safe_filename(model_id)}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing prediction file: {path}")
    return path


def oracle_path(e2_dir: Path, model_id: str) -> Path:
    path = e2_dir / "oracle_sample_scores" / f"{safe_filename(model_id)}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing oracle score file: {path}")
    return path


def _resolve_frozen_path(path_value: str, *, relative_to: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repository_path = REPO_ROOT / path
    if repository_path.is_file():
        return repository_path
    return relative_to / path


def support_artifact_path(e2_dir: Path) -> Path:
    config_path = e2_dir / "generation_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"missing generation config for E3 cell discovery: {config_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    support_record = (config.get("suite_files") or {}).get("support")
    if not isinstance(support_record, dict) or not support_record.get("path"):
        raise ValueError(
            "generation_config.json does not freeze a support artifact"
        )
    path = _resolve_frozen_path(
        str(support_record["path"]),
        relative_to=e2_dir,
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen support artifact: {path}")
    expected_sha256 = support_record.get("sha256")
    if expected_sha256 and sha256_file(path) != str(expected_sha256):
        raise ValueError("frozen support artifact checksum mismatch")
    return path


def sample_shards_by_cell(
    e2_dir: Path,
) -> dict[tuple[str, str, str], Path]:
    shard_dir = e2_dir / "sample_shards"
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"missing generation shards: {shard_dir}")
    result: dict[tuple[str, str, str], Path] = {}
    for path in sorted(shard_dir.glob("*.jsonl")):
        first_row: dict[str, Any] | None = None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    first_row = json.loads(line)
                    break
        if first_row is None:
            raise ValueError(f"generation shard is empty: {path}")
        key = (
            str(first_row["dataset_id"]),
            str(first_row["task_id"]),
            str(first_row["capability_id"]),
        )
        if key in result:
            raise ValueError(f"duplicate generation shards for cell {key}")
        result[key] = path
    if not result:
        raise ValueError("generation shard set is empty")
    return result


def discover_supported_cells(
    e2_dir: Path,
    *,
    dataset_ids: list[str] | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Discover the sealed E3 cell universe without a hand-written mapping."""

    support_path = support_artifact_path(e2_dir)
    support = json.loads(support_path.read_text(encoding="utf-8"))
    supported = {
        (
            str(cell["dataset_id"]),
            str(cell["task_id"]),
            str(cell["capability_id"]),
        )
        for cell in support.get("cells", [])
        if cell.get("status") == "supported"
        or cell.get("supported") is True
    }
    if not supported:
        raise ValueError("frozen support artifact has no supported cells")
    shard_cells = set(sample_shards_by_cell(e2_dir))
    if shard_cells != supported:
        missing_shards = sorted(supported - shard_cells)
        unsupported_shards = sorted(shard_cells - supported)
        raise ValueError(
            "generation shards and frozen supported cells disagree: "
            f"missing_shards={missing_shards}, "
            f"unsupported_shards={unsupported_shards}"
        )
    if dataset_ids is not None:
        requested = {str(value) for value in dataset_ids}
        available = {cell[0] for cell in supported}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(
                "datasets are absent from the sealed supported cells: "
                + ", ".join(unknown)
            )
        supported = {cell for cell in supported if cell[0] in requested}
    if not supported:
        raise ValueError("supported cell selection is empty")
    return tuple(sorted(supported))


def load_selected_samples(
    e2_dir: Path,
    *,
    supported_cells: tuple[tuple[str, str, str], ...],
    max_paired_groups_per_cell: int,
) -> dict[str, dict[str, Any]]:
    if max_paired_groups_per_cell < 0:
        raise ValueError("max_paired_groups_per_cell cannot be negative")
    shard_paths = sample_shards_by_cell(e2_dir)
    selected: dict[str, dict[str, Any]] = {}
    for cell_key in supported_cells:
        if cell_key not in shard_paths:
            raise FileNotFoundError(
                f"missing sample shard for supported cell {cell_key}"
            )
        rows = [
            json.loads(line)
            for line in shard_paths[cell_key].open(encoding="utf-8")
            if line.strip()
        ]
        observed_cells = {
            (
                str(row["dataset_id"]),
                str(row["task_id"]),
                str(row["capability_id"]),
            )
            for row in rows
        }
        if observed_cells != {cell_key}:
            raise ValueError(
                f"generation shard mixes cells: {observed_cells}"
            )
        group_order = sorted(
            {
                (
                    int(row.get("round_index", 0)),
                    int(row["sample_index"]),
                    str(row["paired_group_id"]),
                )
                for row in rows
            }
        )
        if max_paired_groups_per_cell:
            group_order = group_order[:max_paired_groups_per_cell]
        group_ids = {item[2] for item in group_order}
        cell_rows = [
            row for row in rows if str(row["paired_group_id"]) in group_ids
        ]
        counts = pd.Series(
            [str(row["paired_group_id"]) for row in cell_rows]
        ).value_counts()
        if counts.empty or not (counts == 5).all():
            raise ValueError(
                f"{cell_key} does not retain five paired intensity levels"
            )
        for row in cell_rows:
            master_id = str(row["master_sample_id"])
            if master_id in selected:
                raise ValueError(f"duplicate master sample id: {master_id}")
            selected[master_id] = row
    if not selected:
        raise ValueError("sample selection is empty")
    return selected


def load_oracle_selection(
    e2_dir: Path,
    *,
    model_ids: list[str],
    samples: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], pd.DataFrame]:
    master_sample_ids = set(samples)
    desired_by_cell: dict[tuple[str, str, str], set[str]] = {}
    for master_id, sample in samples.items():
        key = (
            str(sample["dataset_id"]),
            str(sample["task_id"]),
            str(sample["capability_id"]),
        )
        desired_by_cell.setdefault(key, set()).add(master_id)
    selections: dict[str, dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for model_id in [*model_ids, BLIND_REFERENCE_MODEL]:
        model_rows: dict[str, dict[str, Any]] = {}
        with oracle_path(e2_dir, model_id).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                master_id = str(row["master_sample_id"])
                if master_id not in master_sample_ids:
                    continue
                model_rows[master_id] = row
                rows.append(
                    {
                        "model_id": model_id,
                        "dataset_id": str(row["dataset_id"]),
                        "task_id": str(row["task_id"]),
                        "capability_id": str(row["capability_id"]),
                        "intensity": int(row["intensity"]),
                        "master_sample_id": master_id,
                        "paired_group_id": str(row["paired_group_id"]),
                        "oracle_mase": float(row["oracle_mase"]),
                    }
                )
        if model_id == BLIND_REFERENCE_MODEL and set(model_rows) != master_sample_ids:
            missing = len(master_sample_ids - set(model_rows))
            raise ValueError(
                f"blind reference oracle selection misses {missing} samples"
            )
        if model_id != BLIND_REFERENCE_MODEL:
            if not model_rows:
                raise ValueError(
                    f"{model_id} has no compatible E3 capability cells"
                )
            observed = set(model_rows)
            for key, desired in desired_by_cell.items():
                count = len(observed & desired)
                if count not in {0, len(desired)}:
                    raise ValueError(
                        f"{model_id} partially covers E3 cell {key}: "
                        f"{count}/{len(desired)}"
                    )
        if model_id != BLIND_REFERENCE_MODEL:
            selections[model_id] = model_rows
    return selections, pd.DataFrame(rows)


def load_selected_predictions(
    e2_dir: Path,
    *,
    model_id: str,
    oracle_selection: dict[str, dict[str, Any]],
    require_input_adaptation: bool = False,
) -> dict[str, dict[str, Any]]:
    view_to_master = {
        str(row["oracle_view_id"]): master_id
        for master_id, row in oracle_selection.items()
    }
    selected: dict[str, dict[str, Any]] = {}
    with prediction_path(e2_dir, model_id).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            master_id = view_to_master.get(str(row["view_id"]))
            if master_id is None:
                continue
            if require_input_adaptation:
                plan = row.get("input_adaptation")
                if not isinstance(plan, dict):
                    raise ValueError(
                        f"{model_id}/{row.get('view_id')} lacks v7 "
                        "input_adaptation provenance"
                    )
                if str(plan.get("policy_id")) != (
                    V7_INPUT_ADAPTATION_POLICY_ID
                ):
                    raise ValueError(
                        f"{model_id}/{row.get('view_id')} has an unexpected "
                        "input adaptation policy"
                    )
            selected[master_id] = row
            if len(selected) == len(view_to_master):
                break
    if set(selected) != set(oracle_selection):
        missing = len(set(oracle_selection) - set(selected))
        raise ValueError(f"{model_id} predictions miss {missing} oracle views")
    return selected


def input_adaptation_provenance(
    prediction: dict[str, Any],
) -> dict[str, Any]:
    plan = prediction.get("input_adaptation")
    if not isinstance(plan, dict):
        return {
            "input_execution_mode": "legacy_native",
            "target_input_mode": "legacy_native",
            "covariate_input_mode": "legacy_native",
            "input_adaptation_policy_id": "legacy_native_only",
            "adapted": False,
        }
    adapted = bool(plan.get("adapted"))
    return {
        "input_execution_mode": "adapted" if adapted else "native",
        "target_input_mode": str(plan.get("target_mode", "unknown")),
        "covariate_input_mode": str(
            plan.get("covariate_mode", "unknown")
        ),
        "input_adaptation_policy_id": str(
            plan.get("policy_id", "unknown")
        ),
        "adapted": adapted,
    }


def expected_counterfactual_mode(
    intact_prediction: dict[str, Any],
) -> str:
    provenance = input_adaptation_provenance(intact_prediction)
    if provenance["covariate_input_mode"] == "omitted_unsupported":
        return "reuse_intact_forecast_covariates_omitted"
    return "native_future_covariate_ablation_http"


def load_covariate_ablation_predictions(
    directory: Path | None,
    *,
    model_ids: list[str],
    samples: dict[str, dict[str, Any]],
    oracle_selections: dict[str, dict[str, dict[str, Any]]],
    intact_predictions: (
        dict[str, dict[str, dict[str, Any]]] | None
    ) = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    if directory is None:
        return {}
    desired_ids = {
        master_id
        for master_id, sample in samples.items()
        if str(sample["capability_id"]) == "covariate_response"
    }
    if not desired_ids:
        return {}
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for model_id in model_ids:
        model_desired_ids = desired_ids & set(oracle_selections[model_id])
        if not model_desired_ids:
            continue
        path = directory / f"{safe_filename(model_id)}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing covariate ablation predictions: {path}"
            )
        model_rows: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                master_id = str(row.get("master_sample_id", ""))
                if master_id not in model_desired_ids:
                    continue
                if str(row.get("ablation")) != "future_covariates_zero":
                    raise ValueError(
                        "covariate ablation rows must declare "
                        "ablation=future_covariates_zero"
                    )
                expected_context = int(
                    oracle_selections[model_id][master_id]["oracle_context"]
                )
                if int(row.get("context_length", -1)) != expected_context:
                    continue
                if intact_predictions is not None:
                    intact = intact_predictions[model_id][master_id]
                    expected_mode = expected_counterfactual_mode(intact)
                    if str(row.get("counterfactual_mode")) != expected_mode:
                        raise ValueError(
                            f"{model_id}/{master_id} has counterfactual mode "
                            f"{row.get('counterfactual_mode')!r}, expected "
                            f"{expected_mode!r}"
                        )
                    if expected_mode.startswith("reuse_"):
                        if int(
                            row.get(
                                "counterfactual_http_request_count",
                                -1,
                            )
                        ) != 0:
                            raise ValueError(
                                "a reused counterfactual cannot issue HTTP "
                                "requests"
                            )
                        if not np.array_equal(
                            np.asarray(row["forecast"], dtype=float),
                            np.asarray(intact["forecast"], dtype=float),
                        ):
                            raise ValueError(
                                "covariate-omitted counterfactual must reuse "
                                "the intact forecast exactly"
                            )
                model_rows[master_id] = row
        if set(model_rows) != model_desired_ids:
            missing = len(model_desired_ids - set(model_rows))
            raise ValueError(
                f"{model_id} covariate ablation misses {missing} oracle views"
            )
        result[model_id] = model_rows
    return result


def covariate_ablation_input_record(
    directory: Path | None,
    *,
    counterfactual_predictions: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    if directory is None:
        return None
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing covariate ablation manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("ablation")) != "future_covariates_zero":
        raise ValueError("covariate ablation manifest has a wrong intervention")
    expected_models = sorted(counterfactual_predictions)
    manifest_models = sorted(
        str(value) for value in manifest.get("models", [])
    )
    if manifest_models != expected_models:
        raise ValueError(
            "covariate ablation manifest model set does not match loaded files"
        )
    files: dict[str, Any] = {}
    for model_id in expected_models:
        path = directory / f"{safe_filename(model_id)}.jsonl"
        files[path.name] = {
            "model_id": model_id,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "directory": str(directory.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "files": files,
    }


def context_view(
    sample: dict[str, Any],
    *,
    context_length: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    master_target = np.asarray(sample["target"], dtype=float)
    expected = MAX_CONTEXT_LENGTH + int(sample["horizon"])
    if master_target.ndim != 2 or len(master_target) != expected:
        raise ValueError("master sample target has an unexpected shape")
    start = MAX_CONTEXT_LENGTH - int(context_length)
    suffix = master_target[start:]
    if str(sample["capability_id"]) == "hierarchical_coherence":
        target = standardize_hierarchy_by_context(suffix, context_length)
    else:
        target = standardize_by_context(suffix, context_length)
    raw_covariates = sample.get("covariates")
    covariates = None
    if raw_covariates is not None:
        suffix_covariates = np.asarray(raw_covariates, dtype=float)[start:]
        covariates = normalize_covariates(
            suffix_covariates,
            context_length,
        )
    return target, covariates


def standardize_by_context(
    values: np.ndarray,
    context_length: int,
) -> np.ndarray:
    context = values[:context_length]
    mean = context.mean(axis=0, keepdims=True)
    scale = context.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (values - mean) / scale


def standardize_hierarchy_by_context(
    values: np.ndarray,
    context_length: int,
) -> np.ndarray:
    context = values[:context_length]
    centered = values - context.mean(axis=0, keepdims=True)
    scale = float(np.std(context[:, 0]))
    if scale <= 1e-6:
        scale = float(np.mean(np.std(context, axis=0)))
    if scale <= 1e-6:
        scale = 1.0
    return centered / scale


def normalize_covariates(
    values: np.ndarray,
    context_length: int,
) -> np.ndarray:
    normalized = np.asarray(values, dtype=float).copy()
    for index in range(normalized.shape[1]):
        column = normalized[:context_length, index]
        if set(np.unique(normalized[:, index])).issubset({0.0, 1.0}):
            continue
        mean = float(column.mean())
        scale = float(column.std())
        if scale <= 1e-12:
            scale = 1.0
        normalized[:, index] = (normalized[:, index] - mean) / scale
    return normalized


def evaluate_selected_predictions(
    samples: dict[str, dict[str, Any]],
    oracle_selections: dict[str, dict[str, dict[str, Any]]],
    predictions: dict[str, dict[str, dict[str, Any]]],
    counterfactual_predictions: (
        dict[str, dict[str, dict[str, Any]]] | None
    ) = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, model_predictions in predictions.items():
        for master_id, prediction in model_predictions.items():
            sample = samples[master_id]
            oracle = oracle_selections[model_id][master_id]
            provenance = input_adaptation_provenance(prediction)
            context_length = int(oracle["oracle_context"])
            target_view, covariate_view = context_view(
                sample,
                context_length=context_length,
            )
            history = target_view[:context_length]
            target_future = np.asarray(prediction["target_future"], dtype=float)
            if not np.allclose(
                target_view[context_length:],
                target_future,
                rtol=1e-8,
                atol=1e-8,
            ):
                raise ValueError(
                    f"{master_id} prediction target does not match rebuilt view"
                )
            counterfactual_row = (
                counterfactual_predictions[model_id][master_id]
                if (
                    counterfactual_predictions
                    and model_id in counterfactual_predictions
                    and master_id in counterfactual_predictions[model_id]
                )
                else None
            )
            counterfactual_forecast = (
                np.asarray(counterfactual_row["forecast"], dtype=float)
                if counterfactual_row is not None
                else None
            )
            result = evaluate_mechanism_fidelity(
                capability_id=str(sample["capability_id"]),
                history=history,
                target_future=target_future,
                forecast=np.asarray(prediction["forecast"], dtype=float),
                season_length=int(sample["season_length"]),
                latent_params=dict(sample["generation_metadata"]),
                intensity=int(sample["intensity"]),
                forecast_start_index=MAX_CONTEXT_LENGTH,
                covariates=covariate_view,
                counterfactual_forecast=counterfactual_forecast,
            )
            rows.append(
                {
                    "model_id": model_id,
                    "dataset_id": str(sample["dataset_id"]),
                    "task_id": str(sample["task_id"]),
                    "capability_id": str(sample["capability_id"]),
                    "intensity": int(sample["intensity"]),
                    "paired_group_id": str(sample["paired_group_id"]),
                    "master_sample_id": master_id,
                    "round_index": int(sample["round_index"]),
                    "sample_index": int(sample["sample_index"]),
                    "analysis_pool_index": int(
                        sample.get("analysis_pool_index", -1)
                    ),
                    "analysis_block_id": str(
                        sample.get("analysis_block_id", "")
                    ),
                    "context_length": context_length,
                    "oracle_mase": float(oracle["oracle_mase"]),
                    "formal_score_eligible": bool(
                        result["formal_score_eligible"]
                    ),
                    "unsupported_reason": result["unsupported_reason"],
                    "mechanism_fidelity_score": float(
                        result["mechanism_fidelity_score"]
                    ),
                    "detection_score": result["detection_score"],
                    "timing_score": result["timing_score"],
                    "magnitude_score": result["magnitude_score"],
                    "selectivity_score": result["selectivity_score"],
                    "truth_mechanism_strength": float(
                        result["truth_mechanism_strength"]
                    ),
                    "forecast_mechanism_strength": float(
                        result["forecast_mechanism_strength"]
                    ),
                    "point_mae": float(result["point_mae"]),
                    **{
                        key: provenance[key]
                        for key in (
                            "input_execution_mode",
                            "target_input_mode",
                            "covariate_input_mode",
                            "input_adaptation_policy_id",
                        )
                    },
                    "counterfactual_mode": (
                        str(counterfactual_row["counterfactual_mode"])
                        if counterfactual_row is not None
                        else "not_applicable"
                    ),
                    "counterfactual_effect_mae": (
                        float(
                            np.mean(
                                np.abs(
                                    np.asarray(
                                        prediction["forecast"],
                                        dtype=float,
                                    )
                                    - counterfactual_forecast
                                )
                            )
                        )
                        if counterfactual_forecast is not None
                        else math.nan
                    ),
                    "counterfactual_http_request_count": (
                        int(
                            counterfactual_row.get(
                                "counterfactual_http_request_count",
                                0,
                            )
                        )
                        if counterfactual_row is not None
                        else 0
                    ),
                    "component_scores_json": json.dumps(
                        result["component_scores"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "diagnostics_json": json.dumps(
                        result["diagnostics"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
    frame = pd.DataFrame(rows, columns=SAMPLE_COLUMNS)
    if frame.empty:
        raise ValueError("mechanism evaluation produced no rows")
    return frame


def _single_value(values: pd.Series) -> Any:
    unique = values.drop_duplicates().tolist()
    if len(unique) != 1:
        raise ValueError(f"expected one provenance value, found {unique}")
    return unique[0]


def intensity_cell_scores(
    sample_scores: pd.DataFrame,
    oracle_scores: pd.DataFrame,
) -> pd.DataFrame:
    sample_scores = sample_scores.copy()
    provenance_defaults = {
        "input_execution_mode": "legacy_native",
        "target_input_mode": "legacy_native",
        "covariate_input_mode": "legacy_native",
        "input_adaptation_policy_id": "legacy_native_only",
        "counterfactual_mode": "not_applicable",
        "counterfactual_effect_mae": math.nan,
        "counterfactual_http_request_count": 0,
    }
    for column, default in provenance_defaults.items():
        if column not in sample_scores:
            sample_scores[column] = default
    keys = [
        "model_id",
        "dataset_id",
        "task_id",
        "capability_id",
        "intensity",
    ]
    cells = (
        sample_scores.groupby(keys, sort=True)
        .agg(
            sample_count=("master_sample_id", "size"),
            oracle_mase_mean=("oracle_mase", "mean"),
            mechanism_fidelity_mean=("mechanism_fidelity_score", "mean"),
            mechanism_fidelity_std=("mechanism_fidelity_score", "std"),
            detection_mean=("detection_score", "mean"),
            timing_mean=("timing_score", "mean"),
            magnitude_mean=("magnitude_score", "mean"),
            selectivity_mean=("selectivity_score", "mean"),
            formal_score_eligible=("formal_score_eligible", "all"),
            input_execution_mode=(
                "input_execution_mode",
                _single_value,
            ),
            target_input_mode=("target_input_mode", _single_value),
            covariate_input_mode=(
                "covariate_input_mode",
                _single_value,
            ),
            input_adaptation_policy_id=(
                "input_adaptation_policy_id",
                _single_value,
            ),
            counterfactual_mode=(
                "counterfactual_mode",
                _single_value,
            ),
            counterfactual_effect_mae=(
                "counterfactual_effect_mae",
                "mean",
            ),
            counterfactual_http_request_count=(
                "counterfactual_http_request_count",
                "sum",
            ),
        )
        .reset_index()
    )
    blind = oracle_scores[
        oracle_scores["model_id"] == BLIND_REFERENCE_MODEL
    ]
    blind = (
        blind.groupby(keys[1:], sort=True)["oracle_mase"]
        .mean()
        .rename("blind_mase_mean")
        .reset_index()
    )
    cells = cells.merge(blind, on=keys[1:], how="left", validate="many_to_one")
    if cells["blind_mase_mean"].isna().any():
        raise ValueError("blind reference MASE is incomplete")
    cells["point_accuracy_gate"] = np.minimum(
        1.0,
        cells["blind_mase_mean"] / cells["oracle_mase_mean"],
    )
    cells["ability_score"] = [
        (
            capability_score(
                mechanism_fidelity_score=float(row.mechanism_fidelity_mean),
                model_point_loss=float(row.oracle_mase_mean),
                blind_point_loss=float(row.blind_mase_mean),
            )
            if bool(row.formal_score_eligible)
            else math.nan
        )
        for row in cells.itertuples(index=False)
    ]
    rank_keys = ["dataset_id", "task_id", "capability_id", "intensity"]
    cells["mase_rank"] = cells.groupby(rank_keys, sort=False)[
        "oracle_mase_mean"
    ].rank(method="average", ascending=True)
    cells["mechanism_rank"] = cells["mechanism_fidelity_mean"].where(
        cells["formal_score_eligible"]
    ).groupby(
        [cells[key] for key in rank_keys],
        sort=False,
    ).rank(method="average", ascending=False)
    cells["ability_rank"] = cells["ability_score"].groupby(
        [cells[key] for key in rank_keys],
        sort=False,
    ).rank(method="average", ascending=False)
    return cells


def paired_dose_response_scores(sample_scores: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_id",
        "dataset_id",
        "task_id",
        "capability_id",
        "paired_group_id",
    ]
    rows: list[dict[str, Any]] = []
    for group_key, group in sample_scores.groupby(keys, sort=True):
        ordered = group.sort_values("intensity")
        if ordered["intensity"].tolist() != [1, 2, 3, 4, 5]:
            raise ValueError("dose-response group must contain intensities 1..5")
        truth = ordered["truth_mechanism_strength"].to_numpy(dtype=float)
        predicted = ordered["forecast_mechanism_strength"].to_numpy(dtype=float)
        rho = spearman_correlation(truth, predicted)
        normalized_truth = minmax_scale(truth)
        normalized_predicted = minmax_scale(predicted)
        ccc = lin_concordance(normalized_truth, normalized_predicted)
        truth_diff = np.diff(truth)
        pred_diff = np.diff(predicted)
        informative = np.abs(truth_diff) > 1e-12
        monotonic = (
            float(
                np.mean(
                    np.sign(pred_diff[informative])
                    == np.sign(truth_diff[informative])
                )
            )
            if np.any(informative)
            else 0.0
        )
        dose_score = geometric_mean(
            (
                (rho + 1.0) / 2.0,
                (ccc + 1.0) / 2.0,
                monotonic,
            )
        )
        rows.append(
            {
                **dict(zip(keys, group_key, strict=True)),
                "formal_score_eligible": bool(
                    ordered["formal_score_eligible"].all()
                ),
                "dose_spearman_rho": rho,
                "dose_lin_ccc_normalized": ccc,
                "dose_monotonic_accuracy": monotonic,
                "dose_response_score": dose_score,
            }
        )
    return pd.DataFrame(rows)


def capability_profiles(
    cells: pd.DataFrame,
    dose_scores: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["model_id", "dataset_id", "task_id", "capability_id"]
    profiles = (
        cells.groupby(keys, sort=True)
        .agg(
            intensity_count=("intensity", "nunique"),
            sample_count=("sample_count", "sum"),
            mase_mean=("oracle_mase_mean", "mean"),
            blind_mase_mean=("blind_mase_mean", "mean"),
            level_mechanism_fidelity=("mechanism_fidelity_mean", "mean"),
            detection_mean=("detection_mean", "mean"),
            timing_mean=("timing_mean", "mean"),
            magnitude_mean=("magnitude_mean", "mean"),
            selectivity_mean=("selectivity_mean", "mean"),
            formal_score_eligible=("formal_score_eligible", "all"),
            input_execution_mode=(
                "input_execution_mode",
                _single_value,
            ),
            target_input_mode=("target_input_mode", _single_value),
            covariate_input_mode=(
                "covariate_input_mode",
                _single_value,
            ),
            input_adaptation_policy_id=(
                "input_adaptation_policy_id",
                _single_value,
            ),
            counterfactual_mode=(
                "counterfactual_mode",
                _single_value,
            ),
            counterfactual_effect_mae=(
                "counterfactual_effect_mae",
                "mean",
            ),
            counterfactual_http_request_count=(
                "counterfactual_http_request_count",
                "sum",
            ),
        )
        .reset_index()
    )
    dose = (
        dose_scores.groupby(keys, sort=True)
        .agg(
            paired_group_count=("paired_group_id", "size"),
            dose_response_score=("dose_response_score", "mean"),
            dose_spearman_rho=("dose_spearman_rho", "mean"),
            dose_lin_ccc_normalized=("dose_lin_ccc_normalized", "mean"),
            dose_monotonic_accuracy=("dose_monotonic_accuracy", "mean"),
            dose_formal_eligible=("formal_score_eligible", "all"),
        )
        .reset_index()
    )
    profiles = profiles.merge(dose, on=keys, how="left", validate="one_to_one")
    profiles["formal_score_eligible"] &= profiles[
        "dose_formal_eligible"
    ].fillna(False)
    profiles["mechanism_fidelity_score"] = (
        0.7 * profiles["level_mechanism_fidelity"]
        + 0.3 * profiles["dose_response_score"]
    )
    profiles["point_accuracy_gate"] = np.minimum(
        1.0,
        profiles["blind_mase_mean"] / profiles["mase_mean"],
    )
    profiles["ability_score"] = (
        profiles["mechanism_fidelity_score"]
        * profiles["point_accuracy_gate"]
    ).where(profiles["formal_score_eligible"])
    rank_keys = ["dataset_id", "task_id", "capability_id"]
    profiles["mase_rank"] = profiles.groupby(rank_keys, sort=False)[
        "mase_mean"
    ].rank(method="average", ascending=True)
    profiles["mechanism_rank"] = profiles[
        "mechanism_fidelity_score"
    ].where(profiles["formal_score_eligible"]).groupby(
        [profiles[key] for key in rank_keys],
        sort=False,
    ).rank(method="average", ascending=False)
    profiles["ability_rank"] = profiles["ability_score"].groupby(
        [profiles[key] for key in rank_keys],
        sort=False,
    ).rank(method="average", ascending=False)
    return profiles


def model_capability_coverage(
    samples: dict[str, dict[str, Any]],
    oracle_selections: dict[str, dict[str, dict[str, Any]]],
    model_ids: list[str],
    predictions: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    cells = sorted(
        {
            (
                str(sample["dataset_id"]),
                str(sample["task_id"]),
                str(sample["capability_id"]),
            )
            for sample in samples.values()
        }
    )
    desired_counts = {
        key: sum(
            (
                str(sample["dataset_id"]),
                str(sample["task_id"]),
                str(sample["capability_id"]),
            )
            == key
            for sample in samples.values()
        )
        for key in cells
    }
    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        selected = oracle_selections[model_id]
        model_predictions = (
            predictions.get(model_id, {}) if predictions is not None else {}
        )
        for dataset_id, task_id, capability_id in cells:
            cell_key = (dataset_id, task_id, capability_id)
            desired_ids = {
                master_id
                for master_id, sample in samples.items()
                if (
                    str(sample["dataset_id"]),
                    str(sample["task_id"]),
                    str(sample["capability_id"]),
                )
                == cell_key
            }
            observed = sum(
                str(row["dataset_id"]) == dataset_id
                and str(row["task_id"]) == task_id
                and str(row["capability_id"]) == capability_id
                for row in selected.values()
            )
            expected = desired_counts[
                cell_key
            ]
            prediction_rows = [
                model_predictions[master_id]
                for master_id in sorted(desired_ids & set(model_predictions))
            ]
            provenance = [
                input_adaptation_provenance(row)
                for row in prediction_rows
            ]
            native_count = sum(
                item["input_execution_mode"] in {"native", "legacy_native"}
                for item in provenance
            )
            adapted_count = sum(item["adapted"] for item in provenance)
            prediction_count = (
                len(prediction_rows)
                if predictions is not None
                else observed
            )
            if prediction_count == expected:
                execution_modes = {
                    str(item["input_execution_mode"])
                    for item in provenance
                }
                execution_mode = (
                    next(iter(execution_modes))
                    if len(execution_modes) == 1
                    else ("mixed" if execution_modes else "legacy_native")
                )
            else:
                execution_mode = "unsupported"
            target_modes = sorted(
                {str(item["target_input_mode"]) for item in provenance}
            )
            covariate_modes = sorted(
                {str(item["covariate_input_mode"]) for item in provenance}
            )
            policy_ids = sorted(
                {
                    str(item["input_adaptation_policy_id"])
                    for item in provenance
                }
            )
            rows.append(
                {
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "task_id": task_id,
                    "capability_id": capability_id,
                    "supported": prediction_count == expected,
                    "selected_sample_count": observed,
                    "prediction_sample_count": prediction_count,
                    "expected_sample_count": expected,
                    "input_execution_mode": execution_mode,
                    "native_view_count": native_count,
                    "adapted_view_count": adapted_count,
                    "target_input_modes": ",".join(target_modes),
                    "covariate_input_modes": ",".join(covariate_modes),
                    "input_adaptation_policy_ids": ",".join(policy_ids),
                    "covariates_omitted_view_count": sum(
                        item["covariate_input_mode"]
                        == "omitted_unsupported"
                        for item in provenance
                    ),
                    "unsupported_reason": (
                        None
                        if prediction_count == expected
                        else (
                            "original_view_prediction_missing"
                            if predictions is not None
                            else "model_input_contract_unsupported"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def profile_group_components(
    sample_scores: pd.DataFrame,
    oracle_scores: pd.DataFrame,
    dose_scores: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["model_id", *PROFILE_KEYS, "paired_group_id"]
    components = (
        sample_scores.groupby(keys, sort=True)
        .agg(
            round_index=("round_index", "min"),
            sample_index=("sample_index", "min"),
            analysis_pool_index=("analysis_pool_index", "min"),
            analysis_block_id=("analysis_block_id", _single_value),
            intensity_count=("intensity", "nunique"),
            mase_group_mean=("oracle_mase", "mean"),
            level_mechanism_group_mean=(
                "mechanism_fidelity_score",
                "mean",
            ),
            formal_score_eligible=("formal_score_eligible", "all"),
            input_execution_mode=(
                "input_execution_mode",
                _single_value,
            ),
            target_input_mode=("target_input_mode", _single_value),
            covariate_input_mode=(
                "covariate_input_mode",
                _single_value,
            ),
            input_adaptation_policy_id=(
                "input_adaptation_policy_id",
                _single_value,
            ),
            counterfactual_mode=(
                "counterfactual_mode",
                _single_value,
            ),
            counterfactual_effect_mae=(
                "counterfactual_effect_mae",
                "mean",
            ),
            counterfactual_http_request_count=(
                "counterfactual_http_request_count",
                "sum",
            ),
        )
        .reset_index()
    )
    if not (components["intensity_count"] == 5).all():
        raise ValueError("profile components must retain five intensities")
    dose_columns = [
        *keys,
        "dose_response_score",
        "formal_score_eligible",
    ]
    renamed_dose = dose_scores[dose_columns].rename(
        columns={
            "formal_score_eligible": "dose_formal_score_eligible",
        }
    )
    components = components.merge(
        renamed_dose,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if components["dose_response_score"].isna().any():
        raise ValueError("profile components miss dose-response scores")
    blind = oracle_scores[
        oracle_scores["model_id"] == BLIND_REFERENCE_MODEL
    ]
    blind = (
        blind.groupby([*PROFILE_KEYS, "paired_group_id"], sort=True)
        .agg(
            blind_mase_group_mean=("oracle_mase", "mean"),
            blind_intensity_count=("intensity", "nunique"),
        )
        .reset_index()
    )
    if not (blind["blind_intensity_count"] == 5).all():
        raise ValueError("blind profile components must retain five intensities")
    components = components.merge(
        blind,
        on=[*PROFILE_KEYS, "paired_group_id"],
        how="left",
        validate="many_to_one",
    )
    if components["blind_mase_group_mean"].isna().any():
        raise ValueError("profile components miss blind MASE")
    components["formal_score_eligible"] &= components[
        "dose_formal_score_eligible"
    ]
    return components


def aggregate_profile_components(
    components: pd.DataFrame,
    *,
    bank_id: str,
) -> pd.DataFrame:
    keys = ["model_id", *PROFILE_KEYS]
    profiles = (
        components.groupby(keys, sort=True)
        .agg(
            paired_group_count=("paired_group_id", "size"),
            mase_mean=("mase_group_mean", "mean"),
            blind_mase_mean=("blind_mase_group_mean", "mean"),
            level_mechanism_fidelity=(
                "level_mechanism_group_mean",
                "mean",
            ),
            dose_response_score=("dose_response_score", "mean"),
            formal_score_eligible=("formal_score_eligible", "all"),
            input_execution_mode=(
                "input_execution_mode",
                _single_value,
            ),
            target_input_mode=("target_input_mode", _single_value),
            covariate_input_mode=(
                "covariate_input_mode",
                _single_value,
            ),
            input_adaptation_policy_id=(
                "input_adaptation_policy_id",
                _single_value,
            ),
            counterfactual_mode=(
                "counterfactual_mode",
                _single_value,
            ),
            counterfactual_effect_mae=(
                "counterfactual_effect_mae",
                "mean",
            ),
            counterfactual_http_request_count=(
                "counterfactual_http_request_count",
                "sum",
            ),
        )
        .reset_index()
    )
    profiles["bank_id"] = bank_id
    profiles["sample_count"] = 5 * profiles["paired_group_count"]
    profiles["mechanism_fidelity_score"] = (
        0.7 * profiles["level_mechanism_fidelity"]
        + 0.3 * profiles["dose_response_score"]
    )
    profiles["point_accuracy_gate"] = np.minimum(
        1.0,
        profiles["blind_mase_mean"] / profiles["mase_mean"],
    )
    profiles["ability_score"] = (
        profiles["mechanism_fidelity_score"]
        * profiles["point_accuracy_gate"]
    ).where(profiles["formal_score_eligible"])
    rank_keys = ["bank_id", *PROFILE_KEYS]
    profiles["mase_rank"] = profiles.groupby(rank_keys, sort=False)[
        "mase_mean"
    ].rank(method="average", ascending=True)
    profiles["mechanism_rank"] = profiles[
        "mechanism_fidelity_score"
    ].where(profiles["formal_score_eligible"]).groupby(
        [profiles[key] for key in rank_keys],
        sort=False,
    ).rank(method="average", ascending=False)
    profiles["ability_rank"] = profiles["ability_score"].groupby(
        [profiles[key] for key in rank_keys],
        sort=False,
    ).rank(method="average", ascending=False)
    return profiles


def stable_bootstrap_seed(
    base_seed: int,
    values: tuple[Any, ...],
) -> int:
    payload = "|".join([str(base_seed), *(str(value) for value in values)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def equivalence_state(
    *,
    ci_low: float,
    ci_high: float,
    margin: float,
) -> str:
    if ci_low > margin:
        return "left_better"
    if ci_high < -margin:
        return "right_better"
    if ci_low >= -margin and ci_high <= margin:
        return "equivalent"
    return "unresolved"


def _effect(
    left: np.ndarray | float,
    right: np.ndarray | float,
    *,
    metric_id: str,
) -> np.ndarray | float:
    if metric_id == "mase":
        denominator = np.maximum(
            np.abs(left) + np.abs(right),
            1e-12,
        )
        return 2.0 * (right - left) / denominator
    return left - right


def bootstrap_profile_statistics(
    components: pd.DataFrame,
    *,
    bank_id: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    ci_level: float,
    equivalence_margins: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    alpha = (1.0 - ci_level) / 2.0
    interval_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for profile_key, cell in components.groupby(PROFILE_KEYS, sort=True):
        model_ids = sorted(cell["model_id"].unique())
        group_ids: list[str] | None = None
        matrices: dict[str, list[np.ndarray]] = {
            "mase": [],
            "blind": [],
            "level": [],
            "dose": [],
        }
        formal_by_model: dict[str, bool] = {}
        for model_id in model_ids:
            model = cell[cell["model_id"] == model_id].sort_values(
                "paired_group_id"
            )
            observed_groups = model["paired_group_id"].astype(str).tolist()
            if group_ids is None:
                group_ids = observed_groups
            elif observed_groups != group_ids:
                raise ValueError(
                    f"models do not share paired groups for {profile_key}"
                )
            matrices["mase"].append(
                model["mase_group_mean"].to_numpy(dtype=float)
            )
            matrices["blind"].append(
                model["blind_mase_group_mean"].to_numpy(dtype=float)
            )
            matrices["level"].append(
                model["level_mechanism_group_mean"].to_numpy(dtype=float)
            )
            matrices["dose"].append(
                model["dose_response_score"].to_numpy(dtype=float)
            )
            formal_by_model[model_id] = bool(
                model["formal_score_eligible"].all()
            )
        if not group_ids:
            continue
        values = {
            name: np.column_stack(columns)
            for name, columns in matrices.items()
        }
        group_count = len(group_ids)
        rng = np.random.default_rng(
            stable_bootstrap_seed(
                bootstrap_seed,
                (bank_id, *profile_key),
            )
        )
        draws = rng.integers(
            0,
            group_count,
            size=(bootstrap_replicates, group_count),
        )
        point_mase = values["mase"].mean(axis=0)
        point_blind = values["blind"].mean(axis=0)
        point_level = values["level"].mean(axis=0)
        point_dose = values["dose"].mean(axis=0)
        point_mechanism = 0.7 * point_level + 0.3 * point_dose
        point_ability = point_mechanism * np.minimum(
            1.0,
            point_blind / point_mase,
        )
        boot_mase = values["mase"][draws].mean(axis=1)
        boot_blind = values["blind"][draws].mean(axis=1)
        boot_level = values["level"][draws].mean(axis=1)
        boot_dose = values["dose"][draws].mean(axis=1)
        boot_mechanism = 0.7 * boot_level + 0.3 * boot_dose
        boot_ability = boot_mechanism * np.minimum(
            1.0,
            boot_blind / boot_mase,
        )
        point_metrics = {
            "mase": point_mase,
            "mechanism": point_mechanism,
            "ability": point_ability,
        }
        boot_metrics = {
            "mase": boot_mase,
            "mechanism": boot_mechanism,
            "ability": boot_ability,
        }
        profile_values = dict(zip(PROFILE_KEYS, profile_key, strict=True))
        for metric_id, point_values in point_metrics.items():
            boot_values = boot_metrics[metric_id]
            formal_metric = metric_id == "mase"
            for model_index, model_id in enumerate(model_ids):
                eligible = formal_metric or formal_by_model[model_id]
                low, high = np.quantile(
                    boot_values[:, model_index],
                    [alpha, 1.0 - alpha],
                )
                interval_rows.append(
                    {
                        **profile_values,
                        "bank_id": bank_id,
                        "model_id": model_id,
                        "metric_id": metric_id,
                        "score": float(point_values[model_index]),
                        "bootstrap_ci_low": float(low),
                        "bootstrap_ci_high": float(high),
                        "bootstrap_replicates": bootstrap_replicates,
                        "ci_level": ci_level,
                        "paired_group_count": group_count,
                        "formal_score_eligible": eligible,
                    }
                )
            for left_index, right_index in combinations(
                range(len(model_ids)),
                2,
            ):
                left_model = model_ids[left_index]
                right_model = model_ids[right_index]
                eligible = (
                    metric_id == "mase"
                    or (
                        formal_by_model[left_model]
                        and formal_by_model[right_model]
                    )
                )
                if not eligible:
                    continue
                point_effect = float(
                    _effect(
                        point_values[left_index],
                        point_values[right_index],
                        metric_id=metric_id,
                    )
                )
                bootstrap_effect = np.asarray(
                    _effect(
                        boot_values[:, left_index],
                        boot_values[:, right_index],
                        metric_id=metric_id,
                    ),
                    dtype=float,
                )
                low, high = np.quantile(
                    bootstrap_effect,
                    [alpha, 1.0 - alpha],
                )
                for margin in equivalence_margins:
                    pair_rows.append(
                        {
                            **profile_values,
                            "bank_id": bank_id,
                            "metric_id": metric_id,
                            "effect_scale": METRIC_SPECS[metric_id][
                                "effect_scale"
                            ],
                            "left_model": left_model,
                            "right_model": right_model,
                            "left_score": float(
                                point_values[left_index]
                            ),
                            "right_score": float(
                                point_values[right_index]
                            ),
                            "effect_left_better_positive": point_effect,
                            "bootstrap_ci_low": float(low),
                            "bootstrap_ci_high": float(high),
                            "bootstrap_replicates": bootstrap_replicates,
                            "ci_level": ci_level,
                            "equivalence_margin": margin,
                            "state": equivalence_state(
                                ci_low=float(low),
                                ci_high=float(high),
                                margin=margin,
                            ),
                        }
                    )
    return pd.DataFrame(interval_rows), pd.DataFrame(pair_rows)


def deterministic_split_components(
    components: pd.DataFrame,
    *,
    split_size: int,
) -> pd.DataFrame:
    if split_size < 1:
        raise ValueError("split_size must be positive")
    assignments: list[dict[str, Any]] = []
    reference = components.drop_duplicates(
        [*PROFILE_KEYS, "paired_group_id"]
    )
    for profile_key, cell in reference.groupby(PROFILE_KEYS, sort=True):
        order_columns = (
            ["analysis_pool_index", "paired_group_id"]
            if (
                "analysis_pool_index" in cell
                and (cell["analysis_pool_index"] >= 0).all()
            )
            else ["round_index", "sample_index", "paired_group_id"]
        )
        ordered = cell.sort_values(order_columns)
        if len(ordered) != 2 * split_size:
            raise ValueError(
                f"{profile_key} has {len(ordered)} groups, "
                f"must equal two complete blocks of {split_size}"
            )
        for bank_id, selected in (
            ("first", ordered.iloc[:split_size]),
            ("second", ordered.iloc[split_size : 2 * split_size]),
        ):
            for paired_group_id in selected["paired_group_id"]:
                assignments.append(
                    {
                        **dict(
                            zip(
                                PROFILE_KEYS,
                                profile_key,
                                strict=True,
                            )
                        ),
                        "paired_group_id": paired_group_id,
                        "bank_id": bank_id,
                    }
                )
    assignment_frame = pd.DataFrame(assignments)
    split = components.merge(
        assignment_frame,
        on=[*PROFILE_KEYS, "paired_group_id"],
        how="inner",
        validate="many_to_one",
    )
    return split


def split_assignment_audit(
    components: pd.DataFrame,
    split_components: pd.DataFrame,
    *,
    split_size: int,
) -> dict[str, Any]:
    reference = components.drop_duplicates(
        [*PROFILE_KEYS, "paired_group_id"]
    )
    split_reference = split_components.drop_duplicates(
        [*PROFILE_KEYS, "paired_group_id", "bank_id"]
    )
    cell_records: list[dict[str, Any]] = []
    for profile_key, all_cell in reference.groupby(PROFILE_KEYS, sort=True):
        selector = np.logical_and.reduce(
            [
                split_reference[key] == value
                for key, value in zip(
                    PROFILE_KEYS,
                    profile_key,
                    strict=True,
                )
            ]
        )
        split_cell = split_reference[selector]
        first = set(
            split_cell.loc[
                split_cell["bank_id"] == "first",
                "paired_group_id",
            ].astype(str)
        )
        second = set(
            split_cell.loc[
                split_cell["bank_id"] == "second",
                "paired_group_id",
            ].astype(str)
        )
        all_groups = set(all_cell["paired_group_id"].astype(str))
        source_block_alignment: bool | None = None
        source_first_blocks: list[str] = []
        source_second_blocks: list[str] = []
        if "analysis_block_id" in split_cell:
            nonempty = split_cell[
                split_cell["analysis_block_id"].astype(str) != ""
            ]
            if not nonempty.empty:
                source_first_blocks = sorted(
                    set(
                        nonempty.loc[
                            nonempty["bank_id"] == "first",
                            "analysis_block_id",
                        ].astype(str)
                    )
                )
                source_second_blocks = sorted(
                    set(
                        nonempty.loc[
                            nonempty["bank_id"] == "second",
                            "analysis_block_id",
                        ].astype(str)
                    )
                )
                source_block_alignment = (
                    len(source_first_blocks) == 1
                    and len(source_second_blocks) == 1
                    and source_first_blocks != source_second_blocks
                )
        record = {
            **dict(zip(PROFILE_KEYS, profile_key, strict=True)),
            "all_group_count": len(all_groups),
            "first_group_count": len(first),
            "second_group_count": len(second),
            "mutually_exclusive": first.isdisjoint(second),
            "covers_all_selected_groups": first | second == all_groups,
            "source_first_analysis_block_ids": source_first_blocks,
            "source_second_analysis_block_ids": source_second_blocks,
            "source_analysis_block_alignment": source_block_alignment,
        }
        if (
            record["first_group_count"] != split_size
            or record["second_group_count"] != split_size
            or not record["mutually_exclusive"]
            or not record["covers_all_selected_groups"]
            or source_block_alignment is False
        ):
            raise ValueError(f"invalid split assignment: {record}")
        cell_records.append(record)
    return {
        "split_size": split_size,
        "cell_count": len(cell_records),
        "all_cells_have_two_complete_blocks": True,
        "all_blocks_mutually_exclusive": True,
        "all_blocks_cover_all_selected_groups": True,
        "all_source_analysis_blocks_aligned": all(
            record["source_analysis_block_alignment"] is not False
            for record in cell_records
        ),
        "cells": cell_records,
    }


def split_half_reliability(
    split_profiles: pd.DataFrame,
    split_pairs: pd.DataFrame,
    *,
    primary_margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    left = split_profiles[split_profiles["bank_id"] == "first"]
    right = split_profiles[split_profiles["bank_id"] == "second"]
    score_rows: list[dict[str, Any]] = []
    for metric_id, spec in METRIC_SPECS.items():
        score_column = str(spec["column"])
        rank_column = (
            "mase_rank"
            if metric_id == "mase"
            else f"{metric_id}_rank"
        )
        keys = ["model_id", *PROFILE_KEYS]
        compared = left[keys + [score_column, rank_column]].merge(
            right[keys + [score_column, rank_column]],
            on=keys,
            suffixes=("_first", "_second"),
            validate="one_to_one",
        )
        compared = compared.dropna(
            subset=[
                f"{score_column}_first",
                f"{score_column}_second",
                f"{rank_column}_first",
                f"{rank_column}_second",
            ]
        )
        for profile_key, cell in compared.groupby(PROFILE_KEYS, sort=True):
            first_scores = cell[f"{score_column}_first"].to_numpy(
                dtype=float
            )
            second_scores = cell[f"{score_column}_second"].to_numpy(
                dtype=float
            )
            first_ranks = cell[f"{rank_column}_first"].to_numpy(dtype=float)
            second_ranks = cell[f"{rank_column}_second"].to_numpy(dtype=float)
            pair_agreements: list[bool] = []
            for left_index, right_index in combinations(
                range(len(cell)),
                2,
            ):
                first_effect = float(
                    _effect(
                        first_scores[left_index],
                        first_scores[right_index],
                        metric_id=metric_id,
                    )
                )
                second_effect = float(
                    _effect(
                        second_scores[left_index],
                        second_scores[right_index],
                        metric_id=metric_id,
                    )
                )
                pair_agreements.append(
                    np.sign(first_effect) == np.sign(second_effect)
                )
            first_top = set(
                cell.loc[
                    cell[f"{rank_column}_first"]
                    == cell[f"{rank_column}_first"].min(),
                    "model_id",
                ]
            )
            second_top = set(
                cell.loc[
                    cell[f"{rank_column}_second"]
                    == cell[f"{rank_column}_second"].min(),
                    "model_id",
                ]
            )
            score_rows.append(
                {
                    **dict(
                        zip(PROFILE_KEYS, profile_key, strict=True)
                    ),
                    "metric_id": metric_id,
                    "model_count": len(cell),
                    "rank_spearman": spearman_correlation(
                        first_ranks,
                        second_ranks,
                    ),
                    "exact_rank_vector": bool(
                        np.array_equal(first_ranks, second_ranks)
                    ),
                    "point_pair_direction_agreement": (
                        float(np.mean(pair_agreements))
                        if pair_agreements
                        else math.nan
                    ),
                    "top_model_exact_match": first_top == second_top,
                    "top_model_jaccard": (
                        len(first_top & second_top)
                        / len(first_top | second_top)
                    ),
                }
            )
    reliability = pd.DataFrame(score_rows)
    primary = split_pairs[
        np.isclose(
            split_pairs["equivalence_margin"],
            primary_margin,
        )
    ]
    pair_first = primary[primary["bank_id"] == "first"]
    pair_second = primary[primary["bank_id"] == "second"]
    pair_keys = [
        *PROFILE_KEYS,
        "metric_id",
        "left_model",
        "right_model",
        "equivalence_margin",
    ]
    pair_comparison = pair_first[pair_keys + ["state"]].merge(
        pair_second[pair_keys + ["state"]],
        on=pair_keys,
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    pair_comparison["state_match"] = (
        pair_comparison["state_first"]
        == pair_comparison["state_second"]
    )
    pair_comparison["direction_conflict"] = (
        (
            pair_comparison["state_first"] == "left_better"
        )
        & (pair_comparison["state_second"] == "right_better")
    ) | (
        (
            pair_comparison["state_first"] == "right_better"
        )
        & (pair_comparison["state_second"] == "left_better")
    )
    pair_comparison["no_direction_contradiction"] = ~pair_comparison[
        "direction_conflict"
    ]
    summary = {
        "profile_metric_count": len(reliability),
        "rank_spearman_mean": float(
            reliability["rank_spearman"].mean()
        ),
        "exact_rank_vector_rate": float(
            reliability["exact_rank_vector"].mean()
        ),
        "point_pair_direction_agreement_mean": float(
            reliability["point_pair_direction_agreement"].mean()
        ),
        "top_model_exact_match_rate": float(
            reliability["top_model_exact_match"].mean()
        ),
        "top_model_jaccard_mean": float(
            reliability["top_model_jaccard"].mean()
        ),
        "primary_equivalence_margin": primary_margin,
        "pair_state_count": len(pair_comparison),
        "pair_state_exact_match_rate": float(
            pair_comparison["state_match"].mean()
        ),
        "pair_no_direction_contradiction_rate": float(
            pair_comparison["no_direction_contradiction"].mean()
        ),
        "pair_direction_conflict_count": int(
            pair_comparison["direction_conflict"].sum()
        ),
    }
    return reliability, pair_comparison, summary


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = pd.Series(np.asarray(left, dtype=float)).rank(method="average").to_numpy()
    y = pd.Series(np.asarray(right, dtype=float)).rank(method="average").to_numpy()
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def minmax_scale(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    span = float(np.max(vector) - np.min(vector))
    if span <= 1e-12:
        return np.zeros_like(vector)
    return (vector - np.min(vector)) / span


def lin_concordance(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = float(x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    if denominator <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.clip(2.0 * covariance / denominator, -1.0, 1.0))


def geometric_mean(values: tuple[float, ...]) -> float:
    vector = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    if np.any(vector <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(vector))))


def write_report(
    path: Path,
    *,
    profiles: pd.DataFrame,
    pair_states: pd.DataFrame,
    coverage: pd.DataFrame,
    sample_count: int,
    models: list[str],
    max_groups: int,
    covariate_ablation_available: bool,
    split_summary: dict[str, Any] | None,
    split_size: int,
    primary_margin: float,
) -> None:
    lines = [
        "# Paper v7 E3：正式机制保真能力画像",
        "",
        "本结果同时保留点预测误差、机制保真度与能力总分。机制分只说明输出行为",
        "与合成机制一致，不表示模型内部识别了因果生成机制。",
        "",
        f"- 模型：{', '.join(models)}",
        f"- 逐模型样本评分总行数：{sample_count}",
        f"- 每个 dataset × task × capability 的 paired groups 上限：{max_groups or '全部'}",
        "- 不支持的模型 × 能力组合记为 N/A，不补成最差名次。",
        "- 单档机制分与 I1–I5 剂量响应按 0.7/0.3 合成。",
        "- 能力总分使用 naive MASE 安全门：MFS × min(1, naive MASE/model MASE)。",
        (
            "- covariate_response 已使用 intact/zero-future-covariate "
            "配对消融。"
            if covariate_ablation_available
            else (
                "- covariate_response 在没有 future-covariate 配对消融时"
                "只作诊断，不进入正式机制/能力排名。"
            )
        ),
        "",
        "## 输入兼容范围",
        "",
        "| Dataset | Task | Capability | Native models | Adapted models | Unsupported models |",
        "|---|---|---|---|---|---|",
    ]
    for profile_key, cell in coverage.groupby(
        PROFILE_KEYS,
        sort=True,
    ):
        native = sorted(
            cell.loc[
                cell["supported"]
                & cell["input_execution_mode"].isin(
                    ["native", "legacy_native"]
                ),
                "model_id",
            ]
        )
        adapted = sorted(
            cell.loc[
                cell["supported"]
                & ~cell["input_execution_mode"].isin(
                    ["native", "legacy_native"]
                ),
                "model_id",
            ]
        )
        unsupported = sorted(cell.loc[~cell["supported"], "model_id"])
        lines.append(
            "| "
            + " | ".join(
                [
                    *[str(value) for value in profile_key],
                    ", ".join(native) or "—",
                    ", ".join(adapted) or "—",
                    ", ".join(unsupported) or "—",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 能力画像",
            "",
            "| Dataset | Capability | Model | Input | Counterfactual | MASE [95% CI] | Level MFS | Dose | MFS [95% CI] | Ability [95% CI] | MASE rank | Mechanism rank | Ability rank | Formal |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in profiles.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.dataset_id),
                    str(row.capability_id),
                    str(row.model_id),
                    str(row.input_execution_mode),
                    str(row.counterfactual_mode),
                    _format_interval(
                        row.mase_mean,
                        row.mase_ci_low,
                        row.mase_ci_high,
                    ),
                    _format_number(row.level_mechanism_fidelity),
                    _format_number(row.dose_response_score),
                    _format_interval(
                        row.mechanism_fidelity_score,
                        row.mechanism_ci_low,
                        row.mechanism_ci_high,
                    ),
                    _format_interval(
                        row.ability_score,
                        row.ability_ci_low,
                        row.ability_ci_high,
                    ),
                    _format_number(row.mase_rank),
                    _format_number(row.mechanism_rank),
                    _format_number(row.ability_rank),
                    "yes" if bool(row.formal_score_eligible) else "diagnostic",
                ]
            )
            + " |"
        )
    primary = pair_states[
        np.isclose(
            pair_states["equivalence_margin"],
            primary_margin,
        )
    ]
    lines.extend(
        [
            "",
            "## Tie-aware 模型对",
            "",
            f"主等价阈值为 {primary_margin:.2%}：MASE 使用对称相对差，"
            "MFS/Ability 使用 `[0,1]` 绝对差。",
            "",
            "| Metric | Left better | Right better | Equivalent | Unresolved |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    state_counts = (
        primary.groupby(["metric_id", "state"], sort=True)
        .size()
        .unstack(fill_value=0)
    )
    for metric_id in METRIC_SPECS:
        counts = (
            state_counts.loc[metric_id]
            if metric_id in state_counts.index
            else pd.Series(dtype=int)
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    metric_id,
                    str(int(counts.get("left_better", 0))),
                    str(int(counts.get("right_better", 0))),
                    str(int(counts.get("equivalent", 0))),
                    str(int(counts.get("unresolved", 0))),
                ]
            )
            + " |"
        )
    if split_summary is not None:
        lines.extend(
            [
                "",
                f"## 前 {split_size} / 后 {split_size} paired groups 可靠性",
                "",
                f"- 平均 rank Spearman：{split_summary['rank_spearman_mean']:.4f}",
                f"- point pair 方向一致率：{split_summary['point_pair_direction_agreement_mean']:.4f}",
                f"- top model 完全一致率：{split_summary['top_model_exact_match_rate']:.4f}",
                f"- tie-aware pair state 完全一致率：{split_summary['pair_state_exact_match_rate']:.4f}",
                f"- tie-aware pair 无方向冲突率：{split_summary['pair_no_direction_contradiction_rate']:.4f}",
                f"- 明确方向冲突数：{split_summary['pair_direction_conflict_count']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.4f}" if math.isfinite(number) else "N/A"


def _format_interval(value: Any, low: Any, high: Any) -> str:
    center = _format_number(value)
    if center == "N/A":
        return center
    return f"{center} [{_format_number(low)}, {_format_number(high)}]"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
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


def main() -> None:
    args = parse_args()
    model_ids = list(dict.fromkeys(str(value) for value in args.models))
    if not model_ids:
        raise ValueError("at least one model is required")
    if BLIND_REFERENCE_MODEL in model_ids:
        raise ValueError("naive is reserved as the point-accuracy reference")
    if int(args.bootstrap_replicates) < 1:
        raise ValueError("bootstrap-replicates must be positive")
    if not 0.0 < float(args.ci_level) < 1.0:
        raise ValueError("ci-level must be between zero and one")
    equivalence_margins = tuple(
        sorted({float(value) for value in args.equivalence_margins})
    )
    if (
        not equivalence_margins
        or any(value <= 0.0 for value in equivalence_margins)
    ):
        raise ValueError("equivalence margins must be positive")
    primary_margin = float(args.primary_equivalence_margin)
    if not any(
        math.isclose(primary_margin, value)
        for value in equivalence_margins
    ):
        raise ValueError(
            "primary-equivalence-margin must be one of equivalence-margins"
        )
    if int(args.split_size) < 0:
        raise ValueError("split-size cannot be negative")
    supported_cells = discover_supported_cells(
        args.e2_dir,
        dataset_ids=(
            [str(value) for value in args.datasets]
            if args.datasets is not None
            else None
        ),
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output directory already exists: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    samples = load_selected_samples(
        args.e2_dir,
        supported_cells=supported_cells,
        max_paired_groups_per_cell=int(args.max_paired_groups_per_cell),
    )
    oracle_selections, oracle_scores = load_oracle_selection(
        args.e2_dir,
        model_ids=model_ids,
        samples=samples,
    )
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for model_id in model_ids:
        print(f"loading selected {model_id} forecasts", flush=True)
        predictions[model_id] = load_selected_predictions(
            args.e2_dir,
            model_id=model_id,
            oracle_selection=oracle_selections[model_id],
            require_input_adaptation=True,
        )
    coverage = model_capability_coverage(
        samples,
        oracle_selections,
        model_ids,
        predictions,
    )
    counterfactual_predictions = load_covariate_ablation_predictions(
        args.covariate_ablation_predictions_dir,
        model_ids=model_ids,
        samples=samples,
        oracle_selections=oracle_selections,
        intact_predictions=predictions,
    )
    ablation_input = covariate_ablation_input_record(
        args.covariate_ablation_predictions_dir,
        counterfactual_predictions=counterfactual_predictions,
    )
    sample_scores = evaluate_selected_predictions(
        samples,
        oracle_selections,
        predictions,
        counterfactual_predictions,
    )
    cells = intensity_cell_scores(sample_scores, oracle_scores)
    dose = paired_dose_response_scores(sample_scores)
    profiles = capability_profiles(cells, dose)
    components = profile_group_components(
        sample_scores,
        oracle_scores,
        dose,
    )
    all_intervals, all_pairs = bootstrap_profile_statistics(
        components,
        bank_id="all",
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.bootstrap_seed),
        ci_level=float(args.ci_level),
        equivalence_margins=equivalence_margins,
    )
    for metric_id in METRIC_SPECS:
        interval = all_intervals[
            all_intervals["metric_id"] == metric_id
        ][
            [
                "model_id",
                *PROFILE_KEYS,
                "bootstrap_ci_low",
                "bootstrap_ci_high",
            ]
        ].rename(
            columns={
                "bootstrap_ci_low": f"{metric_id}_ci_low",
                "bootstrap_ci_high": f"{metric_id}_ci_high",
            }
        )
        profiles = profiles.merge(
            interval,
            on=["model_id", *PROFILE_KEYS],
            how="left",
            validate="one_to_one",
        )

    split_profiles = pd.DataFrame()
    split_intervals = pd.DataFrame()
    split_pairs = pd.DataFrame()
    split_reliability = pd.DataFrame()
    split_pair_comparison = pd.DataFrame()
    split_summary: dict[str, Any] | None = None
    split_audit: dict[str, Any] | None = None
    if int(args.split_size):
        split_components = deterministic_split_components(
            components,
            split_size=int(args.split_size),
        )
        split_audit = split_assignment_audit(
            components,
            split_components,
            split_size=int(args.split_size),
        )
        profile_parts: list[pd.DataFrame] = []
        interval_parts: list[pd.DataFrame] = []
        pair_parts: list[pd.DataFrame] = []
        for bank_id, bank in split_components.groupby(
            "bank_id",
            sort=True,
        ):
            profile_parts.append(
                aggregate_profile_components(bank, bank_id=str(bank_id))
            )
            intervals, pairs = bootstrap_profile_statistics(
                bank,
                bank_id=str(bank_id),
                bootstrap_replicates=int(args.bootstrap_replicates),
                bootstrap_seed=int(args.bootstrap_seed),
                ci_level=float(args.ci_level),
                equivalence_margins=equivalence_margins,
            )
            interval_parts.append(intervals)
            pair_parts.append(pairs)
        split_profiles = pd.concat(profile_parts, ignore_index=True)
        split_intervals = pd.concat(interval_parts, ignore_index=True)
        split_pairs = pd.concat(pair_parts, ignore_index=True)
        (
            split_reliability,
            split_pair_comparison,
            split_summary,
        ) = split_half_reliability(
            split_profiles,
            split_pairs,
            primary_margin=primary_margin,
        )
        split_summary["assignment"] = split_audit

    sample_scores.to_csv(output_dir / "sample_mechanism_scores.csv", index=False)
    cells.to_csv(output_dir / "intensity_cell_scores.csv", index=False)
    dose.to_csv(output_dir / "paired_dose_response_scores.csv", index=False)
    profiles.to_csv(output_dir / "capability_profiles.csv", index=False)
    coverage.to_csv(output_dir / "model_capability_coverage.csv", index=False)
    components.to_csv(output_dir / "profile_group_components.csv", index=False)
    all_intervals.to_csv(
        output_dir / "capability_bootstrap_intervals.csv",
        index=False,
    )
    all_pairs.to_csv(
        output_dir / "capability_pair_states.csv",
        index=False,
    )
    if split_summary is not None:
        split_profiles.to_csv(
            output_dir / "split_half_capability_profiles.csv",
            index=False,
        )
        split_intervals.to_csv(
            output_dir / "split_half_bootstrap_intervals.csv",
            index=False,
        )
        split_pairs.to_csv(
            output_dir / "split_half_pair_states.csv",
            index=False,
        )
        split_reliability.to_csv(
            output_dir / "split_half_reliability.csv",
            index=False,
        )
        split_pair_comparison.to_csv(
            output_dir / "split_half_pair_state_comparison.csv",
            index=False,
        )
        write_json(
            output_dir / "split_half_summary.json",
            split_summary,
        )
    write_report(
        output_dir / "report.md",
        profiles=profiles,
        pair_states=all_pairs,
        coverage=coverage,
        sample_count=len(sample_scores),
        models=model_ids,
        max_groups=int(args.max_paired_groups_per_cell),
        covariate_ablation_available=bool(counterfactual_predictions),
        split_summary=split_summary,
        split_size=int(args.split_size),
        primary_margin=primary_margin,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "models": model_ids,
        "blind_reference_model": BLIND_REFERENCE_MODEL,
        "supported_cells": [
            {
                "dataset_id": dataset_id,
                "task_id": task_id,
                "capability_id": capability_id,
            }
            for dataset_id, task_id, capability_id in supported_cells
        ],
        "max_paired_groups_per_cell": int(
            args.max_paired_groups_per_cell
        ),
        "sample_score_count": len(sample_scores),
        "profile_count": len(profiles),
        "formal_profile_count": int(profiles["formal_score_eligible"].sum()),
        "diagnostic_profile_count": int(
            (~profiles["formal_score_eligible"]).sum()
        ),
        "mechanism_score_composition": {
            "level_fidelity_weight": 0.7,
            "dose_response_weight": 0.3,
        },
        "bootstrap": {
            "unit": "paired_group_id_with_all_five_intensities",
            "shared_draws_across_models": True,
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.bootstrap_seed),
            "ci_level": float(args.ci_level),
            "equivalence_margins": list(equivalence_margins),
            "primary_equivalence_margin": primary_margin,
            "mase_effect_scale": "symmetric_relative_difference",
            "mechanism_and_ability_effect_scale": (
                "absolute_difference_on_unit_interval"
            ),
        },
        "split_half": (
            {
                "split_size": int(args.split_size),
                "ordering": (
                    "analysis_pool_index_then_paired_group_id_with_"
                    "round_index_sample_index_fallback"
                ),
                "assignment": split_audit,
                "summary": split_summary,
            }
            if split_summary is not None
            else None
        ),
        "compatibility": {
            "supported_profile_count": int(coverage["supported"].sum()),
            "unsupported_profile_count": int((~coverage["supported"]).sum()),
            "unsupported_policy": "N/A_not_worst_rank",
            "native_profile_count": int(
                (
                    coverage["supported"]
                    & coverage["input_execution_mode"].isin(
                        ["native", "legacy_native"]
                    )
                ).sum()
            ),
            "adapted_profile_count": int(
                (
                    coverage["supported"]
                    & ~coverage["input_execution_mode"].isin(
                        ["native", "legacy_native"]
                    )
                ).sum()
            ),
            "coverage_unit": "original_forecast_view",
        },
        "ability_score": (
            "mechanism_fidelity_score * "
            "min(1, naive_mase / model_mase)"
        ),
        "covariate_response_status": (
            "formal_paired_future_covariate_ablation_with_native_or_reused_"
            "counterfactual_by_input_provenance"
            if counterfactual_predictions
            else "diagnostic_only_until_paired_future_covariate_ablation"
        ),
        "covariate_ablation_predictions_dir": (
            str(args.covariate_ablation_predictions_dir.resolve())
            if args.covariate_ablation_predictions_dir is not None
            else None
        ),
        "source_inputs": {
            "e2_sample_manifest": {
                "path": str(
                    (args.e2_dir / "sample_manifest.json").resolve()
                ),
                "sha256": sha256_file(
                    args.e2_dir / "sample_manifest.json"
                ),
            },
            "generation_support_artifact": {
                "path": str(support_artifact_path(args.e2_dir).resolve()),
                "sha256": sha256_file(
                    support_artifact_path(args.e2_dir)
                ),
            },
            "e2_inference_manifest": {
                "path": str(
                    (args.e2_dir / "inference_manifest.json").resolve()
                ),
                "sha256": sha256_file(
                    args.e2_dir / "inference_manifest.json"
                ),
            },
            "covariate_ablation": ablation_input,
        },
    }
    write_json(output_dir / "summary.json", summary)
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        outputs[path.name] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        **summary,
        "source_e2_dir": str(args.e2_dir.resolve()),
        "protocol_path": str(DEFAULT_PROTOCOL_PATH),
        "protocol_sha256": (
            sha256_file(DEFAULT_PROTOCOL_PATH)
            if DEFAULT_PROTOCOL_PATH.is_file()
            else None
        ),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": outputs,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"E3 formal mechanism analysis complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
