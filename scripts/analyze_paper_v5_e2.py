#!/usr/bin/env python3
"""Analyze formal Paper v7 E2 view predictions after inference completes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_paper_e2_dynamic_stability as stats  # noqa: E402


DEFAULT_E2_DIR = REPO_ROOT / "runtime/paper_exp/v7/E2_dynamic_stability"
DEFAULT_REAL_SOURCE_DIR = (
    REPO_ROOT / "runtime/paper_exp/v7/02_real_source_window_suite"
)
CONTEXT_LENGTHS = (96, 168, 336, 504)
MIN_PAIRWISE_AGREEMENT = 0.95
BOOTSTRAP_REPLICATES = 10_000
SCHEMA_VERSION = "paper_v7_e2_analysis.v1"
BASELINE_MODELS = ("naive", "seasonal_naive")
FORMAL_ROUND_SEEDS = (
    2026072121,
    2026072122,
    2026072123,
    2026072124,
    2026072125,
)
FORMAL_SAMPLES_PER_ROUND = 64
FORMAL_TOTAL_PER_INTENSITY = 320
FORMAL_ANALYSIS_BLOCK_SIZE = 160
FORMAL_ANALYSIS_BLOCK_IDS = ("A", "B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse four contexts and analyze Paper v7 E2 cell stability "
            "plus synthetic-real source-window rank alignment."
        )
    )
    parser.add_argument("--e2-dir", type=Path, default=DEFAULT_E2_DIR)
    parser.add_argument(
        "--real-source-dir",
        type=Path,
        default=DEFAULT_REAL_SOURCE_DIR,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
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
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error


def safe_filename(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in "-_"
        else "_"
        for character in value
    )


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def prediction_path(
    e2_dir: Path,
    model_id: str,
    *,
    prediction_kind: str,
) -> Path:
    directory = (
        "real_source_predictions"
        if prediction_kind == "real_source"
        else "predictions"
    )
    return e2_dir / directory / f"{safe_filename(model_id)}.jsonl"


def oracle_path(
    e2_dir: Path,
    model_id: str,
    *,
    prediction_kind: str,
) -> Path:
    directory = (
        "real_source_oracle_scores"
        if prediction_kind == "real_source"
        else "oracle_sample_scores"
    )
    return e2_dir / directory / f"{safe_filename(model_id)}.jsonl"


def compact_prediction_file(
    source: Path,
    destination: Path,
    *,
    model_id: str,
    prediction_kind: str,
) -> dict[str, Any]:
    masters: dict[str, dict[str, Any]] = {}
    view_count = 0
    mase_unavailable_view_count = 0
    for row in iter_jsonl(source):
        if str(row["model_id"]) != model_id:
            raise ValueError(f"model mismatch in {source}")
        master_id = str(row["master_sample_id"])
        context = int(row["context_length"])
        if context not in CONTEXT_LENGTHS:
            raise ValueError(f"unexpected context in {source}: {context}")
        mase = row.get("metrics", {}).get("mase")
        if mase is not None and not math.isfinite(float(mase)):
            raise ValueError(
                f"non-finite MASE for {model_id}/{master_id}/L{context}"
            )
        mae = float(row["metrics"]["mae"])
        mse = float(row["metrics"]["mse"])
        if not math.isfinite(mae) or not math.isfinite(mse):
            raise ValueError(
                f"non-finite error metric for "
                f"{model_id}/{master_id}/L{context}"
            )
        record = masters.setdefault(
            master_id,
            {
                "schema_version": "paper_v7_e2_oracle_sample_score.v1",
                "prediction_kind": prediction_kind,
                "model_id": model_id,
                "model_group": row["model_group"],
                "master_sample_id": master_id,
                "dataset_id": row["dataset_id"],
                "task_id": row["task_id"],
                "profile_id": row["profile_id"],
                "contexts": {},
            },
        )
        if str(context) in record["contexts"]:
            raise ValueError(
                f"duplicate context prediction: {model_id}/{master_id}/L{context}"
            )
        context_row = {
            "view_id": row["view_id"],
            "mase": float(mase) if mase is not None else None,
            "mae": mae,
            "mse": mse,
        }
        if mase is None:
            context_row["mase_unavailable_reason"] = str(
                row.get("mase_unavailable_reason") or "unspecified"
            )
            mase_unavailable_view_count += 1
        if "request_seconds" in row:
            context_row["request_seconds"] = float(row["request_seconds"])
            context_row["request_attempts"] = int(row["request_attempts"])
        record["contexts"][str(context)] = context_row
        if prediction_kind == "synthetic":
            round_index = int(row["round_index"])
            sample_index = int(row["sample_index"])
            pool_index = (
                (round_index - 1) * FORMAL_SAMPLES_PER_ROUND
                + sample_index
            )
            analysis_block_index = (
                pool_index // FORMAL_ANALYSIS_BLOCK_SIZE
            )
            if analysis_block_index >= len(FORMAL_ANALYSIS_BLOCK_IDS):
                raise ValueError(
                    f"formal pool index is out of range: {pool_index}"
                )
            analysis_block_id = FORMAL_ANALYSIS_BLOCK_IDS[
                analysis_block_index
            ]
            for field, expected in (
                ("pool_index", pool_index),
                ("analysis_block_id", analysis_block_id),
                (
                    "analysis_block_index",
                    pool_index % FORMAL_ANALYSIS_BLOCK_SIZE,
                ),
            ):
                if field in row and row[field] != expected:
                    raise ValueError(
                        f"{field} mismatch for {model_id}/{master_id}: "
                        f"{row[field]} != {expected}"
                    )
            record.update(
                {
                    "capability_id": row["capability_id"],
                    "intensity": int(row["intensity"]),
                    "round_index": round_index,
                    "round_seed": int(row["round_seed"]),
                    "sample_index": sample_index,
                    "paired_group_id": row["paired_group_id"],
                    "pool_index": pool_index,
                    "analysis_block_id": analysis_block_id,
                    "analysis_block_index": (
                        pool_index % FORMAL_ANALYSIS_BLOCK_SIZE
                    ),
                }
            )
        else:
            record.update(
                {
                    "source_reference_index": int(
                        row["source_reference_index"]
                    ),
                    "supported_capabilities": list(
                        row["supported_capabilities"]
                    ),
                }
            )
        view_count += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for master_id in sorted(masters):
            record = masters[master_id]
            observed = {
                int(context) for context in record["contexts"]
            }
            if observed != set(CONTEXT_LENGTHS):
                raise ValueError(
                    f"incomplete contexts for {model_id}/{master_id}: "
                    f"{sorted(observed)}"
                )
            finite_mase_contexts = [
                context
                for context in CONTEXT_LENGTHS
                if record["contexts"][str(context)]["mase"] is not None
            ]
            if not finite_mase_contexts:
                raise ValueError(
                    f"MASE is unavailable for every context: "
                    f"{model_id}/{master_id}"
                )
            selected_context = min(
                finite_mase_contexts,
                key=lambda context: (
                    record["contexts"][str(context)]["mase"],
                    context,
                ),
            )
            selected = record["contexts"][str(selected_context)]
            fixed = record["contexts"][str(max(CONTEXT_LENGTHS))]
            if fixed["mase"] is None:
                raise ValueError(
                    f"fixed L504 MASE is unavailable: "
                    f"{model_id}/{master_id}"
                )
            record.update(
                {
                    "oracle_context": selected_context,
                    "oracle_view_id": selected["view_id"],
                    "oracle_mase": selected["mase"],
                    "oracle_mae": selected["mae"],
                    "fixed_l504_mase": fixed["mase"],
                    "fixed_l504_mae": fixed["mae"],
                    "context_mase": {
                        context: record["contexts"][context]["mase"]
                        for context in sorted(record["contexts"], key=int)
                    },
                    "context_mae": {
                        context: record["contexts"][context]["mae"]
                        for context in sorted(record["contexts"], key=int)
                    },
                    "context_mase_unavailable_reason": {
                        context: record["contexts"][context][
                            "mase_unavailable_reason"
                        ]
                        for context in sorted(record["contexts"], key=int)
                        if record["contexts"][context]["mase"] is None
                    },
                }
            )
            del record["contexts"]
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, destination)
    return {
        "model_id": model_id,
        "prediction_kind": prediction_kind,
        "view_count": view_count,
        "mase_unavailable_view_count": mase_unavailable_view_count,
        "master_count": len(masters),
        "path": display_path(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
    }


def load_oracle_rows(
    paths: Iterable[Path],
    *,
    prediction_kind: str,
) -> pd.DataFrame:
    columns = [
        "model_id",
        "master_sample_id",
        "dataset_id",
        "oracle_mase",
        "fixed_l504_mase",
    ]
    if prediction_kind == "synthetic":
        columns.extend(
            [
                "task_id",
                "capability_id",
                "intensity",
                "round_index",
                "round_seed",
                "sample_index",
                "paired_group_id",
                "pool_index",
                "analysis_block_id",
                "analysis_block_index",
            ]
        )
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in iter_jsonl(path):
            if row["prediction_kind"] != prediction_kind:
                raise ValueError(f"prediction kind mismatch in {path}")
            rows.append({column: row[column] for column in columns})
    if not rows:
        raise ValueError(f"no {prediction_kind} oracle scores")
    return pd.DataFrame(rows)


CELL_KEYS = [
    "dataset_id",
    "task_id",
    "capability_id",
    "intensity",
]
ROUND_KEYS = [*CELL_KEYS, "round_index"]


def validate_formal_generation_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    round_seeds = tuple(int(value) for value in config.get("round_seeds", []))
    samples_per_round = int(
        config.get("samples_per_round_per_cell", -1)
    )
    total_per_intensity = int(config.get("total_per_intensity", -1))
    if config.get("generation_mode") != "formal_v7_round_pool":
        raise ValueError("E2 analysis requires formal_v7_round_pool generation")
    if round_seeds != FORMAL_ROUND_SEEDS:
        raise ValueError(
            "formal v7 generation_config has unexpected round seeds"
        )
    if samples_per_round != FORMAL_SAMPLES_PER_ROUND:
        raise ValueError(
            "formal v7 generation_config must use 64 samples per round"
        )
    if total_per_intensity != FORMAL_TOTAL_PER_INTENSITY:
        raise ValueError(
            "formal v7 generation_config must declare 320 per intensity"
        )
    contract = config.get("analysis_block_contract")
    expected_contract = {
        "ordering": ["round_index", "sample_index"],
        "total_per_intensity": FORMAL_TOTAL_PER_INTENSITY,
        "block_size": FORMAL_ANALYSIS_BLOCK_SIZE,
        "mutually_exclusive": True,
        "complete_partition": True,
    }
    if not isinstance(contract, dict) or any(
        contract.get(key) != value
        for key, value in expected_contract.items()
    ):
        raise ValueError(
            "formal v7 generation_config has an invalid analysis block contract"
        )
    blocks = contract.get("blocks")
    if not isinstance(blocks, list) or [
        str(block.get("analysis_block_id")) for block in blocks
    ] != list(FORMAL_ANALYSIS_BLOCK_IDS):
        raise ValueError(
            "formal v7 generation_config must declare analysis blocks A and B"
        )
    return {
        "round_seeds": list(round_seeds),
        "round_count": len(round_seeds),
        "samples_per_round": samples_per_round,
        "total_per_intensity": total_per_intensity,
        "analysis_block_size": FORMAL_ANALYSIS_BLOCK_SIZE,
        "analysis_block_ids": list(FORMAL_ANALYSIS_BLOCK_IDS),
        "analysis_blocks_mutually_exclusive": True,
    }


def validate_formal_oracle_identity(
    oracle: pd.DataFrame,
    *,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    required = {
        "model_id",
        "master_sample_id",
        "paired_group_id",
        "round_index",
        "round_seed",
        "sample_index",
        *CELL_KEYS,
    }
    missing = sorted(required - set(oracle.columns))
    if missing:
        raise ValueError(
            "synthetic oracle is missing formal identity columns: "
            + ", ".join(missing)
        )
    result = oracle.copy()
    expected_seeds = {
        round_index: int(seed)
        for round_index, seed in enumerate(
            protocol["round_seeds"],
            start=1,
        )
    }
    result["round_index"] = result["round_index"].astype(int)
    result["round_seed"] = result["round_seed"].astype(int)
    result["sample_index"] = result["sample_index"].astype(int)
    result["pool_index"] = (
        (result["round_index"] - 1) * protocol["samples_per_round"]
        + result["sample_index"]
    )
    result["analysis_block_id"] = np.where(
        result["pool_index"] < protocol["analysis_block_size"],
        FORMAL_ANALYSIS_BLOCK_IDS[0],
        FORMAL_ANALYSIS_BLOCK_IDS[1],
    )
    result["analysis_block_index"] = (
        result["pool_index"] % protocol["analysis_block_size"]
    )
    if not result["round_index"].isin(expected_seeds).all():
        raise ValueError("synthetic oracle contains an unexpected round_index")
    observed_seed = result["round_index"].map(expected_seeds)
    if not (result["round_seed"] == observed_seed).all():
        raise ValueError("synthetic oracle round_seed identity mismatch")
    if not result["sample_index"].between(
        0,
        protocol["samples_per_round"] - 1,
    ).all():
        raise ValueError("synthetic oracle sample_index is out of range")
    if not result["pool_index"].between(
        0,
        protocol["total_per_intensity"] - 1,
    ).all():
        raise ValueError("synthetic oracle pool_index is out of range")

    identity_keys = ["model_id", *CELL_KEYS, "round_index", "sample_index"]
    if result.duplicated(identity_keys).any():
        raise ValueError("synthetic oracle contains duplicate pool identities")
    expected_grid = {
        (round_index, sample_index)
        for round_index in expected_seeds
        for sample_index in range(protocol["samples_per_round"])
    }
    for key, group in result.groupby(["model_id", *CELL_KEYS], sort=True):
        observed_grid = set(
            zip(
                group["round_index"].astype(int),
                group["sample_index"].astype(int),
                strict=True,
            )
        )
        if observed_grid != expected_grid:
            raise ValueError(
                f"formal synthetic oracle pool is incomplete for {key}"
            )
        if group["master_sample_id"].nunique() != protocol[
            "total_per_intensity"
        ]:
            raise ValueError(
                f"formal synthetic oracle master ids are not unique for {key}"
            )
        block_counts = group.groupby("analysis_block_id")[
            "pool_index"
        ].nunique()
        if {
            str(block_id): int(count)
            for block_id, count in block_counts.items()
        } != {
            block_id: protocol["analysis_block_size"]
            for block_id in FORMAL_ANALYSIS_BLOCK_IDS
        }:
            raise ValueError(
                f"formal analysis blocks are incomplete for {key}"
            )

    cross_model_keys = [*CELL_KEYS, "round_index", "sample_index"]
    cross_model_identity = result.groupby(
        cross_model_keys,
        sort=False,
    ).agg(
        master_id_count=("master_sample_id", "nunique"),
        paired_group_id_count=("paired_group_id", "nunique"),
    )
    if not (
        (cross_model_identity["master_id_count"] == 1)
        & (cross_model_identity["paired_group_id_count"] == 1)
    ).all():
        raise ValueError(
            "model oracle files do not share the same formal sample identity"
        )
    return result


def cell_round_scores(
    oracle: pd.DataFrame,
    *,
    score_column: str,
    score_policy: str,
    expected_samples_per_round: int | None = None,
) -> pd.DataFrame:
    grouped = (
        oracle.groupby(["model_id", *ROUND_KEYS], sort=True)[score_column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mase_mean",
                "std": "mase_std",
                "count": "master_sample_count",
            }
        )
    )
    expected = (
        int(expected_samples_per_round)
        if expected_samples_per_round is not None
        else int(grouped["master_sample_count"].iloc[0])
    )
    if not (grouped["master_sample_count"] == expected).all():
        bad = grouped[grouped["master_sample_count"] != expected]
        raise ValueError(
            f"round score sample counts are not {expected}: "
            f"{bad.head().to_dict(orient='records')}"
        )
    grouped["score_policy"] = score_policy
    grouped["model_rank"] = grouped.groupby(
        ROUND_KEYS,
        sort=False,
    )["mase_mean"].rank(method="average", ascending=True)
    grouped["compatible_model_count"] = grouped.groupby(
        ROUND_KEYS,
        sort=False,
    )["model_id"].transform("count")
    return grouped


def ordering_agreement(left: np.ndarray, right: np.ndarray) -> float:
    result = stats.pairwise_ordering_agreement(left, right)
    agreement = result["agreement"]
    return float(agreement) if agreement is not None else 1.0


def rank_stability_rows(
    round_scores: pd.DataFrame,
    *,
    expected_round_count: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in round_scores.groupby(CELL_KEYS, sort=True):
        score_matrix = group.pivot(
            index="model_id",
            columns="round_index",
            values="mase_mean",
        ).sort_index()
        if score_matrix.isna().any().any():
            raise ValueError(f"incomplete round score matrix for {key}")
        rounds = list(score_matrix.columns)
        if (
            expected_round_count is not None
            and len(rounds) != expected_round_count
        ):
            raise ValueError(
                f"cell {key} does not contain {expected_round_count} rounds"
            )
        rank_matrix = score_matrix.rank(
            axis=0,
            method="average",
            ascending=True,
        )
        pair_rows: list[dict[str, Any]] = []
        for left_round, right_round in combinations(rounds, 2):
            left_rank = rank_matrix[left_round].to_numpy(dtype=float)
            right_rank = rank_matrix[right_round].to_numpy(dtype=float)
            left_score = score_matrix[left_round].to_numpy(dtype=float)
            right_score = score_matrix[right_round].to_numpy(dtype=float)
            left_top = set(
                score_matrix[left_round]
                .sort_values(kind="stable")
                .index[: min(3, len(score_matrix))]
            )
            right_top = set(
                score_matrix[right_round]
                .sort_values(kind="stable")
                .index[: min(3, len(score_matrix))]
            )
            top_k = min(3, len(score_matrix))
            pair_rows.append(
                {
                    "left_round": int(left_round),
                    "right_round": int(right_round),
                    "kendall_tau_b": float(
                        stats.kendall_tau_b(left_rank, right_rank)
                    ),
                    "pairwise_ordering_agreement": ordering_agreement(
                        left_score,
                        right_score,
                    ),
                    "exact_rank_vector": bool(
                        np.array_equal(left_rank, right_rank)
                    ),
                    "top1_agreement": bool(
                        score_matrix[left_round].idxmin()
                        == score_matrix[right_round].idxmin()
                    ),
                    "top3_overlap_rate": float(
                        len(left_top & right_top) / max(top_k, 1)
                    ),
                }
            )
        kendall = np.asarray(
            [row["kendall_tau_b"] for row in pair_rows],
            dtype=float,
        )
        agreement = np.asarray(
            [
                row["pairwise_ordering_agreement"]
                for row in pair_rows
            ],
            dtype=float,
        )
        rows.append(
            {
                **dict(zip(CELL_KEYS, key, strict=True)),
                "score_policy": group["score_policy"].iloc[0],
                "model_count": len(score_matrix),
                "models": ";".join(score_matrix.index),
                "round_count": len(rounds),
                "round_pair_count": len(pair_rows),
                "kendall_tau_b_mean": float(kendall.mean()),
                "kendall_tau_b_min": float(kendall.min()),
                "pairwise_agreement_mean": float(agreement.mean()),
                "pairwise_agreement_min": float(agreement.min()),
                "exact_rank_vector_pair_rate": float(
                    np.mean(
                        [row["exact_rank_vector"] for row in pair_rows]
                    )
                ),
                "top1_pair_agreement_rate": float(
                    np.mean([row["top1_agreement"] for row in pair_rows])
                ),
                "top3_overlap_mean": float(
                    np.mean(
                        [row["top3_overlap_rate"] for row in pair_rows]
                    )
                ),
                "passed_min_pairwise_agreement": bool(
                    agreement.min() >= MIN_PAIRWISE_AGREEMENT
                ),
                "round_pair_details": json.dumps(
                    pair_rows,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame(rows)


def score_stability_rows(
    round_scores: pd.DataFrame,
    *,
    expected_round_count: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model_id", *CELL_KEYS]
    for key, group in round_scores.groupby(keys, sort=True):
        values = group.sort_values("round_index")["mase_mean"].to_numpy(
            dtype=float
        )
        if (
            expected_round_count is not None
            and len(values) != expected_round_count
        ):
            raise ValueError(
                "score stability group has an unexpected round count: "
                f"{key}"
            )
        mean = float(values.mean())
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "score_policy": group["score_policy"].iloc[0],
                "round_count": len(values),
                "mase_round_mean": mean,
                "mase_round_std": float(values.std(ddof=1)),
                "mase_round_cv": float(
                    values.std(ddof=1) / max(abs(mean), 1e-12)
                ),
                "mase_round_min": float(values.min()),
                "mase_round_max": float(values.max()),
                "mase_max_min_ratio": float(
                    values.max() / max(values.min(), 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows)


def difficulty_stability_rows(round_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in round_scores.groupby(CELL_KEYS, sort=True):
        matrix = group.pivot(
            index="model_id",
            columns="round_index",
            values="mase_mean",
        ).sort_index()
        normalized = matrix.div(matrix.mean(axis=1), axis=0)
        multipliers = normalized.mean(axis=0).to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(CELL_KEYS, key, strict=True)),
                "score_policy": group["score_policy"].iloc[0],
                "model_count": len(matrix),
                "round_count": matrix.shape[1],
                "difficulty_multiplier_mean": float(multipliers.mean()),
                "difficulty_multiplier_std": float(
                    multipliers.std(ddof=1)
                ),
                "difficulty_multiplier_cv": float(
                    multipliers.std(ddof=1)
                    / max(abs(multipliers.mean()), 1e-12)
                ),
                "difficulty_multiplier_min": float(multipliers.min()),
                "difficulty_multiplier_max": float(multipliers.max()),
                "difficulty_multiplier_range": float(
                    multipliers.max() - multipliers.min()
                ),
                "round_multipliers": json.dumps(
                    {
                        str(round_index): float(value)
                        for round_index, value in zip(
                            matrix.columns,
                            multipliers,
                            strict=True,
                        )
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame(rows)


def cell_full_pool_scores(
    oracle: pd.DataFrame,
    *,
    score_column: str,
    score_policy: str,
    expected_samples_per_cell: int,
) -> pd.DataFrame:
    result = (
        oracle.groupby(["model_id", *CELL_KEYS], sort=True)[score_column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mase_mean",
                "std": "mase_std",
                "count": "master_sample_count",
            }
        )
    )
    if not (
        result["master_sample_count"] == expected_samples_per_cell
    ).all():
        bad = result[
            result["master_sample_count"] != expected_samples_per_cell
        ]
        raise ValueError(
            f"full-pool score sample counts are not "
            f"{expected_samples_per_cell}: "
            f"{bad.head().to_dict(orient='records')}"
        )
    result["score_policy"] = score_policy
    result["model_rank"] = result.groupby(
        CELL_KEYS,
        sort=False,
    )["mase_mean"].rank(method="average", ascending=True)
    result["compatible_model_count"] = result.groupby(
        CELL_KEYS,
        sort=False,
    )["model_id"].transform("count")
    return result


def cell_analysis_block_scores(
    oracle: pd.DataFrame,
    *,
    score_column: str,
    score_policy: str,
    expected_block_size: int,
) -> pd.DataFrame:
    keys = ["analysis_block_id", "model_id", *CELL_KEYS]
    result = (
        oracle.groupby(keys, sort=True)[score_column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mase_mean",
                "std": "mase_std",
                "count": "master_sample_count",
            }
        )
    )
    if not (result["master_sample_count"] == expected_block_size).all():
        bad = result[
            result["master_sample_count"] != expected_block_size
        ]
        raise ValueError(
            f"analysis-block score sample counts are not "
            f"{expected_block_size}: "
            f"{bad.head().to_dict(orient='records')}"
        )
    result["score_policy"] = score_policy
    result["model_rank"] = result.groupby(
        ["analysis_block_id", *CELL_KEYS],
        sort=False,
    )["mase_mean"].rank(method="average", ascending=True)
    result["compatible_model_count"] = result.groupby(
        ["analysis_block_id", *CELL_KEYS],
        sort=False,
    )["model_id"].transform("count")
    return result


def analysis_block_stability_rows(
    block_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in block_scores.groupby(CELL_KEYS, sort=True):
        matrix = group.pivot(
            index="model_id",
            columns="analysis_block_id",
            values="mase_mean",
        ).sort_index()
        if matrix.isna().any().any():
            raise ValueError(f"incomplete analysis-block matrix for {key}")
        if list(matrix.columns) != list(FORMAL_ANALYSIS_BLOCK_IDS):
            raise ValueError(
                f"cell {key} does not contain formal blocks A and B"
            )
        left = matrix[FORMAL_ANALYSIS_BLOCK_IDS[0]]
        right = matrix[FORMAL_ANALYSIS_BLOCK_IDS[1]]
        left_rank = left.rank(method="average", ascending=True)
        right_rank = right.rank(method="average", ascending=True)
        top_k = min(3, len(matrix))
        left_top = set(left.sort_values(kind="stable").index[:top_k])
        right_top = set(right.sort_values(kind="stable").index[:top_k])
        agreement = ordering_agreement(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
        )
        rows.append(
            {
                **dict(zip(CELL_KEYS, key, strict=True)),
                "score_policy": group["score_policy"].iloc[0],
                "model_count": len(matrix),
                "models": ";".join(matrix.index),
                "analysis_block_size": int(
                    group["master_sample_count"].iloc[0]
                ),
                "kendall_tau_b": float(
                    stats.kendall_tau_b(
                        left_rank.to_numpy(dtype=float),
                        right_rank.to_numpy(dtype=float),
                    )
                ),
                "pairwise_ordering_agreement": agreement,
                "exact_rank_vector": bool(
                    np.array_equal(
                        left_rank.to_numpy(dtype=float),
                        right_rank.to_numpy(dtype=float),
                    )
                ),
                "top1_agreement": bool(left.idxmin() == right.idxmin()),
                "top3_overlap_rate": float(
                    len(left_top & right_top) / max(top_k, 1)
                ),
                "passed_min_pairwise_agreement": bool(
                    agreement >= MIN_PAIRWISE_AGREEMENT
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_analysis_block_stability(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    passed = frame["passed_min_pairwise_agreement"].astype(bool)
    return {
        "cell_count": len(frame),
        "block_size": FORMAL_ANALYSIS_BLOCK_SIZE,
        "block_ids": list(FORMAL_ANALYSIS_BLOCK_IDS),
        "passed_cell_count": int(passed.sum()),
        "passed_cell_rate": float(passed.mean()),
        "pairwise_agreement_mean": float(
            frame["pairwise_ordering_agreement"].mean()
        ),
        "pairwise_agreement_minimum": float(
            frame["pairwise_ordering_agreement"].min()
        ),
        "kendall_tau_b_mean": float(frame["kendall_tau_b"].mean()),
        "exact_rank_vector_rate": float(
            frame["exact_rank_vector"].mean()
        ),
        "top1_agreement_rate": float(frame["top1_agreement"].mean()),
        "top3_overlap_mean": float(frame["top3_overlap_rate"].mean()),
    }


def synthetic_model_ranks(
    cell_scores: pd.DataFrame,
    *,
    real_dataset_ids: set[str],
) -> pd.DataFrame:
    eligible = cell_scores[
        cell_scores["dataset_id"].isin(real_dataset_ids)
    ].copy()
    aggregations: dict[str, tuple[str, str]] = {
        "synthetic_average_rank": ("model_rank", "mean"),
        "effective_capability_count": ("capability_id", "nunique"),
        "effective_intensity_count": ("intensity", "nunique"),
        "effective_rank_cell_count": ("model_rank", "count"),
    }
    if "round_index" in eligible.columns:
        aggregations["effective_round_count"] = (
            "round_index",
            "nunique",
        )
    result = eligible.groupby(
        ["dataset_id", "model_id"],
        sort=True,
    ).agg(**aggregations).reset_index()
    result["score_policy"] = eligible["score_policy"].iloc[0]
    return result


def real_model_ranks(
    real_oracle: pd.DataFrame,
    *,
    score_column: str,
    score_policy: str,
) -> pd.DataFrame:
    result = (
        real_oracle.groupby(["dataset_id", "model_id"], sort=True)
        .agg(
            real_source_mean_mase=(score_column, "mean"),
            real_source_mase_std=(score_column, "std"),
            real_master_count=("master_sample_id", "nunique"),
        )
        .reset_index()
    )
    result["real_source_rank"] = result.groupby(
        "dataset_id",
        sort=False,
    )["real_source_mean_mase"].rank(method="average", ascending=True)
    result["score_policy"] = score_policy
    return result


def source_alignment_rows(
    synthetic_ranks: pd.DataFrame,
    real_ranks: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id in sorted(
        set(synthetic_ranks["dataset_id"])
        & set(real_ranks["dataset_id"])
    ):
        synthetic = synthetic_ranks[
            synthetic_ranks["dataset_id"] == dataset_id
        ].set_index("model_id")
        real = real_ranks[
            real_ranks["dataset_id"] == dataset_id
        ].set_index("model_id")
        models = sorted(set(synthetic.index) & set(real.index))
        if len(models) < 3:
            raise ValueError(
                f"insufficient common models for {dataset_id}: {models}"
            )
        synthetic_values = synthetic.loc[
            models,
            "synthetic_average_rank",
        ].to_numpy(dtype=float)
        real_values = real.loc[
            models,
            "real_source_rank",
        ].to_numpy(dtype=float)
        synthetic_order = (
            synthetic.loc[models, "synthetic_average_rank"]
            .sort_values(kind="stable")
        )
        real_mase_order = (
            real.loc[models, "real_source_mean_mase"]
            .sort_values(kind="stable")
        )
        synthetic_top_gap = float(
            synthetic_order.iloc[1] - synthetic_order.iloc[0]
        )
        real_top_gap = float(
            real_mase_order.iloc[1] - real_mase_order.iloc[0]
        )
        top_k = min(3, len(models))
        synthetic_top = set(
            synthetic_order.index[:top_k]
        )
        real_top = set(
            real.loc[models, "real_source_rank"]
            .sort_values(kind="stable")
            .index[:top_k]
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "score_policy": synthetic["score_policy"].iloc[0],
                "model_count": len(models),
                "models": ";".join(models),
                "supported_capability_count": int(
                    synthetic["effective_capability_count"].max()
                ),
                "real_master_count": int(
                    real["real_master_count"].max()
                ),
                "spearman_rho": float(
                    stats.spearman_rank_correlation(
                        synthetic_values,
                        real_values,
                    )
                ),
                "kendall_tau_b": float(
                    stats.kendall_tau_b(
                        synthetic_values,
                        real_values,
                    )
                ),
                "top3_overlap_count": len(
                    synthetic_top & real_top
                ),
                "top3_overlap_rate": float(
                    len(synthetic_top & real_top) / top_k
                ),
                "pairwise_ordering_agreement": ordering_agreement(
                    synthetic_values,
                    real_values,
                ),
                "synthetic_top1_model": str(synthetic_order.index[0]),
                "synthetic_top1_top2_rank_gap": synthetic_top_gap,
                "real_top1_model": str(real_mase_order.index[0]),
                "real_top1_top2_mase_gap": real_top_gap,
                "real_top1_top2_relative_mase_gap": float(
                    real_top_gap
                    / max(abs(float(real_mase_order.iloc[0])), 1e-12)
                ),
                "synthetic_top_models": ";".join(
                    sorted(synthetic_top)
                ),
                "real_top_models": ";".join(sorted(real_top)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_dataset_mean(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.mean(
        rng.choice(
            array,
            size=(replicates, len(array)),
            replace=True,
        ),
        axis=1,
    )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "bootstrap_ci_low": float(low),
        "bootstrap_ci_high": float(high),
    }


def alignment_summary(
    rows: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    metrics = (
        "spearman_rho",
        "kendall_tau_b",
        "top3_overlap_rate",
        "pairwise_ordering_agreement",
    )
    return {
        "dataset_count": len(rows),
        "score_policy": rows["score_policy"].iloc[0],
        "metrics": {
            metric: bootstrap_dataset_mean(
                rows[metric].to_numpy(dtype=float),
                replicates=replicates,
                seed=seed + index,
            )
            for index, metric in enumerate(metrics)
        },
    }


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"refusing to write empty table: {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        float_format="%.10g",
    )
    os.replace(temporary, path)


def summarize_rank_stability(frame: pd.DataFrame) -> dict[str, Any]:
    passed = frame["passed_min_pairwise_agreement"].astype(bool)
    return {
        "cell_count": len(frame),
        "passed_cell_count": int(passed.sum()),
        "failed_cell_count": int((~passed).sum()),
        "passed_cell_rate": float(passed.mean()),
        "pairwise_agreement_minimum": float(
            frame["pairwise_agreement_min"].min()
        ),
        "pairwise_agreement_median": float(
            frame["pairwise_agreement_min"].median()
        ),
        "kendall_tau_b_minimum": float(
            frame["kendall_tau_b_min"].min()
        ),
        "kendall_tau_b_median": float(
            frame["kendall_tau_b_mean"].median()
        ),
        "mean_pairwise_agreement": float(
            frame["pairwise_agreement_mean"].mean()
        ),
        "mean_exact_rank_vector_pair_rate": float(
            frame["exact_rank_vector_pair_rate"].mean()
        ),
        "mean_top1_pair_agreement_rate": float(
            frame["top1_pair_agreement_rate"].mean()
        ),
        "mean_top3_overlap": float(frame["top3_overlap_mean"].mean()),
    }


def variability_summary(
    frame: pd.DataFrame,
    *,
    column: str,
) -> dict[str, float]:
    values = frame[column].to_numpy(dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def rank_breakdown(
    frame: pd.DataFrame,
    *,
    group_key: str,
) -> pd.DataFrame:
    return (
        frame.groupby(group_key, sort=True)
        .agg(
            cell_count=("dataset_id", "size"),
            passed_cell_count=(
                "passed_min_pairwise_agreement",
                "sum",
            ),
            passed_cell_rate=(
                "passed_min_pairwise_agreement",
                "mean",
            ),
            median_worst_pairwise_agreement=(
                "pairwise_agreement_min",
                "median",
            ),
            median_worst_kendall_tau_b=(
                "kendall_tau_b_min",
                "median",
            ),
            mean_top1_pair_agreement_rate=(
                "top1_pair_agreement_rate",
                "mean",
            ),
            mean_top3_overlap=("top3_overlap_mean", "mean"),
        )
        .reset_index()
    )


def render_report(summary: dict[str, Any]) -> str:
    oracle = summary["rank_stability"]["oracle_context"]
    full_pool = summary["full_pool_summary"]["oracle_context"]
    blocks = summary["analysis_block_stability"]["oracle_context"]
    score = summary["score_stability"]["oracle_context"]
    difficulty = summary["difficulty_stability"]["oracle_context"]
    alignment = summary["source_alignment"]["oracle_context"]
    metrics = alignment["metrics"]
    return "\n".join(
        [
            "# Paper v7 E2 inference analysis",
            "",
            "## E2-A formal 320-sample estimate",
            "",
            (
                f"- Main cell estimates use exactly "
                f"{full_pool['samples_per_cell_model']} samples per "
                f"model/cell across {full_pool['cell_count']} cells."
            ),
            "",
            "## E2-B two mutually exclusive 160-sample blocks",
            "",
            (
                f"- Block A/B cell ordering agreement mean/minimum: "
                f"{blocks['pairwise_agreement_mean']:.4f} / "
                f"{blocks['pairwise_agreement_minimum']:.4f}."
            ),
            (
                f"- Exact-rank / top-1 / top-3 agreement: "
                f"{blocks['exact_rank_vector_rate']:.4f} / "
                f"{blocks['top1_agreement_rate']:.4f} / "
                f"{blocks['top3_overlap_mean']:.4f}."
            ),
            "",
            "## E2-C five-round diagnostic stability",
            "",
            (
                f"- Oracle-context cells passing minimum pairwise agreement "
                f"≥ {MIN_PAIRWISE_AGREEMENT:.2f}: "
                f"{oracle['passed_cell_count']}/{oracle['cell_count']} "
                f"({oracle['passed_cell_rate']:.1%})."
            ),
            (
                f"- Median/minimum cell worst-pair agreement: "
                f"{oracle['pairwise_agreement_median']:.4f} / "
                f"{oracle['pairwise_agreement_minimum']:.4f}."
            ),
            (
                f"- Median mean Kendall τ-b / global minimum: "
                f"{oracle['kendall_tau_b_median']:.4f} / "
                f"{oracle['kendall_tau_b_minimum']:.4f}."
            ),
            (
                f"- Mean exact-rank pair rate / top-1 agreement / "
                f"top-3 overlap: "
                f"{oracle['mean_exact_rank_vector_pair_rate']:.4f} / "
                f"{oracle['mean_top1_pair_agreement_rate']:.4f} / "
                f"{oracle['mean_top3_overlap']:.4f}."
            ),
            (
                f"- Median / p90 model score CV across rounds: "
                f"{score['median']:.4f} / {score['p90']:.4f}."
            ),
            (
                f"- Median / p90 common difficulty-multiplier CV: "
                f"{difficulty['median']:.4f} / "
                f"{difficulty['p90']:.4f}."
            ),
            "",
            "## E2-D synthetic–real source-window alignment",
            "",
            (
                f"- Aligned datasets: {alignment['dataset_count']}."
            ),
            (
                f"- Mean Spearman ρ: "
                f"{metrics['spearman_rho']['mean']:.4f} "
                f"[{metrics['spearman_rho']['bootstrap_ci_low']:.4f}, "
                f"{metrics['spearman_rho']['bootstrap_ci_high']:.4f}]."
            ),
            (
                f"- Mean Kendall τ-b: "
                f"{metrics['kendall_tau_b']['mean']:.4f}."
            ),
            (
                f"- Mean top-3 overlap: "
                f"{metrics['top3_overlap_rate']['mean']:.4f}."
            ),
            (
                f"- Mean pairwise ordering agreement: "
                f"{metrics['pairwise_ordering_agreement']['mean']:.4f}."
            ),
            "",
            (
                "Interpretation is limited to dataset-local calibration "
                "source windows; this is not held-out external validity."
            ),
            "",
        ]
    )


def analyze(
    e2_dir: Path,
    real_source_dir: Path,
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    e2_dir = e2_dir.resolve()
    real_source_dir = real_source_dir.resolve()
    config_path = e2_dir / "inference_config.json"
    catalog_path = e2_dir / "inference_model_catalog.json"
    generation_config_path = e2_dir / "generation_config.json"
    sample_manifest_path = e2_dir / "sample_manifest.json"
    for path in (
        config_path,
        catalog_path,
        generation_config_path,
        sample_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing inference input: {path}")
    config = read_json(config_path)
    generation_config = read_json(generation_config_path)
    protocol = validate_formal_generation_config(generation_config)
    sample_manifest = read_json(sample_manifest_path)
    generation_record = sample_manifest.get("files", {}).get(
        "generation_config.json"
    )
    if (
        not isinstance(generation_record, dict)
        or generation_record.get("sha256")
        != file_sha256(generation_config_path)
    ):
        raise ValueError(
            "sample_manifest generation_config identity mismatch"
        )
    synthetic_input = config.get("synthetic_input")
    if not isinstance(synthetic_input, dict):
        raise ValueError("inference_config is missing synthetic_input identity")
    if (
        synthetic_input.get("manifest_sha256")
        != file_sha256(sample_manifest_path)
    ):
        raise ValueError(
            "inference_config sample_manifest identity mismatch"
        )
    sample_record = sample_manifest.get("files", {}).get("samples.jsonl")
    if (
        not isinstance(sample_record, dict)
        or sample_record.get("sha256") != synthetic_input.get("sha256")
    ):
        raise ValueError("inference_config synthetic sample identity mismatch")
    models = [str(model) for model in config["requested_models"]]
    all_models = [*BASELINE_MODELS, *models]
    model_status = read_json(e2_dir / "model_status.json")["models"]
    real_status = read_json(
        e2_dir / "real_source_model_status.json"
    )["models"]
    for model_id in models:
        for label, status in (
            ("synthetic", model_status.get(model_id)),
            ("real_source", real_status.get(model_id)),
        ):
            if (
                not isinstance(status, dict)
                or status.get("status") != "complete"
                or int(status["succeeded_count"])
                != int(status["compatible_sample_count"])
            ):
                raise RuntimeError(
                    f"{model_id} {label} inference is incomplete: {status}"
                )

    oracle_records: list[dict[str, Any]] = []
    synthetic_paths: list[Path] = []
    real_paths: list[Path] = []
    for model_id in all_models:
        for prediction_kind in ("synthetic", "real_source"):
            source = prediction_path(
                e2_dir,
                model_id,
                prediction_kind=prediction_kind,
            )
            if not source.is_file():
                raise FileNotFoundError(f"missing prediction file: {source}")
            destination = oracle_path(
                e2_dir,
                model_id,
                prediction_kind=prediction_kind,
            )
            oracle_records.append(
                compact_prediction_file(
                    source,
                    destination,
                    model_id=model_id,
                    prediction_kind=prediction_kind,
                )
            )
            if model_id in models:
                (
                    synthetic_paths
                    if prediction_kind == "synthetic"
                    else real_paths
                ).append(destination)

    synthetic_oracle = load_oracle_rows(
        synthetic_paths,
        prediction_kind="synthetic",
    )
    synthetic_oracle = validate_formal_oracle_identity(
        synthetic_oracle,
        protocol=protocol,
    )
    real_oracle = load_oracle_rows(
        real_paths,
        prediction_kind="real_source",
    )
    output_frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    score_summaries: dict[str, Any] = {}
    difficulty_summaries: dict[str, Any] = {}
    alignment_summaries: dict[str, Any] = {}
    full_pool_summaries: dict[str, Any] = {}
    analysis_block_summaries: dict[str, Any] = {}
    real_dataset_ids = set(real_oracle["dataset_id"].unique())
    for score_policy, score_column in (
        ("oracle_context", "oracle_mase"),
        ("fixed_l504", "fixed_l504_mase"),
    ):
        suffix = "" if score_policy == "oracle_context" else "_l504"
        round_scores = cell_round_scores(
            synthetic_oracle,
            score_column=score_column,
            score_policy=score_policy,
            expected_samples_per_round=protocol["samples_per_round"],
        )
        full_pool_scores = cell_full_pool_scores(
            synthetic_oracle,
            score_column=score_column,
            score_policy=score_policy,
            expected_samples_per_cell=protocol["total_per_intensity"],
        )
        block_scores = cell_analysis_block_scores(
            synthetic_oracle,
            score_column=score_column,
            score_policy=score_policy,
            expected_block_size=protocol["analysis_block_size"],
        )
        block_stability = analysis_block_stability_rows(
            block_scores
        )
        rank_stability = rank_stability_rows(
            round_scores,
            expected_round_count=protocol["round_count"],
        )
        score_stability = score_stability_rows(
            round_scores,
            expected_round_count=protocol["round_count"],
        )
        difficulty_stability = difficulty_stability_rows(round_scores)
        synthetic_ranks = synthetic_model_ranks(
            full_pool_scores,
            real_dataset_ids=real_dataset_ids,
        )
        real_ranks = real_model_ranks(
            real_oracle,
            score_column=score_column,
            score_policy=score_policy,
        )
        alignment = source_alignment_rows(
            synthetic_ranks,
            real_ranks,
        )
        output_frames.update(
            {
                f"cell_round_scores{suffix}.csv": round_scores,
                f"cell_full_pool_scores{suffix}.csv": full_pool_scores,
                f"cell_analysis_block_scores{suffix}.csv": block_scores,
                f"cell_analysis_block_stability{suffix}.csv": (
                    block_stability
                ),
                f"cell_rank_stability{suffix}.csv": rank_stability,
                f"cell_score_stability{suffix}.csv": score_stability,
                f"cell_difficulty_stability{suffix}.csv": (
                    difficulty_stability
                ),
                f"synthetic_model_ranks{suffix}.csv": synthetic_ranks,
                f"real_source_model_ranks{suffix}.csv": real_ranks,
                f"synthetic_real_source_alignment{suffix}.csv": alignment,
                f"cell_rank_stability_by_capability{suffix}.csv": (
                    rank_breakdown(
                        rank_stability,
                        group_key="capability_id",
                    )
                ),
                f"cell_rank_stability_by_intensity{suffix}.csv": (
                    rank_breakdown(
                        rank_stability,
                        group_key="intensity",
                    )
                ),
                f"cell_rank_stability_by_task{suffix}.csv": (
                    rank_breakdown(
                        rank_stability,
                        group_key="task_id",
                    )
                ),
                f"cell_rank_stability_by_dataset{suffix}.csv": (
                    rank_breakdown(
                        rank_stability,
                        group_key="dataset_id",
                    )
                ),
            }
        )
        summaries[score_policy] = summarize_rank_stability(
            rank_stability
        )
        score_summaries[score_policy] = variability_summary(
            score_stability,
            column="mase_round_cv",
        )
        difficulty_summaries[score_policy] = variability_summary(
            difficulty_stability,
            column="difficulty_multiplier_cv",
        )
        full_pool_summaries[score_policy] = {
            "cell_model_row_count": len(full_pool_scores),
            "samples_per_cell_model": protocol["total_per_intensity"],
            "cell_count": int(
                full_pool_scores[CELL_KEYS].drop_duplicates().shape[0]
            ),
            "model_count": int(full_pool_scores["model_id"].nunique()),
        }
        analysis_block_summaries[score_policy] = (
            summarize_analysis_block_stability(block_stability)
        )
        alignment_summaries[score_policy] = alignment_summary(
            alignment,
            replicates=bootstrap_replicates,
            seed=2026071901
            + (0 if score_policy == "oracle_context" else 100),
        )

    for filename, frame in output_frames.items():
        write_csv(e2_dir / filename, frame)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "baseline_models": list(BASELINE_MODELS),
        "oracle_score_files": oracle_records,
        "formal_generation_protocol": protocol,
        "full_pool_summary": full_pool_summaries,
        "analysis_block_stability": analysis_block_summaries,
        "rank_stability": summaries,
        "score_stability": score_summaries,
        "difficulty_stability": difficulty_summaries,
        "source_alignment": alignment_summaries,
        "minimum_pairwise_agreement": MIN_PAIRWISE_AGREEMENT,
        "bootstrap_replicates": bootstrap_replicates,
        "interpretation": (
            "source-window construct alignment using formal calibration "
            "references; not held-out external validity"
        ),
        "table_rows": {
            filename: len(frame)
            for filename, frame in output_frames.items()
        },
    }
    detailed_split_path = e2_dir / "split_bank_reliability/summary.json"
    summary["detailed_split_bank_analysis"] = {
        "path": display_path(detailed_split_path),
        "status": "complete" if detailed_split_path.is_file() else "pending",
        "required_command": (
            "python scripts/analyze_paper_e2_split_bank_reliability.py "
            "--bank-sizes 160"
        ),
    }
    if detailed_split_path.is_file():
        detailed_split = read_json(detailed_split_path)
        partition = detailed_split.get("formal_v7_partition")
        if not isinstance(partition, dict) or any(
            partition.get(key) != value
            for key, value in {
                "total_per_intensity": FORMAL_TOTAL_PER_INTENSITY,
                "block_size": FORMAL_ANALYSIS_BLOCK_SIZE,
                "mutually_exclusive": True,
                "complete_partition": True,
            }.items()
        ):
            raise ValueError(
                "existing detailed split-bank analysis has stale identity"
            )
        summary["detailed_split_bank_analysis"]["sha256"] = file_sha256(
            detailed_split_path
        )
    inference_shards_path = e2_dir / "inference_shards.json"
    if inference_shards_path.is_file():
        shard_payload = read_json(inference_shards_path)
        summary["distributed_inference"] = {
            "path": display_path(inference_shards_path),
            "sha256": file_sha256(inference_shards_path),
            "shard_count": len(shard_payload.get("shards", [])),
        }
    write_json(e2_dir / "inference_summary.json", summary)
    (e2_dir / "inference_report.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )
    manifest_files = [
        "inference_config.json",
        "inference_model_catalog.json",
        "generation_config.json",
        "sample_manifest.json",
        "model_status.json",
        "real_source_model_status.json",
        "baseline_status.json",
        "real_source_baseline_status.json",
        "inference_summary.json",
        "inference_report.md",
        *sorted(output_frames),
    ]
    if inference_shards_path.is_file():
        manifest_files.append("inference_shards.json")
    manifest = {
        "schema_version": "paper_v7_e2_inference_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "analysis_sha256": file_sha256(Path(__file__)),
        "real_source_manifest": {
            "path": display_path(real_source_dir / "manifest.json"),
            "sha256": file_sha256(real_source_dir / "manifest.json"),
        },
        "files": {
            filename: {
                "size_bytes": (e2_dir / filename).stat().st_size,
                "sha256": file_sha256(e2_dir / filename),
            }
            for filename in manifest_files
        },
        "prediction_files": {
            prediction_kind: {
                model_id: {
                    "path": display_path(
                        prediction_path(
                            e2_dir,
                            model_id,
                            prediction_kind=prediction_kind,
                        )
                    ),
                    "size_bytes": prediction_path(
                        e2_dir,
                        model_id,
                        prediction_kind=prediction_kind,
                    ).stat().st_size,
                    "sha256": file_sha256(
                        prediction_path(
                            e2_dir,
                            model_id,
                            prediction_kind=prediction_kind,
                        )
                    ),
                }
                for model_id in all_models
            }
            for prediction_kind in ("synthetic", "real_source")
        },
    }
    write_json(e2_dir / "inference_manifest.json", manifest)
    return summary


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 1000:
        raise ValueError("bootstrap-replicates must be at least 1000")
    summary = analyze(
        args.e2_dir,
        args.real_source_dir,
        bootstrap_replicates=int(args.bootstrap_replicates),
    )
    oracle = summary["rank_stability"]["oracle_context"]
    print(
        f"E2 analysis complete: cell stability "
        f"{oracle['passed_cell_count']}/{oracle['cell_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
