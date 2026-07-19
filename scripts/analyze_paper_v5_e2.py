#!/usr/bin/env python3
"""Analyze formal Paper v5 E2 view predictions after inference completes."""
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


DEFAULT_E2_DIR = REPO_ROOT / "runtime/paper_exp/v5/E2_dynamic_stability"
DEFAULT_REAL_SOURCE_DIR = (
    REPO_ROOT / "runtime/paper_exp/v5/02_real_source_window_suite"
)
CONTEXT_LENGTHS = (96, 168, 336, 504)
MIN_PAIRWISE_AGREEMENT = 0.95
BOOTSTRAP_REPLICATES = 10_000
SCHEMA_VERSION = "paper_v5_e2_analysis.v1"
BASELINE_MODELS = ("naive", "seasonal_naive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse four contexts and analyze Paper v5 E2 cell stability "
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
    for row in iter_jsonl(source):
        if str(row["model_id"]) != model_id:
            raise ValueError(f"model mismatch in {source}")
        master_id = str(row["master_sample_id"])
        context = int(row["context_length"])
        if context not in CONTEXT_LENGTHS:
            raise ValueError(f"unexpected context in {source}: {context}")
        mase = row.get("metrics", {}).get("mase")
        if mase is None or not math.isfinite(float(mase)):
            raise ValueError(
                f"missing finite MASE for {model_id}/{master_id}/L{context}"
            )
        record = masters.setdefault(
            master_id,
            {
                "schema_version": "paper_v5_e2_oracle_sample_score.v1",
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
            "mase": float(mase),
            "mae": float(row["metrics"]["mae"]),
            "mse": float(row["metrics"]["mse"]),
        }
        if "request_seconds" in row:
            context_row["request_seconds"] = float(row["request_seconds"])
            context_row["request_attempts"] = int(row["request_attempts"])
        record["contexts"][str(context)] = context_row
        if prediction_kind == "synthetic":
            record.update(
                {
                    "capability_id": row["capability_id"],
                    "intensity": int(row["intensity"]),
                    "round_index": int(row["round_index"]),
                    "round_seed": int(row["round_seed"]),
                    "sample_index": int(row["sample_index"]),
                    "paired_group_id": row["paired_group_id"],
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
            selected_context = min(
                CONTEXT_LENGTHS,
                key=lambda context: (
                    record["contexts"][str(context)]["mase"],
                    context,
                ),
            )
            selected = record["contexts"][str(selected_context)]
            fixed = record["contexts"][str(max(CONTEXT_LENGTHS))]
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


def cell_round_scores(
    oracle: pd.DataFrame,
    *,
    score_column: str,
    score_policy: str,
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
    if not (grouped["master_sample_count"] == 32).all():
        bad = grouped[grouped["master_sample_count"] != 32]
        raise ValueError(
            f"round score sample counts are not 32: "
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


def rank_stability_rows(round_scores: pd.DataFrame) -> pd.DataFrame:
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
        if len(rounds) != 5:
            raise ValueError(f"cell {key} does not contain five rounds")
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


def score_stability_rows(round_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model_id", *CELL_KEYS]
    for key, group in round_scores.groupby(keys, sort=True):
        values = group.sort_values("round_index")["mase_mean"].to_numpy(
            dtype=float
        )
        if len(values) != 5:
            raise ValueError(f"score stability group lacks five rounds: {key}")
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


def synthetic_model_ranks(
    round_scores: pd.DataFrame,
    *,
    real_dataset_ids: set[str],
) -> pd.DataFrame:
    eligible = round_scores[
        round_scores["dataset_id"].isin(real_dataset_ids)
    ].copy()
    result = (
        eligible.groupby(["dataset_id", "model_id"], sort=True)
        .agg(
            synthetic_average_rank=("model_rank", "mean"),
            effective_capability_count=("capability_id", "nunique"),
            effective_intensity_count=("intensity", "nunique"),
            effective_round_count=("round_index", "nunique"),
            effective_rank_cell_count=("model_rank", "count"),
        )
        .reset_index()
    )
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
        top_k = min(3, len(models))
        synthetic_top = set(
            synthetic.loc[models, "synthetic_average_rank"]
            .sort_values(kind="stable")
            .index[:top_k]
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
    }


def render_report(summary: dict[str, Any]) -> str:
    oracle = summary["rank_stability"]["oracle_context"]
    alignment = summary["source_alignment"]["oracle_context"]
    metrics = alignment["metrics"]
    return "\n".join(
        [
            "# Paper v5 E2 inference analysis",
            "",
            "## E2-A cell-level dynamic stability",
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
            "",
            "## E2-B synthetic–real source-window alignment",
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
    for path in (config_path, catalog_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing inference input: {path}")
    config = read_json(config_path)
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
    real_oracle = load_oracle_rows(
        real_paths,
        prediction_kind="real_source",
    )
    output_frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    alignment_summaries: dict[str, Any] = {}
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
        )
        rank_stability = rank_stability_rows(round_scores)
        score_stability = score_stability_rows(round_scores)
        difficulty_stability = difficulty_stability_rows(round_scores)
        synthetic_ranks = synthetic_model_ranks(
            round_scores,
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
                f"cell_rank_stability{suffix}.csv": rank_stability,
                f"cell_score_stability{suffix}.csv": score_stability,
                f"cell_difficulty_stability{suffix}.csv": (
                    difficulty_stability
                ),
                f"synthetic_model_ranks{suffix}.csv": synthetic_ranks,
                f"real_source_model_ranks{suffix}.csv": real_ranks,
                f"synthetic_real_source_alignment{suffix}.csv": alignment,
            }
        )
        summaries[score_policy] = summarize_rank_stability(
            rank_stability
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
        "rank_stability": summaries,
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
    write_json(e2_dir / "inference_summary.json", summary)
    (e2_dir / "inference_report.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )
    manifest_files = [
        "inference_config.json",
        "inference_model_catalog.json",
        "model_status.json",
        "real_source_model_status.json",
        "baseline_status.json",
        "real_source_baseline_status.json",
        "inference_summary.json",
        "inference_report.md",
        *sorted(output_frames),
    ]
    manifest = {
        "schema_version": "paper_v5_e2_inference_manifest.v1",
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
