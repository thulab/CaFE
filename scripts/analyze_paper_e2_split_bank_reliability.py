#!/usr/bin/env python3
"""Estimate E2 measurement reliability by splitting one paired-group pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_paper_v5_e2 as e2  # noqa: E402
import analyze_paper_v5_e2_seed_bank_reliability as reliability  # noqa: E402
import run_paper_e2_dynamic_stability as stats  # noqa: E402


SCHEMA_VERSION = "paper_e2_split_bank_reliability.v1"
DEFAULT_E2_DIR = REPO_ROOT / "runtime/paper_exp/v6/E2_dynamic_stability"
DEFAULT_BANK_SIZES = (32, 48, 64, 80)
DEFAULT_SPLIT_SEED = 20260720
DEFAULT_MINIMUM_AGREEMENT = 0.80
NORMAL_95 = 1.959963984540054
PROFILE_KEYS = ["dataset_id", "task_id", "capability_id"]
CELL_KEYS = [*PROFILE_KEYS, "intensity"]
MODEL_CELL_KEYS = ["model_id", *CELL_KEYS]
SCORE_POLICIES = {
    "oracle_context": "oracle_mase",
    "fixed_l504": "fixed_l504_mase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split one E2 paired-group pool into two disjoint banks and "
            "compare continuous scores, capability profiles, tie-aware "
            "model contrasts, and rankings."
        )
    )
    parser.add_argument("--e2-dir", type=Path, default=DEFAULT_E2_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <e2-dir>/split_bank_reliability.",
    )
    parser.add_argument(
        "--bank-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BANK_SIZES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Defaults to requested_models in inference_config.json.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Analyze only these dataset IDs; defaults to all datasets.",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=0,
        help=(
            "Additionally evaluate this many deterministic random disjoint "
            "splits per bank size."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
    )
    parser.add_argument(
        "--minimum-agreement",
        type=float,
        default=DEFAULT_MINIMUM_AGREEMENT,
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def requested_models(e2_dir: Path, explicit: list[str] | None) -> list[str]:
    if explicit is not None:
        models = [str(model_id) for model_id in explicit]
    else:
        config = json.loads(
            (e2_dir / "inference_config.json").read_text(encoding="utf-8")
        )
        models = [str(model_id) for model_id in config["requested_models"]]
    if len(models) < 2 or len(models) != len(set(models)):
        raise ValueError("at least two unique models are required")
    return models


def ensure_oracle_paths(e2_dir: Path, models: list[str]) -> list[Path]:
    paths: list[Path] = []
    for model_id in models:
        destination = e2.oracle_path(
            e2_dir,
            model_id,
            prediction_kind="synthetic",
        )
        if not destination.is_file():
            source = e2.prediction_path(
                e2_dir,
                model_id,
                prediction_kind="synthetic",
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            e2.compact_prediction_file(
                source,
                destination,
                model_id=model_id,
                prediction_kind="synthetic",
            )
        paths.append(destination)
    return paths


def load_oracle_pool(
    paths: Iterable[Path],
    *,
    datasets: set[str] | None,
) -> pd.DataFrame:
    columns = [
        "model_id",
        "master_sample_id",
        "dataset_id",
        "task_id",
        "capability_id",
        "intensity",
        "paired_group_id",
        "oracle_mase",
        "fixed_l504_mase",
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in e2.iter_jsonl(path):
            dataset_id = str(row["dataset_id"])
            if datasets is not None and dataset_id not in datasets:
                continue
            rows.append({column: row[column] for column in columns})
    if not rows:
        raise ValueError("no synthetic oracle scores matched the selection")
    frame = pd.DataFrame(rows)
    frame["intensity"] = frame["intensity"].astype(int)
    for column in ("oracle_mase", "fixed_l504_mase"):
        frame[column] = frame[column].astype(float)
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"non-finite score in {column}")

    group_order = (
        frame[[*PROFILE_KEYS, "paired_group_id"]]
        .drop_duplicates()
        .sort_values([*PROFILE_KEYS, "paired_group_id"], kind="stable")
    )
    group_order["pool_index"] = group_order.groupby(
        PROFILE_KEYS,
        sort=False,
    ).cumcount()
    frame = frame.merge(
        group_order,
        on=[*PROFILE_KEYS, "paired_group_id"],
        how="left",
        validate="many_to_one",
    )
    validate_oracle_pool(frame)
    return frame


def validate_oracle_pool(frame: pd.DataFrame) -> None:
    duplicate_keys = [*MODEL_CELL_KEYS, "paired_group_id"]
    if frame.duplicated(duplicate_keys).any():
        raise ValueError("duplicate model/cell/paired-group oracle score")

    profile_sizes = (
        frame.groupby(PROFILE_KEYS, sort=False)["paired_group_id"]
        .nunique()
        .rename("profile_size")
    )
    if profile_sizes.empty or int(profile_sizes.min()) < 2:
        raise ValueError("each profile requires at least two paired groups")

    observed = (
        frame.groupby(MODEL_CELL_KEYS, sort=False)["paired_group_id"]
        .nunique()
        .rename("observed")
        .reset_index()
        .merge(
            profile_sizes.reset_index(),
            on=PROFILE_KEYS,
            how="left",
            validate="many_to_one",
        )
    )
    incomplete = observed[observed["observed"] != observed["profile_size"]]
    if not incomplete.empty:
        raise ValueError(
            "model/cell does not cover the complete paired-group pool: "
            f"{incomplete.head().to_dict(orient='records')}"
        )

    models_per_cell = frame.groupby(CELL_KEYS, sort=False)["model_id"].nunique()
    if (models_per_cell < 2).any():
        raise ValueError("every analyzed cell requires at least two models")


def pool_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[[*PROFILE_KEYS, "paired_group_id", "pool_index"]]
        .drop_duplicates()
        .sort_values([*PROFILE_KEYS, "pool_index"], kind="stable")
        .reset_index(drop=True)
    )


def stable_profile_seed(
    split_seed: int,
    repeat_index: int,
    profile: tuple[Any, ...],
) -> int:
    payload = "|".join(
        [str(split_seed), str(repeat_index), *(str(value) for value in profile)]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def split_assignments(
    catalog: pd.DataFrame,
    *,
    bank_size: int,
    split_kind: str,
    repeat_index: int,
    split_seed: int,
) -> pd.DataFrame:
    if bank_size < 2:
        raise ValueError("bank size must be at least two")
    if split_kind not in {"ordered", "random"}:
        raise ValueError(f"unsupported split kind: {split_kind}")

    rows: list[dict[str, Any]] = []
    for profile, group in catalog.groupby(PROFILE_KEYS, sort=True):
        ordered = group.sort_values("pool_index", kind="stable")
        group_ids = ordered["paired_group_id"].to_numpy(dtype=object)
        if 2 * bank_size > len(group_ids):
            raise ValueError(
                f"bank size {bank_size} requires {2 * bank_size} groups, "
                f"but profile {profile} contains {len(group_ids)}"
            )
        if split_kind == "random":
            rng = np.random.default_rng(
                stable_profile_seed(split_seed, repeat_index, profile)
            )
            group_ids = group_ids[rng.permutation(len(group_ids))]
        left = group_ids[:bank_size]
        right = group_ids[-bank_size:]
        for bank_id, selected in (("A", left), ("B", right)):
            rows.extend(
                {
                    **dict(zip(PROFILE_KEYS, profile, strict=True)),
                    "paired_group_id": str(group_id),
                    "bank_id": bank_id,
                }
                for group_id in selected
            )
    assignments = pd.DataFrame(rows)
    if assignments.duplicated(
        [*PROFILE_KEYS, "paired_group_id"]
    ).any():
        raise AssertionError("split banks overlap")
    return assignments


def cell_model_scores(
    split_frame: pd.DataFrame,
    *,
    score_column: str,
    bank_size: int,
) -> pd.DataFrame:
    result = (
        split_frame.groupby(["bank_id", *MODEL_CELL_KEYS], sort=True)[
            score_column
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mase_mean",
                "std": "mase_std",
                "count": "sample_count",
            }
        )
    )
    if not (result["sample_count"] == bank_size).all():
        raise ValueError("split cell score has an unexpected sample count")
    result["mase_se"] = result["mase_std"] / np.sqrt(
        result["sample_count"]
    )
    result["mase_ci_low"] = (
        result["mase_mean"] - NORMAL_95 * result["mase_se"]
    )
    result["mase_ci_high"] = (
        result["mase_mean"] + NORMAL_95 * result["mase_se"]
    )
    if (result["mase_mean"] <= 0).any():
        raise ValueError("MASE means must be positive")
    result["log_mase"] = np.log(result["mase_mean"])
    result["relative_log_mase"] = result["log_mase"] - result.groupby(
        ["bank_id", *CELL_KEYS],
        sort=False,
    )["log_mase"].transform("mean")
    result["model_rank"] = result.groupby(
        ["bank_id", *CELL_KEYS],
        sort=False,
    )["mase_mean"].rank(method="average", ascending=True)
    return result


def tie_aware_pair_states(
    split_frame: pd.DataFrame,
    *,
    score_column: str,
    bank_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in split_frame.groupby(
        ["bank_id", *CELL_KEYS],
        sort=True,
    ):
        bank_id, *cell_values = key
        matrix = group.pivot(
            index="paired_group_id",
            columns="model_id",
            values=score_column,
        )
        if len(matrix) != bank_size or matrix.isna().any().any():
            raise ValueError(f"incomplete paired model matrix for {key}")
        for left_model, right_model in combinations(
            sorted(matrix.columns),
            2,
        ):
            difference = (
                matrix[left_model].to_numpy(dtype=float)
                - matrix[right_model].to_numpy(dtype=float)
            )
            mean = float(np.mean(difference))
            se = float(np.std(difference, ddof=1) / math.sqrt(bank_size))
            low = mean - NORMAL_95 * se
            high = mean + NORMAL_95 * se
            if high < 0:
                state = "left_better"
            elif low > 0:
                state = "right_better"
            else:
                state = "indistinguishable"
            rows.append(
                {
                    **dict(
                        zip(CELL_KEYS, cell_values, strict=True)
                    ),
                    "left_model": left_model,
                    "right_model": right_model,
                    "bank_id": bank_id,
                    "mean_mase_difference": mean,
                    "difference_se": se,
                    "ci_low": low,
                    "ci_high": high,
                    "state": state,
                }
            )
    return pd.DataFrame(rows)


def rank_reliability(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    *,
    minimum_agreement: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, left_group in scores_a.groupby(CELL_KEYS, sort=True):
        right_group = scores_b
        for column, value in zip(CELL_KEYS, cell, strict=True):
            right_group = right_group[right_group[column] == value]
        left = left_group.set_index("model_id").sort_index()
        right = right_group.set_index("model_id").sort_index()
        if list(left.index) != list(right.index):
            raise ValueError(f"split-bank model mismatch for {cell}")
        left_score = left["mase_mean"].to_numpy(dtype=float)
        right_score = right["mase_mean"].to_numpy(dtype=float)
        left_rank = left["model_rank"].to_numpy(dtype=float)
        right_rank = right["model_rank"].to_numpy(dtype=float)
        ordering = stats.pairwise_ordering_agreement(
            left_score,
            right_score,
        )["agreement"]
        agreement = 1.0 if ordering is None else float(ordering)
        top_k = min(3, len(left))
        left_top = set(left["mase_mean"].nsmallest(top_k).index)
        right_top = set(right["mase_mean"].nsmallest(top_k).index)
        rows.append(
            {
                **dict(zip(CELL_KEYS, cell, strict=True)),
                "model_count": len(left),
                "pair_count": len(left) * (len(left) - 1) // 2,
                "pairwise_ordering_agreement": agreement,
                "passed": agreement >= minimum_agreement,
                "spearman_rho": float(
                    stats.spearman_rank_correlation(
                        left_rank,
                        right_rank,
                    )
                ),
                "kendall_tau_b": float(
                    stats.kendall_tau_b(left_rank, right_rank)
                ),
                "exact_rank_vector": bool(
                    np.array_equal(left_rank, right_rank)
                ),
                "top1_agreement": bool(
                    left["mase_mean"].idxmin()
                    == right["mase_mean"].idxmin()
                ),
                "top3_overlap_rate": float(
                    len(left_top & right_top) / top_k
                ),
            }
        )
    frame = pd.DataFrame(rows)
    agreement = frame["pairwise_ordering_agreement"].to_numpy(dtype=float)
    summary = {
        "cell_count": len(frame),
        "passed_cell_count": int(frame["passed"].sum()),
        "passed_cell_rate": float(frame["passed"].mean()),
        "pairwise_ordering_agreement": {
            "mean": float(np.mean(agreement)),
            "median": float(np.median(agreement)),
            "minimum": float(np.min(agreement)),
            "maximum": float(np.max(agreement)),
        },
        "spearman_rho": {
            "mean": float(frame["spearman_rho"].mean()),
            "median": float(frame["spearman_rho"].median()),
        },
        "kendall_tau_b": {
            "mean": float(frame["kendall_tau_b"].mean()),
            "median": float(frame["kendall_tau_b"].median()),
        },
        "exact_rank_vector_rate": float(frame["exact_rank_vector"].mean()),
        "top1_agreement_rate": float(frame["top1_agreement"].mean()),
        "top3_overlap_mean": float(frame["top3_overlap_rate"].mean()),
    }
    return frame, summary


def analyze_split(
    oracle: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    score_column: str,
    bank_size: int,
    minimum_agreement: float,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    split_frame = oracle.merge(
        assignments,
        on=[*PROFILE_KEYS, "paired_group_id"],
        how="inner",
        validate="many_to_one",
    )
    scores = cell_model_scores(
        split_frame,
        score_column=score_column,
        bank_size=bank_size,
    )
    scores_a = scores[scores["bank_id"] == "A"].copy()
    scores_b = scores[scores["bank_id"] == "B"].copy()
    compared = reliability.compare_cell_model_scores(scores_a, scores_b)

    profiles = reliability.capability_profiles(scores)
    profile_a = profiles[profiles["bank_id"] == "A"].copy()
    profile_b = profiles[profiles["bank_id"] == "B"].copy()
    profile_comparison, profile_summary = (
        reliability.compare_capability_profiles(profile_a, profile_b)
    )

    pair_states = tie_aware_pair_states(
        split_frame,
        score_column=score_column,
        bank_size=bank_size,
    )
    pair_a = pair_states[pair_states["bank_id"] == "A"].copy()
    pair_b = pair_states[pair_states["bank_id"] == "B"].copy()
    pair_comparison, pair_summary = reliability.compare_pair_states(
        pair_a,
        pair_b,
    )
    rank_comparison, rank_summary = rank_reliability(
        scores_a,
        scores_b,
        minimum_agreement=minimum_agreement,
    )
    summary = reliability.summarize_continuous(
        compared,
        profile_summary,
        pair_summary,
    )
    summary["formal_rank_reliability"] = rank_summary
    return summary, {
        "cell_model_scores": scores,
        "cell_model_reliability": compared,
        "capability_profiles": profiles,
        "capability_profile_reliability": profile_comparison,
        "tie_aware_model_contrasts": pair_comparison,
        "rank_reliability": rank_comparison,
    }


def flat_summary_row(
    summary: dict[str, Any],
    *,
    bank_size: int,
    split_kind: str,
    repeat_index: int,
    score_policy: str,
) -> dict[str, Any]:
    raw = summary["raw_mase_reliability"]
    normalized = summary["cell_normalized_score_reliability"]
    profile = summary["capability_profile_reliability"]["overall"]
    relative = summary["symmetric_relative_difference"]
    pairs = summary["tie_aware_model_contrasts"]
    rank = summary["formal_rank_reliability"]
    return {
        "bank_size": bank_size,
        "split_kind": split_kind,
        "repeat_index": repeat_index,
        "score_policy": score_policy,
        "cell_count": rank["cell_count"],
        "rank_pairwise_agreement_mean": (
            rank["pairwise_ordering_agreement"]["mean"]
        ),
        "rank_pairwise_agreement_median": (
            rank["pairwise_ordering_agreement"]["median"]
        ),
        "rank_pairwise_agreement_minimum": (
            rank["pairwise_ordering_agreement"]["minimum"]
        ),
        "rank_passed_cell_rate": rank["passed_cell_rate"],
        "rank_spearman_mean": rank["spearman_rho"]["mean"],
        "rank_kendall_mean": rank["kendall_tau_b"]["mean"],
        "exact_rank_vector_rate": rank["exact_rank_vector_rate"],
        "top1_agreement_rate": rank["top1_agreement_rate"],
        "top3_overlap_mean": rank["top3_overlap_mean"],
        "raw_mase_lin_ccc": raw["lin_ccc"],
        "raw_mase_spearman": raw["spearman_rho"],
        "normalized_score_lin_ccc": normalized["lin_ccc"],
        "normalized_score_spearman": normalized["spearman_rho"],
        "capability_profile_lin_ccc": profile["lin_ccc"],
        "capability_profile_spearman": profile["spearman_rho"],
        "symmetric_relative_difference_median": relative["median"],
        "symmetric_relative_difference_p90": relative["p90"],
        "tie_state_match_rate": pairs["state_match_rate"],
        "tie_decisive_direction_agreement": (
            pairs["both_decisive_directional_agreement"]
        ),
        "tie_direction_conflict_count": pairs[
            "direction_conflict_count"
        ],
    }


def distribution(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def random_repeat_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    metrics = [
        "rank_pairwise_agreement_mean",
        "rank_passed_cell_rate",
        "rank_spearman_mean",
        "top1_agreement_rate",
        "top3_overlap_mean",
        "normalized_score_lin_ccc",
        "capability_profile_lin_ccc",
        "symmetric_relative_difference_median",
        "tie_state_match_rate",
    ]
    result: dict[str, Any] = {}
    random_frame = frame[frame["split_kind"] == "random"]
    for (policy, bank_size), group in random_frame.groupby(
        ["score_policy", "bank_size"],
        sort=True,
    ):
        result.setdefault(str(policy), {})[str(int(bank_size))] = {
            "repeat_count": len(group),
            "metrics": {
                metric: distribution(group[metric])
                for metric in metrics
            },
        }
    return result


def add_split_metadata(
    frame: pd.DataFrame,
    *,
    bank_size: int,
    split_kind: str,
    repeat_index: int,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "repeat_index", repeat_index)
    result.insert(0, "split_kind", split_kind)
    result.insert(0, "bank_size", bank_size)
    return result


def rank_breakdown(
    frame: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    return (
        frame.groupby(
            ["bank_size", "split_kind", "repeat_index", group_column],
            sort=True,
        )
        .agg(
            cell_count=("passed", "size"),
            passed_cell_count=("passed", "sum"),
            passed_cell_rate=("passed", "mean"),
            pairwise_agreement_mean=(
                "pairwise_ordering_agreement",
                "mean",
            ),
            pairwise_agreement_median=(
                "pairwise_ordering_agreement",
                "median",
            ),
            pairwise_agreement_minimum=(
                "pairwise_ordering_agreement",
                "min",
            ),
            spearman_mean=("spearman_rho", "mean"),
            kendall_mean=("kendall_tau_b", "mean"),
            exact_rank_vector_rate=("exact_rank_vector", "mean"),
            top1_agreement_rate=("top1_agreement", "mean"),
            top3_overlap_mean=("top3_overlap_rate", "mean"),
        )
        .reset_index()
    )


def render_report(summary: dict[str, Any], flat: pd.DataFrame) -> str:
    lines = [
        "# E2 paired-group split-bank reliability",
        "",
        "每个 dataset/task/capability 的 paired-group pool 被直接切成两个"
        "不相交 bank；round 不作为统计层级。Ordered split 按 paired_group_id "
        "确定性排序后取前 N 与后 N。",
        "",
        f"排名 cell 通过阈值：`{summary['minimum_pairwise_agreement']:.2f}`。",
        "",
    ]
    ordered = flat[flat["split_kind"] == "ordered"]
    for policy, title in (
        ("oracle_context", "Oracle context"),
        ("fixed_l504", "固定 L=504"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| N per bank | Rank agreement | Cells passed | Top-1 | "
                "Profile CCC | Tie-state match |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        selected = ordered[ordered["score_policy"] == policy].sort_values(
            "bank_size"
        )
        for row in selected.to_dict(orient="records"):
            lines.append(
                f"| {int(row['bank_size'])} | "
                f"{row['rank_pairwise_agreement_mean']:.4f} | "
                f"{row['rank_passed_cell_rate']:.4f} | "
                f"{row['top1_agreement_rate']:.4f} | "
                f"{row['capability_profile_lin_ccc']:.4f} | "
                f"{row['tie_state_match_rate']:.4f} |"
            )
        lines.append("")
    if summary["random_repeats"] > 0:
        lines.extend(
            [
                "## Repeated random split",
                "",
                f"每个 N 额外执行 {summary['random_repeats']} 次固定种子的"
                "不相交随机二分；完整分布见 `summary.json` 和 "
                "`split_comparison_summary.csv`。",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Split-half reliability 可用于估计当前生成分布下达到稳定测量所需的"
            "样本数，但两个 bank 来自同一个冻结 pool，因此不能替代独立重新生成"
            "的 external seed-bank replication。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    e2_dir: Path,
    output_dir: Path,
    *,
    bank_sizes: list[int],
    models: list[str],
    datasets: set[str] | None,
    random_repeats: int,
    split_seed: int,
    minimum_agreement: float,
) -> dict[str, Any]:
    e2_dir = e2_dir.resolve()
    output_dir = output_dir.resolve()
    if random_repeats < 0:
        raise ValueError("random repeats cannot be negative")
    if not 0.0 <= minimum_agreement <= 1.0:
        raise ValueError("minimum agreement must be between zero and one")
    sizes = sorted(set(int(value) for value in bank_sizes))
    if not sizes:
        raise ValueError("at least one bank size is required")

    oracle_paths = ensure_oracle_paths(e2_dir, models)
    oracle = load_oracle_pool(oracle_paths, datasets=datasets)
    observed_models = set(oracle["model_id"])
    missing_models = sorted(set(models) - observed_models)
    if missing_models:
        raise ValueError(
            "selected oracle pool is missing models: "
            + ", ".join(missing_models)
        )
    catalog = pool_catalog(oracle)
    profile_sizes = catalog.groupby(PROFILE_KEYS, sort=False).size()
    minimum_pool_size = int(profile_sizes.min())
    if 2 * max(sizes) > minimum_pool_size:
        raise ValueError(
            f"largest bank size {max(sizes)} exceeds half the minimum "
            f"profile pool size {minimum_pool_size}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows: list[dict[str, Any]] = []
    ordered_summaries: dict[str, dict[str, Any]] = {
        policy: {} for policy in SCORE_POLICIES
    }
    ordered_frames: dict[str, dict[str, list[pd.DataFrame]]] = {
        policy: {} for policy in SCORE_POLICIES
    }
    split_specs = [("ordered", 0), *(
        ("random", repeat_index)
        for repeat_index in range(1, random_repeats + 1)
    )]
    for split_kind, repeat_index in split_specs:
        for bank_size in sizes:
            assignments = split_assignments(
                catalog,
                bank_size=bank_size,
                split_kind=split_kind,
                repeat_index=repeat_index,
                split_seed=split_seed,
            )
            for score_policy, score_column in SCORE_POLICIES.items():
                policy_summary, frames = analyze_split(
                    oracle,
                    assignments,
                    score_column=score_column,
                    bank_size=bank_size,
                    minimum_agreement=minimum_agreement,
                )
                flat_rows.append(
                    flat_summary_row(
                        policy_summary,
                        bank_size=bank_size,
                        split_kind=split_kind,
                        repeat_index=repeat_index,
                        score_policy=score_policy,
                    )
                )
                if split_kind == "ordered":
                    ordered_summaries[score_policy][str(bank_size)] = (
                        policy_summary
                    )
                    for name, frame in frames.items():
                        ordered_frames[score_policy].setdefault(
                            name,
                            [],
                        ).append(
                            add_split_metadata(
                                frame,
                                bank_size=bank_size,
                                split_kind=split_kind,
                                repeat_index=repeat_index,
                            )
                        )

    flat = pd.DataFrame(flat_rows).sort_values(
        ["split_kind", "repeat_index", "bank_size", "score_policy"],
        kind="stable",
    )
    flat.to_csv(output_dir / "split_comparison_summary.csv", index=False)
    table_rows = {"split_comparison_summary.csv": len(flat)}
    for policy, frame_groups in ordered_frames.items():
        for name, frames in frame_groups.items():
            filename = f"ordered_{name}_{policy}.csv"
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(output_dir / filename, index=False)
            table_rows[filename] = len(combined)
            if name == "rank_reliability":
                for group_column in (
                    "capability_id",
                    "dataset_id",
                    "intensity",
                ):
                    breakdown_name = (
                        "ordered_rank_reliability_by_"
                        f"{group_column.removesuffix('_id')}_{policy}.csv"
                    )
                    breakdown = rank_breakdown(
                        combined,
                        group_column=group_column,
                    )
                    breakdown.to_csv(
                        output_dir / breakdown_name,
                        index=False,
                    )
                    table_rows[breakdown_name] = len(breakdown)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "e2_dir": e2.display_path(e2_dir),
        "models": models,
        "datasets": sorted(oracle["dataset_id"].unique()),
        "sample_unit": "paired_group",
        "pool_order": "lexicographic paired_group_id within profile",
        "round_interpretation": (
            "round fields are ignored; the pool is one flat collection of "
            "independent paired groups"
        ),
        "profile_count": len(profile_sizes),
        "cell_count": int(
            oracle[CELL_KEYS].drop_duplicates().shape[0]
        ),
        "profile_pool_size": {
            "minimum": minimum_pool_size,
            "maximum": int(profile_sizes.max()),
            "unique": sorted(int(value) for value in profile_sizes.unique()),
        },
        "bank_sizes": sizes,
        "random_repeats": random_repeats,
        "split_seed": split_seed,
        "minimum_pairwise_agreement": minimum_agreement,
        "ordered_split": ordered_summaries,
        "random_split": random_repeat_summary(flat),
        "interpretation": (
            "within-pool split-half measurement reliability; useful for "
            "sample-size selection but weaker than an independently "
            "generated seed-bank replication"
        ),
        "inputs": {
            "inference_config": {
                "path": e2.display_path(e2_dir / "inference_config.json"),
                "sha256": file_sha256(
                    e2_dir / "inference_config.json"
                ),
            },
            "oracle_scores": [
                {
                    "path": e2.display_path(path),
                    "sha256": file_sha256(path),
                }
                for path in oracle_paths
            ],
        },
        "table_rows": table_rows,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        render_report(summary, flat),
        encoding="utf-8",
    )
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "paper_e2_split_bank_manifest.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "analysis_path": str(Path(__file__).relative_to(REPO_ROOT)),
            "analysis_sha256": file_sha256(Path(__file__)),
            "files": {
                path.name: {
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in sorted(output_dir.iterdir())
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )
    return summary


def main() -> int:
    args = parse_args()
    e2_dir = args.e2_dir.resolve()
    output_dir = (
        (e2_dir / "split_bank_reliability")
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    models = requested_models(
        e2_dir,
        None if args.models is None else [str(value) for value in args.models],
    )
    summary = analyze(
        e2_dir,
        output_dir,
        bank_sizes=[int(value) for value in args.bank_sizes],
        models=models,
        datasets=(
            None
            if args.datasets is None
            else {str(value) for value in args.datasets}
        ),
        random_repeats=int(args.random_repeats),
        split_seed=int(args.split_seed),
        minimum_agreement=float(args.minimum_agreement),
    )
    oracle_80 = summary["ordered_split"]["oracle_context"].get("80")
    if oracle_80 is None:
        largest = str(max(summary["bank_sizes"]))
        oracle_80 = summary["ordered_split"]["oracle_context"][largest]
    rank = oracle_80["formal_rank_reliability"]
    print(
        "split-bank reliability complete: rank agreement="
        f"{rank['pairwise_ordering_agreement']['mean']:.4f}, "
        f"passed={rank['passed_cell_count']}/{rank['cell_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
