#!/usr/bin/env python3
"""Evaluate Paper v5 E2 measurement reliability across two seed banks."""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_paper_v5_e2 as e2  # noqa: E402
import analyze_paper_v5_e2_seed_bank_pilot as rank_pilot  # noqa: E402
import run_paper_e2_dynamic_stability as stats  # noqa: E402


SCHEMA_VERSION = "paper_v5_e2_seed_bank_reliability.v1"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runtime/paper_exp/v5/E2_dynamic_stability_B/"
    "seed_bank_reliability_ett1"
)
CELL_KEYS = ["dataset_id", "task_id", "capability_id", "intensity"]
MODEL_CELL_KEYS = ["model_id", *CELL_KEYS]
NORMAL_95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare continuous capability scores and tie-aware model "
            "contrasts between two independent Paper v5 E2 seed banks."
        )
    )
    parser.add_argument(
        "--bank-a-dir",
        type=Path,
        default=rank_pilot.DEFAULT_BANK_A_DIR,
    )
    parser.add_argument(
        "--bank-b-dir",
        type=Path,
        default=rank_pilot.DEFAULT_BANK_B_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-id", default="gift_ett1_h")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(rank_pilot.DEFAULT_MODELS),
    )
    return parser.parse_args()


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


def bank_oracle_paths(
    bank_dir: Path,
    output_dir: Path,
    *,
    bank_id: str,
    models: list[str],
) -> list[Path]:
    existing = [
        bank_dir
        / "oracle_sample_scores"
        / f"{e2.safe_filename(model_id)}.jsonl"
        for model_id in models
    ]
    if all(path.is_file() for path in existing):
        return existing

    paths: list[Path] = []
    oracle_dir = output_dir / f"bank_{bank_id.lower()}_oracle_sample_scores"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    for model_id in models:
        source = rank_pilot.find_prediction_path(bank_dir, model_id)
        destination = oracle_dir / f"{e2.safe_filename(model_id)}.jsonl"
        e2.compact_prediction_file(
            source,
            destination,
            model_id=model_id,
            prediction_kind="synthetic",
        )
        paths.append(destination)
    return paths


def load_bank_oracle(
    bank_dir: Path,
    output_dir: Path,
    *,
    bank_id: str,
    dataset_id: str,
    models: list[str],
) -> pd.DataFrame:
    frame = e2.load_oracle_rows(
        bank_oracle_paths(
            bank_dir,
            output_dir,
            bank_id=bank_id,
            models=models,
        ),
        prediction_kind="synthetic",
    )
    frame = frame[frame["dataset_id"] == dataset_id].copy()
    frame["bank_id"] = bank_id
    validate_bank_oracle(frame, models=models)
    return frame


def validate_bank_oracle(frame: pd.DataFrame, *, models: list[str]) -> None:
    if set(frame["model_id"]) != set(models):
        raise ValueError("oracle scores do not contain the requested models")
    if set(frame["round_index"]) != {1, 2, 3, 4, 5}:
        raise ValueError("oracle scores do not contain five batches")
    counts = frame.groupby(MODEL_CELL_KEYS, sort=False).size()
    if counts.empty or not (counts == 160).all():
        raise ValueError("each model/cell must contain 160 samples")
    model_counts = frame.groupby(
        [*CELL_KEYS, "master_sample_id"],
        sort=False,
    )["model_id"].nunique()
    if not (model_counts == len(models)).all():
        raise ValueError("models are not paired on every master sample")


def cell_model_scores(
    frame: pd.DataFrame,
    *,
    score_column: str,
    bank_id: str,
) -> pd.DataFrame:
    result = (
        frame.groupby(MODEL_CELL_KEYS, sort=True)[score_column]
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
    if not (result["sample_count"] == 160).all():
        raise ValueError("cell score did not aggregate 160 samples")
    result["mase_se"] = result["mase_std"] / np.sqrt(
        result["sample_count"]
    )
    result["mase_ci_low"] = (
        result["mase_mean"] - NORMAL_95 * result["mase_se"]
    )
    result["mase_ci_high"] = (
        result["mase_mean"] + NORMAL_95 * result["mase_se"]
    )
    result["bank_id"] = bank_id
    if (result["mase_mean"] <= 0).any():
        raise ValueError("MASE means must be positive for log profiles")
    result["log_mase"] = np.log(result["mase_mean"])
    result["relative_log_mase"] = result["log_mase"] - result.groupby(
        CELL_KEYS,
        sort=False,
    )["log_mase"].transform("mean")
    result["model_rank"] = result.groupby(
        CELL_KEYS,
        sort=False,
    )["mase_mean"].rank(method="average", ascending=True)
    return result


def lin_concordance(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("Lin concordance requires equal non-trivial vectors")
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = float(x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    if denominator <= 1e-15:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(2.0 * covariance / denominator)


def vector_reliability(
    frame: pd.DataFrame,
    *,
    left_column: str,
    right_column: str,
) -> dict[str, float | None]:
    left = frame[left_column].to_numpy(dtype=float)
    right = frame[right_column].to_numpy(dtype=float)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("reliability vectors must be equal and non-empty")
    if left.size == 1:
        absolute_difference = float(np.abs(left[0] - right[0]))
        return {
            "pearson_r": None,
            "spearman_rho": None,
            "lin_ccc": None,
            "mae": absolute_difference,
            "rmse": absolute_difference,
        }
    return {
        "pearson_r": float(np.corrcoef(left, right)[0, 1]),
        "spearman_rho": float(
            stats.spearman_rank_correlation(left, right)
        ),
        "lin_ccc": lin_concordance(left, right),
        "mae": float(np.mean(np.abs(left - right))),
        "rmse": float(np.sqrt(np.mean((left - right) ** 2))),
    }


def compare_cell_model_scores(
    bank_a: pd.DataFrame,
    bank_b: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        *MODEL_CELL_KEYS,
        "mase_mean",
        "mase_std",
        "mase_se",
        "mase_ci_low",
        "mase_ci_high",
        "relative_log_mase",
        "model_rank",
    ]
    result = bank_a[columns].merge(
        bank_b[columns],
        on=MODEL_CELL_KEYS,
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    result["mase_difference_b_minus_a"] = (
        result["mase_mean_b"] - result["mase_mean_a"]
    )
    denominator = (
        np.abs(result["mase_mean_a"]) + np.abs(result["mase_mean_b"])
    )
    result["symmetric_relative_difference"] = (
        2.0 * np.abs(result["mase_difference_b_minus_a"])
        / np.maximum(denominator, 1e-12)
    )
    combined_se = np.sqrt(
        result["mase_se_a"] ** 2 + result["mase_se_b"] ** 2
    )
    result["bank_difference_z"] = (
        result["mase_difference_b_minus_a"]
        / np.maximum(combined_se, 1e-12)
    )
    result["mean_ci_overlap"] = (
        np.maximum(result["mase_ci_low_a"], result["mase_ci_low_b"])
        <= np.minimum(result["mase_ci_high_a"], result["mase_ci_high_b"])
    )
    return result


def capability_profiles(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(
            ["bank_id", "model_id", "dataset_id", "task_id", "capability_id"],
            sort=True,
        )
        .agg(
            relative_log_mase=("relative_log_mase", "mean"),
            intensity_point_count=("intensity", "nunique"),
        )
        .reset_index()
    )


def compare_capability_profiles(
    bank_a: pd.DataFrame,
    bank_b: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["model_id", "dataset_id", "task_id", "capability_id"]
    compared = bank_a[keys + ["relative_log_mase"]].merge(
        bank_b[keys + ["relative_log_mase"]],
        on=keys,
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    compared["difference_b_minus_a"] = (
        compared["relative_log_mase_b"]
        - compared["relative_log_mase_a"]
    )
    per_model = []
    for model_id, group in compared.groupby("model_id", sort=True):
        metrics = vector_reliability(
            group,
            left_column="relative_log_mase_a",
            right_column="relative_log_mase_b",
        )
        per_model.append(
            {
                "model_id": model_id,
                "capability_count": len(group),
                **metrics,
            }
        )
    return compared, {
        "overall": vector_reliability(
            compared,
            left_column="relative_log_mase_a",
            right_column="relative_log_mase_b",
        ),
        "per_model": per_model,
    }


def tie_aware_pair_states(
    oracle: pd.DataFrame,
    *,
    score_column: str,
    bank_id: str,
    models: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in oracle.groupby(CELL_KEYS, sort=True):
        matrix = group.pivot(
            index="master_sample_id",
            columns="model_id",
            values=score_column,
        )
        if len(matrix) != 160 or matrix.isna().any().any():
            raise ValueError(f"incomplete paired model matrix for {key}")
        for left_model, right_model in combinations(sorted(models), 2):
            difference = (
                matrix[left_model].to_numpy(dtype=float)
                - matrix[right_model].to_numpy(dtype=float)
            )
            mean = float(np.mean(difference))
            se = float(np.std(difference, ddof=1) / math.sqrt(len(difference)))
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
                    **dict(zip(CELL_KEYS, key, strict=True)),
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


def compare_pair_states(
    bank_a: pd.DataFrame,
    bank_b: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = [*CELL_KEYS, "left_model", "right_model"]
    compared = bank_a.merge(
        bank_b,
        on=keys,
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    decisive = {"left_better", "right_better"}
    compared["state_match"] = compared["state_a"] == compared["state_b"]
    compared["both_decisive"] = compared["state_a"].isin(decisive) & compared[
        "state_b"
    ].isin(decisive)
    compared["direction_conflict"] = (
        compared["both_decisive"]
        & (compared["state_a"] != compared["state_b"])
    )
    compared["both_indistinguishable"] = (
        (compared["state_a"] == "indistinguishable")
        & (compared["state_b"] == "indistinguishable")
    )
    compared["one_decisive_one_indistinguishable"] = (
        compared["state_a"].isin(decisive)
        ^ compared["state_b"].isin(decisive)
    )
    both_decisive = compared[compared["both_decisive"]]
    directional_agreement = (
        float((both_decisive["state_a"] == both_decisive["state_b"]).mean())
        if len(both_decisive)
        else 1.0
    )
    return compared, {
        "model_pair_cell_count": len(compared),
        "state_match_rate": float(compared["state_match"].mean()),
        "both_decisive_count": int(compared["both_decisive"].sum()),
        "both_decisive_directional_agreement": directional_agreement,
        "direction_conflict_count": int(
            compared["direction_conflict"].sum()
        ),
        "both_indistinguishable_count": int(
            compared["both_indistinguishable"].sum()
        ),
        "one_decisive_one_indistinguishable_count": int(
            compared["one_decisive_one_indistinguishable"].sum()
        ),
    }


def formal_rank_reliability(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = []
    for scores in (scores_a, scores_b):
        frame = scores.copy()
        frame["mean_round_rank"] = frame["model_rank"]
        frame["mean_round_rank_order"] = frame["model_rank"]
        prepared.append(frame)
    compared = rank_pilot.compare_banks(*prepared)
    return compared, rank_pilot.summarize_comparison(compared)


def summarize_continuous(
    compared: pd.DataFrame,
    profile_summary: dict[str, Any],
    pair_summary: dict[str, Any],
) -> dict[str, Any]:
    relative = compared["symmetric_relative_difference"].to_numpy(dtype=float)
    return {
        "cell_model_count": len(compared),
        "raw_mase_reliability": vector_reliability(
            compared,
            left_column="mase_mean_a",
            right_column="mase_mean_b",
        ),
        "cell_normalized_score_reliability": vector_reliability(
            compared,
            left_column="relative_log_mase_a",
            right_column="relative_log_mase_b",
        ),
        "symmetric_relative_difference": {
            "median": float(np.median(relative)),
            "p90": float(np.quantile(relative, 0.90)),
            "maximum": float(np.max(relative)),
        },
        "mean_ci_overlap_rate": float(compared["mean_ci_overlap"].mean()),
        "bank_difference_within_1_96_se_rate": float(
            (np.abs(compared["bank_difference_z"]) <= NORMAL_95).mean()
        ),
        "capability_profile_reliability": profile_summary,
        "tie_aware_model_contrasts": pair_summary,
    }


def analyze_policy(
    oracle_a: pd.DataFrame,
    oracle_b: pd.DataFrame,
    output_dir: Path,
    *,
    policy: str,
    score_column: str,
    models: list[str],
) -> dict[str, Any]:
    scores_a = cell_model_scores(
        oracle_a,
        score_column=score_column,
        bank_id="A",
    )
    scores_b = cell_model_scores(
        oracle_b,
        score_column=score_column,
        bank_id="B",
    )
    compared = compare_cell_model_scores(scores_a, scores_b)
    profiles = capability_profiles(
        pd.concat([scores_a, scores_b], ignore_index=True)
    )
    profile_a = profiles[profiles["bank_id"] == "A"].copy()
    profile_b = profiles[profiles["bank_id"] == "B"].copy()
    profile_comparison, profile_summary = compare_capability_profiles(
        profile_a,
        profile_b,
    )
    pair_a = tie_aware_pair_states(
        oracle_a,
        score_column=score_column,
        bank_id="A",
        models=models,
    )
    pair_b = tie_aware_pair_states(
        oracle_b,
        score_column=score_column,
        bank_id="B",
        models=models,
    )
    pair_comparison, pair_summary = compare_pair_states(pair_a, pair_b)
    summary = summarize_continuous(
        compared,
        profile_summary,
        pair_summary,
    )
    rank_comparison, rank_summary = formal_rank_reliability(
        scores_a,
        scores_b,
    )
    summary["formal_rank_reliability"] = rank_summary

    pd.concat([scores_a, scores_b], ignore_index=True).to_csv(
        output_dir / f"cell_model_scores_{policy}.csv",
        index=False,
    )
    compared.to_csv(
        output_dir / f"cell_model_reliability_{policy}.csv",
        index=False,
    )
    profiles.to_csv(
        output_dir / f"capability_profiles_{policy}.csv",
        index=False,
    )
    profile_comparison.to_csv(
        output_dir / f"capability_profile_reliability_{policy}.csv",
        index=False,
    )
    pair_comparison.to_csv(
        output_dir / f"tie_aware_model_contrasts_{policy}.csv",
        index=False,
    )
    rank_comparison.to_csv(
        output_dir / f"formal_rank_reliability_{policy}.csv",
        index=False,
    )
    return summary


def report_text(summary: dict[str, Any]) -> str:
    lines = [
        "# Paper v5 E2：ETT1 seed-bank 测量可靠性",
        "",
        "主分析共同报告两套独立 N=160 seed bank 的连续能力分数、capability "
        "profile 和模型排名可靠性；tie-aware contrasts 区分显著优劣与统计并列。",
        "",
    ]
    for policy, title in (
        ("oracle_context", "Oracle context"),
        ("fixed_l504", "固定 L=504"),
    ):
        result = summary["score_policies"][policy]
        raw = result["raw_mase_reliability"]
        normalized = result["cell_normalized_score_reliability"]
        profile = result["capability_profile_reliability"]["overall"]
        pairs = result["tie_aware_model_contrasts"]
        rank = result["formal_rank_reliability"]
        relative = result["symmetric_relative_difference"]
        lines.extend(
            [
                f"## {title}",
                "",
                f"- Raw MASE Lin CCC / Spearman："
                f"{raw['lin_ccc']:.4f} / {raw['spearman_rho']:.4f}；",
                f"- Cell-normalized score Lin CCC / Spearman："
                f"{normalized['lin_ccc']:.4f} / "
                f"{normalized['spearman_rho']:.4f}；",
                f"- Capability profile Lin CCC / Spearman："
                f"{profile['lin_ccc']:.4f} / "
                f"{profile['spearman_rho']:.4f}；",
                f"- Cell-model symmetric relative difference median / p90："
                f"{relative['median']:.4f} / {relative['p90']:.4f}；",
                f"- Tie-aware 双侧显著 model-pair 方向一致率："
                f"{pairs['both_decisive_directional_agreement']:.4f}，"
                f"相反方向 {pairs['direction_conflict_count']} 对；",
                f"- 正式排名 agreement mean，cells ≥0.80："
                f"{rank['pairwise_ordering_agreement']['mean']:.4f}，"
                f"{rank['passed_cell_count']}/{rank['cell_count']}。",
                "",
            ]
        )
    return "\n".join(lines)


def analyze(
    bank_a_dir: Path,
    bank_b_dir: Path,
    output_dir: Path,
    *,
    dataset_id: str,
    models: list[str],
) -> dict[str, Any]:
    if len(models) != 6 or len(set(models)) != 6:
        raise ValueError("ETT1 pilot requires six unique models")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = rank_pilot.generation_provenance(bank_a_dir, bank_b_dir)
    oracle_a = load_bank_oracle(
        bank_a_dir,
        output_dir,
        bank_id="A",
        dataset_id=dataset_id,
        models=models,
    )
    oracle_b = load_bank_oracle(
        bank_b_dir,
        output_dir,
        bank_id="B",
        dataset_id=dataset_id,
        models=models,
    )
    policy_summaries = {
        "oracle_context": analyze_policy(
            oracle_a,
            oracle_b,
            output_dir,
            policy="oracle_context",
            score_column="oracle_mase",
            models=models,
        ),
        "fixed_l504": analyze_policy(
            oracle_a,
            oracle_b,
            output_dir,
            policy="fixed_l504",
            score_column="fixed_l504_mase",
            models=models,
        ),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "models": models,
        **provenance,
        "sample_unit": "paired_group",
        "samples_per_bank_per_cell": 160,
        "round_interpretation": (
            "five deterministic 32-sample batches retained for diagnostics; "
            "not a generator latent hierarchy"
        ),
        "primary_estimand": (
            "expected model MASE under the dataset/capability/intensity "
            "conditional generator distribution"
        ),
        "score_policies": policy_summaries,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        report_text(summary),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = analyze(
        args.bank_a_dir.resolve(),
        args.bank_b_dir.resolve(),
        args.output_dir.resolve(),
        dataset_id=str(args.dataset_id),
        models=[str(model) for model in args.models],
    )
    oracle = summary["score_policies"]["oracle_context"]
    profile = oracle["capability_profile_reliability"]["overall"]
    print(
        "E2 reliability complete: oracle capability-profile CCC="
        f"{profile['lin_ccc']:.4f}, "
        "Spearman="
        f"{profile['spearman_rho']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
