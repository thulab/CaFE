#!/usr/bin/env python3
"""Compare two five-round Paper v5 E2 seed banks on one dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_paper_v5_e2 as e2  # noqa: E402
import run_paper_e2_dynamic_stability as stats  # noqa: E402


SCHEMA_VERSION = "paper_v5_e2_seed_bank_pilot.v1"
DEFAULT_BANK_A_DIR = REPO_ROOT / "runtime/paper_exp/v5/E2_dynamic_stability"
DEFAULT_BANK_B_DIR = REPO_ROOT / "runtime/paper_exp/v5/E2_dynamic_stability_B"
DEFAULT_OUTPUT_DIR = DEFAULT_BANK_B_DIR / "seed_bank_comparison_ett1"
DEFAULT_MODELS = (
    "Timer-3.5",
    "Timer-3.0",
    "Chronos-2",
    "moirai2",
    "toto2.0",
    "tirex2",
)
CELL_KEYS = ["dataset_id", "task_id", "capability_id", "intensity"]
MIN_PAIRWISE_AGREEMENT = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare model rankings after independently pooling five rounds "
            "of 32 samples from two Paper v5 E2 seed banks."
        )
    )
    parser.add_argument("--bank-a-dir", type=Path, default=DEFAULT_BANK_A_DIR)
    parser.add_argument("--bank-b-dir", type=Path, default=DEFAULT_BANK_B_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-id", default="gift_ett1_h")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generation_provenance(
    bank_a_dir: Path,
    bank_b_dir: Path,
) -> dict[str, Any]:
    configs = [
        read_json(directory / "generation_config.json")
        for directory in (bank_a_dir, bank_b_dir)
    ]
    comparable = []
    for config in configs:
        record = dict(config)
        record.pop("created_at", None)
        record.pop("round_seeds", None)
        comparable.append(record)
    if comparable[0] != comparable[1]:
        raise ValueError("seed banks differ beyond creation time and seeds")
    seeds = [list(map(int, config["round_seeds"])) for config in configs]
    if any(len(values) != 5 or len(set(values)) != 5 for values in seeds):
        raise ValueError("each seed bank must contain five unique round seeds")
    if set(seeds[0]) & set(seeds[1]):
        raise ValueError("seed banks are not independent")
    manifests = [
        read_json(directory / "sample_manifest.json")
        for directory in (bank_a_dir, bank_b_dir)
    ]
    versions = {str(manifest["generator_version"]) for manifest in manifests}
    hashes = {str(manifest["generator_sha256"]) for manifest in manifests}
    if len(versions) != 1 or len(hashes) != 1:
        raise ValueError("seed banks do not share one frozen generator")
    return {
        "generator_version": next(iter(versions)),
        "generator_sha256": next(iter(hashes)),
        "bank_a_round_seeds": seeds[0],
        "bank_b_round_seeds": seeds[1],
        "configuration_equal_except_created_at_and_round_seeds": True,
    }


def find_prediction_path(bank_dir: Path, model_id: str) -> Path:
    filename = f"{e2.safe_filename(model_id)}.jsonl"
    matches = sorted(bank_dir.glob(f"inference_ett1_*/predictions/{filename}"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one Bank B prediction for {model_id}, found {matches}"
        )
    return matches[0]


def load_bank_a_round_scores(
    bank_a_dir: Path,
    *,
    dataset_id: str,
    models: list[str],
    fixed_l504: bool,
) -> pd.DataFrame:
    filename = (
        "cell_round_scores_l504.csv"
        if fixed_l504
        else "cell_round_scores.csv"
    )
    source = bank_a_dir / filename
    frame = pd.read_csv(source)
    selected = frame[
        (frame["dataset_id"] == dataset_id)
        & frame["model_id"].isin(models)
    ].copy()
    validate_round_scores(selected, models=models)
    return selected


def build_bank_b_round_scores(
    bank_b_dir: Path,
    output_dir: Path,
    *,
    dataset_id: str,
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    oracle_paths: list[Path] = []
    prediction_records: list[dict[str, Any]] = []
    oracle_dir = output_dir / "bank_b_oracle_sample_scores"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    for model_id in models:
        source = find_prediction_path(bank_b_dir, model_id)
        destination = oracle_dir / f"{e2.safe_filename(model_id)}.jsonl"
        record = e2.compact_prediction_file(
            source,
            destination,
            model_id=model_id,
            prediction_kind="synthetic",
        )
        if int(record["view_count"]) != 19_200:
            raise ValueError(f"unexpected Bank B view count: {record}")
        if int(record["master_count"]) != 4_800:
            raise ValueError(f"unexpected Bank B master count: {record}")
        oracle_paths.append(destination)
        prediction_records.append(
            {
                "model_id": model_id,
                "path": str(source.relative_to(REPO_ROOT)),
                "size_bytes": source.stat().st_size,
                "sha256": file_sha256(source),
                **record,
            }
        )
    oracle = e2.load_oracle_rows(
        oracle_paths,
        prediction_kind="synthetic",
    )
    oracle = oracle[oracle["dataset_id"] == dataset_id].copy()
    oracle_round = e2.cell_round_scores(
        oracle,
        score_column="oracle_mase",
        score_policy="oracle_context",
    )
    fixed_round = e2.cell_round_scores(
        oracle,
        score_column="fixed_l504_mase",
        score_policy="fixed_l504",
    )
    validate_round_scores(oracle_round, models=models)
    validate_round_scores(fixed_round, models=models)
    return oracle_round, fixed_round, prediction_records


def validate_round_scores(frame: pd.DataFrame, *, models: list[str]) -> None:
    if set(frame["model_id"]) != set(models):
        raise ValueError("round scores do not contain the requested models")
    if set(frame["round_index"]) != {1, 2, 3, 4, 5}:
        raise ValueError("round scores do not contain exactly five rounds")
    if not (frame["master_sample_count"] == 32).all():
        raise ValueError("round scores do not contain 32 samples per round")
    counts = frame.groupby(CELL_KEYS, sort=False).size()
    expected = len(models) * 5
    if counts.empty or not (counts == expected).all():
        raise ValueError("round score cells are incomplete")


def aggregate_bank_scores(
    round_scores: pd.DataFrame,
    *,
    bank_id: str,
) -> pd.DataFrame:
    grouped = (
        round_scores.groupby(["model_id", *CELL_KEYS], sort=True)
        .agg(
            mase_mean=("mase_mean", "mean"),
            round_count=("round_index", "nunique"),
            master_sample_count=("master_sample_count", "sum"),
            mean_round_rank=("model_rank", "mean"),
        )
        .reset_index()
    )
    if not (grouped["round_count"] == 5).all():
        raise ValueError("bank aggregation did not receive five rounds")
    if not (grouped["master_sample_count"] == 160).all():
        raise ValueError("bank aggregation did not receive 160 samples")
    grouped["bank_id"] = bank_id
    grouped["model_rank"] = grouped.groupby(
        CELL_KEYS,
        sort=False,
    )["mase_mean"].rank(method="average", ascending=True)
    grouped["mean_round_rank_order"] = grouped.groupby(
        CELL_KEYS,
        sort=False,
    )["mean_round_rank"].rank(method="average", ascending=True)
    return grouped


def compare_banks(
    bank_a: pd.DataFrame,
    bank_b: pd.DataFrame,
    *,
    minimum_agreement: float = MIN_PAIRWISE_AGREEMENT,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, left_group in bank_a.groupby(CELL_KEYS, sort=True):
        right_group = bank_b
        for column, value in zip(CELL_KEYS, key, strict=True):
            right_group = right_group[right_group[column] == value]
        left = left_group.set_index("model_id").sort_index()
        right = right_group.set_index("model_id").sort_index()
        if list(left.index) != list(right.index):
            raise ValueError(f"Bank A/B model mismatch for cell {key}")
        left_score = left["mase_mean"].to_numpy(dtype=float)
        right_score = right["mase_mean"].to_numpy(dtype=float)
        left_rank = left["model_rank"].to_numpy(dtype=float)
        right_rank = right["model_rank"].to_numpy(dtype=float)
        left_mean_rank = left["mean_round_rank"].to_numpy(dtype=float)
        right_mean_rank = right["mean_round_rank"].to_numpy(dtype=float)
        agreement = float(
            stats.pairwise_ordering_agreement(left_score, right_score)[
                "agreement"
            ]
        )
        mean_rank_agreement = float(
            stats.pairwise_ordering_agreement(
                left_mean_rank,
                right_mean_rank,
            )["agreement"]
        )
        top_k = min(3, len(left))
        left_top = set(left["mase_mean"].nsmallest(top_k).index)
        right_top = set(right["mase_mean"].nsmallest(top_k).index)
        rows.append(
            {
                **dict(zip(CELL_KEYS, key, strict=True)),
                "model_count": len(left),
                "pair_count": len(left) * (len(left) - 1) // 2,
                "pairwise_ordering_agreement": agreement,
                "passed": agreement >= minimum_agreement,
                "spearman_rho": float(
                    stats.spearman_rank_correlation(left_rank, right_rank)
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
                "top3_overlap_rate": len(left_top & right_top) / top_k,
                "mean_round_rank_ordering_agreement": (
                    mean_rank_agreement
                ),
                "mean_round_rank_exact_vector": bool(
                    np.array_equal(
                        left["mean_round_rank_order"].to_numpy(dtype=float),
                        right["mean_round_rank_order"].to_numpy(dtype=float),
                    )
                ),
            }
        )
    result = pd.DataFrame(rows)
    return result


def summarize_comparison(frame: pd.DataFrame) -> dict[str, Any]:
    agreement = frame["pairwise_ordering_agreement"].to_numpy(dtype=float)
    return {
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
        "mean_round_rank_ordering_agreement": {
            "mean": float(
                frame["mean_round_rank_ordering_agreement"].mean()
            ),
            "median": float(
                frame["mean_round_rank_ordering_agreement"].median()
            ),
            "minimum": float(
                frame["mean_round_rank_ordering_agreement"].min()
            ),
            "passed_cell_count": int(
                (
                    frame["mean_round_rank_ordering_agreement"]
                    >= MIN_PAIRWISE_AGREEMENT
                ).sum()
            ),
        },
        "mean_round_rank_exact_vector_rate": float(
            frame["mean_round_rank_exact_vector"].mean()
        ),
    }


def within_bank_round_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary = e2.summarize_rank_stability(e2.rank_stability_rows(frame))
    keys = (
        "cell_count",
        "passed_cell_count",
        "passed_cell_rate",
        "pairwise_agreement_minimum",
        "pairwise_agreement_median",
        "mean_pairwise_agreement",
        "mean_exact_rank_vector_pair_rate",
        "mean_top1_pair_agreement_rate",
        "mean_top3_overlap",
    )
    return {key: summary[key] for key in keys}


def grouped_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame.groupby(column, sort=True)
        .agg(
            cell_count=("passed", "size"),
            passed_cell_count=("passed", "sum"),
            pairwise_agreement_mean=(
                "pairwise_ordering_agreement",
                "mean",
            ),
            pairwise_agreement_median=(
                "pairwise_ordering_agreement",
                "median",
            ),
            spearman_mean=("spearman_rho", "mean"),
            top1_agreement_rate=("top1_agreement", "mean"),
        )
        .reset_index()
    )


def report_text(summary: dict[str, Any]) -> str:
    oracle = summary["score_policies"]["oracle_context"]
    fixed = summary["score_policies"]["fixed_l504"]
    return "\n".join(
        [
            "# Paper v5 E2 ETT1 五轮 seed-bank 稳定性小试验",
            "",
            "两套样本均由同一 `capts-paper-v3` 生成器和同一套逐数据集校准结果生成，"
            "只替换五个 round seed。每个 bank 在每个 capability × intensity cell "
            "内先汇总 5 × 32 = 160 条样本的模型 MASE，再据此排名。",
            "",
            "## 主结果：逐样本 oracle context",
            "",
            f"- 通过 cell：{oracle['passed_cell_count']} / "
            f"{oracle['cell_count']}；",
            f"- pairwise ordering agreement mean / median / min："
            f"{oracle['pairwise_ordering_agreement']['mean']:.4f} / "
            f"{oracle['pairwise_ordering_agreement']['median']:.4f} / "
            f"{oracle['pairwise_ordering_agreement']['minimum']:.4f}；",
            f"- Spearman ρ mean / median："
            f"{oracle['spearman_rho']['mean']:.4f} / "
            f"{oracle['spearman_rho']['median']:.4f}；",
            f"- top-1 agreement：{oracle['top1_agreement_rate']:.4f}；",
            f"- exact rank-vector rate："
            f"{oracle['exact_rank_vector_rate']:.4f}。",
            f"- 对照：Bank A / B 内部逐轮 mean agreement 为 "
            f"{oracle['within_bank_single_round']['A']['mean_pairwise_agreement']:.4f} / "
            f"{oracle['within_bank_single_round']['B']['mean_pairwise_agreement']:.4f}。",
            "",
            "## 敏感性分析：固定 L=504",
            "",
            f"- 通过 cell：{fixed['passed_cell_count']} / "
            f"{fixed['cell_count']}；",
            f"- pairwise ordering agreement mean / median / min："
            f"{fixed['pairwise_ordering_agreement']['mean']:.4f} / "
            f"{fixed['pairwise_ordering_agreement']['median']:.4f} / "
            f"{fixed['pairwise_ordering_agreement']['minimum']:.4f}；",
            f"- Spearman ρ mean / median："
            f"{fixed['spearman_rho']['mean']:.4f} / "
            f"{fixed['spearman_rho']['median']:.4f}；",
            f"- top-1 agreement：{fixed['top1_agreement_rate']:.4f}；",
            f"- exact rank-vector rate："
            f"{fixed['exact_rank_vector_rate']:.4f}。",
            f"- 对照：Bank A / B 内部逐轮 mean agreement 为 "
            f"{fixed['within_bank_single_round']['A']['mean_pairwise_agreement']:.4f} / "
            f"{fixed['within_bank_single_round']['B']['mean_pairwise_agreement']:.4f}。",
            "",
            "阈值为 pairwise ordering agreement ≥ 0.80。六模型共有 15 个"
            "模型对，因此通过允许最多 3 个模型对改变相对次序。",
            "",
        ]
    )


def analyze(
    bank_a_dir: Path,
    bank_b_dir: Path,
    output_dir: Path,
    *,
    dataset_id: str,
    models: list[str],
) -> dict[str, Any]:
    if len(models) != 6 or len(set(models)) != 6:
        raise ValueError("pilot requires six unique models")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = generation_provenance(bank_a_dir, bank_b_dir)
    bank_b_oracle, bank_b_fixed, prediction_records = (
        build_bank_b_round_scores(
            bank_b_dir,
            output_dir,
            dataset_id=dataset_id,
            models=models,
        )
    )
    policies = {
        "oracle_context": (
            load_bank_a_round_scores(
                bank_a_dir,
                dataset_id=dataset_id,
                models=models,
                fixed_l504=False,
            ),
            bank_b_oracle,
        ),
        "fixed_l504": (
            load_bank_a_round_scores(
                bank_a_dir,
                dataset_id=dataset_id,
                models=models,
                fixed_l504=True,
            ),
            bank_b_fixed,
        ),
    }
    policy_summaries: dict[str, Any] = {}
    for policy, (round_a, round_b) in policies.items():
        bank_a = aggregate_bank_scores(round_a, bank_id="A")
        bank_b = aggregate_bank_scores(round_b, bank_id="B")
        comparison = compare_banks(bank_a, bank_b)
        if len(comparison) != 30:
            raise ValueError(
                f"expected 30 ETT1 cells, observed {len(comparison)}"
            )
        round_scores = pd.concat(
            [round_a.assign(bank_id="A"), round_b.assign(bank_id="B")],
            ignore_index=True,
        )
        cell_scores = pd.concat([bank_a, bank_b], ignore_index=True)
        round_scores.to_csv(
            output_dir / f"bank_round_scores_{policy}.csv",
            index=False,
        )
        cell_scores.to_csv(
            output_dir / f"bank_cell_model_scores_{policy}.csv",
            index=False,
        )
        comparison.to_csv(
            output_dir / f"cell_rank_comparison_{policy}.csv",
            index=False,
        )
        grouped_summary(comparison, "capability_id").to_csv(
            output_dir / f"comparison_by_capability_{policy}.csv",
            index=False,
        )
        grouped_summary(comparison, "intensity").to_csv(
            output_dir / f"comparison_by_intensity_{policy}.csv",
            index=False,
        )
        policy_summaries[policy] = {
            **summarize_comparison(comparison),
            "within_bank_single_round": {
                "A": within_bank_round_summary(round_a),
                "B": within_bank_round_summary(round_b),
            },
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "models": models,
        "minimum_pairwise_ordering_agreement": MIN_PAIRWISE_AGREEMENT,
        **provenance,
        "samples_per_round_per_cell": 32,
        "samples_per_bank_per_cell": 160,
        "aggregation_policy": (
            "choose the best context per model and master sample, average "
            "MASE across five equal 32-sample rounds, then rank models"
        ),
        "score_policies": policy_summaries,
        "bank_b_predictions": prediction_records,
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
    print(
        "seed-bank pilot complete: "
        f"{oracle['passed_cell_count']}/{oracle['cell_count']} cells pass, "
        "mean agreement="
        f"{oracle['pairwise_ordering_agreement']['mean']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
