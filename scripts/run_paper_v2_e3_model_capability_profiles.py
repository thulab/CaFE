#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_VERSION = "v2"
EXPERIMENT_ID = "E3_model_capability_profiles"
SCHEMA_VERSION = "paper_e3_model_capability_profiles.v2"
DEFAULT_SOURCE_DIR = REPO_ROOT / "runtime/paper_exp/v2/E2_dynamic_stability"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/paper_exp/v2/E3_model_capability_profiles"
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/"
    "2026-07-17-paper-v2-e3-model-capability-profiling-protocol.md"
)
EXPECTED_E2_MANIFEST_SHA256 = (
    "91b61c7d4b3d4cd81da28f011d6d6e0810db423d1c16b0bb336a6f17a2e1f34d"
)
CAPABILITY_ORDER = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
)


import run_paper_e3_model_capability_profiles as base  # noqa: E402


def configure_base_module() -> None:
    base.EXPERIMENT_VERSION = EXPERIMENT_VERSION
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.DEFAULT_SOURCE_DIR = DEFAULT_SOURCE_DIR
    base.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    base.PROTOCOL_PATH = PROTOCOL_PATH
    base.RUNNER_PATH = Path(__file__).resolve()
    base.EXPECTED_E2_MANIFEST_SHA256 = EXPECTED_E2_MANIFEST_SHA256
    base.CAPABILITY_ORDER = CAPABILITY_ORDER
    base.UNIVARIATE_CAPABILITIES = CAPABILITY_ORDER
    base.STRUCTURED_CAPABILITIES = ()
    base.TASK_PROTOCOL_BY_CAPABILITY = {
        capability: "univariate" for capability in CAPABILITY_ORDER
    }


configure_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-v2 E3 profiles for the six univariate capabilities "
            "from the sealed E2-v2 output."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=base.DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--allow-existing-empty",
        action="store_true",
        help="Only intended for isolated tests; completed output is immutable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    base.verify_e2_source(source_dir)
    base.prepare_output_dir(
        output_dir,
        allow_existing_empty=args.allow_existing_empty,
    )

    source_config = base.read_json(source_dir / "config.json")
    validate_source_config(source_config)
    config = experiment_config(
        source_dir=source_dir,
        source_config=source_config,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    base.write_json(output_dir / "config.json", config)

    samples = base.load_sample_info(source_dir / "samples.jsonl")
    observations = base.load_prediction_observations(
        source_dir,
        samples=samples,
        foundation_models=tuple(source_config["requested_models"]),
    )
    base.validate_observation_design(
        observations,
        samples=samples,
        config=source_config,
    )

    cell_scores = base.profile_intensity_score_frame(observations)
    intensity_curves = base.intensity_curve_frame(cell_scores)
    bucket_scores = base.bucket_score_frame(cell_scores)
    capability_profiles = base.capability_profile_frame(
        intensity_curves,
        bucket_scores,
    )

    bootstrap = base.capability_bootstrap_results(
        observations,
        cell_scores=cell_scores,
        replicates=args.bootstrap_replicates,
        seed=base.BOOTSTRAP_SEED,
    )
    intensity_curves = base.attach_intensity_bootstrap_ci(
        intensity_curves,
        bootstrap,
    )
    capability_profiles = base.attach_capability_bootstrap_ci(
        capability_profiles,
        bootstrap,
    )
    capability_profiles = base.add_capability_ranks(capability_profiles)
    model_summary = base.model_summary_frame(capability_profiles, bootstrap)
    capability_contrasts = capability_contrast_frame(
        capability_profiles,
        bootstrap,
    )

    outputs = {
        "profile_intensity_scores.csv": cell_scores,
        "intensity_curves.csv": intensity_curves,
        "bucket_scores.csv": bucket_scores,
        "capability_profiles.csv": capability_profiles,
        "model_capability_contrasts.csv": capability_contrasts,
        "model_summary.csv": model_summary,
    }
    for filename, frame in outputs.items():
        base.write_dataframe(output_dir / filename, frame)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    figure_files = base.render_figures(
        figures_dir,
        intensity_curves=intensity_curves,
        capability_profiles=capability_profiles,
        model_summary=model_summary,
    )

    summary = summarize_results(
        config=config,
        capability_profiles=capability_profiles,
        capability_contrasts=capability_contrasts,
        model_summary=model_summary,
        cell_scores=cell_scores,
        table_rows={name: len(frame) for name, frame in outputs.items()},
        figure_files=figure_files,
    )
    base.write_json(output_dir / "summary.json", summary)
    (output_dir / "paper_tables.md").write_text(
        render_paper_tables(capability_profiles, model_summary),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )
    base.write_manifest(output_dir, source_dir=source_dir)

    print(
        f"E3-v2 profiles: {len(capability_profiles)} model-capability rows; "
        f"figures={len(figure_files)}",
        flush=True,
    )
    print(f"E3-v2 output: {output_dir}", flush=True)
    return 0


def validate_source_config(config: dict[str, Any]) -> None:
    if tuple(config.get("requested_models", ())) != base.MODEL_ORDER:
        raise ValueError("E2 requested model order does not match frozen E3 model set")
    if tuple(config.get("intensities", ())) != base.INTENSITIES:
        raise ValueError("E2 intensity levels do not match E3 protocol")
    if int(config.get("expected_generated_sample_count", 0)) != 21_600:
        raise ValueError("E2 sample count is not the frozen 21,600 design")
    if len(config.get("online_conditioning_profile_ids", ())) != 9:
        raise ValueError("E2 must contain nine held-out transfer profiles")
    if int(config.get("profile_capability_count", 0)) != 54:
        raise ValueError("E2 must contain nine profiles x six capabilities")
    if len(config.get("round_seeds", ())) != 5:
        raise ValueError("E2 must contain five rounds")
    if int(config.get("samples_per_round_per_cell", 0)) != 16:
        raise ValueError("E2 must contain 16 samples per round and cell")
    if config.get("fixed_request_shape") != {
        "context_length": 504,
        "horizon": 48,
        "season_length": 24,
        "target_dim": 1,
    }:
        raise ValueError("E2 request shape is not the frozen 504/48/24 univariate design")


def experiment_config(
    *,
    source_dir: Path,
    source_config: dict[str, Any],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment": str(source_dir.relative_to(REPO_ROOT)),
        "source_manifest_sha256": base.sha256_file(source_dir / "manifest.json"),
        "canonical_scale_id": source_config["canonical_scale_id"],
        "canonical_scale_fingerprint": source_config[
            "canonical_scale_fingerprint"
        ],
        "foundation_models": list(source_config["requested_models"]),
        "seasonal_baseline": base.SEASONAL_BASELINE,
        "capabilities": list(CAPABILITY_ORDER),
        "intensities": list(base.INTENSITIES),
        "profiles": list(source_config["online_conditioning_profile_ids"]),
        "request_shape": dict(source_config["fixed_request_shape"]),
        "aggregation": {
            "within_profile_intensity": (
                "equal sample weight across 5 rounds x 16 samples"
            ),
            "profiles": "equal macro weight across nine held-out profiles",
            "intensities": "equal macro weight across five absolute levels",
            "mase_auc": "trapezoidal integral on x=(intensity-1)/4",
            "worst_level": (
                "maximum observed profile-macro MASE among five levels; "
                "intensity is structure strength, not assumed difficulty"
            ),
            "relative_skill": (
                "1-model_mean_mase/seasonal_naive_mean_mase within each "
                "profile-intensity, followed by equal macro averaging"
            ),
            "nmae_abs": (
                "sum_abs_error/sum_abs_future_target within cell, "
                "followed by macro averaging"
            ),
            "cross_bucket_variance": (
                "sample variance ddof=1 of each profile's five-level MASE"
            ),
        },
        "bootstrap": {
            "replicates": int(bootstrap_replicates),
            "seed": base.BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "round_cluster_resampling": True,
            "sample_index_resampling_within_round": True,
            "paired_across_models_intensities_and_baseline": True,
            "profiles_fixed_not_resampled": True,
        },
        "primary_metric": "seasonal_mase",
        "secondary_metric": "nmae_abs",
        "ranking_policy": (
            "continuous scores and paired CIs are primary; hard ranks are "
            "descriptive because E2 strict cell-wise Kendall did not pass"
        ),
    }


def capability_contrast_frame(
    capability_profiles: pd.DataFrame,
    bootstrap: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for capability_id in CAPABILITY_ORDER:
        group = capability_profiles[
            capability_profiles["capability_id"] == capability_id
        ].sort_values(["five_level_mase_mean", "model_id"])
        best = group.iloc[0]
        best_model = str(best["model_id"])
        best_mase_boot = bootstrap[(best_model, capability_id)][
            "five_level_mase_mean"
        ]
        best_skill_boot = bootstrap[(best_model, capability_id)][
            "five_level_skill_mase_mean"
        ]
        for row in group.to_dict(orient="records"):
            model_id = str(row["model_id"])
            mase_gap_boot = (
                bootstrap[(model_id, capability_id)]["five_level_mase_mean"]
                - best_mase_boot
            )
            skill_gap_boot = (
                bootstrap[(model_id, capability_id)][
                    "five_level_skill_mase_mean"
                ]
                - best_skill_boot
            )
            mase_low, mase_high = base.percentile_ci(mase_gap_boot)
            skill_low, skill_high = base.percentile_ci(skill_gap_boot)
            observed_mase_gap = float(
                row["five_level_mase_mean"] - best["five_level_mase_mean"]
            )
            rows.append(
                {
                    "capability_id": capability_id,
                    "model_id": model_id,
                    "reference_best_model": best_model,
                    "model_five_level_mase": float(
                        row["five_level_mase_mean"]
                    ),
                    "reference_five_level_mase": float(
                        best["five_level_mase_mean"]
                    ),
                    "paired_mase_gap_vs_best": observed_mase_gap,
                    "paired_mase_gap_vs_best_ci_low": mase_low,
                    "paired_mase_gap_vs_best_ci_high": mase_high,
                    "relative_mase_gap_vs_best": observed_mase_gap
                    / float(best["five_level_mase_mean"]),
                    "paired_skill_gap_vs_best": float(
                        row["five_level_skill_mase_mean"]
                        - best["five_level_skill_mase_mean"]
                    ),
                    "paired_skill_gap_vs_best_ci_low": skill_low,
                    "paired_skill_gap_vs_best_ci_high": skill_high,
                    "worse_than_best_at_95ci": bool(
                        model_id != best_model and mase_low > 0
                    ),
                    "interpretation": (
                        "descriptive paired contrast to observed capability "
                        "leader; leader selection is not multiplicity-adjusted"
                    ),
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(
        ["capability_id", "paired_mase_gap_vs_best", "model_id"],
        ignore_index=True,
    )


def summarize_results(
    *,
    config: dict[str, Any],
    capability_profiles: pd.DataFrame,
    capability_contrasts: pd.DataFrame,
    model_summary: pd.DataFrame,
    cell_scores: pd.DataFrame,
    table_rows: dict[str, int],
    figure_files: list[str],
) -> dict[str, Any]:
    leaders: dict[str, Any] = {}
    for capability in CAPABILITY_ORDER:
        group = capability_profiles[
            capability_profiles["capability_id"] == capability
        ].sort_values(["five_level_mase_mean", "model_id"])
        best = group.iloc[0]
        worst = group.iloc[-1]
        leaders[capability] = {
            "compatible_model_count": len(group),
            "best_model": best["model_id"],
            "best_five_level_mase": best["five_level_mase_mean"],
            "best_mase_ci": [
                best["five_level_mase_mean_ci_low"],
                best["five_level_mase_mean_ci_high"],
            ],
            "best_relative_skill": best["five_level_skill_mase_mean"],
            "worst_model": worst["model_id"],
            "worst_five_level_mase": worst["five_level_mase_mean"],
        }

    below_baseline = capability_profiles[
        capability_profiles["five_level_skill_mase_mean"] < 0
    ].sort_values("five_level_skill_mase_mean")
    endpoint = capability_profiles.assign(
        endpoint_abs=capability_profiles["mase_endpoint_relative_change"].abs()
    ).sort_values("endpoint_abs", ascending=False).iloc[0]
    bucket = capability_profiles.sort_values(
        "bucket_mase_cv",
        ascending=False,
    ).iloc[0]
    e2_summary = base.read_json(
        REPO_ROOT / config["source_experiment"] / "summary.json"
    )
    rank_stats = e2_summary["statistics"]["rank_stability"]
    score_stats = e2_summary["statistics"]["score_cv"]
    clear_deficits = capability_contrasts[
        capability_contrasts["worse_than_best_at_95ci"]
    ].sort_values(
        ["relative_mase_gap_vs_best", "capability_id", "model_id"],
        ascending=[False, True, True],
    )
    return base.clean_for_json(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "coverage": {
                "foundation_model_count": int(
                    capability_profiles["model_id"].nunique()
                ),
                "capability_count": len(CAPABILITY_ORDER),
                "profile_count": int(cell_scores["profile_id"].nunique()),
                "model_capability_rows": len(capability_profiles),
                "profile_intensity_rows": len(cell_scores),
            },
            "univariate_model_summary": model_summary.to_dict(orient="records"),
            "capability_leaders": leaders,
            "diagnostics": {
                "below_seasonal_naive_capability_count": len(below_baseline),
                "below_seasonal_naive": below_baseline[
                    [
                        "model_id",
                        "capability_id",
                        "five_level_skill_mase_mean",
                    ]
                ].to_dict(orient="records"),
                "paired_deficits_vs_observed_leader": clear_deficits[
                    [
                        "model_id",
                        "capability_id",
                        "reference_best_model",
                        "paired_mase_gap_vs_best",
                        "paired_mase_gap_vs_best_ci_low",
                        "paired_mase_gap_vs_best_ci_high",
                        "relative_mase_gap_vs_best",
                    ]
                ].to_dict(orient="records"),
                "top_paired_deficits_for_e4": clear_deficits.head(10)[
                    [
                        "model_id",
                        "capability_id",
                        "reference_best_model",
                        "paired_mase_gap_vs_best",
                        "paired_mase_gap_vs_best_ci_low",
                        "paired_mase_gap_vs_best_ci_high",
                        "relative_mase_gap_vs_best",
                    ]
                ].to_dict(orient="records"),
                "largest_endpoint_response": {
                    "model_id": endpoint["model_id"],
                    "capability_id": endpoint["capability_id"],
                    "relative_change_i1_to_i5": endpoint[
                        "mase_endpoint_relative_change"
                    ],
                    "mase_i1": endpoint["mase_intensity_1"],
                    "mase_i5": endpoint["mase_intensity_5"],
                },
                "largest_cross_bucket_cv": {
                    "model_id": bucket["model_id"],
                    "capability_id": bucket["capability_id"],
                    "profile_count": bucket["profile_count"],
                    "bucket_mase_cv": bucket["bucket_mase_cv"],
                },
                "e2_stability_boundary": {
                    "score_cv_median": score_stats["median"],
                    "score_cv_p95": score_stats["p95"],
                    "strict_rank_kendall_median": rank_stats[
                        "kendall_mean_median"
                    ],
                    "strict_rank_kendall_p10": rank_stats["kendall_mean_p10"],
                    "strict_rank_criterion_passed": e2_summary["criteria"][
                        "checks"
                    ]["model_ranking"],
                },
            },
            "table_rows": table_rows,
            "figures": figure_files,
            "interpretation_boundary": (
                "E3 is synthetic capability profiling on six univariate "
                "mechanisms. Continuous scores and paired uncertainty support "
                "hypothesis generation; real-data external validity is tested in E4."
            ),
        }
    )


def render_report(summary: dict[str, Any]) -> str:
    model_rows = summary["univariate_model_summary"]
    leaders = summary["capability_leaders"]
    diagnostics = summary["diagnostics"]
    stability = diagnostics["e2_stability_boundary"]
    lines = [
        "# Paper v2 E3：六个单变量能力画像",
        "",
        f"日期：{summary['created_at'][:10]}",
        "",
        "## 概览",
        "",
        (
            "E3 只读复用已封存 E2-v2 的 9 个 profile、5 个 intensity、"
            "5 轮 × 16 条样本以及七个基础模型预测。"
        ),
        (
            f"E2 分数稳定性良好（CV median={stability['score_cv_median']:.3f}，"
            f"p95={stability['score_cv_p95']:.3f}），但严格逐-cell 全排序 "
            f"Kendall τ median={stability['strict_rank_kendall_median']:.3f}、"
            f"p10={stability['strict_rank_kendall_p10']:.3f}，未通过预注册门限。"
            "因此本报告以连续 MASE、相对 skill 与配对 CI 为主，硬排名仅作描述。"
        ),
        "",
        "## 六能力 macro 总览",
        "",
        "| Rank* | Model | Macro MASE [95% CI] | Skill vs SNaive | Mean cap. rank* | Relative strength | Relative weakness |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in model_rows:
        lines.append(
            "| {rank:.0f} | `{model}` | {mase:.4f} [{low:.4f}, {high:.4f}] | "
            "{skill:.1f}% | {mean_rank:.2f} | `{strong}` | `{weak}` |".format(
                rank=row["rank_univariate_macro_mase"],
                model=row["model_id"],
                mase=row["univariate_macro_mase"],
                low=row["univariate_macro_mase_ci_low"],
                high=row["univariate_macro_mase_ci_high"],
                skill=100 * row["univariate_macro_skill_mase"],
                mean_rank=row["mean_capability_rank"],
                strong=row["strongest_relative_capability"],
                weak=row["weakest_relative_capability"],
            )
        )
    lines.extend(
        [
            "",
            "\\* Rank 对近似并列敏感，不作为 E3 的主要证据。",
            "",
            "## 分能力最优模型",
            "",
            "| Capability | Best model | Five-level MASE [95% CI] | Skill vs SNaive | Worst model |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for capability in CAPABILITY_ORDER:
        row = leaders[capability]
        lines.append(
            "| `{cap}` | `{model}` | {mase:.4f} [{low:.4f}, {high:.4f}] | "
            "{skill:.1f}% | `{worst}` |".format(
                cap=capability,
                model=row["best_model"],
                mase=row["best_five_level_mase"],
                low=row["best_mase_ci"][0],
                high=row["best_mase_ci"][1],
                skill=100 * row["best_relative_skill"],
                worst=row["worst_model"],
            )
        )
    below = diagnostics["below_seasonal_naive"]
    below_text = (
        "；".join(
            f"`{row['model_id']} / {row['capability_id']}` "
            f"({100 * row['five_level_skill_mase_mean']:.1f}%)"
            for row in below
        )
        if below
        else "无"
    )
    endpoint = diagnostics["largest_endpoint_response"]
    bucket = diagnostics["largest_cross_bucket_cv"]
    deficits = diagnostics["top_paired_deficits_for_e4"]
    deficit_lines = [
        "| Model | Capability | Reference | MASE gap [95% CI] | Relative gap |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in deficits:
        deficit_lines.append(
            "| `{model}` | `{cap}` | `{reference}` | "
            "{gap:.4f} [{low:.4f}, {high:.4f}] | {relative:.1f}% |".format(
                model=row["model_id"],
                cap=row["capability_id"],
                reference=row["reference_best_model"],
                gap=row["paired_mase_gap_vs_best"],
                low=row["paired_mase_gap_vs_best_ci_low"],
                high=row["paired_mase_gap_vs_best_ci_high"],
                relative=100 * row["relative_mase_gap_vs_best"],
            )
        )
    lines.extend(
        [
            "",
            "## 诊断性发现",
            "",
            f"- capability 宏平均低于 seasonal naive 的模型单元：{below_text}。",
            (
                "- 最大 intensity 端点变化为 "
                f"`{endpoint['model_id']} / {endpoint['capability_id']}`："
                f"{100 * endpoint['relative_change_i1_to_i5']:.1f}% "
                f"(MASE {endpoint['mase_i1']:.4f} → {endpoint['mase_i5']:.4f})。"
            ),
            (
                "- 最大跨 profile CV 为 "
                f"`{bucket['model_id']} / {bucket['capability_id']}`："
                f"{bucket['bucket_mase_cv']:.3f}，基于 "
                f"{bucket['profile_count']} 个固定 profile。"
            ),
            "",
            "## 配对能力缺陷候选",
            "",
            (
                "下表列 paired MASE gap 的 95% CI 完全高于 0 后，按相对 gap "
                "排序的前 10 个模型单元。"
                "参照模型是该能力的 observed leader，未做 leader-selection "
                "multiplicity adjustment，因此用于 E4 假设生成而非独立显著性声明。"
            ),
            "",
            *deficit_lines,
            "",
            "## 解释边界",
            "",
            (
                "intensity 表示结构强度而非难度；最差档由实际五档曲线决定。"
                "跨能力比较优先使用 seasonal-naive-relative skill；原始 MASE "
                "只在同一 capability 或预先定义的等权 macro 下比较。"
            ),
            (
                "E3 用于提出真实缺陷假设，不独自证明外部效度。E4 必须在不看真实"
                "模型结果的前提下选择 high-loading 数据集，并检验合成画像与真实"
                "窗口表现的对应关系。"
            ),
            "",
            "## 输出",
            "",
            "- `capability_profiles.csv`：五档均值、AUC、最差档、skill、CI 与跨 profile 统计。",
            "- `model_capability_contrasts.csv`：相对各能力 observed leader 的配对 gap 与 CI。",
            "- `intensity_curves.csv`：五档 MASE/NMAE/skill 曲线及 CI。",
            "- `profile_intensity_scores.csv` 与 `bucket_scores.csv`：底层可追溯汇总。",
            "- `model_summary.csv`：六能力等权 macro。",
            "- `paper_tables.md`：可直接审阅的主表。",
            "- `figures/`：5 张图，各保留 PNG、SVG 与 PDF。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_paper_tables(
    capability_profiles: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> str:
    lines = [
        "# E3-v2 paper tables",
        "",
        "## Six-capability macro summary",
        "",
        "| Rank* | Model | MASE | 95% CI | NMAE | Skill | Mean cap. rank* |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in model_summary.to_dict(orient="records"):
        lines.append(
            f"| {int(row['rank_univariate_macro_mase'])} | `{row['model_id']}` | "
            f"{row['univariate_macro_mase']:.4f} | "
            f"[{row['univariate_macro_mase_ci_low']:.4f}, "
            f"{row['univariate_macro_mase_ci_high']:.4f}] | "
            f"{row['univariate_macro_nmae_abs']:.4f} | "
            f"{100 * row['univariate_macro_skill_mase']:.1f}% | "
            f"{row['mean_capability_rank']:.2f} |"
        )
    lines.extend(
        [
            "",
            "\\* Descriptive; continuous estimates and paired CIs are primary.",
            "",
            "## Capability profiles",
            "",
            "| Capability | Model | Rank* | MASE | 95% CI | AUC | Worst (I) | Skill | Profile CV |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for capability in CAPABILITY_ORDER:
        group = capability_profiles[
            capability_profiles["capability_id"] == capability
        ].sort_values("rank_five_level_mase")
        for row in group.to_dict(orient="records"):
            bucket_cv = (
                f"{row['bucket_mase_cv']:.3f}"
                if row["bucket_mase_cv"] is not None
                and math.isfinite(float(row["bucket_mase_cv"]))
                else "N/A"
            )
            lines.append(
                f"| `{capability}` | `{row['model_id']}` | "
                f"{int(row['rank_five_level_mase'])} | "
                f"{row['five_level_mase_mean']:.4f} | "
                f"[{row['five_level_mase_mean_ci_low']:.4f}, "
                f"{row['five_level_mase_mean_ci_high']:.4f}] | "
                f"{row['mase_auc']:.4f} | {row['worst_level_mase']:.4f} "
                f"(I{int(row['worst_level_intensity'])}) | "
                f"{100 * row['five_level_skill_mase_mean']:.1f}% | "
                f"{bucket_cv} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
