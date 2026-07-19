#!/usr/bin/env python3
"""Build Paper v5 E3 mechanism-aware capability profiles from sealed E2 output."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
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


SCHEMA_VERSION = "paper_v5_e3_mechanism_fidelity.v1"
EXPERIMENT_ID = "E3_mechanism_fidelity"
DEFAULT_E2_DIR = REPO_ROOT / "runtime/paper_exp/v5/E2_dynamic_stability"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/paper_exp/v5/E3_mechanism_fidelity_pilot"
DEFAULT_PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/"
    "2026-07-20-paper-v5-e3-mechanism-fidelity-protocol.md"
)
DEFAULT_MODELS = ("Chronos-2", "tabpfn-ts3")
BLIND_REFERENCE_MODEL = "naive"
MAX_CONTEXT_LENGTH = 504
HORIZON = 48
DEFAULT_DATASET_CAPABILITIES = {
    "gift_ett1_h": (
        "trend",
        "multi_seasonal",
        "time_varying_seasonality",
        "regime_switching",
        "nonlinear_persistence",
        "predictable_intermittency",
    ),
    "electricity_hourly_panel": ("common_factor",),
    "m5_daily_hierarchy": ("hierarchical_coherence",),
    "gefcom2014_load": ("covariate_response",),
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
    "component_scores_json",
    "diagnostics_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate mechanism-aligned forecast behavior for Paper v5 E3 "
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
        default=8,
        help=(
            "Pilot cap per dataset/capability. Each selected paired group "
            "retains all five intensity levels. Use 0 for all groups."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASET_CAPABILITIES),
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


def selected_dataset_capabilities(
    dataset_ids: list[str],
) -> dict[str, tuple[str, ...]]:
    unknown = sorted(set(dataset_ids) - set(DEFAULT_DATASET_CAPABILITIES))
    if unknown:
        raise ValueError(
            "datasets do not have a frozen E3 pilot capability mapping: "
            + ", ".join(unknown)
        )
    return {
        dataset_id: DEFAULT_DATASET_CAPABILITIES[dataset_id]
        for dataset_id in dataset_ids
    }


def load_selected_samples(
    e2_dir: Path,
    *,
    dataset_capabilities: dict[str, tuple[str, ...]],
    max_paired_groups_per_cell: int,
) -> dict[str, dict[str, Any]]:
    if max_paired_groups_per_cell < 0:
        raise ValueError("max_paired_groups_per_cell cannot be negative")
    selected: dict[str, dict[str, Any]] = {}
    for dataset_id, capabilities in dataset_capabilities.items():
        for capability_id in capabilities:
            matches = sorted(
                (e2_dir / "sample_shards").glob(
                    f"{dataset_id}__*__{capability_id}.jsonl"
                )
            )
            if len(matches) != 1:
                raise FileNotFoundError(
                    "expected one sample shard for "
                    f"{dataset_id}/{capability_id}, found {len(matches)}"
                )
            rows = [json.loads(line) for line in matches[0].open(encoding="utf-8")]
            group_order = sorted(
                {
                    (
                        int(row["round_index"]),
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
                    f"{dataset_id}/{capability_id} does not retain five "
                    "paired intensity levels"
                )
            for row in cell_rows:
                selected[str(row["master_sample_id"])] = row
    if not selected:
        raise ValueError("sample selection is empty")
    return selected


def load_oracle_selection(
    e2_dir: Path,
    *,
    model_ids: list[str],
    master_sample_ids: set[str],
) -> tuple[dict[str, dict[str, dict[str, Any]]], pd.DataFrame]:
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
        if set(model_rows) != master_sample_ids:
            missing = len(master_sample_ids - set(model_rows))
            raise ValueError(
                f"{model_id} oracle selection misses {missing} samples"
            )
        if model_id != BLIND_REFERENCE_MODEL:
            selections[model_id] = model_rows
    return selections, pd.DataFrame(rows)


def load_selected_predictions(
    e2_dir: Path,
    *,
    model_id: str,
    oracle_selection: dict[str, dict[str, Any]],
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
            selected[master_id] = row
            if len(selected) == len(view_to_master):
                break
    if set(selected) != set(oracle_selection):
        missing = len(set(oracle_selection) - set(selected))
        raise ValueError(f"{model_id} predictions miss {missing} oracle views")
    return selected


def load_covariate_ablation_predictions(
    directory: Path | None,
    *,
    model_ids: list[str],
    samples: dict[str, dict[str, Any]],
    oracle_selections: dict[str, dict[str, dict[str, Any]]],
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
                if master_id not in desired_ids:
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
                model_rows[master_id] = row
        if set(model_rows) != desired_ids:
            missing = len(desired_ids - set(model_rows))
            raise ValueError(
                f"{model_id} covariate ablation misses {missing} oracle views"
            )
        result[model_id] = model_rows
    return result


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
                counterfactual_forecast=(
                    np.asarray(
                        counterfactual_predictions[model_id][master_id][
                            "forecast"
                        ],
                        dtype=float,
                    )
                    if (
                        counterfactual_predictions
                        and model_id in counterfactual_predictions
                        and master_id
                        in counterfactual_predictions[model_id]
                    )
                    else None
                ),
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


def intensity_cell_scores(
    sample_scores: pd.DataFrame,
    oracle_scores: pd.DataFrame,
) -> pd.DataFrame:
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
    sample_count: int,
    models: list[str],
    max_groups: int,
    covariate_ablation_available: bool,
) -> None:
    lines = [
        "# Paper v5 E3：机制保真能力画像小试验",
        "",
        "本结果同时保留点预测误差、机制保真度与能力总分。机制分只说明输出行为",
        "与合成机制一致，不表示模型内部识别了因果生成机制。",
        "",
        f"- 模型：{', '.join(models)}",
        f"- 逐模型样本评分总行数：{sample_count}",
        f"- 每个 dataset × capability 的 paired groups 上限：{max_groups or '全部'}",
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
        "## 能力画像",
        "",
        "| Dataset | Capability | Model | MASE | Level MFS | Dose | MFS | Ability | MASE rank | Mechanism rank | Ability rank | Formal |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in profiles.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.dataset_id),
                    str(row.capability_id),
                    str(row.model_id),
                    _format_number(row.mase_mean),
                    _format_number(row.level_mechanism_fidelity),
                    _format_number(row.dose_response_score),
                    _format_number(row.mechanism_fidelity_score),
                    _format_number(row.ability_score),
                    _format_number(row.mase_rank),
                    _format_number(row.mechanism_rank),
                    _format_number(row.ability_rank),
                    "yes" if bool(row.formal_score_eligible) else "diagnostic",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.4f}" if math.isfinite(number) else "N/A"


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
    dataset_capabilities = selected_dataset_capabilities(list(args.datasets))
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
        dataset_capabilities=dataset_capabilities,
        max_paired_groups_per_cell=int(args.max_paired_groups_per_cell),
    )
    oracle_selections, oracle_scores = load_oracle_selection(
        args.e2_dir,
        model_ids=model_ids,
        master_sample_ids=set(samples),
    )
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for model_id in model_ids:
        print(f"loading selected {model_id} forecasts", flush=True)
        predictions[model_id] = load_selected_predictions(
            args.e2_dir,
            model_id=model_id,
            oracle_selection=oracle_selections[model_id],
        )
    counterfactual_predictions = load_covariate_ablation_predictions(
        args.covariate_ablation_predictions_dir,
        model_ids=model_ids,
        samples=samples,
        oracle_selections=oracle_selections,
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

    sample_scores.to_csv(output_dir / "sample_mechanism_scores.csv", index=False)
    cells.to_csv(output_dir / "intensity_cell_scores.csv", index=False)
    dose.to_csv(output_dir / "paired_dose_response_scores.csv", index=False)
    profiles.to_csv(output_dir / "capability_profiles.csv", index=False)
    write_report(
        output_dir / "report.md",
        profiles=profiles,
        sample_count=len(sample_scores),
        models=model_ids,
        max_groups=int(args.max_paired_groups_per_cell),
        covariate_ablation_available=bool(counterfactual_predictions),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "models": model_ids,
        "blind_reference_model": BLIND_REFERENCE_MODEL,
        "dataset_capabilities": {
            key: list(value) for key, value in dataset_capabilities.items()
        },
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
        "ability_score": (
            "mechanism_fidelity_score * "
            "min(1, naive_mase / model_mase)"
        ),
        "covariate_response_status": (
            "formal_paired_future_covariate_ablation"
            if counterfactual_predictions
            else "diagnostic_only_until_paired_future_covariate_ablation"
        ),
        "covariate_ablation_predictions_dir": (
            str(args.covariate_ablation_predictions_dir.resolve())
            if args.covariate_ablation_predictions_dir is not None
            else None
        ),
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
    print(f"E3 mechanism pilot complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
