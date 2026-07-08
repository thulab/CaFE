#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (BACKEND_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.services.synthetic_generation_service import (  # noqa: E402
    CAPABILITIES_BY_ID,
    _generate_accepted_sample_values,
    _seed_for,
    _standardize_by_context,
)
from synthetic_feature_profile import (  # noqa: E402
    DEFAULT_FEATURES,
    WindowSpec,
    feature_vector,
    read_tsf_series,
    window_starts,
)


DEFAULT_M4_PATH = REPO_ROOT / "runtime/research/m4_hourly_dataset.zip"
DEFAULT_US_BIRTHS_PATH = REPO_ROOT / "runtime/research/us_births_dataset.zip"
CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24
DEFAULT_SYNTHETIC_CAPABILITIES = tuple(CAPABILITIES_BY_ID)
FEATURE_DISTANCE_COLUMNS = tuple(DEFAULT_FEATURES)
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class WindowRecord:
    dataset: str
    series_index: int
    start: int
    values: np.ndarray


@dataclass(frozen=True)
class QueryRecord:
    group_id: str
    capability_id: str
    difficulty: int | None
    sample_index: int | None
    target_index: int
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate real-vs-real DCR and synthetic-vs-real NN ratios for paper novelty validation."
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument("--m4-path", type=Path, default=DEFAULT_M4_PATH)
    parser.add_argument("--us-births-path", type=Path, default=DEFAULT_US_BIRTHS_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / f"runtime/research/agent-novelty-calibration-{timestamp}",
    )
    parser.add_argument("--max-train", type=int, default=640)
    parser.add_argument("--max-holdout", type=int, default=320)
    parser.add_argument("--synthetic-count", type=int, default=48, help="Samples per capability and difficulty.")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--capabilities", nargs="*", default=list(DEFAULT_SYNTHETIC_CAPABILITIES))
    parser.add_argument("--include-us-births", action="store_true", help="Also run a small US Births daily check.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    capabilities = validate_capabilities(args.capabilities)
    synthetic_queries = generate_synthetic_queries(
        capabilities,
        sample_count=args.synthetic_count,
        seed=args.seed,
    )
    experiments = [
        DatasetExperiment(
            name="m4_hourly",
            path=args.m4_path,
            context_length=CONTEXT_LENGTH,
            horizon=HORIZON,
            season_length=SEASON_LENGTH,
            split_mode="series",
            max_train=args.max_train,
            max_holdout=args.max_holdout,
        )
    ]
    if args.include_us_births:
        experiments.append(
            DatasetExperiment(
                name="us_births_daily",
                path=args.us_births_path,
                context_length=365,
                horizon=30,
                season_length=7,
                split_mode="time_block",
                max_train=40,
                max_holdout=20,
            )
        )

    dataset_results = []
    for experiment in experiments:
        dataset_results.append(
            run_dataset_experiment(
                experiment,
                synthetic_queries=synthetic_queries,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
            )
        )

    summary = {
        "schema_version": "agent_novelty_calibration_probe.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "max_train": args.max_train,
            "max_holdout": args.max_holdout,
            "synthetic_count_per_capability_difficulty": args.synthetic_count,
            "seed": args.seed,
            "capabilities": capabilities,
            "include_us_births": bool(args.include_us_births),
        },
        "datasets": dataset_results,
        "recommended_rule": recommended_rule_text(),
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(f"wrote output dir: {args.output_dir}")
    print(f"wrote summary: {args.output_dir / 'summary.json'}")
    print(f"wrote report: {args.output_dir / 'report.md'}")
    return 0


@dataclass(frozen=True)
class DatasetExperiment:
    name: str
    path: Path
    context_length: int
    horizon: int
    season_length: int
    split_mode: str
    max_train: int
    max_holdout: int

    @property
    def length(self) -> int:
        return self.context_length + self.horizon


def validate_capabilities(capabilities: list[str]) -> list[str]:
    missing = [capability for capability in capabilities if capability not in CAPABILITIES_BY_ID]
    if missing:
        raise SystemExit(f"unknown synthetic capabilities: {', '.join(missing)}")
    return list(capabilities)


def run_dataset_experiment(
    experiment: DatasetExperiment,
    *,
    synthetic_queries: list[QueryRecord],
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    train_records, holdout_records, split_info = load_real_train_holdout(experiment)
    if not train_records or not holdout_records:
        raise RuntimeError(f"{experiment.name}: need non-empty train and holdout windows")

    train = stack_standardized(train_records, experiment.context_length)
    holdout = stack_standardized(holdout_records, experiment.context_length)
    train_features = feature_matrix(train, experiment.season_length)
    holdout_features = feature_matrix(holdout, experiment.season_length)
    feature_scaler = fit_feature_scaler(train_features)

    real_metrics = compute_all_metrics(
        holdout,
        train,
        holdout_features,
        train_features,
        feature_scaler,
        batch_size=batch_size,
    )
    real_baselines = {
        metric: nearest_summary(result["d1"], result["nndr"])
        for metric, result in real_metrics.items()
    }

    compatible_queries = [
        query
        for query in synthetic_queries
        if query.values.shape[0] == experiment.length
    ]
    if compatible_queries:
        synthetic = stack_queries(compatible_queries, experiment.context_length)
        synthetic_features = feature_matrix(synthetic, experiment.season_length)
        synthetic_metrics = compute_all_metrics(
            synthetic,
            train,
            synthetic_features,
            train_features,
            feature_scaler,
            batch_size=batch_size,
        )
        sample_rows = synthetic_sample_rows(compatible_queries, synthetic_metrics, real_baselines)
    else:
        sample_rows = []
    group_rows = group_synthetic_rows(sample_rows)

    prefix = output_dir / experiment.name
    write_baseline_csv(prefix.with_name(prefix.name + "_real_baselines.csv"), real_baselines)
    write_group_csv(prefix.with_name(prefix.name + "_synthetic_group_ratios.csv"), group_rows)
    write_sample_jsonl(prefix.with_name(prefix.name + "_synthetic_samples.jsonl"), sample_rows)

    return {
        "dataset": experiment.name,
        "source_path": display_path(experiment.path),
        "context_length": experiment.context_length,
        "horizon": experiment.horizon,
        "season_length": experiment.season_length,
        "split": split_info,
        "train_window_count": len(train_records),
        "holdout_window_count": len(holdout_records),
        "synthetic_target_window_count": len(compatible_queries),
        "real_baselines": real_baselines,
        "synthetic_group_rows": group_rows,
        "files": {
            "real_baselines_csv": display_path(prefix.with_name(prefix.name + "_real_baselines.csv")),
            "synthetic_group_ratios_csv": display_path(prefix.with_name(prefix.name + "_synthetic_group_ratios.csv")),
            "synthetic_samples_jsonl": display_path(prefix.with_name(prefix.name + "_synthetic_samples.jsonl")),
        },
    }


def load_real_train_holdout(experiment: DatasetExperiment) -> tuple[list[WindowRecord], list[WindowRecord], dict[str, Any]]:
    if not experiment.path.exists():
        raise FileNotFoundError(f"real dataset not found: {experiment.path}")
    _metadata, series = read_tsf_series(experiment.path)
    spec = WindowSpec(experiment.context_length, experiment.horizon, experiment.horizon)
    records: list[WindowRecord] = []
    for series_index, (_series_id, values) in enumerate(series):
        for start in window_starts(len(values), spec):
            window = values[start : start + spec.length, None]
            if np.isfinite(window).all():
                records.append(WindowRecord(experiment.name, series_index, start, window))
    if experiment.split_mode == "series" and len(series) >= 4:
        train = [record for record in records if record.series_index % 5 != 4]
        holdout = [record for record in records if record.series_index % 5 == 4]
        split_info = {
            "mode": "series_modulo",
            "train_series_rule": "series_index % 5 != 4",
            "holdout_series_rule": "series_index % 5 == 4",
            "series_count": len(series),
        }
    else:
        train, holdout = split_single_series_by_time(records, experiment.length)
        split_info = {
            "mode": "time_block_with_gap",
            "gap_points": experiment.length,
            "series_count": len(series),
            "limitation": "single-series check; useful for sensitivity, not the main paper threshold",
        }

    return (
        even_sample(train, experiment.max_train),
        even_sample(holdout, experiment.max_holdout),
        split_info,
    )


def split_single_series_by_time(records: list[WindowRecord], window_length: int) -> tuple[list[WindowRecord], list[WindowRecord]]:
    by_start = sorted(records, key=lambda record: (record.series_index, record.start))
    if not by_start:
        return [], []
    max_start = max(record.start for record in by_start)
    train_cut = int(max_start * 0.55)
    holdout_cut = train_cut + window_length
    train = [record for record in by_start if record.start <= train_cut]
    holdout = [record for record in by_start if record.start >= holdout_cut]
    return train, holdout


def even_sample(records: list[WindowRecord], count: int) -> list[WindowRecord]:
    if count <= 0 or len(records) <= count:
        return list(records)
    indexes = np.linspace(0, len(records) - 1, count).round().astype(int)
    return [records[index] for index in sorted(set(indexes.tolist()))]


def stack_standardized(records: list[WindowRecord], context_length: int) -> np.ndarray:
    values = [_standardize_by_context(record.values, context_length)[:, 0] for record in records]
    return np.asarray(values, dtype=float)


def stack_queries(records: list[QueryRecord], context_length: int) -> np.ndarray:
    values = [_standardize_by_context(record.values[:, None], context_length)[:, 0] for record in records]
    return np.asarray(values, dtype=float)


def feature_matrix(windows: np.ndarray, season_length: int) -> np.ndarray:
    if windows.size == 0:
        return np.empty((0, len(FEATURE_DISTANCE_COLUMNS)), dtype=float)
    rows = []
    for window in windows:
        features = feature_vector(window[:, None], season_length=season_length)
        rows.append([features.get(column, 0.0) for column in FEATURE_DISTANCE_COLUMNS])
    return np.asarray(rows, dtype=float)


def fit_feature_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(features, axis=0)
    scale = np.nanstd(features, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return center, scale


def compute_all_metrics(
    queries: np.ndarray,
    train: np.ndarray,
    query_features: np.ndarray,
    train_features: np.ndarray,
    feature_scaler: tuple[np.ndarray, np.ndarray],
    *,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    center, scale = feature_scaler
    return {
        "z_l2": nearest_distances(queries, train, metric="l2", batch_size=batch_size),
        "z_mae": nearest_distances(queries, train, metric="mae", batch_size=batch_size),
        "feature_l2": nearest_distances(
            (query_features - center) / scale,
            (train_features - center) / scale,
            metric="l2",
            batch_size=batch_size,
        ),
    }


def nearest_distances(
    queries: np.ndarray,
    refs: np.ndarray,
    *,
    metric: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    nearest_index: list[np.ndarray] = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        diff = batch[:, None, :] - refs[None, :, :]
        if metric == "l2":
            distances = np.sqrt(np.mean(diff * diff, axis=2))
        elif metric == "mae":
            distances = np.mean(np.abs(diff), axis=2)
        else:
            raise ValueError(f"unsupported metric: {metric}")
        if distances.shape[1] == 1:
            order = np.zeros((distances.shape[0], 1), dtype=int)
            d1 = distances[:, 0]
            d2 = np.full_like(d1, np.inf)
        else:
            order = np.argpartition(distances, kth=1, axis=1)[:, :2]
            pair = np.take_along_axis(distances, order, axis=1)
            swap = pair[:, 0] > pair[:, 1]
            pair[swap] = pair[swap][:, ::-1]
            order[swap] = order[swap][:, ::-1]
            d1 = pair[:, 0]
            d2 = pair[:, 1]
        first.append(d1)
        second.append(d2)
        nearest_index.append(order[:, 0])
    d1_all = np.concatenate(first)
    d2_all = np.concatenate(second)
    return {
        "d1": d1_all,
        "d2": d2_all,
        "nndr": d1_all / np.maximum(d2_all, 1e-12),
        "nearest_index": np.concatenate(nearest_index),
    }


def nearest_summary(d1: np.ndarray, nndr: np.ndarray) -> dict[str, Any]:
    return {
        "dcr": summarize_values(d1),
        "nndr": summarize_values(nndr[np.isfinite(nndr)]),
    }


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return {"count": 0}
    out: dict[str, float | int] = {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }
    for quantile in QUANTILES:
        out[f"p{int(quantile * 100):02d}"] = float(np.quantile(finite, quantile))
    return out


def synthetic_sample_rows(
    queries: list[QueryRecord],
    synthetic_metrics: dict[str, dict[str, np.ndarray]],
    real_baselines: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries):
        row: dict[str, Any] = {
            "group_id": query.group_id,
            "capability_id": query.capability_id,
            "difficulty": query.difficulty,
            "sample_index": query.sample_index,
            "target_index": query.target_index,
        }
        for metric, result in synthetic_metrics.items():
            d1 = float(result["d1"][index])
            d2 = float(result["d2"][index])
            nndr = float(result["nndr"][index])
            row[f"{metric}_dcr"] = d1
            row[f"{metric}_d2"] = d2
            row[f"{metric}_nndr"] = nndr
            row[f"{metric}_dcr_to_real_p01"] = d1 / max(float(real_baselines[metric]["dcr"]["p01"]), 1e-12)
            row[f"{metric}_dcr_to_real_p05"] = d1 / max(float(real_baselines[metric]["dcr"]["p05"]), 1e-12)
            row[f"{metric}_dcr_to_real_p50"] = d1 / max(float(real_baselines[metric]["dcr"]["p50"]), 1e-12)
        row["copy_risk_strict"] = strict_copy_risk(row, real_baselines)
        row["copy_risk_combined"] = combined_copy_risk(row, real_baselines)
        rows.append(row)
    return rows


def strict_copy_risk(row: dict[str, Any], real_baselines: dict[str, dict[str, Any]]) -> bool:
    return (
        float(row["z_l2_dcr"]) < float(real_baselines["z_l2"]["dcr"]["p01"])
        and float(row["z_mae_dcr"]) < float(real_baselines["z_mae"]["dcr"]["p01"])
    )


def combined_copy_risk(row: dict[str, Any], real_baselines: dict[str, dict[str, Any]]) -> bool:
    low_dcr = (
        float(row["z_l2_dcr"]) < float(real_baselines["z_l2"]["dcr"]["p05"])
        and float(row["z_mae_dcr"]) < float(real_baselines["z_mae"]["dcr"]["p05"])
    )
    singular_neighbor = (
        float(row["z_l2_nndr"]) < float(real_baselines["z_l2"]["nndr"]["p01"])
        or float(row["z_mae_nndr"]) < float(real_baselines["z_mae"]["nndr"]["p01"])
    )
    feature_too_close = float(row["feature_l2_dcr"]) < float(real_baselines["feature_l2"]["dcr"]["p01"])
    return bool(strict_copy_risk(row, real_baselines) or (low_dcr and (singular_neighbor or feature_too_close)))


def group_synthetic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["capability_id"]), row["difficulty"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (capability_id, difficulty), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1] or 0)):
        record: dict[str, Any] = {
            "capability_id": capability_id,
            "difficulty": difficulty,
            "target_window_count": len(group),
            "strict_copy_risk_rate": mean_bool(row["copy_risk_strict"] for row in group),
            "combined_copy_risk_rate": mean_bool(row["copy_risk_combined"] for row in group),
        }
        for metric in ("z_l2", "z_mae", "feature_l2"):
            dcr = np.asarray([float(row[f"{metric}_dcr"]) for row in group], dtype=float)
            nndr = np.asarray([float(row[f"{metric}_nndr"]) for row in group], dtype=float)
            to_p05 = np.asarray([float(row[f"{metric}_dcr_to_real_p05"]) for row in group], dtype=float)
            to_p50 = np.asarray([float(row[f"{metric}_dcr_to_real_p50"]) for row in group], dtype=float)
            record[f"{metric}_dcr_p05"] = float(np.quantile(dcr, 0.05))
            record[f"{metric}_dcr_p50"] = float(np.quantile(dcr, 0.50))
            record[f"{metric}_nndr_p05"] = float(np.quantile(nndr, 0.05))
            record[f"{metric}_nndr_p50"] = float(np.quantile(nndr, 0.50))
            record[f"{metric}_dcr_to_real_p05_p05"] = float(np.quantile(to_p05, 0.05))
            record[f"{metric}_dcr_to_real_p50_p50"] = float(np.quantile(to_p50, 0.50))
        out.append(record)
    return out


def mean_bool(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    return float(np.mean(items)) if items else 0.0


def generate_synthetic_queries(
    capabilities: list[str],
    *,
    sample_count: int,
    seed: int,
) -> list[QueryRecord]:
    queries: list[QueryRecord] = []
    length = CONTEXT_LENGTH + HORIZON
    for capability_id in capabilities:
        capability = CAPABILITIES_BY_ID[capability_id]
        target_dim = 3 if capability.target_dim_mode == "multi" else 1
        for difficulty in range(1, 6):
            for sample_index in range(sample_count):
                sample_seed = _seed_for(seed, capability_id, difficulty * 10_000 + sample_index)
                values, _latent, _covariates, _features = _generate_accepted_sample_values(
                    capability_id,
                    length,
                    CONTEXT_LENGTH,
                    target_dim,
                    SEASON_LENGTH,
                    difficulty,
                    sample_seed,
                )
                for target_index in range(values.shape[1]):
                    queries.append(
                        QueryRecord(
                            group_id=f"{capability_id}_d{difficulty}",
                            capability_id=capability_id,
                            difficulty=difficulty,
                            sample_index=sample_index,
                            target_index=target_index,
                            values=values[:, target_index],
                        )
                    )
    return queries


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_baseline_csv(path: Path, baselines: dict[str, dict[str, Any]]) -> None:
    fieldnames = ["metric", "kind", "count", "min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric, parts in baselines.items():
            for kind in ("dcr", "nndr"):
                row = {"metric": metric, "kind": kind, **parts[kind]}
                writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sample_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Agent Novelty Calibration Probe",
        "",
        "## Scope",
        "",
        "- Train/holdout calibration on real anchor windows.",
        "- Real-vs-real nearest distances define DCR and NNDR lower-tail baselines.",
        "- Synthetic target channels are compared to real train anchors independently.",
        "- No discriminative or predictive score is computed.",
        "",
        "## Recommended Rule",
        "",
        recommended_rule_text(),
        "",
    ]
    for dataset in summary["datasets"]:
        lines.extend(render_dataset_section(dataset))
    return "\n".join(lines) + "\n"


def render_dataset_section(dataset: dict[str, Any]) -> list[str]:
    baseline_rows = [
        "| Metric | DCR p01 | DCR p05 | DCR p50 | NNDR p01 | NNDR p05 | NNDR p50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric, parts in dataset["real_baselines"].items():
        baseline_rows.append(
            "| "
            + " | ".join(
                [
                    metric,
                    fmt(parts["dcr"]["p01"]),
                    fmt(parts["dcr"]["p05"]),
                    fmt(parts["dcr"]["p50"]),
                    fmt(parts["nndr"]["p01"]),
                    fmt(parts["nndr"]["p05"]),
                    fmt(parts["nndr"]["p50"]),
                ]
            )
            + " |"
        )
    group_rows = [
        "| Capability | d | n | strict risk | combined risk | z-L2 p05/p50 ratio | z-MAE p05/p50 ratio | feature p05/p50 ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset["synthetic_group_rows"]:
        group_rows.append(
            "| "
            + " | ".join(
                [
                    str(row["capability_id"]),
                    str(row["difficulty"]),
                    str(row["target_window_count"]),
                    pct(row["strict_copy_risk_rate"]),
                    pct(row["combined_copy_risk_rate"]),
                    f"{fmt(row['z_l2_dcr_to_real_p05_p05'])}/{fmt(row['z_l2_dcr_to_real_p50_p50'])}",
                    f"{fmt(row['z_mae_dcr_to_real_p05_p05'])}/{fmt(row['z_mae_dcr_to_real_p50_p50'])}",
                    f"{fmt(row['feature_l2_dcr_to_real_p05_p05'])}/{fmt(row['feature_l2_dcr_to_real_p50_p50'])}",
                ]
            )
            + " |"
        )
    return [
        f"## {dataset['dataset']}",
        "",
        f"- Source: `{dataset['source_path']}`",
        f"- Windows: train={dataset['train_window_count']}, holdout={dataset['holdout_window_count']}, synthetic targets={dataset['synthetic_target_window_count']}",
        f"- Split: `{dataset['split']['mode']}`",
        f"- CSV outputs: `{dataset['files']['real_baselines_csv']}`, `{dataset['files']['synthetic_group_ratios_csv']}`",
        "",
        "### Real-vs-real Baselines",
        "",
        *baseline_rows,
        "",
        "### Synthetic-vs-real Ratios",
        "",
        *group_rows,
        "",
    ]


def recommended_rule_text() -> str:
    return (
        "For each real anchor bucket, split train and holdout without overlap leakage. "
        "Compute holdout-to-train DCR lower tails for z-L2 and z-MAE over context-standardized target windows, "
        "plus feature-L2 as a proxy check. A synthetic target window is a strict contamination risk when both "
        "z-L2 and z-MAE nearest-anchor distances are below the real holdout p01. It is a combined risk when both "
        "shape distances are below real p05 and either NNDR is below the real p01 NNDR tail or feature-L2 is below "
        "the real p01 feature-DCR tail. Accept a generated batch only when strict risk is zero and combined risk is "
        "at most 1%; report z-L2/z-MAE p05 and median DCR ratios to the real holdout p05/p50 baselines."
    )


def fmt(value: Any) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%"


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
