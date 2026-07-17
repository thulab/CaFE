#!/usr/bin/env python3
"""Calibrate and independently audit the E4-v3 predictive mechanism gates."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from predictive_capability_gate import (  # noqa: E402
    CAPABILITY_IDS,
    evaluate_capability_fingerprint,
    gate_decision,
)


SCHEMA_VERSION = "paper_v3_mechanism_gate_calibration.v1"
EXPERIMENT_ID = "00_mechanism_gate_freeze"
DEFAULT_INPUT = (
    REPO_ROOT / "runtime/paper_exp/v2/E2_dynamic_stability/samples.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runtime/paper_exp/v3/00_mechanism_gate_freeze"
)
CALIBRATION_ROUNDS = (1, 2, 3)
AUDIT_ROUNDS = (4, 5)
SAMPLE_INDICES = (0, 8)
POSITIVE_INTENSITIES = (3, 4, 5)
WEAK_CONTROL_INTENSITY = 1
GAIN_STATISTIC = "pooled_relative_mae_gain"
MINIMUM_POSITIVE_FOLD_FRACTION_OPTIONS = (0.50, 0.75)
MINIMUM_PARAMETER_STABILITY_OPTIONS = (0.50, 0.75)
MAXIMUM_PERMUTATION_PVALUE_OPTIONS = (0.10, 0.15, 0.20)
CALIBRATION_SEED = 2026071761


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit transparent E4-v3 mechanism-gate thresholds on E2 rounds "
            "1-3 and audit them unchanged on rounds 4-5."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
    )
    parser.add_argument(
        "--sample-indices",
        default=",".join(str(value) for value in SAMPLE_INDICES),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_indices = tuple(
        sorted(
            {
                int(value)
                for value in str(args.sample_indices).split(",")
                if value.strip()
            }
        )
    )
    run_calibration(
        args.input.resolve(),
        args.output_dir.resolve(),
        workers=max(1, int(args.workers)),
        sample_indices=sample_indices,
        force=bool(args.force),
    )
    return 0


def run_calibration(
    input_path: Path,
    output_dir: Path,
    *,
    workers: int,
    sample_indices: tuple[int, ...],
    force: bool,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "probe_diagnostics.jsonl"
    if diagnostics_path.exists() and not force:
        diagnostics = read_jsonl(diagnostics_path)
        print(
            f"reusing {len(diagnostics)} cached mechanism diagnostics",
            flush=True,
        )
    else:
        samples = selected_samples(
            input_path,
            sample_indices=sample_indices,
        )
        print(
            f"evaluating {len(samples)} E2 samples with {workers} workers",
            flush=True,
        )
        diagnostics = compute_diagnostics(samples, workers=workers)
        write_jsonl(diagnostics_path, diagnostics)

    frame = diagnostic_frame(diagnostics)
    write_csv(output_dir / "probe_diagnostics.csv", frame)
    calibration = frame[frame["split"] == "calibration"].reset_index(drop=True)
    audit = frame[frame["split"] == "audit"].reset_index(drop=True)
    thresholds, calibration_metrics = calibrate_all_thresholds(calibration)
    audit_decisions = apply_thresholds(audit, thresholds)
    audit_metrics = summarize_all_capabilities(audit_decisions)
    dose_response = summarize_dose_response(audit_decisions)
    confusion = summarize_confusion(audit_decisions)

    write_csv(output_dir / "calibration_operating_points.csv", calibration_metrics)
    write_csv(output_dir / "audit_operating_points.csv", audit_metrics)
    write_csv(output_dir / "audit_dose_response.csv", dose_response)
    write_csv(output_dir / "audit_confusion.csv", confusion)
    write_csv(output_dir / "audit_decisions.csv", audit_decisions)

    artifact = {
        "schema_version": "predictive_capability_gate_thresholds.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "calibration_source": {
            "path": relative_path(input_path),
            "sha256": sha256_file(input_path),
            "rounds": list(CALIBRATION_ROUNDS),
            "sample_indices": list(sample_indices),
            "positive_definition": (
                "source capability equals probe capability and intensity >= 3"
            ),
            "negative_controls": (
                "all other capability generators at intensities 1-5 plus "
                "same-capability intensity 1"
            ),
        },
        "independent_audit": {
            "rounds": list(AUDIT_ROUNDS),
            "threshold_refit": False,
        },
        "probe_contract": {
            "visible_history_only": True,
            "pseudo_future_fold_count": 4,
            "pseudo_future_horizon": "equal to benchmark horizon",
            "primary_gain_statistic": GAIN_STATISTIC,
            "phase_permutation_count": 19,
            "benchmark_future_access": False,
        },
        "thresholds": thresholds,
        "calibration_metrics": frame_to_records(calibration_metrics),
        "audit_metrics": frame_to_records(audit_metrics),
        "audit_dose_response": frame_to_records(dose_response),
        "audit_confusion": frame_to_records(confusion),
    }
    write_json(output_dir / "gate_thresholds_candidate.json", artifact)
    summary = build_summary(
        artifact=artifact,
        diagnostics=frame,
        audit_decisions=audit_decisions,
    )
    write_json(output_dir / "summary.json", summary)
    write_text(output_dir / "report.md", render_report(summary))
    manifest = build_manifest(output_dir)
    write_json(output_dir / "manifest.json", manifest)
    print(render_console_summary(summary), flush=True)


def selected_samples(
    input_path: Path,
    *,
    sample_indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    allowed_rounds = set(CALIBRATION_ROUNDS) | set(AUDIT_ROUNDS)
    allowed_indices = set(sample_indices)
    rows: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["round_index"]) not in allowed_rounds:
                continue
            if int(row["sample_index"]) not in allowed_indices:
                continue
            if str(row["capability_id"]) not in CAPABILITY_IDS:
                continue
            rows.append(row)
    expected = (
        9
        * len(CAPABILITY_IDS)
        * 5
        * len(allowed_rounds)
        * len(sample_indices)
    )
    if len(rows) != expected:
        raise ValueError(
            f"selected E2 calibration grid is incomplete: {len(rows)}/{expected}"
        )
    return rows


def compute_diagnostics(
    samples: list[dict[str, Any]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if workers == 1:
        nested = [_diagnose_sample(sample) for sample in samples]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
        ) as executor:
            nested = list(
                executor.map(
                    _diagnose_sample,
                    samples,
                    chunksize=max(1, len(samples) // (workers * 8)),
                )
            )
    rows = list(itertools.chain.from_iterable(nested))
    rows.sort(
        key=lambda row: (
            row["split"],
            row["profile_id"],
            row["source_capability_id"],
            row["intensity"],
            row["round_index"],
            row["sample_index"],
            row["probe_capability_id"],
        )
    )
    return rows


def _diagnose_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    context_length = int(sample["context_length"])
    target = np.asarray(sample["target"], dtype=float)
    if target.ndim != 2 or target.shape[1] != 1:
        raise ValueError("gate calibration requires univariate E2 samples")
    history = target[:context_length, 0]
    fingerprint = evaluate_capability_fingerprint(
        history,
        season_length=int(sample["season_length"]),
        pseudo_horizon=int(sample["horizon"]),
    )
    split = (
        "calibration"
        if int(sample["round_index"]) in CALIBRATION_ROUNDS
        else "audit"
    )
    rows: list[dict[str, Any]] = []
    for capability_id, diagnostics in fingerprint.items():
        rows.append(
            {
                "schema_version": "paper_v3_gate_probe_diagnostic.v1",
                "split": split,
                "sample_id": str(sample["sample_id"]),
                "profile_id": str(sample["profile_id"]),
                "source_capability_id": str(sample["capability_id"]),
                "probe_capability_id": capability_id,
                "intensity": int(sample["intensity"]),
                "round_index": int(sample["round_index"]),
                "sample_index": int(sample["sample_index"]),
                "diagnostics": diagnostics,
            }
        )
    return rows


def diagnostic_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    flat: list[dict[str, Any]] = []
    for row in rows:
        diagnostics = row["diagnostics"]
        flat.append(
            {
                key: row[key]
                for key in (
                    "split",
                    "sample_id",
                    "profile_id",
                    "source_capability_id",
                    "probe_capability_id",
                    "intensity",
                    "round_index",
                    "sample_index",
                )
            }
            | {
                "valid_fold_count": int(diagnostics["valid_fold_count"]),
                "gain_mean": float(diagnostics["gain_mean"]),
                "gain_median": float(diagnostics["gain_median"]),
                "gain_lcb": float(diagnostics["gain_lcb"]),
                "pooled_relative_mae_gain": float(
                    diagnostics["pooled_relative_mae_gain"]
                ),
                "mse_gain_mean": float(diagnostics["mse_gain_mean"]),
                "positive_fold_fraction": float(
                    diagnostics["positive_fold_fraction"]
                ),
                "support_median": float(diagnostics["support_median"]),
                "parameter_stability": float(
                    diagnostics["parameter_stability"]
                ),
                "phase_permutation_pvalue": float(
                    diagnostics["phase_permutation_pvalue"]
                ),
            }
        )
    return pd.DataFrame.from_records(flat).sort_values(
        [
            "split",
            "profile_id",
            "source_capability_id",
            "intensity",
            "round_index",
            "sample_index",
            "probe_capability_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def calibrate_all_thresholds(
    frame: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    thresholds: dict[str, dict[str, Any]] = {}
    metrics: list[dict[str, Any]] = []
    for capability_id in CAPABILITY_IDS:
        threshold, summary = calibrate_threshold(
            frame,
            capability_id=capability_id,
        )
        thresholds[capability_id] = threshold
        metrics.append(summary)
    return thresholds, pd.DataFrame.from_records(metrics)


def calibrate_threshold(
    frame: pd.DataFrame,
    *,
    capability_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    subset = frame[frame["probe_capability_id"] == capability_id].copy()
    subset["is_positive"] = (
        (subset["source_capability_id"] == capability_id)
        & subset["intensity"].isin(POSITIVE_INTENSITIES)
    )
    subset["is_negative"] = (
        (subset["source_capability_id"] != capability_id)
        | (
            (subset["source_capability_id"] == capability_id)
            & (subset["intensity"] == WEAK_CONTROL_INTENSITY)
        )
    )
    eligible = subset[subset["is_positive"] | subset["is_negative"]].copy()
    positives = eligible[eligible["is_positive"]]
    negatives = eligible[eligible["is_negative"]]
    if positives.empty or negatives.empty:
        raise ValueError(f"{capability_id} calibration labels are empty")

    gain_options = _threshold_options(
        positives[GAIN_STATISTIC].to_numpy(dtype=float),
        negatives[GAIN_STATISTIC].to_numpy(dtype=float),
        fixed=(0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10),
    )
    support_options = _threshold_options(
        positives["support_median"].to_numpy(dtype=float),
        negatives["support_median"].to_numpy(dtype=float),
        fixed=(0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20),
    )
    candidates: list[dict[str, Any]] = []
    for (
        minimum_gain,
        minimum_support,
        minimum_positive_fraction,
        minimum_stability,
        maximum_pvalue,
    ) in itertools.product(
        gain_options,
        support_options,
        MINIMUM_POSITIVE_FOLD_FRACTION_OPTIONS,
        MINIMUM_PARAMETER_STABILITY_OPTIONS,
        MAXIMUM_PERMUTATION_PVALUE_OPTIONS,
    ):
        qualified = _decision_mask(
            eligible,
            minimum_gain=minimum_gain,
            minimum_support=minimum_support,
            minimum_positive_fraction=minimum_positive_fraction,
            minimum_stability=minimum_stability,
            maximum_pvalue=maximum_pvalue,
        )
        evaluated = _operating_point_metrics(
            eligible,
            qualified=qualified,
            capability_id=capability_id,
        )
        feasible = (
            evaluated["true_positive_rate"] >= 0.35
            and evaluated["intensity5_true_positive_rate"] >= 0.50
            and evaluated["false_positive_rate"] <= 0.10
            and evaluated["maximum_other_capability_false_positive_rate"]
            <= 0.25
            and evaluated["intensity5_minus_intensity1_rate"] >= 0.20
        )
        objective = (
            evaluated["true_positive_rate"]
            - 0.60 * evaluated["false_positive_rate"]
            - 0.25
            * evaluated["maximum_other_capability_false_positive_rate"]
            + 0.15 * evaluated["intensity5_minus_intensity1_rate"]
        )
        candidates.append(
            {
                **evaluated,
                "feasible": bool(feasible),
                "objective": float(objective),
                "minimum_predictive_gain": float(minimum_gain),
                "minimum_support": float(minimum_support),
                "minimum_positive_fold_fraction": float(
                    minimum_positive_fraction
                ),
                "minimum_parameter_stability": float(minimum_stability),
                "maximum_phase_permutation_pvalue": float(maximum_pvalue),
            }
        )
    if not candidates:
        raise ValueError(f"{capability_id} has no threshold candidates")
    best = max(
        candidates,
        key=lambda item: (
            int(item["feasible"]),
            item["objective"],
            item["true_positive_rate"],
            -item["false_positive_rate"],
            -item["maximum_other_capability_false_positive_rate"],
            -item["minimum_predictive_gain"],
            -item["minimum_support"],
        ),
    )
    threshold = {
        "gain_statistic": GAIN_STATISTIC,
        "minimum_predictive_gain": best["minimum_predictive_gain"],
        "minimum_positive_fold_fraction": best[
            "minimum_positive_fold_fraction"
        ],
        "minimum_support": best["minimum_support"],
        "minimum_parameter_stability": best[
            "minimum_parameter_stability"
        ],
        "maximum_phase_permutation_pvalue": best[
            "maximum_phase_permutation_pvalue"
        ],
        "minimum_valid_folds": 3,
        "gain_normalization_scale": max(
            float(best["minimum_predictive_gain"]),
            float(positives[GAIN_STATISTIC].clip(lower=0.0).median()),
            0.01,
        ),
        "support_normalization_scale": max(
            float(best["minimum_support"]),
            float(positives["support_median"].clip(lower=0.0).median()),
            0.01,
        ),
        "selection_rule": (
            "maximize predeclared TPR/FPR/dose objective subject to synthetic "
            "high-intensity recall and nuisance-control constraints"
        ),
        "strict_constraints_satisfied": bool(best["feasible"]),
    }
    summary = {
        "capability_id": capability_id,
        **{
            key: value
            for key, value in best.items()
            if key
            not in {
                "minimum_predictive_gain",
                "minimum_support",
                "minimum_positive_fold_fraction",
                "minimum_parameter_stability",
                "maximum_phase_permutation_pvalue",
            }
        },
        **threshold,
    }
    return threshold, summary


def _threshold_options(
    positives: np.ndarray,
    negatives: np.ndarray,
    *,
    fixed: tuple[float, ...],
) -> tuple[float, ...]:
    values = set(float(value) for value in fixed)
    for array, quantiles in (
        (negatives, (0.80, 0.90, 0.95, 0.975)),
        (positives, (0.05, 0.10, 0.20, 0.30)),
    ):
        finite = array[np.isfinite(array)]
        if len(finite):
            values.update(
                round(float(np.quantile(finite, quantile)), 8)
                for quantile in quantiles
            )
    return tuple(
        sorted(value for value in values if value >= 0.0 and math.isfinite(value))
    )


def _decision_mask(
    frame: pd.DataFrame,
    *,
    minimum_gain: float,
    minimum_support: float,
    minimum_positive_fraction: float,
    minimum_stability: float,
    maximum_pvalue: float,
) -> np.ndarray:
    return (
        (frame["valid_fold_count"].to_numpy(dtype=int) >= 3)
        & (
            frame[GAIN_STATISTIC].to_numpy(dtype=float)
            >= float(minimum_gain)
        )
        & (
            frame["positive_fold_fraction"].to_numpy(dtype=float)
            >= float(minimum_positive_fraction)
        )
        & (
            frame["support_median"].to_numpy(dtype=float)
            >= float(minimum_support)
        )
        & (
            frame["parameter_stability"].to_numpy(dtype=float)
            >= float(minimum_stability)
        )
        & (
            frame["phase_permutation_pvalue"].to_numpy(dtype=float)
            <= float(maximum_pvalue)
        )
    )


def _operating_point_metrics(
    frame: pd.DataFrame,
    *,
    qualified: np.ndarray,
    capability_id: str,
) -> dict[str, Any]:
    positive = frame["is_positive"].to_numpy(dtype=bool)
    negative = frame["is_negative"].to_numpy(dtype=bool)
    true_positive_rate = float(np.mean(qualified[positive]))
    false_positive_rate = float(np.mean(qualified[negative]))
    other_rates = []
    for source_capability in CAPABILITY_IDS:
        if source_capability == capability_id:
            continue
        mask = (
            frame["source_capability_id"].to_numpy(dtype=str)
            == source_capability
        )
        if bool(np.any(mask)):
            other_rates.append(float(np.mean(qualified[mask])))
    intensity_rates: dict[int, float] = {}
    for intensity in range(1, 6):
        mask = (
            (
                frame["source_capability_id"].to_numpy(dtype=str)
                == capability_id
            )
            & (frame["intensity"].to_numpy(dtype=int) == intensity)
        )
        intensity_rates[intensity] = (
            float(np.mean(qualified[mask])) if bool(np.any(mask)) else 0.0
        )
    balanced_precision = true_positive_rate / max(
        true_positive_rate + false_positive_rate,
        1e-12,
    )
    return {
        "capability_id": capability_id,
        "positive_count": int(np.sum(positive)),
        "negative_count": int(np.sum(negative)),
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "maximum_other_capability_false_positive_rate": max(
            other_rates,
            default=0.0,
        ),
        "balanced_precision": float(balanced_precision),
        "intensity1_qualification_rate": intensity_rates[1],
        "intensity2_qualification_rate": intensity_rates[2],
        "intensity3_qualification_rate": intensity_rates[3],
        "intensity4_qualification_rate": intensity_rates[4],
        "intensity5_qualification_rate": intensity_rates[5],
        "intensity5_true_positive_rate": intensity_rates[5],
        "intensity5_minus_intensity1_rate": (
            intensity_rates[5] - intensity_rates[1]
        ),
    }


def apply_thresholds(
    frame: pd.DataFrame,
    thresholds: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        capability_id = str(record["probe_capability_id"])
        diagnostics = {
            key: record[key]
            for key in (
                "valid_fold_count",
                "gain_mean",
                "gain_median",
                "gain_lcb",
                "pooled_relative_mae_gain",
                "positive_fold_fraction",
                "support_median",
                "parameter_stability",
                "phase_permutation_pvalue",
            )
        }
        decision = gate_decision(diagnostics, thresholds[capability_id])
        rows.append(
            {
                **record,
                "qualified": bool(decision["qualified"]),
                "fingerprint_weight": float(
                    decision["fingerprint_weight"]
                ),
                "is_target_capability": (
                    str(record["source_capability_id"]) == capability_id
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_all_capabilities(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for capability_id in CAPABILITY_IDS:
        subset = frame[frame["probe_capability_id"] == capability_id].copy()
        subset["is_positive"] = (
            (subset["source_capability_id"] == capability_id)
            & subset["intensity"].isin(POSITIVE_INTENSITIES)
        )
        subset["is_negative"] = (
            (subset["source_capability_id"] != capability_id)
            | (
                (subset["source_capability_id"] == capability_id)
                & (subset["intensity"] == WEAK_CONTROL_INTENSITY)
            )
        )
        eligible = subset[subset["is_positive"] | subset["is_negative"]]
        rows.append(
            _operating_point_metrics(
                eligible,
                qualified=eligible["qualified"].to_numpy(dtype=bool),
                capability_id=capability_id,
            )
        )
    return pd.DataFrame.from_records(rows)


def summarize_dose_response(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for capability_id, intensity in itertools.product(
        CAPABILITY_IDS,
        range(1, 6),
    ):
        subset = frame[
            (frame["probe_capability_id"] == capability_id)
            & (frame["source_capability_id"] == capability_id)
            & (frame["intensity"] == intensity)
        ]
        rows.append(
            {
                "capability_id": capability_id,
                "intensity": intensity,
                "sample_count": len(subset),
                "qualification_rate": float(
                    subset["qualified"].mean() if len(subset) else 0.0
                ),
                "median_fingerprint_weight": float(
                    subset["fingerprint_weight"].median()
                    if len(subset)
                    else 0.0
                ),
                "median_predictive_gain": float(
                    subset[GAIN_STATISTIC].median()
                    if len(subset)
                    else 0.0
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_confusion(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_capability, probe_capability in itertools.product(
        CAPABILITY_IDS,
        CAPABILITY_IDS,
    ):
        subset = frame[
            (frame["source_capability_id"] == source_capability)
            & (frame["probe_capability_id"] == probe_capability)
            & (frame["intensity"].isin(POSITIVE_INTENSITIES))
        ]
        rows.append(
            {
                "source_capability_id": source_capability,
                "probe_capability_id": probe_capability,
                "sample_count": len(subset),
                "qualification_rate": float(
                    subset["qualified"].mean() if len(subset) else 0.0
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_summary(
    *,
    artifact: dict[str, Any],
    diagnostics: pd.DataFrame,
    audit_decisions: pd.DataFrame,
) -> dict[str, Any]:
    audit = pd.DataFrame.from_records(artifact["audit_metrics"])
    calibration = pd.DataFrame.from_records(
        artifact["calibration_metrics"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": artifact["created_at"],
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_row_count": len(diagnostics),
        "audit_decision_count": len(audit_decisions),
        "calibration_sample_count": int(
            diagnostics[diagnostics["split"] == "calibration"][
                "sample_id"
            ].nunique()
        ),
        "audit_sample_count": int(
            diagnostics[diagnostics["split"] == "audit"][
                "sample_id"
            ].nunique()
        ),
        "strict_constraint_capability_count": int(
            calibration["strict_constraints_satisfied"].sum()
        ),
        "audit_macro_true_positive_rate": float(
            audit["true_positive_rate"].mean()
        ),
        "audit_macro_false_positive_rate": float(
            audit["false_positive_rate"].mean()
        ),
        "audit_worst_other_capability_false_positive_rate": float(
            audit[
                "maximum_other_capability_false_positive_rate"
            ].max()
        ),
        "audit_capabilities": frame_to_records(audit),
        "thresholds": artifact["thresholds"],
        "interpretation": (
            "Passing means a frozen capability probe gives stable, "
            "phase-specific pseudo-future headroom over a nuisance-matched "
            "baseline. It is evidence of mechanism-aligned predictive behavior, "
            "not proof of a causal data-generating mechanism."
        ),
    }


def render_console_summary(summary: dict[str, Any]) -> str:
    return (
        "mechanism gate calibration complete: "
        f"strict={summary['strict_constraint_capability_count']}/6, "
        f"audit macro TPR={summary['audit_macro_true_positive_rate']:.3f}, "
        f"audit macro FPR={summary['audit_macro_false_positive_rate']:.3f}, "
        "output="
        f"{relative_path(DEFAULT_OUTPUT_DIR)}"
    )


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Paper E4-v3：预测机制门控校准",
        "",
        "本实验只使用 E2 合成样本的可见 context。round 1–3 用于选阈值，"
        "round 4–5 在不重拟合阈值的情况下独立审计。",
        "",
        "门控含义是 capability-specific probe 相对 nuisance-matched baseline "
        "在四个 context 内伪未来上获得稳定增益；它不是因果机制识别。",
        "",
        "## 汇总",
        "",
        f"- 校准样本：{summary['calibration_sample_count']}",
        f"- 独立审计样本：{summary['audit_sample_count']}",
        (
            "- 满足严格校准约束的能力："
            f"{summary['strict_constraint_capability_count']} / 6"
        ),
        (
            "- 审计 macro TPR / FPR："
            f"{summary['audit_macro_true_positive_rate']:.4f} / "
            f"{summary['audit_macro_false_positive_rate']:.4f}"
        ),
        (
            "- 审计最差 alternate-capability FPR："
            f"{summary['audit_worst_other_capability_false_positive_rate']:.4f}"
        ),
        "",
        "## 分能力审计",
        "",
        "| capability | TPR | FPR | max alternate FPR | I1 rate | I5 rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["audit_capabilities"]:
        lines.append(
            f"| `{row['capability_id']}` | "
            f"{row['true_positive_rate']:.4f} | "
            f"{row['false_positive_rate']:.4f} | "
            f"{row['maximum_other_capability_false_positive_rate']:.4f} | "
            f"{row['intensity1_qualification_rate']:.4f} | "
            f"{row['intensity5_qualification_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "完整阈值、dose-response、confusion matrix 和逐样本诊断见同目录产物。",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": "paper_experiment_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "files": files,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.10g")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in frame.to_dict(orient="records"):
        records.append(
            {
                key: (
                    value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in row.items()
            }
        )
    return records


if __name__ == "__main__":
    raise SystemExit(main())
