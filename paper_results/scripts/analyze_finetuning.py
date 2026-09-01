#!/usr/bin/env python3
"""Reproduce the CaFE fine-tuning summaries and publication figures.

Run from the CaFE repository root with:

    uv run --with matplotlib python paper_results/work/finetuning/analyze_finetuning.py

The script reads only the local snapshots under ``raw/``.  It never connects to
the experiment server or mutates the remote experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


OBJECTIVES = {
    "default": {
        "label": "Chronos default (quantile)",
        "short_label": "Default quantile",
        "curve": "raw/default/results/default-loss-curve.csv",
        "parts": "raw/default/results/metric-parts",
        "trainer_state": "raw/default/trainer_state.json",
        "manifest": "raw/default/cafe_training_manifest.json",
        "aligned_metric": "mase",
    },
    "effect_nrmse": {
        "label": "Paired effect NRMSE",
        "short_label": "Effect NRMSE",
        "curve": "raw/nrmse/results/nrmse-loss-curve.csv",
        "parts": "raw/nrmse/results/metric-parts",
        "trainer_state": "raw/nrmse/trainer_state.json",
        "manifest": "raw/nrmse/cafe_effect_training_manifest.json",
        "aligned_metric": "effect_nrmse",
    },
}

METRICS = {
    "mase": {
        "column": "macro_stratum_mase",
        "label": "Macro-stratum MASE",
        "short_label": "MASE",
    },
    "effect_nrmse": {
        "column": "macro_stratum_effect_nrmse",
        "label": "Macro-stratum effect NRMSE",
        "short_label": "Effect NRMSE",
    },
}

CORPUS_LABELS = {"train": "Seed A (training)", "cross": "Seed B (cross-seed)"}
EFFECT_FIELDS = (
    "candidate_count",
    "scored_count",
    "squared_error_sum",
    "truth_squared_sum",
    "observed_cell_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Fine-tuning work directory containing raw/ (default: script directory).",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=20_000,
        help="Paired stratum bootstrap replicates for descriptive intervals.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    names = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_curve(path: Path, objective: str) -> list[dict[str, Any]]:
    integer_fields = {
        "step",
        "mase_stratum_count",
        "effect_stratum_count",
        "treatment_count",
        "effect_candidate_count",
        "effect_scored_count",
        "observed_effect_cell_count",
    }
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {"objective_id": objective}
            for key, value in raw.items():
                if key in {"objective", "corpus"}:
                    row[key] = value
                elif key in integer_fields:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    expected = {(corpus, step) for corpus in ("train", "cross") for step in range(0, 40_001, 4_000)}
    observed = {(str(row["corpus"]), int(row["step"])) for row in rows}
    if observed != expected:
        raise ValueError(f"Unexpected checkpoint grid in {path}: missing={expected-observed}, extra={observed-expected}")
    return rows


def aggregate_metric_parts(parts_root: Path) -> tuple[dict[tuple[str, int], dict[tuple[str, str, int], dict[str, float]]], list[dict[str, Any]]]:
    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(parts_root.rglob("rank-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "chronos2.cafe_direct_metric_part.v1":
            continue
        if not payload.get("complete"):
            raise ValueError(f"Incomplete metric part: {path}")
        grouped[(str(payload["corpus"]), int(payload["step"]))].append(payload)

    strata_by_group: dict[tuple[str, int], dict[tuple[str, str, int], dict[str, float]]] = {}
    audit_rows: list[dict[str, Any]] = []
    for (corpus, step), parts in sorted(grouped.items()):
        world_sizes = {int(part["world_size"]) for part in parts}
        ranks = {int(part["rank"]) for part in parts}
        if len(world_sizes) != 1 or ranks != set(range(next(iter(world_sizes)))):
            raise ValueError(f"Incomplete rank set for {parts_root} {corpus=} {step=}: {world_sizes=}, {ranks=}")
        accuracy: defaultdict[tuple[str, str, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
        effect: defaultdict[tuple[str, str, int], dict[str, float]] = defaultdict(
            lambda: {name: 0.0 for name in EFFECT_FIELDS}
        )
        for part in parts:
            for item in part["accuracy_strata"]:
                key = (str(item["dataset_id"]), str(item["capability_id"]), int(item["capability_level"]))
                accuracy[key][0] += float(item["mase_sum"])
                accuracy[key][1] += int(item["row_count"])
            for item in part["effect_strata"]:
                key = (str(item["dataset_id"]), str(item["capability_id"]), int(item["capability_level"]))
                for name in EFFECT_FIELDS:
                    effect[key][name] += float(item[name])

        strata: dict[tuple[str, str, int], dict[str, float]] = {}
        for key in sorted(set(accuracy) | set(effect)):
            values: dict[str, float] = {}
            if key in accuracy and accuracy[key][1] > 0:
                values["mase"] = accuracy[key][0] / accuracy[key][1]
                values["row_count"] = accuracy[key][1]
            if key in effect and effect[key]["truth_squared_sum"] > 0:
                values["effect_nrmse"] = math.sqrt(
                    effect[key]["squared_error_sum"] / effect[key]["truth_squared_sum"]
                )
                values.update(effect[key])
            strata[key] = values
        strata_by_group[(corpus, step)] = strata
        audit_rows.append(
            {
                "corpus": corpus,
                "step": step,
                "world_size": next(iter(world_sizes)),
                "rank_count": len(parts),
                "mase_stratum_count": sum("mase" in values for values in strata.values()),
                "effect_stratum_count": sum("effect_nrmse" in values for values in strata.values()),
                "treatment_count": int(sum(values[1] for values in accuracy.values())),
                "all_parts_complete": True,
            }
        )
    return strata_by_group, audit_rows


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, sufficient for the short no-/few-tie checkpoint curves."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, *, spearman: bool = False) -> float:
    if spearman:
        left, right = rankdata(left), rankdata(right)
    return float(np.corrcoef(left, right)[0, 1])


def find_pareto(rows: Iterable[dict[str, Any]]) -> set[int]:
    points = list(rows)
    pareto: set[int] = set()
    for row in points:
        mase = float(row[METRICS["mase"]["column"]])
        nrmse = float(row[METRICS["effect_nrmse"]["column"]])
        dominated = any(
            (
                float(other[METRICS["mase"]["column"]]) <= mase
                and float(other[METRICS["effect_nrmse"]["column"]]) <= nrmse
                and (
                    float(other[METRICS["mase"]["column"]]) < mase
                    or float(other[METRICS["effect_nrmse"]["column"]]) < nrmse
                )
            )
            for other in points
        )
        if not dominated:
            pareto.add(int(row["step"]))
    return pareto


def paired_bootstrap(
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    differences = current - baseline
    chunk = 1_000
    samples: list[np.ndarray] = []
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        samples.append(differences[indices].mean(axis=1))
    means = np.concatenate(samples)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def prepare_outputs(root: Path, bootstrap_replicates: int) -> dict[str, Any]:
    all_curves: list[dict[str, Any]] = []
    strata: dict[str, dict[tuple[str, int], dict[tuple[str, str, int], dict[str, float]]]] = {}
    audit_rows: list[dict[str, Any]] = []
    for objective, config in OBJECTIVES.items():
        curves = load_curve(root / str(config["curve"]), objective)
        all_curves.extend(curves)
        objective_strata, objective_audit = aggregate_metric_parts(root / str(config["parts"]))
        strata[objective] = objective_strata
        for row in objective_audit:
            audit_rows.append({"objective_id": objective, **row})

        curve_lookup = {(str(row["corpus"]), int(row["step"])): row for row in curves}
        for group, cells in objective_strata.items():
            curve_row = curve_lookup[group]
            for metric, meta in METRICS.items():
                values = [cell[metric] for cell in cells.values() if metric in cell]
                reconstructed = float(np.mean(values))
                expected = float(curve_row[str(meta["column"])])
                if not math.isclose(reconstructed, expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(
                        f"Metric reconstruction mismatch for {objective} {group} {metric}: "
                        f"{reconstructed} != {expected}"
                    )

    curve_index = {
        (str(row["objective_id"]), str(row["corpus"]), int(row["step"])): row for row in all_curves
    }

    tidy_curves: list[dict[str, Any]] = []
    for row in all_curves:
        objective = str(row["objective_id"])
        corpus = str(row["corpus"])
        step = int(row["step"])
        for metric, meta in METRICS.items():
            value = float(row[str(meta["column"])])
            baseline = float(curve_index[(objective, corpus, 0)][str(meta["column"])])
            tidy_curves.append(
                {
                    "objective_id": objective,
                    "objective_label": OBJECTIVES[objective]["label"],
                    "corpus": corpus,
                    "corpus_label": CORPUS_LABELS[corpus],
                    "step": step,
                    "nominal_epoch_equivalents": step
                    / float(json.loads((root / str(OBJECTIVES[objective]["manifest"])).read_text())["epoch_steps"]),
                    "metric": metric,
                    "metric_label": meta["label"],
                    "value": value,
                    "baseline": baseline,
                    "absolute_change": value - baseline,
                    "relative_change_percent": 100.0 * (value / baseline - 1.0),
                    "gain_percent": 100.0 * (baseline - value) / baseline,
                }
            )
    write_csv(root / "curves_long.csv", tidy_curves)

    checkpoint_rows: list[dict[str, Any]] = []
    wide_checkpoint_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        for corpus in ("train", "cross"):
            rows = sorted(
                (row for row in all_curves if row["objective_id"] == objective and row["corpus"] == corpus),
                key=lambda row: int(row["step"]),
            )
            for metric, meta in METRICS.items():
                column = str(meta["column"])
                baseline_row = rows[0]
                best_all = min(rows, key=lambda row: (float(row[column]), int(row["step"])))
                best_nonzero = min(rows[1:], key=lambda row: (float(row[column]), int(row["step"])))
                final_row = rows[-1]
                baseline = float(baseline_row[column])
                for selection, selected in (
                    ("baseline", baseline_row),
                    ("best_all", best_all),
                    ("best_nonzero", best_nonzero),
                    ("final", final_row),
                ):
                    value = float(selected[column])
                    checkpoint_rows.append(
                        {
                            "objective_id": objective,
                            "objective_label": OBJECTIVES[objective]["label"],
                            "corpus": corpus,
                            "corpus_label": CORPUS_LABELS[corpus],
                            "metric": metric,
                            "metric_label": meta["label"],
                            "selection": selection,
                            "step": int(selected["step"]),
                            "value": value,
                            "absolute_change": value - baseline,
                            "relative_change_percent": 100.0 * (value / baseline - 1.0),
                            "gain_percent": 100.0 * (baseline - value) / baseline,
                        }
                    )
                wide_checkpoint_rows.append(
                    {
                        "objective_id": objective,
                        "objective_label": OBJECTIVES[objective]["label"],
                        "corpus": corpus,
                        "metric": metric,
                        "baseline_value": baseline,
                        "best_all_step": int(best_all["step"]),
                        "best_all_value": float(best_all[column]),
                        "best_all_relative_change_percent": 100.0 * (float(best_all[column]) / baseline - 1.0),
                        "best_nonzero_step": int(best_nonzero["step"]),
                        "best_nonzero_value": float(best_nonzero[column]),
                        "best_nonzero_relative_change_percent": 100.0
                        * (float(best_nonzero[column]) / baseline - 1.0),
                        "final_step": int(final_row["step"]),
                        "final_value": float(final_row[column]),
                        "final_relative_change_percent": 100.0 * (float(final_row[column]) / baseline - 1.0),
                    }
                )
    write_csv(root / "key_checkpoints.csv", checkpoint_rows)
    write_csv(root / "checkpoint_summary_wide.csv", wide_checkpoint_rows)

    seed_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        for metric, meta in METRICS.items():
            column = str(meta["column"])
            by_corpus = {
                corpus: {
                    int(row["step"]): float(row[column])
                    for row in all_curves
                    if row["objective_id"] == objective and row["corpus"] == corpus
                }
                for corpus in ("train", "cross")
            }
            steps = sorted(set(by_corpus["train"]) & set(by_corpus["cross"]))
            nonzero_steps = [step for step in steps if step > 0]
            gains = {
                corpus: np.asarray(
                    [
                        100.0
                        * (by_corpus[corpus][0] - by_corpus[corpus][step])
                        / by_corpus[corpus][0]
                        for step in nonzero_steps
                    ]
                )
                for corpus in ("train", "cross")
            }
            train_best_step = min(steps, key=lambda step: by_corpus["train"][step])
            cross_best_step = min(steps, key=lambda step: by_corpus["cross"][step])
            final_train_gain = 100.0 * (
                by_corpus["train"][0] - by_corpus["train"][40_000]
            ) / by_corpus["train"][0]
            final_cross_gain = 100.0 * (
                by_corpus["cross"][0] - by_corpus["cross"][40_000]
            ) / by_corpus["cross"][0]
            seed_rows.append(
                {
                    "objective_id": objective,
                    "objective_label": OBJECTIVES[objective]["label"],
                    "metric": metric,
                    "metric_label": meta["label"],
                    "trajectory_steps": "4000..40000",
                    "pearson_gain_train_vs_cross": correlation(gains["train"], gains["cross"]),
                    "spearman_gain_train_vs_cross": correlation(
                        gains["train"], gains["cross"], spearman=True
                    ),
                    "train_best_step": train_best_step,
                    "train_best_gain_percent": 100.0
                    * (by_corpus["train"][0] - by_corpus["train"][train_best_step])
                    / by_corpus["train"][0],
                    "cross_best_step": cross_best_step,
                    "cross_best_gain_percent": 100.0
                    * (by_corpus["cross"][0] - by_corpus["cross"][cross_best_step])
                    / by_corpus["cross"][0],
                    "final_train_gain_percent": final_train_gain,
                    "final_cross_gain_percent": final_cross_gain,
                    "final_train_minus_cross_gain_pp": final_train_gain - final_cross_gain,
                }
            )
    write_csv(root / "seed_transfer_summary.csv", seed_rows)

    stratum_change_rows: list[dict[str, Any]] = []
    capability_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260902)
    checkpoint_lookup = {
        (row["objective_id"], row["corpus"], row["metric"], row["selection"]): row
        for row in checkpoint_rows
    }
    for objective in OBJECTIVES:
        for corpus in ("train", "cross"):
            baseline_cells = strata[objective][(corpus, 0)]
            for step in range(0, 40_001, 4_000):
                current_cells = strata[objective][(corpus, step)]
                for metric in METRICS:
                    keys = sorted(
                        key
                        for key in baseline_cells.keys() & current_cells.keys()
                        if metric in baseline_cells[key] and metric in current_cells[key]
                    )
                    for key in keys:
                        baseline = float(baseline_cells[key][metric])
                        current = float(current_cells[key][metric])
                        stratum_change_rows.append(
                            {
                                "objective_id": objective,
                                "corpus": corpus,
                                "step": step,
                                "metric": metric,
                                "dataset_id": key[0],
                                "capability_id": key[1],
                                "capability_level": key[2],
                                "baseline": baseline,
                                "value": current,
                                "absolute_change": current - baseline,
                                "relative_change_percent": 100.0 * (current / baseline - 1.0),
                                "improved": current < baseline,
                            }
                        )

            for metric in METRICS:
                final_cells = strata[objective][(corpus, 40_000)]
                capabilities = sorted({key[1] for key in baseline_cells})
                for capability in capabilities:
                    keys = sorted(
                        key
                        for key in baseline_cells.keys() & final_cells.keys()
                        if key[1] == capability and metric in baseline_cells[key] and metric in final_cells[key]
                    )
                    baseline_values = np.asarray([baseline_cells[key][metric] for key in keys], dtype=float)
                    final_values = np.asarray([final_cells[key][metric] for key in keys], dtype=float)
                    capability_rows.append(
                        {
                            "objective_id": objective,
                            "corpus": corpus,
                            "step": 40_000,
                            "metric": metric,
                            "capability_id": capability,
                            "stratum_count": len(keys),
                            "baseline_macro": float(baseline_values.mean()),
                            "value_macro": float(final_values.mean()),
                            "absolute_change": float(final_values.mean() - baseline_values.mean()),
                            "relative_change_percent": 100.0
                            * float(final_values.mean() / baseline_values.mean() - 1.0),
                            "improved_stratum_fraction": float(np.mean(final_values < baseline_values)),
                        }
                    )

                for selection in ("best_all", "best_nonzero", "final"):
                    selected_step = int(
                        checkpoint_lookup[(objective, corpus, metric, selection)]["step"]
                    )
                    selected_cells = strata[objective][(corpus, selected_step)]
                    keys = sorted(
                        key
                        for key in baseline_cells.keys() & selected_cells.keys()
                        if metric in baseline_cells[key] and metric in selected_cells[key]
                    )
                    baseline_values = np.asarray([baseline_cells[key][metric] for key in keys], dtype=float)
                    current_values = np.asarray([selected_cells[key][metric] for key in keys], dtype=float)
                    low, high = paired_bootstrap(
                        baseline_values,
                        current_values,
                        rng=rng,
                        replicates=bootstrap_replicates,
                    )
                    uncertainty_rows.append(
                        {
                            "objective_id": objective,
                            "corpus": corpus,
                            "metric": metric,
                            "selection": selection,
                            "step": selected_step,
                            "stratum_count": len(keys),
                            "macro_absolute_change": float((current_values - baseline_values).mean()),
                            "paired_stratum_bootstrap_95ci_low": low,
                            "paired_stratum_bootstrap_95ci_high": high,
                            "improved_stratum_fraction": float(np.mean(current_values < baseline_values)),
                            "interval_note": "Descriptive paired bootstrap over dataset-capability-level cells; hierarchy ignored.",
                        }
                    )
    write_csv(root / "stratum_changes.csv", stratum_change_rows)
    write_csv(root / "capability_final_changes.csv", capability_rows)
    write_csv(root / "paired_stratum_bootstrap.csv", uncertainty_rows)
    write_csv(root / "metric_parts_audit.csv", audit_rows)

    pareto_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        for corpus in ("train", "cross"):
            rows = sorted(
                (row for row in all_curves if row["objective_id"] == objective and row["corpus"] == corpus),
                key=lambda row: int(row["step"]),
            )
            pareto_steps = find_pareto(rows)
            base_mase = float(rows[0][METRICS["mase"]["column"]])
            base_nrmse = float(rows[0][METRICS["effect_nrmse"]["column"]])
            for row in rows:
                pareto_rows.append(
                    {
                        "objective_id": objective,
                        "corpus": corpus,
                        "step": int(row["step"]),
                        "mase": float(row[METRICS["mase"]["column"]]),
                        "effect_nrmse": float(row[METRICS["effect_nrmse"]["column"]]),
                        "delta_mase_percent": 100.0
                        * (float(row[METRICS["mase"]["column"]]) / base_mase - 1.0),
                        "delta_effect_nrmse_percent": 100.0
                        * (float(row[METRICS["effect_nrmse"]["column"]]) / base_nrmse - 1.0),
                        "pareto_nondominated_within_objective_corpus": int(row["step"]) in pareto_steps,
                    }
                )
    write_csv(root / "pareto_trajectory.csv", pareto_rows)

    training_loss_rows: list[dict[str, Any]] = []
    training_loss_summary: list[dict[str, Any]] = []
    for objective, config in OBJECTIVES.items():
        state = json.loads((root / str(config["trainer_state"])).read_text(encoding="utf-8"))
        logs = [entry for entry in state["log_history"] if "loss" in entry]
        for entry in logs:
            training_loss_rows.append(
                {
                    "objective_id": objective,
                    "step": int(entry["step"]),
                    "loss": float(entry["loss"]),
                    "learning_rate": float(entry["learning_rate"]),
                    "grad_norm": float(entry["grad_norm"]),
                }
            )
        head = np.asarray([float(entry["loss"]) for entry in logs[:10]])
        tail = np.asarray([float(entry["loss"]) for entry in logs[-10:]])
        training_loss_summary.append(
            {
                "objective_id": objective,
                "logged_points": len(logs),
                "first_logged_step": int(logs[0]["step"]),
                "last_logged_step": int(logs[-1]["step"]),
                "first_10_mean_loss": float(head.mean()),
                "last_10_mean_loss": float(tail.mean()),
                "first_to_last_10_relative_change_percent": 100.0 * float(tail.mean() / head.mean() - 1.0),
            }
        )
    write_csv(root / "training_loss.csv", training_loss_rows)
    write_csv(root / "training_loss_summary.csv", training_loss_summary)

    # A compact machine-readable summary for downstream paper assembly.
    summary = {
        "schema_version": "cafe.paper.finetuning_analysis.v1",
        "metric_direction": "lower_is_better",
        "checkpoint_grid": list(range(0, 40_001, 4_000)),
        "seed_a": 2026082701,
        "seed_b": 2026082702,
        "seed_overlap": {
            "train_rows": 50535,
            "cross_rows": 48365,
            "train_unique_official_instances": 2382,
            "cross_unique_official_instances": 2309,
            "official_instance_intersection": 240,
            "official_instance_jaccard": 0.053920467310716695,
            "sample_id_intersection": 0,
            "official_capability_level_intersection": 5035,
        },
        "training": {},
        "seed_transfer": seed_rows,
    }
    for objective, config in OBJECTIVES.items():
        manifest = json.loads((root / str(config["manifest"])).read_text(encoding="utf-8"))
        summary["training"][objective] = {
            "label": config["label"],
            "manifest": manifest,
            "nominal_epoch_equivalents_at_40000": 40_000 / float(manifest["epoch_steps"]),
            "checkpoints": [row for row in wide_checkpoint_rows if row["objective_id"] == objective],
        }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "curves": all_curves,
        "tidy_curves": tidy_curves,
        "capability_rows": capability_rows,
        "pareto_rows": pareto_rows,
    }


def configure_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    return plt


def save_figure(fig: Any, root: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")


def make_plots(root: Path, data: dict[str, Any]) -> None:
    plt = configure_matplotlib()
    curves = data["curves"]
    colors = {"default": "#2b6cb0", "effect_nrmse": "#c05621"}
    markers = {"default": "o", "effect_nrmse": "s"}
    linestyles = {"train": "--", "cross": "-"}

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.2), sharex=True)
    for row_index, corpus in enumerate(("train", "cross")):
        for col_index, metric in enumerate(("mase", "effect_nrmse")):
            ax = axes[row_index, col_index]
            column = str(METRICS[metric]["column"])
            for objective in OBJECTIVES:
                rows = sorted(
                    (row for row in curves if row["objective_id"] == objective and row["corpus"] == corpus),
                    key=lambda row: int(row["step"]),
                )
                x = np.asarray([int(row["step"]) / 1000 for row in rows])
                y = np.asarray([float(row[column]) for row in rows])
                ax.plot(
                    x,
                    y,
                    color=colors[objective],
                    marker=markers[objective],
                    markersize=3.2,
                    linewidth=1.55,
                    label=str(OBJECTIVES[objective]["short_label"]),
                )
                best_index = int(np.argmin(y))
                ax.scatter(
                    [x[best_index]],
                    [y[best_index]],
                    color=colors[objective],
                    marker="*",
                    s=52,
                    zorder=4,
                    edgecolor="white",
                    linewidth=0.5,
                )
            ax.axhline(
                float(
                    next(
                        row[column]
                        for row in curves
                        if row["objective_id"] == "default"
                        and row["corpus"] == corpus
                        and int(row["step"]) == 0
                    )
                ),
                color="#777777",
                linewidth=0.8,
                linestyle=":",
            )
            ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.8)
            ax.set_title(f"{CORPUS_LABELS[corpus]} · {METRICS[metric]['short_label']}")
            if row_index == 1:
                ax.set_xlabel("Optimizer step (thousands)")
            ax.set_ylabel(str(METRICS[metric]["short_label"]))
            ax.set_xticks([0, 8, 16, 24, 32, 40])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle("Fine-tuning trajectories (★ lowest observed checkpoint)", y=1.055, fontsize=11)
    fig.tight_layout()
    save_figure(fig, root, "finetuning_curves_absolute")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), sharex=True)
    for ax, metric in zip(axes, ("mase", "effect_nrmse")):
        column = str(METRICS[metric]["column"])
        for objective in OBJECTIVES:
            for corpus in ("train", "cross"):
                rows = sorted(
                    (row for row in curves if row["objective_id"] == objective and row["corpus"] == corpus),
                    key=lambda row: int(row["step"]),
                )
                baseline = float(rows[0][column])
                x = np.asarray([int(row["step"]) / 1000 for row in rows])
                y = np.asarray([100.0 * (float(row[column]) / baseline - 1.0) for row in rows])
                ax.plot(
                    x,
                    y,
                    color=colors[objective],
                    linestyle=linestyles[corpus],
                    marker=markers[objective],
                    markersize=3,
                    linewidth=1.4,
                    label=f"{OBJECTIVES[objective]['short_label']} · {corpus}",
                )
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.8)
        ax.set_title(str(METRICS[metric]["label"]))
        ax.set_xlabel("Optimizer step (thousands)")
        ax.set_ylabel("Change from step 0 (%)")
        ax.set_xticks([0, 8, 16, 24, 32, 40])
    handles = [
        plt.Line2D([], [], color=colors[objective], marker=markers[objective], linewidth=1.5, label=str(OBJECTIVES[objective]["short_label"]))
        for objective in OBJECTIVES
    ] + [
        plt.Line2D([], [], color="#444444", linestyle=linestyles[corpus], linewidth=1.5, label=CORPUS_LABELS[corpus])
        for corpus in ("train", "cross")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Objective alignment produces opposite metric movement", y=1.115, fontsize=11)
    fig.tight_layout()
    save_figure(fig, root, "finetuning_relative_change")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
    for ax, corpus in zip(axes, ("train", "cross")):
        for objective in OBJECTIVES:
            rows = sorted(
                (row for row in data["pareto_rows"] if row["objective_id"] == objective and row["corpus"] == corpus),
                key=lambda row: int(row["step"]),
            )
            x = np.asarray([float(row["delta_mase_percent"]) for row in rows])
            y = np.asarray([float(row["delta_effect_nrmse_percent"]) for row in rows])
            ax.plot(
                x,
                y,
                color=colors[objective],
                marker=markers[objective],
                markersize=3.4,
                linewidth=1.35,
                label=str(OBJECTIVES[objective]["short_label"]),
            )
            best_nrmse_step = int(min(rows, key=lambda row: float(row["effect_nrmse"]))["step"])
            annotation_steps = {0, 40_000, best_nrmse_step}
            vertical_offsets = {
                "default": {0: 5, 40_000: 5, best_nrmse_step: -12},
                "effect_nrmse": {0: 5, 40_000: -12, best_nrmse_step: 7},
            }
            for row, x_value, y_value in zip(rows, x, y):
                step = int(row["step"])
                if step in annotation_steps:
                    label = f"best {step // 1000}" if step == best_nrmse_step and step != 0 else str(step // 1000)
                    ax.annotate(
                        label,
                        (x_value, y_value),
                        xytext=(4, vertical_offsets[objective].get(step, 4)),
                        textcoords="offset points",
                        fontsize=7,
                        color=colors[objective],
                    )
        ax.axhline(0, color="#777777", linewidth=0.7)
        ax.axvline(0, color="#777777", linewidth=0.7)
        ax.grid(color="#dddddd", linewidth=0.45, alpha=0.8)
        ax.set_title(CORPUS_LABELS[corpus])
        ax.set_xlabel("Δ MASE from step 0 (%)")
        ax.set_ylabel("Δ effect NRMSE from step 0 (%)")
        ax.text(
            0.02,
            0.03,
            "better on both ↙",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("MASE–effect-NRMSE checkpoint trajectories (labels: k steps)", y=1.11, fontsize=11)
    fig.tight_layout()
    save_figure(fig, root, "finetuning_pareto_trajectory")
    plt.close(fig)

    capability_rows = [
        row for row in data["capability_rows"] if row["corpus"] == "cross" and int(row["step"]) == 40_000
    ]
    capability_order = [
        "trend",
        "multi_seasonal",
        "time_varying_seasonality",
        "regime_switching",
        "predictable_intermittency",
        "common_factor",
        "cross_series_dependence",
        "covariate_impulse_response",
    ]
    column_order = [
        ("default", "mase"),
        ("default", "effect_nrmse"),
        ("effect_nrmse", "mase"),
        ("effect_nrmse", "effect_nrmse"),
    ]
    lookup = {
        (str(row["capability_id"]), str(row["objective_id"]), str(row["metric"])): float(
            row["relative_change_percent"]
        )
        for row in capability_rows
    }
    matrix = np.asarray(
        [[lookup.get((capability, objective, metric), np.nan) for objective, metric in column_order] for capability in capability_order]
    )
    fig, ax = plt.subplots(figsize=(7.15, 3.8))
    from matplotlib.colors import TwoSlopeNorm

    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), aspect="auto")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isfinite(value):
                ax.text(col_index, row_index, f"{value:+.1f}%", ha="center", va="center", fontsize=8)
    ax.set_yticks(range(len(capability_order)), [name.replace("_", " ") for name in capability_order])
    ax.set_xticks(
        range(len(column_order)),
        ["Default\nMASE", "Default\nEffect NRMSE", "Effect objective\nMASE", "Effect objective\nEffect NRMSE"],
    )
    ax.set_title("Seed B capability-wise change at 40k steps (lower is better)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Change from step 0 (%)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, root, "finetuning_capability_heatmap")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    data = prepare_outputs(root, args.bootstrap_replicates)
    if not args.no_plots:
        make_plots(root, data)
    print(root / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
