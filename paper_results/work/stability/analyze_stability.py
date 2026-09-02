#!/usr/bin/env python3
"""Reproduce paper-oriented stability summaries and figures from local snapshots."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
RAW_SUITE = HERE / "raw" / "suite_summaries"
RAW_STABILITY = HERE / "raw" / "remote_stability"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
SEEDS = list(range(2026082701, 2026082711))
T_CRITICAL_975_DF9 = 2.2621571627409915

CAPABILITIES = [
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "predictable_intermittency",
    "common_factor",
    "cross_series_dependence",
    "covariate_impulse_response",
]
CAPABILITY_LABELS = {
    "trend": "Trend",
    "multi_seasonal": "Multi-seasonal",
    "time_varying_seasonality": "TV seasonality",
    "regime_switching": "Regime switching",
    "predictable_intermittency": "Intermittency",
    "common_factor": "Common factor",
    "cross_series_dependence": "Cross-series",
    "covariate_impulse_response": "Covariate impulse",
}
CAPABILITY_SHORT = {
    "trend": "Trend",
    "multi_seasonal": "Multi-seas.",
    "time_varying_seasonality": "TV seas.",
    "regime_switching": "Regime",
    "predictable_intermittency": "Intermitt.",
    "common_factor": "Common",
    "cross_series_dependence": "Cross-ser.",
    "covariate_impulse_response": "Cov. impulse",
}
MODEL_SHORT = {
    "Chronos-2": "C2",
    "timesfm2.5": "TF2.5",
    "Timer-3.5": "T3.5",
    "tirex2": "TiRex",
    "moirai2": "M2",
    "toto2.0": "Toto",
}
EXCLUDED_MODELS = {"Timer-4.0"}


def finite(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def describe(values: Iterable[float | None]) -> dict[str, float | int | None]:
    data = finite(values)
    if not data:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "cv": None,
            "median": None,
            "min": None,
            "max": None,
            "range": None,
            "seed_p025": None,
            "seed_p975": None,
            "mean_ci95_low": None,
            "mean_ci95_high": None,
        }
    mean = statistics.fmean(data)
    sd = statistics.stdev(data) if len(data) > 1 else 0.0
    half_width = (
        T_CRITICAL_975_DF9 * sd / math.sqrt(len(data)) if len(data) == 10 else None
    )
    return {
        "n": len(data),
        "mean": mean,
        "sd": sd,
        "cv": sd / abs(mean) if abs(mean) > 1e-12 else None,
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
        "range": max(data) - min(data),
        "seed_p025": float(np.quantile(data, 0.025)),
        "seed_p975": float(np.quantile(data, 0.975)),
        "mean_ci95_low": mean - half_width if half_width is not None else None,
        "mean_ci95_high": mean + half_width if half_width is not None else None,
    }


def rank_low_is_good(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and math.isclose(
            ordered[end][1], ordered[cursor][1], abs_tol=1e-12, rel_tol=0.0
        ):
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for index in range(cursor, end):
            ranks[ordered[index][0]] = average_rank
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    return float(np.corrcoef(left_array, right_array)[0, 1])


def kendall_tau(left: dict[str, float], right: dict[str, float]) -> float:
    models = sorted(left)
    concordant = 0
    discordant = 0
    for first, second in itertools.combinations(models, 2):
        product = (left[first] - left[second]) * (right[first] - right[second])
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
    return (concordant - discordant) / (concordant + discordant)


def kendall_w(rank_by_seed: dict[int, dict[str, float]], models: list[str]) -> float:
    seed_count = len(rank_by_seed)
    model_count = len(models)
    rank_sums = [sum(rank_by_seed[seed][model] for seed in SEEDS) for model in models]
    expected = seed_count * (model_count + 1) / 2
    squared_deviation = sum((rank_sum - expected) ** 2 for rank_sum in rank_sums)
    return 12 * squared_deviation / (
        seed_count**2 * (model_count**3 - model_count)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_input_manifest() -> None:
    rows = []
    for path in sorted((HERE / "raw").glob("**/*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "path": str(path.relative_to(HERE)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    (HERE / "input_manifest.json").write_text(
        json.dumps(
            {"schema_version": "cafe.paper_stability_input_manifest.v1", "files": rows},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def threshold_summary(values: list[float], threshold: float, lower_is_good: bool) -> dict[str, Any]:
    desired = [value < threshold if lower_is_good else value > threshold for value in values]
    count = sum(desired)
    if count == len(values):
        category = "unanimous_desired"
    elif count == 0:
        category = "unanimous_undesired"
    else:
        category = "crosses_threshold"
    return {
        "desired_count": count,
        "desired_rate": count / len(values),
        "direction_category": category,
        "direction_unanimous": category != "crosses_threshold",
    }


def load_inputs() -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str | None, int | None], dict[int, dict[str, Any]]],
    list[str],
    list[dict[str, Any]],
]:
    records: list[dict[str, Any]] = []
    key_sets: list[set[tuple[str, str, str | None, int | None]]] = []
    coverage_rows: list[dict[str, Any]] = []
    task_ids_reference: list[str] | None = None
    for seed in SEEDS:
        path = RAW_SUITE / f"seed{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = [
            row
            for row in payload["rows"]
            if row["model_id"] not in EXCLUDED_MODELS
        ]
        task_ids = list(payload["task_ids"])
        if task_ids_reference is None:
            task_ids_reference = task_ids
        keys = {
            (
                row["metric"],
                row["model_id"],
                row.get("capability_id"),
                row.get("capability_level"),
            )
            for row in rows
        }
        key_sets.append(keys)
        for row in rows:
            records.append({"seed": seed, **row})
        metric_counts = Counter(row["metric"] for row in rows)
        coverage_rows.append(
            {
                "seed": seed,
                "suite_rows": len(rows),
                "effect_cells": metric_counts["capability_effect_nrmse"],
                "ablation_cells": metric_counts["input_ablation_mase_degradation"],
                "official_cells": metric_counts["official_mase"],
                "task_count": len(task_ids),
                "task_ids": "|".join(task_ids),
                "model_count": len({row["model_id"] for row in rows}),
                "capability_count": len(
                    {row["capability_id"] for row in rows if row.get("capability_id")}
                ),
                "levels": "|".join(
                    map(
                        str,
                        sorted(
                            {
                                row["capability_level"]
                                for row in rows
                                if row.get("capability_level") is not None
                            }
                        ),
                    )
                ),
            }
        )
    union = set.union(*key_sets)
    intersection = set.intersection(*key_sets)
    if union != intersection:
        raise RuntimeError(
            f"Suite keys differ across seeds: union={len(union)}, intersection={len(intersection)}"
        )
    if any(row["task_ids"] != coverage_rows[0]["task_ids"] for row in coverage_rows):
        raise RuntimeError("Task lists differ across augmentation seeds")
    write_csv(TABLES / "suite_coverage_audit.csv", coverage_rows)

    by_cell: defaultdict[
        tuple[str, str, str | None, int | None], dict[int, dict[str, Any]]
    ] = defaultdict(dict)
    for row in records:
        key = (
            row["metric"],
            row["model_id"],
            row.get("capability_id"),
            row.get("capability_level"),
        )
        by_cell[key][row["seed"]] = row
    models = sorted({row["model_id"] for row in records})
    return records, by_cell, models, coverage_rows


def analyze() -> dict[str, Any]:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_input_manifest()
    records, by_cell, models, coverage_rows = load_inputs()

    effect_keys = sorted(key for key in by_cell if key[0] == "capability_effect_nrmse")
    ablation_keys = sorted(
        key for key in by_cell if key[0] == "input_ablation_mase_degradation"
    )
    official_keys = sorted(key for key in by_cell if key[0] == "official_mase")

    effect_by_seed_model: dict[tuple[int, str], float] = {}
    ablation_by_seed_model: dict[tuple[int, str], float] = {}
    official_by_seed_model: dict[tuple[int, str], float] = {}
    for seed in SEEDS:
        for model in models:
            effect_by_seed_model[(seed, model)] = statistics.fmean(
                float(by_cell[key][seed]["task_equal_mean"])
                for key in effect_keys
                if key[1] == model
            )
            ablation_by_seed_model[(seed, model)] = statistics.fmean(
                float(by_cell[key][seed]["task_equal_mean"])
                for key in ablation_keys
                if key[1] == model
            )
            official_key = next(key for key in official_keys if key[1] == model)
            official_by_seed_model[(seed, model)] = float(
                by_cell[official_key][seed]["task_equal_mean"]
            )

    rank_by_seed: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        rank_by_seed[seed] = rank_low_is_good(
            {model: effect_by_seed_model[(seed, model)] for model in models}
        )
    model_order = sorted(
        models,
        key=lambda model: statistics.fmean(
            effect_by_seed_model[(seed, model)] for seed in SEEDS
        ),
    )

    effect_cell_rows: list[dict[str, Any]] = []
    for key in effect_keys:
        metric, model, capability, level = key
        values = [float(by_cell[key][seed]["task_equal_mean"]) for seed in SEEDS]
        result = {
            "metric": metric,
            "model_id": model,
            "capability_id": capability,
            "capability_level": level,
            **describe(values),
            **threshold_summary(values, 1.0, True),
        }
        task_ses = [
            (
                float(by_cell[key][seed]["task_bootstrap_95_ci_upper"])
                - float(by_cell[key][seed]["task_bootstrap_95_ci_lower"])
            )
            / (2 * 1.96)
            for seed in SEEDS
        ]
        result["mean_approx_task_bootstrap_se"] = statistics.fmean(task_ses)
        result["seed_sd_to_task_se_ratio"] = (
            float(result["sd"]) / result["mean_approx_task_bootstrap_se"]
            if result["mean_approx_task_bootstrap_se"] > 0
            else None
        )
        result["seed_mean_ci_relation_to_1"] = (
            "below"
            if result["mean_ci95_high"] < 1
            else "above"
            if result["mean_ci95_low"] > 1
            else "includes"
        )
        result["task_count_min"] = min(
            int(by_cell[key][seed]["task_count"]) for seed in SEEDS
        )
        result["task_count_max"] = max(
            int(by_cell[key][seed]["task_count"]) for seed in SEEDS
        )
        result["task_coverage_min"] = min(
            float(by_cell[key][seed]["task_coverage"]) for seed in SEEDS
        )
        result["task_coverage_max"] = max(
            float(by_cell[key][seed]["task_coverage"]) for seed in SEEDS
        )
        result["worst_seed"] = SEEDS[int(np.argmax(values))]
        result["best_seed"] = SEEDS[int(np.argmin(values))]
        effect_cell_rows.append(result)
    write_csv(TABLES / "effect_cell_seed_stability.csv", effect_cell_rows)

    ablation_cell_rows: list[dict[str, Any]] = []
    for key in ablation_keys:
        metric, model, capability, level = key
        values = [float(by_cell[key][seed]["task_equal_mean"]) for seed in SEEDS]
        result = {
            "metric": metric,
            "model_id": model,
            "capability_id": capability,
            "capability_level": level,
            **describe(values),
            **threshold_summary(values, 0.0, False),
        }
        result["seed_mean_ci_relation_to_0"] = (
            "above"
            if result["mean_ci95_low"] > 0
            else "below"
            if result["mean_ci95_high"] < 0
            else "includes"
        )
        ablation_cell_rows.append(result)
    write_csv(TABLES / "ablation_cell_seed_stability.csv", ablation_cell_rows)

    model_rows: list[dict[str, Any]] = []
    seed_model_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for model in model_order:
            seed_model_rows.append(
                {
                    "seed": seed,
                    "model_id": model,
                    "macro_effect_nrmse": effect_by_seed_model[(seed, model)],
                    "macro_input_ablation_degradation": ablation_by_seed_model[(seed, model)],
                    "official_mase": official_by_seed_model[(seed, model)],
                    "effect_rank": rank_by_seed[seed][model],
                }
            )
    write_csv(TABLES / "seed_model_macro_scores.csv", seed_model_rows)

    for model in model_order:
        effect_values = [effect_by_seed_model[(seed, model)] for seed in SEEDS]
        ablation_values = [ablation_by_seed_model[(seed, model)] for seed in SEEDS]
        official_values = [official_by_seed_model[(seed, model)] for seed in SEEDS]
        model_effect_cells = [row for row in effect_cell_rows if row["model_id"] == model]
        model_ablation_cells = [row for row in ablation_cell_rows if row["model_id"] == model]
        ranks = [rank_by_seed[seed][model] for seed in SEEDS]
        model_rows.append(
            {
                "model_id": model,
                **{f"effect_{key}": value for key, value in describe(effect_values).items()},
                "rank_mean": statistics.fmean(ranks),
                "rank_sd": statistics.stdev(ranks),
                "rank_min": min(ranks),
                "rank_max": max(ranks),
                "top1_count": sum(rank == 1 for rank in ranks),
                "effect_desired_rate": sum(
                    int(row["desired_count"]) for row in model_effect_cells
                )
                / (len(model_effect_cells) * len(SEEDS)),
                "effect_unanimous_desired_cells": sum(
                    row["direction_category"] == "unanimous_desired"
                    for row in model_effect_cells
                ),
                "effect_unanimous_undesired_cells": sum(
                    row["direction_category"] == "unanimous_undesired"
                    for row in model_effect_cells
                ),
                "effect_threshold_crossing_cells": sum(
                    row["direction_category"] == "crosses_threshold"
                    for row in model_effect_cells
                ),
                "effect_seed_ci_below_1_cells": sum(
                    row["seed_mean_ci_relation_to_1"] == "below"
                    for row in model_effect_cells
                ),
                **{
                    f"official_{key}": value
                    for key, value in describe(official_values).items()
                },
                **{
                    f"ablation_{key}": value
                    for key, value in describe(ablation_values).items()
                },
                "ablation_desired_rate": sum(
                    int(row["desired_count"]) for row in model_ablation_cells
                )
                / (len(model_ablation_cells) * len(SEEDS)),
                "ablation_unanimous_cells": sum(
                    bool(row["direction_unanimous"]) for row in model_ablation_cells
                ),
            }
        )
    write_csv(TABLES / "model_overall_stability_extended.csv", model_rows)

    model_capability_rows: list[dict[str, Any]] = []
    for model in model_order:
        for capability in CAPABILITIES:
            matching_cells = [
                row
                for row in effect_cell_rows
                if row["model_id"] == model and row["capability_id"] == capability
            ]
            seed_values = []
            raw_values = []
            for seed in SEEDS:
                values = [
                    float(
                        by_cell[
                            ("capability_effect_nrmse", model, capability, level)
                        ][seed]["task_equal_mean"]
                    )
                    for level in range(1, 6)
                ]
                raw_values.extend(values)
                seed_values.append(statistics.fmean(values))
            model_capability_rows.append(
                {
                    "model_id": model,
                    "capability_id": capability,
                    **describe(seed_values),
                    "desired_rate": sum(value < 1 for value in raw_values) / len(raw_values),
                    "unanimous_desired_levels": sum(
                        row["direction_category"] == "unanimous_desired"
                        for row in matching_cells
                    ),
                    "unanimous_undesired_levels": sum(
                        row["direction_category"] == "unanimous_undesired"
                        for row in matching_cells
                    ),
                    "threshold_crossing_levels": sum(
                        row["direction_category"] == "crosses_threshold"
                        for row in matching_cells
                    ),
                    "median_seed_sd_to_task_se_ratio": float(
                        np.median(
                            [row["seed_sd_to_task_se_ratio"] for row in matching_cells]
                        )
                    ),
                    "max_seed_sd_to_task_se_ratio": max(
                        row["seed_sd_to_task_se_ratio"] for row in matching_cells
                    ),
                    "task_count_min": min(row["task_count_min"] for row in matching_cells),
                    "task_count_max": max(row["task_count_max"] for row in matching_cells),
                }
            )
    write_csv(TABLES / "model_capability_stability_extended.csv", model_capability_rows)

    structure_rows = read_csv(RAW_STABILITY / "structure_diversity.csv")
    structure_by_capability = {row["capability_id"]: row for row in structure_rows}
    capability_rows: list[dict[str, Any]] = []
    capability_coverage_rows: list[dict[str, Any]] = []
    for capability in CAPABILITIES:
        matching_cells = [row for row in effect_cell_rows if row["capability_id"] == capability]
        seed_values = []
        all_values = []
        for seed in SEEDS:
            values = [
                float(by_cell[key][seed]["task_equal_mean"])
                for key in effect_keys
                if key[2] == capability
            ]
            all_values.extend(values)
            seed_values.append(statistics.fmean(values))
        structure = structure_by_capability[capability]
        capability_rows.append(
            {
                "capability_id": capability,
                **describe(seed_values),
                "desired_rate": sum(value < 1 for value in all_values) / len(all_values),
                "unanimous_desired_cells": sum(
                    row["direction_category"] == "unanimous_desired"
                    for row in matching_cells
                ),
                "unanimous_undesired_cells": sum(
                    row["direction_category"] == "unanimous_undesired"
                    for row in matching_cells
                ),
                "threshold_crossing_cells": sum(
                    row["direction_category"] == "crosses_threshold"
                    for row in matching_cells
                ),
                "median_seed_sd_to_task_se_ratio": float(
                    np.median([row["seed_sd_to_task_se_ratio"] for row in matching_cells])
                ),
                "p95_seed_sd_to_task_se_ratio": float(
                    np.quantile(
                        [row["seed_sd_to_task_se_ratio"] for row in matching_cells], 0.95
                    )
                ),
                "instance_capability_groups": int(
                    structure["instance_capability_group_count"]
                ),
                "unique_structure_count_mean": float(
                    structure["unique_structure_count_mean"]
                ),
                "all_10_seeds_unique_rate": float(structure["all_10_seeds_unique_rate"]),
            }
        )
        task_counts = sorted({row["task_count_min"] for row in matching_cells})
        task_coverages = sorted({row["task_coverage_min"] for row in matching_cells})
        capability_coverage_rows.append(
            {
                "capability_id": capability,
                "suite_task_count": 10,
                "contributing_task_counts": "|".join(map(str, task_counts)),
                "task_coverage_values": "|".join(f"{value:.3f}" for value in task_coverages),
                "coverage_identical_across_models_levels_seeds": len(task_counts) == 1
                and len(task_coverages) == 1,
                "is_full_10_task_coverage": task_counts == [10],
            }
        )
    write_csv(TABLES / "capability_stability_summary.csv", capability_rows)
    write_csv(TABLES / "capability_task_coverage.csv", capability_coverage_rows)

    level_rows: list[dict[str, Any]] = []
    for capability in CAPABILITIES:
        for level in range(1, 6):
            matching_keys = [
                key for key in effect_keys if key[2] == capability and key[3] == level
            ]
            seed_values = []
            raw_values = []
            for seed in SEEDS:
                values = [
                    float(by_cell[key][seed]["task_equal_mean"]) for key in matching_keys
                ]
                raw_values.extend(values)
                seed_values.append(statistics.fmean(values))
            level_rows.append(
                {
                    "capability_id": capability,
                    "capability_level": level,
                    **describe(seed_values),
                    "desired_rate": sum(value < 1 for value in raw_values) / len(raw_values),
                }
            )
    write_csv(TABLES / "capability_level_stability.csv", level_rows)

    winner_rows: list[dict[str, Any]] = []
    for capability in CAPABILITIES:
        for level in range(1, 6):
            winners: list[str] = []
            margins: list[float] = []
            for seed in SEEDS:
                scores = {
                    model: float(
                        by_cell[
                            ("capability_effect_nrmse", model, capability, level)
                        ][seed]["task_equal_mean"]
                    )
                    for model in models
                }
                ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]))
                winners.append(ordered[0][0])
                margins.append(ordered[1][1] - ordered[0][1])
            counts = Counter(winners)
            modal_winner, modal_count = sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[0]
            winner_rows.append(
                {
                    "capability_id": capability,
                    "capability_level": level,
                    "modal_winner": modal_winner,
                    "winner_consistency": modal_count / len(SEEDS),
                    "modal_winner_count": modal_count,
                    "unique_winner_count": len(counts),
                    "winner_counts": "|".join(
                        f"{model}:{counts[model]}" for model in sorted(counts)
                    ),
                    "winner_sequence_by_seed": "|".join(winners),
                    "unanimous_winner": modal_count == len(SEEDS),
                    **{
                        f"winner_margin_{key}": value
                        for key, value in describe(margins).items()
                    },
                }
            )
    write_csv(TABLES / "capability_level_winner_consistency.csv", winner_rows)

    rank_similarity_rows: list[dict[str, Any]] = []
    for left_seed, right_seed in itertools.combinations(SEEDS, 2):
        left_ranks = rank_by_seed[left_seed]
        right_ranks = rank_by_seed[right_seed]
        rank_similarity_rows.append(
            {
                "seed_a": left_seed,
                "seed_b": right_seed,
                "spearman": pearson(
                    [left_ranks[model] for model in models],
                    [right_ranks[model] for model in models],
                ),
                "kendall_tau": kendall_tau(left_ranks, right_ranks),
            }
        )
    write_csv(TABLES / "seed_rank_similarity.csv", rank_similarity_rows)

    spearman_matrix_rows: list[dict[str, Any]] = []
    for left_seed in SEEDS:
        row: dict[str, Any] = {"seed": left_seed}
        for right_seed in SEEDS:
            row[f"seed_{str(right_seed)[-2:]}"] = pearson(
                [rank_by_seed[left_seed][model] for model in models],
                [rank_by_seed[right_seed][model] for model in models],
            )
        spearman_matrix_rows.append(row)
    write_csv(TABLES / "seed_rank_spearman_matrix.csv", spearman_matrix_rows)

    pairwise_rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(model_order, 2):
        differences = [
            effect_by_seed_model[(seed, second)]
            - effect_by_seed_model[(seed, first)]
            for seed in SEEDS
        ]
        gap_stats = describe(differences)
        pairwise_rows.append(
            {
                "lower_mean_model": first,
                "higher_mean_model": second,
                "lower_model_better_seed_count": sum(value > 0 for value in differences),
                "higher_model_better_seed_count": sum(value < 0 for value in differences),
                "tie_seed_count": sum(value == 0 for value in differences),
                "mean_gap_higher_minus_lower": gap_stats["mean"],
                "paired_gap_sd": gap_stats["sd"],
                "paired_seed_sd_to_mean_gap": (
                    gap_stats["sd"] / abs(gap_stats["mean"])
                    if abs(gap_stats["mean"]) > 1e-12
                    else None
                ),
                "gap_mean_ci95_low": gap_stats["mean_ci95_low"],
                "gap_mean_ci95_high": gap_stats["mean_ci95_high"],
                "mean_order_ci_excludes_zero": gap_stats["mean_ci95_low"] > 0,
            }
        )
    write_csv(TABLES / "model_pairwise_gap_vs_seed_noise.csv", pairwise_rows)
    adjacent_rows = [
        row
        for row in pairwise_rows
        if model_order.index(row["higher_mean_model"])
        - model_order.index(row["lower_mean_model"])
        == 1
    ]
    write_csv(TABLES / "adjacent_model_gap_vs_seed_noise.csv", adjacent_rows)

    ablation_model_capability_rows: list[dict[str, Any]] = []
    for model in model_order:
        for capability in [
            "common_factor",
            "cross_series_dependence",
            "covariate_impulse_response",
        ]:
            matching_cells = [
                row
                for row in ablation_cell_rows
                if row["model_id"] == model and row["capability_id"] == capability
            ]
            seed_values = [
                statistics.fmean(
                    float(
                        by_cell[
                            (
                                "input_ablation_mase_degradation",
                                model,
                                capability,
                                level,
                            )
                        ][seed]["task_equal_mean"]
                    )
                    for level in range(1, 6)
                )
                for seed in SEEDS
            ]
            ablation_model_capability_rows.append(
                {
                    "model_id": model,
                    "capability_id": capability,
                    **describe(seed_values),
                    "desired_rate": sum(
                        int(row["desired_count"]) for row in matching_cells
                    )
                    / (len(matching_cells) * len(SEEDS)),
                    "unanimous_cell_rate": sum(
                        row["direction_unanimous"] for row in matching_cells
                    )
                    / len(matching_cells),
                }
            )
    write_csv(
        TABLES / "ablation_model_capability_stability.csv",
        ablation_model_capability_rows,
    )

    ratio_values = [row["seed_sd_to_task_se_ratio"] for row in effect_cell_rows]
    overall_direction_counts = Counter(
        row["direction_category"] for row in effect_cell_rows
    )
    spearman_values = [row["spearman"] for row in rank_similarity_rows]
    kendall_values = [row["kendall_tau"] for row in rank_similarity_rows]
    structure_group_total = sum(
        int(row["instance_capability_group_count"]) for row in structure_rows
    )
    structure_all_unique_weighted = sum(
        int(row["instance_capability_group_count"])
        * float(row["all_10_seeds_unique_rate"])
        for row in structure_rows
    ) / structure_group_total

    summary: dict[str, Any] = {
        "scope": {
            "seed_count": len(SEEDS),
            "task_count": len(coverage_rows[0]["task_ids"].split("|")),
            "model_count": len(models),
            "capability_count": len(CAPABILITIES),
            "level_count": 5,
            "suite_cells_per_seed": coverage_rows[0]["suite_rows"],
            "effect_cells_per_seed": coverage_rows[0]["effect_cells"],
            "ablation_cells_per_seed": coverage_rows[0]["ablation_cells"],
            "all_suite_keys_identical": True,
            "all_task_lists_identical": True,
        },
        "ranking": {
            "kendall_w": kendall_w(rank_by_seed, models),
            "pairwise_seed_spearman": describe(spearman_values),
            "pairwise_seed_kendall_tau": describe(kendall_values),
            "top1_frequency": dict(
                Counter(
                    min(rank_by_seed[seed], key=rank_by_seed[seed].get)
                    for seed in SEEDS
                )
            ),
        },
        "winner_consistency": {
            "cell_count": len(winner_rows),
            "unanimous_winner_cell_count": sum(
                row["unanimous_winner"] for row in winner_rows
            ),
            "unanimous_winner_cell_rate": sum(
                row["unanimous_winner"] for row in winner_rows
            )
            / len(winner_rows),
            "mean_modal_winner_rate": statistics.fmean(
                row["winner_consistency"] for row in winner_rows
            ),
            "min_modal_winner_rate": min(
                row["winner_consistency"] for row in winner_rows
            ),
        },
        "effect_direction": {
            "cell_count": len(effect_cell_rows),
            **overall_direction_counts,
            "unanimous_any_direction_rate": (
                overall_direction_counts["unanimous_desired"]
                + overall_direction_counts["unanimous_undesired"]
            )
            / len(effect_cell_rows),
            "all_seed_observation_desired_rate": sum(
                row["desired_count"] for row in effect_cell_rows
            )
            / (len(effect_cell_rows) * len(SEEDS)),
            "seed_mean_ci_below_1_cell_count": sum(
                row["seed_mean_ci_relation_to_1"] == "below"
                for row in effect_cell_rows
            ),
        },
        "seed_vs_task_uncertainty": {
            "median_ratio": float(np.median(ratio_values)),
            "p95_ratio": float(np.quantile(ratio_values, 0.95)),
            "max_ratio": max(ratio_values),
            "cell_count_ratio_gt_1": sum(value > 1 for value in ratio_values),
            "cell_rate_ratio_gt_1": sum(value > 1 for value in ratio_values)
            / len(ratio_values),
        },
        "structure_diversity": {
            "instance_capability_group_count": structure_group_total,
            "weighted_all_10_seeds_unique_rate": structure_all_unique_weighted,
        },
        "models": model_rows,
        "capabilities": capability_rows,
        "adjacent_model_gaps": adjacent_rows,
    }
    (HERE / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    make_figures(
        model_order=model_order,
        model_rows=model_rows,
        seed_model_rows=seed_model_rows,
        model_capability_rows=model_capability_rows,
        effect_cell_rows=effect_cell_rows,
        rank_by_seed=rank_by_seed,
        winner_rows=winner_rows,
        capability_rows=capability_rows,
        ablation_model_capability_rows=ablation_model_capability_rows,
    )
    write_markdown_tables(summary, model_rows, capability_rows, adjacent_rows)
    return summary


def setup_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def save_figure(figure: Any, stem: str) -> None:
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")


def annotate_heatmap(axis: Any, matrix: np.ndarray, formatter: Any, threshold: float) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value > threshold else "black"
            axis.text(
                column,
                row,
                formatter(value),
                ha="center",
                va="center",
                fontsize=6,
                color=color,
            )


def make_figures(
    *,
    model_order: list[str],
    model_rows: list[dict[str, Any]],
    seed_model_rows: list[dict[str, Any]],
    model_capability_rows: list[dict[str, Any]],
    effect_cell_rows: list[dict[str, Any]],
    rank_by_seed: dict[int, dict[str, float]],
    winner_rows: list[dict[str, Any]],
    capability_rows: list[dict[str, Any]],
    ablation_model_capability_rows: list[dict[str, Any]],
) -> None:
    plt = setup_matplotlib()

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.35]})
    positions = np.arange(len(model_order))
    rng = np.random.default_rng(20260831)
    for index, model in enumerate(model_order):
        values = [
            row["macro_effect_nrmse"]
            for row in seed_model_rows
            if row["model_id"] == model
        ]
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        axes[0].scatter(
            positions[index] + jitter,
            values,
            color="#6b7280",
            alpha=0.55,
            s=11,
            linewidths=0,
        )
        model_row = next(row for row in model_rows if row["model_id"] == model)
        mean = model_row["effect_mean"]
        lower = mean - model_row["effect_mean_ci95_low"]
        upper = model_row["effect_mean_ci95_high"] - mean
        axes[0].errorbar(
            positions[index],
            mean,
            yerr=np.array([[lower], [upper]]),
            fmt="o",
            color="#0f4c5c",
            capsize=2.5,
            markersize=4,
            zorder=3,
        )
    axes[0].set_xticks(positions, model_order, rotation=45, ha="right")
    axes[0].set_ylabel("Macro effect NRMSE (lower is better)")
    axes[0].set_title("(a) Ten augmentation seeds")
    axes[0].grid(axis="y", alpha=0.2)

    rank_matrix = np.asarray(
        [[rank_by_seed[seed][model] for model in model_order] for seed in SEEDS]
    )
    image = axes[1].imshow(rank_matrix, cmap="viridis_r", vmin=1, vmax=len(model_order), aspect="auto")
    annotate_heatmap(axes[1], rank_matrix, lambda value: f"{value:.0f}", threshold=4.2)
    axes[1].set_xticks(np.arange(len(model_order)), model_order, rotation=45, ha="right")
    axes[1].set_yticks(np.arange(len(SEEDS)), [f"S{index:02d}" for index in range(1, 11)])
    axes[1].set_title("(b) Rank by seed")
    colorbar = figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03)
    colorbar.set_label("Rank")
    figure.tight_layout()
    save_figure(figure, "fig_stability_overall")
    plt.close(figure)

    mean_matrix = np.asarray(
        [
            [
                next(
                    row["mean"]
                    for row in model_capability_rows
                    if row["model_id"] == model and row["capability_id"] == capability
                )
                for capability in CAPABILITIES
            ]
            for model in model_order
        ]
    )
    cv_matrix = np.asarray(
        [
            [
                100
                * next(
                    row["cv"]
                    for row in model_capability_rows
                    if row["model_id"] == model and row["capability_id"] == capability
                )
                for capability in CAPABILITIES
            ]
            for model in model_order
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    image_mean = axes[0].imshow(mean_matrix, cmap="RdYlGn_r", vmin=0, vmax=1.15, aspect="auto")
    annotate_heatmap(axes[0], mean_matrix, lambda value: f"{value:.2f}", threshold=0.72)
    image_cv = axes[1].imshow(cv_matrix, cmap="magma", vmin=0, vmax=30, aspect="auto")
    annotate_heatmap(axes[1], cv_matrix, lambda value: f"{value:.1f}", threshold=16)
    for axis in axes:
        axis.set_xticks(
            np.arange(len(CAPABILITIES)),
            [CAPABILITY_SHORT[capability] for capability in CAPABILITIES],
            rotation=45,
            ha="right",
        )
        axis.set_yticks(np.arange(len(model_order)), model_order)
    axes[0].set_title("(a) Mean effect NRMSE")
    axes[1].set_title("(b) Across-seed CV (%)")
    figure.colorbar(image_mean, ax=axes[0], fraction=0.035, pad=0.02)
    figure.colorbar(image_cv, ax=axes[1], fraction=0.035, pad=0.02)
    save_figure(figure, "fig_capability_mean_cv_heatmap")
    plt.close(figure)

    winner_matrix = np.asarray(
        [
            [
                next(
                    row["winner_consistency"]
                    for row in winner_rows
                    if row["capability_id"] == capability
                    and row["capability_level"] == level
                )
                for level in range(1, 6)
            ]
            for capability in CAPABILITIES
        ]
    )
    figure, axis = plt.subplots(figsize=(4.8, 3.7))
    image = axis.imshow(winner_matrix, cmap="Blues", vmin=0.3, vmax=1, aspect="auto")
    for row_index, capability in enumerate(CAPABILITIES):
        for column_index, level in enumerate(range(1, 6)):
            row = next(
                item
                for item in winner_rows
                if item["capability_id"] == capability
                and item["capability_level"] == level
            )
            axis.text(
                column_index,
                row_index,
                f"{MODEL_SHORT[row['modal_winner']]}\n{row['modal_winner_count']}/10",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if row["winner_consistency"] >= 0.75 else "black",
            )
    axis.set_xticks(np.arange(5), [f"L{level}" for level in range(1, 6)])
    axis.set_yticks(
        np.arange(len(CAPABILITIES)),
        [CAPABILITY_LABELS[capability] for capability in CAPABILITIES],
    )
    axis.set_xlabel("Capability level")
    axis.set_title("Modal winner and consistency across augmentation seeds")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label("Modal winner frequency")
    figure.tight_layout()
    save_figure(figure, "fig_capability_level_winner_consistency")
    plt.close(figure)

    desired_matrix = np.asarray(
        [
            [
                next(
                    row["desired_rate"]
                    for row in model_capability_rows
                    if row["model_id"] == model and row["capability_id"] == capability
                )
                for capability in CAPABILITIES
            ]
            for model in model_order
        ]
    )
    ratio_by_capability = [
        [
            row["seed_sd_to_task_se_ratio"]
            for row in effect_cell_rows
            if row["capability_id"] == capability
        ]
        for capability in CAPABILITIES
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    image = axes[0].imshow(desired_matrix, cmap="YlGn", vmin=0.4, vmax=1, aspect="auto")
    annotate_heatmap(axes[0], desired_matrix, lambda value: f"{100*value:.0f}", threshold=0.78)
    axes[0].set_xticks(
        np.arange(len(CAPABILITIES)),
        [CAPABILITY_SHORT[capability] for capability in CAPABILITIES],
        rotation=45,
        ha="right",
    )
    axes[0].set_yticks(np.arange(len(model_order)), model_order)
    axes[0].set_title("(a) NRMSE < 1 observations (%)")
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.035, pad=0.02)
    colorbar.set_label("Fraction")

    axes[1].boxplot(
        ratio_by_capability,
        tick_labels=[CAPABILITY_SHORT[capability] for capability in CAPABILITIES],
        showfliers=True,
        flierprops={"markersize": 2},
        medianprops={"color": "#b91c1c"},
    )
    axes[1].axhline(1, color="#b91c1c", linestyle="--", linewidth=0.8)
    axes[1].set_xticklabels(
        [CAPABILITY_SHORT[capability] for capability in CAPABILITIES],
        rotation=45,
        ha="right",
    )
    axes[1].set_ylabel("Seed SD / task-bootstrap SE")
    axes[1].set_title("(b) Seed variation relative to task uncertainty")
    axes[1].grid(axis="y", alpha=0.2)
    save_figure(figure, "fig_direction_and_uncertainty")
    plt.close(figure)

    structure_means = [row["unique_structure_count_mean"] for row in capability_rows]
    all_unique_rates = [100 * row["all_10_seeds_unique_rate"] for row in capability_rows]
    positions = np.arange(len(CAPABILITIES))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    axes[0].barh(positions, structure_means, color="#2a9d8f")
    axes[0].set_xlim(0, 10.5)
    axes[0].set_xlabel("Mean unique structures (max. 10)")
    axes[0].set_yticks(
        positions, [CAPABILITY_LABELS[capability] for capability in CAPABILITIES]
    )
    axes[0].invert_yaxis()
    axes[0].set_title("(a) Per instance × capability")
    axes[1].barh(positions, all_unique_rates, color="#457b9d")
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel("Groups with 10 unique structures (%)")
    axes[1].set_yticks(positions, [])
    axes[1].invert_yaxis()
    axes[1].set_title("(b) Full ten-seed uniqueness")
    save_figure(figure, "fig_structure_diversity")
    plt.close(figure)

    ablation_capabilities = [
        "common_factor",
        "cross_series_dependence",
        "covariate_impulse_response",
    ]
    ablation_matrix = np.asarray(
        [
            [
                next(
                    row["desired_rate"]
                    for row in ablation_model_capability_rows
                    if row["model_id"] == model and row["capability_id"] == capability
                )
                for capability in ablation_capabilities
            ]
            for model in model_order
        ]
    )
    figure, axis = plt.subplots(figsize=(3.8, 3.2))
    image = axis.imshow(ablation_matrix, cmap="PuBuGn", vmin=0, vmax=1, aspect="auto")
    annotate_heatmap(axis, ablation_matrix, lambda value: f"{100*value:.0f}", threshold=0.6)
    axis.set_xticks(
        np.arange(len(ablation_capabilities)),
        [CAPABILITY_SHORT[capability] for capability in ablation_capabilities],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(np.arange(len(model_order)), model_order)
    axis.set_title("Legacy input-ablation positive direction (%)")
    figure.colorbar(image, ax=axis, fraction=0.05, pad=0.03)
    figure.tight_layout()
    save_figure(figure, "fig_legacy_ablation_direction")
    plt.close(figure)


def markdown_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def write_markdown_tables(
    summary: dict[str, Any],
    model_rows: list[dict[str, Any]],
    capability_rows: list[dict[str, Any]],
    adjacent_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Stability paper tables (generated)",
        "",
        "## Macro effect NRMSE across augmentation seeds",
        "",
        "| Model | Mean ± seed SD | Seed 95% empirical interval | 95% CI of seed mean | Mean rank (range) | Top-1 | NRMSE<1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            "| {model} | {mean:.3f} ± {sd:.3f} | [{p025:.3f}, {p975:.3f}] | "
            "[{ci_low:.3f}, {ci_high:.3f}] | {rank:.1f} ({rank_min:.0f}–{rank_max:.0f}) | "
            "{top1}/10 | {desired:.1%} |".format(
                model=row["model_id"],
                mean=row["effect_mean"],
                sd=row["effect_sd"],
                p025=row["effect_seed_p025"],
                p975=row["effect_seed_p975"],
                ci_low=row["effect_mean_ci95_low"],
                ci_high=row["effect_mean_ci95_high"],
                rank=row["rank_mean"],
                rank_min=row["rank_min"],
                rank_max=row["rank_max"],
                top1=row["top1_count"],
                desired=row["effect_desired_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Capability-level stability (equal model × level cells)",
            "",
            "| Capability | Mean ± seed SD | CV | NRMSE<1 | Crossing cells | Seed SD / task SE (median) | 10-unique structures |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in capability_rows:
        lines.append(
            "| {capability} | {mean:.3f} ± {sd:.3f} | {cv:.1%} | {desired:.1%} | "
            "{crossing}/35 | {ratio:.3f} | {unique:.1%} |".format(
                capability=CAPABILITY_LABELS[row["capability_id"]],
                mean=row["mean"],
                sd=row["sd"],
                cv=row["cv"],
                desired=row["desired_rate"],
                crossing=row["threshold_crossing_cells"],
                ratio=row["median_seed_sd_to_task_se_ratio"],
                unique=row["all_10_seeds_unique_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Adjacent model gaps versus paired seed variation",
            "",
            "| Better mean | Next model | Mean gap | Paired seed SD | SD / gap | Better seeds | Gap 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in adjacent_rows:
        lines.append(
            "| {better} | {worse} | {gap:.4f} | {sd:.4f} | {ratio:.2f} | {count}/10 | "
            "[{low:.4f}, {high:.4f}] |".format(
                better=row["lower_mean_model"],
                worse=row["higher_mean_model"],
                gap=row["mean_gap_higher_minus_lower"],
                sd=row["paired_gap_sd"],
                ratio=row["paired_seed_sd_to_mean_gap"],
                count=row["lower_model_better_seed_count"],
                low=row["gap_mean_ci95_low"],
                high=row["gap_mean_ci95_high"],
            )
        )
    lines.extend(
        [
            "",
            "## Rank agreement and cell winners",
            "",
            f"- Kendall's W: {summary['ranking']['kendall_w']:.3f}.",
            f"- Pairwise seed Spearman: mean {summary['ranking']['pairwise_seed_spearman']['mean']:.3f}, minimum {summary['ranking']['pairwise_seed_spearman']['min']:.3f}.",
            f"- Pairwise seed Kendall tau: mean {summary['ranking']['pairwise_seed_kendall_tau']['mean']:.3f}, minimum {summary['ranking']['pairwise_seed_kendall_tau']['min']:.3f}.",
            f"- Unanimous capability × level winner: {summary['winner_consistency']['unanimous_winner_cell_count']}/40 ({summary['winner_consistency']['unanimous_winner_cell_rate']:.1%}).",
            f"- Mean modal-winner consistency: {summary['winner_consistency']['mean_modal_winner_rate']:.1%}.",
            "",
        ]
    )
    (HERE / "paper_tables.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = analyze()
    print(json.dumps({
        "ranking": summary["ranking"],
        "winner_consistency": summary["winner_consistency"],
        "effect_direction": summary["effect_direction"],
        "seed_vs_task_uncertainty": summary["seed_vs_task_uncertainty"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
