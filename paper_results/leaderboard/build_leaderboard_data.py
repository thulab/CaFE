#!/usr/bin/env python3
"""Build the self-contained public leaderboard data bundle."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper_results" / "work" / "main_experiments" / "tables"
OUTPUT = Path(__file__).resolve().parent / "leaderboard-data.js"

MODELS = ["Chronos-2", "timesfm2.5", "tirex2", "moirai2", "Timer-3.5", "toto2.0"]
SUITES = ["GIFT-Short", "GIFT-Medium", "GIFT-Long", "FEV-Mini20"]
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
    "time_varying_seasonality": "Time-varying seasonality",
    "regime_switching": "Regime switching",
    "predictable_intermittency": "Predictable intermittency",
    "common_factor": "Common factor",
    "cross_series_dependence": "Cross-series dependence",
    "covariate_impulse_response": "Covariate response",
}


def read_rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | float | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return round(parsed, 10) if math.isfinite(parsed) else None


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return round(fmean(finite), 10) if finite else None


def suite_metric_values() -> dict[str, dict[str, dict[str, float]]]:
    values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in read_rows("official_mase_suite.csv"):
        value = number(row["official_mase_task_equal_mean"])
        if value is not None:
            values[row["suite"]][row["model_id"]]["reference_mase"] = value

    sources = (
        (
            "treatment_mase_by_suite_model_capability_level.csv",
            "treatment_mase_task_equal",
            "probe_mase",
        ),
        (
            "effect_nrmse_by_suite_model_capability_level.csv",
            "effect_nrmse_task_equal",
            "paired_nrmse",
        ),
    )
    for filename, source_column, metric in sources:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in read_rows(filename):
            value = number(row[source_column])
            if value is not None:
                grouped[(row["suite"], row["model_id"])].append(value)
        for (suite, model), observations in grouped.items():
            aggregate = mean(observations)
            if aggregate is not None:
                values[suite][model][metric] = aggregate
    return values


def overall_tables(
    suite_values: dict[str, dict[str, dict[str, float]]],
) -> dict[str, list[dict[str, object]]]:
    metrics = ("reference_mase", "probe_mase", "paired_nrmse")
    tables: dict[str, list[dict[str, object]]] = {}
    for suite_key, selected_suites in [(suite, [suite]) for suite in SUITES] + [
        ("all", SUITES)
    ]:
        rows = []
        for model in MODELS:
            model_values: dict[str, float | None] = {}
            coverage: dict[str, int] = {}
            for metric in metrics:
                observations = [
                    suite_values[suite][model][metric]
                    for suite in selected_suites
                    if metric in suite_values[suite][model]
                ]
                model_values[metric] = mean(observations)
                coverage[metric] = len(observations)
            if any(value is not None for value in model_values.values()):
                rows.append(
                    {
                        "model": model,
                        "values": model_values,
                        "coverage": coverage,
                        "suiteCount": len(selected_suites),
                    }
                )
        tables[suite_key] = rows
    return tables


def capability_source(
    filename: str,
    value_column: str,
) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in read_rows(filename):
        value = number(row[value_column])
        if value is not None:
            grouped[(row["suite"], row["model_id"], row["capability_id"])].append(
                value
            )
    return {
        key: aggregate
        for key, observations in grouped.items()
        if (aggregate := mean(observations)) is not None
    }


def capability_tables() -> dict[str, dict[str, list[dict[str, object]]]]:
    metric_sources = {
        "probe_mase": capability_source(
            "treatment_mase_by_suite_model_capability_level.csv",
            "treatment_mase_task_equal",
        ),
        "paired_nrmse": capability_source(
            "effect_nrmse_by_suite_model_capability_level.csv",
            "effect_nrmse_task_equal",
        ),
    }
    tables: dict[str, dict[str, list[dict[str, object]]]] = {}
    for suite_key, selected_suites in [(suite, [suite]) for suite in SUITES] + [
        ("all", SUITES)
    ]:
        tables[suite_key] = {}
        for metric, source in metric_sources.items():
            rows = []
            for model in MODELS:
                scores: dict[str, float | None] = {}
                coverage: dict[str, int] = {}
                for capability in CAPABILITIES:
                    observations = [
                        source[(suite, model, capability)]
                        for suite in selected_suites
                        if (suite, model, capability) in source
                    ]
                    scores[capability] = mean(observations)
                    coverage[capability] = len(observations)
                available = [value for value in scores.values() if value is not None]
                if available:
                    rows.append(
                        {
                            "model": model,
                            "scores": scores,
                            "coverage": coverage,
                            "average": mean(available),
                            "suiteCount": len(selected_suites),
                        }
                    )
            tables[suite_key][metric] = rows
    return tables


def main() -> None:
    suite_values = suite_metric_values()
    payload = {
        "schemaVersion": "cafe.public_leaderboard.v1",
        "lowerIsBetter": True,
        "suiteAggregation": "equal weight over available benchmark suites",
        "capabilityAggregation": "equal weight over levels within suite, then equal weight over available suites",
        "models": MODELS,
        "suites": [
            {"id": "all", "label": "All benchmarks"},
            *({"id": suite, "label": suite} for suite in SUITES),
        ],
        "capabilities": [
            {"id": capability, "label": CAPABILITY_LABELS[capability]}
            for capability in CAPABILITIES
        ],
        "metrics": {
            "reference_mase": {
                "label": "Reference MASE",
                "shortLabel": "Reference MASE",
                "description": "Forecasting accuracy on the authentic benchmark futures.",
            },
            "probe_mase": {
                "label": "Diagnostic-probe MASE",
                "shortLabel": "Probe MASE",
                "description": "Forecasting accuracy on the treated futures.",
            },
            "paired_nrmse": {
                "label": "Paired forecast-change NRMSE",
                "shortLabel": "Paired NRMSE",
                "description": "Normalized error in the forecast change induced by each treatment.",
            },
        },
        "overall": overall_tables(suite_values),
        "capability": capability_tables(),
    }
    OUTPUT.write_text(
        "window.CAFE_LEADERBOARD_DATA = "
        + json.dumps(payload, indent=2, sort_keys=False)
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
