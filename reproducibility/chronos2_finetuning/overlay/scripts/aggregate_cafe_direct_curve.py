#!/usr/bin/env python3
"""Aggregate compact direct-inference metric parts into a CaFE curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--objective",
        default="default_loss_random_with_replacement",
        help="Training objective label written into every aggregated curve row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.parts.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "chronos2.cafe_direct_metric_part.v1":
            continue
        grouped[(str(payload["corpus"]), int(payload["step"]))].append(payload)
    if not grouped:
        raise ValueError(f"No direct metric parts found below {args.parts}")

    rows: list[dict[str, Any]] = []
    for (corpus, step), parts in sorted(grouped.items()):
        world_sizes = {int(part["world_size"]) for part in parts}
        ranks = {int(part["rank"]) for part in parts}
        if len(world_sizes) != 1 or ranks != set(range(next(iter(world_sizes)))):
            raise ValueError(
                f"Incomplete ranks for {corpus} step {step}: world={world_sizes}, ranks={ranks}"
            )
        accuracy: defaultdict[tuple[str, str, int], list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        effect: defaultdict[tuple[str, str, int], dict[str, float]] = defaultdict(
            lambda: {
                "candidate_count": 0.0,
                "scored_count": 0.0,
                "squared_error_sum": 0.0,
                "truth_squared_sum": 0.0,
                "observed_cell_count": 0.0,
            }
        )
        for part in parts:
            for item in part["accuracy_strata"]:
                key = (
                    str(item["dataset_id"]),
                    str(item["capability_id"]),
                    int(item["capability_level"]),
                )
                accuracy[key][0] += float(item["mase_sum"])
                accuracy[key][1] += int(item["row_count"])
            for item in part["effect_strata"]:
                key = (
                    str(item["dataset_id"]),
                    str(item["capability_id"]),
                    int(item["capability_level"]),
                )
                for name in effect[key]:
                    effect[key][name] += float(item[name])
        mase_values = [values[0] / values[1] for values in accuracy.values()]
        nrmse_values = [
            math.sqrt(values["squared_error_sum"] / values["truth_squared_sum"])
            for values in effect.values()
            if values["truth_squared_sum"] > 0.0
        ]
        treatment_count = int(sum(values[1] for values in accuracy.values()))
        rows.append(
            {
                "objective": args.objective,
                "corpus": corpus,
                "step": step,
                "macro_stratum_mase": float(np.mean(mase_values)),
                "sample_weighted_mase": float(
                    sum(values[0] for values in accuracy.values()) / treatment_count
                ),
                "macro_stratum_effect_nrmse": float(np.mean(nrmse_values)),
                "mase_stratum_count": len(mase_values),
                "effect_stratum_count": len(nrmse_values),
                "treatment_count": treatment_count,
                "effect_candidate_count": int(
                    sum(values["candidate_count"] for values in effect.values())
                ),
                "effect_scored_count": int(
                    sum(values["scored_count"] for values in effect.values())
                ),
                "observed_effect_cell_count": int(
                    sum(values["observed_cell_count"] for values in effect.values())
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"schema_version": "chronos2.cafe_direct_curve.v1", "rows": rows},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with args.output.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
