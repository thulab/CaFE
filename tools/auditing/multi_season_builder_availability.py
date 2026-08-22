from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cafe.benchmark_extension.gift_eval import iter_gift_eval_instances
from cafe.benchmark_extension.mechanisms import (
    MULTI_SEASONAL_COMPONENT_VISIBILITY,
    MULTI_SEASONAL_HARMONIC_RELATIVE_TOLERANCE,
    MULTI_SEASONAL_MAXIMUM_ADDITIONAL_PERIODS,
    MULTI_SEASONAL_MINIMUM_FREQUENCY_SEPARATION_CYCLES,
    MULTI_SEASONAL_MINIMUM_FUTURE_CYCLE_FRACTION,
    MULTI_SEASONAL_MINIMUM_HISTORY_CYCLES,
    MULTI_SEASONAL_MINIMUM_PERIOD,
    MULTI_SEASONAL_PERIOD_CANDIDATE_COUNT,
    MULTI_SEASONAL_REAL_ANCHOR_CANDIDATE_COUNT,
    MULTI_SEASONAL_SHARED_DISTANCE_INTERVAL,
    MULTI_SEASONAL_SPLIT_AMPLITUDE_RATIO_MINIMUM,
    MULTI_SEASONAL_SPLIT_PHASE_COSINE_MINIMUM,
    build_capability_group,
)
from cafe.core import DATASET_REGISTRY


def _percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _local_dataset_ids(gift_eval_dir: Path) -> list[str]:
    return [
        dataset_id
        for dataset_id, spec in DATASET_REGISTRY.items()
        if (gift_eval_dir / spec.asset_name).is_dir()
    ]


def audit_dataset(
    dataset_id: str,
    gift_eval_dir: Path,
    *,
    term: str,
    augmentation_seed: int,
    max_instances: int | None,
) -> dict[str, Any]:
    started = time.monotonic()
    instance_count = 0
    available_count = 0
    reasons: Counter[str] = Counter()
    anchor_sources: Counter[str] = Counter()
    history_anchor_ranks: Counter[str] = Counter()
    anchor_fallback_reasons: Counter[str] = Counter()
    history_anchor_rejection_reasons: Counter[str] = Counter()
    items: dict[str, dict[str, int]] = defaultdict(
        lambda: {"available": 0, "total": 0}
    )
    shared_distances: list[float] = []
    minimum_context_distances: list[float] = []

    for instance in iter_gift_eval_instances(
        dataset_id,
        gift_eval_dir,
        term=term,
        max_instances=max_instances,
    ):
        instance_count += 1
        group = build_capability_group(
            instance,
            "multi_seasonal",
            augmentation_seed=augmentation_seed,
        )
        item = items[instance.item_id]
        item["total"] += 1
        if not group.available:
            reasons[str(group.reason)] += 1
            continue
        available_count += 1
        item["available"] += 1
        first_metadata = group.treatments[0].metadata
        for details in first_metadata["resolved_periods_by_target"].values():
            anchor_sources[str(details["anchor_source"])] += 1
            search = details["history_anchor_search"]
            accepted_rank = search.get("accepted_rank")
            if accepted_rank is not None:
                history_anchor_ranks[str(int(accepted_rank))] += 1
            fallback_reason = search.get("fallback_reason")
            if fallback_reason is not None:
                anchor_fallback_reasons[str(fallback_reason)] += 1
            for attempt in search.get("attempts", []):
                history_anchor_rejection_reasons.update(
                    str(reason)
                    for reason in attempt.get("rejection_reasons", [])
                )
        shared_distances.append(
            float(
                group.group_metadata[
                    "shared_full_history_macro_normalized_rms"
                ]
            )
        )
        minimum_context_distances.append(
            min(
                float(
                    treatment.source_distance_gate[
                        "minimum_observed_macro_distance"
                    ]
                )
                for treatment in group.treatments
            )
        )

    record_count = len(items)
    all_windows = sum(
        int(item["available"] == item["total"] and item["total"] > 0)
        for item in items.values()
    )
    any_window = sum(int(item["available"] > 0) for item in items.values())
    return {
        "dataset_id": dataset_id,
        "instance_count": instance_count,
        "available_instance_count": available_count,
        "available_instance_percent": _percentage(
            available_count, instance_count
        ),
        "unavailable_reason_counts": dict(reasons.most_common()),
        "accepted_anchor_source_counts": dict(anchor_sources.most_common()),
        "accepted_history_anchor_rank_counts": dict(
            history_anchor_ranks.most_common()
        ),
        "anchor_fallback_reason_counts": dict(
            anchor_fallback_reasons.most_common()
        ),
        "history_anchor_candidate_rejection_reason_counts": dict(
            history_anchor_rejection_reasons.most_common()
        ),
        "record_count": record_count,
        "available_record_count_all_windows": all_windows,
        "available_record_percent_all_windows": _percentage(
            all_windows, record_count
        ),
        "available_record_count_any_window": any_window,
        "available_record_percent_any_window": _percentage(
            any_window, record_count
        ),
        "accepted_shared_distance_range": (
            [min(shared_distances), max(shared_distances)]
            if shared_distances
            else None
        ),
        "accepted_minimum_context_distance_range": (
            [min(minimum_context_distances), max(minimum_context_distances)]
            if minimum_context_distances
            else None
        ),
        "elapsed_seconds": time.monotonic() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit final count-controlled multi-seasonal builder availability."
        )
    )
    parser.add_argument("--gift-eval-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", action="append", default=[])
    parser.add_argument(
        "--term", choices=("short", "medium", "long"), default="short"
    )
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gift_eval_dir = args.gift_eval_dir.resolve()
    dataset_ids = args.dataset_id or _local_dataset_ids(gift_eval_dir)
    datasets: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        row = audit_dataset(
            dataset_id,
            gift_eval_dir,
            term=args.term,
            augmentation_seed=args.augmentation_seed,
            max_instances=args.max_instances,
        )
        datasets.append(row)
        print(
            f"{dataset_id}: {row['available_instance_count']}/"
            f"{row['instance_count']} "
            f"({row['available_instance_percent']:.2f}%) "
            f"elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )

    total = sum(row["instance_count"] for row in datasets)
    available = sum(row["available_instance_count"] for row in datasets)
    payload = {
        "schema_version": "cafe.multi_season_builder_availability.v1",
        "gift_eval_dir": str(gift_eval_dir),
        "term": args.term,
        "augmentation_seed": args.augmentation_seed,
        "max_instances_per_dataset": args.max_instances,
        "dataset_ids": dataset_ids,
        "method": {
            "level_coordinate": "additional_independent_period_count",
            "levels": list(
                range(1, MULTI_SEASONAL_MAXIMUM_ADDITIONAL_PERIODS + 1)
            ),
            "total_controlled_period_counts": list(
                range(2, MULTI_SEASONAL_MAXIMUM_ADDITIONAL_PERIODS + 2)
            ),
            "anchor_policy": (
                "first_stable_visible_supported_history_top3_harmonic_else_"
                "protocol_generated"
            ),
            "maximum_history_anchor_candidate_count": (
                MULTI_SEASONAL_REAL_ANCHOR_CANDIDATE_COUNT
            ),
            "history_anchor_component_visibility_threshold": (
                MULTI_SEASONAL_COMPONENT_VISIBILITY
            ),
            "history_anchor_split_half_phase_cosine_minimum": (
                MULTI_SEASONAL_SPLIT_PHASE_COSINE_MINIMUM
            ),
            "history_anchor_split_half_amplitude_ratio_minimum": (
                MULTI_SEASONAL_SPLIT_AMPLITUDE_RATIO_MINIMUM
            ),
            "history_anchor_split_half_visibility_threshold": (
                MULTI_SEASONAL_COMPONENT_VISIBILITY
            ),
            "additional_period_source": "protocol_generated",
            "period_candidate_count": MULTI_SEASONAL_PERIOD_CANDIDATE_COUNT,
            "minimum_period": MULTI_SEASONAL_MINIMUM_PERIOD,
            "minimum_shortest_context_cycles": (
                MULTI_SEASONAL_MINIMUM_HISTORY_CYCLES
            ),
            "minimum_future_cycle_fraction": (
                MULTI_SEASONAL_MINIMUM_FUTURE_CYCLE_FRACTION
            ),
            "minimum_frequency_separation_cycles_in_shortest_context": (
                MULTI_SEASONAL_MINIMUM_FREQUENCY_SEPARATION_CYCLES
            ),
            "harmonic_relative_tolerance": (
                MULTI_SEASONAL_HARMONIC_RELATIVE_TOLERANCE
            ),
            "pre_aggregate_component_energy_policy": (
                "each_component_has_one_target_history_scale_rms"
            ),
            "shared_full_history_macro_rms_interval": list(
                MULTI_SEASONAL_SHARED_DISTANCE_INTERVAL
            ),
            "final_source_distance_gate_included": True,
        },
        "datasets": datasets,
        "summary": {
            "instance_count": total,
            "available_instance_count": available,
            "available_instance_percent": _percentage(available, total),
            "macro_average_available_instance_percent": sum(
                row["available_instance_percent"] for row in datasets
            )
            / len(datasets),
            "dataset_count_at_least_50_percent_available": sum(
                int(row["available_instance_percent"] >= 50.0)
                for row in datasets
            ),
            "dataset_count_at_least_20_percent_available": sum(
                int(row["available_instance_percent"] >= 20.0)
                for row in datasets
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
