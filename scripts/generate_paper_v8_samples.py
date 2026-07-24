#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator

import paper_v8_pipeline_common as v8


DEFAULT_OUTPUT_ROOT = (
    v8.REPO_ROOT / "runtime" / "paper_exp" / "v8_test" / "full_pipeline"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate formal Paper v8 deterministic master samples."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument(
        "--secondary-modulus",
        type=int,
        default=4,
        help="Seeds whose stable hash is divisible by this value enter secondary/robustness.",
    )
    return parser.parse_args()


def selected_sensitivity_seeds(
    dataset_id: str,
    seed_indexes: list[int],
    modulus: int,
) -> set[int]:
    return {
        seed
        for seed in seed_indexes
        if v8.stable_seed(dataset_id, seed, "sensitivity") % modulus == 0
    }


def members_for(capability_id: str) -> tuple[int | None, ...]:
    return (
        (0, 1)
        if capability_id in v8.COUNTERFACTUAL_CAPABILITIES
        else (None,)
    )


def iter_clean_samples(
    dataset: v8.DatasetSpec,
    anchors: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    seed_indexes: list[int],
    sensitivity_seeds: set[int],
) -> Iterator[dict[str, Any]]:
    for capability_id in v8.CAPABILITIES:
        capability_calibration = calibration["capabilities"][capability_id]
        for seed_index in seed_indexes:
            anchor = v8.anchor_for_seed(
                anchors,
                dataset_id=dataset.dataset_id,
                capability_id=capability_id,
                seed_index=seed_index,
            )
            for intensity in v8.INTENSITIES:
                for member in members_for(capability_id):
                    yield v8.generate_master_sample(
                        dataset,
                        anchor,
                        capability_calibration,
                        capability_id=capability_id,
                        family_role="primary",
                        intensity=intensity,
                        seed_index=seed_index,
                        counterfactual_member=member,
                    )
            if seed_index not in sensitivity_seeds:
                continue
            for intensity in (3, 5):
                for member in members_for(capability_id):
                    yield v8.generate_master_sample(
                        dataset,
                        anchor,
                        capability_calibration,
                        capability_id=capability_id,
                        family_role="secondary",
                        intensity=intensity,
                        seed_index=seed_index,
                        counterfactual_member=member,
                    )


def main() -> int:
    args = parse_args()
    if args.seed_start < 0 or args.seed_count < 1:
        raise ValueError("seed_start must be non-negative and seed_count positive")
    if args.secondary_modulus < 1:
        raise ValueError("secondary modulus must be positive")
    dataset = v8.resolve_dataset(args.dataset_id)
    dataset_root = args.output_root.resolve() / dataset.dataset_id
    calibration_dir = dataset_root / "01_calibration"
    bundle = v8.read_json(calibration_dir / "calibration_bundle.json")
    anchors = list(v8.iter_jsonl(calibration_dir / "anchors.jsonl"))
    calibration = v8.read_json(
        calibration_dir / "capability_calibration.json"
    )
    if bundle["generator_version"] != v8.GENERATOR_VERSION:
        raise ValueError("calibration bundle generator version mismatch")

    seed_indexes = list(
        range(args.seed_start, args.seed_start + args.seed_count)
    )
    sensitivity_seeds = selected_sensitivity_seeds(
        dataset.dataset_id,
        seed_indexes,
        args.secondary_modulus,
    )
    generation_dir = dataset_root / "02_generation"
    shard_dir = generation_dir / "sample_shards"
    shard_name = (
        f"seed_{args.seed_start:06d}_{args.seed_start + args.seed_count:06d}"
    )
    clean_path = shard_dir / f"{shard_name}.jsonl"
    clean_count = v8.write_jsonl(
        clean_path,
        iter_clean_samples(
            dataset,
            anchors,
            calibration,
            seed_indexes=seed_indexes,
            sensitivity_seeds=sensitivity_seeds,
        ),
    )

    robustness_path = shard_dir / f"{shard_name}__robustness.jsonl"
    robustness_count = v8.write_jsonl(
        robustness_path,
        (
            v8.robustness_sample(row)
            for row in v8.iter_jsonl(clean_path)
            if row["generator_family_role"] == "primary"
            and int(row["intensity"]) in {3, 5}
            and int(row["seed_index"]) in sensitivity_seeds
        ),
    )
    config = {
        "schema_version": "paper_v8_generation_config.v1",
        "dataset_id": dataset.dataset_id,
        "calibration_bundle_sha256": bundle["bundle_content_sha256"],
        "generator_version": v8.GENERATOR_VERSION,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "seed_indexes": seed_indexes,
        "secondary_seed_policy": {
            "stable_hash_modulus": args.secondary_modulus,
            "selected_seed_indexes": sorted(sensitivity_seeds),
            "intensities": [3, 5],
        },
        "robustness_policy": {
            "source": "clean_primary",
            "selected_seed_indexes": sorted(sensitivity_seeds),
            "intensities": [3, 5],
            "history_noise_ratio": v8.ROBUSTNESS_NOISE_RATIO,
            "scoring_future": "clean_latent",
        },
    }
    manifest = {
        "schema_version": "paper_v8_generation_manifest.v1",
        "created_at": v8.utc_now(),
        "config": config,
        "config_sha256": v8.json_sha256(config),
        "files": {
            "clean": {
                **v8.file_record(clean_path),
                "row_count": clean_count,
            },
            "robustness": {
                **v8.file_record(robustness_path),
                "row_count": robustness_count,
            },
        },
    }
    manifest_path = generation_dir / f"manifest__{shard_name}.json"
    v8.write_json(manifest_path, manifest)
    print(
        v8.canonical_json(
            {
                "dataset_id": dataset.dataset_id,
                "clean_sample_count": clean_count,
                "robustness_sample_count": robustness_count,
                "sensitivity_seed_count": len(sensitivity_seeds),
                "manifest": str(manifest_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
