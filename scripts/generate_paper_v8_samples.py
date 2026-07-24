#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Iterator

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import paper_v8_pipeline_common as v8


DEFAULT_OUTPUT_ROOT = v8.REPO_ROOT / "runtime" / "paper_exp" / "v8"
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate formal Paper v8 deterministic master samples."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Independent capability generation processes. Use 1 for the "
            "serial reference implementation."
        ),
    )
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=v8.CAPABILITIES,
        default=list(v8.CAPABILITIES),
    )
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
        if capability_id in v8.MAIN_COUNTERFACTUAL_CAPABILITIES
        else (None,)
    )


def iter_clean_samples(
    dataset: v8.DatasetSpec,
    anchors: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    capability_ids: tuple[str, ...],
    seed_indexes: list[int],
    sensitivity_seeds: set[int],
) -> Iterator[dict[str, Any]]:
    for capability_id in capability_ids:
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
            if (
                capability_id in v8.STRICT_COUNTERFACTUAL_CAPABILITIES
                and seed_index in sensitivity_seeds
            ):
                for member in (0, 1):
                    yield v8.generate_master_sample(
                        dataset,
                        anchor,
                        capability_calibration,
                        capability_id=capability_id,
                        family_role="primary",
                        intensity=5,
                        seed_index=seed_index,
                        counterfactual_member=member,
                        evaluation_table="strict_counterfactual_audit",
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


def iter_input_ablations(
    clean_rows: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in clean_rows:
        if (
            row["capability_id"] not in v8.INPUT_ABLATION_CAPABILITIES
            or row["generator_family_role"] != "primary"
            or row.get("evaluation_table", "main") != "main"
        ):
            continue
        key = (str(row["capability_id"]), int(row["intensity"]))
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["seed_index"]))
        if len(rows) < 2:
            continue
        for index, clean in enumerate(rows):
            donor = rows[(index + 1) % len(rows)]
            yield v8.multivariate_input_ablation_sample(clean, donor)


def generate_capability_shard(
    dataset: v8.DatasetSpec,
    anchors: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    capability_id: str,
    seed_indexes: list[int],
    sensitivity_seeds: set[int],
    output_path: Path,
) -> tuple[str, int]:
    count = v8.write_jsonl(
        output_path,
        iter_clean_samples(
            dataset,
            anchors,
            calibration,
            capability_ids=(capability_id,),
            seed_indexes=seed_indexes,
            sensitivity_seeds=sensitivity_seeds,
        ),
    )
    return capability_id, count


def merge_jsonl_shards(
    output_path: Path,
    shard_paths: Iterable[Path],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as output:
        for shard_path in shard_paths:
            with shard_path.open("rb") as source:
                shutil.copyfileobj(source, output)
    os.replace(temporary, output_path)


def generate_clean_samples(
    dataset: v8.DatasetSpec,
    anchors: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    capability_ids: tuple[str, ...],
    seed_indexes: list[int],
    sensitivity_seeds: set[int],
    output_path: Path,
    workers: int,
) -> int:
    if workers == 1 or len(capability_ids) == 1:
        return v8.write_jsonl(
            output_path,
            iter_clean_samples(
                dataset,
                anchors,
                calibration,
                capability_ids=capability_ids,
                seed_indexes=seed_indexes,
                sensitivity_seeds=sensitivity_seeds,
            ),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    maximum_workers = min(workers, len(capability_ids))
    with tempfile.TemporaryDirectory(
        prefix=".v8_capability_shards_",
        dir=output_path.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        shard_paths = {
            capability_id: temporary_root / f"{capability_id}.jsonl"
            for capability_id in capability_ids
        }
        with ProcessPoolExecutor(max_workers=maximum_workers) as executor:
            future_capabilities = {
                executor.submit(
                    generate_capability_shard,
                    dataset,
                    anchors,
                    calibration,
                    capability_id=capability_id,
                    seed_indexes=seed_indexes,
                    sensitivity_seeds=sensitivity_seeds,
                    output_path=shard_paths[capability_id],
                ): capability_id
                for capability_id in capability_ids
            }
            for future in as_completed(future_capabilities):
                capability_id, count = future.result()
                counts[capability_id] = count
                print(
                    v8.canonical_json(
                        {
                            "dataset_id": dataset.dataset_id,
                            "generated_capability": capability_id,
                            "sample_count": count,
                        }
                    ),
                    flush=True,
                )
        merge_jsonl_shards(
            output_path,
            (shard_paths[capability_id] for capability_id in capability_ids),
        )
    return sum(counts.values())


def main() -> int:
    args = parse_args()
    if args.seed_start < 0 or args.seed_count < 1:
        raise ValueError("seed_start must be non-negative and seed_count positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
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
    capability_ids = tuple(args.capabilities)
    clean_count = generate_clean_samples(
        dataset,
        anchors,
        calibration,
        capability_ids=capability_ids,
        seed_indexes=seed_indexes,
        sensitivity_seeds=sensitivity_seeds,
        output_path=clean_path,
        workers=args.workers,
    )

    ablation_path = shard_dir / f"{shard_name}__input_ablation.jsonl"
    ablation_count = v8.write_jsonl(
        ablation_path,
        iter_input_ablations(v8.iter_jsonl(clean_path)),
    )

    robustness_path = shard_dir / f"{shard_name}__robustness.jsonl"
    robustness_count = v8.write_jsonl(
        robustness_path,
        (
            v8.robustness_sample(row)
            for row in v8.iter_jsonl(clean_path)
            if row["generator_family_role"] == "primary"
            and row.get("evaluation_table", "main") == "main"
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
        "capabilities": list(args.capabilities),
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
        "input_ablation_policy": {
            "capabilities": sorted(v8.INPUT_ABLATION_CAPABILITIES),
            "source": "clean_primary_main",
            "donor_policy": "next_seed_same_capability_and_intensity",
            "marginal_matching": "affine_mean_and_std",
            "scoring_future": "original_clean_latent",
        },
        "strict_counterfactual_policy": {
            "capabilities": sorted(v8.STRICT_COUNTERFACTUAL_CAPABILITIES),
            "selected_seed_indexes": sorted(sensitivity_seeds),
            "intensities": [5],
            "evaluation_table": "strict_counterfactual_audit",
        },
    }
    manifest = {
        "schema_version": "paper_v8_generation_manifest.v1",
        "created_at": v8.utc_now(),
        "execution": {
            "capability_workers": min(args.workers, len(capability_ids)),
            "blas_threads_per_process": 1,
        },
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
            "input_ablations": {
                **v8.file_record(ablation_path),
                "row_count": ablation_count,
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
                "input_ablation_sample_count": ablation_count,
                "sensitivity_seed_count": len(sensitivity_seeds),
                "manifest": str(manifest_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
