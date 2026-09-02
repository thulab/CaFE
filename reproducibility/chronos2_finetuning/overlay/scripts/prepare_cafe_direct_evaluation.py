#!/usr/bin/env python3
"""Prepare compact baseline inputs and effect metadata for direct CaFE evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import cafe_seed_transfer as adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cafe-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gift-eval-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--fold-salt", required=True)
    parser.add_argument("--maximum-context", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _baseline_features() -> Any:
    import datasets

    return datasets.Features(
        {
            "context": datasets.List(datasets.List(datasets.Value("float32"))),
            "future_covariates": datasets.List(datasets.List(datasets.Value("float32"))),
            "n_targets": datasets.Value("int64"),
            "n_covariates": datasets.Value("int64"),
            "n_future_covariates": datasets.Value("int64"),
            "horizon": datasets.Value("int64"),
            "dataset_id": datasets.Value("string"),
            "official_instance_id": datasets.Value("string"),
            "sample_id": datasets.Value("string"),
            "future_observed_mask": datasets.List(
                datasets.List(datasets.Value("bool"))
            ),
            "mase_scale_by_target": datasets.List(datasets.Value("float64")),
        }
    )


def _prepared_baseline(row: dict[str, Any]) -> dict[str, Any]:
    target = np.asarray(row["target"], dtype=np.float32)
    horizon = int(row["horizon"])
    covariates, n_future_covariates = adapter._ordered_covariates(row)
    context = np.concatenate((target.T, covariates.T), axis=0)
    future = np.full((context.shape[0], horizon), np.nan, dtype=np.float32)
    if n_future_covariates:
        future[-n_future_covariates:] = covariates[-horizon:, -n_future_covariates:].T
    return {
        "context": context.tolist(),
        "future_covariates": future.tolist(),
        "n_targets": int(target.shape[1]),
        "n_covariates": int(covariates.shape[1]),
        "n_future_covariates": n_future_covariates,
        "horizon": horizon,
        "dataset_id": str(row["dataset_id"]),
        "official_instance_id": str(row["official_instance_id"]),
        "sample_id": str(row["sample_id"]),
        "future_observed_mask": np.asarray(
            row["future_observed_mask"], dtype=bool
        ).tolist(),
        "mase_scale_by_target": np.asarray(
            row["mase_scale_by_target"], dtype=np.float64
        ).tolist(),
    }


def main() -> None:
    args = parse_args()
    adapter._configure_cafe_import(args.cafe_root)
    if args.output.exists():
        raise FileExistsError(args.output)

    import datasets
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cafe.benchmark_extension.generation import (
        _replay_contract_instance,
        iter_replay_contract_work_items,
    )

    requested = set(args.datasets)
    roots = [
        root
        for root in adapter._dataset_roots(args.experiment_root)
        if root.name in requested
    ]
    missing = requested - {root.name for root in roots}
    if missing:
        raise ValueError(f"Missing datasets: {sorted(missing)}")

    effect_rows: list[dict[str, Any]] = []

    def generate_baselines() -> Iterator[dict[str, Any]]:
        for root in roots:
            manifest = adapter._load_local_manifest(root)
            if adapter._manifest_horizon(manifest) != args.horizon:
                continue
            for instance, baseline, treatments, _ablations in iter_replay_contract_work_items(
                manifest, gift_eval_dir=args.gift_eval_dir
            ):
                official_id = str(baseline["official_instance_id"])
                if not adapter._fold_matches(
                    official_id,
                    fold_count=args.fold_count,
                    heldout_fold=args.fold_index,
                    role="eval",
                    fold_salt=args.fold_salt,
                ):
                    continue
                selected = [
                    row
                    for row in treatments
                    if str(row["capability_id"]) in adapter.CAPABILITIES
                ]
                if not selected:
                    continue
                history_start = max(
                    0, int(instance.context_length) - args.maximum_context
                )
                dense = _replay_contract_instance(
                    instance,
                    baseline,
                    [],
                    [],
                    history_start=history_start,
                )[0]
                yield _prepared_baseline(dense)
                for treatment in selected:
                    effect_rows.append(
                        {
                            "sample_id": str(treatment["sample_id"]),
                            "baseline_sample_id": str(treatment["baseline_sample_id"]),
                            "official_instance_id": official_id,
                            "dataset_id": str(treatment["dataset_id"]),
                            "capability_id": str(treatment["capability_id"]),
                            "capability_level": int(treatment["capability_level"]),
                            "affected_target_indices": [
                                int(value)
                                for value in treatment["affected_target_indices"]
                            ],
                        }
                    )

    baseline_path = args.output / "baselines"
    baseline_dataset = datasets.Dataset.from_generator(
        generate_baselines,
        features=_baseline_features(),
        writer_batch_size=128,
    )
    baseline_dataset.save_to_disk(str(baseline_path))
    effect_schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("baseline_sample_id", pa.string()),
            ("official_instance_id", pa.string()),
            ("dataset_id", pa.string()),
            ("capability_id", pa.string()),
            ("capability_level", pa.int16()),
            ("affected_target_indices", pa.list_(pa.int16())),
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(effect_rows, schema=effect_schema),
        args.output / "effect_metadata.parquet",
        compression="zstd",
    )
    manifest = {
        "schema_version": "chronos2.cafe_direct_evaluation_data.v1",
        "experiment_root": str(args.experiment_root.resolve()),
        "horizon": args.horizon,
        "fold_count": args.fold_count,
        "fold_index": args.fold_index,
        "fold_salt": args.fold_salt,
        "datasets": sorted(requested),
        "baseline_count": len(baseline_dataset),
        "treatment_count": len(effect_rows),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
