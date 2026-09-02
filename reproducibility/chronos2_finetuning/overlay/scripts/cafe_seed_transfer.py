#!/usr/bin/env python3
"""Fine-tune Chronos-2 on one CaFE augmentation seed and test another.

The adapter deliberately consumes CaFE's compact replay contracts instead of
copying dense generated curves.  It supports deterministic, balanced sampling
by (dataset, capability, level) and an optional official-instance holdout fold.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("cafe_seed_transfer")

CAPABILITIES = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "predictable_intermittency",
    "common_factor",
    "cross_series_dependence",
    "covariate_impulse_response",
)

CHECKPOINT_PROGRESS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _packed_epoch_steps(
    group_size_counts: Mapping[int, int], *, batch_size: int, seed: int
) -> int:
    """Count batches in one deterministic no-replacement pass."""

    sizes = np.concatenate(
        [
            np.full(int(count), int(group_size), dtype=np.uint16)
            for group_size, count in sorted(group_size_counts.items())
            if count
        ]
    )
    if sizes.size == 0:
        raise ValueError("Cannot construct an epoch from an empty dataset")
    return _packed_shuffled_sizes_steps(sizes, batch_size=batch_size, seed=seed)


def _packed_shuffled_sizes_steps(
    sizes: np.ndarray, *, batch_size: int, seed: int
) -> int:
    if sizes.size == 0:
        raise ValueError("Cannot construct an epoch from an empty dataset")
    sizes = sizes.copy()
    np.random.default_rng(seed).shuffle(sizes)
    steps = 0
    current = 0
    for size in sizes:
        current += int(size)
        if current >= batch_size:
            steps += 1
            current = 0
    return steps + int(current > 0)


def _dataset_epoch_steps(dataset: Any, *, batch_size: int, seed: int) -> int:
    sizes = np.empty(len(dataset), dtype=np.uint16)
    offset = 0
    metadata = dataset.select_columns(("n_targets", "n_covariates"))
    for batch in metadata.iter(batch_size=65_536):
        count = len(batch["n_targets"])
        sizes[offset : offset + count] = np.asarray(batch["n_targets"], dtype=np.uint16) + np.asarray(
            batch["n_covariates"], dtype=np.uint16
        )
        offset += count
    if offset != len(dataset):
        raise RuntimeError(f"Read {offset} group sizes for a dataset with {len(dataset)} rows")
    return _packed_shuffled_sizes_steps(sizes, batch_size=batch_size, seed=seed)


def _checkpoint_steps(total_steps: int, interval: int | None = None) -> list[int]:
    if interval is not None:
        return sorted({*range(interval, total_steps + 1, interval), total_steps})
    return sorted({max(1, round(total_steps * progress)) for progress in CHECKPOINT_PROGRESS})


def _hash_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _configure_cafe_import(cafe_root: Path) -> None:
    source = (cafe_root / "src").resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"CaFE source directory does not exist: {source}")
    text = str(source)
    if text not in sys.path:
        sys.path.insert(0, text)


def _dataset_roots(experiment_root: Path) -> list[Path]:
    roots = sorted(
        path
        for path in experiment_root.resolve().iterdir()
        if path.is_dir() and path.name.startswith("gift_")
    )
    if not roots:
        raise ValueError(f"No gift_* dataset directories found in {experiment_root}")
    return roots


def _load_local_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "01_generation" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: dict[str, dict[str, Any]] = {}
    for name, raw_record in (manifest.get("files") or {}).items():
        record = dict(raw_record)
        local = manifest_path.parent / Path(str(record["path"])).name
        if not local.is_file():
            raise FileNotFoundError(local)
        record["path"] = str(local.resolve())
        files[str(name)] = record
    if not files:
        raise ValueError(f"Generation manifest has no replay files: {manifest_path}")
    return {**manifest, "files": files}


def _manifest_horizon(manifest: Mapping[str, Any]) -> int:
    configured = manifest.get("config", {}).get("prediction_length")
    if configured is not None:
        return int(configured)

    import pyarrow.parquet as pq

    official_path = Path(manifest["files"]["official_baselines"]["path"])
    first = pq.ParquetFile(official_path).read_row_group(0, columns=("payload_json",)).slice(0, 1)
    payload = json.loads(first.column("payload_json")[0].as_py())
    return int(payload["horizon"])


def _fold_matches(
    official_instance_id: str,
    *,
    fold_count: int,
    heldout_fold: int,
    role: str,
    fold_salt: str = "",
) -> bool:
    if fold_count == 1 or role == "all":
        return True
    hash_key = official_instance_id if not fold_salt else f"{fold_salt}:{official_instance_id}"
    bucket = _hash_int(hash_key) % fold_count
    return bucket != heldout_fold if role == "train" else bucket == heldout_fold


def _selected_sample_ids(
    treatment_path: Path,
    *,
    capabilities: frozenset[str],
    capability_levels: frozenset[int],
    max_per_stratum: int | None,
    selection_seed: int,
    fold_count: int,
    heldout_fold: int,
    role: str,
    fold_salt: str = "",
) -> tuple[set[str] | None, Counter[tuple[str, int]]]:
    """Select deterministic minimum-hash samples within capability/level strata."""

    import pyarrow.parquet as pq

    counts: Counter[tuple[str, int]] = Counter()
    heaps: defaultdict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    parquet = pq.ParquetFile(treatment_path)
    for batch in parquet.iter_batches(
        batch_size=65_536,
        columns=("sample_id", "official_instance_id", "capability_id", "capability_level"),
    ):
        sample_ids, official_ids, capability_ids, levels = (
            column.to_pylist() for column in batch.columns
        )
        for sample_id, official_id, capability_id, level in zip(
            sample_ids, official_ids, capability_ids, levels, strict=True
        ):
            capability = str(capability_id)
            numeric_level = int(level)
            if (
                capability not in capabilities
                or numeric_level not in capability_levels
                or not _fold_matches(
                    str(official_id),
                    fold_count=fold_count,
                    heldout_fold=heldout_fold,
                    role=role,
                    fold_salt=fold_salt,
                )
            ):
                continue
            key = (capability, numeric_level)
            counts[key] += 1
            if max_per_stratum is None:
                continue
            # sample_id embeds augmentation_seed. Ranking by the stable source
            # treatment key keeps capped selections paired across seed batches.
            rank = _hash_int(
                f"{selection_seed}:{official_id}:{capability}:{numeric_level}"
            )
            heap = heaps[key]
            candidate = (-rank, str(sample_id))
            if len(heap) < max_per_stratum:
                heapq.heappush(heap, candidate)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, candidate)

    if max_per_stratum is None:
        return None, counts
    selected = {sample_id for heap in heaps.values() for _rank, sample_id in heap}
    return selected, counts


def _selection_for_dataset(
    dataset_root: Path,
    args: argparse.Namespace,
) -> tuple[set[str] | None, Counter[tuple[str, int]]]:
    return _selected_sample_ids(
        dataset_root / "01_generation" / "treatment_contracts.parquet",
        capabilities=frozenset(args.capabilities),
        capability_levels=frozenset(args.capability_levels),
        max_per_stratum=args.max_per_stratum,
        selection_seed=args.selection_seed,
        fold_count=args.official_fold_count,
        heldout_fold=args.heldout_fold,
        role=args.fold_role,
        fold_salt=args.fold_salt,
    )


def _iter_selected_dense_treatments(
    *,
    experiment_root: Path,
    gift_eval_dir: Path,
    capabilities: frozenset[str],
    selections: Mapping[str, set[str] | None],
    fold_count: int,
    heldout_fold: int,
    role: str,
    fold_salt: str,
    maximum_context: int,
    horizon: int | None = None,
) -> Iterator[dict[str, Any]]:
    from cafe.benchmark_extension.generation import (
        _replay_contract_instance,
        iter_replay_contract_work_items,
    )

    for dataset_root in _dataset_roots(experiment_root):
        if dataset_root.name not in selections:
            continue
        manifest = _load_local_manifest(dataset_root)
        manifest_horizon = _manifest_horizon(manifest)
        if horizon is not None and manifest_horizon != horizon:
            continue
        selected = selections[dataset_root.name]
        LOGGER.info("Replaying selected treatments from %s", dataset_root.name)
        for instance, baseline, treatments, _ablations in iter_replay_contract_work_items(
            manifest, gift_eval_dir=gift_eval_dir
        ):
            official_id = str(baseline["official_instance_id"])
            if not _fold_matches(
                official_id,
                fold_count=fold_count,
                heldout_fold=heldout_fold,
                role=role,
                fold_salt=fold_salt,
            ):
                continue
            filtered = [
                row
                for row in treatments
                if str(row["capability_id"]) in capabilities
                and (selected is None or str(row["sample_id"]) in selected)
            ]
            if not filtered:
                continue
            history_start = max(0, int(instance.context_length) - maximum_context)
            dense_rows = _replay_contract_instance(
                instance,
                baseline,
                filtered,
                [],
                history_start=history_start,
            )
            yield from dense_rows[1:]


def _ordered_covariates(row: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    target = np.asarray(row["target"], dtype=np.float32)
    raw = row.get("covariates")
    if raw is None:
        return np.empty((target.shape[0], 0), dtype=np.float32), 0
    covariates = np.asarray(raw, dtype=np.float32)
    visible = np.asarray(row.get("future_covariate_visible") or [], dtype=bool)
    if covariates.ndim != 2 or visible.shape != (covariates.shape[1],):
        raise ValueError(f"Invalid covariate schema for {row['sample_id']}")
    past_only = np.flatnonzero(~visible)
    known_future = np.flatnonzero(visible)
    order = np.concatenate((past_only, known_future))
    return covariates[:, order], int(known_future.size)


def _prepared_training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    target = np.asarray(row["target"], dtype=np.float32)
    horizon = int(row["horizon"])
    covariates, n_future_covariates = _ordered_covariates(row)
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
        "capability_id": str(row["capability_id"]),
        "capability_level": int(row["capability_level"]),
        "augmentation_seed": int(row["augmentation_seed"]),
        "future_observed_mask": np.asarray(
            row["future_observed_mask"], dtype=bool
        ).tolist(),
        "mase_scale_by_target": np.asarray(
            row["mase_scale_by_target"], dtype=np.float64
        ).tolist(),
    }


def _prepared_evaluation_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import torch

    target = np.asarray(row["target"], dtype=np.float32)
    horizon = int(row["horizon"])
    origin = target.shape[0] - horizon
    covariates, n_future_covariates = _ordered_covariates(row)
    context = np.concatenate((target[:origin].T, covariates[:origin].T), axis=0)
    future = np.full((context.shape[0], horizon), np.nan, dtype=np.float32)
    if n_future_covariates:
        future[-n_future_covariates:] = covariates[origin:, -n_future_covariates:].T
    prepared = {
        "context": torch.from_numpy(context),
        "future_covariates": torch.from_numpy(future),
        "n_targets": int(target.shape[1]),
        "n_covariates": int(covariates.shape[1]),
        "n_future_covariates": n_future_covariates,
    }
    truth = target[origin:]
    observed = np.asarray(row["future_observed_mask"], dtype=bool)
    return prepared, truth, observed


def _materialized_evaluation_row(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import torch

    context = torch.as_tensor(row["context"], dtype=torch.float32)
    horizon = int(row["horizon"])
    n_targets = int(row["n_targets"])
    origin = context.shape[-1] - horizon
    if origin <= 0:
        raise ValueError(f"Context is shorter than its horizon for {row['sample_id']}")
    prepared = {
        "context": context[:, :origin],
        "future_covariates": torch.as_tensor(row["future_covariates"], dtype=torch.float32),
        "n_targets": n_targets,
        "n_covariates": int(row["n_covariates"]),
        "n_future_covariates": int(row["n_future_covariates"]),
    }
    truth = context[:n_targets, origin:].T.numpy()
    observed = np.asarray(row["future_observed_mask"], dtype=bool)
    return prepared, truth, observed


def _dataset_features() -> Any:
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
            "capability_id": datasets.Value("string"),
            "capability_level": datasets.Value("int64"),
            "augmentation_seed": datasets.Value("int64"),
            "future_observed_mask": datasets.List(
                datasets.List(datasets.Value("bool"))
            ),
            "mase_scale_by_target": datasets.List(datasets.Value("float64")),
        }
    )


def _selection_audit(args: argparse.Namespace) -> tuple[dict[str, set[str] | None], dict[str, Any]]:
    selections: dict[str, set[str] | None] = {}
    source_counts: dict[str, dict[str, int]] = {}
    selected_counts: dict[str, dict[str, int]] = {}
    requested_datasets = set(args.datasets or ())
    dataset_roots = [
        root
        for root in _dataset_roots(args.experiment_root)
        if not requested_datasets or root.name in requested_datasets
    ]
    missing_datasets = requested_datasets - {root.name for root in dataset_roots}
    if missing_datasets:
        raise ValueError(
            f"Requested datasets are missing from {args.experiment_root}: "
            + ", ".join(sorted(missing_datasets))
        )
    for dataset_root in dataset_roots:
        selected, counts = _selection_for_dataset(dataset_root, args)
        selections[dataset_root.name] = selected
        source_counts[dataset_root.name] = {
            f"{capability}/level-{level}": count
            for (capability, level), count in sorted(counts.items())
        }
        selected_counts[dataset_root.name] = {
            key: min(value, args.max_per_stratum)
            if args.max_per_stratum is not None
            else value
            for key, value in source_counts[dataset_root.name].items()
        }
    audit = {
        "schema_version": "chronos2.cafe_seed_transfer_selection.v1",
        "experiment_root": str(args.experiment_root.resolve()),
        "gift_eval_dir": str(args.gift_eval_dir.resolve()),
        "capabilities": list(args.capabilities),
        "capability_levels": list(args.capability_levels),
        "datasets": [root.name for root in dataset_roots],
        "excluded_capabilities": ["nonlinear_persistence"],
        "max_per_dataset_capability_level": args.max_per_stratum,
        "selection_seed": args.selection_seed,
        "official_instance_fold": {
            "count": args.official_fold_count,
            "heldout": args.heldout_fold,
            "role": args.fold_role,
            "salt": args.fold_salt,
        },
        "source_counts": source_counts,
        "selected_counts": selected_counts,
    }
    audit["selection_sha256"] = _sha256_json(audit)
    return selections, audit


def command_audit(args: argparse.Namespace) -> None:
    _configure_cafe_import(args.cafe_root)
    _selections, audit = _selection_audit(args)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


def command_prepare(args: argparse.Namespace) -> None:
    import datasets

    _configure_cafe_import(args.cafe_root)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    selections, audit = _selection_audit(args)
    group_size_counts: Counter[int] = Counter()

    def generate() -> Iterator[dict[str, Any]]:
        rows = _iter_selected_dense_treatments(
            experiment_root=args.experiment_root,
            gift_eval_dir=args.gift_eval_dir,
            capabilities=frozenset(args.capabilities),
            selections=selections,
            fold_count=args.official_fold_count,
            heldout_fold=args.heldout_fold,
            role=args.fold_role,
            fold_salt=args.fold_salt,
            maximum_context=args.maximum_context,
            horizon=args.horizon,
        )
        for row in rows:
            prepared = _prepared_training_row(row)
            group_size_counts[int(prepared["n_targets"]) + int(prepared["n_covariates"])] += 1
            yield prepared

    dataset = datasets.Dataset.from_generator(
        generate,
        features=_dataset_features(),
        writer_batch_size=args.writer_batch_size,
    )
    if not group_size_counts:
        for row in dataset.select_columns(("n_targets", "n_covariates")):
            group_size_counts[int(row["n_targets"]) + int(row["n_covariates"])] += 1
    dataset.save_to_disk(str(args.output))
    audit["materialized_row_count"] = len(dataset)
    audit["maximum_context"] = args.maximum_context
    audit["horizon"] = args.horizon
    audit["training_group_size_counts"] = {
        str(group_size): count for group_size, count in sorted(group_size_counts.items())
    }
    (args.output / "cafe_adapter_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved %d prepared rows to %s", len(dataset), args.output)


def command_finetune(args: argparse.Namespace) -> None:
    import datasets
    import torch

    from chronos import Chronos2Pipeline

    dataset = datasets.load_from_disk(str(args.dataset))
    adapter_manifest_path = args.dataset / "cafe_adapter_manifest.json"
    adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    if adapter_manifest.get("horizon") != args.horizon:
        dataset = dataset.filter(lambda row: int(row["horizon"]) == args.horizon)
    if len(dataset) == 0:
        raise ValueError(f"No rows with horizon={args.horizon} in {args.dataset}")

    dataset = dataset.with_format("torch")
    epoch_steps = _dataset_epoch_steps(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    num_steps = args.num_steps or epoch_steps
    if args.training_sampling == "shuffle_without_replacement" and num_steps > epoch_steps:
        raise ValueError(
            f"--num-steps={num_steps} exceeds the single-epoch length {epoch_steps}"
        )
    checkpoint_steps = _checkpoint_steps(num_steps, args.checkpoint_interval)

    from transformers import TrainerCallback

    class SelectedCheckpointCallback(TrainerCallback):
        def on_step_end(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            if state.global_step in checkpoint_steps:
                control.should_save = True
            return control

    pipeline = Chronos2Pipeline.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
    )
    fitted = pipeline.fit(
        inputs=dataset,
        validation_inputs=None,
        prediction_length=args.horizon,
        finetune_mode=args.finetune_mode,
        context_length=args.context_length,
        learning_rate=args.learning_rate,
        num_steps=num_steps,
        batch_size=args.batch_size,
        min_past=args.min_past,
        output_dir=args.output,
        logging_steps=args.logging_steps,
        seed=args.seed,
        callbacks=[SelectedCheckpointCallback()],
        training_sampling=args.training_sampling,
        shuffle_seed=args.seed,
        save_total_limit=None,
        disable_tqdm=True,
    )
    training_manifest = {
        "schema_version": "chronos2.cafe_seed_transfer_training.v1",
        "dataset": str(args.dataset.resolve()),
        "model": args.model,
        "horizon": args.horizon,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "min_past": args.min_past,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "sampling": args.training_sampling,
        "epoch_steps": epoch_steps,
        "nominal_epoch_progress": num_steps / epoch_steps,
        "trained_steps": num_steps,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_progress": list(CHECKPOINT_PROGRESS),
        "training_rows": len(dataset),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cafe_training_manifest.json").write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Fine-tuned model saved at %s", args.output / "finetuned-ckpt")
    del fitted


class _MetricWriter:
    def __init__(self, path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = pa.schema(
            [
                ("model", pa.string()),
                ("dataset_id", pa.string()),
                ("official_instance_id", pa.string()),
                ("sample_id", pa.string()),
                ("capability_id", pa.string()),
                ("capability_level", pa.int16()),
                ("augmentation_seed", pa.int64()),
                ("horizon", pa.int16()),
                ("mase", pa.float64()),
                ("mae", pa.float64()),
                ("observed_cells", pa.int64()),
            ]
        )
        self._pa = pa
        self._writer = pq.ParquetWriter(self.temporary, self.schema, compression="zstd")
        self._buffer: list[dict[str, Any]] = []
        self.count = 0

    def write(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= 2048:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        table = self._pa.Table.from_pylist(self._buffer, schema=self.schema)
        self._writer.write_table(table)
        self.count += len(self._buffer)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()
        self.temporary.replace(self.path)


def _metric_record(
    *,
    model_label: str,
    row: Mapping[str, Any],
    forecast: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
) -> dict[str, Any]:
    if forecast.shape != truth.shape or observed.shape != truth.shape:
        raise ValueError(
            f"Prediction shape mismatch for {row['sample_id']}: "
            f"forecast={forecast.shape}, truth={truth.shape}, mask={observed.shape}"
        )
    scales = np.asarray(row["mase_scale_by_target"], dtype=np.float64)
    valid = observed & np.isfinite(truth) & np.isfinite(forecast)
    if not np.any(valid):
        raise ValueError(f"No observed finite cells for {row['sample_id']}")
    error = np.abs(forecast.astype(np.float64) - truth.astype(np.float64))
    scaled = error / scales[None, :]
    return {
        "model": model_label,
        "dataset_id": str(row["dataset_id"]),
        "official_instance_id": str(row["official_instance_id"]),
        "sample_id": str(row["sample_id"]),
        "capability_id": str(row["capability_id"]),
        "capability_level": int(row["capability_level"]),
        "augmentation_seed": int(row["augmentation_seed"]),
        "horizon": int(row["horizon"]),
        "mase": float(np.mean(scaled[valid])),
        "mae": float(np.mean(error[valid])),
        "observed_cells": int(np.count_nonzero(valid)),
    }


def command_evaluate(args: argparse.Namespace) -> None:
    import torch

    from chronos import Chronos2Pipeline

    _configure_cafe_import(args.cafe_root)
    if args.output.exists():
        raise FileExistsError(args.output)
    selections, audit = _selection_audit(args)
    pipeline = Chronos2Pipeline.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
    )
    writer = _MetricWriter(args.output)
    aggregates: defaultdict[tuple[str, str, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    batch_inputs: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    batch_truth: list[np.ndarray] = []
    batch_observed: list[np.ndarray] = []

    def flush() -> None:
        if not batch_inputs:
            return
        _quantiles, means = pipeline.predict_quantiles(
            batch_inputs,
            prediction_length=args.horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=args.batch_size,
        )
        for source, mean, truth, observed in zip(
            batch_rows, means, batch_truth, batch_observed, strict=True
        ):
            record = _metric_record(
                model_label=args.model_label,
                row=source,
                forecast=mean.float().cpu().numpy().T,
                truth=truth,
                observed=observed,
            )
            writer.write(record)
            keys = (
                ("overall", "all", 0),
                ("dataset", record["dataset_id"], 0),
                ("capability", record["capability_id"], int(record["capability_level"])),
            )
            for key in keys:
                aggregates[key][0] += record["mase"]
                aggregates[key][1] += 1
        batch_inputs.clear()
        batch_rows.clear()
        batch_truth.clear()
        batch_observed.clear()

    try:
        dense_rows = _iter_selected_dense_treatments(
            experiment_root=args.experiment_root,
            gift_eval_dir=args.gift_eval_dir,
            capabilities=frozenset(args.capabilities),
            selections=selections,
            fold_count=args.official_fold_count,
            heldout_fold=args.heldout_fold,
            role=args.fold_role,
            fold_salt=args.fold_salt,
            maximum_context=args.maximum_context,
            horizon=args.horizon,
        )
        for row in dense_rows:
            if int(row["horizon"]) != args.horizon:
                continue
            prepared, truth, observed = _prepared_evaluation_row(row)
            batch_inputs.append(prepared)
            batch_rows.append(row)
            batch_truth.append(truth)
            batch_observed.append(observed)
            if len(batch_inputs) >= args.input_batch_size:
                flush()
        flush()
        writer.close()
    except BaseException:
        writer._writer.close()
        writer.temporary.unlink(missing_ok=True)
        raise

    summary = {
        "schema_version": "chronos2.cafe_seed_transfer_evaluation.v1",
        "model": args.model_label,
        "model_path": args.model,
        "horizon": args.horizon,
        "metric_rows": writer.count,
        "selection": audit,
        "mase": {
            f"{kind}/{name}/level-{level}": values[0] / values[1]
            for (kind, name, level), values in sorted(aggregates.items())
            if values[1]
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote %d metric rows to %s", writer.count, args.output)


def command_evaluate_prepared(args: argparse.Namespace) -> None:
    import datasets
    import torch

    from chronos import Chronos2Pipeline

    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = datasets.load_from_disk(str(args.dataset))
    dataset = dataset.shard(
        num_shards=args.world_size,
        index=args.rank,
        contiguous=True,
    ).with_format("torch")
    pipeline = Chronos2Pipeline.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
    )
    aggregates: defaultdict[tuple[str, str, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0]
    )
    batch_inputs: list[dict[str, Any]] = []
    batch_rows: list[Mapping[str, Any]] = []
    batch_truth: list[np.ndarray] = []
    batch_observed: list[np.ndarray] = []
    metric_rows = 0

    def flush() -> None:
        nonlocal metric_rows
        if not batch_inputs:
            return
        _quantiles, means = pipeline.predict_quantiles(
            batch_inputs,
            prediction_length=args.horizon,
            quantile_levels=[0.5],
            batch_size=args.batch_size,
        )
        for source, mean, truth, observed in zip(
            batch_rows, means, batch_truth, batch_observed, strict=True
        ):
            record = _metric_record(
                model_label=args.model_label,
                row=source,
                forecast=mean.float().cpu().numpy().T,
                truth=truth,
                observed=observed,
            )
            key = (
                record["dataset_id"],
                record["capability_id"],
                int(record["capability_level"]),
            )
            aggregates[key][0] += record["mase"]
            aggregates[key][1] += 1
            aggregates[key][2] += record["observed_cells"]
            aggregates[key][3] += record["mase"] * record["observed_cells"]
            metric_rows += 1
        batch_inputs.clear()
        batch_rows.clear()
        batch_truth.clear()
        batch_observed.clear()

    for row in dataset:
        if int(row["horizon"]) != args.horizon:
            raise ValueError(
                f"Dataset {args.dataset} contains horizon={row['horizon']}, expected {args.horizon}"
            )
        prepared, truth, observed = _materialized_evaluation_row(row)
        batch_inputs.append(prepared)
        batch_rows.append(row)
        batch_truth.append(truth)
        batch_observed.append(observed)
        if len(batch_inputs) >= args.input_batch_size:
            flush()
    flush()

    result = {
        "schema_version": "chronos2.cafe_seed_transfer_prepared_evaluation_part.v1",
        "corpus": args.corpus,
        "dataset": str(args.dataset.resolve()),
        "model": args.model_label,
        "model_path": args.model,
        "step": args.step,
        "horizon": args.horizon,
        "rank": args.rank,
        "world_size": args.world_size,
        "metric_rows": metric_rows,
        "strata": [
            {
                "dataset_id": dataset_id,
                "capability_id": capability_id,
                "capability_level": level,
                "mase_sum": values[0],
                "row_count": int(values[1]),
                "observed_cells": int(values[2]),
                "cell_weighted_mase_sum": values[3],
            }
            for (dataset_id, capability_id, level), values in sorted(aggregates.items())
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    LOGGER.info("Wrote %d metric rows to %s", metric_rows, args.output)


def command_aggregate_curve(args: argparse.Namespace) -> None:
    import csv

    part_paths = sorted(args.parts.rglob("*.json"))
    if not part_paths:
        raise ValueError(f"No JSON evaluation parts found under {args.parts}")
    grouped: defaultdict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in part_paths:
        part = json.loads(path.read_text(encoding="utf-8"))
        if part.get("schema_version") != "chronos2.cafe_seed_transfer_prepared_evaluation_part.v1":
            continue
        key = (str(part["corpus"]), int(part["horizon"]), int(part["step"]))
        grouped[key].append(part)

    curve_rows: list[dict[str, Any]] = []
    for (corpus, horizon, step), parts in sorted(grouped.items()):
        world_sizes = {int(part["world_size"]) for part in parts}
        if len(world_sizes) != 1:
            raise ValueError(f"Inconsistent world sizes for {(corpus, horizon, step)}")
        world_size = world_sizes.pop()
        ranks = {int(part["rank"]) for part in parts}
        if ranks != set(range(world_size)):
            raise ValueError(
                f"Incomplete ranks for {(corpus, horizon, step)}: {sorted(ranks)}"
            )
        strata: defaultdict[tuple[str, str, int], list[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0, 0.0]
        )
        for part in parts:
            for row in part["strata"]:
                key = (
                    str(row["dataset_id"]),
                    str(row["capability_id"]),
                    int(row["capability_level"]),
                )
                strata[key][0] += float(row["mase_sum"])
                strata[key][1] += int(row["row_count"])
                strata[key][2] += int(row["observed_cells"])
                strata[key][3] += float(row["cell_weighted_mase_sum"])
        stratum_means = [values[0] / values[1] for values in strata.values()]
        datasets_acc: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for (dataset_id, _capability, _level), values in strata.items():
            datasets_acc[dataset_id][0] += values[0]
            datasets_acc[dataset_id][1] += values[1]
        dataset_means = [values[0] / values[1] for values in datasets_acc.values()]
        total_mase = sum(values[0] for values in strata.values())
        total_rows = sum(values[1] for values in strata.values())
        total_cells = sum(values[2] for values in strata.values())
        total_cell_mase = sum(values[3] for values in strata.values())
        curve_rows.append(
            {
                "corpus": corpus,
                "horizon": horizon,
                "step": step,
                "model": str(parts[0]["model"]),
                "row_count": int(total_rows),
                "stratum_count": len(strata),
                "dataset_count": len(datasets_acc),
                "macro_stratum_mase": float(np.mean(stratum_means)),
                "macro_dataset_mase": float(np.mean(dataset_means)),
                "sample_weighted_mase": total_mase / total_rows,
                "cell_weighted_mase": total_cell_mase / total_cells,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": "chronos2.cafe_seed_transfer_curve.v1",
                "rows": curve_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    import matplotlib.pyplot as plt

    horizons = sorted({int(row["horizon"]) for row in curve_rows})
    figure, axes_grid = plt.subplots(
        1,
        len(horizons),
        figsize=(5 * len(horizons), 4.5),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes_grid[0]
    colors = {"v13": "#2563eb", "v14": "#dc2626"}
    labels = {
        "v13": getattr(args, "train_label", "training treatment"),
        "v14": getattr(args, "cross_label", "cross-seed treatment"),
    }
    for axis, horizon in zip(axes, horizons, strict=True):
        for corpus in ("v13", "v14"):
            rows = sorted(
                (
                    row
                    for row in curve_rows
                    if row["horizon"] == horizon and row["corpus"] == corpus
                ),
                key=lambda row: int(row["step"]),
            )
            axis.plot(
                [row["step"] for row in rows],
                [row["macro_stratum_mase"] for row in rows],
                marker="o",
                linewidth=2,
                color=colors[corpus],
                label=labels[corpus],
            )
        axis.set_title(f"Horizon {horizon}")
        axis.set_xlabel("Training steps")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Macro-stratum MASE")
    axes[-1].legend(frameon=False)
    png_path = args.output.with_suffix(".png")
    pdf_path = args.output.with_suffix(".pdf")
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)

    relative_figure, relative_axes_grid = plt.subplots(
        1,
        len(horizons),
        figsize=(5 * len(horizons), 4.5),
        constrained_layout=True,
        squeeze=False,
    )
    relative_axes = relative_axes_grid[0]
    for axis, horizon in zip(relative_axes, horizons, strict=True):
        for corpus in ("v13", "v14"):
            rows = sorted(
                (
                    row
                    for row in curve_rows
                    if row["horizon"] == horizon and row["corpus"] == corpus
                ),
                key=lambda row: int(row["step"]),
            )
            if not rows:
                continue
            baseline = float(rows[0]["macro_stratum_mase"])
            axis.plot(
                [row["step"] for row in rows],
                [
                    100.0 * (float(row["macro_stratum_mase"]) / baseline - 1.0)
                    for row in rows
                ],
                marker="o",
                linewidth=2,
                color=colors[corpus],
                label=labels[corpus],
            )
        axis.axhline(0.0, color="#64748b", linewidth=1, alpha=0.7)
        axis.set_title(f"Horizon {horizon}")
        axis.set_xlabel("Training steps")
        axis.grid(alpha=0.25)
    relative_axes[0].set_ylabel("Macro-stratum MASE change from step 0 (%)")
    relative_axes[-1].legend(frameon=False)
    relative_png_path = args.output.with_name(f"{args.output.stem}-relative.png")
    relative_pdf_path = args.output.with_name(f"{args.output.stem}-relative.pdf")
    relative_figure.savefig(relative_png_path, dpi=180)
    relative_figure.savefig(relative_pdf_path)
    plt.close(relative_figure)
    LOGGER.info(
        "Wrote %d curve rows to %s, %s, %s, and %s",
        len(curve_rows),
        args.output,
        csv_path,
        png_path,
        pdf_path,
    )


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cafe-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gift-eval-dir", type=Path, required=True)
    parser.add_argument("--capabilities", nargs="+", choices=CAPABILITIES, default=list(CAPABILITIES))
    parser.add_argument(
        "--capability-levels",
        nargs="+",
        type=_positive,
        choices=(1, 2, 3, 4, 5),
        default=[1, 2, 3, 4, 5],
        help="Capability levels to include; default keeps levels 1 through 5.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional gift_* dataset ids to include.",
    )
    parser.add_argument(
        "--max-per-stratum",
        type=_positive,
        default=None,
        help="Maximum samples per (dataset, capability, level); default keeps all.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260825)
    parser.add_argument("--official-fold-count", type=_positive, default=1)
    parser.add_argument("--heldout-fold", type=_nonnegative, default=0)
    parser.add_argument("--fold-role", choices=("all", "train", "eval"), default="all")
    parser.add_argument(
        "--fold-salt",
        default="",
        help="Optional salt for deterministic official-instance fold assignment.",
    )
    parser.add_argument("--maximum-context", type=_positive, default=8192)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit capability/fold/sample selection without replaying curves.")
    _add_selection_arguments(audit)
    audit.set_defaults(func=command_audit)

    prepare = subparsers.add_parser("prepare", help="Materialize a lazy Hugging Face training dataset.")
    _add_selection_arguments(prepare)
    prepare.add_argument("--horizon", type=_positive, choices=(30, 48, 60), default=None)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--writer-batch-size", type=_positive, default=128)
    prepare.set_defaults(func=command_prepare)

    finetune = subparsers.add_parser("finetune", help="Fine-tune one horizon-specific Chronos-2 adapter.")
    finetune.add_argument("--dataset", type=Path, required=True)
    finetune.add_argument("--horizon", type=_positive, choices=(30, 48, 60), required=True)
    finetune.add_argument("--model", default="amazon/chronos-2")
    finetune.add_argument("--output", type=Path, required=True)
    finetune.add_argument("--device", default="cuda")
    finetune.add_argument("--finetune-mode", choices=("full", "lora"), default="lora")
    finetune.add_argument("--context-length", type=_positive, default=2048)
    finetune.add_argument("--min-past", type=_positive, default=64)
    finetune.add_argument("--learning-rate", type=float, default=1e-4)
    finetune.add_argument(
        "--num-steps",
        type=_positive,
        default=None,
        help="Number of optimizer steps; default uses one nominal packed epoch.",
    )
    finetune.add_argument(
        "--training-sampling",
        choices=("random_with_replacement", "shuffle_without_replacement"),
        default="random_with_replacement",
        help="Chronos-2 input sampling policy; the official default is random_with_replacement.",
    )
    finetune.add_argument(
        "--checkpoint-interval",
        type=_positive,
        help="Save checkpoints at this step interval and at the final step.",
    )
    finetune.add_argument("--batch-size", type=_positive, default=32)
    finetune.add_argument("--logging-steps", type=_positive, default=50)
    finetune.add_argument("--seed", type=int, default=20260825)
    finetune.set_defaults(func=command_finetune)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate one model on one CaFE horizon.")
    _add_selection_arguments(evaluate)
    evaluate.add_argument("--horizon", type=_positive, choices=(30, 48, 60), required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--model-label", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=_positive, default=64)
    evaluate.add_argument("--input-batch-size", type=_positive, default=256)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(func=command_evaluate)

    evaluate_prepared = subparsers.add_parser(
        "evaluate-prepared",
        help="Evaluate one model on a pre-materialized treatment dataset shard.",
    )
    evaluate_prepared.add_argument("--dataset", type=Path, required=True)
    evaluate_prepared.add_argument("--horizon", type=_positive, choices=(30, 48, 60), required=True)
    evaluate_prepared.add_argument("--model", required=True)
    evaluate_prepared.add_argument("--model-label", required=True)
    evaluate_prepared.add_argument("--step", type=_nonnegative, required=True)
    evaluate_prepared.add_argument("--corpus", choices=("v13", "v14"), required=True)
    evaluate_prepared.add_argument("--rank", type=_nonnegative, default=0)
    evaluate_prepared.add_argument("--world-size", type=_positive, default=1)
    evaluate_prepared.add_argument("--device", default="cuda")
    evaluate_prepared.add_argument("--batch-size", type=_positive, default=64)
    evaluate_prepared.add_argument("--input-batch-size", type=_positive, default=256)
    evaluate_prepared.add_argument("--output", type=Path, required=True)
    evaluate_prepared.set_defaults(func=command_evaluate_prepared)

    aggregate_curve = subparsers.add_parser(
        "aggregate-curve",
        help="Merge all sharded prepared-evaluation summaries into curve JSON and CSV.",
    )
    aggregate_curve.add_argument("--parts", type=Path, required=True)
    aggregate_curve.add_argument("--output", type=Path, required=True)
    aggregate_curve.add_argument("--train-label", default="training treatment")
    aggregate_curve.add_argument("--cross-label", default="cross-seed treatment")
    aggregate_curve.set_defaults(func=command_aggregate_curve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.command in {"audit", "prepare", "evaluate"}
        and args.heldout_fold >= args.official_fold_count
    ):
        parser.error("--heldout-fold must be smaller than --official-fold-count")
    if args.command == "evaluate-prepared" and args.rank >= args.world_size:
        parser.error("--rank must be smaller than --world-size")
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
