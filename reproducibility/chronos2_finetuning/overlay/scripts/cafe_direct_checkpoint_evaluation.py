#!/usr/bin/env python3
"""Directly evaluate one Chronos-2 checkpoint without prediction artifacts."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--treatment-dataset", type=Path, required=True)
    parser.add_argument("--evaluation-data", type=Path, required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--prediction-length", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _median_forecast(model: Any, batch: dict[str, Any]) -> Any:
    import torch
    from einops import rearrange

    encoder_outputs, loc_scale, _future_mask, num_context_patches = model.encode(
        **batch
    )
    hidden = encoder_outputs[0]
    num_output_patches = int(batch["num_output_patches"])
    forecast_embeds = hidden[:, -num_output_patches:]
    block = model.output_patch_embedding
    output_patch_size = int(model.chronos_config.output_patch_size)
    median_index = list(model.chronos_config.quantiles).index(0.5)
    start = median_index * output_patch_size
    end = start + output_patch_size
    # Preserve the exact BF16 computation performed by model.forward().  Slicing
    # the output weights before GEMM changes accumulation/rounding.  We only
    # retain and transfer the q=0.5 forecast after the original projection.
    normalized = block(forecast_embeds)[..., start:end]
    normalized = rearrange(normalized, "b n p -> b (n p)")
    forecast = model.instance_norm.inverse(normalized, loc_scale)
    return forecast


def _predict_dataset(
    *,
    pipeline: Any,
    source: Any,
    context_length: int,
    prediction_length: int,
    batch_size: int,
    consume: Callable[[Mapping[str, Any], np.ndarray, np.ndarray], None],
) -> tuple[int, int]:
    import torch

    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    metadata_columns = [
        name
        for name in source.column_names
        if name
        not in {
            "context",
            "future_covariates",
            "n_targets",
            "n_covariates",
            "n_future_covariates",
        }
    ]
    metadata = source.select_columns(metadata_columns)
    tensor_source = source.with_format("torch")
    evaluation = Chronos2Dataset(
        inputs=tensor_source,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        output_patch_size=pipeline.model_output_patch_size,
        min_past=1,
        mode=DatasetMode.VALIDATION,
    )
    cursor = 0
    batch_count = 0
    model = pipeline.model
    with torch.inference_mode():
        for batch in evaluation._generate_sequential_batches():
            target_ranges = batch.pop("target_idx_ranges")
            future_target = batch.pop("future_target")
            assert isinstance(future_target, torch.Tensor)
            device_batch = {
                key: value.to(model.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            forecasts = _median_forecast(model, device_batch).float().cpu().numpy()
            truths = future_target.float().numpy()
            for offset, (start, end) in enumerate(target_ranges):
                row = metadata[cursor + offset]
                consume(row, forecasts[start:end].T, truths[start:end].T)
            cursor += len(target_ranges)
            batch_count += 1
    if cursor != len(source):
        raise RuntimeError(f"Predicted {cursor} inputs from a dataset of length {len(source)}")
    return cursor, batch_count


def main() -> None:
    args = parse_args()

    import datasets
    import pyarrow.parquet as pq
    import torch

    from chronos import Chronos2Pipeline

    started = time.perf_counter()
    effect_table = pq.read_table(args.evaluation_data / "effect_metadata.parquet")
    effects = {
        str(row["sample_id"]): row for row in effect_table.to_pylist()
    }
    baseline_source = datasets.load_from_disk(
        str(args.evaluation_data / "baselines")
    )
    treatment_source = datasets.load_from_disk(str(args.treatment_dataset)).shard(
        num_shards=args.world_size,
        index=args.rank,
        contiguous=False,
    )
    pipeline = Chronos2Pipeline.from_pretrained(
        str(args.model),
        device_map=args.device,
        dtype=(
            torch.bfloat16
            if args.dtype == "bfloat16" and args.device != "cpu"
            else torch.float32
        ),
    )
    pipeline.model.eval()

    baselines: dict[str, dict[str, Any]] = {}

    def consume_baseline(
        row: Mapping[str, Any], forecast: np.ndarray, truth: np.ndarray
    ) -> None:
        baselines[str(row["sample_id"])] = {
            "forecast": forecast,
            "truth": truth,
            "mask": np.asarray(row["future_observed_mask"], dtype=bool),
            "scales": np.asarray(row["mase_scale_by_target"], dtype=float),
        }

    baseline_count, baseline_batches = _predict_dataset(
        pipeline=pipeline,
        source=baseline_source,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        consume=consume_baseline,
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

    def consume_treatment(
        row: Mapping[str, Any], forecast: np.ndarray, truth: np.ndarray
    ) -> None:
        sample_id = str(row["sample_id"])
        metadata = effects[sample_id]
        baseline = baselines[str(metadata["baseline_sample_id"])]
        mask = baseline["mask"]
        scales = baseline["scales"]
        valid = mask & np.isfinite(truth) & np.isfinite(forecast)
        if not np.any(valid):
            raise ValueError(f"No observed values for {sample_id}")
        mase = float(np.mean((np.abs(forecast - truth) / scales[None, :])[valid]))
        key = (
            str(row["dataset_id"]),
            str(row["capability_id"]),
            int(row["capability_level"]),
        )
        accuracy[key][0] += mase
        accuracy[key][1] += 1.0

        values = effect[key]
        values["candidate_count"] += 1.0
        affected = [int(value) for value in metadata["affected_target_indices"]]
        assessed = np.zeros_like(mask, dtype=bool)
        assessed[:, affected] = mask[:, affected]
        observed_count = int(np.count_nonzero(assessed))
        if observed_count == 0:
            return
        truth_delta = truth - baseline["truth"]
        forecast_delta = forecast - baseline["forecast"]
        truth_standardized = (truth_delta / scales[None, :])[assessed]
        forecast_standardized = (forecast_delta / scales[None, :])[assessed]
        truth_squared_sum = float(np.sum(np.square(truth_standardized)))
        truth_mase_rms = math.sqrt(truth_squared_sum / observed_count)
        if truth_mase_rms < 0.05 - 1e-12:
            return
        difference = forecast_standardized - truth_standardized
        values["scored_count"] += 1.0
        values["squared_error_sum"] += float(np.sum(np.square(difference)))
        values["truth_squared_sum"] += truth_squared_sum
        values["observed_cell_count"] += observed_count

    treatment_count, treatment_batches = _predict_dataset(
        pipeline=pipeline,
        source=treatment_source,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        consume=consume_treatment,
    )

    payload = {
        "schema_version": "chronos2.cafe_direct_metric_part.v1",
        "corpus": args.corpus,
        "step": args.step,
        "rank": args.rank,
        "world_size": args.world_size,
        "model": str(args.model.resolve()),
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "baseline_count": baseline_count,
        "baseline_batches": baseline_batches,
        "treatment_count": treatment_count,
        "treatment_batches": treatment_batches,
        "accuracy_strata": [
            {
                "dataset_id": key[0],
                "capability_id": key[1],
                "capability_level": key[2],
                "mase_sum": values[0],
                "row_count": int(values[1]),
            }
            for key, values in sorted(accuracy.items())
        ],
        "effect_strata": [
            {
                "dataset_id": key[0],
                "capability_id": key[1],
                "capability_level": key[2],
                **{
                    name: int(value)
                    if name.endswith("count")
                    else float(value)
                    for name, value in values.items()
                },
            }
            for key, values in sorted(effect.items())
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "complete": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "corpus": args.corpus,
                "step": args.step,
                "rank": args.rank,
                "baseline_count": baseline_count,
                "treatment_count": treatment_count,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
