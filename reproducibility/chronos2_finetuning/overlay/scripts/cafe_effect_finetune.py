#!/usr/bin/env python3
"""Fine-tune Chronos-2 directly on paired CaFE treatment effects."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import IterableDataset

from cafe_seed_transfer import CHECKPOINT_PROGRESS, _checkpoint_steps
from chronos import Chronos2Pipeline
from chronos.chronos2 import Chronos2Model
from chronos.chronos2.dataset import left_pad_and_cat_2D
from chronos.chronos2.trainer import Chronos2Trainer


class PairedEffectDataset(IterableDataset):
    """Pack fixed-origin official/treatment pairs into Chronos-2 batches."""

    def __init__(
        self,
        rows: Any,
        *,
        context_length: int,
        prediction_length: int,
        batch_size: int,
        shuffle_seed: int,
        training_sampling: Literal[
            "random_with_replacement", "shuffle_without_replacement"
        ] = "random_with_replacement",
    ) -> None:
        super().__init__()
        self.rows = rows
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.training_sampling = training_sampling

    def _task(
        self, row: Mapping[str, Any], prefix: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        full = torch.as_tensor(row[f"{prefix}_context"], dtype=torch.float32)
        horizon = self.prediction_length
        origin = full.shape[-1] - horizon
        if origin <= 0:
            raise ValueError("Paired input is shorter than the prediction horizon")
        context = full[:, max(0, origin - self.context_length) : origin]
        future_target = full[:, origin:].clone()
        n_targets = int(row["n_targets"])
        future_target[n_targets:] = torch.nan
        future_covariates = torch.as_tensor(
            row[f"{prefix}_future_covariates"], dtype=torch.float32
        )
        return context, future_target, future_covariates, n_targets

    def _build_batch(self, indices: list[int]) -> dict[str, torch.Tensor | int]:
        contexts: list[torch.Tensor] = []
        futures: list[torch.Tensor] = []
        covariates: list[torch.Tensor] = []
        group_ids: list[torch.Tensor] = []
        ranges: list[tuple[int, int, int, int]] = []
        masks: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        row_offset = 0
        group_id = 0

        for index in indices:
            row = self.rows[index]
            pair_ranges: list[tuple[int, int]] = []
            pair_mask = torch.as_tensor(
                row["future_observed_mask"], dtype=torch.bool
            ).T
            affected = torch.as_tensor(row["affected_target_indices"], dtype=torch.long)
            assessed = torch.zeros_like(pair_mask)
            assessed[affected] = pair_mask[affected]
            pair_scales = torch.as_tensor(
                row["mase_scale_by_target"], dtype=torch.float32
            )

            for prefix in ("baseline", "treatment"):
                context, future, future_covariates, n_targets = self._task(
                    row, prefix
                )
                group_size = context.shape[0]
                pair_ranges.append((row_offset, row_offset + n_targets))
                contexts.append(context)
                futures.append(future)
                covariates.append(future_covariates)
                group_ids.append(torch.full((group_size,), group_id, dtype=torch.long))
                masks.append(
                    torch.cat(
                        (
                            assessed,
                            torch.zeros(
                                (group_size - n_targets, self.prediction_length),
                                dtype=torch.bool,
                            ),
                        )
                    )
                )
                scales.append(
                    torch.cat(
                        (
                            pair_scales,
                            torch.ones(group_size - n_targets, dtype=torch.float32),
                        )
                    )
                )
                row_offset += group_size
                group_id += 1
            ranges.append((*pair_ranges[0], *pair_ranges[1]))

        return {
            "context": left_pad_and_cat_2D(contexts),
            "future_target": torch.cat(futures),
            "future_covariates": torch.cat(covariates),
            "group_ids": torch.cat(group_ids),
            "num_output_patches": math.ceil(self.prediction_length / 16),
            # Keep ranges on CPU as plain metadata. Moving this tiny index table
            # to CUDA makes every Python int conversion below synchronize.
            "effect_pair_ranges": ranges,
            "effect_mask": torch.cat(masks),
            "effect_scale": torch.cat(scales),
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | int]]:
        if self.training_sampling == "random_with_replacement":
            rng = np.random.default_rng(self.shuffle_seed)
            while True:
                batch: list[int] = []
                used = 0
                while used < self.batch_size:
                    index = int(rng.integers(len(self.rows)))
                    row = self.rows[index]
                    pair_size = 2 * (
                        int(row["n_targets"]) + int(row["n_covariates"])
                    )
                    batch.append(index)
                    used += pair_size
                yield self._build_batch(batch)

        indices = np.arange(len(self.rows))
        np.random.default_rng(self.shuffle_seed).shuffle(indices)
        batch: list[int] = []
        used = 0
        for raw_index in indices:
            index = int(raw_index)
            row = self.rows[index]
            pair_size = 2 * (int(row["n_targets"]) + int(row["n_covariates"]))
            batch.append(index)
            used += pair_size
            if used >= self.batch_size:
                yield self._build_batch(batch)
                batch = []
                used = 0
        if batch:
            yield self._build_batch(batch)


def effect_loss(
    predictions: torch.Tensor,
    truth: torch.Tensor,
    pair_ranges: Sequence[Sequence[int] | torch.Tensor],
    mask: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Return batch-pooled squared effect NRMSE, excluding weak effects."""

    numerator = predictions.new_zeros(())
    denominator = predictions.new_zeros(())
    scored = 0
    for baseline_start, baseline_end, treatment_start, treatment_end in pair_ranges:
        bs, be = int(baseline_start), int(baseline_end)
        ts, te = int(treatment_start), int(treatment_end)
        assessed = mask[bs:be]
        pair_scale = scale[bs:be, None]
        truth_delta = (truth[ts:te] - truth[bs:be]) / pair_scale
        prediction_delta = (predictions[ts:te] - predictions[bs:be]) / pair_scale
        truth_values = truth_delta[assessed]
        if truth_values.numel() == 0 or torch.sqrt(torch.mean(truth_values.square())) < 0.05:
            continue
        numerator = numerator + (prediction_delta[assessed] - truth_values).square().sum()
        denominator = denominator + truth_values.square().sum()
        scored += 1
    if scored == 0:
        raise ValueError("Batch contains no scoreable CaFE effects")
    return numerator / denominator.clamp_min(1e-12)


class EffectTrainer(Chronos2Trainer):
    def __init__(self, *args: Any, median_index: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.median_index = median_index

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        truth = inputs.pop("future_target")
        pair_ranges = inputs.pop("effect_pair_ranges")
        mask = inputs.pop("effect_mask")
        scale = inputs.pop("effect_scale")
        outputs = model(**inputs)
        median = outputs.quantile_preds[:, self.median_index, : truth.shape[-1]].float()
        loss = effect_loss(median, truth.float(), pair_ranges, mask, scale.float())
        return (loss, outputs) if return_outputs else loss


def _epoch_steps(rows: Any, batch_size: int, seed: int) -> int:
    sizes = np.asarray(
        [2 * (int(row["n_targets"]) + int(row["n_covariates"])) for row in rows],
        dtype=np.int32,
    )
    np.random.default_rng(seed).shuffle(sizes)
    steps = 0
    used = 0
    for size in sizes:
        used += int(size)
        if used >= batch_size:
            steps += 1
            used = 0
    return steps + int(used > 0)


def train(args: argparse.Namespace) -> None:
    import datasets
    from transformers import TrainerCallback, TrainingArguments

    if args.output.exists():
        raise FileExistsError(args.output)
    rows = datasets.load_from_disk(str(args.dataset)).with_format("torch")
    epoch_steps = _epoch_steps(rows, args.batch_size, args.seed)
    num_steps = args.num_steps or epoch_steps
    checkpoints = _checkpoint_steps(num_steps, args.checkpoint_interval)

    class SaveSelectedSteps(TrainerCallback):
        def on_step_end(
            self, _args: Any, state: Any, control: Any, **_kwargs: Any
        ) -> Any:
            if state.global_step in checkpoints:
                control.should_save = True
            return control

    pipeline = Chronos2Pipeline.from_pretrained(
        args.model, device_map=args.device, dtype=torch.bfloat16
    )
    config = deepcopy(pipeline.model.config)
    model = Chronos2Model(config).to(pipeline.model.device)
    model.load_state_dict(pipeline.model.state_dict())
    if args.finetune_mode == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=[
                    "self_attention.q",
                    "self_attention.v",
                    "self_attention.k",
                    "self_attention.o",
                    "output_patch_embedding.output_layer",
                ],
            ),
        )
    dataset = PairedEffectDataset(
        rows,
        context_length=args.context_length,
        prediction_length=args.horizon,
        batch_size=args.batch_size,
        shuffle_seed=args.seed,
        training_sampling=args.training_sampling,
    )
    has_sm80 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=0,
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        disable_tqdm=True,
        report_to="none",
        max_steps=num_steps,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        tf32=has_sm80,
        bf16=has_sm80,
        save_only_model=True,
        save_total_limit=None,
        save_strategy="no",
        remove_unused_columns=False,
    )
    training_args._n_gpu = 1
    trainer = EffectTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[SaveSelectedSteps()],
        median_index=list(model.chronos_config.quantiles).index(0.5),
    )
    trainer.train()
    final_path = args.output / "finetuned-ckpt"
    Chronos2Pipeline(model=model).save_pretrained(final_path)
    manifest = {
        "schema_version": "chronos2.cafe_paired_effect_training.v1",
        "dataset": str(args.dataset.resolve()),
        "model": args.model,
        "objective": "batch_pooled_squared_mase_standardized_effect_nrmse",
        "finetune_mode": args.finetune_mode,
        "sampling": args.training_sampling,
        "fixed_forecast_origin": True,
        "horizon": args.horizon,
        "batch_size_series_budget": args.batch_size,
        "context_length": args.context_length,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "epoch_steps": epoch_steps,
        "trained_steps": num_steps,
        "checkpoint_steps": checkpoints,
        "checkpoint_progress": list(CHECKPOINT_PROGRESS),
        "training_pairs": len(rows),
    }
    (args.output / "cafe_effect_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"trained {num_steps} paired-effect steps at {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument(
        "--training-sampling",
        choices=("random_with_replacement", "shuffle_without_replacement"),
        default="random_with_replacement",
    )
    parser.add_argument("--finetune-mode", choices=("full", "lora"), default="lora")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
