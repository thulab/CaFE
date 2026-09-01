from __future__ import annotations

import importlib.metadata
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cafe.inference.native_catalog import NATIVE_MODEL_SPECS, model_weight_path


NATIVE_RUNTIME_SCHEMA = "cafe.native_model_runtime.v1"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _import_timer_runtime(model_code_root: Path) -> tuple[Any, Any]:
    root = str(model_code_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import torch
        from core.model.storage import model_loader
        from core.model.storage.model_info import get_builtin_hf_model_info
        from core.model.storage.utils import (
            eager_load_pretrained,
            get_classes_by_builtin_code,
            get_model_files_path,
        )

        def load_model_direct(model_info: Any, **model_kwargs: Any) -> Any:
            """Load only model weights; bypass service backend discovery."""

            device = torch.device(model_kwargs.get("device_map", "cpu"))
            model_class, config = get_classes_by_builtin_code(model_info)
            if model_class is None:
                raise ValueError(f"model class is unavailable for {model_info.model_id}")
            weights = get_model_files_path(model_info)
            if model_info.auto_map is not None:
                if model_loader._overrides_from_pretrained(model_class):
                    model = model_class.from_pretrained(
                        weights,
                        config=config,
                        trust_remote_code=True,
                        dtype=torch.float32 if device.type == "cpu" else "auto",
                    )
                else:
                    model = eager_load_pretrained(
                        model_class,
                        weights,
                        dtype=torch.float32 if device.type == "cpu" else "auto",
                        config=config,
                    )
            else:
                model = model_class.from_pretrained(weights)
            return model.to(device).eval()

        # BasicPipeline imports this symbol once when its module is initialized.
        # Install the direct loader before importing pipeline_loader.
        model_loader.load_model = load_model_direct
        from inference.pipeline.pipeline_loader import load_pipeline
    except ImportError as error:
        raise RuntimeError(
            "the native runtime needs the staged model implementation tree; "
            f"could not import it from {root}"
        ) from error
    return get_builtin_hf_model_info, load_pipeline


def _covariate_keys(child: dict[str, Any]) -> list[str]:
    configured = child.get("covariate_column_names")
    dimension = int(child.get("covariate_dim", 0))
    if configured is None:
        return [f"covariate_{index}" for index in range(dimension)]
    names = [str(value) for value in configured]
    if len(names) != dimension or len(set(names)) != dimension:
        raise ValueError("covariate_column_names must be unique and match covariate_dim")
    return names


def children_to_pipeline_inputs(children: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild the tensors that the old HTTP router created from bulk payloads."""

    import torch

    output: list[dict[str, Any]] = []
    for child in children:
        context = int(child["context_length"])
        horizon = int(child["horizon"])
        target = np.asarray(child["target"], dtype=np.float32)[:context].T
        target_finite = np.isfinite(target)
        item: dict[str, Any] = {
            "targets": torch.from_numpy(
                np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            ),
            "targets_observed_mask": torch.from_numpy(target_finite),
            "history_covs": {},
            "future_covs": {},
            "history_covs_observed_mask": {},
            "future_covs_observed_mask": {},
        }
        covariate_dim = int(child.get("covariate_dim", 0))
        if covariate_dim:
            covariates = np.asarray(child["covariates"], dtype=np.float32)
            keys = _covariate_keys(child)
            visible = [
                bool(value)
                for value in child.get(
                    "future_covariate_visible", [True] * covariate_dim
                )
            ]
            if len(visible) != covariate_dim:
                raise ValueError("future_covariate_visible must match covariate_dim")
            for index, key in enumerate(keys):
                history = covariates[:context, index]
                history_finite = np.isfinite(history)
                item["history_covs"][key] = torch.from_numpy(
                    np.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)
                )
                item["history_covs_observed_mask"][key] = torch.from_numpy(
                    history_finite
                )
                if visible[index]:
                    future = covariates[context : context + horizon, index]
                    future_finite = np.isfinite(future)
                    item["future_covs"][key] = torch.from_numpy(
                        np.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0)
                    )
                    item["future_covs_observed_mask"][key] = torch.from_numpy(
                        future_finite
                    )
        output.append(item)
    return output


class _PipelineAdapter:
    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def forecast(self, children: list[dict[str, Any]]) -> list[np.ndarray]:
        horizon = int(children[0]["horizon"])
        infer_kwargs = {"output_length": horizon}
        inputs = children_to_pipeline_inputs(children)
        prepared = self.pipeline.preprocess(inputs, **infer_kwargs)
        outputs = self.pipeline.forecast(prepared, **infer_kwargs)
        outputs = self.pipeline.postprocess(outputs, **infer_kwargs)
        return [
            np.asarray(
                output.detach().cpu().numpy()
                if hasattr(output, "detach")
                else output,
                dtype=np.float32,
            )
            for output in outputs
        ]


class _Chronos2Adapter:
    def __init__(self, weight_path: Path, device: str):
        import torch
        from chronos import Chronos2Pipeline

        self.pipeline = Chronos2Pipeline.from_pretrained(
            str(weight_path),
            device_map=device,
            torch_dtype=torch.float32,
        )

    def forecast(self, children: list[dict[str, Any]]) -> list[np.ndarray]:
        horizon = int(children[0]["horizon"])
        inputs: list[dict[str, Any]] = []
        for child in children:
            context = int(child["context_length"])
            target = np.asarray(child["target"], dtype=np.float32)[:context].T
            item: dict[str, Any] = {"target": target}
            covariate_dim = int(child.get("covariate_dim", 0))
            if covariate_dim:
                covariates = np.asarray(child["covariates"], dtype=np.float32)
                keys = _covariate_keys(child)
                visible = child.get(
                    "future_covariate_visible", [True] * covariate_dim
                )
                item["past_covariates"] = {
                    key: covariates[:context, index]
                    for index, key in enumerate(keys)
                }
                item["future_covariates"] = {
                    key: covariates[context : context + horizon, index]
                    for index, key in enumerate(keys)
                    if bool(visible[index])
                }
            inputs.append(item)
        _quantiles, medians = self.pipeline.predict_quantiles(
            inputs,
            prediction_length=horizon,
            quantile_levels=[0.5],
            batch_size=sum(
                int(child["target_dim"]) + int(child.get("covariate_dim", 0))
                for child in children
            ),
        )
        return [np.asarray(value, dtype=np.float32) for value in medians]


def _timesfm_xreg(
    target: np.ndarray,
    covariates: np.ndarray,
    *,
    context: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    history_x = np.column_stack(
        [np.ones(context, dtype=np.float64), covariates[:context]]
    )
    future_x = np.column_stack(
        [np.ones(horizon, dtype=np.float64), covariates[context : context + horizon]]
    )
    beta = np.linalg.pinv(history_x) @ target.astype(np.float64)
    return (
        np.asarray(target - history_x @ beta, dtype=np.float32),
        np.asarray(future_x @ beta, dtype=np.float32),
    )


class _TimesFM2p5Adapter:
    def __init__(self, weight_path: Path, device: str):
        import torch
        import timesfm

        if not device.startswith("cuda:"):
            raise ValueError("TimesFM 2.5 native runtime currently requires CUDA")
        torch.cuda.set_device(int(device.split(":", 1)[1]))
        torch.set_float32_matmul_precision("high")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            str(weight_path)
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=15360,
                max_horizon=1024,
                normalize_inputs=True,
                per_core_batch_size=1,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    def forecast(self, children: list[dict[str, Any]]) -> list[np.ndarray]:
        horizon = int(children[0]["horizon"])
        contexts: list[np.ndarray] = []
        xreg_futures: list[np.ndarray | None] = []
        for child in children:
            context = int(child["context_length"])
            target = np.asarray(child["target"], dtype=np.float32)[:context, 0]
            if int(child.get("covariate_dim", 0)):
                covariates = np.asarray(child["covariates"], dtype=np.float32)
                residual, future = _timesfm_xreg(
                    target, covariates, context=context, horizon=horizon
                )
                contexts.append(residual)
                xreg_futures.append(future)
            else:
                contexts.append(target)
                xreg_futures.append(None)
        points, _quantiles = self.model.forecast(horizon=horizon, inputs=contexts)
        output: list[np.ndarray] = []
        for point, xreg in zip(points, xreg_futures, strict=True):
            values = np.asarray(point, dtype=np.float32)
            if xreg is not None:
                values = values + xreg
            output.append(values[np.newaxis, :])
        return output


class _Toto2Adapter:
    def __init__(self, weight_path: Path, device: str):
        import torch
        from toto2 import Toto2Model

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.set_float32_matmul_precision("high")
        self.model = Toto2Model.from_pretrained(
            str(weight_path), map_location="cpu"
        ).to(self.device).eval()

    def forecast(self, children: list[dict[str, Any]]) -> list[np.ndarray]:
        import torch.nn.functional as functional

        horizon = int(children[0]["horizon"])
        patch_size = int(self.model.config.patch_size)
        targets = []
        masks = []
        target_counts = []
        for child in children:
            context = int(child["context_length"])
            values = self.torch.from_numpy(
                np.asarray(child["target"], dtype=np.float32)[:context].T
            )
            mask = self.torch.isfinite(values)
            values = self.torch.nan_to_num(values)
            padding = (-values.shape[-1]) % patch_size
            if padding:
                values = functional.pad(values, (padding, 0), value=0.0)
                mask = functional.pad(mask, (padding, 0), value=False)
            targets.append(values)
            masks.append(mask)
            target_counts.append(int(child["target_dim"]))
        target = self.torch.stack(targets).to(self.device)
        target_mask = self.torch.stack(masks).to(self.device)
        series_ids = self.torch.zeros(
            target.shape[:2], dtype=self.torch.long, device=self.device
        )
        with self.torch.inference_mode():
            quantiles = self.model.forecast(
                {
                    "target": target,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=horizon,
                decode_block_size=None,
                has_missing_values=not bool(target_mask.all().item()),
            )
        median_index = self.model.output_head.knots.index(0.5)
        median = quantiles[median_index]
        return [
            median[index, :target_count, :horizon]
            .detach()
            .cpu()
            .to(self.torch.float32)
            .numpy()
            for index, target_count in enumerate(target_counts)
        ]


@dataclass
class NativeForecastRuntime:
    model_id: str
    model_root: Path
    model_code_root: Path
    device: str
    adapter: Any
    load_seconds: float

    @classmethod
    def load(
        cls,
        *,
        model_id: str,
        model_root: Path,
        model_code_root: Path,
        device: str,
    ) -> "NativeForecastRuntime":
        if model_id not in NATIVE_MODEL_SPECS:
            raise ValueError(f"unknown native model: {model_id}")
        interpreter_bin = str(Path(sys.executable).parent.resolve())
        current_path = os.environ.get("PATH", "")
        if interpreter_bin not in current_path.split(os.pathsep):
            os.environ["PATH"] = interpreter_bin + os.pathsep + current_path
        weights = model_weight_path(model_root, model_id)
        if not weights.is_dir():
            raise FileNotFoundError(f"native model weights are missing: {weights}")

        import torch

        runtime_device = torch.device(device)
        if runtime_device.type == "cuda":
            torch.cuda.set_device(runtime_device)
            torch.set_float32_matmul_precision("high")

        started = time.monotonic()
        if model_id == "Chronos-2":
            adapter: Any = _Chronos2Adapter(weights, device)
        elif model_id == "timesfm2.5":
            adapter = _TimesFM2p5Adapter(weights, device)
        elif model_id == "toto2.0":
            adapter = _Toto2Adapter(weights, device)
        else:
            # The retained model-only implementation reads this setting while
            # loading a builtin checkpoint. It does not start an HTTP server,
            # ZMQ transport, database, or coordinator.
            os.environ["TIMER_MODELS_DIR"] = str(model_root.resolve())
            get_model_info, load_pipeline = _import_timer_runtime(model_code_root)
            model_info = get_model_info(model_id)
            if model_info is None:
                raise ValueError(
                    f"model implementation has no registry row for {model_id}"
                )
            adapter = _PipelineAdapter(
                load_pipeline(model_info, torch.device(device))
            )
        return cls(
            model_id=model_id,
            model_root=model_root.resolve(),
            model_code_root=model_code_root.resolve(),
            device=device,
            adapter=adapter,
            load_seconds=time.monotonic() - started,
        )

    def forecast(self, children: list[dict[str, Any]]) -> list[np.ndarray]:
        if not children:
            return []
        horizons = {int(child["horizon"]) for child in children}
        if len(horizons) != 1:
            raise ValueError("one native batch must use one forecast horizon")
        horizon = horizons.pop()
        outputs = self.adapter.forecast(children)
        if len(outputs) != len(children):
            raise ValueError(
                f"native model returned {len(outputs)} rows for {len(children)} inputs"
            )
        result: list[np.ndarray] = []
        for child, output in zip(children, outputs, strict=True):
            if hasattr(output, "detach"):
                output = output.detach().cpu().numpy()
            values = np.asarray(output, dtype=np.float32)
            expected = (int(child["target_dim"]), horizon)
            if values.shape != expected:
                raise ValueError(
                    f"native forecast shape {values.shape} != {expected} for "
                    f"{child['sample_id']}"
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"native model {self.model_id} returned non-finite predictions"
                )
            result.append(values)
        return result

    def provenance(self) -> dict[str, Any]:
        import torch

        return {
            "schema_version": NATIVE_RUNTIME_SCHEMA,
            "model_id": self.model_id,
            "device": self.device,
            "model_root": str(self.model_root),
            "model_code_root": str(self.model_code_root),
            "provider": NATIVE_MODEL_SPECS[self.model_id].provider,
            "torch_version": torch.__version__,
            "package_versions": {
                name: _package_version(name)
                for name in (
                    "transformers",
                    "chronos-forecasting",
                    "timesfm",
                    "toto-2",
                    "uni2ts",
                    "tirex2",
                )
            },
        }
