from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


class BackendError(RuntimeError):
    pass


def _model_dir(repo_root: Path, env_key: str, default_name: str) -> Path:
    return Path(os.getenv(env_key, str(repo_root / "models" / default_name)))


def _as_context(context: np.ndarray) -> np.ndarray:
    arr = np.asarray(context, dtype=float)
    if arr.ndim != 1:
        raise BackendError(f"expected 1D context, got shape {arr.shape}")
    return arr


def _resolve_device() -> tuple[str, "object"]:
    try:
        import torch
    except Exception as exc:
        raise BackendError(f"PyTorch import failed: {exc}") from exc
    requested = os.getenv("TSBENCHMARK_DEVICE", "cpu").strip().lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise BackendError("TSBENCHMARK_DEVICE=cuda requested but CUDA is unavailable")
        return "cuda", torch.device("cuda:0")
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise BackendError("TSBENCHMARK_DEVICE=mps requested but MPS is unavailable")
        return "mps", torch.device("mps")
    if requested == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", torch.device("mps")
        if torch.cuda.is_available():
            return "cuda", torch.device("cuda:0")
    return "cpu", torch.device("cpu")


def _resolve_backend_device(backend_name: str) -> tuple[str, "object"]:
    device_name, torch_device = _resolve_device()
    if backend_name == "moirai_moe_base" and device_name == "mps":
        import torch

        return "cpu", torch.device("cpu")
    return device_name, torch_device


@lru_cache(maxsize=8)
def _load_timesfm(repo_root: str, max_context: int):
    model_dir = _model_dir(Path(repo_root), "TSBENCHMARK_TIMESFM_DIR", "google_timesfm_2p5_200m")
    try:
        import timesfm

        model_cls = getattr(timesfm, "TimesFM_2p5_200M_torch")
        model = model_cls._from_pretrained(model_id=str(model_dir), local_files_only=True)
        _, torch_device = _resolve_backend_device("timesfm_2_5_200m")
        model.model.device = torch_device
        model.model.device_count = 1
        model.model.to(torch_device)
        model.compile(timesfm.ForecastConfig(max_context=int(max_context), max_horizon=128, per_core_batch_size=1))
        return model
    except Exception as exc:
        raise BackendError(f"TimesFM load failed from {model_dir}: {exc}") from exc


def predict_timesfm_2_5_200m(context: np.ndarray, horizon: int, season_length: int, seed: int, repo_root: Path) -> np.ndarray:
    arr = _as_context(context)
    model = _load_timesfm(str(repo_root), int(len(arr)))
    try:
        forecast, _ = model.forecast(horizon=horizon, inputs=[arr])
        return np.asarray(forecast[0][:horizon], dtype=float)
    except Exception as exc:
        raise BackendError(f"TimesFM forecast failed: {exc}") from exc


@lru_cache(maxsize=1)
def _load_chronos(repo_root: str):
    model_dir = _model_dir(Path(repo_root), "TSBENCHMARK_CHRONOS_DIR", "amazon_chronos_bolt_base")
    try:
        from chronos import BaseChronosPipeline

        device_name, _ = _resolve_backend_device("chronos_bolt_base")
        return BaseChronosPipeline.from_pretrained(str(model_dir), device_map=device_name)
    except Exception as exc:
        raise BackendError(f"Chronos-Bolt load failed from {model_dir}: {exc}") from exc


def predict_chronos_bolt_base(context: np.ndarray, horizon: int, season_length: int, seed: int, repo_root: Path) -> np.ndarray:
    try:
        import torch

        model = _load_chronos(str(repo_root))
        _, torch_device = _resolve_backend_device("chronos_bolt_base")
        forecast = model.predict(torch.tensor(_as_context(context), dtype=torch.float32, device=torch_device), prediction_length=horizon)
        out = np.asarray(forecast, dtype=float)
        if out.ndim == 3:
            out = np.median(out, axis=1)
        if out.ndim == 2:
            out = out[0]
        return out[:horizon]
    except Exception as exc:
        raise BackendError(f"Chronos-Bolt predict failed: {exc}") from exc


@lru_cache(maxsize=1)
def _load_sundial(repo_root: str):
    model_dir = _model_dir(Path(repo_root), "TSBENCHMARK_SUNDIAL_DIR", "thuml_sundial_base_128m")
    try:
        from transformers import AutoModelForCausalLM

        _, torch_device = _resolve_backend_device("sundial_base_128m")
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True)
        return model.to(torch_device)
    except Exception as exc:
        raise BackendError(f"Sundial load failed from {model_dir}: {exc}") from exc


def predict_sundial_base_128m(context: np.ndarray, horizon: int, season_length: int, seed: int, repo_root: Path) -> np.ndarray:
    try:
        import torch

        model = _load_sundial(str(repo_root))
        _, torch_device = _resolve_backend_device("sundial_base_128m")
        forecast = model.generate(torch.tensor(_as_context(context)[None, :], dtype=torch.float32, device=torch_device), max_new_tokens=horizon, num_samples=5)
        out = np.asarray(forecast, dtype=float)
        if out.ndim == 3:
            out = np.median(out, axis=1)
        if out.ndim >= 2:
            out = out[0]
        return out[:horizon]
    except Exception as exc:
        raise BackendError(f"Sundial forecast failed: {exc}") from exc


def predict_moirai_moe_base(context: np.ndarray, horizon: int, season_length: int, seed: int, repo_root: Path) -> np.ndarray:
    try:
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from gluonts.evaluation import make_evaluation_predictions
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        model_dir = _model_dir(repo_root, "TSBENCHMARK_MOIRAI_DIR", "salesforce_moirai_moe_base")
        module = MoiraiModule.from_pretrained(str(model_dir), local_files_only=True)
        arr = _as_context(context)
        device_name, torch_device = _resolve_backend_device("moirai_moe_base")
        model = MoiraiForecast(
            prediction_length=int(horizon),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=int(min(len(arr), 512)),
            module=module,
            patch_size="auto",
            num_samples=20,
        ).to(torch_device)
        predictor = model.create_predictor(batch_size=1, device=device_name)
        dataset = ListDataset([{"start": pd.Period("2020-01-01 00:00", freq="h"), "target": arr.astype(np.float32)}], freq="h")
        forecast_it, _ = make_evaluation_predictions(dataset=dataset, predictor=predictor, num_samples=20)
        return np.median(np.asarray(next(forecast_it).samples, dtype=float), axis=0)[:horizon]
    except Exception as exc:
        raise BackendError(f"Moirai-MoE predict failed: {exc}") from exc


def predict_lag_llama(context: np.ndarray, horizon: int, season_length: int, seed: int, repo_root: Path) -> np.ndarray:
    try:
        import pandas as pd
        import torch
        from gluonts.dataset.common import ListDataset
        from gluonts.evaluation import make_evaluation_predictions
        from lag_llama.gluon.estimator import LagLlamaEstimator

        ckpt_path = _model_dir(repo_root, "TSBENCHMARK_LAG_LLAMA_DIR", "lag_llama") / "lag-llama.ckpt"
        if not ckpt_path.exists():
            raise BackendError(f"Lag-Llama checkpoint not found at {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        hparams = checkpoint["hyper_parameters"]
        model_kwargs = hparams["model_kwargs"]
        arr = _as_context(context)
        context_length = int(min(max(32, len(arr)), 512))
        rope_factor = float(max(1.0, context_length / max(1, int(hparams.get("context_length", 32)))))
        estimator = LagLlamaEstimator(
            prediction_length=int(horizon),
            context_length=context_length,
            input_size=int(model_kwargs.get("input_size", 1)),
            n_layer=int(model_kwargs["n_layer"]),
            n_embd_per_head=int(model_kwargs["n_embd_per_head"]),
            n_head=int(model_kwargs["n_head"]),
            max_context_length=int(model_kwargs.get("max_context_length", 2048)),
            rope_scaling={"type": "linear", "factor": rope_factor},
            scaling=str(model_kwargs.get("scaling", "robust")),
            batch_size=1,
            num_parallel_samples=20,
            time_feat=bool(model_kwargs.get("time_feat", True)),
            dropout=float(model_kwargs.get("dropout", 0.0)),
            lags_seq=["Q", "M", "W", "D", "H", "T", "S"],
            ckpt_path=str(ckpt_path),
            device=_resolve_backend_device("lag_llama")[1],
        )
        predictor = estimator.create_predictor(estimator.create_transformation(), estimator.create_lightning_module())
        dataset = ListDataset([{"start": pd.Period("2020-01-01 00:00", freq="h"), "target": arr.astype(np.float32)}], freq="h")
        forecast_it, _ = make_evaluation_predictions(dataset=dataset, predictor=predictor, num_samples=20)
        return np.median(np.asarray(next(forecast_it).samples, dtype=float), axis=0)[:horizon]
    except Exception as exc:
        raise BackendError(f"Lag-Llama predict failed: {exc}") from exc


BACKENDS = {
    "timesfm_2_5_200m": predict_timesfm_2_5_200m,
    "chronos_bolt_base": predict_chronos_bolt_base,
    "sundial_base_128m": predict_sundial_base_128m,
    "moirai_moe_base": predict_moirai_moe_base,
    "lag_llama": predict_lag_llama,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    backend = payload["backend"]
    if backend not in BACKENDS:
        raise BackendError(f"unknown backend {backend}")
    repo_root = Path(payload["repo_root"])
    mode = payload.get("mode", "single")
    if mode == "single":
        forecast = BACKENDS[backend](
            context=np.asarray(payload["context"], dtype=float),
            horizon=int(payload["horizon"]),
            season_length=int(payload["season_length"]),
            seed=int(payload["seed"]),
            repo_root=repo_root,
        )
        output = {"forecast": np.asarray(forecast, dtype=float).tolist()}
    elif mode == "batch":
        forecasts = []
        rows = payload["rows"]
        total = len(rows)
        print(f"[{backend}] batch start total={total}", file=sys.stderr, flush=True)
        for index, row in enumerate(rows, start=1):
            forecast = BACKENDS[backend](
                context=np.asarray(row["context"], dtype=float),
                horizon=int(row["horizon"]),
                season_length=int(row["season_length"]),
                seed=int(payload["seed"]),
                repo_root=repo_root,
            )
            forecasts.append(np.asarray(forecast, dtype=float).tolist())
            if index == 1 or index % 100 == 0 or index == total:
                print(f"[{backend}] progress {index}/{total}", file=sys.stderr, flush=True)
        output = {"forecasts": forecasts}
    else:
        raise BackendError(f"unknown mode {mode}")
    Path(args.output).write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(str(exc))

