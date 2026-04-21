from __future__ import annotations

import importlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .baselines import baseline_by_name


class AdapterError(RuntimeError):
    """Raised when a benchmark v1 model adapter cannot run in this environment."""


@dataclass(slots=True)
class ModelAdapter:
    name: str

    def predict(self, context: np.ndarray, horizon: int, season_length: int, seed: int) -> np.ndarray:
        raise NotImplementedError

    def predict_many(self, rows: list[dict[str, object]], seed: int) -> list[np.ndarray]:
        return [
            self.predict(
                context=np.asarray(row["context"], dtype=float),
                horizon=int(row["horizon"]),
                season_length=int(row["season_length"]),
                seed=seed,
            )
            for row in rows
        ]


class BaselineAdapter(ModelAdapter):
    def predict(self, context: np.ndarray, horizon: int, season_length: int, seed: int) -> np.ndarray:
        return baseline_by_name(self.name).predict(context, horizon, season_length)


class ExternalCallableAdapter(ModelAdapter):
    def __init__(self, name: str, env_key: str) -> None:
        super().__init__(name=name)
        self.env_key = env_key

    def predict(self, context: np.ndarray, horizon: int, season_length: int, seed: int) -> np.ndarray:
        target = os.getenv(self.env_key)
        if not target:
            raise AdapterError(f"{self.name} requires env var {self.env_key}=module:function")
        module_name, func_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        return np.asarray(func(np.asarray(context, dtype=float), int(horizon), int(season_length), int(seed)), dtype=float)


class ExternalPythonAdapter(ModelAdapter):
    def __init__(self, name: str, backend_name: str) -> None:
        super().__init__(name=name)
        self.backend_name = backend_name

    def _run_backend(self, payload: dict[str, object]) -> dict[str, object]:
        python_path = Path(os.getenv("TSBENCHMARK_MODEL_PYTHON", ".venv-models/bin/python"))
        if not python_path.exists():
            raise AdapterError(
                f"{self.name} requires a model environment. Expected interpreter at {python_path} "
                "or set TSBENCHMARK_MODEL_PYTHON."
            )
        repo_root = Path(__file__).resolve().parents[4]
        payload = {"backend": self.backend_name, "repo_root": str(repo_root), **payload}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload, handle)
            input_path = Path(handle.name)
        output_path = input_path.with_suffix(".out.json")
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else str(repo_root)
        if env.get("TSBENCHMARK_DEVICE", "auto").strip().lower() in {"auto", "mps"}:
            env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        cmd = [
            str(python_path),
            "-m",
            "backend.app.datasets.benchmark_v1.model_backends",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        try:
            completed = subprocess.run(cmd, env=env, text=True, check=False)
        finally:
            input_path.unlink(missing_ok=True)
        if completed.returncode != 0:
            raise AdapterError(f"{self.name} backend failed with exit code {completed.returncode}")
        if not output_path.exists():
            raise AdapterError(f"{self.name} backend did not produce output at {output_path}")
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        finally:
            output_path.unlink(missing_ok=True)

    def predict(self, context: np.ndarray, horizon: int, season_length: int, seed: int) -> np.ndarray:
        result = self._run_backend(
            {
                "mode": "single",
                "context": np.asarray(context, dtype=float).tolist(),
                "horizon": int(horizon),
                "season_length": int(season_length),
                "seed": int(seed),
            }
        )
        return np.asarray(result["forecast"], dtype=float)

    def predict_many(self, rows: list[dict[str, object]], seed: int) -> list[np.ndarray]:
        result = self._run_backend(
            {
                "mode": "batch",
                "seed": int(seed),
                "rows": [
                    {
                        "context": np.asarray(row["context"], dtype=float).tolist(),
                        "horizon": int(row["horizon"]),
                        "season_length": int(row["season_length"]),
                    }
                    for row in rows
                ],
            }
        )
        return [np.asarray(forecast, dtype=float) for forecast in result["forecasts"]]


def build_model_adapter(model_name: str) -> ModelAdapter:
    baseline_names = {"last_value", "seasonal_naive", "auto_theta", "ridge_ar"}
    if model_name in baseline_names:
        return BaselineAdapter(name=model_name)
    explicit_callables = {
        "timesfm_2_5_200m": "TSBENCHMARK_TIMESFM_CALLABLE",
        "chronos_bolt_base": "TSBENCHMARK_CHRONOS_CALLABLE",
        "sundial_base_128m": "TSBENCHMARK_SUNDIAL_CALLABLE",
        "moirai_moe_base": "TSBENCHMARK_MOIRAI_CALLABLE",
        "lag_llama": "TSBENCHMARK_LAG_LLAMA_CALLABLE",
    }
    if explicit_callables.get(model_name) and os.getenv(explicit_callables[model_name]):
        return ExternalCallableAdapter(name=model_name, env_key=explicit_callables[model_name])
    builtin_backends = {
        "timesfm_2_5_200m": "timesfm_2_5_200m",
        "chronos_bolt_base": "chronos_bolt_base",
        "sundial_base_128m": "sundial_base_128m",
        "moirai_moe_base": "moirai_moe_base",
        "lag_llama": "lag_llama",
    }
    if model_name in builtin_backends:
        return ExternalPythonAdapter(name=model_name, backend_name=builtin_backends[model_name])
    raise KeyError(f"unknown model: {model_name}")


def available_models() -> list[str]:
    return [
        "last_value",
        "seasonal_naive",
        "auto_theta",
        "ridge_ar",
        "timesfm_2_5_200m",
        "chronos_bolt_base",
        "sundial_base_128m",
        "moirai_moe_base",
        "lag_llama",
    ]
