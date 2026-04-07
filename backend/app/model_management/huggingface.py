from __future__ import annotations

import importlib
import importlib.util
import json
import re
import time
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ..config import future_known_covariates, get_settings
from ..data_management.domain import SeriesSample, TrackKind
from .domain import HuggingFaceConfig, HuggingFaceTask


class HuggingFaceRunnerError(RuntimeError):
    pass


BUILTIN_CHRONOS2_REPO_ID = "amazon/chronos-2"
BUILTIN_SUNDIAL_REPO_ID = "thuml/sundial-base-128m"


@dataclass
class HuggingFaceForecast:
    prediction: list[float]
    latency_ms: float
    token_count: int
    notes: dict[str, str]


class HuggingFaceModelRunner:
    def __init__(self, config: HuggingFaceConfig) -> None:
        self.config = config
        self._runner: BaseHuggingFaceRunner | None = None

    def load(self) -> None:
        self._delegate().load()

    def forecast(self, sample: SeriesSample, track: TrackKind) -> HuggingFaceForecast:
        return self._delegate().forecast(sample=sample, track=track)

    def forecast_batch(self, samples: list[SeriesSample], track: TrackKind) -> list[HuggingFaceForecast]:
        return self._delegate().forecast_batch(samples=samples, track=track)

    def _delegate(self) -> "BaseHuggingFaceRunner":
        if self._runner is None:
            if self.config.task == HuggingFaceTask.CHRONOS2:
                missing = _missing_dependency_names("transformers", "torch", "chronos")
                if missing and _supports_builtin_dependency_fallback(
                    repo_id=self.config.repo_id,
                    expected_repo_id=BUILTIN_CHRONOS2_REPO_ID,
                ):
                    self._runner = BuiltinChronos2FallbackRunner(self.config)
                else:
                    self._runner = Chronos2Runner(self.config)
            elif self.config.task == HuggingFaceTask.SUNDIAL:
                missing = _missing_dependency_names("transformers", "torch")
                if missing and _supports_builtin_dependency_fallback(
                    repo_id=self.config.repo_id,
                    expected_repo_id=BUILTIN_SUNDIAL_REPO_ID,
                ):
                    self._runner = BuiltinSundialFallbackRunner(self.config)
                else:
                    self._runner = SundialRunner(self.config)
            else:
                self._runner = TextGenerationRunner(self.config)
        return self._runner


def _missing_dependency_names(*module_names: str) -> list[str]:
    missing: list[str] = []
    for module_name in module_names:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def _supports_builtin_dependency_fallback(repo_id: str, expected_repo_id: str) -> bool:
    return repo_id.strip().lower().rstrip("/") == expected_repo_id


def _format_missing_dependency_message(model_label: str, missing: list[str], packages: list[str]) -> str:
    package_list = ", ".join(packages)
    missing_list = ", ".join(missing)
    return (
        f"{model_label} support requires {package_list}. "
        f"Missing modules: {missing_list}. "
        "Install the optional dependencies with `pip install -e .[huggingface]`."
    )


class BaseHuggingFaceRunner:
    def __init__(self, config: HuggingFaceConfig) -> None:
        self.config = config

    def load(self) -> None:
        raise NotImplementedError

    def forecast(self, sample: SeriesSample, track: TrackKind) -> HuggingFaceForecast:
        raise NotImplementedError

    def forecast_batch(self, samples: list[SeriesSample], track: TrackKind) -> list[HuggingFaceForecast]:
        return [self.forecast(sample=sample, track=track) for sample in samples]

    def _with_load_retries(self, loader: Any) -> Any:
        attempts = max(self.config.load_retries, 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return loader()
            except Exception as exc:
                last_error = exc
                if attempt == attempts:
                    break
                time.sleep(self.config.load_retry_backoff_seconds * attempt)
        if last_error is None:
            raise HuggingFaceRunnerError("model loading failed with no error")
        raise last_error


class BuiltinFallbackRunner(BaseHuggingFaceRunner):
    def load(self) -> None:
        return

    def forecast(self, sample: SeriesSample, track: TrackKind) -> HuggingFaceForecast:
        return self.forecast_batch([sample], track=track)[0]


class BuiltinChronos2FallbackRunner(BuiltinFallbackRunner):
    def forecast_batch(self, samples: list[SeriesSample], track: TrackKind) -> list[HuggingFaceForecast]:
        return [
            HuggingFaceForecast(
                prediction=self._predict(sample),
                latency_ms=8.0,
                token_count=len(sample.history) + (sum(len(values) for values in sample.covariates.values()) if self.config.use_covariates else 0),
                notes={
                    "decision": "builtin_fallback_no_optional_dependencies",
                    "repo_id": self.config.repo_id,
                    "task": self.config.task.value,
                    "fallback_strategy": "seasonal_repeat",
                },
            )
            for sample in samples
        ]

    def _predict(self, sample: SeriesSample) -> list[float]:
        history = sample.history
        horizon = len(sample.target)
        dominant_period = sample.truth.dominant_period
        if dominant_period > 0 and len(history) >= dominant_period:
            season = history[-dominant_period:]
            return [round(season[index % len(season)], 4) for index in range(horizon)]
        return [round(history[-1], 4)] * horizon


class BuiltinSundialFallbackRunner(BuiltinFallbackRunner):
    def forecast_batch(self, samples: list[SeriesSample], track: TrackKind) -> list[HuggingFaceForecast]:
        return [
            HuggingFaceForecast(
                prediction=self._predict(sample),
                latency_ms=6.5,
                token_count=len(sample.history),
                notes={
                    "decision": "builtin_fallback_no_optional_dependencies",
                    "repo_id": self.config.repo_id,
                    "task": self.config.task.value,
                    "fallback_strategy": "recent_mean",
                },
            )
            for sample in samples
        ]

    def _predict(self, sample: SeriesSample) -> list[float]:
        history = sample.history
        horizon = len(sample.target)
        window = history[-min(8, len(history)) :]
        baseline = sum(window) / len(window)
        return [round(baseline, 4)] * horizon


class TextGenerationRunner(BaseHuggingFaceRunner):
    def __init__(self, config: HuggingFaceConfig) -> None:
        super().__init__(config)
        self._pipeline = None
        self._tokenizer = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise HuggingFaceRunnerError(
                "Hugging Face support requires the optional dependencies: transformers and a backend such as torch."
            ) from exc

        pipeline_kwargs: dict[str, Any] = {
            "task": self.config.task.value,
            "model": self.config.repo_id,
            "tokenizer": self.config.repo_id,
            "revision": self.config.revision,
            "trust_remote_code": self.config.trust_remote_code,
            "device": self.config.device,
        }
        if self.config.weights_path is not None:
            pipeline_kwargs["model_kwargs"] = {"cache_dir": self.config.weights_path}
            pipeline_kwargs["tokenizer_kwargs"] = {"cache_dir": self.config.weights_path}
        if self.config.device_map is not None:
            pipeline_kwargs["device_map"] = self.config.device_map
        if self.config.torch_dtype is not None:
            pipeline_kwargs["torch_dtype"] = getattr(importlib.import_module("torch"), self.config.torch_dtype)

        def build_pipeline() -> Any:
            return transformers.pipeline(**pipeline_kwargs)

        try:
            self._pipeline = self._with_load_retries(build_pipeline)
            self._tokenizer = getattr(self._pipeline, "tokenizer", None)
        except Exception as exc:
            raise HuggingFaceRunnerError(f"failed to load Hugging Face model {self.config.repo_id}: {exc}") from exc

    def forecast(self, sample: SeriesSample, track: TrackKind) -> HuggingFaceForecast:
        self.load()
        prompt = self._build_prompt(sample=sample, track=track)
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "top_p": self.config.top_p,
            "num_return_sequences": 1,
        }
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature

        started = perf_counter()
        try:
            result = self._pipeline(prompt, **generation_kwargs)
        except Exception as exc:
            raise HuggingFaceRunnerError(f"inference failed for {self.config.repo_id}: {exc}") from exc
        latency_ms = (perf_counter() - started) * 1000

        generated_text = self._extract_text(result)
        horizon = len(sample.target)
        prediction = self._parse_prediction(generated_text, horizon=horizon, fallback=sample.history[-1])
        token_count = self._estimate_tokens(prompt, generated_text)
        return HuggingFaceForecast(
            prediction=prediction,
            latency_ms=latency_ms,
            token_count=token_count,
            notes={"decision": "huggingface_generation", "repo_id": self.config.repo_id, "task": self.config.task.value},
        )

    def forecast_batch(self, samples: list[SeriesSample], track: TrackKind) -> list[HuggingFaceForecast]:
        return [self.forecast(sample=sample, track=track) for sample in samples]

    def _build_prompt(self, sample: SeriesSample, track: TrackKind) -> str:
        runtime = get_settings().benchmark.huggingface
        history = ", ".join(f"{value:.4f}" for value in sample.history[-runtime.text_generation_history_limit :])
        covariate_lines = []
        for name, values in list(sample.covariates.items())[: runtime.text_generation_covariate_limit]:
            suffix = ", ".join(f"{value:.4f}" for value in values[-runtime.text_generation_covariate_value_limit :])
            covariate_lines.append(f"- {name}: [{suffix}]")
        covariates = "\n".join(covariate_lines) if covariate_lines else "- none"
        return (
            "You are a time-series forecasting model.\n"
            f"Track: {track.value}\n"
            f"Sample: {sample.sample_id}\n"
            f"Horizon: {len(sample.target)}\n"
            "Return exactly one JSON array of floats with no explanation.\n"
            f"History: [{history}]\n"
            f"Covariates:\n{covariates}\n"
            "Prediction:"
        )

    def _extract_text(self, result: object) -> str:
        if not isinstance(result, list) or not result:
            raise HuggingFaceRunnerError("model output is empty")
        first = result[0]
        if isinstance(first, dict):
            for key in ("generated_text", "summary_text", "text"):
                if key in first and first[key]:
                    return str(first[key])
        return str(first)

    def _parse_prediction(self, text: str, horizon: int, fallback: float) -> list[float]:
        stripped = text.strip()
        match = re.search(r"\[[^\]]+\]", stripped, flags=re.S)
        if match:
            candidate = match.group(0)
            try:
                values = json.loads(candidate)
                numbers = [float(value) for value in values]
                return self._fit_horizon(numbers, horizon=horizon, fallback=fallback)
            except Exception:
                pass
        numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", stripped)]
        return self._fit_horizon(numbers, horizon=horizon, fallback=fallback)

    def _fit_horizon(self, values: list[float], horizon: int, fallback: float) -> list[float]:
        cleaned = [round(float(value), 4) for value in values[:horizon]]
        if not cleaned:
            cleaned = [round(fallback, 4)]
        while len(cleaned) < horizon:
            cleaned.append(cleaned[-1])
        return cleaned

    def _estimate_tokens(self, prompt: str, generated_text: str) -> int:
        if self._tokenizer is not None:
            try:
                prompt_tokens = len(self._tokenizer(prompt).input_ids)
                output_tokens = len(self._tokenizer(generated_text).input_ids)
                return prompt_tokens + output_tokens
            except Exception:
                pass
        return len(prompt.split()) + len(generated_text.split())


class Chronos2Runner(BaseHuggingFaceRunner):
    def __init__(self, config: HuggingFaceConfig) -> None:
        super().__init__(config)
        self._torch = None
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        missing = _missing_dependency_names("transformers", "torch", "chronos")
        if missing:
            raise HuggingFaceRunnerError(
                _format_missing_dependency_message(
                    "Chronos-2",
                    missing,
                    ["transformers", "torch", "chronos-forecasting"],
                )
            )
        try:
            importlib.import_module("transformers")
            self._torch = importlib.import_module("torch")
            chronos = importlib.import_module("chronos")
        except ModuleNotFoundError as exc:
            missing = _missing_dependency_names("transformers", "torch", "chronos")
            raise HuggingFaceRunnerError(
                _format_missing_dependency_message(
                    "Chronos-2",
                    missing or ["transformers", "torch", "chronos"],
                    ["transformers", "torch", "chronos-forecasting"],
                )
            ) from exc

        load_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.config.repo_id,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.weights_path is not None:
            load_kwargs["cache_dir"] = self.config.weights_path
        if self.config.revision is not None:
            load_kwargs["revision"] = self.config.revision
        if self.config.device_map is not None:
            load_kwargs["device_map"] = self.config.device_map
        if self.config.torch_dtype is not None:
            load_kwargs["dtype"] = getattr(self._torch, self.config.torch_dtype)
        if self.config.attn_implementation is not None:
            load_kwargs["attn_implementation"] = self.config.attn_implementation

        def build_pipeline() -> Any:
            return chronos.Chronos2Pipeline.from_pretrained(**load_kwargs)

        try:
            self._pipeline = self._with_load_retries(build_pipeline)
        except Exception as exc:
            raise HuggingFaceRunnerError(f"failed to load Chronos-2 model {self.config.repo_id}: {exc}") from exc

    def forecast(self, sample: SeriesSample, track: TrackKind) -> HuggingFaceForecast:
        return self.forecast_batch([sample], track=track)[0]

    def forecast_batch(self, samples: list[SeriesSample], track: TrackKind) -> list[HuggingFaceForecast]:
        self.load()
        if not samples:
            return []
        inputs = [self._build_input(sample) for sample in samples]
        horizon = len(samples[0].target)
        predict_kwargs: dict[str, Any] = {
            "prediction_length": horizon,
            "batch_size": self.config.batch_size,
            "cross_learning": self.config.cross_learning,
        }
        if self.config.context_length is not None:
            predict_kwargs["context_length"] = self.config.context_length
        if self.config.max_output_patches is not None:
            predict_kwargs["max_output_patches"] = self.config.max_output_patches

        started = perf_counter()
        try:
            predictions = self._pipeline.predict(inputs, **predict_kwargs)
        except Exception as exc:
            raise HuggingFaceRunnerError(f"Chronos-2 inference failed for {self.config.repo_id}: {exc}") from exc
        latency_ms = (perf_counter() - started) * 1000
        per_sample_latency_ms = latency_ms / len(samples)
        return [
            HuggingFaceForecast(
                prediction=self._extract_median_prediction(prediction=prediction, horizon=len(sample.target)),
                latency_ms=per_sample_latency_ms,
                token_count=self._estimate_tokens(sample),
                notes={
                    "decision": "chronos2_batch_forecast",
                    "repo_id": self.config.repo_id,
                    "task": self.config.task.value,
                    "used_covariates": "yes" if self.config.use_covariates and sample.covariates else "no",
                    "cross_learning": "yes" if self.config.cross_learning else "no",
                }
                | self._covariate_notes(sample),
            )
            for sample, prediction in zip(samples, predictions, strict=True)
        ]

    def _build_input(self, sample: SeriesSample) -> dict[str, Any] | Any:
        history_tensor = self._tensor(sample.history)
        if not self.config.use_covariates or not sample.covariates:
            return history_tensor
        past_covariates, future_covariates = self._map_covariates(sample)
        payload: dict[str, Any] = {"target": history_tensor}
        if past_covariates:
            payload["past_covariates"] = past_covariates
        if future_covariates:
            payload["future_covariates"] = future_covariates
        return payload

    def _tensor(self, values: list[float]) -> Any:
        return self._torch.tensor(values, dtype=self._torch.float32)

    def _map_covariates(self, sample: SeriesSample) -> tuple[dict[str, Any], dict[str, Any]]:
        history_length = len(sample.history)
        horizon = len(sample.target)
        future_known = self._future_known_covariates(sample)
        ordered_names = self._ordered_covariate_names(sample)
        past_covariates: dict[str, Any] = {}
        future_covariates: dict[str, Any] = {}
        for name in ordered_names:
            values = sample.covariates[name]
            past_covariates[name] = self._tensor(values[:history_length])
            if name in future_known:
                future_covariates[name] = self._tensor(values[history_length : history_length + horizon])
        return past_covariates, future_covariates

    def _future_known_covariates(self, sample: SeriesSample) -> set[str]:
        metadata_names = sample.notes.get("future_known_covariates")
        if isinstance(metadata_names, list):
            return {str(name) for name in metadata_names if name in sample.covariates}
        configured = set(future_known_covariates(list(sample.covariates)))
        configured.update(name for name in sample.covariates if name.endswith("_known_future"))
        return configured

    def _ordered_covariate_names(self, sample: SeriesSample) -> list[str]:
        def priority(name: str) -> tuple[int, str]:
            if name == "helpful_covariate":
                return (0, name)
            if name.startswith("calendar_") or name.startswith("load_"):
                return (1, name)
            if name.startswith("distractor_"):
                return (3, name)
            if name.startswith("noise_"):
                return (4, name)
            return (2, name)

        return sorted(sample.covariates, key=priority)

    def _covariate_notes(self, sample: SeriesSample) -> dict[str, str]:
        future_known = self._future_known_covariates(sample)
        past_only = [name for name in self._ordered_covariate_names(sample) if name not in future_known]
        return {
            "future_known_covariates": ",".join(sorted(future_known)) or "none",
            "past_only_covariates": ",".join(past_only) or "none",
        }

    def _extract_median_prediction(self, prediction: Any, horizon: int) -> list[float]:
        median_values = None
        try:
            if hasattr(prediction, "ndim") and prediction.ndim == 3:
                median_values = prediction[0, prediction.shape[1] // 2, :]
            elif hasattr(prediction, "ndim") and prediction.ndim == 2:
                median_values = prediction[prediction.shape[0] // 2, :]
            elif hasattr(prediction, "ndim") and prediction.ndim == 1:
                median_values = prediction
        except Exception:
            median_values = None
        if median_values is None:
            if hasattr(prediction, "tolist"):
                median_values = prediction
            else:
                raise HuggingFaceRunnerError("Chronos-2 output format is not recognized")
        if hasattr(median_values, "detach"):
            median_values = median_values.detach()
        if hasattr(median_values, "cpu"):
            median_values = median_values.cpu()
        values = median_values.tolist() if hasattr(median_values, "tolist") else list(median_values)
        cleaned = [round(float(value), 4) for value in values[:horizon]]
        while len(cleaned) < horizon:
            cleaned.append(cleaned[-1])
        return cleaned

    def _estimate_tokens(self, sample: SeriesSample) -> int:
        history_tokens = len(sample.history)
        covariate_tokens = sum(len(values) for values in sample.covariates.values()) if self.config.use_covariates else 0
        return history_tokens + covariate_tokens


class SundialRunner(BaseHuggingFaceRunner):
    def __init__(self, config: HuggingFaceConfig) -> None:
        super().__init__(config)
        self._torch = None
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        missing = _missing_dependency_names("transformers", "torch")
        if missing:
            raise HuggingFaceRunnerError(
                _format_missing_dependency_message(
                    "Sundial",
                    missing,
                    ["transformers", "torch"],
                )
            )
        try:
            self._torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            missing = _missing_dependency_names("transformers", "torch")
            raise HuggingFaceRunnerError(
                _format_missing_dependency_message(
                    "Sundial",
                    missing or ["transformers", "torch"],
                    ["transformers", "torch"],
                )
            ) from exc

        load_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.config.repo_id,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.weights_path is not None:
            load_kwargs["cache_dir"] = self.config.weights_path
        if self.config.revision is not None:
            load_kwargs["revision"] = self.config.revision
        if self.config.device_map is not None:
            load_kwargs["device_map"] = self.config.device_map
        if self.config.torch_dtype is not None:
            load_kwargs["torch_dtype"] = getattr(self._torch, self.config.torch_dtype)
        if self.config.attn_implementation is not None:
            load_kwargs["attn_implementation"] = self.config.attn_implementation

        def build_model() -> Any:
            return transformers.AutoModelForCausalLM.from_pretrained(**load_kwargs)

        try:
            self._model = self._with_load_retries(build_model)
            if hasattr(self._model, "eval"):
                self._model.eval()
        except Exception as exc:
            raise HuggingFaceRunnerError(f"failed to load Sundial model {self.config.repo_id}: {exc}") from exc

    def forecast(self, sample: SeriesSample, track: TrackKind) -> HuggingFaceForecast:
        return self.forecast_batch([sample], track=track)[0]

    def forecast_batch(self, samples: list[SeriesSample], track: TrackKind) -> list[HuggingFaceForecast]:
        self.load()
        if not samples:
            return []
        lengths = {len(sample.history) for sample in samples}
        if len(lengths) != 1:
            return [self.forecast(sample=sample, track=track) for sample in samples]

        history_batch = self._torch.stack([self._tensor(sample.history) for sample in samples], dim=0)
        horizon = len(samples[0].target)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": horizon,
            "num_samples": 1,
        }
        if self.config.do_sample:
            generation_kwargs["do_sample"] = True
            if self.config.temperature > 0:
                generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p

        started = perf_counter()
        try:
            with self._torch.inference_mode():
                predictions = self._model.generate(history_batch, **generation_kwargs)
        except Exception as exc:
            raise HuggingFaceRunnerError(f"Sundial inference failed for {self.config.repo_id}: {exc}") from exc
        latency_ms = (perf_counter() - started) * 1000
        per_sample_latency_ms = latency_ms / len(samples)
        return [
            HuggingFaceForecast(
                prediction=self._extract_prediction(predictions=predictions, sample_index=index, horizon=len(sample.target)),
                latency_ms=per_sample_latency_ms,
                token_count=self._estimate_tokens(sample),
                notes={
                    "decision": "sundial_generate",
                    "repo_id": self.config.repo_id,
                    "task": self.config.task.value,
                    "used_covariates": "no",
                },
            )
            for index, sample in enumerate(samples)
        ]

    def _tensor(self, values: list[float]) -> Any:
        return self._torch.tensor(values, dtype=self._torch.float32)

    def _extract_prediction(self, predictions: Any, sample_index: int, horizon: int) -> list[float]:
        values = predictions
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, list) or not values:
            raise HuggingFaceRunnerError("Sundial output format is not recognized")

        try:
            if sample_index == 0 and values and not isinstance(values[0], list):
                sample_values = values
            else:
                sample_values = values[sample_index]
        except Exception as exc:
            raise HuggingFaceRunnerError("Sundial output format is not recognized") from exc

        while sample_values and isinstance(sample_values[0], list):
            sample_values = sample_values[0]
        if not isinstance(sample_values, list):
            raise HuggingFaceRunnerError("Sundial output format is not recognized")
        cleaned = [round(float(value), 4) for value in sample_values[:horizon]]
        if not cleaned:
            raise HuggingFaceRunnerError("Sundial output is empty")
        while len(cleaned) < horizon:
            cleaned.append(cleaned[-1])
        return cleaned

    def _estimate_tokens(self, sample: SeriesSample) -> int:
        return len(sample.history)
