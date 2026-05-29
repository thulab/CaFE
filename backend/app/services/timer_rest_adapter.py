"""通过 timer-rest-service 的 REST API 执行推理的模型适配器。

契约见 docs/reference/rest-api.md（POST /ai/api/v1/forecast）。
本地无真实服务时，可启动桩服务（backend/stub_service）并将
TSBENCHMARK_TIMER_SERVICE_BASE_URL 指向它。
"""
from __future__ import annotations

import time

import httpx


class TimerServiceError(RuntimeError):
    """timer-rest-service 调用失败（非 200 或响应不符合契约）。"""


class TimerRestAdapter:
    """实现 ModelAdapter 协议：把内部 sample 转成 /forecast 请求并解析回预测。"""

    _TIME_COL = "time"

    def __init__(
        self,
        base_url: str,
        api_prefix: str = "/ai/api/v1",
        client: httpx.Client | None = None,
    ):
        self._base = base_url.rstrip("/") + "/" + api_prefix.strip("/")
        self._client = client

    def forecast(self, sample: dict, model: dict, timeout_seconds: int) -> list[list[float]]:
        horizon: int = sample.get("horizon") if sample.get("horizon") is not None else len(sample["target_future"])
        body = self._build_request(sample, model, horizon=horizon)
        try:
            response = self._post(f"{self._base}/forecast", body, timeout_seconds)
        except httpx.HTTPError as exc:
            raise TimerServiceError(f"forecast request failed: {exc}") from exc
        return self._parse_response(response, horizon=horizon)

    def list_models(self, timeout_seconds: int = 10) -> list[dict]:
        """GET /models/list → 返回模型列表（data.models）。"""
        try:
            response = self._get(f"{self._base}/models/list", timeout_seconds)
        except httpx.HTTPError as exc:
            raise TimerServiceError(f"models/list request failed: {exc}") from exc
        try:
            return response["data"]["models"]
        except (KeyError, TypeError) as exc:
            raise TimerServiceError(f"unexpected models/list response shape: {response}") from exc

    def load_model(self, model_id: str, timeout_seconds: int = 600, replicas_per_device: int | None = None) -> dict:
        """POST /models/load → 加载已注册模型；timer-rest-service 端保证幂等。"""
        body: dict = {"model_id": model_id}
        if replicas_per_device is not None:
            body["replicas_per_device"] = replicas_per_device
        try:
            response = self._post(f"{self._base}/models/load", body, timeout_seconds)
        except httpx.HTTPError as exc:
            raise TimerServiceError(f"models/load request failed: {exc}") from exc
        try:
            return response["data"] or {}
        except (KeyError, TypeError) as exc:
            raise TimerServiceError(f"unexpected models/load response shape: {response}") from exc

    def unload_model(self, model: dict, timeout_seconds: int = 600) -> None:
        """POST /models/unload → 卸载一个模型；已卸载视为成功。"""
        model_id = model.get("remote_model_id") or model.get("model_id")
        if not model_id:
            raise TimerServiceError("model_id is required before unloading model")
        try:
            self._post(f"{self._base}/models/unload", {"model_id": str(model_id)}, timeout_seconds)
        except TimerServiceError as exc:
            if self._is_already_unloaded_error(exc):
                return
            raise
        except httpx.HTTPError as exc:
            raise TimerServiceError(f"models/unload request failed: {exc}") from exc

    def unload_all_models(self, timeout_seconds: int = 600) -> None:
        """卸载 timer-rest-service 当前已加载的全部模型。"""
        errors: list[str] = []
        for service_model in self.list_models(timeout_seconds=timeout_seconds):
            if service_model.get("loaded"):
                model_id = str(service_model.get("model_id") or "")
                try:
                    self.unload_model(
                        {"model_id": model_id, "remote_model_id": model_id},
                        timeout_seconds=timeout_seconds,
                    )
                except TimerServiceError as exc:
                    errors.append(f"{model_id}: {exc}")
        if errors:
            raise TimerServiceError("failed to unload all loaded models: " + "; ".join(errors))

    def ensure_model_loaded(self, model: dict, timeout_seconds: int = 600) -> None:
        """确保 forecast 前模型已加载；未加载则调用 /models/load。"""
        model_id = model.get("remote_model_id") or model.get("model_id")
        if not model_id:
            raise TimerServiceError("model_id is required before loading model")
        model_id = str(model_id)
        deadline = time.monotonic() + timeout_seconds
        service_model = self._find_service_model(model_id)
        if service_model.get("loaded"):
            return
        remaining = max(1, int(deadline - time.monotonic()))
        self.load_model(model_id, timeout_seconds=remaining)
        while True:
            service_model = self._find_service_model(model_id)
            if service_model.get("loaded"):
                return
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise TimerServiceError(f"timed out waiting for model [{model_id}] to load")
            time.sleep(min(2.0, remaining_seconds))

    def _find_service_model(self, model_id: str) -> dict:
        service_model = next((item for item in self.list_models() if item.get("model_id") == model_id), None)
        if service_model is None:
            raise TimerServiceError(f"model [{model_id}] is not registered in timer-rest-service")
        return service_model

    # -- 内部实现 ------------------------------------------------------------ #
    def _build_request(self, sample: dict, model: dict, horizon: int) -> dict:
        history = sample["target_history"]
        history_ts = sample.get("history_timestamps") or list(range(len(history)))
        value_cols = sample.get("target_column_names") or [
            f"value{i}" for i in range(len(history[0]) if history else 0)
        ]
        columns = [self._TIME_COL, *value_cols]
        data = [[history_ts[i], *row] for i, row in enumerate(history)]
        return {
            "model_id": model.get("remote_model_id") or str(model["model_id"]),
            "targets": [{"columns": columns, "data": data}],
            "output_length": [horizon],
            "time_col": [self._TIME_COL],
        }

    def _post(self, url: str, body: dict, timeout_seconds: int) -> dict:
        if self._client is not None:
            # 注入的 client 自带超时/传输配置，由调用方负责。
            resp = self._client.post(url, json=body)
        else:
            # trust_env=False：模型服务是内网服务，不应经由系统代理（HTTP/SOCKS）。
            with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
                resp = client.post(url, json=body)
        payload = resp.json()
        if resp.status_code != 200:
            message = payload.get("message") if isinstance(payload, dict) else resp.text
            raise TimerServiceError(f"{url} returned {resp.status_code}: {message}")
        self._raise_for_service_envelope(url, payload)
        return payload

    def _get(self, url: str, timeout_seconds: int) -> dict:
        if self._client is not None:
            resp = self._client.get(url)
        else:
            with httpx.Client(timeout=timeout_seconds, trust_env=False) as client:
                resp = client.get(url)
        payload = resp.json()
        if resp.status_code != 200:
            message = payload.get("message") if isinstance(payload, dict) else resp.text
            raise TimerServiceError(f"{url} returned {resp.status_code}: {message}")
        self._raise_for_service_envelope(url, payload)
        return payload

    def _raise_for_service_envelope(self, url: str, payload: dict) -> None:
        """timer-rest-service reports some business errors as HTTP 200 + code != 200."""
        code = payload.get("code") if isinstance(payload, dict) else None
        if code is not None and code != 200:
            message = payload.get("message") if isinstance(payload, dict) else payload
            raise TimerServiceError(f"{url} returned service code {code}: {message}")

    def _is_already_unloaded_error(self, exc: TimerServiceError) -> bool:
        message = str(exc).lower()
        return "409" in message and "not loaded" in message

    def _parse_response(self, payload: dict, horizon: int) -> list[list[float]]:
        try:
            result = payload["data"]["results"][0]
            columns, rows = result["columns"], result["data"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TimerServiceError(f"unexpected forecast response shape: {payload}") from exc
        value_idx = [i for i, col in enumerate(columns) if col != self._TIME_COL]
        forecast = [[float(row[i]) for i in value_idx] for row in rows]
        return forecast[:horizon]
