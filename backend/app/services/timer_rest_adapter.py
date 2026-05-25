"""通过 timer-rest-service 的 REST API 执行推理的模型适配器。

契约见 docs/reference/rest-api.md（POST /ai/api/v1/forecast）。
本地无真实服务时，可启动桩服务（backend/stub_service）并将
TSBENCHMARK_TIMER_SERVICE_BASE_URL 指向它。
"""
from __future__ import annotations

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
        body = self._build_request(sample, model)
        try:
            response = self._post(f"{self._base}/forecast", body, timeout_seconds)
        except httpx.HTTPError as exc:
            raise TimerServiceError(f"forecast request failed: {exc}") from exc
        return self._parse_response(response, horizon=len(sample["target_future"]))

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

    # -- 内部实现 ------------------------------------------------------------ #
    def _build_request(self, sample: dict, model: dict) -> dict:
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
            "output_length": [len(sample["target_future"])],
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
        return payload

    def _parse_response(self, payload: dict, horizon: int) -> list[list[float]]:
        try:
            result = payload["data"]["results"][0]
            columns, rows = result["columns"], result["data"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TimerServiceError(f"unexpected forecast response shape: {payload}") from exc
        value_idx = [i for i, col in enumerate(columns) if col != self._TIME_COL]
        forecast = [[float(row[i]) for i in value_idx] for row in rows]
        return forecast[:horizon]
