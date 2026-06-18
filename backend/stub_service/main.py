"""timer-rest-service 桩服务。

覆盖 docs/reference/rest-api.md 的应用层端点，供本地无真实推理服务时让
TSBenchmark 后端通过 REST 正常跑通：

  - GET  /                                         → 302 跳转 /docs
  - GET  /health/{startup,liveness,readiness}
  - POST /reboot                                   （安全桩：不真正发 SIGTERM）
  - GET  /metrics                                  （Prometheus 文本）
  - POST /ai/api/v1/forecast
  - GET  /ai/api/v1/models/list
  - POST /ai/api/v1/models/{register,load,unload,delete}
  - GET  /ai/api/v1/models/list_loaded
  - POST /ai/api/v1/dataset/evaluate/execute
  - GET  /ai/api/v1/dataset/evaluate/list_dimensions
  - POST /ai/api/v1/dataset/govern/execute
  - GET  /ai/api/v1/dataset/govern/list_dimensions

推理与治理结果是**确定性的**，便于本地开发与可复现演示。模型注册/加载状态
保存在内存中（每个 app 实例独立），桩把内置模型视为已加载。

本地启动：
    cd backend && uv run uvicorn stub_service.main:app --host 127.0.0.1 --port 10810
或：
    ./scripts/stub-service.sh start
"""
from __future__ import annotations

import hashlib
import os
import random
import time
from datetime import datetime
from statistics import fmean

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse

STUB_VERSION = "0.1.2-stub"
API_PREFIX = "/ai/api/v1"

# 与 rest-api.md 的 /models/list 示例一致的内置模型集合。
BUILTIN_MODELS = [
    {
        "model_id": "Timer-3.5",
        "model_type": "Timer-S1",
        "category": "builtin",
        "base_model_id": None,
        "forecast_limits": {"min_input_length": 16, "max_input_length": 11520, "max_future_covs_length": None, "max_target_count": 1, "max_covariate_count": 0, "max_output_length": 720, "default_output_length": 272},
    },
    {
        "model_id": "Timer-3.0",
        "model_type": "sundial",
        "category": "builtin",
        "base_model_id": None,
        "forecast_limits": {"min_input_length": 16, "max_input_length": 2880, "max_future_covs_length": None, "max_target_count": 1, "max_covariate_count": 0, "max_output_length": 720, "default_output_length": 96},
    },
    {
        "model_id": "Chronos-2",
        "model_type": "t5",
        "category": "builtin",
        "base_model_id": None,
        "forecast_limits": {"min_input_length": 16, "max_input_length": 8192, "max_future_covs_length": 720, "max_target_count": 1, "max_covariate_count": 50, "max_output_length": 720, "default_output_length": 96},
    },
    {
        "model_id": "toto2.0",
        "model_type": "toto2p0",
        "category": "builtin",
        "base_model_id": None,
        "forecast_limits": {"min_input_length": 16, "max_input_length": 8192, "max_future_covs_length": None, "max_target_count": None, "max_covariate_count": 0, "max_output_length": 720, "default_output_length": 96},
    },
    {
        "model_id": "AutoARIMA",
        "model_type": "auto_arima",
        "category": "builtin",
        "base_model_id": None,
        "forecast_limits": {"min_input_length": 16, "max_input_length": 2880, "max_future_covs_length": None, "max_target_count": 1, "max_covariate_count": 0, "max_output_length": 720, "default_output_length": 96},
    },
    {
        "model_id": "Holt-Winters",
        "model_type": "holtwinters",
        "category": "builtin",
        "base_model_id": None,
        "forecast_limits": {"min_input_length": 16, "max_input_length": 2880, "max_future_covs_length": None, "max_target_count": 1, "max_covariate_count": 0, "max_output_length": 720, "default_output_length": 96},
    },
]

EVALUATE_DIMENSIONS = [
    {"name": "integrity", "description": "统一的时间戳完整性检查，涵盖完整性、一致性和时效性",
     "supported_params": [{"name": "downtime", "type": "boolean", "default": True, "description": "计算时效性时是否考虑已知停机时段"}]},
    {"name": "forecastability", "description": "基于频域的可预测性度量，使用谱熵（Goerg 2013）计算", "supported_params": []},
    {"name": "pearson", "description": "皮尔逊相关系数，返回序列间的相关矩阵",
     "supported_params": [{"name": "targets", "type": "array of strings", "default": None, "description": "限制相关性矩阵的行仅为这些目标序列"}]},
]
EVALUATE_DIMENSION_NAMES = [d["name"] for d in EVALUATE_DIMENSIONS]

GOVERN_DIMENSIONS = [
    {"name": "timestamp_repair", "description": "时间戳修复 — 检测并修复时间戳缺失、冗余、错乱", "supported_params": []},
    {"name": "causal_mean_imputation", "description": "因果均值填充 — 使用之前窗口的均值填充 NaN", "supported_params": []},
    {"name": "flat_series_removal", "description": "平稳序列剔除 — 移除方差低于阈值的序列",
     "supported_params": [{"name": "min_variance", "type": "number", "default": 1e-9, "description": "方差阈值"}]},
    {"name": "zscore_normalization", "description": "Z-score 标准化 — 按序列做零均值单位方差变换", "supported_params": []},
    {"name": "extreme_value_clipping", "description": "极值裁剪 — 将超出 ±threshold 标准差的点裁剪到边界",
     "supported_params": [{"name": "threshold", "type": "number", "default": 3.0, "description": "裁剪阈值（标准差倍数）"}]},
]
GOVERN_DIMENSION_NAMES = [d["name"] for d in GOVERN_DIMENSIONS]


def _stub_seed() -> int:
    return int(os.getenv("TSBENCHMARK_STUB_SEED", "0"))


def _envelope(code: int, message: str, data) -> JSONResponse:
    body = {
        "code": code,
        "message": message,
        "service_info": {"timestamp": int(time.time()), "version": STUB_VERSION},
        "data": data,
    }
    return JSONResponse(status_code=code, content=body)


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def _future_timestamps(times: list, horizon: int) -> list:
    """根据输入末两个时间戳推断节奏并外推；无法解析时退化为整数序号。"""
    if len(times) >= 2 and all(isinstance(t, (int, float)) for t in times[-2:]):
        step = times[-1] - times[-2]
        return [times[-1] + step * (k + 1) for k in range(horizon)]
    if len(times) >= 2:
        try:
            prev = datetime.fromisoformat(str(times[-2]))
            last = datetime.fromisoformat(str(times[-1]))
            delta = last - prev
            return [(last + delta * (k + 1)).isoformat() for k in range(horizon)]
        except ValueError:
            pass
    base = len(times)
    return [base + k for k in range(horizon)]


def _forecast_task(model_id: str, columns: list[str], data: list[list], horizon: int, time_col: str) -> dict:
    value_idx = [i for i, col in enumerate(columns) if col != time_col]
    time_idx = columns.index(time_col) if time_col in columns else 0
    last_row = data[-1] if data else []
    last_vals = [float(last_row[i]) for i in value_idx] if last_row else []

    bias = ((int(hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:4], 16) % 21) - 10) / 100.0
    digest = hashlib.sha256(f"{model_id}:{len(data)}:{_stub_seed()}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))

    future_ts = _future_timestamps([row[time_idx] for row in data], horizon)
    out_columns = [time_col] + [columns[i] for i in value_idx]
    out_data = []
    for step in range(horizon):
        values = [round(v + bias + rng.uniform(-0.05, 0.05), 6) for v in last_vals]
        out_data.append([future_ts[step], *values])
    return {"columns": out_columns, "data": out_data}


def _forecast_validation_error(model: dict | None, columns: list[str], data: list[list], horizon: int, time_col: str, task_index: int) -> str | None:
    limits = (model or {}).get("forecast_limits") or {}
    value_count = len([col for col in columns if col != time_col])
    min_input = limits.get("min_input_length")
    max_input = limits.get("max_input_length")
    max_output = limits.get("max_output_length")
    max_targets = limits.get("max_target_count")
    if min_input is not None and len(data) < int(min_input):
        return f"Task-{task_index}'s input length {len(data)} is illegal, acceptable minimum is {min_input}."
    if max_input is not None and len(data) > int(max_input):
        return f"Task-{task_index}'s input length {len(data)} is illegal, acceptable maximum is {max_input}."
    if max_output is not None and horizon > int(max_output):
        return f"Task-{task_index}'s output length {horizon} is illegal, acceptable maximum is {max_output}."
    if max_targets is not None and value_count > int(max_targets):
        return f"Task-{task_index} has {value_count} targets, acceptable maximum is {max_targets}."
    return None


# --------------------------------------------------------------------------- #
# 数据集：解析与计算
# --------------------------------------------------------------------------- #
def _parse_inline(inline: dict) -> tuple[list[str], list[list], str | None, dict[str, list]]:
    columns = inline.get("columns") or []
    data = inline.get("data") or []
    time_col = inline.get("time_col")
    series: dict[str, list] = {}
    for ci, col in enumerate(columns):
        if col == time_col:
            continue
        series[col] = [row[ci] if ci < len(row) else None for row in data]
    return columns, data, time_col, series


def _stub_entropy(series_id: str, values: list) -> float:
    digest = hashlib.sha256(f"{series_id}:{len(values)}:{_stub_seed()}".encode("utf-8")).hexdigest()
    return round((int(digest[:8], 16) % 1000) / 1000.0, 4)


def _pearson(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    if len(pairs) < 2:
        return 0.0
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = fmean(xs), fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return round(num / (dx * dy), 4)


def _evaluate(inline: dict, dimensions: list[str], params: dict) -> dict:
    _columns, _data, _time_col, series = _parse_inline(inline)
    reports: dict[str, dict] = {sid: {} for sid in series}
    dim_scores: dict[str, float] = {}

    if "integrity" in dimensions:
        scores = []
        for sid, vals in series.items():
            total = len(vals)
            non_null = sum(1 for v in vals if v is not None)
            completeness = round(100 * non_null / total, 2) if total else 0.0
            score = round((completeness + 100.0 + 100.0) / 3, 2)
            reports[sid]["integrity"] = {"completeness": completeness, "consistency": 100.0, "timeliness": 100.0, "score": score}
            scores.append(score)
        dim_scores["integrity"] = round(fmean(scores), 2) if scores else 0.0

    if "forecastability" in dimensions:
        scores = []
        for sid, vals in series.items():
            entropy = _stub_entropy(sid, vals)
            score = round((1 - entropy) * 100, 2)
            reports[sid]["forecastability"] = {"spectral_entropy": entropy, "score": score}
            scores.append(score)
        dim_scores["forecastability"] = round(fmean(scores), 2) if scores else 0.0

    if "pearson" in dimensions:
        targets = params.get("targets") or list(series.keys())
        rows = [t for t in targets if t in series]
        corr_vals = []
        for sid in rows:
            matrix = {other: _pearson(series[sid], series[other]) for other in series}
            reports[sid]["pearson"] = matrix
            corr_vals += [abs(v) for other, v in matrix.items() if other != sid]
        dim_scores["pearson"] = round(fmean(corr_vals) * 100, 2) if corr_vals else 0.0

    overall = round(fmean(dim_scores.values()), 2) if dim_scores else 0.0
    return {
        "overall_score": overall,
        "dimension_scores": dim_scores,
        "series_reports": [{"series_id": sid, **rep} for sid, rep in reports.items()],
    }


def _to_ms(value) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(datetime.fromisoformat(str(value)).timestamp() * 1000)
    except ValueError:
        return 0


def _govern(inline: dict, dimension: str, params: dict) -> dict:
    columns, data, time_col, series = _parse_inline(inline)
    new_series = {sid: list(vals) for sid, vals in series.items()}
    changes = 0

    if dimension == "causal_mean_imputation":
        for vals in new_series.values():
            seen: list[float] = []
            for i, v in enumerate(vals):
                if v is None:
                    vals[i] = round(fmean(seen), 6) if seen else 0.0
                    changes += 1
                else:
                    seen.append(v)
    elif dimension == "zscore_normalization":
        for vals in new_series.values():
            nums = [v for v in vals if isinstance(v, (int, float))]
            if len(nums) >= 2:
                m = fmean(nums)
                sd = (sum((x - m) ** 2 for x in nums) / len(nums)) ** 0.5 or 1.0
                for i, v in enumerate(vals):
                    if isinstance(v, (int, float)):
                        vals[i] = round((v - m) / sd, 6)
                        changes += 1
    elif dimension == "extreme_value_clipping":
        threshold = float(params.get("threshold", 3.0))
        for vals in new_series.values():
            nums = [v for v in vals if isinstance(v, (int, float))]
            if len(nums) >= 2:
                m = fmean(nums)
                sd = (sum((x - m) ** 2 for x in nums) / len(nums)) ** 0.5
                lo, hi = m - threshold * sd, m + threshold * sd
                for i, v in enumerate(vals):
                    if isinstance(v, (int, float)):
                        clipped = min(max(v, lo), hi)
                        if clipped != v:
                            vals[i] = round(clipped, 6)
                            changes += 1
    elif dimension == "flat_series_removal":
        min_variance = float(params.get("min_variance", 1e-9))
        for sid in list(new_series.keys()):
            nums = [v for v in new_series[sid] if isinstance(v, (int, float))]
            if len(nums) >= 2:
                m = fmean(nums)
                variance = sum((x - m) ** 2 for x in nums) / len(nums)
                if variance < min_variance:
                    del new_series[sid]
                    changes += 1
    # timestamp_repair：桩按 no-op 处理（changes_count=0）。

    out_columns = [c for c in columns if c == time_col or c in new_series]
    times = [row[columns.index(time_col)] for row in data] if time_col in columns else list(range(len(data)))
    out_data = []
    for i in range(len(data)):
        row = []
        for col in out_columns:
            row.append(_to_ms(times[i]) if col == time_col else new_series[col][i])
        out_data.append(row)
    return {
        "summary": {dimension: {"changes_count": changes}},
        "inline": {"columns": out_columns, "data": out_data, "time_col": time_col},
    }


# --------------------------------------------------------------------------- #
# 应用
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    app = FastAPI(title="timer-rest-service (stub)", version=STUB_VERSION)

    # 内存模型注册表（每个 app 实例独立，便于测试隔离）。
    models: dict[str, dict] = {
        m["model_id"]: {**m, "state": "active", "loaded": True} for m in BUILTIN_MODELS
    }
    loaded: dict[str, dict] = {
        m["model_id"]: {"status": "loaded", "devices": ["cpu"], "endpoints": [{"device": "cpu", "worker_pid": 10000 + i}]}
        for i, m in enumerate(BUILTIN_MODELS)
    }

    # -- 基础 / 健康 / 运维 ------------------------------------------------- #
    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs", status_code=302)

    @app.get("/health/startup")
    def startup() -> dict:
        return {"status": "started", "version": STUB_VERSION}

    @app.get("/health/liveness")
    def liveness() -> dict:
        return {"status": "alive", "version": STUB_VERSION}

    @app.get("/health/readiness")
    def readiness() -> dict:
        return {"status": "ready", "version": STUB_VERSION}

    @app.post("/reboot")
    def reboot() -> JSONResponse:
        # 安全桩：返回契约形状，但不真正向自身发送 SIGTERM。
        return JSONResponse(status_code=202, content={"status": "rebooting", "version": STUB_VERSION, "main_pid": os.getpid()})

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        lines = [
            "# HELP timer_service:model_workers_loaded Loaded ModelWorker processes",
            "# TYPE timer_service:model_workers_loaded gauge",
            f"timer_service:model_workers_loaded {len(loaded)}",
            "# HELP timer_service:request_active In-flight requests",
            "# TYPE timer_service:request_active gauge",
            "timer_service:request_active 0",
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    # -- 推理 --------------------------------------------------------------- #
    @app.post(f"{API_PREFIX}/forecast")
    def forecast(body: dict = Body(...)) -> JSONResponse:
        targets = body.get("targets")
        if not targets:
            return _envelope(422, "targets is required and must be non-empty", {"results": []})

        model_id = str(body.get("model_id") or "Timer-3.5")
        # 已注册但被显式卸载 → 503（让 unload 可被观测）；未注册的 model_id 视为
        # 可用（桩是通用替身，后端的 toto/TimesFM 等也能跑通）。
        if model_id in models and model_id not in loaded:
            return _envelope(503, f"Model [{model_id}] is not available (not loaded)", {"results": []})

        time_cols = body.get("time_col") or []
        out_lens = body.get("output_length") or []
        results = []
        for i, target in enumerate(targets):
            columns = target.get("columns") or []
            data = target.get("data") or []
            if not columns or not data:
                return _envelope(422, f"Task-{i}'s target is missing columns or data", {"results": []})
            time_col = time_cols[i] if i < len(time_cols) else (columns[0] if columns else "time")
            horizon = out_lens[i] if i < len(out_lens) else (out_lens[0] if out_lens else 96)
            validation_error = _forecast_validation_error(models.get(model_id), columns, data, int(horizon), time_col, i)
            if validation_error:
                return _envelope(422, validation_error, {"results": []})
            results.append(_forecast_task(model_id, columns, data, int(horizon), time_col))
        return _envelope(200, "Forecast tasks completed successfully", {"results": results})

    # -- 模型管理 ----------------------------------------------------------- #
    @app.get(f"{API_PREFIX}/models/list")
    def list_models() -> JSONResponse:
        return _envelope(200, "Success", {"models": list(models.values())})

    @app.get(f"{API_PREFIX}/models/list_loaded")
    def list_loaded() -> JSONResponse:
        items = [{"model_id": mid, **info} for mid, info in loaded.items()]
        return _envelope(200, "Success", {"models": items})

    @app.post(f"{API_PREFIX}/models/register")
    def register_model(body: dict = Body(...)) -> JSONResponse:
        model_id = body.get("model_id")
        uri = body.get("uri")
        base_model_id = body.get("base_model_id")
        if not model_id or not uri:
            return _envelope(422, "Failed to register model: model_id and uri are required", None)
        if model_id in models:
            return _envelope(409, f"Failed to register model: Model {model_id} already exists", None)
        if base_model_id is not None and base_model_id not in models:
            return _envelope(404, f"Failed to register model: Model {base_model_id} does not exist", None)
        models[model_id] = {
            "model_id": model_id,
            "model_type": models.get(base_model_id, {}).get("model_type", "user_defined"),
            "category": "user_defined",
            "state": "active",
            "loaded": False,
            "base_model_id": base_model_id,
        }
        return _envelope(200, f"Model '{model_id}' registered successfully", {"model_id": model_id})

    @app.post(f"{API_PREFIX}/models/load")
    def load_model(body: dict = Body(...)) -> JSONResponse:
        model_id = body.get("model_id")
        if not model_id:
            return _envelope(422, "Failed to load model: model_id is required", None)
        if model_id not in models:
            return _envelope(404, f"Failed to load model: Failed to resolve model info: Model {model_id} does not exist", None)
        loaded[model_id] = {"status": "loaded", "devices": ["cpu"], "endpoints": [{"device": "cpu", "worker_pid": os.getpid()}]}
        models[model_id]["loaded"] = True
        return _envelope(200, f"Model '{model_id}' loaded successfully on devices ['cpu']", {"model_id": model_id, "devices": ["cpu"]})

    @app.post(f"{API_PREFIX}/models/unload")
    def unload_model(body: dict = Body(...)) -> JSONResponse:
        model_id = body.get("model_id")
        if not model_id:
            return _envelope(422, "Failed to unload model: model_id is required", None)
        if model_id not in loaded:
            return _envelope(409, f"Failed to unload model: Model '{model_id}' is not loaded", None)
        del loaded[model_id]
        if model_id in models:
            models[model_id]["loaded"] = False
        return _envelope(200, f"Model '{model_id}' unloaded successfully", {"model_id": model_id})

    @app.post(f"{API_PREFIX}/models/delete")
    def delete_model(body: dict = Body(...)) -> JSONResponse:
        model_id = body.get("model_id")
        if not model_id:
            return _envelope(422, "Failed to delete model: model_id is required", None)
        if model_id not in models:
            return _envelope(404, f"Failed to delete model: Model {model_id} does not exist", None)
        if models[model_id]["category"] == "builtin":
            return _envelope(403, f"Failed to delete model: Cannot delete built-in model: {model_id}", None)
        del models[model_id]
        loaded.pop(model_id, None)
        return _envelope(200, f"Model '{model_id}' deleted successfully", {"model_id": model_id})

    # -- 数据集：评估 ------------------------------------------------------- #
    @app.post(f"{API_PREFIX}/dataset/evaluate/execute")
    def evaluate_execute(body: dict = Body(...)) -> JSONResponse:
        source = body.get("input") or {}
        inline = source.get("inline")
        if inline is None and source.get("tsfile") is None:
            return _envelope(422, "Either 'input.inline' or 'input.tsfile' must be provided", {})
        dimensions = body.get("dimensions") or EVALUATE_DIMENSION_NAMES
        invalid = [d for d in dimensions if d not in EVALUATE_DIMENSION_NAMES]
        if invalid:
            return _envelope(422, f"Unknown evaluate dimension(s): {invalid}", {})
        if inline is None:
            # 桩不读取 TsFile，返回固定的占位结果。
            return _envelope(200, "Dataset evaluation completed successfully",
                             {"overall_score": 80.0, "dimension_scores": {d: 80.0 for d in dimensions}, "series_reports": []})
        data = _evaluate(inline, dimensions, body.get("params") or {})
        return _envelope(200, "Dataset evaluation completed successfully", data)

    @app.get(f"{API_PREFIX}/dataset/evaluate/list_dimensions")
    def evaluate_list_dimensions() -> JSONResponse:
        return _envelope(200, "Success", {"dimensions": EVALUATE_DIMENSIONS})

    # -- 数据集：治理 ------------------------------------------------------- #
    @app.post(f"{API_PREFIX}/dataset/govern/execute")
    def govern_execute(body: dict = Body(...)) -> JSONResponse:
        source = body.get("input") or {}
        inline = source.get("inline")
        dimension = body.get("dimension")
        if inline is None and source.get("tsfile") is None:
            return _envelope(422, "Either 'input.inline' or 'input.tsfile' must be provided", {})
        if not dimension or dimension not in GOVERN_DIMENSION_NAMES:
            return _envelope(422, f"dimension must be one of {GOVERN_DIMENSION_NAMES}", {})
        if inline is None:
            return _envelope(422, "stub govern only supports inline input", {})

        started = time.perf_counter()
        result = _govern(inline, dimension, body.get("params") or {})
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        output_path = body.get("output_tsfile_path")
        if output_path:
            # 桩不真正写盘，仅返回契约形状。
            payload = {"dimension": dimension, "summary": result["summary"], "output_tsfile_path": output_path, "elapsed_ms": elapsed_ms}
        else:
            payload = {"dimension": dimension, "summary": result["summary"], "inline": result["inline"], "elapsed_ms": elapsed_ms}
        return _envelope(200, "Dataset governance completed successfully", payload)

    @app.get(f"{API_PREFIX}/dataset/govern/list_dimensions")
    def govern_list_dimensions() -> JSONResponse:
        return _envelope(200, "Success", {"dimensions": GOVERN_DIMENSIONS})

    return app


app = create_app()
