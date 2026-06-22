<!-- 本文件由 scripts/sync-feishu-docs.py 自动生成，请勿手工编辑。 -->
<!-- 内容更新：修改飞书原文后重新运行同步脚本。 -->

> **来源**：[飞书文档](https://timechor.feishu.cn/docx/CwaMdyCmhovwIGxRbGicKuExnVh)（docx token `CwaMdyCmhovwIGxRbGicKuExnVh`）  
> **最后同步**：2026-06-04  
> **更新方式**：`python3 scripts/sync-feishu-docs.py`

---

# REST API 参考文档

基础 URL: `http://<host>:<port>` (默认: `http://127.0.0.1:10810`)

所有 API 端点使用前缀 `/ai/api/v1`。响应为 JSON 格式（通过 FastAPI 原生 Pydantic 序列化）。

`GET /` 重定向到 Swagger UI（`302 Redirect` → `/docs`）。

---

## 健康探针

### `GET /health/startup`

启动探针。`InferenceRouter` 初始化完成后返回 HTTP 200，否则返回 HTTP 503。

Kubernetes 配置建议：`failureThreshold` 设置较大值（如 30），允许模型加载耗时较长。

**响应（已启动）:**

```json
{
  "status": "started",
  "version": "0.1.2"
}
```

**响应（启动中 — HTTP 503）:**

```json
{
  "status": "starting",
  "version": "0.1.2"
}
```

---

### `GET /health/liveness`

存活探针。覆盖两类容器级 liveness 应当捕获的失效：

- **ZMQ 订阅线程存活** — 该线程死掉后，Coordinator 的路由表更新将不再到达本 worker，端点会悄无声息地过期。
- **Coordinator PING 可达** — 每个 Uvicorn worker 每 5s 通过独立 REQ socket 向 Coordinator REP 端发 PING（与按需命令通道隔离）。若最近一次成功 pong 已超过 15s，判定 Coordinator 不可达。

若该探针持续返回 503，Kubernetes 应重启容器。

**响应（存活）:**

```json
{
  "status": "alive",
  "version": "0.1.2"
}
```

**响应（ZMQ 订阅线程异常 — HTTP 503）:**

```json
{
  "status": "unhealthy",
  "version": "0.1.2",
  "reason": "zmq subscriber dead"
}
```

**响应（Coordinator 不可达 — HTTP 503）:**

```json
{
  "status": "unhealthy",
  "version": "0.1.2",
  "reason": "coordinator unreachable",
  "last_pong_age_seconds": 18.4
}
```

---

### `GET /health/readiness`

就绪探针。仅当 **服务启动时配置自动加载的所有模型** 都至少有一个可用端点时返回 HTTP 200。预期模型集合：

- 所有 `BUILTIN_SKTIME_MODEL_MAP` 中的 sktime 内置模型（无条件加载）。
- 由 `TIMER_AUTO_LOAD_MODEL` 环境变量决定的 DL 模型：
  - `"all"` → 所有启动时已下载权重的内置 HF 模型。
  - `"<model_id>"` → 指定的具体模型。
  - 空 / `"none"` → 无 DL 模型。

Coordinator 在每次发布路由表时同步广播这份预期集合，Uvicorn worker 无需额外 RPC 即可独立比对。

若返回 503，Kubernetes 会将 Pod 从 Service 端点列表中摘除，停止分配新流量。

**响应（就绪）:**

```json
{
  "status": "ready",
  "version": "0.1.2"
}
```

**响应（预期模型缺失 — HTTP 503）:**

```json
{
  "status": "not_ready",
  "version": "0.1.2",
  "reason": "expected models not loaded",
  "missing": ["Chronos-2"],
  "expected": ["AutoARIMA", "Chronos-2", "Holt-Winters", "Timer-3.0", "Timer-3.5"]
}
```

**自动恢复**：当某个模型超过 `MAX_RESTART_ATTEMPTS` 连续 worker 重启次数后，Coordinator 将其标记为 abandoned。每隔 `ABANDONED_RETRY_INTERVAL` 秒（默认 300s），Coordinator 会以 restart_count=0 的全新计数重试所有仍在预期集合内的 `(model_id, device)` 组合。一旦重试出的 worker 就绪，`/readiness` 会重新转绿。

---

### `POST /reboot`

通过向 timer-rest-service 主进程发送 `SIGTERM` 触发整个服务的优雅重启。Uvicorn 接到信号后会平滑停止所有 worker，随后主进程拆除 `ModelCoordinator` 并退出。

**重启动作交给进程 supervisor 处理。** 请在部署侧配置 `Restart=always`（systemd）或 `restartPolicy: Always`（Kubernetes），以便干净退出后自动拉起。如果没有 supervisor，服务退出后将停止运行。

该端点**目前未做鉴权**。在引入应用层鉴权之前，请在网络层（防火墙、Ingress 规则、Service Mesh）限制访问。

**响应（202 Accepted）:**

```json
{
  "status": "rebooting",
  "version": "0.1.2",
  "main_pid": 418903
}
```

实际的 `SIGTERM` 信号会在响应返回约 0.5s 后才发送，确保响应主体可以完整 flush 到客户端再触发 worker 关停。

**失败场景**：若环境变量 `TIMER_SERVICE_MAIN_PID` 未设置（例如服务以非常规方式启动），端点返回 HTTP 500 并包含错误信封。正常 `timer-rest-service start` 启动流程会自动写入该变量。

---

## 推理

### `POST /ai/api/v1/forecast`

核心推理端点。在已加载的模型上执行时间序列预测。

请求/响应模型使用 FastAPI 原生 Pydantic 序列化（直接转为 JSON 字节，性能最优）。

**请求体:**

```json
{
  "model_id": "Timer-3.5",
  "targets": [
    {
      "columns": ["time", "value"],
      "data": [
        ["2024-01-01T00:00:00Z", 1.0],
        ["2024-01-01T01:00:00Z", 1.1]
      ]
    }
  ],
  "history_covs": null,
  "future_covs": null,
  "output_length": [96],
  "output_start_time": null,
  "output_interval": null,
  "time_col": ["time"]
}
```

**字段说明:**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `model_id` | string | 否 | 带协变量时为 `"Chronos-2"`，多变量目标时为 `"toto2.0"`，否则为 `"Timer-3.5"`；匹配大小写不敏感 | 使用的模型 ID |
| `targets` | 对象数组 | **是** | — | 每个任务的目标时间序列数据 |
| `targets[i].columns` | 字符串数组 | 是 | — | 列名（包括时间列） |
| `targets[i].data` | 二维数组 | 是 | — | 行导向数据 |
| `history_covs` | 对象数组 | 否 | `null` | 每个任务的历史协变量（与 targets 相同格式） |
| `future_covs` | 对象数组 | 否 | `null` | 每个任务的未来协变量 |
| `output_length` | 整数数组 | 否 | 按模型而定（Timer-3.5: 272、Timer-3.0: 96、Chronos-2: 96） | 每个任务的预测步长 |
| `output_start_time` | 字符串数组 | 否 | `null` | 每个任务的预测起始时间戳（ISO 8601） |
| `output_interval` | 数字数组 | 否 | `null` | 每个任务的时间间隔（秒） |
| `time_col` | 字符串数组 | 否 | `null` | 每个任务中 `targets` 的时间列名称 |

**校验规则（按模型）:**

| 约束 | Timer-3.5 | Timer-3.0 | Chronos-2 | 错误信息 |
| --- | --- | --- | --- | --- |
| `output_length[i]` | [1, 720] | [1, 720] | [1, 720] | `Task-{i}'s output length is illegal` |
| 目标输入长度 (`len(targets[i].data)`) | [16, 11520] | [16, 2880] | [16, 8192] | `Task-{i}'s input length is illegal` |
| 目标变量数 | = 1（仅单变量） | = 1（仅单变量） | = 1（仅单变量） | `Task-{i} has {n} targets...` |
| 历史协变量数量 | 0（被忽略） | 0（被忽略） | [1, 50] | `Task-{i} has {n} history covariates...` |
| 历史协变量输入长度 | — | — | [16, 8192] | `Task-{i}'s history covariate length is illegal` |
| 未来协变量数量 | 0（被忽略） | 0（被忽略） | [1, 50] | `Task-{i} has {n} future covariates...` |
| 未来协变量输入长度 | — | — | [1, 720] | `Task-{i}'s future covariate length is illegal` |
| 模型必须已加载 | — | — | — | `Model [{model_id}] is not available (not loaded)` |

说明：

- `output_length` 由 REST 层统一限制为 720，对所有模型一致，与各模型更大的原生上限无关（Timer-3.5 的 KV-Cache 自回归、Chronos-2 的 1024 单次前向上限）。Timer-3.5 的单次前向上限仍为 272。
- 未来协变量覆盖预测时域，因此其每行长度与 `output_length` 共享同一 720 步上限（基础大模型中仅 Chronos-2 接受协变量）。
- 上述基础大模型在 REST 层均为单变量目标（每个
- 任务一条目标序列）。多变量目标预测由 `toto2.0` 提供（目标数无上限、不支持协变量）；自动选择会将多变量目标输入路由到该模型。
- Sktime 模型（如 `AutoARIMA`、`Holt-Winters`）沿用 Timer-3.0 的 `output_length` 默认范围 `[1, 720]`。
- 上述按模型的限制也可通过 `GET /models/list` 在每个模型的 `forecast_limits` 对象中以编程方式获取。

**成功响应 (200):**

```json
{
  "code": 200,
  "message": "Forecast tasks completed successfully",
  "service_info": {
    "timestamp": 1712345678,
    "version": "0.0.1.dev"
  },
  "data": {
    "results": [
      {
        "columns": ["time", "value1", "value2"],
        "data": [
          ["2024-01-02T00:00:00", 1.23, 4.56],
          ["2024-01-02T01:00:00", 1.34, 4.67]
        ]
      }
    ]
  }
}
```

**校验错误 (HTTP 422 — Unprocessable Entity):**

请求体 parse 成功但参数语义校验失败（`output_length`、`input_length`、目标变量数、协变量数量/长度等）。

```json
{
  "code": 422,
  "message": "Task-0's output length 800 is illegal, acceptable range for model Timer-3.0 is [1, 720].",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": { "results": [] }
}
```

**模型未加载 (HTTP 503 — Service Unavailable):**

请求合法，但目标模型当前未加载到任何 worker 上。调用方应先用 `POST /models/load` 加载模型或改用其他已加载的 `model_id`。

```json
{
  "code": 503,
  "message": "Model [Timer-3.0] is not available (not loaded)",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": { "results": [] }
}
```

**内部错误 (HTTP 500 — Internal Server Error):**

```json
{
  "code": 500,
  "message": "Forecast failed, because ...",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": null
}
```

---

## 模型管理

### `GET /ai/api/v1/models/list`

列出模型存储管理的所有模型（包括内置模型和用户自定义模型）。

排序规则：时序基础大模型始终排在前面，固定顺序为 `Timer-3.5` → `Timer-3.0` → `Chronos-2`，其后是 sktime 统计/ML 模型，最后是用户自定义模型。

**响应:**

```json
{
  "code": 200,
  "message": "Success",
  "service_info": {
    "timestamp": 1712345678,
    "version": "0.0.1.dev"
  },
  "data": {
    "models": [
      {
        "model_id": "Timer-3.5",
        "model_type": "Timer-S1",
        "category": "builtin",
        "state": "active",
        "loaded": true,
        "base_model_id": null,
        "forecast_limits": {
          "min_input_length": 16,
          "max_input_length": 11520,
          "max_future_covs_length": null,
          "max_output_length": 720,
          "max_target_count": 1,
          "max_covariate_count": 0,
          "default_output_length": 272
        }
      },
      {
        "model_id": "Timer-3.0",
        "model_type": "sundial",
        "category": "builtin",
        "state": "active",
        "loaded": true,
        "base_model_id": null,
        "forecast_limits": {
          "min_input_length": 16,
          "max_input_length": 2880,
          "max_future_covs_length": null,
          "max_output_length": 720,
          "max_target_count": 1,
          "max_covariate_count": 0,
          "default_output_length": 96
        }
      },
      {
        "model_id": "Chronos-2",
        "model_type": "t5",
        "category": "builtin",
        "state": "active",
        "loaded": true,
        "base_model_id": null,
        "forecast_limits": {
          "min_input_length": 16,
          "max_input_length": 8192,
          "max_future_covs_length": 720,
          "max_output_length": 720,
          "max_target_count": 1,
          "max_covariate_count": 50,
          "default_output_length": 96
        }
      },
      {
        "model_id": "AutoARIMA",
        "model_type": "auto_arima",
        "category": "builtin",
        "state": "active",
        "loaded": true,
        "base_model_id": null,
        "forecast_limits": {
          "min_input_length": 16,
          "max_input_length": 2880,
          "max_future_covs_length": 2880,
          "max_output_length": 720,
          "max_target_count": 1,
          "max_covariate_count": 50,
          "default_output_length": 96
        }
      },
      {
        "model_id": "Holt-Winters",
        "model_type": "holtwinters",
        "category": "builtin",
        "state": "active",
        "loaded": true,
        "base_model_id": null,
        "forecast_limits": {
          "min_input_length": 16,
          "max_input_length": 2880,
          "max_future_covs_length": 2880,
          "max_output_length": 720,
          "max_target_count": 1,
          "max_covariate_count": 50,
          "default_output_length": 96
        }
      }
    ]
  }
}
```

**字段说明（每个模型）:**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `model_id` | string | 模型唯一标识 |
| `model_type` | string | 模型架构类型（如 `sundial`、`t5`） |
| `category` | string | `"builtin"` 或 `"user_defined"` |
| `state` | string | `"active"`、`"inactive"` 或 `"activating"` |
| `loaded` | boolean | 模型是否已加载到 GPU |
| `base_model_id` | string \| null | 该模型所扩展的内置模型 ID |
| `forecast_limits` | object \| null | 该模型在 REST 层强制的预测请求限制（见下） |

**`forecast_limits`****对象：**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `min_input_length` | int | 最小目标/历史输入长度（时间点数） |
| `max_input_length` | int | 最大目标/历史输入长度（时间点数） |
| `max_future_covs_length` | int \| null | 最大未来协变量长度；模型不接受协变量时为 `null` |
| `max_output_length` | int | 最大预测步长（输出长度） |
| `max_target_count` | int \| null | 最大目标变量数；`null` 表示无上限（多变量，如 `toto2.0`） |
| `max_covariate_count` | int | 最大协变量数（历史或未来）；模型不接受协变量时为 `0` |
| `default_output_length` | int | 省略 `output_length` 时使用的默认预测步长 |

上述取值与 `POST /forecast` 的校验规则一致，便于客户端在提交请求前自行约束输入。

---

### `POST /ai/api/v1/models/register`

从本地 URI 注册新的用户自定义模型。可选通过 `base_model_id` 指定复用已有内置模型的 pipeline 和模型代码。

**请求体:**

```json
{
  "model_id": "self_trained_timer_3p0",
  "uri": "file:///data/models/self_trained_timer_3p0",
  "base_model_id": "Timer-3.0"
}
```

**字段说明:**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `model_id` | string | **是** | — | 新模型的唯一标识 |
| `uri` | string | **是** | — | 本地 URI（`file://<路径>`），指向包含模型权重文件（`.safetensors`）和配置文件（`.json`）的目录 |
| `base_model_id` | string | 否 | `null` | 要扩展的内置模型 ID。设置后，新模型复用基座模型的 pipeline 和模型代码，仅从提供的 URI 加载权重 |

**典型用法 — 扩展内置模型:**

假设你有一组 sundial 架构的微调权重，存放在 `/data/models/self_trained_timer_3p0/` 目录下（包含 `model.safetensors` 和 `config.json`）：

```bash
curl -X POST http://localhost:10810/ai/api/v1/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "self_trained_timer_3p0",
    "uri": "file:///data/models/self_trained_timer_3p0",
    "base_model_id": "Timer-3.0"
  }'
```

注册完成后，原有的 `Timer-3.0` 模型不受影响，同时 `self_trained_timer_3p0` 可被独立加载和使用。

**成功响应 (200):**

```json
{
  "code": 200,
  "message": "Model 'self_trained_timer_3p0' registered successfully",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": { "model_id": "self_trained_timer_3p0" }
}
```

**错误响应：**

| 场景 | HTTP / `code` | 示例 `message` |
| --- | --- | --- |
| `model_id` 已存在 | **409 Conflict** | `Failed to register model: Model self_trained_timer_3p0 already exists` |
| `base_model_id` 不存在 | **404 Not Found** | `Failed to register model: Model timer_x does not exist` |
| URI 非法 / 权重缺失 | **400 Bad Request** | `Failed to register model: Model registration failed because the specified uri is invalid: ...` |
| 缺 `model_id` 或 `uri` 字段 | **422 Unprocessable Entity** | `Failed to register model: model_id is required` |
| 协调器不可达 | **503 Service Unavailable** | `Failed to register model: Coordinator unreachable: ...` |
| 未捕获异常 | **500 Internal Server Error** | `Failed to register model: ...` |

```json
{
  "code": 409,
  "message": "Failed to register model: Model self_trained_timer_3p0 already exists",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": null
}
```

---

### `POST /ai/api/v1/models/load`

将已注册的模型加载到所有可用 GPU 上（若无 GPU 则加载到 CPU）。为指定模型在每个检测到的设备上生成 ModelWorker 进程。

**请求体:**

```json
{
  "model_id": "timer_3p5",
  "replicas_per_device": 2
}
```

**字段说明:**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `model_id` | string | **是** | 要加载的模型唯一标识 |
| `replicas_per_device` | integer | 否 | 每张设备上的 worker 实例数（默认：`1`） |

**成功响应 (200):**

```json
{
  "code": 200,
  "message": "Model 'timer_3p5' loaded successfully on devices ['cuda:0', 'cuda:1']",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": { "model_id": "timer_3p5", "devices": ["cuda:0", "cuda:1"] }
}
```

**错误响应：**

| 场景 | HTTP / `code` | 示例 `message` |
| --- | --- | --- |
| 模型未注册 | **404 Not Found** | `Failed to load model: Failed to resolve model info: Model timer_3p5 does not exist` |
| 设备生成 worker 失败 | **503 Service Unavailable** | `Failed to load model: Failed to spawn ModelWorker` |
| 缺 `model_id` 字段 | **422 Unprocessable Entity** | `Failed to load model: model_id is required` |
| 协调器不可达 | **503 Service Unavailable** | `Failed to load model: Coordinator unreachable: ...` |
| 未捕获异常 | **500 Internal Server Error** | `Failed to load model: ...` |

```json
{
  "code": 404,
  "message": "Failed to load model: Failed to resolve model info: Model timer_3p5 does not exist",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": null
}
```

注：模型已经 loaded 或正在 loading 时仍返回 **200**（操作幂等）。

---

### `POST /ai/api/v1/models/unload`

从所有 GPU 上卸载模型。停止指定模型的所有 ModelWorker 进程并从路由表中移除。模型仍保持注册状态，可以再次加载。

**请求体:**

```json
{
  "model_id": "timer_3p5"
}
```

**字段说明:**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `model_id` | string | **是** | 要卸载的模型唯一标识 |

**成功响应 (200):**

```json
{
  "code": 200,
  "message": "Model 'timer_3p5' unloaded successfully",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": { "model_id": "timer_3p5" }
}
```

**错误响应：**

| 场景 | HTTP / `code` | 示例 `message` |
| --- | --- | --- |
| 模型未加载 | **409 Conflict** | `Failed to unload model: Model 'timer_3p5' is not loaded` |
| 缺 `model_id` 字段 | **422 Unprocessable Entity** | `Failed to unload model: model_id is required` |
| 协调器不可达 | **503 Service Unavailable** | `Failed to unload model: Coordinator unreachable: ...` |
| 未捕获异常 | **500 Internal Server Error** | `Failed to unload model: ...` |

```json
{
  "code": 409,
  "message": "Failed to unload model: Model 'timer_3p5' is not loaded",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": null
}
```

---

### `POST /ai/api/v1/models/delete`

删除一个用户自定义模型。如果该模型当前已加载，会先自动卸载。内置模型不可删除。模型的磁盘文件将被永久移除。

**请求体:**

```json
{
  "model_id": "self_trained_timer_3p0"
}
```

**字段说明:**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `model_id` | string | **是** | 要删除的用户自定义模型唯一标识 |

**成功响应 (200):**

```json
{
  "code": 200,
  "message": "Model 'self_trained_timer_3p0' deleted successfully",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": { "model_id": "self_trained_timer_3p0" }
}
```

**错误响应：**

| 场景 | HTTP / `code` | 示例 `message` |
| --- | --- | --- |
| 模型不存在 | **404 Not Found** | `Failed to delete model: Model self_trained_timer_3p0 does not exist` |
| 删除内置模型 | **403 Forbidden** | `Failed to delete model: Cannot delete built-in model: Timer-3.0` |
| 缺 `model_id` 字段 | **422 Unprocessable Entity** | `Failed to delete model: model_id is required` |
| 协调器不可达 | **503 Service Unavailable** | `Failed to delete model: Coordinator unreachable: ...` |
| 未捕获异常 | **500 Internal Server Error** | `Failed to delete model: ...` |

```json
{
  "code": 404,
  "message": "Failed to delete model: Model self_trained_timer_3p0 does not exist",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": null
}
```

```json
{
  "code": 403,
  "message": "Failed to delete model: Cannot delete built-in model: Timer-3.0",
  "service_info": { "timestamp": 1712345678, "version": "0.0.1.dev" },
  "data": null
}
```

---

### `GET /ai/api/v1/models/list_loaded`

列出当前处于任意加载阶段（下载中 / 加载中 / 已加载）的所有模型，包括设备分配和工作进程信息。

**响应:**

```json
{
  "code": 200,
  "message": "Success",
  "service_info": {
    "timestamp": 1712345678,
    "version": "0.0.1.dev"
  },
  "data": {
    "models": [
      {
        "model_id": "Timer-3.5",
        "status": "loading",
        "devices": ["cuda:1"],
        "endpoints": []
      },
      {
        "model_id": "Timer-3.0",
        "status": "loaded",
        "devices": ["cuda:0", "cuda:1"],
        "endpoints": [
          { "device": "cuda:0", "worker_pid": 12345 },
          { "device": "cuda:1", "worker_pid": 12346 }
        ]
      },
      {
        "model_id": "Chronos-2",
        "status": "loaded",
        "devices": ["cuda:0"],
        "endpoints": [
          { "device": "cuda:0", "worker_pid": 12347 }
        ]
      }
    ]
  }
}
```

**字段说明（每个模型）:**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `model_id` | string | 模型唯一标识 |
| `status` | string | 加载阶段：`"downloading"`（正在下载权重，尚无 worker）、`"loading"`（worker 正在启动，尚未就绪）、`"loaded"`（所有 worker 已就绪） |
| `devices` | 字符串数组 | 设备标识符（如 `"cuda:0"`）；`downloading` 状态下为空数组 |
| `endpoints[i].device` | string | 该端点的设备 |
| `endpoints[i].worker_pid` | integer | ModelWorker 进程的 PID；仅 `loaded` 状态的设备会出现在 endpoints 中 |

**错误响应：** coordinator 不可达 → **503**；未捕获异常 → **500**。`/models/list` 同上。

---

## 数据集

数据集模块同时提供**评估**与**治理**两类能力，输入既支持内联 JSON（columns + rows）也支持本地 TsFile 路径。评估维度衡量数据"长得怎么样"，治理维度则在数据上做一次清洗或变换并返回新数据集。

### 数据集评估

#### `POST /ai/api/v1/dataset/evaluate/execute`

时序数据质量评估端点。运行选定的评估维度，返回综合分数与每条序列的明细。

当前支持以下 3 个评估维度：

| 维度 | 说明 |
| --- | --- |
| `integrity` | 统一的时间戳完整性检查 — 返回完整性（completeness）、一致性（consistency）、时效性（timeliness）三个子分数 |
| `forecastability` | 基于频域的可预测性度量，使用谱熵（Goerg 2013）计算 |
| `pearson` | 皮尔逊相关系数 — 返回序列间的相关矩阵 |

**请求体：**

```json
{
  "input": {
    "inline": {
      "columns": ["time", "temperature", "humidity"],
      "data": [
        ["2024-01-01T00:00:00", 20.1, 65.0],
        ["2024-01-01T01:00:00", 20.4, 63.5],
        ["2024-01-01T02:00:00", 19.8, 64.2],
        ["2024-01-01T03:00:00", null, 62.8],
        ["2024-01-01T04:00:00", 21.2, 66.1]
      ],
      "time_col": "time"
    }
  },
  "dimensions": ["integrity", "forecastability"],
  "params": {"downtime": true}
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input` | 对象 | **是** | — | 数据源。`inline` 与 `tsfile` 二选一 |
| `input.inline` | 对象 | 二选一 | — | 内联时间序列数据 |
| `input.inline.columns` | 字符串数组 | 是 | — | 列名（包括时间列） |
| `input.inline.data` | 二维数组 | 是 | — | 行导向数据 |
| `input.inline.time_col` | string | 否 | — | 时间列名称 |
| `input.tsfile` | 对象 | 二选一 | — | 本地 TsFile 数据源 |
| `input.tsfile.path` | string | 是 | — | TsFile 文件或目录路径 |
| `input.tsfile.sample_per_file` | integer | 否 | `0` | 从每个 TsFile 中随机采样的序列数；`0` 表示读取全部序列 |
| `dimensions` | 字符串数组 | 否 | 全部 3 个 | 要运行的评估维度列表。可选值：`"integrity"`、`"forecastability"`、`"pearson"`。省略则运行全部 |
| `params` | 对象 | 否 | — | 转发给评估维度的参数。支持的 key：`downtime`（布尔值，用于 `integrity`）；`targets`（字符串数组，用于 `pearson`，将相关性矩阵限定为目标序列对应的行） |

**成功响应 (200)：**

```json
{
  "code": 200,
  "message": "Dataset evaluation completed successfully",
  "service_info": { "timestamp": 1712345678, "version": "0.2.1" },
  "data": {
    "overall_score": 82.5,
    "dimension_scores": {
      "integrity": 90.0,
      "forecastability": 75.0
    },
    "series_reports": [
      {
        "series_id": "temperature",
        "integrity": {
          "completeness": 80.0,
          "consistency": 100.0,
          "timeliness": 100.0,
          "score": 93.33
        },
        "forecastability": {
          "spectral_entropy": 0.65,
          "score": 35.0
        }
      }
    ]
  }
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `overall_score` | number | 综合质量评分（0–100） |
| `dimension_scores` | 对象 | 各评估维度的评分（仅包含请求中指定的维度） |
| `series_reports` | 数组 | 每条序列的详细评估报告 |
| `series_reports[i].series_id` | string | 标识该序列的列名 |
| `series_reports[i].integrity` | 对象 | 子分数：`completeness`、`consistency`、`timeliness`，以及汇总 `score` |
| `series_reports[i].forecastability` | 对象 | `spectral_entropy` 值及其衍生 `score` |
| `series_reports[i].pearson` | 对象 | 该序列的皮尔逊相关矩阵条目 |

**错误响应：**

| 场景 | HTTP / `code` | 示例 `message` |
| --- | --- | --- |
| `inline` 与 `tsfile` 都未提供 / 维度名非法 / 参数越界 | **422 Unprocessable Entity** | `Either 'input.inline' or 'input.tsfile' must be provided` |
| TsFile 路径不存在 | **404 Not Found** | （来自底层 `FileNotFoundError` 的原文） |
| 未捕获异常 | **500 Internal Server Error** | `Dataset evaluation failed: ...` |

---

#### `GET /ai/api/v1/dataset/evaluate/list_dimensions`

列出所有可用的时序数据评估维度。

**响应：**

```json
{
  "code": 200,
  "message": "Success",
  "service_info": { "timestamp": 1712345678, "version": "0.2.1" },
  "data": {
    "dimensions": [
      {
        "name": "integrity",
        "description": "统一的时间戳完整性检查，涵盖完整性、一致性和时效性",
        "supported_params": [
          {"name": "downtime", "type": "boolean", "default": true, "description": "计算时效性时是否考虑已知停机时段"}
        ]
      },
      {
        "name": "forecastability",
        "description": "基于频域的可预测性度量，使用谱熵（Goerg 2013）计算",
        "supported_params": []
      },
      {
        "name": "pearson",
        "description": "皮尔逊相关系数，返回序列间的相关矩阵",
        "supported_params": [
          {"name": "targets", "type": "array of strings", "default": null, "description": "限制相关性矩阵的行仅为这些目标序列。score 与 series_details 也仅覆盖目标序列；省略则包含全部序列。"}
        ]
      }
    ]
  }
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `dimensions` | 数组 | 可用的评估维度列表 |
| `dimensions[i].name` | string | 维度名称 |
| `dimensions[i].description` | string | 维度的人类可读描述 |
| `dimensions[i].supported_params` | 数组 | 支持的参数列表，每项包含 `name`、`type`、`default` 和 `description` |

---

### 数据集治理

#### `POST /ai/api/v1/dataset/govern/execute`

时序数据质量治理端点。指定一个治理维度，对输入数据集执行一次清洗或变换，返回新数据集。输入与评估端点完全一致（`inline` 或 `tsfile`），输出默认以 `inline` 形式返回；如需写出为 TsFile，可设置 `output_tsfile_path`。

当前支持以下 5 个治理维度：

| 维度 | 说明 |
| --- | --- |
| `timestamp_repair` | 时间戳修复 — 检测并修复时间戳缺失、冗余、错乱 |
| `causal_mean_imputation` | 因果均值填充 — 使用之前窗口的均值填充 NaN |
| `flat_series_removal` | 平稳序列剔除 — 移除方差低于阈值的序列 |
| `zscore_normalization` | Z-score 归一化 — 按序列做标准化 |
| `extreme_value_clipping` | 极值裁剪 — 将偏离均值 `threshold` 倍标准差以外的点拉回边界 |

**请求体：**

```json
{
  "input": {
    "inline": {
      "columns": ["time", "temperature", "humidity"],
      "data": [
        ["2024-01-01T00:00:00", 20.1, 65.0],
        ["2024-01-01T01:00:00", 20.4, 63.5],
        ["2024-01-01T02:00:00", 19.8, 64.2],
        ["2024-01-01T03:00:00", null, 62.8],
        ["2024-01-01T04:00:00", 21.2, 66.1]
      ],
      "time_col": "time"
    }
  },
  "dimension": "causal_mean_imputation",
  "params": {}
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `input` | 对象 | **是** | — | 数据源。`inline` 与 `tsfile` 二选一（结构同评估端点） |
| `dimension` | string | **是** | — | 要应用的治理维度名称。可用值通过 `/dataset/govern/list_dimensions` 获取 |
| `params` | 对象 | 否 | `{}` | 转发给治理维度的参数（如 `extreme_value_clipping` 的 `threshold`） |
| `output_tsfile_path` | string | 否 | — | 如设置，清洗结果会写入该 TsFile 路径；否则以 `inline` 形式返回 |

**成功响应 (200)（inline 输出）：**

```json
{
  "code": 200,
  "message": "Dataset governance completed successfully",
  "service_info": { "timestamp": 1712345678, "version": "0.2.1" },
  "data": {
    "dimension": "causal_mean_imputation",
    "summary": {
      "causal_mean_imputation": {"changes_count": 1}
    },
    "inline": {
      "columns": ["time", "temperature", "humidity"],
      "data": [
        [1704067200000, 20.1, 65.0],
        [1704070800000, 20.4, 63.5],
        [1704074400000, 19.8, 64.2],
        [1704078000000, 20.1, 62.8],
        [1704081600000, 21.2, 66.1]
      ],
      "time_col": "time"
    },
    "elapsed_ms": 6.6
  }
}
```

写入 TsFile 时，响应将以 `output_tsfile_path` 替代 `inline`：

```json
{
  "data": {
    "dimension": "causal_mean_imputation",
    "summary": {"causal_mean_imputation": {"changes_count": 1}},
    "output_tsfile_path": "/data/cleaned.tsfile",
    "elapsed_ms": 12.4
  }
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `dimension` | string | 本次应用的治理维度名称 |
| `summary` | 对象 | 每个维度的变更摘要（当前仅含 `changes_count`） |
| `inline` | 对象 | 治理后的数据（`columns` / `data` / `time_col`，时间列以毫秒整数返回）。未设置 `output_tsfile_path` 时返回 |
| `output_tsfile_path` | string | 写入的 TsFile 路径。设置 `output_tsfile_path` 时返回 |
| `elapsed_ms` | number | 本次治理总耗时（毫秒，含数据解析、维度执行、结果序列化或 TsFile 写入） |

**错误响应：** 与 `/dataset/evaluate/execute` 同语义（422 / 404 / 500）。

---

#### `GET /ai/api/v1/dataset/govern/list_dimensions`

列出所有可用的时序数据治理维度。

**响应：**

```json
{
  "code": 200,
  "message": "Success",
  "service_info": { "timestamp": 1712345678, "version": "0.2.1" },
  "data": {
    "dimensions": [
      {"name": "timestamp_repair", "description": "...", "supported_params": []},
      {"name": "causal_mean_imputation", "description": "...", "supported_params": []},
      {"name": "flat_series_removal", "description": "...", "supported_params": [{"name": "min_variance", ...}]},
      {"name": "zscore_normalization", "description": "...", "supported_params": []},
      {"name": "extreme_value_clipping", "description": "...", "supported_params": [{"name": "threshold", ...}]}
    ]
  }
}
```

字段含义同 `evaluate/list_dimensions`。

---

## 监控

### `GET /metrics`

Prometheus 指标端点。**仅在配置文件中****`enable_prometheus_metrics=true`****时可用**。

指标端点在**独立端口**上提供（默认 `8080`，可通过 `timer_rest_service_metrics_port` 配置），以隔离抓取流量与 API 流量。

返回标准 Prometheus 文本格式的指标。所有维度名称使用 `timer_service:` 前缀。

**主要指标:**

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `timer_service:request_total` | Counter | 按模型和状态统计的总请求数 |
| `timer_service:e2e_request_latency_seconds` | Histogram | 端到端延迟 |
| `timer_service:inference_latency_seconds` | Histogram | 模型前向传播延迟 |
| `timer_service:request_active` | Gauge | 正在处理的请求数 |
| `timer_service:model_workers_loaded` | Gauge | 已加载的 ModelWorker 进程数 |
| `timer_service:worker_restart_total` | Counter | 自动重启次数 |
| `timer_service:gpu_memory_used_bytes` | Gauge | GPU 显存使用量 |
| `timer_service:gpu_utilization_percent` | Gauge | GPU 计算利用率 % |

预构建的 Grafana 仪表盘位于 `resources/grafana/timer-rest-service-dashboard.json`。

---

## 通用响应信封

除 `/health/*` 探针外，所有 `/ai/api/v1/*` 端点都返回如下统一结构：

```json
{
  "code": 200,
  "message": "...",
  "service_info": {
    "timestamp": 1712345678,
    "version": "0.0.1.dev"
  },
  "data": { ... }
}
```

| 字段 | 说明 |
| --- | --- |
| `code` | **HTTP 状态码**，与响应状态行完全一致（详见下表）。上游服务可只看 HTTP 状态行做路由/告警，无需解析 body |
| `message` | 人类可读的状态信息（按构建语言翻译） |
| `service_info.timestamp` | Unix 纪元秒数 |
| `service_info.version` | 服务版本字符串 |
| `data` | 端点特定载荷（错误时通常为 `null`，部分端点为 `{"results": []}` 等占位） |

### HTTP 状态码语义

| `code` / HTTP | 含义 | 典型场景 |
| --- | --- | --- |
| **200** OK | 成功 | 所有正常路径；幂等操作（如 `load` 已 loaded） |
| **400** Bad Request | 入参不合法 | URI 非法、权重路径不存在 |
| **403** Forbidden | 操作被禁止 | 试图删除内置模型 |
| **404** Not Found | 引用的资源不存在 | `model_id` / `base_model_id` / 数据文件未找到 |
| **409** Conflict | 资源状态冲突 | 模型已存在、模型未加载（unload 时） |
| **422** Unprocessable Entity | 请求 parse 成功但语义校验失败 | `output_length` / `input_length` 越界；目标变量数 ≠ 1；缺必填字段 |
| **500** Internal Server Error | 未捕获异常 | 推理/序列化/未知异常 |
| **503** Service Unavailable | 服务暂时不可用 | 模型未加载（forecast 时）；coordinator 不可达；worker 拉起失败 |

> **`/health/*`****探针不使用本信封**：探针端点直接返回 `{"status": "<state>", "version": "<ver>", "reason": "..."}` 的裸 JSON，便于 Kubernetes 等编排系统消费。

---

## 自动生成文档

服务运行时，交互式 API 文档可通过以下地址访问:

| URL | 格式 |
| --- | --- |
| `http://<host>:<port>/docs` | Swagger UI |
| `http://<host>:<port>/redoc` | ReDoc |
| `http://<host>:<port>/openapi.json` | OpenAPI 3.x JSON |
