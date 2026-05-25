# TSBenchmark 用户手册

本文档面向本地 MVP 使用者，说明如何启动 TSBenchmark、加载真实 CSV 数据集、运行模型评测、查看榜单/报告/样本预测结果，以及如何排查常见问题。

## 1. 适用范围

当前手册覆盖仓库内的 MVP Web 平台：

- 后端：FastAPI + SQLModel + SQLite。
- 前端：Vue + Vite。
- 数据集：仅支持 CSV。
- 目标列：仅支持单变量单 target column。
- 模型：内置 5 个可复现 stub 模型，分别是 Timer 3.5、Timer 3.0、Chronos 2、toto、TimesFM 2.5。
- 指标：MSE、MAE。
- 使用场景：本地可信环境，不包含登录、权限管理或生产模型服务接入。

## 2. 环境准备

在项目根目录执行命令。建议先确认以下工具可用：

```bash
uv --version
node --version
npm --version
```

前端依赖 Vite 7，Node.js 必须为 `20.19+` 或 `22.12+`。如果 `node --version` 显示 `v15.14.0`、`v16` 或 `v18`，请先切换到受支持版本。

首次运行时，启动脚本会自动准备后端虚拟环境和前端依赖。如果需要手动安装，也可以执行：

```bash
cd backend && uv sync
cd ../frontend && npm install
```

## 3. 启动、停止和查看状态

项目提供统一脚本管理前后端：

```bash
./scripts/start-system.sh
./scripts/status-system.sh
./scripts/stop-system.sh
```

默认地址：

- 前端页面：`http://127.0.0.1:5173`
- 后端 API：`http://127.0.0.1:8000`

脚本会在 `.tsbenchmark-system/` 下写入：

- `backend.pid` / `frontend.pid`：进程 ID。
- `backend.log` / `frontend.log`：启动日志和运行日志。

常用端口覆盖方式：

```bash
TSBENCHMARK_BACKEND_PORT=8010 TSBENCHMARK_FRONTEND_PORT=5174 ./scripts/start-system.sh
```

常用运行目录覆盖方式：

```bash
TSBENCHMARK_SYSTEM_DIR=/tmp/tsbenchmark-system ./scripts/status-system.sh
```

停止系统：

```bash
./scripts/stop-system.sh
```

## 4. 数据文件要求

上传的 CSV 必须满足以下规则：

- 编码为 UTF-8 或 UTF-8 BOM。
- 必须有 header row。
- 列名必须唯一。
- 分隔符支持逗号、制表符和分号，系统会自动识别。
- 必须显式选择时间列和 target 列。
- 时间格式支持 ISO 8601、`YYYY-MM-DD`、`YYYY-MM-DD HH:mm:ss`。
- 时间列必须严格单调递增，不允许重复时间戳。
- 时间间隔必须等距，系统会自动推断 frequency。
- MVP 只允许选择 1 个 target column。
- target 值必须能安全转换为有限浮点数，不允许缺失、NaN 或 Inf。
- `context_length + horizon` 不能超过数据行数。
- `stride` 默认等于 `horizon`，且必须为正数。

仓库中有一个可用于试跑的示例文件：

```text
backend/tests/fixtures/valid_hourly_20.csv
```

## 5. 完成一次评测

打开前端页面：

```text
http://127.0.0.1:5173
```

按以下步骤操作：

1. 在上传区域选择 CSV 文件。
2. 确认页面展示了预览行和列名。
3. 选择时间列，通常为 `time`。
4. 选择唯一 target 列，通常为 `target`。
5. 配置切分参数：
   - `Context`：历史窗口长度，例如 `6`。
   - `Horizon`：预测长度，例如 `3`。
   - `Stride`：窗口滑动步长，例如 `3`。
6. 点击 `Load shard`，等待页面显示 sample 数量。
7. 点击 `Create track`，系统会基于真实数据 shard 创建真实数据评测赛道。
8. 在模型列表中选择一个或多个模型。
9. 点击 `Run`，系统会创建 benchmarking run 并轮询进度。
10. 运行完成后，页面会出现 Report 链接。

使用示例 CSV 和参数 `context=6 / horizon=3 / stride=3` 时，应生成 4 个 sample。

## 6. 查看结果

### 6.1 榜单

榜单按 Track 维度展示模型成绩。后端接口支持：

```text
GET /tracks/{track_id}/ranking?metric=mse&policy=latest_valid_result
```

可选参数：

- `metric=mse|mae`
- `policy=latest_valid_result|best_result`

排序方向为 lower is better。

### 6.2 报告

报告按 run 生成，保存为 JSON 产物，默认在：

```text
runtime/reports/{run_id}.json
```

报告包含：

- 模型级指标表。
- task 摘要。
- sample forecast 链接。
- 取消运行时的取消原因。

接口：

```text
GET /reports/{report_id}
```

### 6.3 样本预测

样本预测视图用于检查单个 sample 上多个模型的预测曲线和指标。

接口：

```text
GET /samples/{sample_id}/forecast?run_id={benchmarking_run_id}
```

返回内容包括：

- history timestamps。
- future timestamps。
- target history。
- target future。
- 每个模型的 forecast。
- 每个模型的 sample-level MSE/MAE。
- 模型失败时的错误信息。

## 7. 运行产物

默认运行目录是 `runtime/`，主要结构如下：

```text
runtime/
  uploads/
  samples/
  forecasts/
  reports/
  tsbenchmark.db
```

说明：

- `uploads/` 保存上传后的 CSV 文件。
- `samples/` 保存 materialized sample JSONL。
- `forecasts/` 保存模型预测 JSONL。
- `reports/` 保存 run summary JSON。
- `tsbenchmark.db` 保存 SQLite 元数据。

如需使用隔离运行目录：

```bash
TSBENCHMARK_RUNTIME_DIR=/tmp/tsbenchmark-runtime \
TSBENCHMARK_DATABASE_URL=sqlite:////tmp/tsbenchmark-runtime/tsbenchmark.db \
./scripts/start-system.sh
```

## 8. 常见问题

### 8.1 启动失败

先查看状态：

```bash
./scripts/status-system.sh
```

再查看日志：

```bash
sed -n '1,160p' .tsbenchmark-system/backend.log
sed -n '1,160p' .tsbenchmark-system/frontend.log
```

常见原因：

- 端口已被占用。
- `uv`、`node` 或 `npm` 未安装。
- 依赖安装被网络或权限限制阻止。
- 上一次异常退出留下了 stale PID；再次执行 `stop-system.sh` 后重试。

### 8.2 页面能打开但 API 请求失败

确认后端是否运行：

```bash
curl http://127.0.0.1:8000/models
```

确认前端代理配置仍指向后端：

```text
frontend/vite.config.ts
```

默认前端通过 `/api` 代理到 `http://127.0.0.1:8000`。

### 8.3 CSV 上传后加载失败

检查错误响应中的 `error_code` 和 `details`。常见错误包括：

- `csv_time_parse_failed`：时间列无法解析。
- `csv_duplicate_timestamp`：时间戳重复。
- `csv_time_not_monotonic`：时间列不是严格递增。
- `csv_time_not_equidistant`：时间间隔不等距。
- `csv_single_target_only`：选择了多个 target。
- `csv_target_missing`：target 值缺失。
- `csv_target_not_float`：target 无法转换为浮点数。
- `csv_target_not_finite`：target 是 NaN 或 Inf。
- `split_length_exceeds_rows`：切分窗口长度超过数据行数。

### 8.4 想重新开始一次干净测试

先停止服务：

```bash
./scripts/stop-system.sh
```

再清理运行产物：

```bash
rm -rf runtime .tsbenchmark-system
```

然后重新启动：

```bash
./scripts/start-system.sh
```

## 9. 模型推理服务与本地桩程序

评测的实际推理通过外部 **timer-rest-service** 的 REST API 完成（契约见 [`docs/reference/rest-api.md`](../reference/rest-api.md)）。服务地址被抽象成配置项，本地无真实服务时可启动桩程序顶上。

### 9.1 配置项

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TSBENCHMARK_TIMER_SERVICE_BASE_URL` | `http://127.0.0.1:10810` | 服务前缀 `http://<host>:<port>` |
| `TSBENCHMARK_TIMER_SERVICE_API_PREFIX` | `/ai/api/v1` | 文档约定的统一路径前缀 |
| `TSBENCHMARK_MODEL_ADAPTER` | `rest` | `rest`=走 HTTP；`stub`=进程内确定性桩，无需网络 |

### 9.2 本地启动桩服务

桩程序实现了文档的精简子集（`/forecast`、`/models/list`、`/health/*`），默认监听 `127.0.0.1:10810`，与 `base_url` 默认值一致，因此后端开箱即用：

```bash
./scripts/stub-service.sh start     # 启动（日志写入 .tsbenchmark-system/stub-service.log）
./scripts/stub-service.sh status
./scripts/stub-service.sh stop
```

桩的预测是确定性的（last-value + 按 model_id 计算的偏置 + 固定种子噪声），便于可复现演示。

### 9.3 离线 / 不起 HTTP

无需启动桩服务时，可让后端使用进程内桩：

```bash
TSBENCHMARK_MODEL_ADAPTER=stub ./scripts/start-system.sh
```

### 9.4 指向真实服务

```bash
TSBENCHMARK_TIMER_SERVICE_BASE_URL=http://<gpu-host>:<port> ./scripts/start-system.sh
```

## 10. 开发者验证命令

后端测试：

```bash
cd backend && uv run pytest
```

前端测试：

```bash
cd frontend && npm test
```

前端 smoke 测试：

```bash
cd frontend && npm run test:e2e
```

脚本测试：

```bash
bash scripts/tests/test_system_scripts.sh
```

这些命令通过后，说明当前 MVP 的主要加载、评测、报告、榜单、样本预测和脚本启停行为处于可用状态。
