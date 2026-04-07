# TSBenchmark

基于 `detail.pdf` 和 `paper.png` 实现的时序预测动态 Benchmark 系统原型。当前版本包含两个独立系统：

- 后端：`FastAPI`，负责动态数据集、模型注册、任务执行、报告和排行榜。
- 前端：`Flask`，拆分为用户页面与管理页面。

## 架构落地

整体设计与讨论材料一一对应：

- `Dataset Manager`：动态生成纯合成多周期数据，按赛道执行变换并校验。
- `Model Manager`：管理模型元信息和 adapter；除内置桩模型外，支持通过 Hugging Face repo 提交、加载和执行模型，当前优先支持 `amazon/chronos-2`。
- `Mission Manager / Executor`：把“模型 + 批次”组合为可回溯任务并执行。
- `Reporter`：聚合 MSE、MAE、sMAPE、延迟、Token 成本，输出 bad case 与总结。
- `Online System`：独立前端系统，仅通过 API 访问后端。

详细设计见 [docs/architecture.md](docs/architecture.md)。

## 目录

```text
backend/app/     FastAPI 后端
frontend/        Flask 前端
conf/            系统统一配置
scripts/         启停与兼容测试入口
test/            单元测试与集成测试
runtime/         运行时生成内容
docs/            架构说明
```

## 配置

系统的运行参数已经统一收敛到 `conf/system.toml`，包括：

- 后端/前端 host、port、timeout
- 启停脚本的 PID、日志、健康检查和关闭等待参数
- 赛道默认参数、数据生成参数、评分阈值、报告阈值
- 前端表单默认值和页面展示条目上限

默认情况下，后端、前端和脚本都会读取 `conf/system.toml`。如果需要切换配置文件，可设置环境变量：

```bash
export TSBENCHMARK_CONF=/absolute/path/to/system.toml
```

## 启动

后端：

```bash
python -m backend.app.main
```

前端：

```bash
python -m frontend.app
```

也可以直接使用脚本后台启动/关闭整套系统：

```bash
bash scripts/start_system.sh
bash scripts/stop_system.sh
```

启动脚本会将 PID 和日志写入 `runtime/system/`。

默认配置下：

- 用户页面：`http://127.0.0.1:8501/`
- 管理页面：`http://127.0.0.1:8501/admin`

如果修改了 `conf/system.toml` 中的 host/port，启动地址会随配置变化。前端仍可通过环境变量 `TSBENCHMARK_BACKEND_URL` 额外覆盖后端地址。

数据加载模块位于 `backend/app/data_management/data_loader/`，当前提供统一的 `DatasetLoader` 抽象，底层已实现 `CSV` 格式加载，后续可以按相同接口扩展更多来源。

数据处理模块位于 `backend/app/data_management/processors/`，负责在数据被 loader 读入后做统一变换。当前内置 `identity`、`scale`、`clip`、`covariate_filter` 四种 processor，并通过通用接口支持按顺序串联处理。

数据验证模块位于 `backend/app/data_management/validators/`，负责在批次最终落盘前验证数据集是否有效。当前默认执行上下文长度、预测长度、有限值、低方差等校验；对于系统自动生成的数据，如果验证失败会自动重新生成，直到通过或超过重试上限。

CSV 批次加载 API：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/datasets/load/csv \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type": "csv",
    "csv_path": "/absolute/path/to/data.csv",
    "track": "forecast_accuracy",
    "context_length": 96,
    "horizon": 24,
    "sample_id_column": "sample_id",
    "step_column": "step",
    "target_column": "target",
    "covariate_columns": ["calendar_signal"],
    "processors": [
      {"processor_type": "scale", "params": {"factor": 10}},
      {"processor_type": "clip", "params": {"min_value": -20, "max_value": 20}}
    ]
  }'
```

如需启用 Hugging Face 模型加载，可安装可选依赖：

```bash
pip install -e .[huggingface]
```

如需运行测试，可安装测试依赖：

```bash
pip install -e .[test]
```

`amazon/chronos-2` 使用 `chronos-forecasting` 提供的 `Chronos2Pipeline` 进行推理，并透传 Hugging Face Transformers 的 `from_pretrained(...)` 参数。

真实端到端验证现在位于 `test/integration/test_verify_chronos2_e2e.py`，默认通过环境变量控制是否纳入测试发现；保留 `scripts/verify_chronos2_e2e.py` 作为兼容入口：

```bash
python scripts/verify_chronos2_e2e.py
```

也可以直接按目录执行测试：

```bash
python -m unittest discover -s test -t . -p 'test_*.py'
```

## 测试

```bash
python scripts/smoke_test.py
```

冒烟流程已经迁移到 `test/integration/test_smoke_flow.py`，脚本入口会直接复用该测试逻辑。测试数据会写入仓库内的 `test/.tmp/` 临时目录，执行完后自动清理，不污染正式 `runtime/` 目录。

当前测试覆盖以下几类行为：

- 用户页面提交 Hugging Face 模型
- 管理页面加载模型
- 管理页面生成批次与运行任务
- 用户页面查看总榜与分赛道榜单
- CSV loader、processor、validator 的单元测试
- Data/Model/Task/Leaderboard Manager 的核心行为单元测试
