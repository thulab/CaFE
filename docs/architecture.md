# TSBenchmark Dynamic Benchmark Architecture

## 设计映射

该实现直接对应 `detail.pdf` 与 `paper.png` 中的四个后端核心模块和一个在线系统：

- `Dataset Manager`
  负责动态批次生成、变换、校验和文件系统持久化。
- `Model Manager`
  负责注册模型元信息，并通过 adapter 绑定到可执行的 Hugging Face / 兼容基线路径。
- `Mission Manager / Executor`
  负责把“模型 + 批次 + 赛道”组合成最小任务单元并执行评测。
- `Reporter`
  负责聚合指标、定位 bad case、生成报告与排行榜。
- `Online System`
  以前端独立应用消费后端 API，模拟“客户端开源 / 服务端闭源”的部署边界。

## 后端系统

后端技术栈：`FastAPI + Pydantic + 文件系统存储`

核心 API：

- `GET /api/v1/tracks`
- `GET /api/v1/models`
- `POST /api/v1/models/register`
- `POST /api/v1/datasets/generate`
- `POST /api/v1/datasets/load/csv`
- `GET /api/v1/datasets/batches`
- `POST /api/v1/tasks/run`
- `GET /api/v1/tasks`
- `GET /api/v1/reports/{report_id}`
- `GET /api/v1/leaderboard`
- `GET /api/v1/overview`

数据加载模块使用独立目录 `backend/app/data_management/data_loader/` 组织：

- `base.py` 定义统一的 `DatasetLoader` 抽象。
- `registry.py` 负责按 `source_type` 注册与分发 loader。
- `csv_loader.py` 提供当前的 CSV 实现。

服务层只依赖 dataloader 抽象，不依赖具体文件格式；未来扩展 `parquet`、`jsonl` 或数据库来源时，只需要新增 loader 并注册即可。

数据处理模块使用独立目录 `backend/app/data_management/processors/` 组织：

- `base.py` 定义统一的 `DataProcessor` 抽象。
- `registry.py` 负责按 `processor_type` 注册与分发 processor。
- `pipeline.py` 负责按请求中的配置顺序执行 processor 链。
- `builtin.py` 提供当前内置的通用 processor。

数据验证模块使用独立目录 `backend/app/data_management/validators/` 组织：

- `base.py` 定义统一的 `DataValidator` 抽象和验证上下文。
- `registry.py` 负责注册 validator。
- `pipeline.py` 负责聚合 validator 输出并生成最终 `ValidationReport`。
- `builtin.py` 提供当前内置的通用校验规则。

当前外部数据链路为：`loader -> data processor pipeline -> data validator pipeline -> batch persistence`。
当前动态生成链路为：`sample generation -> data validator pipeline -> fail then regenerate -> batch persistence`。

当前实现的赛道：

- `forecast_accuracy`
- `covariate_robustness`
- `noise_robustness`
- `cost_intensive`

当前默认内置模型：

- `amazon-chronos-2`
- `thuml-sundial-base-128m`

## 前端系统

前端技术栈：`Flask + Jinja2`

职责：

- 展示赛道、批次、最近任务、排行榜。
- 提供生成动态批次与运行评测任务的最小表单。
- 只通过 HTTP provider 调用后端，不依赖后端内部类。

## 冒烟路径

`scripts/smoke_test.py` 验证以下链路：

1. 启动临时后端应用。
2. 获取模型目录。
3. 生成协变量赛道数据批次。
4. 运行内置 Hugging Face 模型评测任务。
5. 拉取报告与排行榜。
6. 启动临时前端应用。
7. 通过前端页面再次触发“生成批次 / 运行任务”。

这条路径覆盖了当前系统的最小闭环。 
