# 开发者手册 · 数据模型篇

> 返回[开发者手册总览](./README.md) ｜ 相关：[架构与关键流程](./key-flows.md)

本篇覆盖 TSBenchmark 后端的**全部实体设计**，以代码为唯一事实来源。所有字段表均按实体定义文件逐字核对，状态/类型枚举值取自 `services/` 中的实际赋值，不做臆测。

> **分层约定**：`routes` 校验并委派、`services` 承载行为、`models` 仅做持久化。本文聚焦 `models`（SQLModel 表实体）、`schemas`（传输层 DTO）与落盘产物（非 DB 的 JSONL / JSON 结构）。

## 0. 通用约定

### 0.1 主键与 ID

所有 SQLModel 表实体的主键都使用 `Field(default_factory=new_id, primary_key=True)`，其中 `new_id` 即 UUID4 字符串（`backend/app/core/ids.py:4`）：

```python
def new_id() -> str:
    return str(uuid4())
```

### 0.2 时间字段

`created_at` / `updated_at` / `*_at` 等时间字段默认工厂为 `utc_now`，返回带 UTC 时区的 `datetime`（`backend/app/core/time.py:4`）：

```python
def utc_now() -> datetime:
    return datetime.now(UTC)
```

### 0.3 JSON 列

凡是 `list[...]` / `dict[...]` 类型且需要持久化的字段，都通过 `sa_column=Column(JSON)` 落到 SQLite 的 JSON 列。本文在字段表中以「JSON 列」标注。

### 0.4 建表与元数据注册

`init_db(engine)` 通过导入所有 model 模块触发 SQLModel 元数据注册，再调用 `SQLModel.metadata.create_all(engine)` 建表（`backend/app/db/init_db.py:13`）。没有显式的外键约束（SQLite + SQLModel 此处仅以 `index=True` 字段表达逻辑外键），关系靠 service 层维护。

---

## 1. 总览

### 1.1 SQLModel 表实体清单（17 个）

| 实体 | 文件 | 职责 |
| --- | --- | --- |
| `DatasetManifest` | `backend/app/models/dataset.py:11` | 描述真实数据源（CSV 文件、列配置） |
| `DatasetLoadJob` | `backend/app/models/dataset.py:28` | 一次数据加载/校验/切分/索引任务 |
| `Shard` | `backend/app/models/dataset.py:47` | 统一数据单元；MVP 用 `shard_type=real` |
| `SampleIndex` | `backend/app/models/sample.py:11` | 样本切片位置 + 物化产物引用 |
| `CapabilityBlock` | `backend/app/models/benchmark.py:11` | 统一能力块；MVP 用 `block_type=real` |
| `Track` | `backend/app/models/benchmark.py:27` | 评测赛道 |
| `Model` | `backend/app/models/model_registry.py:10` | 可评测模型 + adapter 配置 |
| `BenchmarkingRun` | `backend/app/models/benchmark.py:41` | 一次评测执行 |
| `Unit` | `backend/app/models/benchmark.py:63` | 某模型在某次 run 中的完整结果 |
| `Task` | `backend/app/models/benchmark.py:76` | 某模型在某 `CapabilityBlock` 上的结果 |
| `ForecastArtifact` | `backend/app/models/benchmark.py:94` | 预测产物位置与 schema |
| `RunEvent` | `backend/app/models/benchmark.py:108` | run/unit/task 过程事件与日志 |
| `MetricDefinition` | `backend/app/models/metric.py:10` | 指标注册表（MVP: mse / mae） |
| `MetricResult` | `backend/app/models/metric.py:24` | 统一多层级指标结果 |
| `Report` | `backend/app/models/report.py:11` | 基础评测报告 |
| `RankingList` | `backend/app/models/ranking.py:10` | 每条 Track 一个的榜单定义 |
| `RankingEntry` | `backend/app/models/ranking.py:21` | 榜单中的模型成绩行 |

### 1.2 ER 图

```mermaid
erDiagram
    DatasetManifest ||--o{ DatasetLoadJob : "dataset_manifest_id"
    DatasetManifest ||--o| Shard : "dataset_manifest_id (real 唯一)"
    DatasetLoadJob ||--o| Shard : "load_job_id / output_shard_id"
    CapabilityBlock ||--o{ Shard : "capability_block_id"
    Shard ||--o{ SampleIndex : "shard_id"

    Track ||--o{ CapabilityBlock : "track_id"
    Track ||--|| RankingList : "track_id (1:1)"
    Track ||--o{ BenchmarkingRun : "track_id"
    Track ||--o{ Report : "track_id"

    BenchmarkingRun ||--o{ Unit : "benchmarking_run_id"
    BenchmarkingRun ||--o{ Task : "benchmarking_run_id"
    Unit ||--o{ Task : "unit_id"
    Task ||--o{ ForecastArtifact : "task_id"
    Shard ||--o{ ForecastArtifact : "shard_id"
    BenchmarkingRun ||--o| Report : "benchmarking_run_id"
    BenchmarkingRun ||--o{ RunEvent : "benchmarking_run_id"

    Model ||--o{ Unit : "model_id"
    Model ||--o{ Task : "model_id"

    MetricDefinition ||--o{ MetricResult : "metric_id"
    BenchmarkingRun ||--o{ MetricResult : "benchmarking_run_id"
    Unit ||--o{ MetricResult : "unit_id (level=unit)"
    Task ||--o{ MetricResult : "task_id (level=task)"
    Shard ||--o{ MetricResult : "shard_id (level=shard)"
    SampleIndex ||--o{ MetricResult : "sample_id (level=sample)"
    CapabilityBlock ||--o{ MetricResult : "capability_block_id"

    RankingList ||--o{ RankingEntry : "ranking_list_id"
    Unit ||--o{ RankingEntry : "unit_id"
    BenchmarkingRun ||--o{ RankingEntry : "benchmarking_run_id"
    Model ||--o{ RankingEntry : "model_id"
```

### 1.3 核心层级速览

数据侧（数据集 → 样本）：

```text
DatasetManifest → DatasetLoadJob → Shard(real) → SampleIndex
                                         ↓
Track → CapabilityBlock → Shard → SampleIndex
```

执行侧（评测 → 结果）：

```text
BenchmarkingRun → Unit → Task → (遍历 CapabilityBlock 下的 Shard) → ForecastArtifact
                                                                   → MetricResult(sample/shard/task/unit)
```

`MetricResult` 是**单表多层级**：通过 `result_level` 字段区分 `sample` / `shard` / `task` / `unit`（MVP 实际写入这四级；定义上还预留 `run` / `ranking`）。

> **逻辑外键说明**：模型层未声明数据库级 FOREIGN KEY，关系仅以带 `index=True` 的 ID 字段表达，并由 service 层维护引用完整性。ER 图中的关系基数据此还原。

---

## 2. 数据加载域实体

### 2.1 DatasetManifest

描述一个真实数据源。源文件 `backend/app/models/dataset.py:11`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `dataset_manifest_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `name` | `str` | 必填 | 数据集展示名 |
| `domain` | `str` | 必填 | 数据领域 |
| `source_type` | `str` | `"managed_file"` | MVP 固定为托管上传文件 |
| `source_uri` | `str` | 必填 | 原始数据位置（`runtime/uploads/` 下） |
| `file_format` | `str` | `"csv"` | MVP 仅 csv |
| `time_column` | `str` | 必填 | 时间列列名 |
| `target_columns` | `list[str]` | `[]`，**JSON 列** | 目标列；MVP 实际只允许 1 个（见 §6.3） |
| `frequency` | `str \| None` | `None` | 时间频率；load 成功后由推断结果回填 |
| `timezone` | `str \| None` | `None` | 仅作元数据，不做时区转换 |
| `schema_version` | `str` | `"dataset_manifest.v1"` | manifest schema 版本 |
| `status` | `str` | `"ready_to_load"` | 见下方枚举 |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

**status 取值**（代码实际出现）：
- `"ready_to_load"`：默认初始值（`models/dataset.py:23`）。
- `"loaded"`：加载成功后置位（`services/dataset_load_service.py:149`）。

> 设计稿（spec §4.1）列出 `draft / ready_to_load / loaded / disabled` 四态，但当前代码只实际写入 `ready_to_load` 和 `loaded`，未见 `draft` / `disabled` 的赋值。

### 2.2 DatasetLoadJob

记录一次加载过程。源文件 `backend/app/models/dataset.py:28`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `load_job_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `dataset_manifest_id` | `str` | `index=True`（逻辑外键 → DatasetManifest） | 来源 manifest |
| `status` | `str` | `"created"` | 见下方枚举 |
| `current_step` | `str \| None` | `None` | 当前步骤 |
| `reader_type` | `str` | `"csv_dataset_reader"` | 读取器类型 |
| `reader_version` | `str` | `"1"` | 读取器版本 |
| `validation_summary` | `dict[str, Any]` | `{}`，**JSON 列** | 校验摘要（见下） |
| `split_config` | `dict[str, Any]` | `{}`，**JSON 列** | 切分配置（见下） |
| `seed` | `int` | `0` | 物化/stub 默认 seed |
| `error_code` | `str \| None` | `None` | 失败错误码 |
| `error_message` | `str \| None` | `None` | 失败说明 |
| `output_shard_id` | `str \| None` | `None` | 成功后产生的唯一 Shard |
| `started_at` | `datetime \| None` | `None` | |
| `finished_at` | `datetime \| None` | `None` | |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

**status 取值**（来自 `services/dataset_load_service.py`）：
- `"created"`：默认初始值（`models/dataset.py:31`）。
- `"loading"`：进入物化阶段（`dataset_load_service.py:117`）。
- `"succeeded"`：成功（`dataset_load_service.py:152`）。
- `"failed"`：捕获 `ApiError` 后失败（`dataset_load_service.py:92`）。

**current_step 取值**（来自同文件）：
- `"validating"`：创建 job 时（`dataset_load_service.py:82`）。
- `"materializing_samples"`：物化样本时（`dataset_load_service.py:118`）。
- `"succeeded"`：成功收尾时（`dataset_load_service.py:153`）。

> spec §4.2 的状态机额外提到 `validating`、`materializing_samples`，但它们在代码里落在 `current_step` 字段，而非 `status`。`status` 实际未写入独立的 `validating` / `materializing_samples` 值。

**`split_config` 结构**（service 读取的 key，`dataset_load_service.py:104-115`）：
- `context_length`（int，必填）
- `horizon`（int，必填）
- `stride`（int，可选；缺省取 `horizon`）

**`validation_summary` 结构**（成功时写入，`dataset_load_service.py:155-160`）：
```json
{
  "row_count": 20,
  "sample_count": 4,
  "frequency": "1h",
  "columns": ["time", "value"]
}
```

### 2.3 Shard

统一数据单元。源文件 `backend/app/models/dataset.py:47`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `shard_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `shard_type` | `str` | `"real"` | MVP 固定 real |
| `dataset_manifest_id` | `str` | `index=True`（逻辑外键 → DatasetManifest） | 数据来源 |
| `load_job_id` | `str \| None` | `None`，`index=True`（逻辑外键 → DatasetLoadJob） | 产生该 shard 的 job |
| `capability_block_id` | `str \| None` | `None`，`index=True`（逻辑外键 → CapabilityBlock） | 所属能力块；归属前为 None |
| `source_uri` | `str` | 必填 | 原始数据位置 |
| `storage_uri` | `str \| None` | `None` | 规范化产物位置（指向 `samples/{shard_id}.jsonl`） |
| `checksum` | `str \| None` | `None` | 校验值 |
| `time_range_start` | `str \| None` | `None` | 时间范围起（ISO 8601 字符串） |
| `time_range_end` | `str \| None` | `None` | 时间范围止（ISO 8601 字符串） |
| `row_count` | `int` | `0` | 行数 |
| `target_columns` | `list[str]` | `[]`，**JSON 列** | 目标列 |
| `target_dim` | `int` | `1` | 目标维度 |
| `frequency` | `str \| None` | `None` | 时间频率 |
| `context_length` | `int` | `0` | 样本历史长度 |
| `horizon` | `int` | `0` | 预测长度 |
| `stride` | `int` | `0` | 滑窗步长 |
| `sample_count` | `int` | `0` | 样本数 |
| `status` | `str` | `"created"` | 见下方枚举 |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

**shard_type 取值**：`"real"`（默认值；`init_db.py:37` 的唯一性断言也按 `shard_type=="real"` 过滤）。spec §2.2 预留 `synthetic`，当前代码未写入。

**status 取值**：
- `"created"`：默认初始值（`models/dataset.py:66`）。
- `"ready"`：成功加载后创建 shard 即置为 ready（`dataset_load_service.py:137`）；`init_db.py:38` 的唯一性断言也按 `status=="ready"` 过滤。

> spec §4.3 列出 `created / ready / failed / disabled`，代码实际只出现 `created` 和 `ready`。

### 2.4 SampleIndex

记录样本切片位置和物化产物引用。源文件 `backend/app/models/sample.py:11`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `sample_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `shard_id` | `str` | `index=True`（逻辑外键 → Shard） | 所属 shard |
| `sample_index` | `int` | 必填 | shard 内序号（从 0 起） |
| `context_start` | `int` | 必填 | history 窗口起（行偏移） |
| `context_end` | `int` | 必填 | history 窗口止（含） |
| `horizon_start` | `int` | 必填 | future 窗口起 |
| `horizon_end` | `int` | 必填 | future 窗口止（含） |
| `target_columns` | `list[str]` | `[]`，**JSON 列** | 目标列 |
| `context_length` | `int` | `0` | history 长度 |
| `horizon` | `int` | `0` | future 长度 |
| `storage_ref` | `dict[str, Any]` | `{}`，**JSON 列** | 读取引用，形如 `{"line": <行号>}` |
| `materialized` | `bool` | `True` | MVP 固定物化 |
| `materialized_sample_uri` | `str \| None` | `None` | 物化产物文件路径 |
| `checksum` | `str \| None` | `None` | 该 sample canonical JSON 的 sha256 |
| `sample_metadata` | `dict[str, Any]` | `{}`，**JSON 列** | 可选样本摘要 |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

写入逻辑见 `services/sample_store.py:38-55`：`storage_ref={"line": line_number}`，`materialized=True`，`materialized_sample_uri` 指向 `samples/{shard_id}.jsonl`，`checksum` 为该行 canonical JSON 的 sha256。

> 注意字段名为 `sample_metadata`（不是 `metadata`，后者与 SQLAlchemy 保留名冲突）。spec §4.4 写作「metadata」，以代码为准。

---

## 3. 评测组织域实体

### 3.1 CapabilityBlock

统一能力块。源文件 `backend/app/models/benchmark.py:11`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `capability_block_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `track_id` | `str \| None` | `None`，`index=True`（逻辑外键 → Track） | 所属 Track；挂到 Track 前为 None |
| `block_type` | `str` | `"real"` | MVP 固定 real |
| `capability_type` | `str` | `"real_data"` | 能力类型 |
| `name` | `str` | 必填 | 展示名 |
| `task_type` | `str` | `"univariate_forecast"` | 任务类型 |
| `target_dim` | `int` | `1` | 目标维度 |
| `shard_count` | `int` | `0` | shard 数 |
| `sample_count` | `int` | `0` | 汇总样本数 |
| `aggregation_policy` | `str` | `"mean_over_shards"` | shard 聚合策略 |
| `status` | `str` | `"ready"` | 见下方枚举 |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

`block_type`/`capability_type` 在 `services/track_service.py:32-33` 创建时显式赋为 `"real"` / `"real_data"`。
**status 取值**：仅 `"ready"`（默认值，`models/benchmark.py:22`），代码无其它写入。spec §4.5 预留 `draft / disabled`。

### 3.2 Track

评测赛道。源文件 `backend/app/models/benchmark.py:27`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `track_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `name` | `str` | 必填 | 赛道名 |
| `track_type` | `str` | `"real_dataset"` | MVP 固定 |
| `description` | `str \| None` | `None` | 描述 |
| `primary_metric_id` | `str` | `"mse"` | 默认榜单指标 |
| `default_ranking_policy` | `str` | `"latest_valid_result"` | 默认榜单策略 |
| `benchmark_version` | `str` | `"mvp"` | benchmark 版本 |
| `data_version` | `str` | `"v1"` | 数据版本 |
| `status` | `str` | `"ready"` | 见下方枚举 |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

**status 取值**：仅 `"ready"`（默认值）。spec §4.6 预留 `draft / disabled`。

### 3.3 Model

可评测模型注册项。源文件 `backend/app/models/model_registry.py:10`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `model_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `name` | `str` | 必填 | 展示名 |
| `model_family` | `str` | 必填 | 模型族（Timer / Chronos / toto / TimesFM 等） |
| `model_version` | `str` | 必填 | 版本 |
| `adapter_type` | `str` | `"timer_service"` | adapter 类型 |
| `endpoint_uri` | `str \| None` | `None` | 推理服务地址 |
| `supported_task_types` | `list[str]` | `["univariate_forecast"]`，**JSON 列** | 支持的任务类型 |
| `input_schema_version` | `str` | `"sample.v1"` | 输入协议版本 |
| `stub_seed` | `int` | `0` | stub 可复现 seed |
| `status` | `str` | `"available"` | 见下方枚举 |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

**status 取值**：仅 `"available"`（默认值，`models/model_registry.py:20`）。spec §4.7 预留 `registered / disabled`。

**adapter_type**：默认 `"timer_service"`。
- 注意实际选用哪种 adapter 由**全局配置**决定而非该字段：`get_model_adapter(settings)` 在 `settings.model_adapter == "stub"` 时返回 `StubTimerAdapter`，否则返回 `TimerRestAdapter`（`services/model_adapter.py:13-20`）。
- `endpoint_uri == "stub://fail"` 是约定的失败注入：会让对应 unit 直接以 `adapter_error` 失败（`run_executor.py:142-143`）。
- 种子模型由 `seed_mvp_models` 写入（`track_service.py:79-93`），共 5 个：Timer 3.5 / Timer 3.0 / Chronos 2 / toto / TimesFM 2.5，`endpoint_uri` 形如 `stub://timer-service/{slug}`。
- `remote_model_id(model)` 把本地模型映射为 REST 服务的 model_id，规则为 `{model_family}-{model_version}`（如 `Timer-3.5`），缺失时退回 `name` 或 `model_id`（`model_adapter.py:23-29`）。

---

## 4. 评测执行域实体

### 4.1 BenchmarkingRun

一次评测执行。源文件 `backend/app/models/benchmark.py:41`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `benchmarking_run_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `track_id` | `str` | `index=True`（逻辑外键 → Track） | 被评测 Track |
| `model_ids` | `list[str]` | `[]`，**JSON 列** | 参评模型 ID 列表 |
| `benchmark_version` | `str` | `"mvp"` | |
| `data_version` | `str` | `"v1"` | |
| `status` | `str` | `"created"` | 见下方枚举 |
| `execution_mode` | `str` | `"background_thread"` | 执行方式 |
| `cancel_requested` | `bool` | `False` | 是否已请求取消 |
| `cancel_requested_at` | `datetime \| None` | `None` | 取消请求时间 |
| `model_count` | `int` | `0` | 模型数 |
| `task_count` | `int` | `0` | task 数 |
| `sample_count` | `int` | `0` | 样本数 |
| `metric_set` | `list[str]` | `["mse", "mae"]`，**JSON 列** | 计算的指标集 |
| `report_id` | `str \| None` | `None`（逻辑外键 → Report） | 关联报告 |
| `ranking_list_id` | `str \| None` | `None`（逻辑外键 → RankingList） | 关联榜单 |
| `started_at` | `datetime \| None` | `None` | |
| `finished_at` | `datetime \| None` | `None` | |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

**status 取值**（来自 `services/run_executor.py`）：
- `"created"`：模型定义的默认值（`models/benchmark.py:47`）。
- `"queued"`：`create_benchmarking_run` 创建时实际写入（`run_executor.py:30`）。
- `"running"`：开始执行（`run_executor.py:101`）。
- `"cancel_requested"`：用户请求取消（`run_executor.py:70`）。
- `"cancelled"`：执行前发现已请求取消则进入 cancelled（`run_executor.py:93`）。
- `"succeeded"`：全部 unit 成功（`run_executor.py:117`）。
- `"partial_succeeded"`：部分 unit 成功、部分失败（`run_executor.py:115`）。
- `"failed"`：全部失败，或服务重启时把未完成 run 标记失败（`run_executor.py:83,119`）。

终态判定逻辑（`run_executor.py:112-119`）：统计 `unit.status == "succeeded"` 与 `== "failed"` 的数量——同时存在 → `partial_succeeded`；只有成功 → `succeeded`；否则 → `failed`。

### 4.2 Unit

某模型在某次 run 中的完整结果。源文件 `backend/app/models/benchmark.py:63`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `unit_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `benchmarking_run_id` | `str` | `index=True`（逻辑外键 → BenchmarkingRun） | 所属 run |
| `model_id` | `str` | `index=True`（逻辑外键 → Model） | 模型 |
| `status` | `str` | `"created"` | 见下方枚举 |
| `task_count` | `int` | `0` | task 数 |
| `sample_count` | `int` | `0` | 样本数 |
| `started_at` | `datetime \| None` | `None` | |
| `finished_at` | `datetime \| None` | `None` | |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

**status 取值**（来自 `run_executor.py`）：
- `"created"`：默认值（`models/benchmark.py:67`），创建 unit 时未显式改写。
- `"running"`：开始执行 unit（`run_executor.py:138`）。
- `"succeeded"`：所有 task metric 非空（`run_executor.py:156`）。
- `"partial_succeeded"`：部分 task 成功（`run_executor.py:156`）。
- `"failed"`：`_fail_unit` 整体失败（`run_executor.py:170`，如 `endpoint_uri=="stub://fail"`）。

> spec §4.9 还列出 `skipped` / `cancelled`，但当前 run_executor 未写入这两个值（进度统计在 `build_run_progress` 中把它们纳入「已完成」集合，但执行链路不会产生）。

### 4.3 Task

某模型在某 `CapabilityBlock` 上的结果。源文件 `backend/app/models/benchmark.py:76`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `task_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `benchmarking_run_id` | `str` | `index=True`（逻辑外键 → BenchmarkingRun） | 所属 run |
| `unit_id` | `str` | `index=True`（逻辑外键 → Unit） | 所属 unit |
| `model_id` | `str` | `index=True`（逻辑外键 → Model） | 模型 |
| `capability_block_id` | `str` | `index=True`（逻辑外键 → CapabilityBlock） | 能力块 |
| `status` | `str` | `"created"` | 见下方枚举 |
| `shard_count` | `int` | `0` | shard 数 |
| `sample_count` | `int` | `0` | 样本数 |
| `aggregation_policy` | `str` | `"mean_over_shards"` | shard 聚合策略 |
| `error_code` | `str \| None` | `None` | 失败错误码 |
| `error_message` | `str \| None` | `None` | 失败说明 |
| `started_at` | `datetime \| None` | `None` | |
| `finished_at` | `datetime \| None` | `None` | |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

**status 取值**（来自 `run_executor.py`）：
- `"created"`：默认值（`models/benchmark.py:82`）。
- `"running"`：开始执行 task（`run_executor.py:177`）。
- `"succeeded"`：所有 shard metric 非空（`run_executor.py:192`）。
- `"partial_succeeded"`：部分 shard 成功（`run_executor.py:192`）。
- `"failed"`：随 unit 失败被批量置位，并写入 `error_code`/`error_message`（`run_executor.py:165-167`）。

> spec §4.10 同样列出 `skipped` / `cancelled`，代码未写入。

### 4.4 ForecastArtifact

预测产物位置记录（不含 forecast 数组）。源文件 `backend/app/models/benchmark.py:94`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `forecast_artifact_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `benchmarking_run_id` | `str` | `index=True`（逻辑外键 → BenchmarkingRun） | 所属 run |
| `unit_id` | `str` | `index=True`（逻辑外键 → Unit） | 所属 unit |
| `task_id` | `str` | `index=True`（逻辑外键 → Task） | 所属 task |
| `model_id` | `str` | `index=True`（逻辑外键 → Model） | 模型 |
| `shard_id` | `str` | `index=True`（逻辑外键 → Shard） | 数据 shard |
| `storage_uri` | `str` | 必填 | forecast JSONL 路径 |
| `schema_version` | `str` | `"forecast.v1"` | 产物 schema 版本 |
| `sample_count` | `int` | `0` | 覆盖样本数 |
| `checksum` | `str \| None` | `None` | 整文件 sha256 |
| `created_at` | `datetime` | `utc_now` | |

写入见 `services/forecast_store.py:46-56`：`storage_uri` 为 `forecasts/{run_id}/{task_id}/{model_id}_{shard_id}.jsonl`，`sample_count=len(rows)`，`checksum` 为逐行累积的 sha256。`unit_id` 在写入返回后由 `run_executor.py:233` 补齐。

### 4.5 RunEvent

run/unit/task 过程事件与日志。源文件 `backend/app/models/benchmark.py:108`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `run_event_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `benchmarking_run_id` | `str` | `index=True`（逻辑外键 → BenchmarkingRun） | 所属 run |
| `unit_id` | `str \| None` | `None`，`index=True`（逻辑外键 → Unit） | 可选 |
| `task_id` | `str \| None` | `None`，`index=True`（逻辑外键 → Task） | 可选 |
| `level` | `str` | `"info"` | `info` / `warning` / `error` |
| `event_type` | `str` | `"status_changed"` | 事件类型 |
| `message` | `str` | 必填 | 文本说明 |
| `payload` | `dict[str, Any]` | `{}`，**JSON 列** | 结构化补充 |
| `created_at` | `datetime` | `utc_now` | |

**level 取值**（代码实际使用）：`"info"`（默认/run 排队、开始、完成）、`"warning"`（取消相关，`run_executor.py:71,95`）、`"error"`（重启中断，`run_executor.py:85`）。

**event_type 取值**（代码实际使用）：
- `"status_changed"`：默认值。
- `"cancel_requested"`：请求取消（`run_executor.py:71`）。
- `"cancelled"`：取消完成（`run_executor.py:95`）。
- `"interrupted_by_server_restart"`：服务重启中断（`run_executor.py:85`）。

---

## 5. 指标 / 报告 / 榜单域实体

### 5.1 MetricDefinition

指标注册表。源文件 `backend/app/models/metric.py:10`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `metric_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `name` | `str` | 必填 | 指标名（`mse` / `mae`） |
| `display_name` | `str` | 必填 | 展示名 |
| `direction` | `str` | `"lower_is_better"` | 优劣方向 |
| `supported_levels` | `list[str]` | `["sample","shard","task","unit","run","ranking"]`，**JSON 列** | 支持层级 |
| `status` | `str` | `"active"` | 默认 active（spec 预留 `disabled`） |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

> 注意 `MetricResult.metric_id` / `_metric()` 直接使用字符串 `"mse"` / `"mae"`（指标 name），并非引用 `MetricDefinition.metric_id`（UUID）。即指标在执行链路里以 name 作为业务键使用。

### 5.2 MetricResult

统一多层级指标结果。源文件 `backend/app/models/metric.py:24`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `metric_result_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `metric_id` | `str` | `index=True` | 指标键（实际存 `"mse"`/`"mae"`） |
| `result_level` | `str` | 必填 | `sample` / `shard` / `task` / `unit`（写入这四级） |
| `benchmarking_run_id` | `str` | `index=True`（逻辑外键 → BenchmarkingRun） | 所属 run |
| `unit_id` | `str \| None` | `None`，`index=True` | unit 级及以下携带 |
| `task_id` | `str \| None` | `None`，`index=True` | task 级及以下携带 |
| `sample_id` | `str \| None` | `None`，`index=True` | sample 级必填 |
| `shard_id` | `str \| None` | `None`，`index=True` | shard 级/sample 级携带 |
| `model_id` | `str` | `index=True`（逻辑外键 → Model） | 模型 |
| `capability_block_id` | `str \| None` | `None`，`index=True` | 可选 |
| `value` | `float` | 必填 | 指标值 |
| `aggregation` | `str` | `"raw"` | 见下方命名规则 |
| `metadata_json` | `dict` | `{}`，**JSON 列** | 附加信息 |
| `created_at` | `datetime` | `utc_now` | |

**各层级写入的 ID 组合**（来自 `run_executor._execute_*` 与 `_metric`）：

| result_level | unit_id | task_id | shard_id | sample_id | capability_block_id | aggregation |
| --- | --- | --- | --- | --- | --- | --- |
| `sample` | ✓ | ✓ | ✓ | ✓ | ✓ | `raw` |
| `shard` | ✓ | ✓ | ✓ | — | ✓ | `mean_over_shards` |
| `task` | ✓ | ✓ | — | — | ✓ | `mean_over_tasks` |
| `unit` | ✓ | — | — | — | — | `mean_over_units` |

**aggregation 命名规则**（`run_executor._metric`，`run_executor.py:268`）：

```python
aggregation="raw" if level == "sample" else f"mean_over_{level}s"
```

即 `sample → "raw"`，其余层级为 `mean_over_{level}s`：`shard → "mean_over_shards"`、`task → "mean_over_tasks"`、`unit → "mean_over_units"`。

> 这是一个**值得注意的命名细节**：聚合标签描述的是「在该层级内对下层求平均后落到该层」，但字面用了该层级自身的复数（如 task 级写 `mean_over_tasks`），与直觉上的「mean_over_shards 才得到 task 值」略有出入——以代码为准。

**指标计算与聚合**（`services/metric_service.py`）：
- sample 级：把 `target_future` 与 `forecast` 各自 flatten 后逐元素求误差，`mse = mean(err²)`、`mae = mean(|err|)`（`metric_service.py:8-17`）。
- 上层聚合：`aggregate_metric` 对下层成功项取算术平均，并返回 `success_count` / `failure_count`；若全部失败返回 `None`（`metric_service.py:20-34`）。

### 5.3 Report

基础评测报告。源文件 `backend/app/models/report.py:11`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `report_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `report_type` | `str` | `"run_summary"` | 报告类型 |
| `benchmarking_run_id` | `str` | `index=True`（逻辑外键 → BenchmarkingRun） | 所属 run |
| `track_id` | `str` | `index=True`（逻辑外键 → Track） | 所属 track |
| `status` | `str` | `"created"` | 见下方枚举 |
| `storage_uri` | `str \| None` | `None` | 报告 JSON 路径 |
| `summary` | `dict[str, Any]` | `{}`，**JSON 列** | 基础摘要 |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

**status 取值**：
- `"created"`：默认值（`models/report.py:15`）。
- `"ready"`：报告生成完成（`services/report_service.py:35`）。

**summary 结构**（`report_service.py:37`）：`{"status": <run.status>, "model_count": <unit 数>, "task_count": <task 数>}`。`storage_uri` 为 `reports/{run_id}.json`。

### 5.4 RankingList

每条 Track 一个的榜单定义。源文件 `backend/app/models/ranking.py:10`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `ranking_list_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `track_id` | `str` | `index=True`（逻辑外键 → Track） | 所属 track |
| `default_metric_id` | `str` | 必填 | 默认排序指标（来自 `Track.primary_metric_id`） |
| `default_policy` | `str` | `"latest_valid_result"` | 默认策略 |
| `supported_policies` | `list[str]` | `["latest_valid_result","best_result"]`，**JSON 列** | 支持策略 |
| `status` | `str` | `"active"` | 默认 active（spec 预留 `disabled`） |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

### 5.5 RankingEntry

榜单中的模型成绩行。源文件 `backend/app/models/ranking.py:21`。

| 字段 | 类型 | 默认值/约束 | 说明 |
| --- | --- | --- | --- |
| `ranking_entry_id` | `str` | **主键**，`default_factory=new_id` | UUID4 |
| `ranking_list_id` | `str` | `index=True`（逻辑外键 → RankingList） | 榜单 |
| `track_id` | `str` | `index=True`（逻辑外键 → Track） | 赛道 |
| `metric_id` | `str` | `index=True` | 当前视图指标（`mse`/`mae`） |
| `policy` | `str` | `"latest_valid_result"` | 视图策略 |
| `model_id` | `str` | `index=True`（逻辑外键 → Model） | 模型 |
| `benchmarking_run_id` | `str` | `index=True`（逻辑外键 → BenchmarkingRun） | 采用的 run |
| `unit_id` | `str` | `index=True`（逻辑外键 → Unit） | 对应 unit |
| `metric_value` | `float` | 必填 | 排序指标值 |
| `rank` | `int` | 必填 | 排名（从 1 起，按 value 升序） |
| `status` | `str` | `"active"` | 默认 active（spec 预留 `stale`） |
| `created_at` | `datetime` | `utc_now` | |
| `updated_at` | `datetime` | `utc_now` | |

**policy 取值**：`"latest_valid_result"`、`"best_result"`（`refresh_ranking` 对两种 policy 各刷新一套 entry，`ranking_service.py:11`）。

---

## 6. 落盘产物的数据结构（非 DB）

这些结构不进入 SQLite，而是以文件形式存于 `runtime/` 目录，DB 实体只保存其 `storage_uri` / `storage_ref`。

### 6.1 物化样本（sample.v1）

每个 shard 对应一个 JSONL 文件 `runtime/samples/{shard_id}.jsonl`，每行一个 sample。结构来自 `services/sample_store.py:75-89`（`_record_for_sample`）。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | `str` | 固定 `"sample.v1"` |
| `sample_id` | `str` | 对应 `SampleIndex.sample_id` |
| `shard_id` | `str` | 所属 shard |
| `sample_index` | `int` | shard 内序号 |
| `target_column_names` | `list[str]` | 目标列名 |
| `history_timestamps` | `list[str]` | history 窗口 ISO 8601 时间戳，长度 = context_length |
| `future_timestamps` | `list[str]` | future 窗口 ISO 8601 时间戳，长度 = horizon |
| `target_history` | `list[list[float]]` | shape `[context_length, target_dim]`，二维数组（单变量也是二维） |
| `target_future` | `list[list[float]]` | shape `[horizon, target_dim]`，作为 ground truth |
| `history_cov` | `list` | 历史协变量；MVP 为空数组 `[]` |
| `future_cov` | `list` | 未来协变量；MVP 为空数组 `[]` |
| `source_row_start` | `int` | 校验后原始数据行起（左闭） |
| `source_row_end` | `int` | 校验后原始数据行止（含） |

序列化使用 canonical JSON（`ensure_ascii=False, sort_keys=True, separators=(",", ":")`），`SampleIndex.checksum` 即该行内容的 sha256（`sample_store.py:11-16,53`）。

示例片段：
```json
{"future_cov":[],"future_timestamps":["2020-01-01T06:00:00","2020-01-01T07:00:00","2020-01-01T08:00:00"],"history_cov":[],"history_timestamps":["2020-01-01T00:00:00","2020-01-01T01:00:00","2020-01-01T02:00:00","2020-01-01T03:00:00","2020-01-01T04:00:00","2020-01-01T05:00:00"],"sample_id":"...","sample_index":0,"schema_version":"sample.v1","shard_id":"...","source_row_end":8,"source_row_start":0,"target_column_names":["value"],"target_future":[[6.0],[7.0],[8.0]],"target_history":[[0.0],[1.0],[2.0],[3.0],[4.0],[5.0]]}
```

### 6.2 预测产物（forecast.v1）

每个 (run, task, model, shard) 组合一个 JSONL 文件 `runtime/forecasts/{run_id}/{task_id}/{model_id}_{shard_id}.jsonl`，每行一个 sample 的 forecast。结构来自 `services/forecast_store.py:29-42`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | `str` | 固定 `"forecast.v1"` |
| `benchmarking_run_id` | `str` | run |
| `task_id` | `str` | task |
| `model_id` | `str` | 模型 |
| `shard_id` | `str` | shard |
| `sample_id` | `str` | 对应样本 |
| `status` | `str` | sample forecast 状态，默认 `"succeeded"`（来自 `row.get("status","succeeded")`） |
| `forecast` | `list[list[float]] \| None` | 预测数组，shape `[horizon, target_dim]`；失败行为 `null` |
| `future_timestamps` | `list[str]` | 与对应 sample 的 `future_timestamps` 一致；默认 `[]` |
| `metrics` | `dict` | 该 sample 的 sample-level 指标（如 `{"mse":..,"mae":..}`）；默认 `{}` |
| `error_code` | `str \| None` | 失败错误码（成功行为 `null`） |
| `error_message` | `str \| None` | 失败说明（成功行为 `null`） |

> 说明：当前 `run_executor._execute_shard` 写入的 row 均为成功行（含 `forecast` / `future_timestamps` / `metrics`），未携带 `status`/`error_*`，故落盘时 `status` 取默认 `"succeeded"`、`error_*` 为 `null`。失败/跳过行字段（`status="failed"/"skipped"` + `error_code`/`error_message`）是 schema 预留位，由 `ForecastStore` 的 `row.get(...)` 兜底支持。
> forecast.v1 **不保存 ground truth**——读取 sample.v1 的 `target_future` 才能算 metric。

示例片段（成功行）：
```json
{"benchmarking_run_id":"...","error_code":null,"error_message":null,"forecast":[[6.01],[6.99],[8.02]],"future_timestamps":["2020-01-01T06:00:00","2020-01-01T07:00:00","2020-01-01T08:00:00"],"metrics":{"mae":0.013,"mse":0.0002},"model_id":"...","sample_id":"...","schema_version":"forecast.v1","shard_id":"...","status":"succeeded","task_id":"..."}
```

### 6.3 报告 JSON

每个 run 一个 `runtime/reports/{run_id}.json`，由 `services/report_service.py:22-31` 生成（`json.dumps(payload, indent=2, sort_keys=True)`）。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `benchmarking_run_id` | `str` | run |
| `track_id` | `str` | track |
| `status` | `str` | run 终态 |
| `model_metrics` | `list[dict]` | 每个 unit 一项（见下） |
| `task_summaries` | `list[dict]` | 每个 task 一项（见下） |
| `sample_forecast_links` | `list[dict]` | sample → forecast artifact 链接 |
| `cancellation_reason` | `str \| None` | run 为 `cancelled` 时为 `"cancel_requested"`，否则 `null` |

`model_metrics[*]`（`_unit_metrics`，`report_service.py:50-62`）：`unit_id`、`model_id`、`model_name`、`status`、`metrics`（仅 `result_level=="unit"` 且匹配 unit 的指标，形如 `{"mse":..,"mae":..}`）。

`task_summaries[*]`（`_task_summary`，`report_service.py:65-79`）：`task_id`、`unit_id`、`model_id`、`capability_block_id`、`status`、`error_code`、`error_message`、`metrics`（仅 `result_level=="task"` 且匹配 task）。

`sample_forecast_links[*]`（`_sample_links`，`report_service.py:82-89`）：逐个读取 ForecastArtifact 文件，每行产出 `{"sample_id":..,"run_id":..,"forecast_artifact_id":..}`。

> 报告 DB 实体（`Report`）的 `summary` 字段与此 JSON 不同：`summary` 只存 `{status, model_count, task_count}`（§5.3），完整内容落在 `storage_uri` 指向的 JSON 文件里。

---

## 7. 传输层 DTO（schemas）

`backend/app/schemas/*.py` 是 API 读/写模型（Pydantic `BaseModel`），与持久化实体分离。当前都很薄：

| DTO | 文件 | 字段 |
| --- | --- | --- |
| `RealDatasetTrackCreateDTO` | `schemas/benchmark.py:4` | `name: str`、`shard_ids: list[str]`、`primary_metric_id: str = "mse"` |
| `DatasetLoadJobCreateDTO` | `schemas/dataset.py:4` | `dataset_manifest_id: str`、`split_config: dict`、`seed: int = 0` |
| `ModelDTO` | `schemas/model_registry.py:4` | `model_id: str`、`name: str`、`adapter_type: str` |
| `RankingRowDTO` | `schemas/ranking.py:4` | `model_id: str`、`metric_value: float`、`rank: int` |
| `ReportDTO` | `schemas/report.py:4` | `report_id: str`、`benchmarking_run_id: str`、`status: str` |
| `SamplePreviewDTO` | `schemas/sample.py:4` | `sample_id: str`、`target_history: list[list[float]]`、`target_future: list[list[float]]` |

> spec §3.2 / §7 还提到 `SampleForecastDTO`、`RunProgressDTO` 等读模型，但当前它们没有独立的 schema 类——而是由 service 直接构造 `dict` 返回（`build_sample_forecast`、`build_run_progress`）。

---

## 8. 关键不变量与生命周期

### 8.1 DatasetManifest 只能成功 load 一次

`assert_manifest_can_succeed_load`（`backend/app/db/init_db.py:18-30`）：若已存在 `status=="succeeded"` 的 `DatasetLoadJob`，抛 `dataset_manifest_already_loaded`。创建 load job 时调用（`dataset_load_service.py:74`）。

### 8.2 real shard 唯一性

`assert_manifest_can_create_successful_real_shard`（`init_db.py:33-46`）：若该 manifest 已有 `shard_type=="real"` 且 `status=="ready"` 的 Shard，抛 `dataset_manifest_already_has_real_shard`。同样在创建 load job 时调用（`dataset_load_service.py:75`）。二者共同保证：一个 manifest → 至多一个成功 load job → 至多一个 real shard（spec §1.2 / §5）。

### 8.3 加载失败时清理中间产物

`DatasetLoadService.create_load_job` 捕获 `ApiError` 后调用 `_cleanup_job_artifacts` 删除已生成的 `samples/{output_shard_id}.jsonl`，再把 job 置 `failed` 并记录 `error_code`/`error_message`（`dataset_load_service.py:90-100,170-174`）。成功加载会回填 `manifest.status="loaded"`、`manifest.frequency`、`job.output_shard_id`、`shard.storage_uri` 等。

### 8.4 样本切分（滑动窗口）

`build_windows`（`dataset_load_service.py:26-54`）：`stride` 缺省取 `horizon`；要求 `context_length/horizon/stride` 均为正；`context_length + horizon` 超过行数抛 `split_length_exceeds_rows`；窗口为空抛 `sample_count_empty`。窗口左闭右含，`source_row_start..source_row_end` 即原始（校验后）数据行范围。

### 8.5 capability block 与 shard 归属

`create_real_capability_block`（`track_service.py:19-47`）：要求至少一个 shard；shard 不存在抛 `shard_not_found`；已归属其它 block 的 shard 抛 `shard_already_assigned`（保证 shard 只属于一个 block）。block 的 `shard_count`/`sample_count`/`target_dim` 由所含 shard 汇总。

### 8.6 track 与 ranking 同生

`create_track_with_blocks`（`track_service.py:50-76`）：把指定 capability block 挂到新 track（`block.track_id = track.track_id`），并为该 track 创建唯一 `RankingList`（`default_metric_id = primary_metric_id`）。即 Track ↔ RankingList 一对一（spec §1.1）。

### 8.7 run 执行生命周期

创建（`create_benchmarking_run`，`run_executor.py:19-63`）：要求非空 `model_ids` 且 track 有 capability block；按 `模型数 × block 数` 预生成 Unit 与 Task，`status="queued"`。

执行（`execute_run`，`run_executor.py:90-133`）：
1. 若已 `cancel_requested` → 直接 `cancelled` 并返回。
2. 否则 `running`，逐 unit → 逐 task → 逐 shard → 逐 sample 执行；每 sample 调 adapter 产 forecast、算 sample 指标、写 forecast 行；shard/task/unit 逐层聚合写指标。
3. 终态判定（§4.1），写 RunEvent，调 `generate_run_report` 生成报告，回填 `run.report_id`。
4. 非 cancelled 时对 `mse`/`mae` 各刷新一次榜单。

取消（`cancel_run`，`run_executor.py:66-75`）：协作式取消——置 `cancel_requested=True`、`cancel_requested_at`、`status="cancel_requested"`，写 warning 事件。

崩溃恢复（`recover_interrupted_runs`，`run_executor.py:78-87`）：服务启动时把仍处于 `queued`/`running`/`cancel_requested` 的 run 标记 `failed` 并写 `interrupted_by_server_restart` 事件。

### 8.8 榜单刷新规则

`refresh_ranking`（`ranking_service.py:8-37`）：对 `latest_valid_result` 与 `best_result` 两种 policy 各重建一套 entry。
- **有效 unit 过滤**（`_valid_unit_metric_rows`，`ranking_service.py:53-69`）：只取 `result_level=="unit"` 的指标，且 run 属于该 track、`run.status ∈ {succeeded, partial_succeeded}`、`unit.status == "succeeded"`。即 `partial_succeeded`/`failed` 的 unit **不进榜**（spec §8.4-48）。
- `latest_valid_result`：每模型取 `run.created_at` 最新的有效 unit（`_select_latest`）。
- `best_result`：每模型取 metric 值最小的有效 unit（`_select_best`，因 `direction=lower_is_better`）。
- 排名按 value 升序，`rank` 从 1 起。

### 8.9 指标聚合命名

见 §5.2。核心一句：`aggregation = "raw" if level=="sample" else f"mean_over_{level}s"`（`run_executor.py:268`）。

### 8.10 stub forecast 可复现

`StubTimerAdapter.forecast`（`stub_timer_adapter.py:9-27`）：以 `target_history` 最后一个值做 naive 预测，叠加由 `sha256(f"{model_id}:{sample_id}:{seed}")` 决定的确定性噪声与 `model_bias`。相同 `model_id + sample_id + seed` 必得相同 forecast（spec §1.7）。

---

## 9. 设计稿与代码的差异汇总

逐条精读后发现的「设计稿 ↔ 代码」不一致之处（均**以代码为准**）：

1. **DatasetLoadJob 的 `validating` / `materializing_samples` 不是 `status` 值**，而是 `current_step` 字段的取值；`status` 实际只取 `created/loading/succeeded/failed`（spec §4.2 把它们画进 status 状态机）。
2. **多数实体的 `status` 枚举代码只实现了「快乐路径」子集**：`DatasetManifest`（仅 `ready_to_load/loaded`）、`Shard`（仅 `created/ready`）、`CapabilityBlock`/`Track`（仅 `ready`）、`Model`（仅 `available`）。spec 中列出的 `draft/disabled/registered/failed` 等当前无代码写入。
3. **Unit / Task 的 `skipped` / `cancelled` 状态未实现**：run_executor 不会写入这两个值，尽管 `build_run_progress` 的进度统计把它们算作「已完成」（spec §4.9/§4.10 列出）。
4. **`SampleIndex` 字段名是 `sample_metadata`，不是 spec §4.4 写的 `metadata`**（规避 SQLAlchemy 保留名）。
5. **指标在执行链路里用 name（`"mse"`/`"mae"`）作业务键**，`MetricResult.metric_id`/`RankingEntry.metric_id` 存的是指标 name 而非 `MetricDefinition.metric_id`（UUID）。
6. **聚合标签命名**：task 级写 `mean_over_tasks`、unit 级写 `mean_over_units`（即「本层级复数」），而非直觉上的「下层复数」；spec §4.13 仅举例 `mean_over_samples/mean_over_shards`，未覆盖这两个值。
7. **`BenchmarkingRun.model_ids`、`Shard` 大量物理切分字段（context_length/horizon/stride 等）** 在 spec 字段表中未逐一列出，代码中确有这些列。
8. **`SampleForecastDTO` / `RunProgressDTO` 没有独立 schema 类**：service 直接返回 dict（`build_sample_forecast` / `build_run_progress`）。
9. **adapter 选择由全局配置 `settings.model_adapter` 决定**（`stub` vs REST），`Model.adapter_type` 字段当前不参与运行期分支判断；spec §4.7 偏向把 adapter 绑在 Model 上。
