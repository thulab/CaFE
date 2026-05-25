# TSBenchmark 整体设计实现计划（真实数据通路 + TsFile 重构）

> **给实现者：** 本计划是 [`2026-05-25-tsbenchmark-overall-design-discussion.md`](../specs/2026-05-25-tsbenchmark-overall-design-discussion.md) 的执行细化。所有生产行为遵循 TDD：先写失败测试（RED）→ 验证 RED → 最小实现（GREEN）→ 验证 GREEN → 在测试保护下重构。步骤用 `- [ ]` 复选框跟踪。Git 操作由用户负责，本计划不分配版本控制动作。

**目标：** 在现有 MVP（CSV→JSONL→桩推理→MSE/MAE→榜单，已端到端跑通）之上，把数据通路重构为「**全列摄入 → per-dataset TsFile 单一真值源 → 指针化切片器 → 输入/答案分离 → MASE 主排名**」，并先用真实 CSV 跑通现状路径拿到可信对照基线。

**性质：** 本文档只做设计与任务拆分，不执行开发。

---

## 本轮锁定决策（2026-05-25，经确认）

| 议题 | 决策 | 影响范围 |
|---|---|---|
| 本轮交付 | **只产出本 plan/task 文档**，评审通过后再开工 | 本文档即交付物 |
| forecast 输出落盘 | **维持 JSONL**（`forecast.v1` 不变） | Layer 4 只重构输入/样本侧；`forecast_store.py` / `sample_forecast_service.py` 读写链不动 |
| 指标主排名 | **切 MASE 主排名**，mse/mae 降为诊断 | Layer 6：新增 MASE + `Track.primary_metric_id` 默认切 `mase` + 榜单/前端默认指标连锁调整 |

**继承自 spec 的既定路线：** 先基线后重构；TsFile 写入已 de-risk（spec §2，PASS）；推理仅本地桩（真模型接入待定）。

---

## 实现状态（2026-05-25 已完成 Track B 全部 7 层）

> 本计划评审通过后已按 TDD + 多智能体并发实现完毕。**后端 `uv run pytest` 136 passed；前端 `npm test` 18 passed。**

| 层 | 状态 | 关键落点 |
|---|---|---|
| B1 依赖&DB | ✅ | `pyproject.toml` 加 `tsfile==2.3.0`/numpy/pandas/pyarrow；**`requires-python` 收紧到 `>=3.14`**（tsfile 2.3.0 在 <3.14 钉 pandas<2.3，与 pandas3 冲突）；`test_tsfile_roundtrip.py` 固化 spike；删旧 dev 库 |
| B2 全列摄入 | ✅ | `CsvDatasetReader` 全列有限-float 校验（`csv_value_*`）；`DatasetReadResult.value_columns/values`；manifest `target_columns`→`value_columns` |
| B3 选择期 | ✅ | load-job `split_config.target_columns`(恰好1,校验⊆value_columns) + `max_samples` 均匀采样（`subsample_windows`） |
| B4 TsFile+切片器 | ✅ | `services/tsfile_store.py`（`TsFileStore.write`/`TsFileSlicer.slice`）；load 落 `tsfiles/{shard}.tsfile`；`SampleStore` 指针化（`storage_ref` 行号区间，读取现切） |
| B5 输入/答案分离 | ✅ | `services/model_input.py`（无 `target_future`、带 `horizon`）；两适配器 `len(target_future)`→`horizon` |
| B6 MASE 主排名 | ✅ | `compute_sample_metrics` 加 mase（context naive 尺度，scale=0 不产出）；各层聚合 `METRIC_NAMES=["mase","mse","mae"]`；`Track.primary_metric_id`/wizard/ranking 路由默认切 mase |
| B7 测试+前端+文档 | ✅ | 17 后端测试同步 + 新增 `test_real_csv_tsfile_flow.py`（真实 CSV，验证 extra 列入 TsFile + mase 默认榜）；前端 contract 同步；`docs/developer/{data-model,key-flows}.md` 更新 |

**并发执行记录（满足「允许多智能体并发 + TDD」）：** B4-core/B5/B6-core 三个隔离叶子模块由并发子代理 TDD 产出（各只跑自身测试、写集不相交），主流程串行做 B2→B3 与全部共享文件（`run_executor.py` 等）集成；前端 contract 同步由第 4 个子代理完成。

**Track A 现场基线（A0）已补做（2026-05-25）：** `scripts/baseline-run.sh` + `scripts/baseline_run.py` 起一个隔离的 live uvicorn 后端（进程内确定性桩），用真实 `test/flow_template.csv` 走完整 API 链，落 `docs/superpowers/baselines/2026-05-25-baseline-record.md`。实跑结果：run `succeeded`、`extra` 列入 TsFile、默认榜单指标 `mase`；记录含 mase/mse/mae 三榜 + 首样本真值/各模型预测对照。

**诚实边界 / 偏差：**
- 除 live 基线外，回归仍由测试套件兜底：原有 e2e（JSONL→TsFile 在 API 契约层等价，全绿）+ `test_real_csv_tsfile_flow.py`（真实 FastAPI app + 真实 TsFile I/O + 桩推理 + 指标/榜单）。
- **桩 forecast 绝对值不跨重载可比**：`SampleIndex.sample_id` 为随机 UUID，桩以 `sha256(model_id:sample_id:seed)` 播种——这是**重构前既有特性**，非本次引入；逐位对照只在单次 load 内成立。
- **`requires-python` 由 `>=3.12` 提到 `>=3.14`**：tsfile 2.3.0 的轮子约束所致，已记入风险；他人环境需 Python 3.14。
- 推理仍仅桩；协变量（`history_cov`/`future_cov` 仍空）、多目标、forecast 输出 TsFile、数据指纹复现 —— 按 spec §6 延后。

---

## 源文档与现状事实（已读代码核对）

- 设计讨论：`docs/superpowers/specs/2026-05-25-tsbenchmark-overall-design-discussion.md`
- 实体结构：`docs/superpowers/specs/2026-05-16-tsbenchmark-mvp-entity-structure-design.md`
- 开发者手册（代码为唯一事实源）：`docs/developer/{key-flows,data-model}.md`
- TsFile 参考：`docs/reference/tsfile-dataframe-manual.md`

**环境事实（本轮实测）：**
- venv 为 **Python 3.14.4**，已装 `numpy 2.4.6` / `pandas 3.0.3`，但 **`tsfile` 当前不可 import**（spike 时的 `uv pip install` 未持久化）→ Layer 1 必须真正把 tsfile 写入依赖。
- `backend/pyproject.toml` 当前 `requires-python = ">=3.12"`，依赖仅 fastapi/httpx/pydantic-settings/python-multipart/sqlmodel/uvicorn，**无数值栈**。
- 真实样本：`test/flow_template.csv` = `time,target,extra` 三列、30 行（`extra` 现状被丢弃，正好演示全列摄入）。

**两处「答案泄露面」已定位（Layer 5 改这两行）：**
- `backend/app/services/timer_rest_adapter.py:36` → `horizon=len(sample["target_future"])`
- `backend/app/services/stub_timer_adapter.py:15` → `horizon = len(sample["target_future"])`
两处只取 `target_future` 的**长度**当 horizon，不用其值；换成 `sample["horizon"]` 即可，不破坏推理。

---

## 执行约束

- 每个后端测试必须用临时 `TSBENCHMARK_RUNTIME_DIR` + 临时 SQLite，绝不碰真实 `runtime/`。已有 `tests/conftest.py` 提供该隔离，沿用。
- 服务边界不变：routes 校验+委派、services 承载行为、models 仅持久化。
- 落盘产物用临时文件 + 原子 rename。
- 重构期间**每一层结束都跑该层相关测试 + 全量 `uv run pytest` 保持绿**（已有 ~50 个测试文件是回归网）。
- Track A 不写新生产代码，只跑脚本与核对。

---

## 整体改动总览（现状 → 目标）

```
现状：CSV --CsvDatasetReader(强制单目标,只读target列)--> DatasetReadResult
        --build_windows--> SampleStore.write_samples --每窗物化--> samples/{shard}.jsonl(含target_future值)
        --run_executor: read_by_ref(JSONL) --> adapter.forecast(sample含target_future) --> MSE/MAE --> 榜单(mse主)

目标：CSV --CsvDatasetReader(全列数值校验)--> DatasetReadResult(全列矩阵)
        --DatasetLoadService: 选target(恰好1)+max_samples均匀采样--> build_windows(指针,不物化值)
        --dataframe_to_tsfile--> tsfiles/{shard}.tsfile(表模型 tsbench.<id>.<列>, 单一真值源)
        --run_executor: TsFileSlicer 按行号现切窗口 --> ModelInput(无target_future,带horizon)
        --> adapter.forecast(ModelInput) --> 服务端用target_future算 MSE/MAE/MASE --> 榜单(mase主)
                                                                              --> forecast.v1 JSONL(不变)
```

**实体字段连带改动（§6 #2 整体复审结果）：**

| 实体 | 字段改动 | 来源 |
|---|---|---|
| `DatasetManifest` (`models/dataset.py:11`) | `target_columns` → 语义改为 `value_columns`（全部数值列，摄入用）；目标列选择移到 load-job | Layer 2 |
| `DatasetLoadJob.split_config` | 新增 `target_columns`（校验恰好 1）、`max_samples`（可选） | Layer 3 |
| `Shard` (`models/dataset.py:47`) | 新增 `value_columns`（全列）、`tsfile_uri`（per-dataset TsFile 路径）；`target_columns` 保留=被选目标；`storage_uri` 指向 `.tsfile` | Layer 4 |
| `SampleIndex` (`models/sample.py:11`) | `storage_ref` 由 `{"line": n}` 改为指针切片引用；样本不再逐窗物化值 | Layer 4 |
| `MetricDefinition`/`MetricResult` | 业务键集合加入 `"mase"` | Layer 6 |
| `Track` (`models/benchmark.py:27`) | `primary_metric_id` 默认 `"mse"` → `"mase"` | Layer 6 |

**明确「本轮不做 / 延后」（spec §6 复审）：**
- §6 #1 forecast 输出 TsFile → 维持 JSONL（已决策）。
- §6 #3 协变量正式实现 → 延后；`value_columns` 全列摄入为其打地基，但 `history_cov`/`future_cov` 仍空。
- §6 #4 多目标 / 多变量预测 → 延后；MVP 选择期仍恰好 1 个 target。
- §6 #6 真实推理接入 → 延后；仅桩。
- §6 #7 数据指纹复现 → 延后；维持 `checksum=hash(配置)` 路线。

---

# Track A · 基线跑通（现状 JSONL 路径 + 桩推理）— 先做

> 管线已存在，本 track 几乎不写生产代码，重点是**用真实 CSV 实跑并核对**，产出重构前后的对照基线。诚实边界：推理是桩；`extra` 列在现状 reader 下被忽略（单目标）。

### Task A0：脚本化基线 runner

**Files:**
- Create: `scripts/baseline-run.sh`（或 `backend/scripts/baseline_flow.py`）— 一条可重跑的端到端调用脚本
- Create: `docs/superpowers/baselines/2026-05-25-baseline-record.md`（基线结果留档）

- [ ] **A0.1 起后端 + 桩**
  - 动作：`./scripts/stub-service.sh start` + `./scripts/start-system.sh`（或仅后端 + `TSBENCHMARK_MODEL_ADAPTER=stub`）。
  - 验收：`./scripts/status-system.sh` 全绿；`./scripts/stub-service.sh status` 在线。

- [ ] **A0.2 真实 CSV 走完整 API 链**
  - 动作：脚本依次调用 `POST /dataset-manifests/upload`（传 `test/flow_template.csv`）→ `POST /dataset-manifests`（`time_column="time"`, `target_columns=["target"]`）→ `POST /dataset-load-jobs`（`context_length`/`horizon`/`stride`，建议 `context_length=12,horizon=6,stride=6`）→ `POST /wizard/real-dataset-track` → `POST /benchmarking-runs` → 轮询 `GET /benchmarking-runs/{id}/progress`。
  - 验收：run 终态 `succeeded`；load-job `succeeded` 且 `sample_count > 0`。

- [ ] **A0.3 核对榜单 + 样本预测视图并留档**
  - 动作：`GET /tracks/{id}/ranking`、`GET /samples/{id}/forecast?run_id=...`，把 5 个桩模型的 mse/mae 排名、某 sample 的真值+各模型预测记入基线留档。
  - 验收：ranking 有完整排名；sample forecast 取到真值 + 预测；**数值因桩确定性可复现**（同输入同输出）。

→ **产出**：`baseline-record.md` 固化「真实 CSV + 桩」的榜单与样本数值，作为 Track B 重构后 `B7` 复跑的逐位对照锚点。

---

# Track B · spec 重构（按依赖分层，TDD）

> Layer 2/3/5/6 不依赖 TsFile，可独立推进；Layer 4 才用上 `dataframe_to_tsfile`。建议顺序 1→2→3→4→5→6→7；但 5、6 可与 4 并行（不同写集）。

## Task B1：依赖入库 + DB 重建（Layer 1）

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/uv.lock`（由 `uv` 生成）
- Create: `backend/tests/unit/test_tsfile_roundtrip.py`（把 spike 固化成回归测试）

- [ ] **B1.1 RED：TsFile 往返回归测试**
  - 写测试：用临时目录把一段 `time,value` DataFrame 经 `tsfile.dataframe_to_tsfile(df, path, table_name="tsbench", time_column="time", tag_column=["dataset_id"])` 落盘，再 `TsFileDataFrame(path)["tsbench.<id>.value"][a:b]` 读回，断言切片值与源逐位一致、ms epoch 往返正确。
  - Run: `cd backend && uv run pytest tests/unit/test_tsfile_roundtrip.py -v`
  - Expected: FAIL（`import tsfile` 失败 / 依赖未入库）。

- [ ] **B1.2 GREEN：把数值栈写入依赖**
  - 动作：`cd backend && uv add tsfile numpy pandas pyarrow`（spike 验证组合：`tsfile==2.3.0` + numpy 2.4 + pandas 3.0 + pyarrow）。确认 `requires-python` 与 tsfile 轮子兼容（venv 已是 3.14.4；若 tsfile 仅发布 3.14 轮子，按需把 `requires-python` 收紧并在本文档记风险）。
  - Run: `cd backend && uv sync && uv run pytest tests/unit/test_tsfile_roundtrip.py -v`
  - Expected: PASS。

- [ ] **B1.3 开发库重建**
  - 动作：删除本地 `runtime/tsbenchmark.db`（dev 库；测试用临时库不受影响），下次启动由 `init_db` 重建含新字段的表。**仅删 dev 库，不提交 runtime 产物。**
  - 验收：`./scripts/start-system.sh` 后 `init_db` 成功建表，`status-system.sh` 全绿。

> 风险：tsfile 与 numpy 2.4/pandas 3.0 的二进制兼容性；若 `uv sync` 解析失败，回退到 spike 验证过的精确版本钉死（`tsfile==2.3.0`）。

## Task B2：全列数值摄入（Layer 2）

**Files:**
- Modify: `backend/app/services/dataset_reader.py`（`DatasetReadResult` 增全列矩阵）
- Modify: `backend/app/services/csv_dataset_reader.py`
- Modify: `backend/app/models/dataset.py`（`DatasetManifest.target_columns` → `value_columns` 语义）
- Modify: `backend/app/api/routes/dataset_manifests.py`（DTO + 创建）
- Create: `backend/tests/unit/test_csv_reader_all_columns.py`
- Modify: `backend/tests/unit/test_csv_reader_*.py`（去掉「强制恰好 1 列」预期）

- [ ] **B2.1 RED：全列数值校验**
  - 写测试：`test/flow_template.csv`（`time,target,extra`）经 reader 读出后，`value_columns == ["target","extra"]`，返回的全列矩阵 shape `[30,2]`，每列均为有限 float；非数值列 / NaN / Inf 仍按既有错误码（`csv_target_not_float` 泛化为 `csv_value_not_numeric` 等）报错。
  - Run: `cd backend && uv run pytest tests/unit/test_csv_reader_all_columns.py -v`
  - Expected: FAIL（现状 reader 强制 `len(target_columns)==1` 且只解析 target 列）。

- [ ] **B2.2 GREEN：reader 改为全列摄入**
  - 改 `CsvDatasetReader.read`：去掉 `csv_single_target_only` 强制；对除 `time_column` 外的**所有列**做有限 float 校验，产出 `value_columns` 顺序与全列 `values` 矩阵（`list[list[float]]`，shape `[row_count, n_value_cols]`）。
  - 改 `DatasetReadResult`：把 `target_values: list[list[float]]` 泛化为 `value_columns: list[str]` + `values: list[list[float]]`（保留 `target_values` 派生属性给过渡，或一次替换并同步调用点）。
  - 改 `DatasetManifest`：`target_columns` 字段语义改为 `value_columns`（manifest 只描述「这些数值列被摄入」，目标选择移到 load-job）。同步 `dataset_manifests.py` 的 `DatasetManifestCreate` DTO 与 `create_dataset_manifest`。
  - Run: `cd backend && uv run pytest tests/unit/test_csv_reader_all_columns.py tests/unit/test_csv_reader_happy_path.py tests/unit/test_csv_reader_targets.py tests/unit/test_csv_reader_format.py -v`
  - Expected: PASS。

> 边界：时间列校验（严格递增/不重复/等间隔/频率推断）逻辑不变，只把「单 target 解析」扩成「全数值列解析」。

## Task B3：选择期目标列 + 采样（Layer 3）

**Files:**
- Modify: `backend/app/schemas/dataset.py`（`DatasetLoadJobCreateDTO.split_config` 文档化新键）
- Modify: `backend/app/services/dataset_load_service.py`（`build_windows` 加 `max_samples` 均匀采样；load-job 读 `target_columns` 并校验恰好 1）
- Modify: `backend/app/api/routes/dataset_load_jobs.py`（透传）
- Create: `backend/tests/unit/test_load_target_selection.py`
- Create: `backend/tests/unit/test_sample_max_samples.py`

- [ ] **B3.1 RED：选择期 target 校验**
  - 写测试：load-job `split_config.target_columns` 必须是 manifest `value_columns` 的子集且**恰好 1 个**；缺失/多于 1 报 `load_target_columns_invalid`；选中的 target 写入 `Shard.target_columns`、`target_dim=1`。
  - Run: `cd backend && uv run pytest tests/unit/test_load_target_selection.py -v`
  - Expected: FAIL。

- [ ] **B3.2 GREEN：选择期 target**
  - 在 `DatasetLoadService._execute_job` 从 `split_config["target_columns"]` 取目标列，校验 ⊆ `value_columns` 且 len==1，回填 shard 的 `target_columns`/`target_dim`。
  - Run: `cd backend && uv run pytest tests/unit/test_load_target_selection.py -v`
  - Expected: PASS。

- [ ] **B3.3 RED：max_samples 均匀采样**
  - 写测试：30 行、`context_length=6,horizon=3,stride=1` 产 22 窗，`max_samples=5` 时沿序列**均匀**取 5 个（含首尾，索引近似等距），可复现（同配置同结果）；`max_samples` 缺省或 ≥ 窗数时全取。
  - Run: `cd backend && uv run pytest tests/unit/test_sample_max_samples.py -v`
  - Expected: FAIL。

- [ ] **B3.4 GREEN：均匀采样**
  - 在 `build_windows` 后追加 `subsample_windows(windows, max_samples)`：沿序列均匀取样（`numpy.linspace` 取整索引去重），保持 `sample_index` 连续重排。
  - Run: `cd backend && uv run pytest tests/unit/test_sample_max_samples.py tests/unit/test_sample_windowing.py -v`
  - Expected: PASS。

## Task B4：TsFile 落盘 + 指针化切片器（Layer 4）

**Files:**
- Modify: `backend/app/models/dataset.py`（`Shard` 加 `value_columns`/`tsfile_uri`）
- Modify: `backend/app/models/sample.py`（`SampleIndex.storage_ref` 指针化）
- Create: `backend/app/services/tsfile_store.py`（`dataframe_to_tsfile` 写 + `TsFileSlicer` 读）
- Modify: `backend/app/services/dataset_load_service.py`（落 TsFile 而非逐窗 JSONL）
- Modify: `backend/app/services/sample_store.py`（`read_by_ref` 改走切片器；`write_samples` 只建 `SampleIndex` 不再写值 JSONL，或整体由 `tsfile_store` 取代）
- Modify: `backend/app/services/run_executor.py:201,211`（读样本改走切片器）
- Modify: `backend/app/services/sample_forecast_service.py`（读 history/future 改走切片器）
- Create: `backend/tests/unit/test_tsfile_store.py`
- Modify: `backend/tests/unit/test_sample_jsonl_schema.py`、`test_sample_index_checksum.py`（适配新读路径）

- [ ] **B4.1 RED：TsFile 落盘 + 按行切片**
  - 写测试：给定全列矩阵 + 时间戳，`TsFileStore.write(shard_id, dataset_id, read_result)` 落出 `runtime/tsfiles/{shard_id}.tsfile`（表模型 `tsbench.<dataset_id>.<列>`）；`TsFileSlicer(path).slice(dataset_id, columns, row_start, row_end)` 返回 `np.ndarray`，值与源逐位一致。断言 history/future 两段切片拼回 `sample.v1` 等价结构。
  - Run: `cd backend && uv run pytest tests/unit/test_tsfile_store.py -v`
  - Expected: FAIL。

- [ ] **B4.2 GREEN：tsfile_store**
  - 实现 `TsFileStore.write`（构造 pandas DataFrame：`time`(ms epoch) + 各 value 列 + `dataset_id` tag，调 `dataframe_to_tsfile`）与 `TsFileSlicer.slice`（`TsFileDataFrame(path)["tsbench.<id>.<列>"][a:b]` → ndarray）。
  - Run: `cd backend && uv run pytest tests/unit/test_tsfile_store.py -v`
  - Expected: PASS。

- [ ] **B4.3 RED：load 走 TsFile + 样本指针化**
  - 写测试（service 级）：load-job 成功后产出 `Shard.tsfile_uri` 指向 `.tsfile`、`storage_uri` 指向同一文件；每条 `SampleIndex.storage_ref` 为指针引用（如 `{"dataset_id":..,"context":[s,e],"horizon":[s,e]}`），**不再写 `samples/{shard}.jsonl` 值文件**；`SampleStore.read_by_ref` 经切片器现切窗口，返回与旧 `sample.v1` 字段一致的 dict（含 `target_history`/`target_future`/时间戳/列名）。
  - Run: `cd backend && uv run pytest tests/unit/test_sample_jsonl_schema.py -v`
  - Expected: FAIL。

- [ ] **B4.4 GREEN：接通 load → TsFile → 切片器**
  - `DatasetLoadService._execute_job`：建 shard 后调 `TsFileStore.write` 落 TsFile，`SampleStore` 改为只建 `SampleIndex`（带指针 `storage_ref` + checksum 仍对「现切出的 canonical sample」算）；`read_by_ref` 用切片器组装 sample。run_executor / sample_forecast_service 的样本读取点同步改造（接口签名尽量不变，吸收在 `SampleStore` 内）。
  - Run: `cd backend && uv run pytest tests/unit/test_tsfile_store.py tests/unit/test_sample_jsonl_schema.py tests/unit/test_sample_index_checksum.py tests/api/test_dataset_load_flow.py -v`
  - Expected: PASS。

> 注意：`forecast.v1` JSONL 与 `ForecastStore` / `sample_forecast_service` 的**预测侧**读写不变（本轮决策）；仅样本（真值）侧从 JSONL 改 TsFile。`_cleanup_job_artifacts` 同步清理 `.tsfile` 半成品。

## Task B5：输入/答案分离（Layer 5）

**Files:**
- Modify: `backend/app/services/sample_store.py` 或新建 `backend/app/services/model_input.py`（构造 `ModelInput` 视图）
- Modify: `backend/app/services/run_executor.py:212`（传 ModelInput 给 adapter）
- Modify: `backend/app/services/timer_rest_adapter.py:36`
- Modify: `backend/app/services/stub_timer_adapter.py:15`
- Create: `backend/tests/unit/test_model_input_no_leak.py`
- Modify: `backend/tests/unit/test_stub_timer_adapter.py`、`test_timer_rest_adapter.py`

- [ ] **B5.1 RED：ModelInput 不含答案**
  - 写测试：`build_model_input(sample)` 返回的 dict **不含 `target_future`**，含 `horizon`(int)、`target_history`、`history_timestamps`、`future_timestamps`、`target_column_names`；两个 adapter 在收到无 `target_future` 的输入时，用 `model_input["horizon"]` 决定步数、产出 `[horizon, target_dim]`，结果与旧路径数值一致（桩确定性）。
  - Run: `cd backend && uv run pytest tests/unit/test_model_input_no_leak.py -v`
  - Expected: FAIL（adapter 仍读 `sample["target_future"]`，KeyError）。

- [ ] **B5.2 GREEN：分离输入/答案**
  - 新增 `build_model_input(sample)`；run_executor 在 `_execute_shard` 用它构造 adapter 入参，`target_future` 仅留服务端算指标用。
  - `timer_rest_adapter.py:36`：`horizon=sample["horizon"]`；`_build_request` 的 `output_length=[sample["horizon"]]`。
  - `stub_timer_adapter.py:15`：`horizon = sample["horizon"]`。
  - Run: `cd backend && uv run pytest tests/unit/test_model_input_no_leak.py tests/unit/test_stub_timer_adapter.py tests/unit/test_stub_forecast_rule.py tests/unit/test_timer_rest_adapter.py -v`
  - Expected: PASS。

## Task B6：MASE 指标 + 主排名切换（Layer 6）

**Files:**
- Modify: `backend/app/services/metric_service.py`（`compute_sample_metrics` 加 `mase`，context 算 naive 基线）
- Modify: `backend/app/services/run_executor.py:130,150,186,235`（各层聚合 + 刷新列表加 `mase`；`metric_set`）
- Modify: `backend/app/models/benchmark.py:27`（`Track.primary_metric_id` 默认 `"mase"`）
- Modify: `backend/app/services/track_service.py`（seed 的 MetricDefinition 含 mase；RankingList default 跟随）
- Modify: `backend/app/api/routes/ranking_lists.py`（默认 `metric` 跟随 primary）
- Modify: `frontend/src/...`（榜单默认指标展示，见 Task B7 前端同步）
- Create: `backend/tests/unit/test_mase_metric.py`
- Modify: `backend/tests/unit/test_sample_metrics.py`、`test_*_metrics.py`、ranking 测试

- [ ] **B6.1 RED：MASE sample 级**
  - 写测试：给定 `target_history`（算 naive 基线尺度 `scale = mean_{t}|y_t - y_{t-1}|` over context）、`target_future`、`forecast`，`compute_sample_metrics` 返回的 dict 含 `mase = MAE(forecast,target_future)/scale`；`scale==0`（平稳历史）时 `mase` 记为 `None`/失败（文档化）。
  - Run: `cd backend && uv run pytest tests/unit/test_mase_metric.py -v`
  - Expected: FAIL。

- [ ] **B6.2 GREEN：MASE 计算**
  - `compute_sample_metrics(target_future, forecast, target_history)`：加 `mase` 键（last-value naive，m=1）。注意签名新增 `target_history`，同步 `run_executor._execute_shard` 调用点传入 history。
  - Run: `cd backend && uv run pytest tests/unit/test_mase_metric.py tests/unit/test_sample_metrics.py -v`
  - Expected: PASS。

- [ ] **B6.3 RED：mase 进各层聚合 + 榜单主指标**
  - 写测试：run 执行后 `MetricResult` 出现 `metric_id="mase"` 的 sample/shard/task/unit 四级；`Track.primary_metric_id` 默认 `"mase"`；`refresh_ranking` 对 `mase` 也建 entry；`GET /tracks/{id}/ranking` 默认 `metric=mase` 返回排名（lower is better）。
  - Run: `cd backend && uv run pytest tests/unit/test_run_execution_success.py tests/unit/test_latest_valid_result.py tests/api/test_ranking_list_api.py -v`
  - Expected: FAIL。

- [ ] **B6.4 GREEN：接通 mase 全链**
  - `run_executor`：sample 级 `metrics` 已含 mase（B6.2）；`_execute_shard/_execute_task/_execute_unit` 对 mase 调 `aggregate_metric` 并写 `MetricResult`；`execute_run` 刷新列表 `["mse","mae"]` → `["mase","mse","mae"]`；`BenchmarkingRun.metric_set` 默认含 mase。
  - `Track.primary_metric_id` 默认改 `"mase"`；`seed` 的 `MetricDefinition` 增 mase（`direction=lower_is_better`，supported_levels 同 mse）；`ranking_lists.py` 路由默认 `metric` 取 track 的 `primary_metric_id`。
  - Run: `cd backend && uv run pytest tests/unit/test_run_execution_success.py tests/unit/test_shard_metrics.py tests/unit/test_task_unit_metrics.py tests/unit/test_latest_valid_result.py tests/unit/test_best_result.py tests/api/test_ranking_list_api.py -v`
  - Expected: PASS。

> 连锁清单（spec §6 #5）：① `Track.primary_metric_id` 默认；② `RankingList.default_metric_id`（由 primary 派生，自动跟随）；③ ranking 路由默认 metric；④ 前端榜单默认指标列（Task B7）；⑤ 报告/进度展示。mse/mae 仍全程计算并入榜，只是不再是默认视图。

## Task B7：测试同步 + 真实 CSV 复跑对照基线（Layer 7）

**Files:**
- Modify: 引用 `target_columns`/`target_future`/`single_target` 的测试（实测 17 个文件，见下）
- Modify: `frontend/src/api/*`、榜单/向导组件（manifest `value_columns`、load-job `target_columns`、默认指标 mase）
- Modify: `docs/developer/{data-model,key-flows}.md`（实体/流程随重构更新）
- Create: `backend/tests/e2e/test_real_csv_tsfile_flow.py`（真实 CSV + TsFile 端到端）

实测需同步的后端测试文件（grep `target_columns|csv_single_target|target_future|single_target`）：
```
tests/run_helpers.py
tests/unit/test_dataset_load_constraints.py        tests/unit/test_csv_reader_targets.py
tests/unit/test_sample_metrics.py                  tests/unit/test_csv_reader_happy_path.py
tests/unit/test_csv_reader_format.py               tests/unit/test_dataset_entities.py
tests/unit/test_stub_timer_adapter.py              tests/unit/test_sample_jsonl_schema.py
tests/unit/test_stub_forecast_rule.py              tests/unit/test_timer_rest_adapter.py
tests/unit/test_dataset_load_job_service.py        tests/api/test_dataset_to_track_flow.py
tests/api/test_dataset_load_flow.py                tests/api/test_benchmarking_run_create.py
tests/api/test_sample_forecast_api.py              tests/e2e/test_mvp_benchmarking_flow.py
```

- [ ] **B7.1 同步后端测试与 helpers**
  - 把 `manifest.target_columns` → `value_columns`、load-job `split_config` 增 `target_columns`、adapter 测试去掉对输入里 `target_future` 的依赖、metric 测试补 mase 维度。`tests/run_helpers.py` 是公共构造器，优先改它收敛大部分用例。
  - Run: `cd backend && uv run pytest -q`
  - Expected: PASS（全量绿）。

- [ ] **B7.2 前端同步**
  - manifest 创建表单用 `value_columns`；列/切分步骤新增「选目标列（恰好 1）」+「max_samples」；榜单默认指标列改 `mase`，mse/mae 作为可切换诊断列。
  - Run: `cd frontend && npm test`（必要时 `npm run test:e2e`）
  - Expected: PASS。

- [ ] **B7.3 真实 CSV + TsFile 端到端**
  - 用 `test/flow_template.csv` 走新通路（全列摄入→TsFile→指针切片→ModelInput→MASE）端到端，断言 run `succeeded`、`extra` 列被摄入到 TsFile（`tsbench.<id>.extra` 可读）、target 仍为 `target`。
  - Run: `cd backend && uv run pytest tests/e2e/test_real_csv_tsfile_flow.py -v`
  - Expected: PASS。

- [ ] **B7.4 复跑对照 Track A 基线**
  - 用 A0 的脚本对新通路复跑，与 `baseline-record.md` 逐位核对：桩确定性下，**相同 (context/horizon/stride/seed/target) 应得相同 forecast 与 mse/mae**；mase 为新增列（基线无，单独记录）。差异需能解释（如采样窗集变化）。
  - 验收：mse/mae 与基线一致或差异可解释；记入 `baseline-record.md` 的「重构后」对照段。

---

## 依赖与并行

- B1 必须最先（依赖 + DB）。
- B2 → B3 → B4 线性（摄入→选择→落盘）。
- B5（输入/答案分离）、B6（MASE）不依赖 TsFile，可与 B4 并行（写集不同：B5 动 adapter + model_input，B6 动 metric + ranking + track）。
- B7 收口，依赖 B2–B6 全部完成。
- 写集冲突点：`run_executor.py` 被 B4/B5/B6 都会改 → 指定一个集成 owner 串行合并这三处，或按 `_execute_shard` 内分段分工。

## 风险控制

- **tsfile 二进制兼容**：B1 解析失败则钉死 spike 验证版本（`tsfile==2.3.0`）。
- **切片器性能**：指针化每次切片触发 I/O；若 e2e 变慢，evaluate 是否对单 shard 缓存 `TsFileDataFrame` 句柄（懒加载视图，初始化只扫元数据）。
- **checksum 语义变化**：样本不再物化为 JSONL，`SampleIndex.checksum` 改为对「切片器现切出的 canonical sample」算，保持可复现断言。
- **MASE 退化**：平稳历史 `scale==0` 必须有确定规则（记 None=该 sample 失败），否则 NaN 污染聚合与榜单。
- **答案泄露回归**：B5 后加断言「adapter 收到的 dict 不含 `target_future` 键」防回退。
- **DB schema 漂移**：dev 库删后重建；生产/他人环境需同样重建（MVP 无迁移工具，记入交接）。

## 评审清单

- [ ] spec §4 两条 track 的每一步在本计划有对应 Task 或显式边界。
- [ ] spec §6 七个待讨论议题均有「本轮做/延后」结论。
- [ ] 本轮三项锁定决策（文档先行 / 输出 JSONL / MASE 主排名）在计划中落实。
- [ ] 所有生产行为以失败测试起步（TDD）。
- [ ] 后端可由 `cd backend && uv run pytest` 验收；前端由 `npm test` 验收。
- [ ] 无 git 动作分配给实现者。

---

### 更新记录
- 2026-05-25：创建。基于 spec 整体设计讨论 + 已读代码核对，细化为 Track A（基线脚本化）+ Track B（7 层 TDD 任务），锁定「文档先行 / forecast 维持 JSONL / MASE 主排名」三项决策，列出实体字段连带改动、17 个待同步测试文件、依赖并行与风险控制。
- 2026-05-25：评审通过后实现 Track B 全 7 层（TDD + 多智能体并发）。后端 136 passed / 前端 18 passed。见上「实现状态」节（含偏差：requires-python→3.14、桩值跨重载不可比）。
- 2026-05-25：补做 Track A 现场基线（A0）——`scripts/baseline-run.sh` 实跑落 `docs/superpowers/baselines/2026-05-25-baseline-record.md`（run succeeded、extra 入 TsFile、默认榜 mase）。
