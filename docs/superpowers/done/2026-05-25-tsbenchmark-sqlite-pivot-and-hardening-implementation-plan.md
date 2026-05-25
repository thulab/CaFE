# TSBenchmark SQLite 存储改造 + 清理加固 实现计划

> ✅ **已执行完成（2026-05-26）**。P0–P6 全部落地：SQLite `SeriesPoint` 逐点行存储、CSV/TsFile 双输入 reader、MASE 主排名 + 缺席可见化、榜单 direction、终态/队列健壮性、组织层守卫与 wizard 原子性、回看链接、文档同步、基线复跑对照。后端全量 **176 passed**。cleanup 24 项除排除的 #10 外全部化解（代码或文档登记）。详见文末「更新记录」。本文件随移入 `done/`。

> **给实现者：** 本计划把 [SQLite 存储设计 spec](./2026-05-25-tsbenchmark-input-readers-and-sqlite-storage-design.md) + [清理计划 20 项](./2026-05-25-tsbenchmark-data-path-cleanup-plan.md) 收敛成**有顺序、配测试**的执行计划。所有生产改动遵循 TDD：先写失败测试（RED）→ 验证 RED → 最小实现（GREEN）→ 全量 `uv run pytest` 保持绿。步骤用 `- [ ]` 跟踪。Git 操作由用户负责。

**起点：** 当前已提交「TsFile-当存储」版本（Track B 已落地：全列摄入 / 切窗+采样 / TsFile 存储 / 指针样本 / 输入答案分离 / MASE）。

**目标：** ① 把存储从 per-shard TsFile 改为 **SQLite 逐点行**；② 输入支持 **CSV 或 TsFile**；③ 按优先级清掉 cleanup 20 项里**本轮纳入**的部分。

**本轮排除（不做）：** 多序列/面板（#10）、协变量角色模型 / 多目标、真实推理接入（仅做 #16 的健壮性前置）、数据指纹复现、evaluate/govern 接入、Anchor/合成数据。单序列 + 单设备仍是边界。

---

## 阶段总览与 cleanup 映射

| 阶段 | 内容 | 化解的 cleanup 项 |
|---|---|---|
| **P0** | 立基线锚点（改动前） | —（立 anchor） |
| **P1** | P1 健壮性修复（与存储无关，先做） | #16 #17 #1 #2 |
| **P2** | SQLite 逐点行存储改造（spec 核心） | #4 #5 #6 #7 #8 + L4 非原子写 |
| **P3** | CSV/TsFile 双输入 reader | #2(共享 helper) #3 |
| **P4** | MASE / 榜单 / 终态 / 组织层一致性 | #12 #13 #14 #15 #18 #21 |
| **P5** | P2/P3 余账 + 文档登记 | #9 #11 #19 #20 #22 #23 #24 + 回看链接 None |
| **P6** | 测试同步 + 复跑对照基线 | —（验收） |

> 依赖：P0 最先；P1 独立可先做；P2→P3 线性（存储先稳，输入再加）；P4/P5 可与 P2/P3 并行（写集不同）；P6 收口。

---

## P0 · 立基线锚点（改动前必做）

> 当前代码是确定性桩，先把现状行为固化，作为"改完没改坏"的逐位对照。

**Files:** Create `scripts/baseline_flow.py`（或 .sh）、`docs/superpowers/baselines/2026-05-25-baseline-record.md`

- [ ] **P0.1** 起后端 + 桩（`TSBENCHMARK_MODEL_ADAPTER=stub` 最省）。
- [ ] **P0.2** `test/flow_template.csv` 走完整链：upload → manifest(`value_columns`) → load-job(`target_columns=["target"]`, ctx/horizon/stride) → wizard 建赛道 → run → 轮询 progress 到终态。
- [ ] **P0.3** 记录到 `baseline-record.md`：每样本 forecast、sample/shard/task/unit 的 mase/mse/mae、三指标榜单排名。**这是 P6 复跑的逐位锚点。**

---

## P1 · P1 健壮性修复（与存储无关，先做）

### Task P1.1：adapter 错误不崩 run、不卡队列（#16）
**Files:** `services/run_executor.py`、`api/routes/benchmarking_runs.py`、`services/forecast_store.py`、`tests/unit/test_run_adapter_failure.py`(new)
- [ ] RED：注入一个 `forecast` 抛 `TimerServiceError` 的 adapter，断言：run 落终态（非卡 `running`）、该样本 forecast 行 `status=="failed"` 且有 `error_code`、shard/unit 据失败正确判级、**且后续提交的 run 能执行**。
- [ ] GREEN：`_execute_shard` 包 try/except——单样本失败写 forecast 失败行（不崩整 run）；`_execute_in_background` 用 `try/finally` 确保 `queue.complete` 必被调用。

### Task P1.2：排队 run 自动执行（#17）
**Files:** `api/routes/benchmarking_runs.py`、`workers/run_queue.py`、`tests/unit/test_run_queue_drain.py`(new)
- [ ] RED：连提两个 run，断言**第二个最终也跑到终态**。
- [ ] GREEN：`_execute_in_background` 末尾按 `queue.complete` 返回的下一个 run_id 起新线程（或队列驱动）。

### Task P1.3：时间轴校验加固（#1 混合时区、#2 频率按时长比较）
**Files:** `services/csv_dataset_reader.py` → 抽出 `services/time_axis.py`（`validate_time_axis`）、`tests/unit/test_time_axis.py`(new)
- [ ] RED：① 混合 offset/naive 的时间 → 断言抛 `ApiError("csv_mixed_timezone")` 而非 `TypeError`；② manifest 频率 `"60m"` 实际 hourly → 断言**加载成功**（按时长比较，非字符串）。
- [ ] GREEN：抽 `validate_time_axis(timestamps, declared_freq) -> frequency`（含递增/重复/等间隔/频率按秒数比较 + 时区一致性预检），CsvDatasetReader 改用它。**P3 的 TsFile reader 复用同一函数。**

---

## P2 · SQLite 逐点行存储改造（spec 核心）

### Task P2.1：SeriesPoint 表 + 序列存取（逐点行）
**Files:** Create `models/series_point.py`（`SeriesPoint`）、`services/series_store.py`（`write` 批量插 + `slice(shard_id, columns, start, end)` 范围查询 + `slice_timestamps`）、`tests/unit/test_series_store.py`
- [ ] RED：给定全列矩阵 + ISO 时间戳，`SeriesStore.write` 落库；`slice` 用 `WHERE shard_id=? AND row_index BETWEEN ? AND ?`（**闭区间，无 +1**）返回逐位一致的值；`slice_timestamps` 返回 ISO 原文。
- [ ] GREEN：`SeriesPoint(series_point_id, shard_id idx, row_index, ts:ISO, values_json:[float])` + 复合索引 `(shard_id, row_index)`；`SeriesStore` 实现写/切。

### Task P2.2：加载落库 + 样本指针化改走 SQLite
**Files:** `services/dataset_load_service.py`、`services/sample_store.py`、`models/dataset.py`(Shard 删 `tsfile_uri`/`dataset_id`)、`models/sample.py`(SampleIndex `storage_ref` 瘦身、`materialized`→False/删、删 `materialized_sample_uri`)、`core/config.py`(删 `tsfiles_dir`)、`tests/unit/test_sample_jsonl_schema.py`/`test_sample_index_checksum.py`(改)
- [ ] RED：load-job 成功后**无 `.tsfile`**；`SampleStore.read_by_ref` 经 SQL 拼出的 `sample.v1` 与现状逐字段一致；**同一数据+切分两次加载，样本 checksum 相等**（#7）；`ts` 为 ISO 原文（#8）；失败时事务回滚、库内无半成品（替代非原子写）。
- [ ] GREEN：`_execute_job` 调 `SeriesStore.write` 取代 `TsFileStore.write`；`SampleStore` 切片走 `SeriesStore`；`storage_ref` 改 `{shard_id, target_columns, context, horizon}`（去 `dataset_id`，#5）；`checksum` 对内容算、排除随机 ID（#7）；`materialized=False`（#4），删 `materialized_sample_uri`（#6）。

### Task P2.3：删除 TsFile-当存储
**Files:** `services/tsfile_store.py`（`TsFileStore.write`/`TsFileSlicer` 存储用途删除；如 P3 需要保留读输入能力则改造）、`services/run_executor.py`/`sample_forecast_service.py`（确认仅经 `SampleStore.read_by_ref`，无直接 TsFile 调用）
- [ ] 验收：全量 `uv run pytest` 绿；grep 无残留 `tsfile_uri`/`tsfiles_dir`/存储侧 `TsFileSlicer`。

---

## P3 · CSV / TsFile 双输入 reader

### Task P3.1：TsFileDatasetReader（单设备 + 表模型）
**Files:** Create `services/tsfile_dataset_reader.py`、`tests/unit/test_tsfile_dataset_reader.py`
- [ ] RED：给一个**单设备表模型** TsFile，`read(...)` 产出 `DatasetReadResult`（`value_columns`/`values`/`timestamps(ISO)`/`frequency`），与等价 CSV 结构一致；**多设备 → `tsfile_multiple_devices`**、树模型 → `tsfile_tree_model_unsupported`、不等间隔 → 拒（复用 P1.3 的 `validate_time_axis`）。
- [ ] GREEN：用 `TsFileDataFrame` 读单设备全列 → 内存矩阵 → DatasetReadResult；`time_column` 忽略；CSV 专属字段填惰性值。

### Task P3.2：reader 工厂 + 上传嗅探分支（#3）
**Files:** `services/dataset_reader.py`(加 `get_dataset_reader`)、`services/dataset_load_service.py:115`、`api/routes/dataset_manifests.py`、`models/dataset.py`(`file_format` 取值/`reader_type`)、`tests/api/test_dataset_load_flow.py`(改)
- [ ] RED：`file_format="tsfile"` 的 manifest 走 TsFileDatasetReader；上传 TsFile 的嗅探返回 device/物理量/行数（非分隔符/编码）；CSV 上传嗅探的 `has_header` 用真判、`inferred_type` 真推断（#3，或明确标注"仅预览"）。
- [ ] GREEN：`get_dataset_reader(file_format)` 替两处硬编码；上传路由按 `file_format` 分支。

---

## P4 · MASE / 榜单 / 终态一致性

### Task P4.1：建赛道默认主指标对齐（#12）
**Files:** `api/routes/tracks.py:14`、`tests/api/...`
- [ ] RED：`POST /tracks` 不传 `primary_metric_id` → 断言 `Track.primary_metric_id=="mase"`。
- [ ] GREEN：`tracks.py:14` 默认改 `"mase"`。

### Task P4.2：榜单排序尊重 direction（#15）
**Files:** `services/ranking_service.py`、`tests/unit/test_ranking_direction.py`(new)
- [ ] RED：注册一个 `direction=higher_is_better` 的指标，断言其 `RankingEntry` 按 value 降序、rank=1 为最大。
- [ ] GREEN：`refresh_ranking` 排序按 `MetricDefinition.direction`（lower→asc / higher→desc）。

### Task P4.3：run 终态纳入 partial unit（#18）
**Files:** `services/run_executor.py:116-123`、`tests/unit/test_run_terminal_status.py`(new)
- [ ] RED：`[1 succeeded + 1 partial]`→ run `partial_succeeded`；`[全 partial]`→ `partial_succeeded`（非 `failed`/非 `succeeded`）。
- [ ] GREEN：终态把 partial_succeeded unit 纳入判定。

### Task P4.4：MASE 边界可见 + m 决策（#14 #13）
**Files:** `services/metric_service.py`、`services/report_service.py`、`docs/developer/data-model.md`
- [ ] **决策（#13）**：确认 `_mase_scale` 维持 m=1（文档登记"季节 m 暂不支持"），还是实现"按频率推 m"。**默认按 m=1 登记**（与 #11 等间隔限制一致）。
- [ ] **#14**：平稳历史导致 mase 缺席时，在 report 显式标注（如该 unit `metrics` 含 `mase: null` + 原因），不静默缺席。

### Task P4.5：建赛道补「block 已属别的 track」守卫（#21）
**Files:** `services/track_service.py`、`tests/unit/test_track_block_assignment.py`(new)
- [ ] RED：同一 capability block 连建两次 track，第二次断言报 `capability_block_already_assigned`（镜像 shard 的 `shard_already_assigned`）。
- [ ] GREEN：`create_track_with_blocks` 建 track 前，若任一 block `track_id` 已非空 → 抛 `capability_block_already_assigned`。

---

## P5 · P2/P3 余账 + 文档

- [ ] **#19** 取消：`cancel_run` 把排队中的 run 移出队列（或 `complete` 跳过 `cancel_requested` 的）；`tests/unit/test_cancel_queued.py`。
- [ ] **#20** progress：`build_run_progress` 的 `completed_samples`/`failed_samples` 按已写 forecast 行真统计，或文档标注"进度仅到 task 粒度"。
- [ ] **回看链接**：`build_sample_forecast` 的 `links.report`/`links.ranking` 回填真实 id（前端可跳回）。
- [ ] **#9 / #11 文档登记**：max_samples 抽稀与 stride 的相互作用、等间隔挡掉日历/亚秒频率——写进 `key-flows.md`/`data-model.md` 已知约束。
- [ ] **#23** 模型重名：`create_model` 校验 name 唯一（或允许重名则 `list_models` 不去重），消除"建得了、列表看不到"的不一致。
- [ ] **#24** wizard 事务：`create_real_dataset_track` 两步包一个事务（失败回滚 block 与 shard 归属），避免 orphan capability block。
- [ ] **#22 文档登记（后议）**：adapter 选择只看全局 `settings.model_adapter`、不看 `Model.adapter_type`——接真实推理时再议按模型路由；当前文档标注即可。
- [ ] **文档同步**：`data-model.md`（SeriesPoint 新表、Shard 字段、SampleIndex 变更）、`key-flows.md`（存储改 SQLite、双输入 reader）。

---

## P6 · 测试同步 + 复跑对照基线

**Files:** 主 plan B7 列的 17 个引用 `target_columns`/`value_columns`/`single_target` 的测试 + `tests/run_helpers.py`；Create `tests/e2e/test_real_csv_sqlite_flow.py`、`tests/e2e/test_tsfile_input_flow.py`
- [ ] **P6.1** 同步后端测试与 helpers，全量 `uv run pytest -q` 绿。
- [ ] **P6.2** e2e：CSV 走新通路（SQLite 存储）+ TsFile 单设备输入，端到端断言 run `succeeded`、`extra` 列入库、target 正确。
- [ ] **P6.3** 复跑对照 **P0 基线**：桩确定性下，相同 (ctx/horizon/stride/seed/target) **forecast 与 mse/mae/mase 逐位一致或差异可解释**。
- [ ] **P6.4** 前端连带（SQLite pivot 影响）：manifest `value_columns`、load-job `target_columns`、榜单默认 `mase`——`cd frontend && npm test`。

---

## 风险与回退

- **DB schema 漂移**：删 `tsfile_uri`/`dataset_id` + 加 `SeriesPoint` 需删 dev 库重建（无迁移工具，记入交接）；测试用临时库不受影响。
- **tsfile 依赖**：P2 后存储不再用 tsfile，但 P3 读 TsFile 输入仍需 → `pyproject` 保留 `tsfile`/`numpy`/`pandas`。
- **基线对照**：P0 必须先跑出来，否则 P6.3 无锚点。
- **大改集中点 `run_executor.py`**：P1.1/P1.2/P4.3 都改它 → 指定串行 owner 合并。

---

## 评审清单
- [ ] SQLite spec 的每条决策（D1–D7）在 P2/P3 有对应 Task。
- [ ] cleanup 本轮纳入项（#1–#9,#11–#24，除 #10）均落到某阶段；#10 多序列明确排除。
- [ ] 所有生产改动以失败测试起步。
- [ ] P0 基线先于一切改动；P6.3 复跑对照。
- [ ] 未纳入项（#10 多序列、协变量/多目标、真实推理接入、数据指纹、evaluate/govern）在"本轮排除"显式列出。

---

### 更新记录
- 2026-05-25：创建。把 SQLite 存储 spec + cleanup 20 项收敛成 P0–P6 六阶段 TDD 计划；P0 立基线、P1 健壮性(#16/#17/#1/#2)、P2 SQLite 逐点行存储(#4–#8+非原子)、P3 双输入 reader(#2/#3)、P4 MASE/榜单/终态一致性(#12–#15,#18)、P5 余账+文档(#9/#11/#19/#20+回看链接)、P6 测试+复跑对照。按用户决定**排除多序列(#10)**。
- 2026-05-25：补完组织层/模型管理 review，纳入 cleanup #21–#24——P4 增 Task P4.5（#21 建 track 缺 block-已占守卫）、P5 增 #23 模型重名 / #24 wizard 事务 / #22 adapter 不看 adapter_type（后议文档登记）。至此后端各层 review 全覆盖，cleanup 共 24 项全部映射到阶段（除排除的 #10）。
- **2026-05-26：执行完成**。落地纪要：
  - **P1**：#16 adapter 异常不崩 run（写 failed forecast 行）+ try/finally 保证 `queue.complete`；#17 排队 run 自动起线程；#1/#2 抽出 `time_axis.validate_time_axis`（混合时区 → `csv_mixed_timezone`，频率按时长比较）。
  - **P2**：`SeriesPoint` 表 + `SeriesStore`（逐点写/闭区间切片）；样本指针化、`read_by_ref(session, ...)` 走 SQLite；删 `Shard.tsfile_uri`/`dataset_id`、`config.tsfiles_dir`；checksum 内容化排除随机 ID（#7）；`materialized=False`（#4）；ts 存 ISO（#8）；加载原子化（事务 + 失败 rollback）。
  - **P3**：`TsFileDatasetReader`（单设备表模型）+ `get_dataset_reader(file_format)` 工厂；#3 上传嗅探按扩展名分支（CSV `has_header`/类型真判、TsFile 设备/物理量）。
  - **P4**：#12 `POST /tracks` 默认 mase；#15 榜单按 `direction` 排序；#18 终态纳入 partial unit；#14 MASE 缺席经 `SampleMetrics.mase_unavailable_reason` 可见（`flat_history`/`no_history_diffs`）并在 report 标注；#13 m=1 决策文档登记；#21 建 track 守卫 `capability_block_already_assigned`。
  - **P5**：#19 取消排队 run 出队（`RunQueue.remove`）；#20 progress 真实统计 forecast 行；#23 `model_name_taken`；#24 wizard 失败补偿清理（不留 orphan block）；回看链接回填 report/ranking id；#9/#11/#13/#22 写入 `data-model.md §10` 已知约束；data-model/key-flows 全面同步 SQLite + 双输入。
  - **集成竞态修复**：终态 `status`/`report_id`/三张榜单原本跨多次 commit 暴露，轮询会「看到 succeeded 却查不到榜单」。改为 `refresh_ranking(commit=False)` 攒进 `generate_run_report` 的单次提交，原子可见（`run_executor.py` + `ranking_service.py`）。
  - **P6**：全量 **176 passed**（含新 e2e、双输入、12×e2e 抗 flaky）；`baseline_run.py` 适配 SQLite 后复跑，与 P0 基线（2026-05-25）对照——**真值 `target_future` 逐位一致、结构一致**，仅 forecast 值因 stub 噪声以**随机 `sample_id`** 为种子而变（既有特性，P0 记录已声明，与存储 pivot 无关）→ 满足 P6.3「差异可解释」。
