# TSBenchmark 输入读取器 + SQLite 存储设计

> ✅ **已实现（2026-05-26）**。SQLite `SeriesPoint` 逐点行存储 + CSV/TsFile 双输入 reader 全部落地，后端 176 passed。实现详见 [sqlite-pivot-and-hardening 实现计划](./2026-05-25-tsbenchmark-sqlite-pivot-and-hardening-implementation-plan.md) 的「更新记录」。本文件随移入 `done/`。

**日期：** 2026-05-25

**性质：** 设计 spec。对**当前已落地的 TsFile-当存储**通路做一次方向修正：**TsFile 从「内部存储格式」降为「输入格式之一」，内部真值源改为 SQLite 逐点行存储**。

**与既有文档的关系：**
- 取代实现计划 [`../done/2026-05-25-tsbenchmark-overall-design-implementation-plan.md`](./2026-05-25-tsbenchmark-overall-design-implementation-plan.md) 中 **Layer 4「per-dataset TsFile 单一真值源」** 的存储决策；该计划其余层（全列摄入、采样、输入/答案分离、MASE）保持有效。
- 直接化解清理计划 [`../plans/2026-05-25-tsbenchmark-data-path-cleanup-plan.md`](./2026-05-25-tsbenchmark-data-path-cleanup-plan.md) 的 #7（checksum 含随机 ID）、#8（ms-epoch 时区往返）、L4-非原子写 三项（见 §11）。
- 实体链路仍遵循 [`2026-05-16-tsbenchmark-mvp-entity-structure-design.md`](../specs/2026-05-16-tsbenchmark-mvp-entity-structure-design.md) 的「逻辑实体稳定、物理存储可替换」原则——本次正是替换物理存储。

**输入材料：** 当前实现 `backend/app/{services,models}/*`（已提交，工作树干净）、`docs/developer/{key-flows,data-model}.md`、`docs/reference/tsfile-dataframe-manual.md`。

> 标注：✅ = 已读当前代码核实（附 `file:line`）；⚠️ = 推断/待实现时确认。

---

## 0. 目标与边界

- **目标：** 真实数据集**输入**支持 **CSV 或 TsFile** 两种文件格式（同一 `DatasetReader` 契约）；**内部存储统一用 SQLite**（per-shard 逐点行），不再有独立的 per-shard `.tsfile` 真值源。
- **不做（本轮）：** 多序列/面板数据、不规则（非等间隔）时间序列、真实推理服务接入、数据指纹复现。这些维持既有边界（见 §12）。

---

## 1. 现状（已读当前代码核实 ✅）

当前已落地的是 **TsFile-当存储** 版本：

| 现状组件 | 行为 | 位置 |
|---|---|---|
| `CsvDatasetReader.read` | 全列数值摄入，产 `DatasetReadResult(value_columns, values, ...)` | `services/csv_dataset_reader.py` |
| `DatasetReadResult` | `columns/rows/timestamps/value_columns/values/frequency/encoding/delimiter` + `row_count`/`column_matrix` | `services/dataset_reader.py:8` |
| `DatasetReader` Protocol | `read(path, time_column, value_columns, frequency)`——扩展点已预留 | `services/dataset_reader.py:28` |
| `DatasetLoadService._execute_job` | reader → 选 target(恰好1) → `build_windows`+`subsample_windows` → `TsFileStore.write` 落 `.tsfile` → `SampleStore.write_samples` 建指针 | `services/dataset_load_service.py` |
| `TsFileStore.write` / `TsFileSlicer` | 写/切 per-shard `.tsfile`（表模型 `tsbench.<dataset_id>.<列>`） | `services/tsfile_store.py` |
| `SampleStore` | 指针化：`SampleIndex.storage_ref` 存行号区间，`read_by_ref` 从 `.tsfile` 现切拼 `sample.v1` | `services/sample_store.py` |
| `Shard` | 含 `tsfile_uri`/`dataset_id`/`value_columns`/`target_columns`；`storage_uri` 指 `.tsfile` | `models/dataset.py:47` |
| `config.tsfiles_dir` | `runtime/tsfiles/` | `core/config.py:43` |

**与存储无关、本次保留不动的：** 全列摄入（L2）、`build_windows`+`subsample_windows`（L2/L3 采样）、输入/答案分离 `model_input.py`（L5）、MASE `metric_service`（L6）、选择期单目标（L3）。

---

## 2. 本轮锁定决策（经讨论确认）

| # | 议题 | 决策 |
|---|---|---|
| D1 | 输入格式 | **CSV 或 TsFile**；两个 reader 实现同一 `DatasetReader` 契约，**都产出 `DatasetReadResult`（内存矩阵）** |
| D2 | TsFile 输入路径 | **对称**：`TsFileDatasetReader` 读进内存 → 同管线存库（不短路、不引用原文件） |
| D3 | TsFile 输入边界 | **单设备 + 表模型**；多设备/树模型拒绝 |
| D4 | 内部存储 | **SQLite 逐点行**：新增 `SeriesPoint` 表，**每点一行 + 该点各列值打成 `values_json` 向量**（备选 EAV，见 §5） |
| D5 | 样本 | **仍是指针**（行号区间），切片改走 **SQL 范围查询**；不物化、不再切 TsFile |
| D6 | 时间戳 | `SeriesPoint.ts` 存 **ISO 8601 原文**（保留原始 offset，不经 ms-epoch 往返） |
| D7 | 不变量 | **等间隔仍强制**、**单序列仍是边界**（见 §12） |

---

## 3. 架构总览

```
输入 CSV   ──CsvDatasetReader─────┐
输入 TsFile ──TsFileDatasetReader──┤  get_dataset_reader(file_format)
   (单设备/表模型)                  │
                                   ▼
                          DatasetReadResult(内存矩阵)
                                   │
                ┌──────────────────┼─────────────────────┐
                ▼                  ▼                       ▼
        选 target(恰好1)    build_windows+采样      批量 INSERT SeriesPoint(N 行)
                                   │                  （单一真值源，在 SQLite）
                                   ▼
                        SampleIndex 指针(行号区间)
                                   │
   run_executor / sample_forecast ── read_by_ref ──► SELECT … WHERE row_index BETWEEN … ──► sample.v1
```

一句话：**两种输入文件 → 同一内存矩阵 → 序列逐点写进 SQLite（唯一真值源）→ 样本是行号指针 → 读样本 = SQL 范围查询现拼。**

---

## 4. 输入层设计

### 4.1 契约与工厂

- `DatasetReader` Protocol 不变（`read(path, time_column, value_columns, frequency) -> DatasetReadResult`）。
- 新增工厂 `get_dataset_reader(file_format) -> DatasetReader`：`"csv" → CsvDatasetReader`、`"tsfile" → TsFileDatasetReader`。**替换两处硬编码** `CsvDatasetReader()`：加载路径（`dataset_load_service.py:115`）与上传嗅探（`dataset_manifests.py:38`）。`DatasetLoadJob.reader_type` 随之填。

### 4.2 `TsFileDatasetReader`（新）

读一个 TsFile **输入**文件 → `DatasetReadResult`，逐字段对齐 CsvDatasetReader 的产物，下游零改：

| 步 | 动作 | 错误码（建议） |
|---|---|---|
| 1 | 打开 TsFile，确认**表模型**；树模型拒 | `tsfile_tree_model_unsupported` |
| 2 | 列 device/tag，要求**恰好 1 个** | `tsfile_multiple_devices` |
| 3 | 取该 device 物理量为 `value_columns`；manifest 给了则按子集选 | `tsfile_value_column_missing` |
| 4 | **全量读**各列 → `values[N][C]`（对称路径接受 eager） | — |
| 5 | 读时间轴 → `timestamps`（ISO） | — |
| 6 | 共享校验：严格递增/不重复、**等间隔**、值有限、频率推断/核对 | 复用 §4.3 |

- `time_column` 参数对 TsFile **忽略**（时间是内建轴）。
- `DatasetReadResult` 的 CSV 专属字段（`encoding`/`delimiter`/`rows`）对 TsFile 填惰性值：`encoding="" / delimiter=""`，`rows` 仅需满足 `row_count==N`（填 `[{} for _ in range(N)]` 或后续把 `rows` 瘦身为 `row_count`，见 §11 债务）。
- **`tsfile` 依赖保留，但仅用于此处读输入**（不再用于存储）。

### 4.3 共享时间轴校验

把 `CsvDatasetReader` 里的「严格递增/不重复/等间隔/频率推断与核对」抽成 reader 无关的 `validate_time_axis(timestamps, declared_freq) -> frequency`，两个 reader 复用，避免重抄；顺手把频率比较改为**按时长**而非字符串（化解清理计划 #2）。

---

## 5. 存储层设计（核心）：`SeriesPoint` 逐点行

新增 SQLModel 表，作为 per-shard 序列的**唯一真值源**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `series_point_id` | `str` | 主键，`default_factory=new_id` |
| `shard_id` | `str` | `index=True` |
| `row_index` | `int` | 0-based，对应校验后数据行 |
| `ts` | `str` | ISO 8601 原文（保留原始 offset，**不经 ms 往返**） |
| `values_json` | `list[float]` | JSON 列；该点各 value 列的值，**顺序 = `Shard.value_columns`** |

- **复合索引 `(shard_id, row_index)`**：样本读取靠它做范围扫。
- **示例**（flow_template，`value_columns=["target","extra"]`）：
  ```
  (sp1, shard, 0, "2026-01-01T00:00:00", [100.0, 20.0])
  (sp2, shard, 1, "2026-01-01T01:00:00", [101.5, 20.4])
  ```
- **为何"每点一行 + values_json 向量"而非纯 EAV：** 全列摄入下列是动态的（任意命名/个数），固定列存不了；逐点向量保证「一行一时间点」(D5) 且 `ts` 每点只存一次。**备选 EAV**（`SeriesPoint(shard_id,row_index,ts,column_name,value)`，N×C 行、`ts` 跨列冗余）作为「需按单列查询」时的演进项，本轮不采用。

> 遵循实体设计「大数组不进元数据实体」：序列值落 `SeriesPoint`，不塞进 `Shard`。

---

## 6. 样本层设计

- **`SampleIndex` 结构基本不变**（仍是行号区间指针）。`storage_ref` **瘦身**：去掉 `dataset_id`，留 `{shard_id, target_columns, context:[s,e], horizon:[s,e]}`。
- `read_by_ref` 改走 SQL：
  ```sql
  SELECT row_index, ts, values_json FROM series_point
   WHERE shard_id = :shard AND row_index BETWEEN :start AND :end
   ORDER BY row_index
  ```
  history 查 `[context_start, context_end]`、future 查 `[horizon_start, horizon_end]`，各行从 `values_json` 按 `value_columns` 下标取目标列 → 拼出与现状逐字段一致的 `sample.v1`（`target_history/target_future/history_timestamps/future_timestamps/...`）。**对上层（run_executor / sample_forecast）契约不变。**
- **差一坑消失**：`SampleWindow` 是闭区间 `[start,end]`，SQL `BETWEEN` 也含两端——直接对上，不需现状 `_assemble` 里的 `+1`（那是 TsFile 半开切片的补偿）。
- `SampleIndex.materialized` 语义重定为 `False`（值在 SeriesPoint，不物化为样本产物）或删除该字段；`materialized_sample_uri` 删除（无文件）。`read_by_ref` 不再需要 uri 参数（按 `shard_id` 查库）。

---

## 7. 加载流程（`_execute_job` 改造）

```
manifest → get_dataset_reader(manifest.file_format).read(...)  → DatasetReadResult
        → 选 target_columns(校验 ⊆ value_columns 且恰好 1)
        → build_windows + subsample_windows
        → 建 Shard(status=ready, value_columns, target_columns, row_count, frequency, …)
        → 批量 INSERT SeriesPoint(逐行：row_index, ts(ISO), values_json)
        → SampleStore.write_samples(建 SampleIndex 指针 + checksum)
        → 回填 manifest.status=loaded / job.succeeded / validation_summary
```

- **失败 = 事务回滚**：所有 SeriesPoint / SampleIndex / Shard 写入在同一事务，`ApiError` 时回滚——**不再有 `.tsfile` 半成品要清理**（删除 `_cleanup_job_artifacts` 的文件删除逻辑）。
- 校验摘要 `validation_summary` 维持（row_count/sample_count/frequency/columns/value_columns/target_columns）。

---

## 8. 读取流程

`run_executor._execute_shard` / `sample_forecast_service` 仍调 `SampleStore.read_by_ref(...)`，签名收敛为按 `shard_id + storage_ref` 查库；返回的 `sample.v1` 字段不变 → **这两处业务逻辑零改**（吸收在 SampleStore 内）。

---

## 9. 实体字段改动表

| 实体 | 改动 |
|---|---|
| `DatasetManifest` | `file_format` 取值集加 `"tsfile"`；TsFile 下 `time_column` 忽略（仍可填） |
| `Shard` | **删** `tsfile_uri`、`dataset_id`；`storage_uri` 留空或指逻辑位置（真值在 SeriesPoint）；`value_columns`/`target_columns`/`row_count`/`frequency`/切分字段保留 |
| **`SeriesPoint`（新）** | 见 §5 |
| `SampleIndex` | `storage_ref` 瘦身（去 `dataset_id`）；`materialized` 改 `False` 或删；删 `materialized_sample_uri` |
| `DatasetLoadJob` | `reader_type` 随 `file_format`（`csv_dataset_reader` / `tsfile_dataset_reader`） |

---

## 10. 删除 / 改造清单（TsFile-当存储 退场）

- **删：** `TsFileStore.write`、`TsFileSlicer`（存储用途）、`Shard.tsfile_uri`/`dataset_id`、`config.tsfiles_dir`、per-shard `.tsfile` 落盘、`_assemble` 的 `_ms_to_iso` 与 `+1` 补偿。
- **改：** `sample_store.py` 切片源 TsFile → SQLite；`dataset_load_service._execute_job` 落 SeriesPoint。
- **`tsfile_store.py`：** 仅保留/改造为 `TsFileDatasetReader` 读**输入**所需的最小封装，或整体并入 `TsFileDatasetReader`。
- **依赖：** `tsfile`/`numpy`/`pandas` 保留（读 TsFile 输入仍需）。

---

## 11. 这一刀顺带还清的债（清理计划项）

| 清理计划项 | 本设计如何化解 |
|---|---|
| #8 ms-epoch 时区往返 | `SeriesPoint.ts` 存 ISO 原文，不经 `dt.timestamp()` 往返 |
| L4 非原子写 | DB 事务天然原子，替代「临时文件 + rename」 |
| #7 checksum 含随机 ID | `SampleIndex.checksum` 改对「行区间的 (ts,值) 内容」算，可排除随机 `sample_id` → 跨加载可比 |
| #2 frequency 字符串等值误报 | 共享 `validate_time_axis` 时改为按时长比较 |
| #5/#6 行号两份冗余 / `materialized_sample_uri` 命名漂 | 随 `storage_ref` 瘦身 + 删 uri 一并收 |

---

## 12. 不变量与边界

- **等间隔仍强制**：位置切片（`row_index`）与频率依赖它；**TsFile 输入也照样校验等间隔**——接受 TsFile ≠ 支持不规则序列（清理 #11 维持「按 (a) 现状」）。
- **单序列/单设备仍是边界**：CSV 单序列、TsFile 单设备；多序列（清理 #10）单列一轮再议。
- **延后：** 多序列/面板、不规则时间、真实推理接入、数据指纹复现。

---

## 13. 自查 / 验收标准

- [ ] CSV 与 TsFile 两路输入经各自 reader 产出**结构一致**的 `DatasetReadResult`。
- [ ] 加载后序列落在 `SeriesPoint`（每点一行、`ts` 为 ISO、`values_json` 顺序 = `value_columns`），无 `.tsfile` 产物。
- [ ] `SampleStore.read_by_ref` 经 SQL 范围查询拼出的 `sample.v1` 与现状逐字段一致（`target_history/future`、时间戳、列名）。
- [ ] 加载失败时事务回滚，库内无半成品 Shard/SeriesPoint/SampleIndex。
- [ ] 单设备约束生效：多设备 TsFile 输入被拒。
- [ ] 等间隔约束对两路输入均生效。
- [ ] 同一数据集 + 同一切分两次加载，对应样本 checksum 相等（#7 已解）。
- [ ] run_executor / sample_forecast 业务逻辑未因存储切换而改动（契约稳定）。

---

### 更新记录
- 2026-05-25：创建。把 TsFile 从「内部存储」修正为「输入格式之一」，内部存储改 SQLite 逐点行（`SeriesPoint`，每点一行 + values_json 向量）；样本维持指针、切片改 SQL 范围查询；锁定 7 项决策，给出实体改动 / 删除清单 / 连带还债项 / 不变量边界 / 自查标准。基于当前已提交的 TsFile-当存储 代码核实现状。
