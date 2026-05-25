# TSBenchmark 整体设计方案讨论

**日期：** 2026-05-25（**活文档**，随讨论持续更新）

**性质：** 整体设计讨论画布。从「真实数据读取 → 按框架流程跑通」出发，逐步收敛 TSBenchmark 的整体设计。

**与旧文档的关系：** 取代已删除的窄范围设计 `2026-05-25-csv-ingestion-and-sample-generation-design.md`（`git show 04f5fad` 可查原文）。该窄 spec 的设计结论在第 5 节作为**待整体复审的输入**保留。保留中的相关文档：`2026-05-16-tsbenchmark-mvp-entity-structure-design.md`（实体结构设计，更基础）。

**输入材料：** `README.md`、`docs/developer/{key-flows,data-model}.md`、`docs/reference/tsfile-dataframe-manual.md`、现有实现 `backend/app/{services,models,schemas}/*`。

> 标注约定：✅ = 已读代码 / 已实跑验证；⚠️ = 推断或未验证。

---

## 0. 本文目标与边界

- **目标**：把"真实数据读取并按框架流程跑通"所需的整体方案讨论清楚——既覆盖数据侧（摄入→落盘→样本），也为评测/输出侧（forecast 存储、聚合、榜单）的整体协调留位。
- **当前不做**：任何代码改动。本轮是设计讨论，先把方案与步骤讨论收敛，再开工。

---

## 1. 现状：已读代码确认的事实

✅ **管线本来就能端到端跑通**，走旧路径：CSV → `CsvDatasetReader`（强制单目标）→ `build_windows` 滑窗 → `SampleStore` 把每窗**物化成 JSONL**（含 `target_future`）→ `run_executor` 逐样本 `adapter.forecast` → MSE/MAE → 榜单。e2e 测试 `tests/e2e/test_mvp_benchmarking_flow.py` 覆盖。

✅ **答案泄露面比想象小**：两个适配器都读 `sample["target_future"]`，**但只取长度**当 horizon（`timer_rest_adapter.py:61` `output_length=[len(target_future)]`、`stub_timer_adapter.py:15` `horizon=len(target_future)`），**没用它的值**。→「输入/答案分离」落地干净：两处换成 `model_input["horizon"]` 即可，不破坏推理。

✅ **后端目前纯 stdlib**：`backend/pyproject.toml` 只有 fastapi/httpx/sqlmodel 等，**无 pandas/numpy/tsfile**。引入 TsFile = 同时引入整套数值栈。

✅ **真实样本 CSV 已就位**：`test/flow_template.csv` 为 `time,target,extra` 多列、30 行——正好演示"多列摄入、`extra` 被丢弃"。

✅ **桩服务不读写 TsFile**（开发者手册 §3）：传 `tsfile` 时 evaluate 返回占位、govern 直接报错。→ 即便上 TsFile，**后端自己**切片送 inline 数据给 `/forecast`，REST 链路仍可用。

---

## 2. 关键验证：TsFile 落盘地基（spike 已 PASS ✅）

整个"TsFile 当单一真值源"的可行性曾是最大未知（手册通篇只讲**读**，`TsFileDataFrame` 是只读视图）。2026-05-25 实跑验证（`/tmp/tsfile_spike.py`），用 `test/flow_template.csv` 跑 CSV → TsFile(表模型) → 读回切片，结果：

```
[read-back] list_timeseries: ['tsbench.ds1.target', 'tsbench.ds1.extra']
[read-back] type(ts[5:10]) = numpy.ndarray  value = [106. 107.4 108.8 110.1 111.]
[expected ] csv target[5:10]              = [106. 107.4 108.8 110.1 111.]
RESULT: PASS ✅ 切片值与 CSV 一致
```

✅ **已验证事实：**

- `tsfile==2.3.0` 在 Python 3.14 可装可用（连带 numpy 2.4 / pandas 3.0 / pyarrow）。目前只 `uv pip install` 进 venv，**未写入 `pyproject.toml`**（属重构 track）。
- **写得通**：`tsfile.dataframe_to_tsfile(df, path, table_name="tsbench", time_column="time", tag_column=["dataset_id"])` 直接落出表模型 `tsbench.<dataset_id>.<列>`，ms epoch 时间戳正确往返。比手搓 `TsFileTableWriter`+`Tablet` 省事，很可能就是生产写法。
- **读得回**：`TsFileDataFrame(path)["tsbench.<id>.<列>"][a:b]` → `np.ndarray`，值与源 CSV 逐位一致，正是"按行号现切窗口"所需。

→ "落盘格式 = TsFile" 的决策**站得住**，不必回退 parquet/npy。

---

## 3. 已定路线决策（本次讨论）

| 议题 | 决策 | 说明 |
|---|---|---|
| 跑通路线 | **先基线后重构** | 先用现状 JSONL 路径把真实 CSV 端到端跑通拿可信基线，再分阶段上重构 |
| TsFile 写入 de-risk | **已做，PASS** | 见第 2 节 |
| 推理服务 | **只有本地桩** | 基线的"真跑通"打桩推理（确定性），不是真模型；真模型接入待定 |

---

## 4. 步骤计划（两条 track）

### Track A · 基线跑通（现状 JSONL 路径 + 桩推理）— 先做

> 管线已存在，本 track 几乎不写新代码，重点是**用真实 CSV 实跑并核对**。诚实边界：推理是桩；`extra` 列在现状 reader 下被忽略（单目标）。

| 步 | 动作 | 验收 |
|---|---|---|
| A1 | 起后端 + 桩（`stub-service.sh start` + `start-system.sh`，或 `TSBENCHMARK_MODEL_ADAPTER=stub`） | `status-system.sh` 全绿 |
| A2 | `test/flow_template.csv` 走完整 API 链：upload → manifest(`target_columns=["target"]`) → load-job → wizard 建赛道+榜单 → benchmarking-run → progress | run 终态 `succeeded` |
| A3 | 核对榜单 + sample forecast 视图 | `/tracks/{id}/ranking` 有排名；`/samples/{id}/forecast` 取到真值+预测 |

→ **产出**：一条可重跑的端到端脚本，作为重构前后的对照基线。

### Track B · spec 重构（地基已 de-risk，按依赖分层）

```
Layer 1  依赖&DB：tsfile/pandas/numpy 入 pyproject(uv add) + 删库重建
Layer 2  摄入：CsvDatasetReader 全列数值校验 + manifest target_columns→value_columns
Layer 3  选择期：load-job 加 target_columns(校验恰好1) + max_samples + 均匀采样
Layer 4  落盘&切片器：dataframe_to_tsfile 写 per-dataset TsFile；SampleStore→指针化切片器
Layer 5  输入/答案分离：ModelInput(无 target_future,带 horizon)；两适配器 len(target_future)→horizon
Layer 6  MASE：metric_service 加 MASE(context 算 naive 基线) + 榜单/primary_metric 切 mase
Layer 7  测试同步(~11 文件引用 target_columns/csv_single_target) + 用真实 CSV 复跑对照基线
```

Layer 2/3/5/6 不依赖 TsFile，可独立推进；Layer 4 才用上 `dataframe_to_tsfile`。

---

## 5. 前一版窄范围设计的结论（待整体复审）

> 来自已删除的 `csv-ingestion-and-sample-generation-design.md`（`git show 04f5fad`）。**保留为输入，整体设计中重新审视**，不默认照搬。

| # | 议题 | 旧结论 | 整体复审备注 |
|---|---|---|---|
| 摄入-1 | CSV 列 | 多列通用，全列数值校验 | |
| 摄入-2 | 目标选择 | 选择期单目标，其余丢弃；协变量预留 | |
| 1 | 评测范式 | 纯零样本，所有窗口=测试题，无训练步 | |
| 2 | 样本数量 | stride 为主 + 可选 `max_samples` | |
| 2b | 超量采样 | 沿序列均匀采样 | |
| 3 | target 存值 | 原始值，不标准化（避免双重归一化） | |
| 4 | 指标 | MASE 主排名，MAE/MSE 诊断 | 需与评测/榜单侧整体对齐 |
| 5 | 物化 | 指针方案（单一真值源） | |
| 6 | 落盘粒度 | per-dataset + 全列矩阵 | |
| 7 | 落盘格式 | TsFile | ✅ 已 de-risk（第 2 节） |
| 8 | 序列命名 | 表模型 `tsbench.<dataset_id>.<列>` | ✅ spike 已用此命名 |
| 9 | 输入/答案 | 结构分离，模型只拿输入视图 | ✅ 泄露面已查清（第 1 节） |
| 10 | 复现 | `checksum=hash(配置)` | |

---

## 6. 待讨论议题（整体设计，驱动下一轮）

1. **评测/输出侧是否随数据侧重设计**：forecast 存储（是否也用 TsFile/`output_tsfile_path`）、逐级指标聚合、榜单刷新——旧 spec 把这块划到范围外，整体设计要不要一起拉通？
2. **实体模型整体调整**：`DatasetManifest` / `Shard` / `SampleIndex` 字段是否随新数据通路重排？是否需要新的"原始序列存储"实体？
3. **协变量正式实现的时机与角色模型**（`target`/`known_future_covariate`/`past_covariate`/`ignore`）。
4. **多目标 / 多变量预测**放开的边界。
5. **指标体系**：MASE 主排名与现有 mse/mae 榜单刷新（`run_executor.py:130`、`ranking_service.py`）如何整体协调；`Track.primary_metric_id` 默认切换的连锁影响。
6. **真实推理服务接入**：目前只有桩，何时 / 如何接真模型（`TimerRestAdapter` 已就绪，缺真实 endpoint）。
7. **复现与漂移**：`checksum=hash(配置)` vs 含数据指纹，整体取舍。

---

### 更新记录

- 2026-05-25：创建。取代窄范围 csv-ingestion spec；落入现状事实核查、TsFile 写入 spike 验证（PASS）、两条 track 步骤计划、路线决策，并列出整体设计待讨论议题。
- 2026-05-25：细化为执行计划 [`../plans/2026-05-25-tsbenchmark-overall-design-implementation-plan.md`](../plans/2026-05-25-tsbenchmark-overall-design-implementation-plan.md)（Track A 基线脚本化 + Track B 7 层 TDD 任务）。锁定三项决策：本轮文档先行待评审、forecast 输出维持 JSONL（§6 #1）、切 MASE 主排名（§6 #5）。
