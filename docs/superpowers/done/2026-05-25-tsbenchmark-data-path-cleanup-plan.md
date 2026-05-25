# TSBenchmark 数据通路 review · 清理与加固计划

> ✅ **已执行完成（2026-05-26）**。24 项中 **23 项已化解**（代码修复或文档登记），仅 #10 多序列按用户决定**本轮排除**（需单列设计）。实现见 [sqlite-pivot-and-hardening 实现计划](./2026-05-25-tsbenchmark-sqlite-pivot-and-hardening-implementation-plan.md)（已移入 `done/`）的「更新记录」。状态：代码修复 = #1 #2 #3 #4 #5 #6 #7 #8 #12 #14 #15 #16 #17 #18 #19 #20 #21 #23 #24；文档登记（已知约束/决策）= #9 #11 #13 #22（见 `docs/developer/data-model.md §10`）。后端全量 176 passed。本文件随移入 `done/`。

> **来源：** 2026-05-25 自底向上逐层 review（L1 CSV 读取 → L2 切窗 → L3 物化/指针化）过程中发现的「可优化点」。
>
> **与主计划的关系：** [`2026-05-25-tsbenchmark-overall-design-implementation-plan.md`](./2026-05-25-tsbenchmark-overall-design-implementation-plan.md) 是主重构（全列摄入 → TsFile → 指针化 → MASE）的 happy path；**本份专收 happy path 之外的潜在缺陷、命名/语义债、口径不一致与正确性边角**，独立跟踪，按优先级择机并入 TDD 流程，不与主计划重复。
>
> **性质：** 只做问题登记与修法建议，不执行开发。所有代码修复建议遵循 TDD（先写失败测试 RED → 最小实现 GREEN）。Git 操作由用户负责。
>
> **标注：** ✅ = 已对当前代码核实（附 `file:line`）；⚠️ = 需进一步确认。优先级：**P1** 会产生错误结果/崩溃/契约违背；**P2** 语义/命名/复现债，不影响当前跑通；**P3** 需设计决策或正确性边角。

---

## 优先级速览

| # | 级别 | 项 | 位置 | 一句话 |
|---|---|---|---|---|
| 1 | P1 | 混合时区崩成 `TypeError` | `csv_dataset_reader.py:54` | aware/naive datetime 比较抛非受控异常，漏出统一错误信封 |
| 2 | P1 | frequency 字符串等值误报 | `csv_dataset_reader.py:73` | `"60m"` vs `"1h"` 同时长不同串 → 误判 `csv_frequency_mismatch` |
| 3 | P2 | 上传嗅探与加载校验口径不一致 | `dataset_manifests.py:43-54` | 预览乐观（假表头/恒 string/5 行/零校验），真失败只在 load 暴露 |
| 4 | P2 | `materialized=True` 名不副实 | `sample_store.py:56` | 已指针化无物化，flag 仍 True，与决策（false）/文档冲突 |
| 5 | P2 | 切片行号存两份冗余 | `sample_store.py:65-66` + `models/sample.py:16-18` | 顶层字段与 `storage_ref` 各一份，`_assemble` 只用后者 |
| 6 | P2 | `materialized_sample_uri` 命名漂 | `sample_store.py:57` | 现指向 `.tsfile`，已非「物化样本 uri」语义 |
| 7 | P2 | checksum 含随机 ID，跨加载不可比 | `sample_store.py:69,93-94` | 同数据每次加载 checksum 不同，验不了内容同一性 |
| 8 | P2 | ms-epoch 时区往返保真 | `sample_store.py:21-23` + `tsfile_store.py:51` | naive→`timestamp()` 按本地 TZ，跨机/跨 TZ 漂（已注释认领） |
| 9 | P3 | max_samples 均匀抽稀仍可能重叠 | `dataset_load_service.py:59-65` | 抽后有效间距 < horizon → 答案段重叠、指标重复计权 |
| 10 | P3 决策 | 单序列假设 / 不支持多序列 | `csv_dataset_reader.py:52-55` | 一 CSV = 一序列，与 TsFile `tag=dataset_id` 多序列能力错配 |
| 11 | P3 决策 | 等间隔挡掉日历频率 | `csv_dataset_reader.py:115` | 月度数据进不来，连带 MASE `monthly→12` 不可达 |
| 12 | P2 | `POST /tracks` 主指标默认仍 `mse` | `tracks.py:14` | 与 wizard(`mase`)/MASE-主排名决策不一致 |
| 13 | P3 决策 | `_mase_scale` 只 m=1，未按频率推季节 m | `metric_service.py:8` | 决策曾定「按频率推 m，缺省 1」；确认简化或补全 |
| 14 | P2 | 平稳历史模型从 MASE 主榜静默消失 | `metric_service.py:32` + `run_executor.py:158` | scale==0 跳过 mase → succeeded 模型缺席主榜；部分平稳两套指标样本集不一致 |
| 15 | P2 | 榜单排序硬编码升序，无视 `MetricDefinition.direction` | `ranking_service.py:21` | 未来加 higher-is-better 指标会排反；direction 字段存而不用 |
| 16 | P1 | `adapter.forecast` 无错误捕获 → 崩 run + 卡死队列 | `run_executor.py:212` | 真实服务异常未捕获；`complete` 不被调用→队列永久卡；failed 行/error_code 设计未落地 |
| 17 | P1 | 排队 run 永不自动执行 | `run_queue.py:16` + `benchmarking_runs.py:30` | `complete` 只推指针不起线程；线程只在 create_run 起→并发第二个 run 挂到重启 |
| 18 | P2 | run 终态忽略 partial_succeeded unit | `run_executor.py:116-123` | 只数 succeeded/failed；混合→判 succeeded(掩盖)、全 partial→判 failed(夸大) |
| 19 | P2 | 取消入口-only + 排队中取消卡住 | `run_executor.py:96` + `cancel_run` | 执行中不响应取消(已知)；排队中被取消的 run 不出队→卡到重启 |
| 20 | P3 | progress 的 sample 计数恒 0 | `run_executor.py` `build_run_progress` | completed_samples/failed_samples 写死 0，无 sample 级进度 |
| 21 | P2 | `create_track_with_blocks` 缺「block 已属别的 track」守卫 | `track_service.py:50-76` | 静默把 block 改挂到新 track，违背 block 只属一条 track（镜像缺失 shard_already_assigned 同款守卫） |
| 22 | P3 | adapter 选择只看全局配置，不看 `Model.adapter_type` | `model_adapter.py:13` | 无法按模型选 adapter / 混用 stub+rest（spec §4.7 偏向绑 Model） |
| 23 | P3 | `create_model` name 不唯一 + `list_models` 按 name 去重隐藏 | `models.py:23,37` | 同名模型可建但列表静默只显第一个 |
| 24 | P3 | wizard 两步非事务 → orphan capability block | `wizard.py:19-20` | create_track 失败会留下已提交的 block（shards 已占、无 track） |

---

## P1 · 会产生错误结果 / 崩溃 / 契约违背

### [ ] 1. 混合时区比较崩成 `TypeError`（漏出错误信封） ✅

- **现象：** `_parse_time`（`csv_dataset_reader.py:104`）用 `datetime.fromisoformat` 能同时解析「带 offset」和「不带 offset」的时间。若同一文件里二者混用，`:54` 的 `parsed_time < timestamps[-1]`（以及 `:52` 的 `in seen_times`）会拿 aware 与 naive datetime 比较 → 抛 `TypeError`。
- **影响：** 该 `TypeError` 不是 `ApiError`，会绕过统一错误信封（`core/errors.py`），表现为 500 而非干净的 400 + error_code。违背「所有受控错误走统一信封」的约定。
- **建议：** load 时做时区一致性预检——要么全 aware、要么全 naive，混用则抛 `ApiError("csv_mixed_timezone", ...)`；或统一归一到 UTC 后再比较。
- **RED：** 构造一份「第一行 naive、第二行带 +08:00」的 CSV，断言抛 `ApiError(csv_mixed_timezone)` 而非 `TypeError`。

### [ ] 2. frequency 校验用字符串等值，误报 mismatch ✅

- **现象：** `:73` `if frequency is not None and frequency != inferred_frequency`。推断结果是规范串（`_infer_frequency` 产 `"1h"/"7d"/...`）。用户在 manifest 填 `"60m"`、`"1H"`、`"hourly"` 等**同时长不同写法**，会因字符串不等被判 `csv_frequency_mismatch`。
- **影响：** 合法数据被误拒。当前潜伏（manifest.frequency 加载时多为 None），一旦用户显式填频率即触发。
- **建议：** 比较「时长」而非「字符串」——把 provided 也解析成秒数（或都归一到规范串）再比；解析不了的 provided 给明确错误。
- **RED：** manifest.frequency=`"60m"`、实际 hourly，断言加载**成功**（而非 mismatch）。

---

## P2 · 语义 / 命名 / 复现债（不影响当前跑通）

### [ ] 3. 上传嗅探与加载校验是两套口径不一致的解析 ✅

- **现象：** `upload`（`dataset_manifests.py:29`）只做浅嗅探：`has_header = bool(columns)`（`:54`，不调 `_looks_like_data_row`）、`inferred_type` **恒为 `"string"`**（`:51`，从不真推断）、预览 `rows[1:6]` = **5 行**（`:43`）、且**不做**时间/target/单调/等间隔任何校验。真正严格校验只在 L1 `read`（加载期）发生。
- **影响：** 用户上传时看到「一片绿」，点加载却可能直接红；且与 spec §8.6-68（「前 **20** 行、**推断类型**、validation summary」）对不上。
- **建议：** 让 upload 复用 reader 的同一套校验/类型推断（至少把 `has_header` 用 `_looks_like_data_row` 真判、`inferred_type` 真推断数值/字符串、预览行数对齐 spec 的 20），或明确把 upload 定位为「纯预览、不保证可加载」并在响应里标注。
- **RED：** 一份「无表头（首行即数据）」CSV，断言 upload 的 `validation_summary.has_header == False`。

### [ ] 4. `materialized=True` 名不副实 ✅

- **现象：** 指针化后样本不再物化为产物（值是从 TsFile 现切的），但 `sample_store.py:56` 仍写 `materialized=True`。主计划/决策本是 `materialized=false`。
- **影响：** 代码与 `data-model.md`、与主计划的语义自相矛盾；未来若用该 flag 走「按需读 vs 物化」分支会误判。
- **建议：** 翻成 `materialized=False`；或重定义该字段含义并同步 `data-model.md`。
- **RED：** 加载后断言 `SampleIndex.materialized == False`。

### [ ] 5. 切片行号在两处冗余存储 ✅

- **现象：** `SampleIndex` 顶层有 `context_start/end`、`horizon_start/end`（`models/sample.py:16-18`），`storage_ref` 里又存一份 `context:[s,e]`、`horizon:[s,e]`（`sample_store.py:65-66`）。而 `_assemble`（`:85-86`）只读 `storage_ref` 那份。
- **影响：** 两份冗余、有不一致风险（改一处忘另一处）。
- **建议：** 二选一为单一真值源——要么顶层字段为准、`storage_ref` 不再重复，要么反之。

### [ ] 6. `materialized_sample_uri` 命名已漂 ✅

- **现象：** 该字段现指向 `.tsfile`（`sample_store.py:57`），不再是「物化样本文件」。
- **影响：** 命名误导，阅读/维护成本。
- **建议：** 重命名为 `source_tsfile_uri`/`storage_uri` 一类（与 #4/#5 一并做，避免多次改 schema）。

### [ ] 7. checksum 含随机 ID，跨加载不可比 ✅

- **现象：** `SampleIndex.checksum = sha256(canonical(record))`（`:69`），而 record 含随机 `sample_id`/`shard_id`（UUID4，`:93-94`）。
- **影响：** 同一份数据每次加载 checksum 都不同——只能验「单次加载内这条没被改坏」，**验不了「与上次是同一份数据」**。主计划 B4「checksum 保持可复现」在含随机 ID 时只在单次加载内成立。
- **建议：** 把随机 ID 排除出 checksum 输入（只对 `target_history/target_future/timestamps/列名/行区间` 等**内容**算），使「相同数据 → 相同 checksum」跨加载成立。
- **RED：** 同一 CSV + 同一 split 加载两次，断言对应样本 checksum **相等**。

### [ ] 8. ms-epoch 时区往返保真 ⚠️

- **现象：** `TsFileStore.write`（`tsfile_store.py:51`）用 `int(dt.timestamp()*1000)` 落时间——对 **naive** datetime，`dt.timestamp()` 按**服务器本地时区**解释；`_ms_to_iso`（`sample_store.py:21-23`）反算同理。代码注释已说明「同机本地时区往返为恒等」。
- **影响：** 同机一致，但**跨机/跨 TZ** 会漂；且与原始 CSV 若带 offset 的时间语义不完全等价（与 #1 同源）。当前是「已认领的已知限制」。
- **建议（择机）：** 落盘统一按 UTC（`dt.astimezone(timezone.utc)` 或显式假定 UTC）并在 manifest 记录原始时区；或在文档里把「时间戳按服务器本地 TZ 往返」列为已知约束。

---

## P3 · 正确性边角 / 需设计决策

### [ ] 9. max_samples 均匀抽稀不保证答案段仍铺满 ✅

- **现象：** `subsample_windows`（`dataset_load_service.py:59-65`）用 `np.linspace` 沿窗序均匀取。若先用小 stride 多产窗、再抽稀，抽后相邻被选窗的**有效间距 ≈ (窗数-1)/(max_samples-1)**；该间距 < `horizon` 时，被选窗的**答案段仍重叠**（L2 的「每点考一次」被破坏，指标重复计权）。
- **影响：** 指标对重叠区过度计权，且随 max_samples 变化样本集变化 → 跨配置不可比。
- **建议：** 若要「每点考一次 + 又封顶数量」，更稳的是直接调 `stride` 让答案段铺满，而非 stride=1 再抽稀；或在抽稀时按「答案段不重叠」约束选窗。至少在文档说明 max_samples 与 stride 的相互作用。

### [ ] 10. 单序列假设 / 不支持多序列（需先决策） ✅

- **现象：** L1 要求时间轴「全表严格递增 + 不重复」（`csv_dataset_reader.py:52-55`），所以**一个 CSV 只能是一条序列**：长表/面板（`time,item_id,value`，时间戳重复）直接被拒。而 TsFile 落盘用 `tag_column=["dataset_id"]`（`tsfile_store.py:63`），表模型 `tsbench.<dataset_id>.<列>` **天生支持一张表多条序列**。
- **错配：** 存储层支持多序列，摄入层只喂得出单序列；而真实 TS benchmark（M4、electricity、traffic…）几乎都是多序列面板，N 条序列需传 N 个 CSV / 建 N 个 manifest。
- **决策点（非纯优化，需拍板）：** TSBenchmark 里「一个数据集」到底是**一条序列**还是**一组序列**？答案会反推 manifest / shard / 时间轴校验 / `dataset_id` tag 的用法。**建议单列一轮设计讨论，不在本清理 plan 内直接动手。**

### [ ] 11. 等间隔校验挡掉日历型频率（需先决策） ✅

- **现象：** `_infer_frequency`（`:115`）要求所有相邻间隔严格相等（`timedelta`）。月度数据（31/28/31 天）间隔不齐 → `csv_time_not_equidistant` 直接拒；周度（固定 7×86400s）反而能过。
- **连带：** MASE 的季节 m 映射里 `monthly→12` 现实**不可达**（月度数据进不了 L1）。
- **决策点：** (a) 接受现状——MASE 的 m 映射现实只到 `hourly→24 / daily→7 / weekly→52`，文档标注 monthly/yearly 暂不支持；或 (b) 放宽 L1 为「日历等间隔」以支持月/季/年。**本轮建议按 (a)，心里有数即可。**

---

## 附：跨层接线缺口（属现有主计划范围，仅登记防遗漏） ⚠️

- L5 已把两个 adapter 改成读 `sample["horizon"]`（`stub_timer_adapter.py` / `timer_rest_adapter.py`），并新建了 `model_input.py`；但 **`run_executor.py` 是否已改为用 `build_model_input(sample)` 构造 adapter 入参尚未核实**。若未接，`_assemble` 出的 dict 不含 `horizon` 键 → 评测会 `KeyError "horizon"`。
- **归属：** 主计划 Task **B5.2**（「run_executor 在 `_execute_shard` 用 `build_model_input` 构造 adapter 入参」）已覆盖此项；本 plan 不重复，仅提示**收口时务必核实 run_executor 已接线**，并按 B5「答案泄露回归断言」加测试。

---

## 建议处理顺序

1. **随主重构顺手做（同一改 schema 的窗口）：** #4 #5 #6 #7 —— 都动 `SampleIndex`/`sample_store`，与主计划 B4 收尾合并一次改完，避免多次改 schema + 重建 dev 库。
2. **独立小修（低风险，TDD 各自一条）：** #2 #3 #1 —— 都在 reader/upload，互不耦合。
3. **文档/约束登记即可：** #8 #9 #11 —— 先在 `data-model.md`/`key-flows.md` 记为已知约束，择机再动。
4. **单列设计讨论：** #10 —— 决定「数据集 = 一条 vs 一组序列」后再排实现。

---

## 追加（L6 MASE review，2026-05-25）

### [ ] 12. `POST /tracks` 主指标默认仍是 `mse`（与 MASE-主排名决策不一致） ✅  [P2]

- **现象：** `wizard.py:14` 默认 `primary_metric_id="mase"`，但直连的 `tracks.py:14` 仍 `"mse"`。
- **影响：** 走 `POST /tracks`（非向导）建的赛道主指标=mse、榜单默认=mse，违背已锁定的「MASE 主排名」决策；两个建赛道入口行为不一致。
- **建议：** `tracks.py:14` 默认改 `"mase"`，与 wizard 对齐。
- **RED：** `POST /tracks` 不传 `primary_metric_id`，断言建出的 `Track.primary_metric_id == "mase"`。

### [ ] 13. `_mase_scale` 只实现 m=1，未做「按频率推季节 m」 ✅  [P3 决策]

- **现象：** `metric_service.py:8` `_mase_scale` 写死 last-value naive（m=1）。
- **背景：** 决策曾定「按频率推 m（hourly→24…），缺省 m=1」；但 L1 等间隔已挡掉月度（#11），季节 m 大半不可达，m=1 是务实简化。
- **决策点：** 确认 m=1-only 为最终（文档登记），还是补「按频率推 m」。强季节序列下 last-value 基线偏弱 → MASE 偏小。

### [ ] 14. 平稳历史样本会让「成功」模型从 MASE 主榜静默消失 ✅  [P2]

- **现象：** `scale==0`（平稳历史）→ 该样本无 `mase`（`metric_service.py:32,53`）；若 unit 全平稳，逐层 `aggregate_metric(...,"mase")` 全 None → unit 无 mase 指标，但 unit 仍判 `succeeded`（`run_executor.py:158`，终态只看返回 dict 非空，mse/mae 撑着它非空）。
- **影响：** 一个 `succeeded` 的模型会在 MASE（主）榜缺席、却仍在 mse/mae 榜——主榜少人。另：**部分平稳**时 `mase` 对非平稳子集求均、`mse/mae` 对全部求均（`aggregate_metric` 把缺 mase 的样本计 failure 跳过）→ 同 shard 两套指标样本集不一致。
- **建议：** 明确规则——平稳历史的 mase 记为该样本失败并在报告/榜单上可见（而非静默缺席），或对平稳序列定义退化基线；至少文档登记此行为。

---

## 追加（L7 榜单 review，2026-05-25）

### [ ] 15. 榜单排序硬编码升序，无视 `MetricDefinition.direction` ✅  [P2]

- **现象：** `ranking_service.py:21` `sorted(..., key=lambda item: item["value"])` 永远升序；`_valid_unit_metric_rows` 也未取 `direction`。`MetricDefinition.direction`（存了 `lower_is_better`）形同虚设。
- **影响：** MVP 三指标（mase/mse/mae）都越小越好，当前无误；但**一旦加 higher-is-better 指标（R²、skill score、accuracy）就会排反**。
- **建议：** 排序时按 `MetricDefinition.direction` 决定升/降序（lower→asc、higher→desc）；或在 metric 注册处约束。
- **RED：** 注册一个 `direction=higher_is_better` 的指标，断言其 `RankingEntry` 按 value 降序、rank=1 是最大值。

---

## 追加（L9 编排 review，2026-05-25）

> 说明：「重启把 queued/running/cancel_requested 全判 failed」（`recover_interrupted_runs`）是 spec §8.5-57 的**有意设计**（内存执行态重启即丢，保守判失败），不作缺陷列入——但它恰是 #17/#19 卡住 run 的最终归宿。

### [ ] 16. `adapter.forecast` 无错误捕获 → 崩 run + 卡死队列（接真实推理前必补） ✅  [P1]

- **现象：** `_execute_shard`（`run_executor.py:212`）`forecast = adapter.forecast(...)` 无 try/except。桩不抛异常；真实 REST（`TimerServiceError`/超时）会让异常一路冒到 `_execute_in_background`，**其后的 `queue.complete(run_id)` 不被调用**。
- **影响：** ① 队列 `running_run_id` 永久卡死，之后所有 run 排队不跑；② 该 run 状态停 `running`（终态未走到）；③ forecast.v1 预留的 `status=failed`/`error_code` 失败行**没人写**，与 spec §8.5-52 不符。
- **建议：** 包 try/except——单样本失败记 forecast 失败行（`status=failed`+`error_code`）、不崩整 run；`_execute_in_background` 用 try/finally 确保 `queue.complete` 必被调用。
- **RED：** 注入一个 forecast 抛异常的 adapter，断言 run 落终态（非卡 `running`）、该样本 forecast 行 `status=="failed"`、且后续提交的 run 能执行。

### [ ] 17. 排队的 run 永远不会自动执行 ✅  [P1]

- **现象：** 线程只在 `create_run` 中 `submit=="running"` 时起（`benchmarking_runs.py:30`）；`queue.complete`（`run_queue.py:16`）只推进 `running_run_id` 指针、不起线程，返回值还被忽略。
- **影响：** Run A 在跑、B 提交→queued；A 完成后 B 被设为 running_run_id 却**无人起线程** → B 永远 `queued` 到重启。并发提交第二个 run 即挂。
- **建议：** `complete` 后驱动下一个 queued run 起线程（或在 `_execute_in_background` 末尾按 `complete` 返回值起下一个）。
- **RED：** 连提两个 run，断言第二个最终也跑到终态。

### [ ] 18. run 终态判定忽略 partial_succeeded 的 unit ✅  [P2]

- **现象：** `:116-123` 只统计 `unit.status=="succeeded"` 与 `=="failed"`（后者只来自 `stub://fail`）。
- **影响：** `[1 succeeded + 1 partial]`→ run 判 `succeeded`（掩盖 partial）；`[全 partial]`→ run 判 `failed`（夸大）。与 spec「至少一成功一失败→partial」意图有差。
- **建议：** 终态把 partial_succeeded unit 纳入考量（有成功也有非成功→partial；全非成功→failed）。

### [ ] 19. 取消入口-only + 排队中取消会卡住 ✅  [P2]

- **现象：** `execute_run` 只在入口查 `cancel_requested`（`:96`），执行中不轮询（已知 MVP 边界）；`cancel_run` 不把排队中的 run 移出队列。
- **影响：** 执行中的 run 取消不掉（已知）；**排队中被取消的 run** 仍会被 `complete` 弹成 running_run_id 却没人跑 → 卡到重启。
- **建议：** 排队中取消时从队列移除（或 `complete` 跳过 `cancel_requested` 的）；执行中取消按 spec §8.5-55 在逐 sample 处轮询（较大，后议）。

### [ ] 20. progress 的 sample 计数恒 0 ✅  [P3]

- **现象：** `build_run_progress` 的 `completed_samples`/`failed_samples` 写死 0，无 sample 级进度统计。
- **建议：** 按已写 forecast 行/已算 sample 指标统计，或文档标注「进度仅到 task 粒度」。

---

## 追加（组织层 / 模型管理 review，2026-05-25）

### [ ] 21. `create_track_with_blocks` 缺「block 已属别的 track」守卫 ✅  [P2]
- **现象：** `track_service.py:50-76` 校验 block 存在后直接 `block.track_id = track.track_id`，**不查 `block.track_id` 是否已非空**。
- **影响：** 把已属 track X 的 block 传进来建 track Y → block 被静默改挂到 Y，违背「每个 CapabilityBlock 只属一条 Track」（spec §1.4）。`create_real_capability_block` 对 shard 有 `shard_already_assigned` 守卫，track 这端却没有对称守卫。
- **建议：** 建 track 前若任一 block `track_id` 已非空 → 抛 `capability_block_already_assigned`。
- **RED：** 同一 block 连建两次 track，第二次断言报 `capability_block_already_assigned`。

### [ ] 22. adapter 选择只看全局配置，不看 `Model.adapter_type` ✅  [P3 / 已知]
- **现象：** `get_model_adapter`（`model_adapter.py:13`）只按 `settings.model_adapter`（全局）选 stub/rest，`Model.adapter_type` 字段不参与。
- **影响：** 不能按模型选 adapter，也无法混用（部分 stub、部分真实）。spec §4.7 偏向把 adapter 绑在 Model 上；data-model §9.9 已记此差异。
- **建议（后议）：** 接真实推理时再决定是否按 `Model.adapter_type`/`endpoint_uri` 路由；MVP 全局够用，文档登记即可。

### [ ] 23. `create_model` name 不唯一，`list_models` 却按 name 去重 ✅  [P3]
- **现象：** `create_model`（`models.py:37`）无唯一性校验；`list_models`（`:23`）`unique.setdefault(model.name, ...)` 按 name 去重。
- **影响：** 可建同名模型，但列表静默只显最早一个 → 存储与展示不一致。
- **建议：** create 时校验 name 唯一（或允许重名则 list 不去重）。

### [ ] 24. wizard 两步非事务 → orphan capability block ✅  [P3]
- **现象：** `wizard.py:19-20` 先 `create_real_capability_block`（提交）再 `create_track_with_blocks`（提交），非同一事务。
- **影响：** 第二步失败 → 留下已提交的 block（`track_id=None`、其 shards 已被占 → 后续 `shard_already_assigned` 挡住复用）。
- **建议：** 两步包一个事务（失败回滚 block 与 shard 归属），或失败时清理。

---

### 更新记录
- 2026-05-25：创建。汇总自底向上 review（L1→L3）发现的 11 项可优化点 + 1 项跨层接线提醒，按 P1/P2/P3 分级，标注 file:line 与修法/RED 建议，给出与主实现计划不重复的处理顺序。
- 2026-05-25：追加 L6 MASE review 三项——#12 `POST /tracks` 主指标默认仍 `mse`（与决策不一致）、#13 `_mase_scale` 只 m=1（待确认简化/补全）、#14 平稳历史模型从 MASE 主榜静默消失。
- 2026-05-25：追加 L7 榜单 review 一项——#15 榜单排序硬编码升序、无视 `MetricDefinition.direction`。
- 2026-05-25：追加 L9 编排 review 五项——#16 adapter 无错误捕获(P1，崩 run+卡队列)、#17 排队 run 不自动执行(P1)、#18 终态忽略 partial unit、#19 取消入口-only/排队中取消卡住、#20 progress sample 计数恒 0。
- 2026-05-25：追加 组织层/模型管理 review 四项——#21 建 track 缺 block-已占守卫(P2)、#22 adapter 不看 Model.adapter_type、#23 create_model name 不唯一+list 去重、#24 wizard 两步非事务留 orphan block。至此 cleanup 共 24 项（+1 跨层提醒）。
- **2026-05-26：全部执行完成**。23/24 项化解（#10 排除）。代码项均以失败测试起步（TDD）；文档项（#9/#11/#13/#22）登记到 `docs/developer/data-model.md §10「已知约束与边界」`。跨层接线提醒已核实：`run_executor` 经 `build_model_input(sample)` 构造 adapter 入参（输入/答案分离）。本文件移入 `done/`。
