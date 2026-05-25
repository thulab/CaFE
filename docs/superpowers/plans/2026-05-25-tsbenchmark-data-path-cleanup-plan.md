# TSBenchmark 数据通路 review · 清理与加固计划

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

### 更新记录
- 2026-05-25：创建。汇总自底向上 review（L1→L3）发现的 11 项可优化点 + 1 项跨层接线提醒，按 P1/P2/P3 分级，标注 file:line 与修法/RED 建议，给出与主实现计划不重复的处理顺序。
