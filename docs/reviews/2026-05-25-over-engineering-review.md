# 代码洁癖 Review：over-engineering 排查报告（2026-05-25）

> 视角：对代码洁癖到变态的 Staff Engineer。范围：`design` 分支工作区相对 `main` 的本次增量改动（TsFile 指针化样本存储 + MASE 指标 + 全列摄入）。
> 判断基线：**能用更少的概念、更短的代码、更直的调用路径达到同样行为，就是 over-engineering。**

## 结论

发现并**已直接简化** 5 处 over-engineering，行为不变，全量测试 136 passed、改动文件 ruff 干净。另有 3 处**存疑未改**（涉及契约/设计取舍，列在末尾交由你拍板）。

---

## 已简化（已动手）

| # | 文件 | 命中的坏味道 | 改了什么 / 为什么更简单 |
|---|---|---|---|
| 1 | `app/services/tsfile_store.py` | 冗余代码 | 删掉 `from typing import TYPE_CHECKING` 和空的 `if TYPE_CHECKING: pass`。该块什么都不做——纯死代码，少一个读者要费解的概念。 |
| 2 | `app/services/sample_store.py` | 兼容性代码 | 删掉 `SampleStore.__init__(self, _unused: Path \| None=None)`。注释写着"构造参数保留以兼容旧调用点"——正是 SpriCoder 规则禁止的兼容垫片。改为无构造器，并把 3 个仍传废参的调用点（`samples.py`、`sample_forecast_service.py`、`run_executor.py`）统一改成 `SampleStore()`。 |
| 3 | `app/services/sample_store.py` | 用不到的参数（死参） | `write_samples(...)` 的 `read_result: DatasetReadResult` 形参在函数体内**从未被引用**（数据全部经 `TsFileSlicer` 从 .tsfile 现切）。删除该位置参数 + 不再需要的 `DatasetReadResult` import，并同步 3 个调用点（`dataset_load_service.py` 及 2 个单测）。读者不再需要猜"read_result 在写样本时起什么作用"——答案是不起作用。 |
| 4 | `app/services/sample_store.py` | 一个资源两套清理惯例 | `write_samples` / `read_by_ref` 原本手写 `try/finally + slicer.close()`，而 `TsFileSlicer` 本就实现了（且被单测覆盖的）上下文管理器。改用 `with TsFileSlicer(...) as slicer:`，去掉 try/finally 样板，全仓库统一一种资源清理惯例。 |
| 5 | `app/services/timer_rest_adapter.py` | 过度防御 / 重复逻辑 | `_build_request(self, sample, model, horizon: int \| None=None)` 内部又写了一遍 `if horizon is None: horizon = sample.get("horizon") ... else len(target_future)`。但唯一调用方 `forecast` 永远传入已解析好的 `horizon`，该兜底分支不可达。改签名为必填 `horizon: int`、删除内部兜底。horizon 的解析现在只有一处（`forecast`），DRY。 |

## 判断为"克制、不动"的部分

- `app/services/metric_service.py::_mase_scale`：MASE 的朴素（m=1）尺度实现。`<2 行`、`flat history` 返回 `None` 是正确性需要（避免除零），不是过度防御。docstring 详尽与统计函数的复杂度相称。**合理。**
- `app/services/model_input.py`：`_REQUIRED_KEYS` 白名单式裁剪 `target_future`。这是**安全关键**的"防答案泄漏"白名单，集中化反而更清晰。**合理。**
- `dataset_load_service.py::subsample_windows` 用 `np.linspace`：纯 Python 等价式在 `max_samples==1` 时会除零，需特判；numpy 顺手处理了边界且 numpy 本就是传递依赖。**合理，不值得为省一个 import 引入 off-by-one 风险。**
- 各类小改动（`mase` 成为默认指标、`target_columns`→`value_columns` 改名、`Shard` 新增 `tsfile_uri`/`dataset_id`、`ranking_lists` 从 track 默认解析 metric/policy）均为最小化、行为驱动的改动。**克制。**

---

## 存疑未改（需你拍板）

1. **适配器里的 `len(sample["target_future"])` 兜底**（`stub_timer_adapter.py:15`、`timer_rest_adapter.py:31`）
   - 现状：`sample.get("horizon") if ... is not None else len(sample["target_future"])`。生产路径走 `build_model_input`——只带 `horizon`、**剥掉** `target_future`，所以 `else` 分支在生产中不可达（真走到反而会 KeyError）。它现在只为**让单测直接传完整 sample** 而活着。
   - 选项 A（推荐）：把适配器契约收紧为"必带 `horizon`"，删掉 `target_future` 兜底，相应更新 `test_stub_timer_adapter` / `test_timer_rest_adapter` 的 fixture（让它们传 `horizon` 或直接传 model_input）。契约更干净、与 `model_input` 的设计意图一致。
   - 选项 B：保持现状（双输入形状）。代价是生产代码留着一条只为测试存在的、且实际不可达的兼容分支。
   - 之所以没擅自改：这改的是适配器**对外输入契约** + 多个测试 fixture，属于设计取舍，按规则先问。

2. **`DatasetReadResult.column_matrix`（`dataset_reader.py:22`）只被测试用**
   - 生产代码直接用 `read_result.values` / `value_columns`；`column_matrix` 仅出现在 4 个单测里。属于"挂在生产类上的测试便利方法"。
   - 选项：留着（测试更易读）/ 下沉为测试辅助函数。倾向留着但标注，影响很小。

3. **`samples_dir` 可能已成残留**（`core/config.py` 的 `samples_dir` + `main.py` 的 `samples_dir.mkdir`）
   - 样本已改为 TsFile 指针存储，不再写 `samples/*.jsonl`。`samples_dir` 这条配置/建目录是否还有用途需确认；若确无人用可一并删。属于跨改动的清理，未擅动。

## 顺带发现（pre-existing，未在本次范围内修）

- `sample_forecast_service.py:3` 的 `CapabilityBlock` 为**既有**未使用 import（早于本次改动，与 over-engineering 无关）。我这次只改了该文件的 `SampleStore()` 调用行，未顺手清它以守住 review 范围；如需可一行 `ruff --fix`。

---

## 验证

- `uv run pytest -q` → **136 passed**（含 run_executor / dataset_load 全流程、API、e2e）。
- 针对性单测（sample_store、tsfile、两个适配器、model_input_no_leak、mase）→ 31 passed。
- `ruff check` 改动文件干净（仅上述 pre-existing 警告）。

---

## 下一步建议

> 对应"存疑未改 #1"，建议采用**选项 A**：把适配器输入契约收紧为"必带 `horizon`"，彻底删掉只为测试存活、且在生产中不可达的 `target_future` 兜底。

落地步骤：

1. `stub_timer_adapter.py:15` 与 `timer_rest_adapter.py:31`：把
   `sample.get("horizon") if sample.get("horizon") is not None else len(sample["target_future"])`
   收敛为直接读取 `sample["horizon"]`（与 `build_model_input` 的输出契约一致）。
2. 更新单测 fixture：`test_stub_timer_adapter.py` 的 `sample()`、`test_timer_rest_adapter.py` 的 `_sample()` 改为提供 `horizon` 键（或直接传 `build_model_input(...)` 的结果），不再依赖适配器从 `target_future` 反推。
3. 跑 `uv run pytest -q` 回归，确认生产路径（`run_executor` 已走 `build_model_input`）与测试均绿。

收益：生产代码不再保留"只为测试存在、实际不可达"的兼容分支，适配器契约单一明确（输入即 model_input）。

可选的轻量清理（非阻塞）：

- **存疑 #3 `samples_dir` 残留**：确认无人再写 `samples/*.jsonl` 后，删除 `core/config.py` 的 `samples_dir` 与 `main.py` 的 `samples_dir.mkdir`。
- **pre-existing**：`sample_forecast_service.py:3` 未使用的 `CapabilityBlock` import，一行 `ruff --fix` 即可清。
