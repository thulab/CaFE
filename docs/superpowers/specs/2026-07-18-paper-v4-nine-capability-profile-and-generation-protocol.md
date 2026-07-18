# Paper v4：逐 Dataset 九能力、四档 Lookback 协议

日期：2026-07-18

## 1. 固定实验形状

- Prediction length：`H=48`
- Lookback：`L ∈ {96, 168, 336, 504}`
- Intensity：dataset-local `q10/q30/q50/q70/q90`

四档 lookback 不是四次独立生成。每个任务先构造一条 `L=504,H=48` 母样本，再取
history 后缀形成四个 view。四个 view 指向同一段 future，并按自己的 context 重新
标准化、提取特征和执行 gate。

## 2. Dataset 是唯一独立校准单位

每个 dataset 独立完成：

```text
windowing
  -> three-way split
  -> four lookback profiles
  -> nine-capability support matrix
  -> dataset-local five-level targets
  -> inverse conditioning
  -> feature-support gate
  -> near-distance gate
  -> generation qualification
```

不生成跨 dataset profile，不汇总真实分位数，不共享 nuisance、强度 target 或 gate
阈值。一个 dataset 有多个 task view 时仍保留同一个 `dataset_id`，但每个 task view
单独 profile 和校准。

## 3. 数据集与 task view

### 3.1 单变量 task view

当前 profile suite 的 13 个 dataset 配置均可进入单变量候选：

- M4 Hourly
- Electricity Hourly
- Solar
- ETT1 / ETT2
- Jena Weather
- KDD Cup
- Loop Seattle
- SZ-Taxi
- M_DENSE
- Bitbrains Fast / RND
- BizITObs L2C

它们分别审计：

`trend`、`multi_seasonal`、`time_varying_seasonality`、`regime_switching`、
`nonlinear_persistence`、`predictable_intermittency`。

### 3.2 Structured task view

- `common_factor`：需要多目标 panel；
- `hierarchical_coherence`：需要可验证的 parent = sum(children)；
- `covariate_response`：需要预测期已知协变量。

当前结构化来源包括 Electricity/Traffic/Jena/BizITObs panel、M5 hierarchy、
GEFCom2014 Load/Solar covariates。原始资产存在不代表所有结构都存在；必须由 task
view 审计决定。

## 4. 九能力 Support Matrix

每个 dataset 固定输出九行，每行嵌套四个 lookback 的 `view_support`。允许
dataset 不支持某些能力，并如实记录：

- `missing_required_task_view`
- `variable_structure_not_supported`
- `insufficient_windows`
- `insufficient_local_target_range`
- `insufficient_local_intensity_spacing`
- `inverse_calibration_failed`
- `feature_gate_calibration_failed`
- `near_distance_calibration_failed`

只有四个 view 均具备该实验所需条件的 cell 才进入母样本 qualification。unsupported
cell 不生成样本、不进入模型汇总，也不阻断同一 dataset 的其他能力。

## 5. Dataset-local intensity

对固定 dataset/task/capability/lookback，仅在 parameter split 上计算主特征：

\[
T_k=Q(f_c(R_{param}),p_k),\quad
p=(0.10,0.30,0.50,0.70,0.90).
\]

这五档是 dataset 内部的相对强度，不具有跨 dataset 绝对可比性。五档必须有限、严格
递增且相邻间距足够；不满足时标记 unsupported，不修改真实分位点。

在固定该 dataset profile 的 nuisance 后，反求 `structure_scale` 和
`intensity_lambda[1..5]`。独立 seed bank 验证 realized mean 单调，最大相对本地
target range 的归一化误差不得超过 0.20。

## 6. 数据切分与防泄漏

每个 dataset/task/lookback 独立做三路拆分：

- generator parameter；
- gate reference；
- gate calibration。

多序列按 group 隔离；单序列按时间阻塞，并在边界 embargo 至少 `L+H`。官方 test
tail 或内部 evaluation tail 先排除。四档 view 虽共享母窗口身份，但只能在各自
lookback 的校准中使用，不能假装成相互独立样本。

## 7. 最终合格判定

一个 supported `dataset × capability × intensity` 母样本必须同时满足：

- capability-specific construction predictability contract；
- 四个 lookback view 共享完全相同的原始 future；
- 四个 view 的 dataset-local realized target 审计；
- 适用时四个 view 的 dataset-local feature-support gate；
- 四个 view 的 dataset-local near-distance gate。

qualification 只遍历 supported cells。任何 gate 或样本生成失败只影响当前 cell；
失败原因和 attempt audit 必须落盘。

## 8. 输出

输出目录：

```text
runtime/paper_exp/v4/01_nine_capability_suite/
```

核心文件：

```text
dataset_capability_support_matrix.json
dataset_capability_support_matrix.csv
profile_suite.json
generator_conditioning_artifact.json
feature_gate_artifact.json
near_distance_artifact.json
qualification.json
manifest.json
```

旧的 pooled profile、全局强度、source-to-global mapping 和跨 dataset aggregate
artifact 不再生成。

## 9. 复现命令

```bash
cd backend
uv run python ../scripts/build_paper_v4_profile_suite.py
uv run python ../scripts/build_paper_v4_nine_capability_suite.py
uv run pytest \
  tests/unit/test_paper_v4_profile_suite_script.py \
  tests/unit/test_paper_v4_nine_capability_suite_script.py
```

在正式大规模运行前，先选择一个真实 dataset 做 smoke。smoke 必须同时展示：

1. 四个 lookback profile；
2. 完整九能力 support matrix；
3. 至少一个 supported cell 的五档 target 与逆校准；
4. 至少一个真实的 unsupported reason（若该 dataset 结构确实不支持）。
