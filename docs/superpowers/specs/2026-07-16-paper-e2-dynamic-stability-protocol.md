# E2：动态稳定性与同 Dataset 真实排名一致性协议

更新日期：2026-07-18

## 1. 目的

E2 包含两个互补部分：

- **E2-A Dynamic stability**：同一
  `dataset × profile × capability × intensity` 更换生成 seed 后，模型分数和相对
  排名是否稳定；
- **E2-B Synthetic–real alignment**：在同一个 dataset 上，模型在 supported
  synthetic capabilities 上的平均相对排名，是否与独立真实 test windows 上的模型
  排名一致。

E2-A 回答动态生成的 Monte Carlo 波动；E2-B 回答合成能力测试的外部效度。二者不得
通过跨 dataset 汇总真实 profile 或共享强度标尺来人为提高一致性。

## 2. 输入与版本

正式输入：

```text
runtime/paper_exp/v4/01_nine_capability_suite/
  generator_conditioning_artifact.json
  feature_gate_artifact.json
  near_distance_artifact.json
  dataset_capability_support_matrix.json

runtime/paper_exp/v4/02_real_evaluation_suite/
  real_samples.jsonl
  dataset_support.json
  manifest.json
```

生成器必须使用 v4 dataset-local policy：

```text
policy_id = dataset-local-real-bounded-generator-feasible-v1
real_tolerance = [q05, 1.2*q95]
relative_dose_levels = 0/0.25/0.50/0.75/1.00
```

旧 global/canonical conditioning artifact 明确拒绝。运行目录固定为：

```text
runtime/paper_exp/v4/E2_dynamic_stability/
```

## 3. Dataset 与能力支持

support matrix 对每个 dataset 列出九个能力。E2 只运行 `status=supported` 且
conditioning、feature gate、near-distance 三者齐全的 cells。

以下情况写入 skipped audit，不补值、不计为失败或最差名次：

- task view / 变量结构不支持；
- 有效窗口不足；
- dataset-local 真实容忍区间与生成器响应区间没有足够宽的可行交集；
- conditioning 或 gate 校准失败；
- 某模型不支持目标数、协变量数、context 或 horizon。

所有样本和预测表必须保留 `dataset_id/profile_id`。一次 synthetic shard 只绑定一个
dataset/profile，不在样本层轮转 dataset。

## 4. E2-A 动态生成设计

对每个 eligible cell：

- intensity 1–5 全部运行；
- 五轮独立 root seed：
  `2026071621..2026071625`；
- 每轮每档默认 32 个母样本；
- 同一 `dataset/profile/capability/round/sample_index` 的五档使用配对 base seed；
- 所有模型接收完全相同的 accepted samples；
- construction、dataset-local feature-support 和同 dataset near-distance gate 全部
  执行。

样本总数由 eligible cell 数动态决定：

\[
N_{syn}=N_{eligible}\times5\times5\times32.
\]

不得在协议中硬编码“所有 dataset 都支持九能力”或旧 profile 数量。

## 5. E2-B 独立真实评测样本

真实评测窗口不能来自 profile/gate 使用的 development windows：

- GIFT-Eval 数据使用 profile 阶段已排除的 official short-term test tail；
- 只有 training history 的 TSF 数据使用 profile 阶段已排除的 final 48 internal
  validation；
- 固定 `H=48`；
- `L={96,168,336,504}` 四个 view 使用同一个 forecast origin 和 raw future；
- 历史不足或 test tail 不可用的 dataset 写 `unsupported`。

真实窗口只用于模型预测和 E2-B 分析，不反向修改 profile、五档 target、gate、模型
集合或统计阈值。

主分析使用 `L=504`，其他 lookback 作为敏感性分析。这样 synthetic 与 real 具有相同
任务形状，又避免把四个共享 future 的 view 当成四个独立真实样本。

## 6. 模型与推理

基础模型集合：

`Timer-3.5`、`Timer-3.0`、`Chronos-2`、`moirai2`、`toto2.0`、
`timesfm2.5`、`tirex2`。

`naive` 与 `seasonal_naive` 作为诊断基线。模型兼容性由运行时 catalog 快照决定；
不兼容 cell 排除并记录。runner 每次只加载一个基础模型，保存模型版本、并发配置、
成功/失败样本和完整 forecast。`--resume` 只补未成功的 sample id。

主误差指标为逐样本 seasonal MASE，MAE 为辅助指标。season length 来自对应
dataset profile。

## 7. E2-A 统计量

### 7.1 分数轮次变异

最小单元为
`model × dataset × profile × capability × intensity × round` 的样本均值。
对五个 round score 计算：

\[
CV=\frac{sd(score_1,\ldots,score_5)}
{\max(|mean(score)|,\epsilon)}.
\]

操作性标准：CV 中位数不超过 0.10，p95 不超过 0.25。

### 7.2 ICC

逐模型以 `dataset/profile/capability/intensity` cells 为 subjects、五轮为 raters，
计算 two-way absolute-agreement single-measure `ICC(A,1)`。最低值应不低于 0.90。

### 7.3 排名稳定性

在每个 `dataset/profile/capability/intensity` 内，按 MASE 从低到高排列兼容模型，
计算五轮两两组合的 Kendall τ-b。cell-level 平均 τ 的中位数应不低于 0.80，p10
不低于 0.50。

### 7.4 分层 bootstrap

对每个模型 cell 做 1000 次 bootstrap：先重采样 round，再在 round 内重采样样本。
报告 mean MASE 的 95% percentile CI 和相对宽度。中位相对宽度应不超过 0.20，p95
不超过 0.50。

### 7.5 跨轮新颖性

在同一 dataset/profile/capability/intensity 内，对十对 round 计算完整轨迹 DCR、
NNDR、float64 hash、六位小数 hash 和 `DCR≤1e-6` 近重复率。三种重复率必须为 0。

## 8. E2-B Synthetic–real 排名构造

### 8.1 Synthetic 侧

先在每个 supported
`dataset × capability × intensity` 内按模型 MASE 得到 rank。随后：

1. 对五档 intensity 等权平均，得到 `dataset × capability × model` rank；
2. 对该 dataset 的 supported capabilities 等权平均，得到
   `dataset × model` synthetic mean rank。

不能让样本更多的 capability 权重更高，也不能把 unsupported capability 补成最差
rank。

### 8.2 Real 侧

在同 dataset 的独立 `L=504,H=48` 真实 test windows 上，计算每个模型 mean MASE 并
排序，得到 `dataset × model` real rank。模型必须同时在 synthetic 与 real 两侧有
有效预测才进入该 dataset 的一致性统计。

### 8.3 一致性指标

逐 dataset 报告：

- Spearman rank correlation；
- Kendall τ-b；
- top-k overlap（主报告 `k=min(3, model_count)`）；
- pairwise ordering agreement；
- 有效 supported capability 数；
- 两侧有效模型数和真实/合成样本数。

最后以 dataset 为统计单位报告均值、median、bootstrap CI 和逐 dataset 散点。不能先
把不同 dataset 的预测行混在一起再排名。

当一个 dataset 只有一个 supported capability 时仍可报告一致性，但必须标记
`single_capability_only=true`，不能把它作为“多能力平均排名”的主要证据。主结论至少
要求两个 supported capabilities 和三个共同模型。

## 9. 失败、恢复与产物

- 推理失败逐样本落盘，不伪造分数；
- unsupported 和不兼容分别落盘；
- 任一正式分析缺少 real suite 时 fail closed；纯 E2-A smoke 可显式使用
  `--skip-real-alignment`；
- 看到模型结果后不得重定义 supported cells、五档或 real window。

核心输出：

```text
samples.jsonl
predictions/<model>.jsonl
real_predictions/<model>.jsonl
skipped_profile_capability_cells.json
round_scores.csv
score_cv.csv
rank_stability.csv
model_profile_icc.csv
bootstrap_ci.csv
cross_round_distance.csv
synthetic_real_rank_alignment.csv
synthetic_real_rank_alignment.json
summary.json
report.md
manifest.json
```

manifest 封存 runner、协议、四类 suite artifact、real samples、model catalog 与全部
输出的 SHA-256。
