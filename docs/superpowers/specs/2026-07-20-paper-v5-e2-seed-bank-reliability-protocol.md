# Paper v5 E2：独立 seed-bank 能力测量可靠性协议

日期：2026-07-20

## 1. 修正原因

`capts-paper-v3` 没有 round-level latent structure。每条 paired sample 的 seed 为
`Hash(round_seed, profile × capability, sample_index)`，同一 round 内不同
`sample_index` 也会重新抽取离散机制参数、连续参数和噪声。round 因而只是预先划定
的 32 样本 batch。

旧 E2-A 实际检验的是“一个 N=32 小测试集能否独立复现完整模型排名”，不能解释为
生成器离散模式只在轮次之间变化。修正后的目标是检验论文最终能力估计量在重新生成
完整测试套件后能否复现。

## 2. 估计对象与实验单位

对模型 \(m\)、dataset/task \(d\)、capability \(c\)、intensity \(i\)，目标量为：

\[
\theta_{m,d,c,i}
=
\mathbb{E}_{X\sim G(d,c,i)}[MASE(m,X)].
\]

每个独立 seed bank 包含：

- 每个 `dataset × task × capability × intensity` 160 条样本；
- 160 条来自 160 个独立 paired groups；
- 每个 paired group 的五档 intensity 共享通过验收的结构和 nuisance realization；
- 原 `5 rounds × 32` 只保留为可恢复 batch 和小样本敏感性切片。

正式可靠性至少比较 Bank A 与独立 Bank B。两者必须使用完全相同的生成器、校准
artifact、模型集合、context/horizon 和统计协议，只允许 seed 不同。

## 3. 共同正式结果

E2 同时报告连续测量、能力画像和排名。不得只选择其中表现最好的一类指标。

### 3.1 连续 cell-model 分数

每个模型、每条 master 先按既定 oracle-context 规则选择最低 MASE；随后在每个
`model × dataset × task × capability × intensity × bank` 内对 160 条样本求均值。
固定 L=504 重复全部分析。

Bank A/B 报告：

- raw MASE Pearson、Spearman 和 Lin concordance correlation coefficient；
- cell 内模型中心化后的 relative log-MASE 的相同统计；
- symmetric relative difference 的 median、p90 和 maximum；
- 独立均值 95% CI overlap；
- bank difference 是否落在合并标准误的 ±1.96 范围。

Lin CCC 同时惩罚相关性不足和整体位置/尺度偏移，不能用单独的高相关替代。

### 3.2 Capability profile

在每个 cell 内定义：

\[
r_{m,d,c,i}
=
\log\theta_{m,d,c,i}
-
\frac{1}{M}\sum_{m'}\log\theta_{m',d,c,i}.
\]

随后对五档 intensity 等权平均得到 `model × dataset × capability` profile。
Bank A/B 报告整体和逐模型的 Pearson、Spearman、Lin CCC、MAE 与 RMSE。

### 3.3 正式模型排名

模型排名作为正式结果保留。每个 bank 在汇总 160 条样本后按 mean MASE 排名，
Bank A/B 逐 cell 报告：

- pairwise ordering agreement；
- Spearman ρ 与 Kendall τ-b；
- exact rank vector；
- Top-1 agreement 和 Top-3 overlap；
- agreement `≥0.80` 的 cell 数与比例。

该排名是连续能力估计量的派生结果，不再要求任意 N=32 batch 之间均达到阈值。

### 3.4 Tie-aware 模型对

同一 bank 内的模型共享 master samples。对每个 cell、每个模型对计算 160 个配对
MASE 差值的均值、标准误和双侧 95% CI：

- CI 全部小于 0：left model 显著更优；
- CI 全部大于 0：right model 显著更优；
- 其余：statistically indistinguishable。

Bank A/B 报告状态完全一致率、双侧均显著时的方向一致率、相反显著方向数、双侧
均并列数，以及仅一侧显著的模型对数。完整名次变化但两侧均为统计并列时，不解释为
实质能力反转。

## 4. Batch 敏感性

原五个 32 样本 batch 的 CV、排序 agreement 和 split-half 结果作为有限样本敏感性
分析保留，用于说明 N=32 与 N=160 的估计精度差异。它们不再被描述为生成机制的
round-level 稳定性。

不以 round 为 bootstrap cluster。能力分数不确定性以 paired group 为抽样单位；
intensity 对比必须保持同一 paired group 的五档配对关系。

## 5. 小试验与正式扩展

2026-07-20 先使用 ETT1、六个推理最快的 foundation models 和两个独立 N=160
bank 验证实现，不运行全数据集推理。

该小试验用于检查统计口径、产物完整性和异常模式，不用于根据观察结果反向选择正式
阈值。扩展到更多 dataset 或八模型前，应冻结实际容忍阈值和所需 seed-bank 数量。

## 6. 产物

分析器：

`scripts/analyze_paper_v5_e2_seed_bank_reliability.py`

每种 context policy 输出：

- `cell_model_scores_*.csv`；
- `cell_model_reliability_*.csv`；
- `capability_profiles_*.csv`；
- `capability_profile_reliability_*.csv`；
- `formal_rank_reliability_*.csv`；
- `tie_aware_model_contrasts_*.csv`；
- `summary.json` 与 `report.md`。
