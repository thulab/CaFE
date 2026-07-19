# Paper v5 E2：ETT1 独立 seed-bank 测量可靠性

日期：2026-07-20

## 1. 结论

按照修正后的 E2 协议，使用 ETT1、六个推理最快的 foundation models 和两套独立
N=160 seed banks 重新分析后，能力测量本身高度可靠；完整排名低于连续分数的
可靠性，主要是相近模型之间的次序交换，而不是稳定的显著优劣发生反转。

Oracle-context 结果：

- raw cell-model MASE：Lin CCC `0.9948`，Spearman `0.9928`；
- cell 内中心化 relative log-MASE：Lin CCC `0.9834`，Spearman `0.9311`；
- model × capability profile：Lin CCC `0.9879`，Spearman `0.9416`；
- 180 个 cell-model 分数的 symmetric relative difference：
  median `2.35%`，p90 `4.21%`，maximum `7.13%`；
- 180/180 对 Bank A/B mean MASE 95% CI 均有重叠；
- 450 个 cell-model pairs 中，204 对在两个 bank 均显著，方向一致率 `100%`，
  相反显著方向为 `0`；
- 正式完整排名 mean pairwise agreement `0.8178`，`20/30` cells 达到 `0.80`，
  Top-1 agreement `0.9000`。

固定 L=504 结论一致且完整排名更稳定：

- capability profile Lin CCC `0.9854`，Spearman `0.9519`；
- 正式完整排名 mean agreement `0.9022`，`27/30` cells 达到 `0.80`；
- 两个 bank 均显著的 249 个模型对方向一致率 `100%`，相反显著方向为 `0`。

因此 E2 不应再表述为“任意 32 样本轮次均复现完整名次”。更准确的结论是：

> 对 ETT1 六模型小试验，重新生成独立 N=160 测试套件后，连续能力分数和
> capability profile 高度一致；完整模型排名也呈较高一致性，但对统计上接近的模型
> 更敏感。所有在两个 bank 均达到显著的模型优劣关系均保持同一方向。

## 2. 输入与估计量

两套 bank 的唯一实验性差异是 seed：

- Bank A：`2026071621–2026071625`；
- Bank B：`2026071921–2026071925`；
- generator：`capts-paper-v3`；
- generator SHA-256：
  `a03238149c27651b58384be5ec471ef298e1bf79077b87cd1eeef5a8ad46243f`；
- 每个 capability × intensity × bank：160 条样本；
- 每个 bank 共 4,800 条 ETT1 masters；
- H=48，context=`96,168,336,504`。

六模型为 Timer-3.5、Timer-3.0、Chronos-2、moirai2、toto2.0 和 tirex2。
Bank B 共完成 `115,200/115,200` 个 views，失败为 0；Bank A 复用正式 v5
冻结预测。

每个模型、每条样本先选择 MASE 最低的 context，再对 160 条样本求 mean MASE。
能力画像使用 cell 内中心化 relative log-MASE，并对五档 intensity 等权平均。

## 3. 为什么连续分数与排名不同

六模型每个 cell 有 15 个模型对。很多模型的 mean MASE 差距很小，有限样本波动可以
改变这些模型的严格先后顺序，所以 exact rank vector 只有 `5/30` 完全一致。

但 tie-aware 结果显示：

| Oracle model-pair 状态 | 数量 |
|---|---:|
| 两个 bank 均显著且方向一致 | 204 |
| 两个 bank 显著但方向相反 | 0 |
| 两个 bank 均统计并列 | 193 |
| 仅一个 bank 显著 | 53 |

完整排名交换主要发生在后两类，不能等价解释为模型能力反转。排名仍作为正式直观结果
报告，但必须与连续 effect 和统计并列共同解释。

## 4. 方法学修正

v3 的 round seed 与 sample index 共同哈希为每条 paired sample 的 seed。round
内部没有共享结构抽样，因此 `5 rounds × 32` 在生成层面是 160 个独立 paired
groups，而不是五种 round-level 模式。

修正后的 E2：

1. 以独立 `N=160 vs N=160` seed-bank 复现为正式单位；
2. 连续能力分数、capability profile、正式排名和 tie-aware 模型对共同报告；
3. 原五个 N=32 batch 只作有限样本敏感性分析；
4. 不再要求任意两个 N=32 batch 的完整排名都达到阈值；
5. 后续 bootstrap 以 paired group 为单位，不以 round 为 cluster。

完整协议见
`docs/superpowers/specs/2026-07-20-paper-v5-e2-seed-bank-reliability-protocol.md`。

## 5. 产物

分析脚本：

`scripts/analyze_paper_v5_e2_seed_bank_reliability.py`

运行时结果：

`runtime/paper_exp/v5/E2_dynamic_stability_B/seed_bank_reliability_ett1/`

包括：

- `summary.json`、`report.md`；
- 两种 context policy 的 cell-model 连续分数及 A/B reliability；
- capability profiles 及其 A/B 对比；
- 正式 ranking reliability；
- tie-aware model-pair contrasts。
