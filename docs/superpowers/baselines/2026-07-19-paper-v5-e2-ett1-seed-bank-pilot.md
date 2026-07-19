# Paper v5 E2：ETT1 五轮 seed-bank 稳定性小试验

日期：2026-07-19

## 1. 目的与结论

此前 E2-A 逐轮检验要求任意一个 32 样本随机轮次都独立复现模型排名。由于
`capts-paper-v3` 的 round seed 同时改变能力机制内部的具体形态，这项检验比论文
最终使用五轮汇总结果的统计口径更严格。

本试验改为比较两套彼此独立的五轮 seed bank。对每个
`dataset × capability × intensity × model`，先按模型、逐样本选择 MASE 最低的
context，再对每个 bank 的 `5 × 32 = 160` 条样本求平均 MASE，最后据此排名。

ETT1 六模型小试验支持采用这一口径：

- Oracle-context 主结果：`20/30` cells 达到 pairwise ordering agreement
  `≥ 0.80`，mean / median agreement 为 `0.8178 / 0.8333`；
- 固定 L=504：`27/30` cells 通过，mean / median agreement 为
  `0.9022 / 0.9333`；
- Oracle-context 的 Bank A、Bank B 内部逐轮 mean agreement 仅为
  `0.7284`、`0.7338`，五轮汇总后明显提高；
- 固定 L=504 的两个 bank 内部逐轮 mean agreement 为 `0.8151`、`0.7369`，
  五轮汇总后的 bank 间 agreement 为 `0.9022`。

因此，五轮 bank 间复现性比“任意单轮都稳定”更贴合最终实验估计量，也确实显著
降低了 v3 的 seed 波动。不过 oracle-context 仍只有三分之二 cells 通过，当前结果
只能视为有希望的小规模证据，不能据此宣称全数据集稳定。

## 2. 冻结输入

Dataset：`gift_ett1_h`。

两个 bank 使用同一套逐数据集校准、同一生成器代码和完全相同的生成配置，唯一的
实验性差异是 round seed：

- Bank A：`2026071621–2026071625`；
- Bank B：`2026071921–2026071925`；
- generator version：`capts-paper-v3`；
- generator SHA-256：
  `a03238149c27651b58384be5ec471ef298e1bf79077b87cd1eeef5a8ad46243f`；
- 每个 bank：6 个能力 × 5 档 intensity × 5 轮 × 32 条 =
  4,800 条母样本；
- 每条母样本保留 L=96、168、336、504 四个共享 future 的 views，H=48。

虽然本试验运行在机制回退实验分支上，但 B bank 样本已由 v3 生成器冻结，且当前
推理 runner 与正式 v3 提交无差异；推理只读取冻结样本，没有调用当前分支的生成器。

## 3. 模型与推理

从正式全量运行的八个 foundation models 中按实测耗时选择六个最快模型：

1. Timer-3.5；
2. Timer-3.0；
3. Chronos-2；
4. moirai2；
5. toto2.0；
6. tirex2。

排除较慢的 `tabpfn-ts3` 和 `timesfm2.5`。B bank 使用三台相同推理服务并行：

- 本机 `127.0.0.1:10810`：toto2.0；
- `192.168.99.18:10810`：Timer-3.5、moirai2、tirex2；
- `192.168.99.17:10811`：Timer-3.0、Chronos-2。

每个模型均完成 `19,200/19,200` 个兼容 views，六模型共
`115,200/115,200`，失败请求为 0。Bank A 复用正式 E2 已冻结预测。

## 4. 主结果

六模型共有 15 个模型对。通过阈值为 agreement `≥ 0.80`，即最多允许 3 个模型对
改变相对次序。

### 4.1 Oracle context

| 指标 | 结果 |
|---|---:|
| 通过 cells | 20 / 30 |
| Agreement mean / median / minimum | 0.8178 / 0.8333 / 0.5333 |
| Spearman ρ mean / median | 0.7333 / 0.8000 |
| Kendall τ-b mean / median | 0.6356 / 0.6667 |
| Top-1 agreement | 0.9000 |
| Top-3 overlap | 0.8111 |
| Exact rank-vector rate | 0.1667 |

如果按用户原始提议，先对五轮 ordinal rank 求平均再比较，结果同样有 `20/30`
cells 达到 0.80，mean / median agreement 为 `0.8345 / 0.9310`。由于 MASE 在轮次
间同尺度且每轮样本数相同，主口径仍采用“先平均 MASE，再排名”，避免丢失模型分数
差距信息。

### 4.2 固定 L=504

| 指标 | 结果 |
|---|---:|
| 通过 cells | 27 / 30 |
| Agreement mean / median / minimum | 0.9022 / 0.9333 / 0.7333 |
| Spearman ρ mean / median | 0.8819 / 0.9429 |
| Kendall τ-b mean / median | 0.8044 / 0.8667 |
| Top-1 agreement | 0.9333 |
| Top-3 overlap | 0.8444 |
| Exact rank-vector rate | 0.3667 |

平均五轮 ordinal rank 的固定 L=504 结果为 `26/30` cells 通过，mean / median
agreement 为 `0.9082 / 0.9286`，与主聚合方向一致。

## 5. 分能力与强度

Oracle-context：

| Capability | 通过 | Mean agreement | Top-1 |
|---|---:|---:|---:|
| multi_seasonal | 5 / 5 | 0.9733 | 1.0 |
| nonlinear_persistence | 2 / 5 | 0.7333 | 0.8 |
| predictable_intermittency | 3 / 5 | 0.7467 | 1.0 |
| regime_switching | 2 / 5 | 0.7200 | 0.8 |
| time_varying_seasonality | 5 / 5 | 0.9467 | 1.0 |
| trend | 3 / 5 | 0.7867 | 0.8 |

按 intensity：

| Intensity | 通过 | Mean agreement | Top-1 |
|---:|---:|---:|---:|
| 1 | 2 / 6 | 0.7444 | 0.8333 |
| 2 | 3 / 6 | 0.7667 | 0.6667 |
| 3 | 3 / 6 | 0.7889 | 1.0000 |
| 4 | 6 / 6 | 0.9111 | 1.0000 |
| 5 | 6 / 6 | 0.8778 | 1.0000 |

主要失败仍集中在 nonlinear persistence、regime switching、trend 的低 intensity。
高 intensity 的 12 个 cells 全部通过，符合“主能力模式越突出，随机子模式对模型
次序的影响越小”的解释。

固定 L=504 只剩 3 个失败 cell：

- nonlinear persistence intensity 1、2；
- regime switching intensity 4。

这说明逐样本选择最佳 context 会引入额外的模型特异性选择波动，但它不是唯一原因。

## 6. 判断与边界

本试验回答的是：将五轮作为一个完整 Monte Carlo 测试套件后，重新生成另一套五轮
是否会得到近似一致的模型排名。它不再要求任意 32 样本单轮都成为独立、稳定的
benchmark。

结果支持把五轮 bank 间比较改为 E2-A 主口径，并把逐轮比较降为敏感性分析。不过：

1. 当前只有 ETT1、六个较快模型；
2. 只有 A/B 两个独立 bank，尚不能估计不同 bank 组合间的方差；
3. oracle-context 仍有 10/30 cells 未通过，尤其是低 intensity；
4. 选择最快模型改变了模型集合，不能直接外推到正式八模型排名。

合理的下一步是先对另外一至两个代表性数据集做同样的六模型小试验；若模式持续，
再为全量八模型生成和推理更多独立 bank。

## 7. 产物

分析脚本：

`scripts/analyze_paper_v5_e2_seed_bank_pilot.py`

运行时结果：

`runtime/paper_exp/v5/E2_dynamic_stability_B/seed_bank_comparison_ett1/`

核心文件：

- `summary.json`；
- `report.md`；
- `cell_rank_comparison_oracle_context.csv`；
- `cell_rank_comparison_fixed_l504.csv`；
- 两种口径的逐轮分数、bank cell-model 分数以及按 capability/intensity 汇总表。
