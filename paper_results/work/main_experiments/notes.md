# CaFE 主实验：论文结果分析备忘

## 口径与边界

本备忘只分析四个冻结主实验：GIFT-Eval short、medium、long，以及
FEV-Bench Mini-20。服务器产物仅以只读方式访问，所有二次统计均来自各任务
`04_analysis` 的冻结 JSON 汇总和 `04_analysis_suite/task_equal_summary.json`。

协议主结果有两类，且不应合并成一个“官方总分”：

1. 官方原始实例上的 task-equal MASE；
2. 每个 capability、每个 level 独立报告的 task-equal pooled effect NRMSE。

Effect NRMSE 比较模型预测变化
`forecast(treatment) - forecast(baseline)` 与真实 treatment effect。数值越低越好；
当模型完全不响应 treatment 时，预测变化为零，对应 NRMSE 恰为 1。因此，
`<1` 表示比零响应更接近真实机制，`>1` 表示响应误差反而大于忽略 treatment。

为方便画热力图和比较排名，本分析额外给出“派生宏 NRMSE”：先对一个
capability 的 5 个 level 等权平均，再对 8 个 capability 等权平均。它不是协议
定义的单一 leaderboard 分数，尤其不能替代逐 capability × level 结果。其区间
来自 20,000 次 capability 分层、模型配对的 task bootstrap；每个 capability
内部只重采样该能力的合格任务，然后再对 8 个能力等权平均。所有显著胜负均为
未做多重比较校正的 95% bootstrap 区间。

## 实验规模与完整性

| Suite | Tasks | Horizon | Official instances | Treatments | Models | Predictions | Inference failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| GIFT-Short | 20 | 30–60 | 120,784 | 2,662,855 | 7 | 19,761,938 | 0 |
| GIFT-Medium | 11 | 480–600 | 4,077 | 70,995 | 6 | 456,972 | 0 |
| GIFT-Long | 11 | 720–900 | 2,902 | 50,090 | 6 | 322,332 | 0 |
| FEV-Mini20 | 20 | 5–288 | 23,062 | 563,570 | 7 | 4,738,349 | 0 |

四组实验合计产生 25,279,591 条模型预测，所有 inference manifest 均为
`complete`，failure count 为 0。TiRex2 因最大输出长度为 320，不参加
GIFT medium/long。派生结果通过了 62 个任务的 treatment 数量恒等式、1,040 个
模型 × capability × level suite cell，以及逐级、逐能力和宏平均之间的一致性检查。

机制分覆盖率为：GIFT-Short 2,660,523 / 2,662,855 = 99.912%，
GIFT-Medium 与 GIFT-Long 均为 100%，FEV-Mini20 为
555,816 / 563,570 = 98.624%。Short 的 2,332 个未评分 treatment 全部是
low-signal；FEV 的 7,754 个未评分 treatment 中 414 个为 low-signal，7,340 个
为 unobserved-effect status。Treatment MASE 不因 effect NRMSE 不可用而丢弃。

## 总体结果

下表每个单元格为“official MASE（排名） / 派生宏 effect NRMSE（排名）”。两者
均越低越好；宏 NRMSE 只作展示，正式报告仍应使用后续的逐能力、逐 level 表。

| Model | GIFT-Short | GIFT-Medium | GIFT-Long | FEV-Mini20 |
|---|---:|---:|---:|---:|
| Chronos-2 | 1.696 (6) / 0.533 (1) | 2.266 (1) / 0.647 (1) | 2.814 (1) / 0.578 (1) | 1.828 (7) / 0.640 (5) |
| Timer-4.0 | 1.684 (5) / 0.631 (6) | 2.370 (4) / 0.710 (3) | 3.074 (4) / 0.746 (3) | 1.596 (4) / 0.706 (7) |
| Timer-3.5 | 1.648 (4) / 0.601 (3) | 2.299 (2) / 0.821 (4) | 2.818 (2) / 0.775 (4) | 1.706 (5) / 0.624 (4) |
| TimesFM-2.5 | 1.602 (3) / 0.580 (2) | 2.308 (3) / 0.700 (2) | 2.920 (3) / 0.674 (2) | 1.434 (1) / 0.567 (2) |
| TiRex2 | 1.588 (2) / 0.605 (4) | — | — | 1.746 (6) / 0.582 (3) |
| Moirai2 | 1.822 (7) / 0.643 (7) | 3.083 (5) / 0.882 (6) | 4.138 (5) / 0.955 (6) | 1.555 (3) / 0.644 (6) |
| Toto2.0 | 1.573 (1) / 0.617 (5) | 5.012 (6) / 0.858 (5) | 12.343 (6) / 0.834 (5) | 1.553 (2) / 0.519 (1) |

### 官方准确率

- GIFT-Short 的最低 MASE 是 Toto2.0 的 1.573，95% task-bootstrap CI
  `[0.871, 2.572]`。按未校正配对区间，它显著优于 Timer-4.0、Chronos-2 和
  Moirai2，但与 TiRex2、TimesFM-2.5、Timer-3.5 的差异未达到 95% 水平。
- GIFT-Medium 与 GIFT-Long 的最低点估计均来自 Chronos-2，分别为
  2.266 `[0.968, 4.338]` 和 2.814 `[0.990, 5.658]`。两组中 Chronos-2 均只对
  Moirai2 与 Toto2.0 有显著优势；与 Timer-3.5、TimesFM-2.5、Timer-4.0 的
  pairwise CI 跨零。
- FEV-Mini20 的最低 MASE 是 TimesFM-2.5 的 1.434
  `[0.765, 2.368]`。它显著优于 Moirai2、Timer-3.5 和 Chronos-2；与 Toto2.0、
  Timer-4.0、TiRex2 的差异未达到 95% 水平。
- 所有 suite 的 task-bootstrap 区间都较宽，说明任务异质性远大于许多模型间的
  平均差距。论文不宜仅凭点估计宣称细微的全局排名优势。

### 派生宏能力概览

| Suite | Best model | Macro NRMSE [95% CI] | Mean rank over 40 cells | Cell wins / 40 | Significant cell W/L |
|---|---|---:|---:|---:|---:|
| GIFT-Short | Chronos-2 | 0.533 [0.463, 0.608] | 2.40 | 13 | 88 / 7 |
| GIFT-Medium | Chronos-2 | 0.647 [0.534, 0.780] | 2.25 | 25 | 105 / 1 |
| GIFT-Long | Chronos-2 | 0.578 [0.513, 0.642] | 1.15 | 34 | 141 / 1 |
| FEV-Mini20 | Toto2.0 | 0.519 [0.455, 0.589] | 2.03 | 19 | 112 / 5 |

这里的“显著 cell W/L”统计模型在 40 个 capability × level cell 中，对其余模型
逐一做协议自带 paired task bootstrap 后的显著胜/负次数；Short/FEV 每个模型
共有 240 次 pairwise 比较，Medium/Long 为 200 次。该计数未做 family-wise 或
FDR 校正，只适合作为强弱模式的描述。

派生宏分的配对 bootstrap 显示：

- Short 中，Chronos-2 显著优于 Timer-3.5、Timer-4.0、Moirai2；相对
  TimesFM-2.5、TiRex2、Toto2.0 的区间跨零。
- Medium 中，Chronos-2 显著优于 Timer-3.5、Moirai2、Toto2.0；相对
  TimesFM-2.5 与 Timer-4.0 的区间跨零。
- Long 中，Chronos-2 对其余五个模型的派生宏差值区间均低于零；相对第二名
  TimesFM-2.5 的差值为 -0.095，95% CI `[-0.134, -0.058]`。
- FEV 中，Toto2.0 对其余六个模型的派生宏差值区间均低于零；相对第二名
  TimesFM-2.5 的差值为 -0.048，95% CI `[-0.078, -0.020]`。

将低覆盖结构能力排除，只平均 task coverage ≥ 80% 的五个能力（trend、regime、
TVS、multi-seasonal、intermittency）时，四个 suite 的第一名不变：Short、
Medium、Long 仍为 Chronos-2，FEV 仍为 Toto2.0。这是描述性敏感性检查，不是
预注册主分析。

## 逐能力发现

下表对 5 个 level 等权平均，并给出能力第一名、NRMSE 和该能力实际覆盖的任务数。
任务数较少的结构能力必须与覆盖率表一起解释。

| Capability | GIFT-Short | GIFT-Medium | GIFT-Long | FEV-Mini20 |
|---|---|---|---|---|
| Trend | Chronos-2, 0.082 (18) | Chronos-2, 0.051 (9) | Chronos-2, 0.062 (10) | Moirai2, 0.087 (18) |
| Regime switching | TiRex2, 0.071 (20) | Chronos-2, 0.053 (11) | Chronos-2, 0.064 (11) | Moirai2, 0.103 (20) |
| Common factor | Chronos-2, 0.257 (4) | Chronos-2, 0.257 (4) | Chronos-2, 0.385 (4) | Chronos-2, 0.286 (9) |
| Covariate impulse | TimesFM-2.5, 0.495 (6) | TimesFM-2.5, 0.373 (4) | TimesFM-2.5, 0.335 (3) | Toto2.0, 0.551 (8) |
| Time-varying seasonality | Chronos-2, 0.529 (19) | Chronos-2, 0.787 (10) | Chronos-2, 0.859 (10) | Toto2.0, 0.663 (20) |
| Multi-seasonality | Toto2.0, 0.637 (20) | TimesFM-2.5, 0.961 (11) | Chronos-2, 0.914 (11) | Toto2.0, 0.809 (20) |
| Predictable intermittency | TimesFM-2.5, 0.837 (20) | Moirai2, 1.115 (11) | Chronos-2, 1.051 (11) | Toto2.0, 0.852 (18) |
| Cross-series dependence | Chronos-2, 0.989 (8) | Chronos-2, 0.992 (6) | Chronos-2, 0.893 (6) | TiRex2, 0.717 (9) |

主要模式如下：

1. **Trend 和 regime switching 已接近“可解能力”。** 所有 suite、模型平均后，
   这两项通常是最低 NRMSE。最优模型的 level-average NRMSE 约为 0.05–0.10。
   这说明基础模型普遍能将大尺度趋势和明确状态切换反映到预测变化中。
2. **组合周期、稀疏事件和跨序列传递是主要瓶颈。** Medium/Long 的
   multi-seasonal、intermittency 与 TVS 经常接近或超过 1。GIFT 的
   cross-series 平均也长期停留在零响应参考附近；即使 Long 最优的 Chronos-2
   也只有 0.893。该结果更适合表述为“模型没有稳定恢复受控跨序列响应”，而非
   笼统声称模型完全不使用多变量输入。
3. **模型优势具有明显的 benchmark 依赖性。** Chronos-2 在 GIFT-Long 的 8 个
   level-average 能力中赢得 7 个，仅 covariate impulse 由 TimesFM-2.5 领先；
   但 FEV 上 Toto2.0 领先 covariate impulse、multi-seasonal、intermittency、TVS，
   Moirai2 领先 trend 与 regime，Chronos-2 只领先 common factor。
4. **没有一个模型在所有结构上占优。** Short 的 8 项能力第一名分布在
   Chronos-2（4 项）、TimesFM-2.5（2 项）、Toto2.0（1 项）和 TiRex2（1 项）；
   FEV 的第一名分布在 Toto2.0（4 项）、Moirai2（2 项）、Chronos-2 与 TiRex2
   （各 1 项）。这比单一平均排名更能体现 CaFE 的诊断价值。

## Level 曲线的可解释模式

Level 不是统一的“幅度越大越难”。Trend、common factor、covariate impulse 与
TVS 的高 level 通常提供更强可见信号；multi-seasonal 的 5 个 level 表示在固定
总能量下加入更多独立周期；regime 与 intermittency 的高 level 表示证据更少。

- 对 trend，模型均值从 level 1 到 level 5 分别由 Short 的 0.307 降至 0.150、
  Medium 的 0.401 降至 0.172、Long 的 0.209 降至 0.129、FEV 的 0.185 降至
  0.113。Common factor、covariate impulse 和 TVS 也大体随信号增强而改善。
- TVS 在长 horizon 的弱信号端最困难：Medium 从 1.395 降至 0.874，Long 从
  1.279 降至 0.894。模型只有在调制更可见时才稳定优于零响应参考。
- Multi-seasonal 随受控周期数量从 2 增至 6 而系统变难：模型均值由
  Short 0.749→0.931、Medium 1.090→1.260、Long 0.998→1.207、
  FEV 0.966→1.169。因为五档总 RMS 固定，这一趋势主要反映分解与外推更多周期
  的结构复杂度，而不是简单幅度增加。
- Intermittency 同样随证据减少显著恶化：Short 0.586→1.164，FEV
  0.664→1.204；Medium/Long 从 level 1 起已约为 1.22，后续总体不再改善。
- Regime switching 虽随 level 增大而变难，但模型均值仍保持在约 0.08–0.29，
  显著好于其他“少证据”结构。
- GIFT cross-series 对 level 的响应很弱：Short 五档模型均值为
  1.021、1.090、1.164、1.146、1.127，Medium/Long 也约为 1。FEV 稍好，但由
  0.638 恶化到 0.834。这是最值得在主文热力图和附录曲线中强调的能力缺口。

## MASE 与 NRMSE 的互补性

主实验直接支持“整体误差小不等于识别了受控规律”这一论点：

- Short 中，Toto2.0 的 official MASE 排名第 1，但派生宏能力排名第 5；
  Chronos-2 的 MASE 排名第 6，却在派生宏能力中第 1。七模型的两种排名
  Spearman 相关仅为 0.143（双侧 `p=0.760`）。
- FEV 中，TimesFM-2.5 的 official MASE 第 1、能力第 2；Toto2.0 的 official
  MASE 第 2、能力第 1。两种排名的 Spearman 相关为 0.393（`p=0.383`）。
- Medium/Long 的相关较高，均为 0.771，但只有 6 个模型，双侧 `p=0.072`；这不
  足以把两种指标视为等价。
- 直接在 treatment MASE 与 effect NRMSE 间比较逐能力第一名，两者只在
  Short 2/8、Medium 4/8、Long 5/8、FEV 3/8 的能力上吻合。Short 中例如
  Toto2.0 的 cross-series treatment MASE 最低，但该能力 effect NRMSE 为
  1.460，明显差于零响应参考；Chronos-2 则以 0.989 获得该能力第一。

这些结果应表述为“准确率与机制响应提供互补证据”，而不是“MASE 完全无用”。
Treatment MASE 衡量最终预测是否接近处理后的真实未来；effect NRMSE 则隔离了
模型是否沿正确方向、幅度响应受控变化。两者相互关联，但在同一能力上的最佳
模型经常不同。

## 覆盖率与异常值审计

结构能力不是所有任务都适用，尤其 common factor、cross-series 与 covariate
impulse。论文表格必须同时报告 task count，不能把低覆盖结果外推到整个 suite。

| Suite | Common factor | Cross-series | Covariate impulse |
|---|---:|---:|---:|
| GIFT-Short | 4 tasks / 65 instances | 8 / 3,538 | 6 / 4,296 |
| GIFT-Medium | 4 / 11 | 6 / 42 | 4 / 165 |
| GIFT-Long | 4 / 9 | 6 / 25 | 3 / 112 |
| FEV-Mini20 | 9 / 280 | 9 / 280 | 8 / 17,495 |

其中 GIFT Medium/Long 的 common-factor 只有 11/9 个 instance，Long covariate
只有 3 个任务。它们可以作为诊断结果，但不宜用窄 bootstrap CI 作强普遍性结论；
task bootstrap 只描述任务间变异，不包含任务内 instance 不确定性。

Official MASE 还存在强任务离群值：

- GIFT-Long 的 Toto2.0 task-equal MASE 为 12.343，主要由
  `gift_bizitobs_service` 的 116.655 拉高；去掉该最大任务后的描述性均值为
  1.912。Medium 中同一模型在该任务为 39.146，去掉最大任务后为 1.599。
- GIFT-Long 的其他模型在 `gift_bizitobs_service` 上也高达 13.652–24.136，
  因此 Chronos-2 的总体 2.814 去掉该最大任务后为 1.730。
- FEV 的最大误差任务通常是 `fev__uk_covid_nation_1D-cumulative`，不同模型为
  7.731–12.011；TimesFM-2.5 去掉该最大任务后的均值由 1.434 降为 1.093。

这些 sensitivity 数值不能替代预定的 task-equal 主结果；它们解释了为什么官方
MASE 的 bootstrap CI 很宽，并提示主文应搭配 task 分布图或附录逐任务表。

现有四个主实验中的 input ablation 使用旧的 temporal-shift 设计。根据项目已确认
的新决策，这些 ablation 行不应作为最终论文结构归因证据；正式消融应引用新的
target-only 实验。主实验的 official MASE、treatment MASE 与 effect NRMSE 不受
这一替换影响。

## 可直接改写进论文的结果段落

> Across the four benchmark suites, conventional forecasting accuracy and
> treatment-response fidelity induced markedly different model orderings. On
> GIFT-Short, Toto2.0 achieved the best task-equal official MASE (1.573), while
> Chronos-2 ranked only sixth in MASE (1.696) but first in the descriptive
> capability macro (effect NRMSE 0.533). The resulting rank correlation was
> weak (Spearman's rho = 0.143). The discrepancy persisted under external
> evaluation: TimesFM-2.5 led FEV-Mini20 in official MASE (1.434), whereas
> Toto2.0 led the capability macro (0.519) and won 19 of the 40
> capability-by-level cells. These results show that low aggregate forecast
> error does not by itself establish that a model recovers the controlled
> dynamics introduced by CaFE.

> The capability profiles reveal a consistent hierarchy of difficulty. Trend
> and regime-switching responses were recovered accurately across suites, with
> the best level-averaged NRMSE typically between 0.05 and 0.10. In contrast,
> multi-seasonality, predictable intermittency, and cross-series dependence
> remained challenging. At fixed total seasonal energy, increasing the number
> of controlled periods raised the model-averaged NRMSE from 0.749 to 0.931 on
> GIFT-Short and from 0.998 to 1.207 on GIFT-Long. For GIFT cross-series
> treatments, the average NRMSE stayed close to the zero-response reference of
> one across levels, indicating that current foundation models do not reliably
> recover the injected driver-to-target response.

> Model strengths were also benchmark dependent. Chronos-2 dominated the
> long-horizon GIFT capability profile, winning 34 of 40 cells and seven of
> eight level-averaged capabilities. FEV-Mini20 instead favored Toto2.0 for
> covariate response, multi-seasonality, intermittency, and time-varying
> seasonality, while Moirai2 led trend and regime switching. This heterogeneity
> motivates reporting capability vectors rather than collapsing model behavior
> into a single accuracy leaderboard.

中文版本：

> 四个主实验表明，传统预测准确率与 treatment 机制响应会产生显著不同的模型
> 排序。GIFT-Short 上，Toto2.0 取得最低 task-equal official MASE（1.573），
> Chronos-2 的 MASE 仅列第六（1.696），但其派生能力宏 NRMSE 最低（0.533）。
> 两种排名的 Spearman 相关仅为 0.143。外部 FEV-Mini20 也出现类似分化：
> TimesFM-2.5 的 official MASE 最低（1.434），Toto2.0 则在能力宏指标上领先
> （0.519），并赢得 40 个 capability × level cell 中的 19 个。由此可见，较低的
> 聚合预测误差并不足以证明模型识别并延续了 CaFE 注入的受控动态。

> 能力剖面呈现出稳定的难度层次。Trend 与 regime switching 在各 suite 中均可
> 被较准确恢复，最优模型的五档平均 NRMSE 通常为 0.05–0.10；multi-seasonal、
> predictable intermittency 与 cross-series dependence 则仍是主要瓶颈。在总
> 季节能量固定时，随着受控周期数量增加，模型平均 NRMSE 在 GIFT-Short 上由
> 0.749 升至 0.931，在 GIFT-Long 上由 0.998 升至 1.207。GIFT 的跨序列处理在
> 各 level 上长期接近 NRMSE=1 的零响应参考，说明现有基础模型尚不能稳定恢复
> 注入的 driver-to-target 响应。

## 图表建议与现成产物

1. 主文表：使用 `official_mase_suite.csv`，报告 task-equal MASE、95% CI 和任务数。
2. 主文热力图：使用 `fig_capability_heatmaps.pdf`。每格为五档等权平均 NRMSE，
   行按派生宏排名排列；caption 必须注明它是描述性 level average。
3. 主文或附录曲线：四张 `fig_level_curves_*.pdf` 同时展示五档结构趋势和
   NRMSE=1 零响应参考。
4. 排名互补图：`fig_rank_divergence.pdf` 重点展示 Short 与 FEV 的 MASE/能力
   排名反转。
5. 附录逐任务表：`official_mase_by_task.csv` 与
   `effect_nrmse_by_task_model_capability_level.csv`，用于呈现离群值、任务覆盖和
   复现实验。

所有详细数值、配对 bootstrap、覆盖率和绘图数据均位于同目录的 `tables/`。

