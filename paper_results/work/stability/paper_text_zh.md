# 稳定性实验：论文结果分析草稿

> 本节只讨论 augmentation seed 稳定性实验。数值来自 10 个固定 short-horizon task、10 个 augmentation seeds、7 个模型、8 个 capability 和 5 个强度 level。以下结论均应限定为“在固定 short task panel 上对 augmentation seed 的稳定性”，不能外推成完整 20-task benchmark、medium/long horizon 或 FEV 的稳定性结论。

## 实验设置与统计口径

我们通过改变 augmentation seed、保持官方测试实例与 task panel 不变，检验 CaFE 合成机制对结构随机性的敏感程度。十个实验使用 seeds 2026082701–2026082710；每个 seed 均包含相同的 10 个 GIFT-Eval short tasks、7 个模型、8 个 capabilities 和 5 个 levels。对于模型 (m) 与 seed (s)，宏观分数定义为 40 个 capability × level 单元的等权平均：

\[
S_m^{(s)} = \frac{1}{40}\sum_{c=1}^{8}\sum_{\ell=1}^{5}
E_{m,c,\ell}^{(s)},
\]

其中每个 (E_{m,c,\ell}^{(s)}) 已先在参与该 capability 的 benchmark tasks 上做 task-equal aggregation。Capability-effect NRMSE 越低越好，NRMSE < 1 表示优于“不响应 treatment”的零响应基线。我们报告十个 seeds 的均值、样本标准差、CV、2.5%–97.5% 经验分位范围，以及基于 (t_9) 的 seed-mean 95% CI。经验分位范围在 (n=10) 时仅作范围描述，不应被解释为精确总体分位数。

为比较 seed 波动与 task 不确定性，我们将每个 model × capability × level 单元的跨-seed SD 除以该单元在单个 seed 内的平均 task-bootstrap SE。该比值小于 1 表示 augmentation-seed 波动小于固定 seed 下由 task panel 产生的抽样不确定性。模型间差距则使用同一 seed 上的配对差，报告“配对差的 seed SD / 平均模型差距”；这种口径保留了不同模型对同一随机合成批次的共同波动。

## 完整性与可比性审计

十个 experiments 均产生 392 个完全相同的 suite keys：280 个 capability-effect cells、105 个 legacy input-ablation cells 和 7 个 official-MASE cells。总计 100 个 dataset-level validation reports 全部 accepted，700 个 model × task inference statuses 全部 complete，validation 与 inference failure 均为 0，且 100 个 analysis manifests 齐全。每个 seed 均包含 10,528 个 official instances、239,030 个 treatments 与 39,495 个 input-ablation samples；各 capability 的 available-instance 数在十个 seeds 间完全一致。因此，下述波动没有混入 seed 间样本量或可用性构成变化。

“task panel 完全相同”不等于每个 capability 都覆盖全部 10 个 tasks。由机制适用性决定，trend、multi-seasonal、regime switching 和 predictable intermittency 覆盖 10/10 tasks；time-varying seasonality 覆盖 9/10，cross-series dependence 覆盖 8/10，covariate impulse response 覆盖 6/10，common factor 仅覆盖 4/10。这些 coverage 在所有模型、levels 与 seeds 上保持完全相同，因此不会造成 seed 间 composition drift，但 common-factor 等结果的跨-task 外推应更谨慎。

Official histories 不随 augmentation seed 改变，因此 official MASE 可作为执行可复现性的诊断。五个模型在十次运行中逐位一致；toto2.0 的最大范围仅为 (2.54\times 10^{-7})；Timer-3.5 存在轻微波动，但 CV 仅 0.076%、最大范围 0.00539。由此可见，没有足以解释主要 effect 波动的系统性执行异常。由于 official MASE 与 effect NRMSE 的量纲不同，这一检查仅是技术噪声诊断，不能直接从 effect 方差中相减。

## 总体稳定性与模型排序

| Model | Macro effect NRMSE（mean ± seed SD） | Seed 95% 经验范围 | Seed-mean 95% CI | Mean rank（range） | Top-1 | Cell observations with NRMSE < 1 |
|---|---:|---:|---:|---:|---:|---:|
| Chronos-2 | 0.529 ± 0.020 | [0.501, 0.555] | [0.515, 0.543] | 1.0 (1–1) | 10/10 | 92.2% |
| timesfm2.5 | 0.563 ± 0.012 | [0.549, 0.577] | [0.554, 0.572] | 2.3 (2–3) | 0/10 | 87.5% |
| Timer-3.5 | 0.568 ± 0.011 | [0.555, 0.585] | [0.560, 0.576] | 2.7 (2–3) | 0/10 | 90.0% |
| tirex2 | 0.630 ± 0.022 | [0.597, 0.657] | [0.614, 0.645] | 4.6 (4–7) | 0/10 | 82.5% |
| Timer-4.0 | 0.638 ± 0.022 | [0.604, 0.665] | [0.622, 0.654] | 5.4 (4–7) | 0/10 | 81.5% |
| moirai2 | 0.638 ± 0.009 | [0.629, 0.656] | [0.632, 0.645] | 5.3 (4–6) | 0/10 | 77.5% |
| toto2.0 | 0.671 ± 0.024 | [0.634, 0.708] | [0.653, 0.688] | 6.7 (5–7) | 0/10 | 75.8% |

所有模型的宏观 effect NRMSE 在十个 seeds 间仅有 1.4%–3.7% 的 CV。Chronos-2 在 10/10 seeds 中均排名第一；十个排名向量的 Kendall’s (W=0.889)，45 对 seed 排名的 Spearman 相关均值为 0.876（最小 0.643），Kendall τ 均值为 0.767（最小 0.429）。这些结果支持“宏观结论对 augmentation seed 稳定”，尤其支持 Chronos-2 的首位具有 seed 鲁棒性。

但不应把 (W=0.889) 解释为完整排序逐项固定。timesfm2.5 与 Timer-3.5 分别在 7/10 和 3/10 seeds 中领先对方；tirex2、Timer-4.0 与 moirai2 的内部顺序也频繁变化。宏观聚合能够稳定地区分大的模型差距，但不足以稳定解析彼此接近的模型。

## Seed 波动相对于模型间差距

| Mean 更优模型 | 相邻模型 | Mean gap | Paired seed SD | SD / gap | 更优 seed 数 | Gap 的 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Chronos-2 | timesfm2.5 | 0.0338 | 0.0133 | 0.39 | 10/10 | [0.0243, 0.0433] |
| timesfm2.5 | Timer-3.5 | 0.0050 | 0.0067 | 1.34 | 7/10 | [0.0002, 0.0098] |
| Timer-3.5 | tirex2 | 0.0614 | 0.0146 | 0.24 | 10/10 | [0.0509, 0.0718] |
| tirex2 | Timer-4.0 | 0.0086 | 0.0143 | 1.66 | 8/10 | [−0.0016, 0.0188] |
| Timer-4.0 | moirai2 | 0.0003 | 0.0210 | 79.45 | 5/10 | [−0.0148, 0.0153] |
| moirai2 | toto2.0 | 0.0323 | 0.0256 | 0.79 | 9/10 | [0.0140, 0.0506] |

Chronos-2 与第二名的平均差距约为配对 seed SD 的 2.5 倍，并在全部 seeds 中保持领先；Timer-3.5 与 tirex2 之间也表现出清晰分离。相反，timesfm2.5 与 Timer-3.5 的 gap 小于配对 seed SD，尽管小样本 (t)-interval 的下界略高于 0，逐-seed 胜负仍发生 3 次反转，因此更适合报告为同一性能组而不是宣称严格次序。tirex2、Timer-4.0 与 moirai2 的相邻 gap 均被 seed 波动覆盖，其中 Timer-4.0 与 moirai2 的均值几乎相同，严格排序没有统计或实际意义。

## 细粒度 winner 一致性

宏观 rank 稳定并不意味着每个 capability × level 的 winner 都稳定。在 40 个 capability × level cells 中，仅 12 个（30.0%）在十个 seeds 上具有相同 winner；modal winner 的平均出现率为 79.0%，最低为 50%。十二个完全一致的 cells 包括：trend 的全部五个 levels（均为 Chronos-2）、time-varying seasonality 的全部五个 levels（均为 Chronos-2），以及 covariate impulse response 的 levels 4–5（均为 timesfm2.5）。regime switching L4–L5、predictable intermittency L3、common factor L3–L4、cross-series dependence L1/L5 的 modal winner 仅出现于 5/10 seeds。

这一结果给出两个互补结论：CaFE 足以稳定区分宏观能力差距和总体 leader；但当多个模型在某个细粒度单元上的差距很小时，单 seed 的 cell winner 不宜作为确定性结论。论文主表应报告多-seed 均值或 winner frequency，避免只用一次运行标粗最佳值。

## Capability 方向一致性与不确定性来源

在 280 个 model × capability × level cells 中，206 个在 10/10 seeds 上均满足 NRMSE < 1，28 个在 10/10 seeds 上均不满足，另有 46 个跨越阈值。因此，83.6% 的 cells 在十个 seeds 上保持相同的阈值方向，其中 73.6% 是“稳定优于零响应”；全部 2,800 个 cell-seed observations 中有 83.9% 满足 NRMSE < 1。按 seed-mean 的 (t_9) CI 判断，220/280 cells 的 CI 上界低于 1。

| Capability | Mean ± seed SD | CV | NRMSE < 1 | Threshold-crossing cells | Median seed SD / task SE | Seed-mean 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Trend | 0.307 ± 0.024 | 7.9% | 100.0% | 0/35 | 0.226 | [0.289, 0.324] |
| Multi-seasonal | 0.891 ± 0.033 | 3.7% | 76.6% | 12/35 | 0.299 | [0.867, 0.915] |
| Time-varying seasonality | 0.521 ± 0.017 | 3.4% | 98.3% | 1/35 | 0.217 | [0.509, 0.534] |
| Regime switching | 0.128 ± 0.011 | 8.4% | 100.0% | 0/35 | 0.353 | [0.120, 0.136] |
| Predictable intermittency | 1.006 ± 0.039 | 3.9% | 46.3% | 8/35 | 0.623 | [0.978, 1.034] |
| Common factor | 0.311 ± 0.019 | 6.2% | 100.0% | 0/35 | 0.250 | [0.298, 0.325] |
| Cross-series dependence | 1.033 ± 0.047 | 4.6% | 60.6% | 22/35 | 0.383 | [1.000, 1.067] |
| Covariate impulse response | 0.646 ± 0.030 | 4.6% | 89.1% | 3/35 | 0.396 | [0.624, 0.667] |

跨-seed SD / task-bootstrap SE 的全体中位数为 0.330，95 分位数为 0.749；仅 9/280（3.2%）cells 的比值大于 1。也就是说，对绝大多数单元，augmentation-seed 波动明显小于 task panel 本身的不确定性。九个例外全部来自 predictable intermittency，这一 capability 的宏观均值也接近阈值 1，提示其随机事件位置或间歇结构选择对评分更敏感，应优先使用多 seed 汇总。

方向最稳健的能力是 trend、regime switching 与 common factor：三者的全部 35 个模型 × level cells 在全部 seeds 上均满足 NRMSE < 1；time-varying seasonality 也只有一个 threshold-crossing cell。主要不确定性集中在 cross-series dependence（22/35 cells 跨阈值）、multi-seasonal（12/35）与 predictable intermittency（8/35）。其中 cross-series dependence 的大量阈值反转更多反映均值靠近 1，而 predictable intermittency 同时表现出相对较大的结构 seed 波动。

需要注意，低均值时 CV 会被放大。例如 toto2.0 × regime switching 的 CV 达到 29.9%，但其均值仅 0.187 且所有方向均为 NRMSE < 1，因此它是“幅值有波动、结论方向稳定”，不应被简单列为失败或异常。

## 结构随机性审计

十个 seeds 的可用实例数完全相同，但 treatment structure hash 确实发生变化。在 47,806 个 instance × capability groups 中，按 group 数加权后有 65.2% 在十个 seeds 上得到十个不同的 structure hashes。Trend 与 predictable intermittency 的 full-ten-seed uniqueness 均为 100%，regime switching 为 91.8%，multi-seasonal 为 56.9%。其余能力的候选结构重复更明显：time-varying seasonality 2.7%、common factor 1.5%、cross-series dependence 0.2%、covariate impulse response 0%。

这一审计说明稳定性不是因为所有 seeds 生成了同一套结构；至少对 trend、regime switching 与 intermittency，结构随机化非常充分。但 full-ten-seed uniqueness 是离散结构哈希指标，较低比例既可能来自有限候选结构池，也不等价于所有连续 treatment 参数完全相同。因此论文中宜将它表述为“structure-level diversity audit”，不宜直接解释为总体样本多样性的充分统计量。

## Legacy input-ablation 结果的处理建议

本稳定性实验中的 input ablation 比较“正确对齐辅助输入”与旧版错位/shift 输入，其设计已被新的 target-only ablation 替代。旧指标 105 个 cells 中只有约 30.5% 在十个 seeds 上保持同一符号，而且多个不使用或几乎不响应辅助变量的模型的 degradation 接近数值零，导致 CV 与符号率不具备可解释性。因此：

- 不应把 legacy ablation 的单-seed attribution 放入论文主结论；
- 可以在附录把它作为促使我们改用 target-only ablation 的诊断证据；
- 正式的“模型是否利用辅助变量”结论应完全由新的 target-only experiments 给出。

## 可直接用于论文的结果段落

**稳定性。** We evaluated the sensitivity of CaFE to the augmentation seed by repeating the short-horizon experiment with ten deterministic seeds while keeping the official instances and the ten-task panel fixed. All ten runs produced the same 392 analysis cells and identical capability availability, with no validation or inference failures. Across seeds, model-level macro effect NRMSE had only 1.4%–3.7% coefficient of variation. Model rankings were strongly concordant (Kendall’s (W=0.889); mean pairwise Spearman ρ = 0.876), and Chronos-2 ranked first in all ten runs. The median ratio of across-seed SD to within-seed task-bootstrap SE was 0.330, and only 9 of 280 model–capability–level cells exceeded one; all nine belonged to predictable intermittency. These results indicate that, conditional on the fixed short-task panel, variation induced by the treatment seed is generally smaller than uncertainty due to task composition.

**细粒度不确定性。** The aggregate ordering was more stable than cell-wise winners. Only 12 of 40 capability–level cells had the same best model in all ten runs, although the modal winner appeared in 79% of seeds on average. Similarly, 83.6% of model–capability–level cells preserved their relation to the NRMSE = 1 zero-response baseline, while threshold crossings concentrated in cross-series dependence, multi-seasonality, and predictable intermittency. We therefore report seed-averaged scores and uncertainty intervals and avoid interpreting a single-seed cell winner as a deterministic ranking.

## 图注草稿

- **Figure S1 (`fig_stability_overall`)**：左图为每个模型十个 augmentation seeds 的宏观 effect NRMSE，深色圆点与误差线分别表示 seed 均值和基于 (t_9) 的 95% CI；右图为逐 seed 排名。Chronos-2 始终第一，而中游模型的局部排序会随 seed 变化。
- **Figure S2 (`fig_capability_mean_cv_heatmap`)**：模型 × capability 的十-seed 平均 effect NRMSE 与跨-seed CV。高 CV 不必然意味着阈值方向不稳，尤其在 NRMSE 均值很低时。
- **Figure S3 (`fig_capability_level_winner_consistency`)**：40 个 capability × level cells 的 modal winner 与其十-seed 出现频率。只有 12 个 cells 的 winner 在全部 seeds 上一致。
- **Figure S4 (`fig_direction_and_uncertainty`)**：左图为 NRMSE < 1 的 cell-seed observation 比例；右图为跨-seed SD / task-bootstrap SE。Predictable intermittency 是唯一出现比值大于 1 单元的 capability。
- **Figure S5 (`fig_structure_diversity`)**：每个 instance × capability group 在十个 seeds 中的平均 unique structure 数与 full-ten-seed uniqueness rate。
- **Figure S6 (`fig_legacy_ablation_direction`)**：旧 input-ablation 符号率，仅建议作为补充诊断，不用于正式多变量结论。

## 论文表述边界

可以支持的表述：

- “在固定的 10-task short panel 上，CaFE 的宏观效应分数和首名对 augmentation seed 稳定。”
- “绝大多数 cell 的 seed 波动小于 task-bootstrap 不确定性。”
- “细粒度 winner 不如总体排名稳定，因此多-seed 报告是必要的。”
- “结构随机化确实生效，但不同 capability 的离散结构多样性不同。”

不应支持的表述：

- “CaFE 在完整 20-task、medium/long 或 FEV 上已经证明 seed-invariant。”
- “七个模型的完整严格排序在任何 seed 下都不变。”
- “只要一个 cell 在某个 seed 上 NRMSE < 1，就能断言模型掌握了对应规律。”
- “低 structure-hash uniqueness 等价于 treatment 没有变化。”
- “旧 shift ablation 可以作为最终的多变量利用证据。”

## 局限

1. 本实验固定了同一组 10 个 short tasks，seed-mean CI 只反映结构 seed 变化，不同时整合 benchmark task sampling uncertainty。
2. 部分 capabilities 只在 4–9 个 tasks 上可评估；虽然 coverage 在 seeds 间严格一致，但对完整 benchmark 的外推仍受限。
3. 十个 deterministic seeds 可视为有限 Monte Carlo 重复，2.5%–97.5% 经验分位在 (n=10) 下较粗糙。
4. Winner consistency 会同时受到真实近似平局与 seed 随机性影响；低一致性不等于生成过程失效。
5. 结构哈希只度量 structure-level draw 的不同，不是 treatment 曲线或连续参数差异的完整距离。

