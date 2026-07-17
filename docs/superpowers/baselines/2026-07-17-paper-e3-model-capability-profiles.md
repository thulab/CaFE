# Paper E3：模型能力画像正式结果

日期：2026-07-17

## 结论

E3 已完成。基于 E2 中完全相同、已经通过动态稳定性检验的 probes，七个基础模型呈现出清晰且不同的能力轮廓；这种差异不是一个“总榜单”能够替代的。

- 单变量六能力的点估计 macro MASE 由 `toto2.0`（0.4555）和 `Chronos-2`（0.4582）领先，两者的边际 95% CI 重叠，不能仅凭本表宣称绝对胜负。
- `toto2.0` 在 `multi_seasonal`、`time_varying_seasonality`、`nonlinear_persistence`、`predictable_intermittency` 四项点估计第一；`Chronos-2` 在 `trend`、`regime_switching` 两项第一。
- 其余模型不是按一个固定次序整体变差，而是具有局部强弱：`Timer-3.5` 的时变季节性相对突出、`tirex2` 的趋势相对突出、`timesfm2.5` 的可预测间歇性相对较弱、`moirai2` 的状态切换相对较弱、`Timer-3.0` 的非线性持续性甚至略低于 seasonal naive。
- MASE 与预注册辅助指标 absolute-target NMAE 在九个 capability 上给出的模型次序完全一致，说明画像不是 MASE 分母选择造成的偶然排序。
- intensity 并不等于难度。更显著的多周期、时变季节、层级异质性和协变量响应通常更容易预测；更强趋势、非线性递推、稀疏脉冲和公共因子则常使误差上升或基本不变。模型间对同一强度变化的敏感度差异本身就是能力画像的一部分。
- 结构化任务存在多目标权衡：`toto2.0` 的 hierarchy forecast MASE 最低，但 `Chronos-2` 的 prediction coherence NMAE 最低。只报告预测误差会漏掉这一差异。

E3 支持“动态合成 benchmark 可以产生稳定、可解释的模型能力画像”这一主张，但仍不证明这些缺陷必然迁移到真实数据。外部效度需要后续合成—真实对应实验；公共因子与协变量能力还需要方法定义中预定的配对消融。

## 冻结输入、实现与覆盖

- runner/protocol commit：`1d469c7bc7bced720de42e980ce72ed07298e807`。
- 唯一源数据：`runtime/paper_exp/v1/E2_dynamic_stability/`；E2 manifest SHA-256 为 `5e91a4a4dadba842939754c8ad3e2efa22c8af3e247bf169c94ef0afbf27cfe0`。
- E3 在运行前复核了 E2 manifest 中全部 29 个文件的大小和 SHA-256；没有重新生成样本，也没有重新调用模型。
- canonical scale：`synthetic-v2-paper-v1-frozen-2026-07-16`，fingerprint `a76b66924562be4f`。
- 23 个 `profile × capability` cells、5 档 intensity、5 轮、每轮每格 32 条；七个基础模型形成 705 个 `model × profile × capability × intensity` 单元和 51 个兼容的 `model × capability` 画像。
- 每个最小单元含 160 条配对预测。capability 汇总对真实 conditioning profiles 等权、对五档 intensity 等权。
- 主要指标为 seasonal MASE；辅助指标为 `Σ|forecast-target| / Σ|target|`；relative skill 在每个 `profile × intensity` 内相对 seasonal naive 计算后再宏平均。
- 2,000 次配对分层 bootstrap 同时重抽 round 和 round 内 sample index，并在模型、intensity 和 seasonal-naive 间共享抽样索引。

正式不可变输出位于 `runtime/paper_exp/v1/E3_model_capability_profiles/`，约 3.8 MiB。E3 manifest SHA-256 为 `180618c0a5f25cf740b1d92b4517c597f5e1192af0c2f0acc16630f0a6d19259`；本次复核确认 manifest 所列 24 个输出文件、共 3,827,130 bytes 的大小和 SHA-256 全部一致。

## 单变量能力总览

| Rank | Model | Macro MASE [95% CI] | Macro NMAE | Skill vs SNaive | Mean capability rank | Top-1 数 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `toto2.0` | 0.4555 [0.4510, 0.4604] | 0.2487 | 48.9% | 1.33 | 4 |
| 2 | `Chronos-2` | 0.4582 [0.4534, 0.4632] | 0.2488 | 48.2% | 1.83 | 2 |
| 3 | `Timer-3.5` | 0.5001 [0.4947, 0.5054] | 0.2698 | 43.9% | 3.50 | 0 |
| 4 | `timesfm2.5` | 0.5354 [0.5300, 0.5409] | 0.3001 | 40.6% | 4.50 | 0 |
| 5 | `Timer-3.0` | 0.5554 [0.5496, 0.5614] | 0.2989 | 37.2% | 5.33 | 0 |
| 6 | `tirex2` | 0.5723 [0.5667, 0.5781] | 0.3258 | 36.7% | 5.33 | 0 |
| 7 | `moirai2` | 0.6392 [0.6317, 0.6474] | 0.3619 | 29.8% | 6.17 | 0 |

这里的 macro MASE 是预先定义的六个单变量 capability 等权汇总，而不是把所有时间点池化。`toto2.0` 与 `Chronos-2` 的差值只有 0.0027，且边际 CI 大幅重叠；论文应描述为点估计前两名和互补画像，而不是强调一个脆弱的总体冠亚军。

### 六能力名次矩阵

| Model | Trend | Multi-seasonal | TV seasonality | Regime | Nonlinear | Intermittency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `toto2.0` | 2 | 1 | 1 | 2 | 1 | 1 |
| `Chronos-2` | 1 | 2 | 3 | 1 | 2 | 2 |
| `Timer-3.5` | 5 | 4 | 2 | 3 | 4 | 3 |
| `timesfm2.5` | 4 | 3 | 5 | 5 | 3 | 7 |
| `Timer-3.0` | 7 | 5 | 4 | 4 | 7 | 5 |
| `tirex2` | 3 | 6 | 6 | 6 | 5 | 6 |
| `moirai2` | 6 | 7 | 7 | 7 | 6 | 4 |

51 个 capability 级模型结果中，只有 `Timer-3.0 / nonlinear_persistence` 的五档宏平均低于 seasonal naive，skill 为 -1.5%。下钻到 705 个 profile-intensity 单元，共 14 个为负：`Timer-3.0 / nonlinear_persistence` 10 个，`moirai2 / trend` 2 个，`moirai2 / nonlinear_persistence` 2 个；最差单元 skill 为 -21.0%。

## 分能力点估计与不确定性

| Capability | Point-estimate leader | MASE [95% CI] | Runner-up | Gap | 两条边际 CI 重叠 |
| --- | --- | ---: | --- | ---: | --- |
| `trend` | `Chronos-2` | 0.7747 [0.7596, 0.7903] | `toto2.0` | 0.0028 | 是 |
| `multi_seasonal` | `toto2.0` | 0.1203 [0.1169, 0.1243] | `Chronos-2` | 0.0006 | 是 |
| `time_varying_seasonality` | `toto2.0` | 0.1746 [0.1709, 0.1783] | `Timer-3.5` | 0.0499 | 否 |
| `regime_switching` | `Chronos-2` | 0.4396 [0.4293, 0.4501] | `toto2.0` | 0.0690 | 否 |
| `nonlinear_persistence` | `toto2.0` | 0.3831 [0.3754, 0.3918] | `Chronos-2` | 0.0205 | 否 |
| `predictable_intermittency` | `toto2.0` | 0.7687 [0.7539, 0.7838] | `Chronos-2` | 0.0164 | 是 |
| `common_factor` | `toto2.0` | 0.2974 [0.2905, 0.3048] | `Chronos-2` | 0.0634 | 否 |
| `hierarchical_coherence` | `toto2.0` | 0.5507 [0.5386, 0.5627] | `Chronos-2` | 0.0505 | 否 |
| `covariate_response` | `timesfm2.5` | 0.5196 [0.4986, 0.5399] | `Chronos-2` | 0.0324 | 是 |

“边际 CI 重叠”不是配对差异检验。该列只用于阻止过度解读点估计，不能反过来证明模型相等。论文主表应同时给出点估计、CI 和完整画像，而不是只报 rank。

MASE 与 NMAE 在九个 capability 上的 Spearman 模型排序相关均为 1.0，且每个模型的具体名次也完全一致。这是本轮最重要的指标稳健性检查。

## Intensity-response 画像

| Capability | I1→I5 MASE 相对变化范围 | 共同形态 | 通常最差档 |
| --- | ---: | --- | ---: |
| `trend` | +0.2% 至 +33.5% | 除 Chronos-2 近乎不变外，多数随趋势增强而变差 | I5 |
| `multi_seasonal` | -58.8% 至 -78.4% | 所有模型随多周期结构增强而显著改善 | I1 |
| `time_varying_seasonality` | -43.5% 至 -60.8% | 所有模型随调制结构增强而改善 | I1 |
| `regime_switching` | -20.0% 至 0.0% | Chronos/Timer/Toto 改善，moirai2 与 tirex2 基本不响应且绝对误差较高 | I1；后两者近乎平坦 |
| `nonlinear_persistence` | -1.0% 至 +6.6% | 多数轻微变差，模型次序较稳定 | I5 |
| `predictable_intermittency` | +0.2% 至 +20.5% | 多数在强脉冲档变差；toto2.0 与 Chronos-2 最稳定 | I5 |
| `common_factor` | +7.3% 至 +32.3% | 三个兼容模型均随共享因子增强而变差 | I5 |
| `hierarchical_coherence` | -19.6% 至 -22.5% | 三个兼容模型均随子节点异质性增强而改善 | I1 |
| `covariate_response` | -16.9% 至 -20.7% | 三个兼容模型均随协变量效应增强而改善 | I1 |

最强端点响应是 `moirai2 / multi_seasonal`，MASE 从 0.4243 降至 0.0916（-78.4%）。对模型缺陷更有诊断意义的两个对照是：`Chronos-2` 的 trend 仅变化 +0.2%，而 `moirai2` 变化 +33.5%；`toto2.0` 的 intermittency 仅变化 +0.2%，而 `timesfm2.5` 变化 +20.5%。这说明五档 intensity 不只是重复五种相似样本，而是能揭示模型对结构显著性的不同响应。

需要特别谨慎解释 `common_factor`：三个模型在更强共享因子下都没有改善，可能意味着未有效利用跨通道结构，也可能受该任务的误差暴露方式影响。按照方法定义，必须增加 channel-independent/channel-permutation 对照后才能归因。

## 跨 bucket 稳健性

逐 capability 的跨 bucket MASE CV 中位数为：

| Capability | Profile 数 | 模型 CV 中位数 | 最大值 |
| --- | ---: | ---: | ---: |
| `trend` | 3 | 0.012 | 0.107 |
| `multi_seasonal` | 3 | 0.680 | 0.745 |
| `time_varying_seasonality` | 3 | 0.360 | 0.653 |
| `regime_switching` | 3 | 0.194 | 0.329 |
| `nonlinear_persistence` | 3 | 0.135 | 0.163 |
| `predictable_intermittency` | 3 | 0.017 | 0.028 |
| `common_factor` | 2 | 0.212 | 0.278 |
| `hierarchical_coherence` | 1 | N/A | N/A |
| `covariate_response` | 2 | 0.210 | 0.227 |

多周期和时变季节性的绝对 MASE 明显依赖真实 conditioning profile，但这并未改变其 leader：`toto2.0` 在这两项的三个 buckets 中均第一。`Chronos-2` 也在 regime 的三个 buckets 中均第一，`toto2.0` 在 nonlinear 和 intermittency 的三个 buckets 中均第一；common factor 与 covariate 的 point-estimate leader 也分别在两个 buckets 中保持不变。trend 是例外：`toto2.0` 在 electricity/M4 两个 buckets 第一，`Chronos-2` 在 traffic 第一，与两者总体差距极小相吻合。

因此，高 CV 主要说明 bucket 的绝对难度不同，不等同于模型排名不稳；反过来，低 CV 也不代表能力强，例如 `tirex2 / regime_switching` 的 CV 只有 0.003，但总体名次仍为第 6。论文应把跨 bucket 方差作为 transport robustness 描述量，与准确度和 rank 分开报告。

## 结构化能力

| Capability | Model | Forecast MASE rank | Forecast MASE | Skill vs SNaive | 额外结构指标 |
| --- | --- | ---: | ---: | ---: | --- |
| `common_factor` | `toto2.0` | 1 | 0.2974 | 70.4% | 待 channel control |
|  | `Chronos-2` | 2 | 0.3608 | 64.1% | 待 channel control |
|  | `tirex2` | 3 | 0.3938 | 60.8% | 待 channel control |
| `hierarchical_coherence` | `toto2.0` | 1 | 0.5507 | 45.1% | coherence NMAE 0.0737（rank 2） |
|  | `Chronos-2` | 2 | 0.6011 | 40.1% | coherence NMAE 0.0546（rank 1） |
|  | `tirex2` | 3 | 0.6262 | 37.6% | coherence NMAE 0.1310（rank 3） |
| `covariate_response` | `timesfm2.5` | 1 | 0.5196 | 46.7% | 待 future-covariate ablation |
|  | `Chronos-2` | 2 | 0.5520 | 43.1% | 待 future-covariate ablation |
|  | `tirex2` | 3 | 0.5781 | 40.6% | 待 future-covariate ablation |

hierarchy 的 accuracy 与 coherence 排名不完全一致，直接证明 structured capability 必须保留多指标报告。`covariate_response` 当前只能说明“在完整 known-future 输入协议下的端到端预测表现”；在完成 drop/shuffle/event-flip 消融前，不能声称优胜模型确实使用了未来协变量。

## 论文表述与下一步

E3 适合支撑以下表述：

1. 动态生成轮次改变时，模型画像稳定（由 E2 建立）；在固定的聚合协议下，不同模型表现出可区分的 capability fingerprint（由 E3 展示）。
2. 单一总分掩盖了模型互补性。例如总体接近的 `toto2.0` 与 `Chronos-2` 分别擅长复杂季节/非线性/间歇结构和趋势/状态切换。
3. intensity 是结构剂量，不是统一难度；响应曲线比单档分数提供了额外诊断信息。
4. 辅助 NMAE、跨 bucket 结果和 hierarchy coherence 让结论不依赖单一指标或单一真实基底。

目前不应写成：合成结果已经证明真实数据上的同类缺陷；模型在 common factor 上的收益一定来自跨通道建模；完整 covariate 输入下的好成绩一定来自使用了未来协变量。

建议后续优先补齐两类证据：

- structured mechanism controls：common-factor 的 channel-independent/permutation，对 covariate 的 intact/drop/shuffle/event-flip；
- synthetic—real external validity：在 GIFT-Eval 的真实数据与基于同一数据提取分布后生成的 probes 上做模型表现/排名 concordance，并针对 E3 暴露的缺陷做真实数据 case study。

## 输出索引

- `report.md` / `summary.json`：正式摘要与机器可读结果。
- `paper_tables.md`：单变量总览、51 个 capability 画像和 hierarchy coherence 主表。
- `capability_profiles.csv`：five-level mean、AUC、最差档、skill、NMAE、intensity slope、跨 bucket 方差、bootstrap CI 与 rank。
- `intensity_curves.csv`：逐 intensity 的 profile-macro MASE/NMAE/skill 与 95% CI。
- `profile_intensity_scores.csv` / `bucket_scores.csv`：最小汇总单元和逐 bucket 结果。
- `figures/figure_1_capability_skill_heatmap.*`：跨能力 relative-skill 热图。
- `figures/figure_2_intensity_response_mase.*`：九能力 intensity-response small multiples。
- `figures/figure_3_univariate_capability_fingerprints.*`：七模型单变量能力指纹。
- `figures/figure_4_cross_bucket_variability.*`：真实 conditioning profile 敏感性。
- `figures/figure_5_univariate_model_summary.*`：单变量 macro MASE 与 skill 总览。

每张图均保留 300-DPI PNG、可编辑 SVG 和 PDF。实现的 8 个定向测试以及完整 backend suite 均通过；全量结果为 381 passed。
