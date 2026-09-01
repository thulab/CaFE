# CaFE short 子集十种子初步稳定性分析

生成时间：2026-08-31T07:51:06.325960+00:00

## 口径

- 10 个固定 benchmark task、10 个 augmentation seed、7 个模型、8 个能力、5 个 level。
- 先使用每批 `04_analysis_suite` 的 task-equal 指标，再对共同的 capability×level 单元等权汇总；十批共有完全一致的 392 个 suite rows。
- capability effect 以 NRMSE 越低越好；NRMSE < 1 表示优于“预测完全不响应 treatment”的零响应基线。
- input ablation degradation > 0 表示正确对齐的辅助输入相较错位输入有正贡献。

## 初步结论

1. **执行噪声很小。** official baseline MASE 在十批间的最大绝对 range 为 5.393e-03、最大 CV=0.076%；5/7 个模型完全一致。后续 seed 波动远大于这一技术噪声下限时，才解释为结构随机性影响。
2. **总体模型排序稳定。** Kendall's W=0.889，十批两两 Spearman 相关均值=0.876、最小值=0.643。
3. **seed 波动小于 task 不确定性。** 280 个模型×能力×level 单元中，seed SD / 单批 task-bootstrap SE 的中位数为 0.330，P95 为 0.749。
4. **主 effect 方向较稳定，但 attribution 更敏感。** effect 单元跨十批方向完全一致率为 83.6%；input-ablation 单元仅为 30.5%，后者不宜只报告单 seed。
5. **样本集合与可用量严格固定。** 8 个能力及总 treatment 数在十 seed 间的 CV 均为 0；因此没有 availability/composition 波动混入效果稳定性。
6. **结构随机性确实生效。** 共 47,806 个 instance×capability group；其中 65.2% 在十个 seed 上得到 10 个不同的 `structure_draw_sha256`。
7. **结构多样性仍不均匀。** trend 与 predictable intermittency 的十 seed 全异率为 100%，而 time-varying seasonality、common factor、cross-series dependence 和 covariate impulse 的候选结构重复更明显；下一轮随机性扩展应优先针对这些能力。

## 模型总体稳定性（effect NRMSE，低者更好）

| 模型 | 十批均值 | SD | CV | 平均排名 | 排名范围 | Top-1 次数 | NRMSE<1 比例 | 单元方向全一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chronos-2 | 0.5294 | 0.0197 | 3.7% | 1.00 | 1–1 | 10 | 92.2% | 82.5% |
| timesfm2.5 | 0.5632 | 0.0122 | 2.2% | 2.30 | 2–3 | 0 | 87.5% | 82.5% |
| Timer-3.5 | 0.5682 | 0.0113 | 2.0% | 2.70 | 2–3 | 0 | 90.0% | 85.0% |
| tirex2 | 0.6296 | 0.0218 | 3.5% | 4.60 | 4–7 | 0 | 82.5% | 82.5% |
| moirai2 | 0.6384 | 0.0088 | 1.4% | 5.30 | 4–6 | 0 | 77.5% | 87.5% |
| Timer-4.0 | 0.6382 | 0.0225 | 3.5% | 5.40 | 4–7 | 0 | 81.5% | 70.0% |
| toto2.0 | 0.6707 | 0.0243 | 3.6% | 6.70 | 5–7 | 0 | 75.8% | 95.0% |

## 可用量稳定性

| 能力 | available instances 均值 | SD | CV | 最小–最大 |
|---|---:|---:|---:|---:|
| trend | 4866.0 | 0.0 | 0.0% | 4866–4866 |
| multi_seasonal | 10528.0 | 0.0 | 0.0% | 10528–10528 |
| time_varying_seasonality | 3457.0 | 0.0 | 0.0% | 3457–3457 |
| regime_switching | 10528.0 | 0.0 | 0.0% | 10528–10528 |
| predictable_intermittency | 10528.0 | 0.0 | 0.0% | 10528–10528 |
| common_factor | 65.0 | 0.0 | 0.0% | 65–65 |
| cross_series_dependence | 3538.0 | 0.0 | 0.0% | 3538–3538 |
| covariate_impulse_response | 4296.0 | 0.0 | 0.0% | 4296–4296 |
| __all_treatments__ | 239030.0 | 0.0 | 0.0% | 239030–239030 |

## 结构多样性

| 能力 | instance×capability groups | 平均不同结构数（最多10） | 最小–最大 | 十 seed 全异比例 |
|---|---:|---:|---:|---:|
| trend | 4866 | 10.00 | 10–10 | 100.0% |
| multi_seasonal | 10528 | 9.36 | 5–10 | 56.9% |
| time_varying_seasonality | 3457 | 4.99 | 1–10 | 2.7% |
| regime_switching | 10528 | 9.91 | 7–10 | 91.8% |
| predictable_intermittency | 10528 | 10.00 | 10–10 | 100.0% |
| common_factor | 65 | 6.15 | 1–10 | 1.5% |
| cross_series_dependence | 3538 | 6.51 | 2–10 | 0.2% |
| covariate_impulse_response | 4296 | 5.91 | 3–8 | 0.0% |

## 波动最大的模型×能力组合

| 模型 | 能力 | 十批均值 | SD | CV | NRMSE<1 比例 | level 方向全一致率 |
|---|---|---:|---:|---:|---:|---:|
| toto2.0 | regime_switching | 0.1874 | 0.0560 | 29.9% | 100.0% | 100.0% |
| toto2.0 | trend | 0.3740 | 0.0848 | 22.7% | 100.0% | 100.0% |
| Timer-4.0 | regime_switching | 0.1385 | 0.0309 | 22.3% | 100.0% | 100.0% |
| tirex2 | regime_switching | 0.0860 | 0.0183 | 21.3% | 100.0% | 100.0% |
| tirex2 | trend | 0.3636 | 0.0632 | 17.4% | 100.0% | 100.0% |
| Timer-4.0 | covariate_impulse_response | 0.7604 | 0.1314 | 17.3% | 76.0% | 60.0% |
| toto2.0 | common_factor | 0.3047 | 0.0505 | 16.6% | 100.0% | 100.0% |
| Chronos-2 | covariate_impulse_response | 0.6378 | 0.1051 | 16.5% | 100.0% | 100.0% |
| timesfm2.5 | covariate_impulse_response | 0.4064 | 0.0556 | 13.7% | 100.0% | 100.0% |
| timesfm2.5 | trend | 0.3071 | 0.0397 | 12.9% | 100.0% | 100.0% |

## 数据波动与模型效果波动的区分

- official baseline 仅 Timer-3.5 有轻微波动（总体最大 CV 仍低于 0.1%），给出了推理技术噪声下限。
- 所有 seed 的 available-instance 数和 treatment 总量完全一致，而结构哈希大量变化；因此本实验把样本选择/可用量固定住，观测到的主要是结构选择导致的模型效果波动。
- 因 availability 的 seed 方差为 0，availability–effect Pearson 相关不可识别；CSV 中保留空值以明确这一点，而不虚构相关性。
- capability effect 的总体结论可视为稳定，但 input-ablation attribution 的方向全一致率仅约三成，应采用多 seed 分布或置信区间，而不是单次点估计。
- 论文正式结果建议同时报告：十 seed 均值±seed SD、task-bootstrap CI，以及最差 seed；本次结果是 short 子集的初步稳定性证据，不能替代全 benchmark 复核。

## 产物

- `stability_summary.json`：完整机器可读汇总。
- `model_overall_stability.csv`、`model_capability_stability.csv`：总体与能力级稳定性。
- `effect_cell_stability.csv`、`ablation_cell_stability.csv`：模型×能力×level 明细。
- `availability_stability.csv`、`structure_diversity.csv`：固定可用量与 seed 结构多样性。
- `availability_effect_correlations.csv`：由于可用量恒定，相关系数为空，作为 composition 已受控的审计记录。
