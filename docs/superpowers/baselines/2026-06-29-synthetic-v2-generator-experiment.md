# Synthetic v2 Generator Experiment

日期：2026-06-29

## 目的

对比旧公式、v2 pilot 公式和 M4 Hourly 真实窗口，检查显式特征、真实分布 cap 和 naive / seasonal naive 基线响应是否符合 synthetic v2 契约。

## 输入

- M4 Hourly 本地数据：`runtime/research/m4_hourly_dataset.zip`
- JSON 输出：`runtime/research/synthetic-v2-generator-experiment/summary.json`
- 每组样本数：`256`
- context/horizon/season：`168/24/24`

## 汇总

| Group | Difficulty | Trend | Seasonal | Slope | Curvature | Noise | Naive MASE | SNaive MASE | SNaive MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy_trend | 1 | 0.6491 | 0.7158 | 0.7206 | 0.1544 | 0.1565 | 1.3417 | 1.1607 | 0.6369 |
| legacy_trend | 2 | 0.7291 | 0.7409 | 0.7752 | 0.1831 | 0.1228 | 1.3246 | 1.1604 | 0.6003 |
| legacy_trend | 3 | 0.787 | 0.755 | 0.8194 | 0.2635 | 0.1015 | 1.2632 | 1.127 | 0.5739 |
| legacy_trend | 4 | 0.8139 | 0.7685 | 0.8412 | 0.3085 | 0.0861 | 1.2535 | 1.0898 | 0.5459 |
| legacy_trend | 5 | 0.8339 | 0.7734 | 0.8484 | 0.35 | 0.0792 | 1.2554 | 1.0714 | 0.535 |
| v2_trend | 1 | 0.0014 | 0.86 | 0.0865 | 0.0745 | 0.1385 | 2.112 | 1.0699 | 0.4607 |
| v2_trend | 2 | 0.0804 | 0.8628 | 0.1219 | 0.1171 | 0.1334 | 2.1099 | 1.0897 | 0.4666 |
| v2_trend | 3 | 0.25 | 0.8647 | 0.2101 | 0.1571 | 0.1273 | 2.1206 | 1.0854 | 0.4625 |
| v2_trend | 4 | 0.4502 | 0.8675 | 0.3212 | 0.2368 | 0.1172 | 2.1009 | 1.1062 | 0.4661 |
| v2_trend | 5 | 0.6091 | 0.8708 | 0.4066 | 0.3196 | 0.1072 | 2.0875 | 1.2206 | 0.5126 |
| legacy_multi_seasonal | 1 | 0.015 | 0.656 | 0.1 | 0.0509 | 0.3401 | 1.2233 | 0.996 | 1.0007 |
| legacy_multi_seasonal | 2 | 0.0159 | 0.6696 | 0.0982 | 0.0492 | 0.3265 | 1.3234 | 0.9983 | 0.97 |
| legacy_multi_seasonal | 3 | 0.0138 | 0.6456 | 0.0946 | 0.0515 | 0.3506 | 1.296 | 0.998 | 0.9989 |
| legacy_multi_seasonal | 4 | 0.0149 | 0.6691 | 0.0931 | 0.055 | 0.3272 | 1.3616 | 0.9997 | 0.9522 |
| legacy_multi_seasonal | 5 | 0.0138 | 0.6432 | 0.0905 | 0.0548 | 0.3528 | 1.3701 | 1.0088 | 0.9937 |
| v2_multi_seasonal | 1 | 0.0156 | 0.977 | 0.0546 | 0.0537 | 0.0228 | 7.1562 | 1.0072 | 0.1627 |
| v2_multi_seasonal | 2 | 0.013 | 0.9512 | 0.0584 | 0.0596 | 0.0484 | 3.4495 | 0.9997 | 0.3295 |
| v2_multi_seasonal | 3 | 0.0129 | 0.8791 | 0.0697 | 0.0627 | 0.1199 | 1.9327 | 0.9892 | 0.5879 |
| v2_multi_seasonal | 4 | 0.0126 | 0.7747 | 0.0834 | 0.0673 | 0.2232 | 1.4562 | 1.007 | 0.8421 |
| v2_multi_seasonal | 5 | 0.015 | 0.6818 | 0.0935 | 0.0698 | 0.3148 | 1.1399 | 0.9988 | 1.0051 |
| real_m4_hourly | - | 0.2707 | 0.8629 | 0.1354 | 0.171 | 0.1291 | 14.0328 | 1.0416 | 0.2446 |

## 验收检查

- v2 trend strength 单调：`True`
- v2 trend slope 均值不超过 cap：`True`
- legacy trend slope 均值不超过 cap：`False`
- v2 multi-seasonal seasonal naive MAE 单调：`True`
- v2 multi-seasonal seasonal naive MAE 增长倍数：`6.1774`

## 结论

- 旧 trend 公式低难度已经有很强趋势，且 slope 均值超过真实 cap；v2 pilot 把 trend strength 调成随 difficulty 单调增强，并把 slope 均值压回 cap 内。
- 旧 multi-seasonal 公式没有稳定制造“单周期 seasonal naive 更难”的响应；v2 pilot 通过 48 点次级周期让 seasonal naive MAE 随 difficulty 明显上升。
- M4 真实窗口保留在同表里，作为当前特征和基线误差的真实参照；后续可以把更多真实数据集加入同一脚本。

## 复现

```bash
cd backend && PYTHONPATH=.:../scripts uv run python ../scripts/run_synthetic_v2_generator_experiment.py
```
