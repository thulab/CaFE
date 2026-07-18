# Paper v4：逐 Dataset 九能力 Suite 重建记录

日期：2026-07-18
状态：旧结果作废；real-bounded intensity 的 ETT1 smoke 通过；正式 freeze 待重建

## 作废结论

此前记录的 `360/360` 母样本合格和 4 个 task-level conditioning profile 基于跨
dataset 汇总后的真实分布与统一强度，不再是有效论文证据。旧
`capability_dataset_mapping.csv`、全局 conditioning/gate artifact 和相关哈希全部
作废，运行目录已清理。

## 新契约

新 suite 使用：

```text
schema = paper_v4_nine_capability_suite.v2
intensity_policy = dataset-local-real-bounded-generator-feasible-v1
real tolerance = [q05, 1.2*q95]
relative levels = 0/0.25/0.50/0.75/1.00
```

每个 dataset：

- 独立构造四个 lookback profile；
- 固定输出九能力 support matrix；
- 缺 task view、变量结构、窗口、真实—生成器可行交集或档间距时如实记
  `unsupported`；
- supported cell 独立拟合 conditioning、feature-support 和 near-distance；
- qualification 不读取其他 dataset 的统计量。

真实数据只给出主特征容忍区间，五档由该区间与生成器响应区间的交集决定。上限放大
`1.2` 允许合成样本有限地凸显能力，但不允许生成器外推到二者交集之外。
`regime_switching` 的 recurring-clock future qualification 只保留为诊断，不再要求
真实序列具有与合成方波相同的潜在机制。

## ETT1 真实 Smoke

配置：

```text
dataset = gift_ett1_h
master windows = 120
H = 48
L = 96/168/336/504
conditioning calibration samples = 4（smoke）
qualification samples = 每个 supported capability × intensity 1 条
```

完整九能力 support matrix：

| capability | status | reason / calibration |
| --- | --- | --- |
| `trend` | supported | max normalized error 0.0730 |
| `multi_seasonal` | supported | max normalized error 0.0262 |
| `time_varying_seasonality` | supported | max normalized error 0.1329 |
| `regime_switching` | supported | max normalized error 0.0855；recurring qualification 仅作诊断 |
| `nonlinear_persistence` | supported | max normalized error 0.0117 |
| `predictable_intermittency` | supported | max normalized error 0.1131 |
| `common_factor` | unsupported | variable structure not supported |
| `hierarchical_coherence` | unsupported | variable structure not supported |
| `covariate_response` | unsupported | variable structure not supported |

6 个单变量能力全部在各自真实容忍与生成器响应的交集内形成五个单调档位；3 个
structured 能力因 task view 不支持而如实跳过。没有从其他 dataset 借 target 或
task view。

qualification 覆盖 6 个 supported capabilities × 5 档 × 1 条，共 30 条母样本；
`30/30` 通过 construction、四 lookback feature-support 和四 lookback
near-distance；120 个 suffix view 全部通过，样本全部首轮通过，失败数为 0。

feature-gate artifact 同时冻结每个 dataset/profile/capability 的
gate-reference 与 gate-calibration 标准化 control vectors，供 E1 直接复用同一 split；
不再在 E1 运行时从原始数据重建 split。

## 三个额外 Dataset 的 Smoke

相同 smoke 参数另运行 Electricity Hourly、ETT2/H 和 Jena Weather：

| dataset | 单变量 supported | structured unsupported | 最大 calibration error | qualification |
| --- | ---: | ---: | ---: | ---: |
| `gift_electricity_h` | 6/6 | 3/3 | 0.0850 | samples 30/30；views 120/120 |
| `gift_ett2_h` | 6/6 | 3/3 | 0.1096 | samples 30/30；views 120/120 |
| `gift_jena_weather_h` | 6/6 | 3/3 | 0.0677 | samples 30/30；views 120/120 |

三个 dataset 共 18 个 supported cells、90 条母样本和 360 个 lookback views，全部
一次验收通过，失败数为 0。regime 的真实 recurring-clock 诊断分别为
`1/42`、`0/46`、`0/32`，但三者的 dataset-local 可观察强度范围、五档逆标定和生成
资格均通过，支持将该诊断保留为描述性审计而非硬门槛。

额外 smoke 输出位于：

```text
runtime/paper_exp/v4/04_multi_dataset_real_bounded_check/
```

## Smoke 输出

```text
runtime/paper_exp/v4/03_real_bounded_intensity_check/
  profile_suite.json
  dataset_capability_support_matrix.json
  dataset_capability_support_matrix.csv
  generator_conditioning_artifact.json
  feature_gate_artifact.json
  near_distance_artifact.json
  qualification.json
  manifest.json
```

这些是 `calibration_samples=4`、每 cell 一条 qualification 样本的 smoke artifact，
不替代正式多 dataset、正式 calibration sample count 的 freeze，因此不冻结哈希。

协议：

`docs/superpowers/specs/2026-07-18-paper-v4-nine-capability-profile-and-generation-protocol.md`
