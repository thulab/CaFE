# Paper v4：逐 Dataset 九能力 Suite 重建记录

日期：2026-07-18
状态：旧结果作废；真实 ETT1 dataset-local smoke 通过；正式多 dataset freeze 待重建

## 作废结论

此前记录的 `360/360` 母样本合格和 4 个 task-level conditioning profile 基于跨
dataset 汇总后的真实分布与统一强度，不再是有效论文证据。旧
`capability_dataset_mapping.csv`、全局 conditioning/gate artifact 和相关哈希全部
作废，运行目录已清理。

## 新契约

新 suite 使用：

```text
schema = paper_v4_nine_capability_suite.v2
intensity_policy = dataset-local-relative-quantiles-v1
percentiles = q10/q30/q50/q70/q90
```

每个 dataset：

- 独立构造四个 lookback profile；
- 固定输出九能力 support matrix；
- 缺 task view、变量结构、窗口或档间距时如实记 `unsupported`；
- supported cell 独立拟合 conditioning、feature-support 和 near-distance；
- qualification 不读取其他 dataset 的统计量。

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
| `trend` | supported | max normalized error 0.0955 |
| `time_varying_seasonality` | supported | max normalized error 0.1061 |
| `predictable_intermittency` | supported | max normalized error 0.0692 |
| `multi_seasonal` | unsupported | conditioning calibration failed |
| `regime_switching` | unsupported | 46 个 parameter windows 中 recurring-regime qualified 为 0 |
| `nonlinear_persistence` | unsupported | conditioning calibration failed |
| `common_factor` | unsupported | variable structure not supported |
| `hierarchical_coherence` | unsupported | variable structure not supported |
| `covariate_response` | unsupported | variable structure not supported |

这正是新协议期望的行为：一个 dataset 固定审计九项，但只对真实结构与校准充分的能力
生成样本。没有从其他 dataset 借 target 或 task view。

qualification 覆盖 3 个 supported capabilities × 5 档 × 1 条，共 15 条母样本；
`15/15` 通过 construction、四 lookback feature-support 和四 lookback
near-distance，全部首轮通过，失败数为 0。

feature-gate artifact 同时冻结每个 dataset/profile/capability 的
gate-reference 与 gate-calibration 标准化 control vectors，供 E1 直接复用同一 split；
不再在 E1 运行时从原始数据重建 split。

## 新输出

```text
runtime/paper_exp/v4/01_nine_capability_suite/
  profile_suite.json
  dataset_capability_support_matrix.json
  dataset_capability_support_matrix.csv
  generator_conditioning_artifact.json
  feature_gate_artifact.json
  near_distance_artifact.json
  qualification.json
  manifest.json
```

当前 smoke 哈希：

```text
dataset_capability_support_matrix.json
  0fbf59a4057d643645d73da639cc26ab491c5c17b03f46ba2a0c447a8ac410d1
generator_conditioning_artifact.json
  947f94d15635bad4e8c2dc325794397879c5ced4981eaf5873c7bddee8a1b2c6
feature_gate_artifact.json
  bfce9a9878d104b84e70dfbeb625c5b94702b3b422588383908b9d2c6b848094
near_distance_artifact.json
  63000d88d074bc5d05ead8465be65147b7f9b4c0d3e433d577d6505edb481546
qualification.json
  9339f9ae8a05245ea666434d8af3c2954a324e160807a36722f6b1e360570d90
manifest.json
  052a914bc2c1001af347a465693b0447e2ec21cc91731f768aa3543e1f36793b
```

这些是 smoke artifact，不替代正式多 dataset、正式 calibration sample count 的
freeze。

协议：

`docs/superpowers/specs/2026-07-18-paper-v4-nine-capability-profile-and-generation-protocol.md`
