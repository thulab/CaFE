# Paper v2 E2：六个单变量能力动态稳定性

日期：2026-07-17

## 结论

paper-v2 E2 正式实验完成。七个基础模型在五个独立生成轮次上的连续分数稳定，模型的整体
能力轮廓高度一致，bootstrap 区间足够窄，跨轮没有曲线复刻；但预注册的严格逐-cell 模型
全排序判据未通过。因此 E2 的正式结论是 **4/5 类稳定标准通过**，不能表述为所有标准通过。

该结果支持后续 E3 使用连续 MASE、relative skill、intensity response 与配对 CI 形成能力
画像；不支持把每个 `profile × capability × intensity` 内七个模型的精确名次当作稳定事实。

## 冻结设计与覆盖

- 输入：paper-v2 transfer freeze，`context=504`、`horizon=48`、
  `season_length=24`、单目标。
- 9 个 held-out GIFT hourly profiles × 6 capabilities × 5 intensities。
- 5 个独立 generation rounds，每轮每 cell 16 条，共 21,600 条样本。
- intensity 间和模型间使用配对 seed；270 个最小 cell 均为 80 条。
- 7 个基础模型与 `naive`、`seasonal_naive` 各完成 21,600 条预测，共 194,400 条。
- 所有在线请求第一次尝试成功，正式 `failures/*.jsonl` 为空，模型覆盖率均为 100%。
- 4,320 个 `profile × capability × round × sample_index` 配对组内，五档 intensity 的
  `sample_seed` 基数全部为 1。
- 跨轮精确、六位小数和 DCR≤1e-6 重复率最大值均为 0。

## 预注册判据

| 判据 | 阈值 | 正式结果 | 通过 |
| --- | --- | --- | --- |
| Round-score CV | median ≤ 0.10；p95 ≤ 0.25 | 0.0341；0.0774 | 是 |
| 模型能力轮廓 ICC(A,1) | 所有基础模型最低值 ≥ 0.90 | minimum 0.9823；median 0.9841 | 是 |
| 模型排名 Kendall τ-b | cell mean τ median ≥ 0.80；p10 ≥ 0.50 | 0.6571；0.2952 | **否** |
| Bootstrap 95% CI 相对宽度 | median ≤ 0.20；p95 ≤ 0.50 | 0.0973；0.1931 | 是 |
| 跨轮多样性 | 三类重复率均为 0 | 三者最大值均为 0 | 是 |

跨轮 DCR q01 的全局最低值为 0.084876，NNDR q05 的全局最低值为 0.319010。

## 分能力稳定性

| Capability | CV median | CV p95 | CI width median | CI width p95 | Kendall τ median | Kendall τ p10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `trend` | 0.0351 | 0.0638 | 0.1051 | 0.1731 | 0.7524 | 0.2990 |
| `multi_seasonal` | 0.0411 | 0.0782 | 0.1288 | 0.2009 | 0.6000 | 0.2838 |
| `time_varying_seasonality` | 0.0317 | 0.1114 | 0.0930 | 0.2664 | 0.7143 | 0.5086 |
| `regime_switching` | 0.0402 | 0.1086 | 0.1033 | 0.2065 | 0.6381 | 0.4743 |
| `nonlinear_persistence` | 0.0257 | 0.0477 | 0.0807 | 0.0996 | 0.3524 | 0.1695 |
| `predictable_intermittency` | 0.0290 | 0.0623 | 0.0856 | 0.1698 | 0.8095 | 0.6648 |

排名不稳定最明显的是 `nonlinear_persistence`，但它同时拥有最小的 CV 与最窄 CI。这说明主要
问题不是分数本身漂移，而是七个模型在这些 cells 上非常接近，微小且稳定的分数波动会改变
完整排序。

一个未改变正式判据的 post-hoc 诊断把互不重叠的两轮合并为 32 条后再比较：Kendall τ
median 从 0.657 提升到 0.759，p10 从 0.295 提升到 0.479，仍未达到 0.80/0.50。七模型
相邻分数的 cell-level median relative margin 只有 0.64%。因此单纯把每轮样本翻倍成本
很高，也不能保证严格全排序稳定；E3 应优先报告连续量和等价/不确定性，而不是事后放宽门限。

## 模型执行

| 模型 | replica/卡 | HTTP 并发 | 预测数 | 请求耗时 | 模型阶段总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Timer-3.5` | 1 | 64 | 21,600 | 160.2s | 230.4s |
| `Timer-3.0` | 1 | 32 | 21,600 | 111.3s | 128.4s |
| `Chronos-2` | 4 | 32 | 21,600 | 121.0s | 139.7s |
| `moirai2` | 2 | 16 | 21,600 | 101.2s | 117.5s |
| `toto2.0` | 2 | 16 | 21,600 | 525.2s | 576.9s |
| `timesfm2.5` | 8 | 32 | 21,600 | 1,346.0s | 1,377.2s |
| `tirex2` | 1 | 32 | 21,600 | 143.2s | 161.4s |

七模型阶段总耗时为 2,731.5 秒，其中请求耗时 2,508.0 秒。每个模型加载后的 endpoint 数、
device 分布与 replica/卡均通过 runner 强校验。

## 封存信息

- runner/freeze commit：`08d752cbd8ed1418f70db63b310d2203d390cfca`
- 输出：`runtime/paper_exp/v2/E2_dynamic_stability/`
- 大小：617,699,983 bytes
- manifest SHA-256：
  `91b61c7d4b3d4cd81da28f011d6d6e0810db423d1c16b0bb336a6f17a2e1f34d`
- manifest 文件数：29

完整输出包含 `round_scores.csv`、`score_cv.csv`、`bootstrap_ci.csv`、
`rank_stability.csv`、`model_profile_icc.csv`、`cross_round_distance.csv`、
模型状态、预测 JSONL 和输入/输出哈希。

