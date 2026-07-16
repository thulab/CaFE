# Paper E2：动态评测稳定性正式结果

日期：2026-07-17

## 结论

paper-v1 E2 正式实验完成，预注册的 5 类操作性稳定标准全部通过。五个独立生成轮次下，基础模型的分数波动低、能力表现轮廓高度一致、模型排名总体稳定，且各轮不存在曲线复刻。因此，后续实验采用每个 `profile × capability × intensity` 每轮 32 条样本，可以把观察到的主要模型差异解释为能力差异，而不是某一次动态生成的偶然结果。

这一结论只建立动态 benchmark 的重复性，不替代后续合成能力与真实数据表现之间的外部效度验证。

## 冻结配置与覆盖

- runner/protocol commit：`e9d03669b561ab510a26455133921659157ea4a5`。
- 正式推理服务基线：timer-rest-service `c7f822bbd166917eec9470af64b026419dd64c27`，叠加 benchmark 已使用的 TiRex CUDA device-context 修复；该文件 SHA-256 为 `77b69002b920cc47b695f43e660064aa46aebffa3e75978239970716bc964e09`，相对该 commit 的 diff SHA-256 为 `9ea4603408e985ead82dc11b4f3ca8181b49dddcb19fd44646f0e0e7db965020`。
- canonical scale：`synthetic-v2-paper-v1-frozen-2026-07-16`，fingerprint `a76b66924562be4f`。
- 23 个 `profile × capability` cells，5 档 intensity，5 个独立轮次，每轮每格 32 条，共 18,400 条生成样本。
- 7 个基础模型共 112,800 条预测；`naive` 与 `seasonal_naive` 共 36,800 条预测。
- 所有基础模型和基线覆盖率均为 100%；112,800 个基础模型 HTTP 请求全部第一次尝试成功，正式 `failures/*.jsonl` 均为空。
- 每个模型均按冻结的双卡 replica 配置加载，实际 endpoint 数通过强校验；请求按完整 shape signature 连续执行。
- 正式墙钟时间约 37 分 40 秒，其中模型加载、推理和卸载合计 1,998.9 秒。

完整不可变输出位于 `runtime/paper_exp/v1/E2_dynamic_stability/`，共约 312 MiB。manifest 记录 29 个输出文件、5 个输入文件和代码 commit；本次复核确认全部 326,214,239 bytes 的 SHA-256 与文件大小一致。

## 预注册判据

| 判据 | 阈值 | 正式结果 | 通过 |
| --- | --- | --- | --- |
| Round-score CV | median ≤ 0.10；p95 ≤ 0.25 | 0.0304；0.0812 | 是 |
| 模型能力轮廓 ICC(A,1) | 所有基础模型最低值 ≥ 0.90 | minimum 0.9895；median 0.9951 | 是 |
| 模型排名 Kendall τ-b | cell mean τ 的 median ≥ 0.80；p10 ≥ 0.50 | 0.8857；0.7333 | 是 |
| Bootstrap 95% CI 相对宽度 | median ≤ 0.20；p95 ≤ 0.50 | 0.1043；0.2034 | 是 |
| 跨轮多样性 | 精确、六位小数、DCR ≤ 1e-6 重复率均为 0 | 三者最大值均为 0 | 是 |

补充统计：cell ranking mean Kendall τ-b 最小值为 0.5048；cell-level model-by-round ICC 中位数为 0.9215；跨轮 DCR q01 的全局最低值为 0.076889，NNDR q05 的全局最低值为 0.241163。

## 分模型稳定性

| 模型 | 预测数 | CV median | CV p95 | CI width median | ICC(A,1) | 模型阶段耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Timer-3.5` | 14,400 | 0.0281 | 0.0593 | 0.1034 | 0.9970 | 155.0s |
| `Timer-3.0` | 14,400 | 0.0308 | 0.0829 | 0.1113 | 0.9953 | 82.6s |
| `Chronos-2` | 18,400 | 0.0300 | 0.0824 | 0.0990 | 0.9951 | 105.1s |
| `moirai2` | 14,400 | 0.0390 | 0.1055 | 0.1248 | 0.9895 | 55.9s |
| `toto2.0` | 16,800 | 0.0313 | 0.0529 | 0.1021 | 0.9960 | 453.3s |
| `timesfm2.5` | 16,000 | 0.0306 | 0.0774 | 0.1009 | 0.9942 | 1,033.8s |
| `tirex2` | 18,400 | 0.0286 | 0.0783 | 0.0990 | 0.9950 | 113.2s |

所有模型的能力表现轮廓 ICC 均高于 0.989。`moirai2` 的轮次 CV 相对最高，但 p95 仍只有 0.1055，远低于预注册上限。

## 分能力诊断

| Capability | CV median | CV p95 | CI width median | CI width p95 | Kendall τ median | Kendall τ p10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `common_factor` | 0.0250 | 0.0357 | 0.0763 | 0.0929 | 1.0000 | 0.9600 |
| `covariate_response` | 0.0360 | 0.0593 | 0.1189 | 0.1536 | 0.7333 | 0.7200 |
| `hierarchical_coherence` | 0.0150 | 0.0260 | 0.0568 | 0.0684 | 1.0000 | 1.0000 |
| `multi_seasonal` | 0.0382 | 0.0910 | 0.1233 | 0.2629 | 0.9429 | 0.8743 |
| `nonlinear_persistence` | 0.0382 | 0.0637 | 0.1097 | 0.1409 | 0.8286 | 0.7181 |
| `predictable_intermittency` | 0.0278 | 0.0385 | 0.0906 | 0.1062 | 0.7524 | 0.6267 |
| `regime_switching` | 0.0249 | 0.0451 | 0.0939 | 0.1332 | 0.9429 | 0.8667 |
| `time_varying_seasonality` | 0.0584 | 0.1134 | 0.1700 | 0.2629 | 0.9238 | 0.8667 |
| `trend` | 0.0258 | 0.0385 | 0.0990 | 0.1280 | 0.8667 | 0.8171 |

结构化任务的分数稳定性尤其高。`covariate_response` 的 Kendall τ 数值较低，部分原因是该任务只有 3 个兼容基础模型，τ 的取值较离散；其 CV 和 CI 仍稳定。

`predictable_intermittency` 的低 intensity 排名最容易交换：最低 cell 是 M4 profile、intensity 1，mean τ 为 0.5048。按 intensity 汇总时，τ p10 从 intensity 1 的 0.6000 提升到 intensity 5 的 0.7562，符合弱结构下模型得分更接近、少量波动更容易交换名次的解释。

## 尾部风险与论文表述

- 705 个基础模型 cells 中，11 个（1.56%）CV 高于 0.10，只有 1 个（0.14%）高于 0.25。
- 37 个（5.25%）bootstrap 相对 CI 宽度高于 0.20，只有 1 个（0.14%）高于 0.50。
- 唯一同时形成全局最大 CV 和 CI width 的 cell 是 `Chronos-2 × m4_hourly_daily_168ctx × time_varying_seasonality × intensity 5`：CV 0.2781，CI width 0.5337。
- `time_varying_seasonality` 和 `multi_seasonal` 构成大部分宽区间尾部。因此论文可以主张总体和能力级结论稳定，但不应把每一个单模型、单 profile、单 intensity cell 都描述成同等精确。若后续需要围绕上述尾部 cell 作单格结论，应报告其 CI、跨 profile 聚合，或增加该格样本数。

## 输出索引

- `summary.json` / `report.md`：总结果和判据。
- `round_scores.csv` / `score_cv.csv`：逐轮得分及分数变异。
- `bootstrap_ci.csv`：分层 bootstrap 95% CI。
- `rank_stability.csv`：基础模型与含基线两种 scope 的 Kendall τ-b 和 cell ICC。
- `model_profile_icc.csv`：逐模型能力轮廓 ICC(A,1)。
- `cross_round_distance.csv`：双向跨轮 DCR、NNDR 和重复率。
- `model_status.json` / `model_coverage.csv`：replica 拓扑、shape 吞吐和完整性。
- `manifest.json`：输入与输出哈希及正式代码 commit。
