# Paper v6 E2 正式重跑记录

日期：2026-07-20

## 1. 结论摘要

九能力生成机制更新后的 v6 Bank A 已完成逐数据集校准、正式样本生成、三机并行推理
和统计分析。

- 校准支持 `64` 个 `dataset × task × capability` cells；
- qualification 为 `2,560/2,560`，失败 `0`；
- 正式生成 `51,200` 条 synthetic masters，即每个
  `dataset × task × capability × intensity` 恰有 `160` 条；
- 八个 foundation models 共完成 `1,561,600` 条 synthetic view 预测和
  `13,344` 条 real-source view 预测；
- 所有模型均为 `complete`，请求失败总数为 `0`。

本轮只有一个 N=160 seed bank。全部 160 条样本用于能力分数和正式模型排名；
沿用的 `5 × 32` 划分只用于观察有限样本下的 batch sensitivity，不作为跨 seed-bank
稳定性证据。要评价“更换整套随机种子后结果能否复现”，仍需生成独立的 N=160
Bank B，并按 seed-bank reliability protocol 比较两套完整排名和能力分数。

synthetic–real source-window 对齐为中等正向，而不是“排名一致”：

- oracle-context mean Spearman ρ = `0.4024`，dataset bootstrap 95% CI
  `[0.1476, 0.6262]`；
- mean Kendall τ-b = `0.3035`；
- mean top-3 overlap = `0.6333`；
- mean pairwise ordering agreement = `0.6517`。

## 2. 校准与生成

本轮从当前生成机制重新校准，不复用 v5 calibration 或 synthetic samples。
生成器为 `capts-paper-v4`，Horizon 为 `48`，Context views 为
`96/168/336/504`。每条 master 使用 `L=504,H=48`，四个 context 共享同一 future。

20 个 dataset/task views 逐数据集独立校准，unsupported capability 如实保留。
支持能力 cell 数如下：

| Capability | Supported cells |
|---|---:|
| trend | 10 |
| multi_seasonal | 10 |
| time_varying_seasonality | 9 |
| regime_switching | 10 |
| nonlinear_persistence | 9 |
| predictable_intermittency | 10 |
| common_factor | 3 |
| hierarchical_coherence | 1 |
| covariate_response | 2 |

正式生成规模：

- supported cells：`64`；
- paired groups：`10,240`；
- master samples：`51,200`；
- dataset/task/capability/intensity cells：`320`；
- 每个 cell：`160` 条；
- 每模型最大 four-context views：`204,800`；
- real-source masters/views：`417/1,668`。

每个 paired group 的 I1–I5 共享 base seed，所有 group 均具有完整五档 intensity；
每条落盘样本的四个 context 均通过 feature gate 和 near-distance gate。

关键输入哈希：

- calibration support matrix：
  `fc167a4c78935931634e6eef0ed0ea6bdfd7e11653c00357ed12afe65c475f46`；
- calibration qualification：
  `567ac86b9c289513c7ef40f044fca7ac83f415614af66b9146cfc442f44a167c`；
- calibration manifest：
  `c7934bac2b1a843bb590bee70347f0bba3732724d0c00569a4c96955bde3f0aa`；
- generation config：
  `0482dc4c2082aba78816666e672f6566fff368fcf7c14b24ebb5d59984d65c9a`；
- sample manifest：
  `f510bf66868d88e749c37ec4fc11df8cee1446da0e9abde8113b2a778c27dde6`；
- synthetic samples：
  `8b9f2c0ce7016f135819a7361b7dc3f3d093b0ed584ddef230fc56899c59a2ba`；
- real-source manifest：
  `a2caed0c0c0a33330ab69fbca8664fd2345a726124f48176c47d9a8049c5b93f`；
- real-source samples：
  `0e140e302eec5151543250b1a1a31ee89608339dd0701c8a3d66b0f0a116fde8`。

## 3. 三机推理与完成情况

按 v5 实测耗时做模型级负载分配，同一台服务依次加载模型：

| Service | Models |
|---|---|
| `127.0.0.1:10810` | `timesfm2.5 → moirai2` |
| `192.168.99.18:10810` | `tabpfn-ts3 → Timer-3.0` |
| `192.168.99.17:10811` | `toto2.0 → Chronos-2 → tirex2 → Timer-3.5` |

远端请求显式设置 `NO_PROXY/no_proxy`。三台机器分别写独立 shard；推理结束后检查
输入身份、模型互斥覆盖、完成状态、JSONL 行数和 SHA-256，再以硬链接汇总到主目录。

| Model | Synthetic succeeded | Real succeeded | Synthetic elapsed |
|---|---:|---:|---:|
| Chronos-2 | 204,800 | 1,668 | 2,216.750 s |
| Timer-3.0 | 185,600 | 1,668 | 1,188.941 s |
| Timer-3.5 | 185,600 | 1,668 | 1,774.026 s |
| moirai2 | 185,600 | 1,668 | 1,001.211 s |
| tabpfn-ts3 | 204,800 | 1,668 | 11,895.220 s |
| timesfm2.5 | 192,000 | 1,668 | 12,366.418 s |
| tirex2 | 204,800 | 1,668 | 2,269.469 s |
| toto2.0 | 198,400 | 1,668 | 5,313.792 s |

不同模型的 synthetic 数量不同，是能力/输入结构兼容范围不同所致；各模型的成功数
均等于自身 compatible count。`naive` 和 `seasonal_naive` 也已完整计算，但只作诊断
基线，不进入八模型 headline 排名。

## 4. 正式排名与 batch sensitivity

主分数对每个模型、每条 master 在四个 context 中选择 MASE 最低者；完全相同则选择
更短 context。固定 `L=504` 作为敏感性分析。每个正式 cell 的模型能力分数和排名均
汇总全部 `160` 条样本。

为兼容旧分析脚本，产物仍包含把 160 条拆成 `5 × 32` 后计算的 round tables。
这些指标回答的是“只抽 32 条时排名有多敏感”，不是“生成器更换 seed 是否稳定”。
oracle-context 下：

- 320 个 cells 的 mean pairwise agreement 为 `0.7932`；
- mean top-1 agreement 为 `0.6078`；
- mean top-3 overlap 为 `0.7425`；
- 旧 `0.95` 最差轮次对阈值下为 `10/320`，该通过率不作为正式可靠性判定。

固定 `L=504` 的 mean pairwise agreement 为 `0.8175`，说明 oracle context 会引入
少量额外波动，但不是小批次排名变化的唯一来源。

## 5. Synthetic–real source-window alignment

10 个单变量数据集的 oracle-context Spearman ρ：

| Dataset | Spearman ρ |
|---|---:|
| Bitbrains Fast | 0.9286 |
| Electricity | 0.0476 |
| ETT1 | 0.3095 |
| ETT2 | 0.5952 |
| Jena Weather | -0.3571 |
| Loop Seattle | 0.6667 |
| M Dense | 0.5000 |
| Solar | -0.0240 |
| SZ Taxi | 0.5238 |
| M4 Hourly | 0.8333 |

固定 `L=504` 的 mean Spearman ρ 为 `0.3690`，95% CI 为
`[0.1500, 0.5762]`。因此正向关系并非完全由逐样本 oracle-context 选择制造，但不同
数据集差异很大。该结果只表示与 calibration source windows 的 construct alignment，
不是 held-out external validity，也不能表述为 synthetic 与 real 排名保持一致。

## 6. 产物与复核

主目录：

`runtime/paper_exp/v6/E2_dynamic_stability/`

主要结果：

- `inference_summary.json`；
- `inference_report.md`；
- `inference_manifest.json`；
- `inference_shards.json`；
- `cell_round_scores.csv`；
- `cell_rank_stability.csv`；
- `cell_score_stability.csv`；
- `synthetic_model_ranks.csv`；
- `real_source_model_ranks.csv`；
- `synthetic_real_source_alignment.csv`；
- 对应的 `_l504` 敏感性表。

最终分析与推理记录哈希：

- inference model catalog：
  `4144ea3b99e6afb969d1d4da5ff0e2e565086464c70f8f3beb39666eacaa826c`；
- distributed shard record：
  `bfdf29a1f4204a13bc9ea58ea730fc5eff6601b1591a3730cbb7c461deb0712d`；
- inference summary：
  `57b9ed914bee06509ec345e3cb7b2d4a12c2e68a1b291765023117e696e2bb6f`；
- inference manifest：
  `c5bd25d16e13f7d65431b89589ab507710fd25c201bf91f3a2b05e0c8ec1f0e2`。

验证：

- 相关生成、推理和 seed-bank tests：`20 passed`；
- backend 全量测试：`551 passed`；
- shard merger 最终修改后的定向测试：通过；
- 运行产物约 `9.6G`，位于 ignored 的 `runtime/paper_exp/v6/`，未纳入 Git。

当前分析器和部分 schema 名仍保留 `paper_v5`，表示沿用文件格式，不表示复用 v5
数据。v6 的实验边界和可靠性口径以本记录及对应正式计划为准。
