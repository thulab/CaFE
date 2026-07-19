# Paper v5 E2 正式推理与分析记录

> 2026-07-20 解释修正：本报告中的 E2-A 逐轮严格排名是 N=32 batch
> 敏感性结果，不再作为生成器具有 round-level latent stability 的证据。正式 E2
> 可靠性改为比较独立 N=160 seed banks，同时报告连续能力分数、capability
> profile、正式排名和 tie-aware 模型对。原始结果与产物保持封存，不回写。

日期：2026-07-19

## 1. 结论摘要

本轮 H=48 正式推理已完整结束，8 个 foundation models 的全部兼容
synthetic views 与 10 个单变量数据集的 real source views 均成功落盘，失败请求为
0。

预注册的 E2-A 严格主判据没有通过：315 个
`dataset × task × capability × intensity` cells 中只有 9 个满足“任意两轮
pairwise ordering agreement 均不低于 0.95”，通过率为 2.9%。因此本轮结果不支持
“更换随机 seed 后模型排名近乎完全一致”的强结论。

E2-B 呈正向但仅中等程度的 synthetic–real source-window alignment：

- dataset mean Spearman ρ = 0.4667，dataset bootstrap 95% CI
  `[0.2286, 0.6738]`；
- mean Kendall τ-b = 0.3286；
- mean top-3 overlap = 0.6333；
- mean pairwise ordering agreement = 0.6643。

因此当前结果支持“存在正向 construct alignment”，但不足以表述为“synthetic
多能力平均排名与真实排名保持一致”。该结论仍仅限 dataset-local calibration
source windows，不是 held-out external validity。

## 2. 冻结输入与协议

- synthetic masters：50,400；
- synthetic four-context views：201,600；
- synthetic samples SHA-256：
  `23dae5c78e96a4ce8ba5f6169fa1ce879421f92ed0394d92095d80b23a4b698f`；
- real source masters：417；
- real source four-context views：1,668；
- real source samples SHA-256：
  `9faef8605b1b9edecf994a74e404c637bef85da425f02d0a402ac9a0204d1ddc`；
- protocol SHA-256：
  `8dba6511a2aa2824bf95e83b8720f004adcb02807673c081f00a3b98ab3fdd46`；
- inference runner SHA-256：
  `85cccd52c64d7793e912befdfc4bb8ddeb41bb46b0b4fd1264ee2b657ca1f964`。

每个 model/master 均保留 L=96、168、336、504 四个 view。主分析逐模型、逐
master 选择 MASE 最低的 context；完全相同的 MASE 选择更短 context。固定 L=504
结果作为敏感性分析。

## 3. 模型与完成情况

Foundation models：

1. Timer-3.5；
2. Timer-3.0；
3. Chronos-2；
4. moirai2；
5. toto2.0；
6. timesfm2.5；
7. tirex2；
8. tabpfn-ts3。

Foundation-model synthetic 成功 views 共 1,536,000，real source 成功 views
共 13,344。各模型兼容条数分别为：

| Model | Synthetic | Real source |
|---|---:|---:|
| Timer-3.5 | 182,400 / 182,400 | 1,668 / 1,668 |
| Timer-3.0 | 182,400 / 182,400 | 1,668 / 1,668 |
| Chronos-2 | 201,600 / 201,600 | 1,668 / 1,668 |
| moirai2 | 182,400 / 182,400 | 1,668 / 1,668 |
| toto2.0 | 195,200 / 195,200 | 1,668 / 1,668 |
| timesfm2.5 | 188,800 / 188,800 | 1,668 / 1,668 |
| tirex2 | 201,600 / 201,600 | 1,668 / 1,668 |
| tabpfn-ts3 | 201,600 / 201,600 | 1,668 / 1,668 |

`naive` 与 `seasonal_naive` 仅作诊断基线，不进入 headline 排名统计。

## 4. 双机推理 provenance

本机运行 Timer、Chronos、moirai、toto、timesfm 和主版本 tabpfn。第二台相同
服务运行 tirex 与一份独立 tabpfn replica：

- tirex synthetic 与 real source 文件复制到主结果目录后 SHA-256 与 shard
  原文件逐字节一致；
- tabpfn 在两台服务上独立完成。按 `view_id` 忽略请求耗时和 JSONL 行序后，
  201,600 个 synthetic views 与 1,668 个 real source views 的 forecast 和 metrics
  全部一致，semantic mismatch 为 0；
- shard 配置、目录、状态、结果哈希和比对结论封存在
  `runtime/paper_exp/v5/E2_dynamic_stability/inference_shards.json`。

## 5. E2-A：轮次排名稳定性

Oracle-context 主结果：

- 通过 cells：9 / 315（2.9%）；
- cell 最差轮次对 agreement 的 median / global minimum：
  0.6786 / 0.1071；
- cell mean Kendall τ-b 的 median / global minimum：
  0.6000 / -0.7857；
- 所有 cells 的 mean exact-rank pair rate：0.0854；
- mean top-1 pair agreement：0.6222；
- mean top-3 overlap：0.7324；
- model score round CV median / p90：0.0396 / 0.0833；
- common difficulty-multiplier CV median / p90：0.0380 / 0.0805。

固定 L=504 不能改变主结论：同样只有 9 / 315 cells 通过。其最差轮次对
agreement median 从 0.6786 提升至 0.7143，说明逐样本 oracle context 会增加一部分
排名波动，但不是失败的根因。

### 按 intensity

| Intensity | Cells | Passed | Median worst-pair agreement | Mean top-1 agreement |
|---:|---:|---:|---:|---:|
| 1 | 63 | 2 | 0.5357 | 0.4810 |
| 2 | 63 | 1 | 0.5357 | 0.5429 |
| 3 | 63 | 2 | 0.6786 | 0.6413 |
| 4 | 63 | 0 | 0.7143 | 0.7048 |
| 5 | 63 | 4 | 0.7500 | 0.7413 |

强度升高时排名与 top-1 稳定性总体改善，说明更突出的能力机制确实减少了部分 seed
敏感性；但 intensity=5 仍远未达到预注册阈值。

### 按 capability

| Capability | Cells | Passed | Median worst-pair agreement | Mean top-1 agreement |
|---|---:|---:|---:|---:|
| trend | 50 | 1 | 0.4107 | 0.3260 |
| regime_switching | 45 | 0 | 0.4643 | 0.4200 |
| nonlinear_persistence | 45 | 0 | 0.4643 | 0.5333 |
| common_factor | 15 | 0 | 0.6667 | 0.4333 |
| covariate_response | 10 | 1 | 0.6667 | 0.6600 |
| predictable_intermittency | 50 | 1 | 0.7500 | 0.9740 |
| time_varying_seasonality | 45 | 1 | 0.8214 | 0.8711 |
| hierarchical_coherence | 5 | 0 | 0.8333 | 0.8800 |
| multi_seasonal | 50 | 5 | 0.8929 | 0.6280 |

`trend`、`regime_switching` 和 `nonlinear_persistence` 是最主要的不稳定来源。
`predictable_intermittency` 的 top-1 几乎固定，但完整八模型次序仍会交换；这说明
“赢家稳定”和“完整排名稳定”不能混为一个结论。

## 6. E2-B：synthetic–real source-window alignment

10 个 dataset 的 Spearman ρ 范围为 -0.2857 到 0.8810：

- 强正向：Bitbrains Fast 0.8810、M4 Hourly 0.8333、M Dense 0.7857；
- 中等正向：Loop Seattle 0.6667、ETT2 0.6190、SZ Taxi 0.5714；
- 较弱或不一致：ETT1 0.3095、Electricity 0.1429、Solar 0.1429、
  Jena Weather -0.2857。

固定 L=504 的 mean Spearman ρ 为 0.3929，低于 oracle-context 的 0.4667；因此
synthetic–real 的正向对齐并非由 oracle context 人为制造，但整体效应仍属中等。

## 7. MASE 边界处理

5 / 417 个 real masters 的部分短 history 完全平坦，使 MASE 分母为 0：

- 共 7 / 1,668 个 views；
- 仅出现在 L=96 或 L=168；
- 涉及 Bitbrains Fast 与 ETT2；
- 同一 master 的 L=336 和 L=504 MASE 均有效。

处理规则为：oracle context 只在 MASE 有定义的候选中选择，并在逐样本记录中保留
`flat_history` 原因；不插值、不将 MAE 冒充 MASE，也不丢弃整个 master。固定 L=504
敏感性分析不受影响。

## 8. 产物与复核

主目录：

`runtime/paper_exp/v5/E2_dynamic_stability/`

核心文件：

- `inference_summary.json`；
- `inference_report.md`；
- `inference_manifest.json`；
- `inference_shards.json`；
- `cell_round_scores.csv`；
- `cell_rank_stability.csv`；
- `cell_score_stability.csv`；
- `cell_difficulty_stability.csv`；
- `synthetic_model_ranks.csv`；
- `real_source_model_ranks.csv`；
- `synthetic_real_source_alignment.csv`；
- 对应的 `_l504` 敏感性文件；
- 按 dataset、task、capability、intensity 汇总的 rank-stability tables。

最终封存哈希：

- inference config：
  `05feb7b431f4f8b1f80a9b5c1508eb2462b427ca88d66482d43c8c58f3087448`；
- inference summary：
  `fe2b5c0c15b73e7b8121256158e85e52996682e85b3c7eff6fa0ccb611e8e7b2`；
- inference manifest：
  `902e858b67957ce7f3ed382dfbaf3d80523fbc4feaf4d394a823b75bd318afc9`；
- distributed shard record：
  `7a8361f2882a3826ab23af3f1560438641eca5b5dfb3b71979c18f1ef6bc0ada`。

说明：协议正文中一处沿用“七模型 cell”措辞，实际冻结模型数和本轮分析均为 8。
为避免改变已记录的 protocol SHA-256，本次不回写冻结协议，而在本记录中作勘误。
