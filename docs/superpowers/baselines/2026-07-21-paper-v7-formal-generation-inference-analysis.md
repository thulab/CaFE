# Paper v7 正式生成、推理与 E2/E3 分析记录

- 日期：2026-07-21
- 范围：结构化数据扩展后的正式校准、生成、八模型推理、E2 稳定性与 E3 能力画像
- 状态：`complete`
- 解释边界：真实源窗口结果是同源构造对齐，不是 held-out external validity

数据准备与准入证据见
[`2026-07-21-paper-v7-p0-dataset-preparation.md`](2026-07-21-paper-v7-p0-dataset-preparation.md)，
预注册决策见
[`2026-07-21-paper-v7-structured-dataset-expansion-protocol.md`](../specs/2026-07-21-paper-v7-structured-dataset-expansion-protocol.md)。
本轮不补跑 v6 Bank B。

## 1. 最终实验面

正式 support matrix 在冻结阈值和最多 512 次尝试下 fail closed。资格复验淘汰
`gefcom2014_load::covariate` 与
`swiss_hierarchical_demand::common_factor`；正式五档联合生成又淘汰
`swiss_hierarchical_demand::covariate`。没有放宽 gate、阈值、条件或样本数。

| Capability | Supported cells | Datasets |
|---|---:|---:|
| common factor | 7 | 7 |
| hierarchical coherence | 3 | 3 |
| covariate response | 4 | 4 |
| multi-seasonal | 7 | 7 |
| nonlinear persistence | 7 | 7 |
| predictable intermittency | 7 | 7 |
| regime switching | 7 | 7 |
| time-varying seasonality | 7 | 7 |
| trend | 6 | 6 |
| **合计** | **55** | **15 个去重数据集** |

正式 support matrix SHA-256：
`1f0e6691cfe85edfe95472533a566194d8db6cac8cfb65608fbf3976bdd737a0`。

每个 supported cell 生成 320 个 paired groups，每组同时生成五档 intensity 和
L96/L168/L336/L504 四个视图：

- paired groups：17,600；
- master samples：88,000；
- 每个 `cell × intensity`：320；
- 每个模型 synthetic original views：352,000；
- A/B 分块：每块每 cell 160 groups、8,800 groups、44,000 masters；两块互斥且完备；
- real-source suite：1,136 masters、4,544 views。

生成配置 SHA-256：
`d361236a758c2b88da9612255e824bfa8b2fcde6652098e487ceecfc4caecd3d`；
样本 manifest SHA-256：
`53f411603a31e7ec39482f577038cf924793118c61f97de9b478cbc20bb46146`。

## 2. v7 输入适配与推理验收

比较单元冻结为“模型 + 预注册输入适配”：

- 不支持多目标时，把 D 个目标拆成 D 个独立单变量请求，再按原列顺序原子回组装；
- 不支持 known-future covariates 时，完整省略协变量；
- 只有全部 child requests 成功才写入一个 original view；
- 原生支持能力的模型继续使用原生多变量或协变量输入。

八模型 synthetic 均为 352,000/352,000，real-source 均为 4,544/4,544；另有
naive 与 seasonal-naive 各 352,000 synthetic 和 4,544 real。模型预测总计
2,816,000 synthetic + 36,352 real，统计基线总计 704,000 synthetic + 9,088 real。

| Model | Synthetic HTTP | Native views | Adapted views | Split target | Covariates omitted | Final failures |
|---|---:|---:|---:|---:|---:|---:|
| Chronos-2 | 352,000 | 352,000 | 0 | 0 | 0 | 0 |
| TiRex | 352,000 | 352,000 | 0 | 0 | 0 | 0 |
| TabPFN-TS | 352,000 | 352,000 | 0 | 0 | 0 | 0 |
| toto2.0 | 352,000 | 326,400 | 25,600 | 0 | 25,600 | 0 |
| TimesFM 2.5 | 492,800 | 281,600 | 70,400 | 70,400 | 0 | 0 |
| Timer-3.5 | 492,800 | 262,400 | 89,600 | 70,400 | 25,600 | 0 |
| Timer-3.0 | 492,800 | 262,400 | 89,600 | 70,400 | 25,600 | 0 |
| Moirai 2 | 492,800 | 262,400 | 89,600 | 70,400 | 25,600 | 0 |

Real-source 中四个拆分模型各执行 9,352 次 HTTP，其余模型各 4,544 次。所有
original-view 文件逐行核验为准确行数且没有重复键。

TabPFN 在第一次正式运行中发生一次远端 worker 卸载：34,421 个唯一视图留下
`503 model-not-loaded` 历史台账，共 103,263 次失败 HTTP。原目录 `--resume` 后这些
view 全部补齐，最终 missing=0、final failure=0。该台账属于基础设施瞬态事件，不计为
模型预测失败，也不删除。

推理配置与最终 manifest SHA-256 分别为
`4ae46add4465b8f0d5e1c41fa930a0fcedea087b0da207ecdb04a81afdb3974c`、
`fdb9eef17f15a6549b6aba8349853ddfe645a4826074e8d573a0e6f2683e562d`。

## 3. E2：160-group 稳定性

主估计使用每个 `model × dataset × task × capability × intensity` 的全部 320 条；
稳定性使用确定性的 A/B 两块，各 160 条求均值。不能把五个 64-group generation
round 当作 160-group 结果。

### 3.1 主要结果

| 指标 | Oracle context | Fixed L504 |
|---|---:|---:|
| raw MASE A/B CCC | 0.9959 | 0.9966 |
| capability-profile A/B CCC | 0.9921 | 0.9905 |
| capability-profile A/B Spearman | 0.9539 | 0.9640 |
| cell 排序 pairwise agreement 均值 | 0.8835 | 0.9042 |
| cell 排序 Kendall tau-b 均值 | 0.7670 | 0.8083 |
| Top-1 一致率 | 0.7345 | 0.7236 |
| Top-3 overlap 均值 | 0.8291 | 0.8436 |
| `agreement >= 0.80` | 220/275 (80.0%) | 238/275 (86.5%) |
| `agreement >= 0.95` | 101/275 (36.7%) | 105/275 (38.2%) |

在 0.02 实用等价边界下，oracle-context 的结论兼容率为 98.82%；成对 bootstrap
对比没有方向冲突。另一方面，精确完整 rank vector 的 A/B 一致率只有 16.7%，说明
160 条足以稳定能力画像、误差均值和多数相对顺序，但不支持把细小差异解释成稳定的
完整名次。五个独立 64-group rounds 的旧严格门槛仅通过 19/275，也支持“不用 64 条
做主结论”的设计。

详细 split summary SHA-256：
`5564a896d030ab82fcc953295ef3e61d306de47e78eea0afcbbd22ec4777dd6e`；
E2 summary SHA-256：
`19e602cd0494fc6f36ab205a7f9d63162945d7b4220b2de96e6eae3e2ac8896b`。

### 3.2 Synthetic 与真实源窗口的构造对齐

15 个数据集上的 oracle-context 平均 pairwise ordering agreement 为 0.6116
（bootstrap 95% CI 0.5214--0.7045），平均 Kendall tau-b 为 0.2223
（0.0429--0.4066），Top-3 overlap 为 0.5778。它是同源窗口的 construct alignment，
不能表述为外部泛化。

新增/结构化数据中：

- **GEFCom2012 Load**：pairwise agreement 0.7857，Top-3 overlap 1.0，三个
  supported capabilities 提供本轮最强的结构化对齐证据；
- **Swiss Hierarchical Demand**：pairwise agreement 0.5357，适合保留为 publisher-native
  严格层级压力集，但不应作为外部有效性的主证据；
- **M5**：pairwise agreement 0.2857，说明强业务语义不保证与当前合成机制排名对齐；
  保留作语义/日频反例，不用其单独支撑有效性声明；
- GEFCom2014 Solar/Wind 分别为 0.3929/0.5357，提示 NWP/任务发布语义值得保留，
  但跨真实窗口的模型排序仍有明显异质性。

## 4. E3：能力画像与协变量反事实

E3 使用全 320 groups 形成 440 个正式画像（8 模型 × 55 cells），共 704,000 条
sample mechanism scores；A/B 各 160 groups。所有 cell 均通过两块互斥、完备与源
`analysis_block_id` 对齐审计。`analysis_pool_index` 已修正为读取 v7 `pool_index`，
实际范围 0--319，704,000 行中没有 `-1`。

分块画像的平均 Spearman 为 0.8379，point-pair 方向一致率为 0.8799，4,620 个模型
对中方向冲突为 0，top-model 一致率为 70.9%；精确 rank vector 率为 18.8%。因此 E3
也应以能力形状、方向与实用等价为主，不以完整精确排名为主。

以下是对 55 个 cells 等权平均的描述性画像，不替代逐 cell 的 bootstrap 区间：

| Model | Mean ability | Mean MASE | Ability-rank wins |
|---|---:|---:|---:|
| TimesFM 2.5 | 0.6573 | 0.5575 | 4 |
| Chronos-2 | 0.6491 | 0.5577 | 5 |
| TiRex | 0.6353 | 0.5749 | 5 |
| toto2.0 | 0.6346 | 0.5535 | 18 |
| TabPFN-TS | 0.6301 | 0.5791 | 7 |
| Timer-3.5 | 0.5897 | 0.5749 | 7 |
| Timer-3.0 | 0.5635 | 0.5925 | 8 |
| Moirai 2 | 0.5534 | 0.6089 | 1 |

能力级最高等权均值不是同一个模型：

- common factor：Timer-3.0，0.5249；模型间差距很小；
- hierarchy：toto2.0，0.4386，TiRex/TimesFM 接近；
- covariate response：TimesFM，0.5235；
- multi-seasonal：TabPFN-TS，0.9252；
- nonlinear persistence：Timer-3.5，0.5555；
- predictable intermittency：toto2.0，0.9068；
- regime switching：toto2.0，0.6937；
- time-varying seasonality：TiRex，0.9192；
- trend：Chronos-2，0.3704。

这比单一总榜更有解释力：toto 的平均总分不是第一，但在 18/55 cells 获得 ability
第一，且优势集中在间歇性、状态切换和层级；TimesFM 的主要优势来自更均衡的结构能力
以及协变量响应。

协变量反事实只覆盖最终四个 covariate cells。TimesFM/Chronos/TiRex/TabPFN 分别执行
9,600/6,400/6,400/6,400 次 native counterfactual HTTP，失败均为 0；
Timer-3.5、Timer-3.0、Moirai、toto 因 E2 已按契约省略协变量，精确复用 intact
forecast，HTTP=0、effect=0。四个原生模型的等权 covariate ability 为
0.5235/0.5005/0.4808/0.4696；其 mean counterfactual effect MAE 为
0.4422/0.3639/0.3219/0.2009。这里的 0 是输入契约结论，不是“模型看到了协变量但没有
使用”。

E3 ablation manifest、summary 与 final manifest SHA-256 分别为：

- `1f4f14ca1c93249fd7c79dd790e2e48615e1e41336cee7a694c0f5f7a8c4c733`；
- `6930e1b50626bd66b28c1950fba684612eaedc7fba7aa96fb5567869d91d129f`；
- `6a167a7d347bf37dac3b1c81bdb350ce1a342662bc3a4b3ef6b94bc07b52fa4a`。

## 5. 数据集决策与下一步

1. 不再扩充主单变量集合；当前 6--7 个跨域 cells/能力已足够，新增预算应投向结构化
   能力与真实 forecast-vintage 证据。
2. GEFCom2012 Load 升为 v7 结构化主锚点；它同时提供 factor、严格 hierarchy 和安全
   日历 covariates，且本轮对齐最好。
3. Swiss 保留作 publisher-native hierarchy 与 NWP issue-time 质量锚点；其 factor 和
   covariate cell 因正式生成不可行已 fail closed，不做事后阈值放宽。
4. M5 保留为日频语义/多层层级压力集，不把其较差 alignment 当成删集理由，也不把它
   当作主要 construct-validity 证据。
5. 下一批优先审计 FreshRetailNet-50K 的 stockout censoring、活动发布时间和元数据层级；
   若这些证据不足，优先转向 Low Carbon London 的 day-ahead tariff，补一个真正有计划
   known-future signal 的新领域。UCI Hierarchical Sales 只作为低成本 PoC，Bike
   Sharing 仍只作 smoke test。

## 6. 主要 runtime 产物

- `runtime/paper_exp/v7/01_nine_capability_suite/`
- `runtime/paper_exp/v7/02_real_source_window_suite/`
- `runtime/paper_exp/v7/E2_dynamic_stability/`
- `runtime/paper_exp/v7/E2_dynamic_stability/split_bank_reliability/`
- `runtime/paper_exp/v7/E3_mechanism_fidelity/covariate_ablation_predictions/`
- `runtime/paper_exp/v7/E3_mechanism_fidelity/formal_analysis/`

这些目录均为 ignored runtime，不提交大文件；本记录提交可审计的规模、协议、结论和
manifest 哈希。
