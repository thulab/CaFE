# Paper v2 E4：合成能力画像到真实缺陷的迁移

日期：2026-07-17

## 执行完整性

- Selection freeze commits：`0cc2bae`，输入别名修复后最终边界为 `71f4a44`。
- 在最终边界之前没有发出真实模型 forecast 请求；第一次入口在读取 task 文件前失败。
- 9 个 frozen GIFT-Eval hourly profiles，4,192 个真实 tasks。
- 7 个基础模型各完成 4,192 条预测，共 29,344 条；naive 与 seasonal-naive 另各
  4,192 条。
- 7 个模型全部一次完成，failure 文件均为 0；replica/卡与 HTTP concurrency 均符合
  冻结配置。
- 七模型阶段耗时合计 688.544 秒。最慢为 timesfm2.5：291.311 秒（其中请求桶
  262.0 秒）。
- 全量后端测试：403 passed；E4 专项测试在输入别名修复后为 6 passed。

Runtime 目录：
`runtime/paper_exp/v2/E4_synthetic_real_transfer/`

精确身份：

- Selection manifest：
  `ffb4e8d63daa2538e4e06ab9f400d5ad449f479bd8baac0b62d45a9e57b1cd8a`
- Final manifest：
  `0ab1c6b666f2098c526d4ecdac2def1604771d757f043830bb15d28f5311015a`
- Summary：
  `a51950c697857c74856e799ee25a20247335b0945ab5de04dcc7fd3e8d675c10`

## 确认性结果

本轮结果不支持“当前 dataset-local synthetic capability 排序可以普遍预测 held-out
真实模型排序”的确认性主张。

| Predictor | Family-macro Kendall tau-b | 95% CI | Spearman | Pair direction | Centered Pearson |
|---|---:|---:|---:|---:|---:|
| v2 dataset-local capability | 0.0881 | [-0.2048, 0.3810] | 0.1016 | 0.5440 | 0.2595 |
| v2 global capability | 0.1095 | [-0.1513, 0.3741] | 0.1369 | 0.5548 | 0.2338 |
| v1 development global | 0.0554 | [-0.1398, 0.2540] | 0.0638 | 0.5277 | 0.1521 |
| v2 scalar macro | 0.0952 | [-0.2063, 0.4286] | 0.0774 | 0.5476 | 0.2769 |

Dataset-local predictor 相对忽略 capability 的 scalar baseline 的 tau 增量为
`-0.0071`，95% CI `[-0.1726, 0.1647]`。719 个非 identity capability-label 精确置换
的 null mean tau 为 `0.1376`，identity label 的单侧精确 `p=0.9972`。因此当前能力标签
没有提供超出模型总体强弱的增量证据。

Leave-one-family-out 的 local tau 范围为 `[-0.0110, 0.1952]`，结果也不具备跨 family
稳健性。

### Capability 分解

| Capability | High-loading families | Local Kendall tau-b |
|---|---:|---:|
| multi_seasonal | 5 | -0.1429 |
| predictable_intermittency | 4 | 0.1786 |
| regime_switching | 4 | -0.0595 |
| time_varying_seasonality | 6 | 0.1825 |
| trend | 3 | -0.1429 |

`nonlinear_persistence` 在所有 held-out profiles 的 train-only coordinate 均为 1，按
冻结规则未进入真实外部验证。

10 个由 E3 预注册的 pair hypotheses 中，4 个真实 family-macro gap 方向一致，没有一个
family-bootstrap CI 完全大于 0。相对较接近预期的两个结果是：

- predictable intermittency：Timer-3.0 vs toto2.0，真实 log-MASE gap
  `+0.0746`，95% CI `[-0.0048, 0.1553]`；
- time-varying seasonality：Timer-3.0 vs toto2.0，真实 log-MASE gap
  `+0.0562`，95% CI `[-0.0019, 0.1205]`。

multi-seasonal 的两个 pair、moirai2/Chronos-2 的 regime pair，以及
moirai2/Chronos-2 的 trend pair 在真实 high-loading families 上出现反向结果。

## 异质性发现

总体不成立不等于所有 cell 都无对应性：

- Bitbrains fast 的 time-varying seasonality、predictable intermittency、
  regime switching 分别达到 tau `0.9048`、`0.8095`、`0.6190`；
- Bitbrains rnd 的 predictable intermittency 和 time-varying seasonality 分别为
  `0.7143`、`0.6190`；
- Loop Seattle 的 predictable intermittency 与 time-varying seasonality 分别为
  `0.5238`、`0.4286`；
- 相反，KDD 的 predictable intermittency 为 `-0.6190`，multi-seasonal 与
  regime switching 均为 `-0.5238`；
- ETT1 的 multi-seasonal 为 `-0.5238`，trend/regime 均为 `-0.4286`。

因此目前更准确的结论是“迁移具有强 domain heterogeneity”，而不是“六维画像已获得普遍
真实外部效度”。

## Post-hoc 诊断（不改变确认性结论）

### 多能力混合

8 个进入主终点的 profiles 平均同时有 3.625 个 high-loading capabilities。
multi-seasonal 与 time-varying seasonality 的 high-loading profile Jaccard 为 0.75，
time-varying seasonality 与 regime switching 也是 0.75。真实 profile 的同一个模型
排序因此被重复用于多个高度重叠的 capability labels。

使用 train-only coordinate 对六个 dataset-local synthetic capability scores 做
`max(coordinate-1, 0)` 加权后，family-macro tau 为约 `0.150`，高于确认性
`0.088`，但仍然很弱；失败不只是逐 capability cell 对照过严造成的。

### 真实排名稳定性

按真实 series cluster 做 1,000 次 bootstrap，九个 profile 的 real-rank Kendall tau
中位数范围为 0.714–1.000。ETT2、SZ Taxi 和 Bitbrains rnd 的第一名因近似并列而较不
稳定，但 KDD、Loop、M_DENSE、Solar、Bitbrains fast 的整体排序较稳定。总体反向结果
不能仅用真实排名抽样噪声解释。

### 真实能力资格与生成机制不等价

一个明确的上游口径问题是：KDD、SZ Taxi、ETT1/2、Bitbrains 的
`change_point_shift_energy` 可映射到高 regime coordinate，但 train-only
`history-selected recurring clock` qualification rate 全部为 0。E4 selection 只使用
coordinate `>=3`，没有同时要求该 qualification，因此这些 cell 的高 feature energy
不等价于生成器定义的“历史重复且未来继续”的 predictable regime。

Post-hoc 去掉所有 regime cells 后 local tau 仅从 `0.0881` 变为约 `0.0992`，所以这不是
总体失败的唯一原因。更一般的问题是，除 regime 外的真实标签也主要基于特征量：
高 spike rate 未证明存在可外推的周期性脉冲时钟，高 modulation 未证明其平滑调制规律
可由历史外推，高 multi-period score 也未证明多频率在历史子区间中稳定。

## 论文决策与下一步

本 E4 必须保留为完整、未删案例的确认性结果，不应只抽取 Bitbrains/Loop 的正例来支持
普遍外部效度。按目前证据，论文不能声称 synthetic capability profile 已经普遍预测真实
缺陷。

下一版外部验证应先解决“real mechanism qualification”，而不是增加同协议的模型或
随机种子：

1. 将本轮九个 profiles 降为 development/audit set，不再用于新规则的确认性检验；
2. 在真实 training prefix 内对每个 capability 做 cross-fitted predictability gate：
   - trend：早期子窗拟合的趋势律能预测后期 history；
   - multi-seasonal：至少两个频率在多个 history folds 中保持周期、相位和显著性；
   - time-varying seasonality：早期 modulation law 能外推后期振幅/相位；
   - regime：recurring-clock pseudo-future qualification 必须为正并达到预设下限；
   - intermittency：早期 pulse period/phase 能预测后期 event mask；
   - nonlinear：非线性多滞后项必须获得 out-of-fold 增量预测收益；
3. 同时冻结 capability purity/overlap 规则，优先选择有单一或少数明确机制的真实 profiles；
4. 把 loading-weighted capability mixture 作为预注册 secondary predictor；
5. 在尚未查看模型结果的新 held-out families 上运行 E4-v3。

该结果对项目是有用的：生成器与 E3 能给出清晰、稳定的机制画像，但“真实特征统计量高”
到“真实序列遵循同一可预测机制”的桥梁尚未建立，这正是下一轮方法工作应补齐的部分。
