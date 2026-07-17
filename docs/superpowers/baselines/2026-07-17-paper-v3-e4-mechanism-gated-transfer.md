# Paper E4-v3：机制门控后的合成—真实缺陷迁移

日期：2026-07-17

## 结论

E4-v3 得到一个有边界的部分正结果：

- 机制门控后，合成分数与真实窗口模型排序呈稳定正相关；
- 但当前结果主要仍可由模型的整体强弱解释，尚不能证明合成 benchmark 已识别出可迁移的
  **能力特异性缺陷**。

因此，本实验支持“合成评测包含真实表现信息”，但不支持更强的“正确能力标签比模型整体
能力更能预测真实缺陷”主张。

## 冻结协议

门控、真实窗口选择和模型推理在读取 E4-v3 结果前已由提交
`4029493ac10ec0663022dcbe03ec3fc1c28a5b85` 冻结。核心约束为：

- gate 只读取 504 点 context 内四个互不重叠的 48 点伪未来；
- E4-v3 outcome 使用 GIFT 官方 test tail 前、此前未推理的单个 validation horizon；
- `context=504`、`horizon=48`、单变量、小时频率；
- 每个 `profile × capability` cell 至少 12 条且来自至少 12 条 series；
- 主分析为 inclusive gate，exclusive 与 fingerprint-weighted 为冻结稳健性分析；
- 主 predictor 为 E3-v2 的 dataset-local capability score；
- 主终点为 family-macro Kendall tau-b，使用 5,000 次 family-cluster bootstrap；
- 使用 capability-label exact permutation 和 scalar macro predictor 检验能力特异性。

完整协议见
`docs/superpowers/specs/2026-07-17-paper-v3-e4-mechanism-gated-transfer-protocol.md`。

## Gate 校准与真实支持

门控在 E2-v2 合成数据上以 round 1–3 校准、round 4–5 独立审计。五个能力通过冻结
审计；`nonlinear_persistence` 的审计 TPR 仅为 0.0648，因此没有通过后放宽阈值。

在 4,063 个合格 GIFT validation contexts 上重新筛选后：

| capability | confirmatory cells | families | selected memberships | 状态 |
|---|---:|---:|---:|---|
| `multi_seasonal` | 4 | 4 | 205 | 有真实支持 |
| `predictable_intermittency` | 2 | 1 | 92 | 有限支持，仅一个 family |
| `trend` | 3 | 2 | 295 | 有限支持 |
| `time_varying_seasonality` | 0 | 0 | 0 | gate 已验证，当前 slices 无支持 |
| `regime_switching` | 0 | 0 | 0 | gate 已验证，当前 slices 无支持 |
| `nonlinear_persistence` | — | — | — | gate 未通过独立审计 |

全部候选中有 18 个 context 同时通过两个能力 gate；确定性限额抽样后保留了其中 10
个。9 个 confirmatory cells 共含 592 个 membership，并集为 582 条唯一任务。主分析
允许这种真实模式重叠。

## 推理完整性

七个模型全部按冻结 replica 与 HTTP 并发配置运行：

| model | replicas/GPU | HTTP concurrency | success |
|---|---:|---:|---:|
| Timer-3.5 | 1 | 64 | 582/582 |
| Timer-3.0 | 1 | 32 | 582/582 |
| Chronos-2 | 4 | 32 | 582/582 |
| moirai2 | 2 | 16 | 582/582 |
| toto2.0 | 2 | 16 | 582/582 |
| timesfm2.5 | 8 | 32 | 582/582 |
| tirex2 | 1 | 32 | 582/582 |

两种基线也各产生 582 条结果。最终共有 5,238 条 observation，七个模型的失败文件均为
空。

## 确认性结果

### 主终点与稳健性

| scope | cells | families | capability tau-b | 95% CI | scalar tau-b | delta vs scalar |
|---|---:|---:|---:|---:|---:|---:|
| inclusive | 9 | 5 | 0.2984 | [0.1619, 0.4159] | 0.3302 | -0.0317 |
| exclusive | 8 | 5 | 0.3095 | [0.1619, 0.4429] | 0.3333 | -0.0238 |
| fingerprint-weighted | 9 | 5 | 0.2778 | [0.1048, 0.4508] | 0.3095 | -0.0317 |

主结果的 paired delta 95% CI 为 `[-0.0762, 0.0127]`。门控方式改变后，正相关方向
保持不变，但 capability-aware predictor 始终没有超过 scalar macro。

正确 dataset-local capability label 的 exact-permutation 结果为：

```text
observed tau       = 0.2984
wrong-label mean   = 0.3304
wrong-label q95    = 0.4114
p(one-sided)       = 0.7500
```

这与 scalar 对照给出相同结论：当前正相关主要反映跨能力共享的模型强弱，而不是正确
能力标签带来的额外解释力。

### 分能力

| capability | families | Kendall tau-b | Spearman rho | pair direction |
|---|---:|---:|---:|---:|
| `multi_seasonal` | 4 | 0.3333 | 0.4375 | 0.6667 |
| `predictable_intermittency` | 1 | 0.6667 | 0.7679 | 0.8333 |
| `trend` | 2 | 0.2619 | 0.3482 | 0.6310 |

`predictable_intermittency` 的数值最高，但两个 cell 都来自 Bitbrains，不能视为跨数据族
复现。`multi_seasonal` 覆盖四个独立 family，是目前最可靠的分能力估计。

### 模型层面的匹配与失配

以下为九个 cell 上的平均名次，名次越小越好：

| model | synthetic mean rank | real mean rank | 观察 |
|---|---:|---:|---|
| toto2.0 | 1.78 | 1.89 | 整体强弱预测准确 |
| Chronos-2 | 1.33 | 3.33 | 合成侧系统性高估 |
| Timer-3.5 | 4.33 | 3.00 | 合成侧系统性低估 |
| tirex2 | 4.56 | 4.22 | 接近 |
| timesfm2.5 | 4.11 | 4.33 | 接近 |
| moirai2 | 5.78 | 4.67 | 合成侧偏低估 |
| Timer-3.0 | 6.11 | 6.56 | 整体弱项预测准确 |

较清晰的局部匹配包括：

- 两个 Bitbrains intermittency cells 中，合成与真实都将 toto2.0 排在第一、
  Timer-3.0 排在最后，cell tau 分别为 0.6190 和 0.7143；
- Bitbrains-fast multi-seasonal 的 cell tau 为 0.6190；
- M_DENSE multi-seasonal 的 cell tau 为 0.4286。

明显失配包括：

- Bitbrains-rnd trend 的 tau 为 -0.0476；
- Solar multi-seasonal 的 tau 仅为 0.0476；
- Chronos-2 在合成侧平均排名第一附近，但真实平均仅排名第三；
- Timer-3.5 的真实平均排名显著好于合成画像。

这些失配说明目前不能把某个模型的合成能力短板直接解释为真实能力缺陷。

## 与 E4-v2 的关系

E4-v2 的主 tau 为 0.0881，95% CI `[-0.2048, 0.3810]`；E4-v3 为 0.2984，区间完全
为正。该变化与“机制 gate 比静态特征统计量更接近所需的模式概念”一致。

但 E4-v3 同时改变了真实时间留出和 eligible cell 集合，因此两版不是同 outcome 上的
随机对照消融，不能把 tau 的提升全部因果归功于 gate。它只能作为支持继续采用
mechanism-aligned gate 的版本间证据。

## 论文表述边界

可以写：

> 在历史内机制门控选出的真实窗口上，合成评测排名与模型真实表现呈稳定正相关。

当前不能写：

> 合成 benchmark 已经证明能识别并迁移模型的能力特异性缺陷。

后一句还需要同时满足：

1. capability predictor 显著优于 scalar/global model-quality 对照；
2. 正确 capability label 优于错标签 null；
3. 每个能力由多个独立真实数据 family 支持；
4. `time_varying_seasonality`、`regime_switching` 和
   `nonlinear_persistence` 不再缺失。

## 下一步建议

下一次确认性实验不应继续扩大同一批 GIFT slices 后再挑窗口，而应：

1. 预先补充具备重复 switch、平滑幅相调制和稳定非线性递推行为的真实数据 family；
2. 每个 capability 至少准备 4 个独立 family，避免 Bitbrains 一族决定能力结果；
3. 先在独立合成 audit 上修复或淘汰 nonlinear gate，再看真实结果；
4. 把能力特异性 interaction 设为主终点，例如先移除 model general-skill 与 dataset
   difficulty，再检验 synthetic capability residual 是否预测 real residual；
5. dataset-local synthetic predictor 需要提高估计精度；本轮 global capability
   predictor 的 tau 为 0.3841，而 dataset-local 仅为 0.2984，说明当前 conditioning
   可能增加了方差。

以上都应在新增真实模型结果前冻结，不应在本轮 9 个 cell 上继续调 gate。

## 产物与校验

运行目录：

```text
runtime/paper_exp/v3/E4_mechanism_gated_transfer/
```

关键产物：

- `summary.json`：`83c25154ee89293cc4c7d6ebb9ef56890fc7ea58712f472c1e89446176948683`
- `manifest.json`：`4db164967a3b6532fc883617d5b63150fca7e89b652549fc7182bbe2ea51d4f3`
- `model_status.json`：`481078351c9fa7eead03f51312d96b97cc35f2cb5a5e47e41f3586d6341b2512`
- `report.md`：`feae00c9ee904dba80e2edb4304831d5d62021e8459876bb7c012d8e7b5f7b17`
- 4 张主图均保存为 PNG、SVG 和 PDF。

`runtime/` 按仓库约定不提交；本文件是论文实验结果的版本控制摘要。
