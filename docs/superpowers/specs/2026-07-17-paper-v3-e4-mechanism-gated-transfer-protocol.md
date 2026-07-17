# Paper E4-v3：机制门控后的合成—真实缺陷迁移

日期：2026-07-17

## 1. 研究问题与版本边界

E4-v3 检验：合成 benchmark 得到的模型能力画像，能否预测具有相同
**机制对齐预测行为**的真实窗口上的模型相对表现。

E4-v2 以 profile 级特征统计量筛选能力，已经作为开发性负结果保留，不能再充当独立确认
集。E4-v3 作如下改变：

1. 能力资格下沉到 `real window × capability`；
2. 资格只由模型可见的 504 点 context 内滚动伪未来确定；
3. 使用 GIFT 官方 test tail 之前、此前从未请求过模型推理的单个 validation horizon
   作为新的时间留出结果；
4. 门控开发和阈值冻结只读取 E2 合成 context，不读取 E4-v2 或 E4-v3 的真实模型成绩；
5. E4-v2 九个 profile 仍提供 dataset-local synthetic predictor，但它们的 E4-v2 test
   成绩不进入 E4-v3。

这里的 gate 不是因果机制识别，也不要求真实曲线复刻生成器。它要求一个抽象能力 probe
在未参与拟合的伪未来上稳定优于包含 nuisance 结构的匹配基线。论文中统一称为
`mechanism-aligned predictive capability fingerprint`。

## 2. 门控开发冻结

门控使用 E2-v2 合成样本：

- round 1–3、每格 sample index 0 与 8：阈值校准；
- round 4–5、同样的 sample index：不重拟合阈值的独立审计；
- 正样本：source capability 与 probe capability 相同且 intensity ≥ 3；
- 负对照：其他五类生成器的全部 intensity，以及同能力 intensity 1；
- 每条 504 点 context 切出四个互不重叠的 48 点伪未来；
- 真实 benchmark future 不在门控 API 中，也不允许被读取。

六种 probe 的抽象契约为：

| capability | nuisance-matched baseline | capability probe |
|---|---|---|
| trend | 固定平滑季节项 | history-only 选择记忆长度的一/二次趋势外推 |
| multi-seasonal | 主周期、慢周期与线性项 | 增加 `P/2` 与 `2P` 的稳定周期项 |
| time-varying seasonality | 固定 `P`、`2P` 季节项 | 增加平滑幅相调制对应的两侧边带 |
| regime switching | 趋势与候选 clock 的平滑谐波 | 历史选择、伪未来延续的重复二状态 clock |
| nonlinear persistence | 去除平滑 nuisance 后的线性多滞后递推 | 增加稳定非线性滞后响应 |
| predictable intermittency | 趋势与平滑周期项 | 历史选择、伪未来延续的窄周期脉冲 clock |

每个 gate 同时要求：

- 至少三个有效伪未来 fold；
- 冻结的 pooled relative MAE gain 下界；
- 正增益 fold 比例；
- construction-support 指标；
- 参数、方向或事件时钟稳定性；
- 错相位伪未来检验达到冻结门限。

阈值由预声明的 TPR/FPR/dose-response 目标确定。若某能力未通过合成独立审计的校准约束，
它不得进入 E4-v3 确认性主分析。当前 `nonlinear_persistence` 属于该情况：作为门控开发
结果完整报告，但真实迁移结论记为“gate 未验证”，而不是通过放宽门限补入主表。

## 3. 新的真实时间留出

沿用九个 dataset-local synthetic profiles 与固定形状：

```text
context = 504
horizon = 48
season_length = 24
target_dim = 1
```

对每条 GIFT series/channel，只使用官方 test tail 之前的一个 validation origin：

```text
validation_origin = series_length - official_test_tail_steps - 48
```

这正是 GIFT `training_dataset` 与 `validation_dataset` 之间的 48 点区间。E2-v2 的
conditioning artifact 已明确排除该区间和全部 test tail；E4-v2 也没有对此 origin
请求过预测。

缺失值、MASE denominator 与 timestamp 规则沿用 E4-v2：

- context 至少 50% observed，并仅在 context 内插值；
- future 不插值，至少 24/48 observed；
- MASE 以 context 的 lag-24 seasonal difference 为 scale；
- 近常数、scale 不稳定的窗口拒绝；
- native multivariate row 继续按 GIFT `to_univariate=True` 语义拆 channel。

门控先扫描所有合格 validation context，再作模型无关的确定性选择。一个
`profile × capability` cell 至少需要 12 条、来自至少 12 条 series/channel 的合格
窗口，最多等距保留 160 条。推理 task 是所有确认性 cell 所选窗口的并集，因此同一窗口
即使承载多个真实机制也只请求一次。

## 4. 主分析与稳健性分析

### 4.1 确认性 cell

主分析使用 inclusive mechanism gate：窗口只要通过能力 `c` 的冻结 gate，就可进入
`profile × c`。真实机制可以重叠，不强迫互斥。

同时冻结两个稳健性分析：

1. `exclusive`：只保留恰好通过一个已验证能力 gate 的窗口；
2. `fingerprint-weighted`：在 inclusive 窗口内按连续 gate weight 计算模型 MASE。

exclusive cell 同样要求至少 12 条与 12 条 series；不满足时报告缺失，不作补抽样。

### 4.2 合成 predictor 与真实分数

合成 predictor 沿用 sealed E3-v2/E3-v1：

- `v2_dataset_local_capability`（确认性主 predictor）；
- `v2_global_capability`；
- `v1_development_global_capability`；
- `v2_scalar_macro`。

对模型 `m`、profile `p`、capability `c`：

```text
R(m,p,c) = log(real gated MASE(m,p,c) /
                 real gated MASE(seasonal-naive,p,c))
```

每个 cell 在七个模型上比较 synthetic predictor 与 `R`，报告 Kendall tau-b、
Spearman rho、pair-direction concordance、centered Pearson 与 z-RMSE。

主终点为 `v2_dataset_local_capability` 的 family-macro Kendall tau-b。宏平均顺序固定
为 capability within profile、profile within family、family；family-cluster
bootstrap 5000 次。还报告：

- 相对 scalar macro 的配对 tau 增量；
- capability-label exact permutation null；
- leave-one-family-out；
- 分 capability 结果；
- inclusive、exclusive 与 fingerprint-weighted 三种选择的差异。

### 4.3 解释边界

支持论文主张需要主 tau 为正、区间不与大幅负相关相容，并且 capability-aware predictor
相对 scalar macro 和错标签 null 有改善。局部正 cell 只能称为案例证据。一个 gate
通过也只表明该能力 probe 在可见 history 中具有可外推 headroom，不能把它写成真实
系统的因果生成机制。

## 5. 冻结与产物

目录固定为：

```text
runtime/paper_exp/v3/E4_mechanism_gated_transfer/
```

`prepare` 必须先生成：

- 全部 context 的 gate diagnostics 与 decisions；
- `qualified_cells.csv`、`cell_task_map.csv`；
- `tasks.jsonl` 与 task/selection manifest；
- selection receipt candidate。

只有门控阈值、protocol、runner 与 selection receipt 全部提交后，`infer` 才允许访问
推理服务。`analyze` 只能读取已冻结 task、predictor 与 prediction 文件。任何看过
E4-v3 validation 成绩后新增的阈值、数据删除、subgroup 或案例都必须标为探索性。
