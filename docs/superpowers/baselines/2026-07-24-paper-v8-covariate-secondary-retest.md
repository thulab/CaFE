# Paper v8 covariate secondary 修复与复测

## 范围

- 数据集：`gift_electricity_h`
- seeds：64
- secondary sensitivity seeds：12，仅 I3/I5
- 模型：Chronos-2、toto2.0、tirex2、timesfm2.5
- context：固定 L504 与 oracle-context
- 本轮只生成、推理和分析 `covariate_response`

## 原问题

旧 secondary 同时改变了三类因素：

1. 将即时线性响应换成非线性与 distributed-lag 响应；
2. 将 effect 额外乘以 `1.7`；
3. 将 primary 的 LDS weather driver 换成近周期 spline driver。

同时，两族用偏向当前线性响应的 `covariate_incremental_r2` 作为共同剂量坐标。旧校准中 primary 支持约为 `0.018–0.568`，secondary 仅为
`0.072–0.238`，因此 secondary 无法反解 primary 的 I1–I5，只能退回自己的 relative lambda grid。

spline driver 还产生了明显的 MASE denominator 混杂。matched seeds 的 seasonal MASE scale 中位数如下：

| intensity | primary | old secondary |
|---:|---:|---:|
| I3 | 0.545 | 0.091 |
| I5 | 0.730 | 0.105 |

因此相近的绝对误差会在旧 secondary 上被放大约 5–7 倍。

## 最终修复

- primary 数值响应路径保持不变。
- matched seed 的两族共享完全相同的 weather、event、baseline、符号和 effect 参数。
- secondary 只把响应律替换为：
  - `0.60 * weather`
  - `0.25 * tanh(weather)`
  - `0.15 * tanh(lag1(weather))`
  - event 使用 `[0.50, 0.30, 0.20]` distributed lag。
- secondary raw response 在 history 上仿射匹配同 seed primary response 的均值和标准差。
- 删除额外 `1.7` 放大。
- I1–I5 改用生成器已知的 history
  `covariate_effect_variance_share` 标定。
- `covariate_incremental_r2` 保留为线性可解释性审计，不再承担跨 family 剂量匹配。

新校准中两族 21 点 response curve 的最大绝对差为
`1.1e-16`，selected lambda 的最大差为 `2.2e-16`，secondary 状态为
`matched_primary_target_values`。

生成 gate 对 48 个 secondary members 全部通过：

| 检查 | 最大差异 |
|---|---:|
| covariate path | 0 |
| effect dose | 2.2e-16 |
| effect strength | 0 |
| MASE scale relative difference | 6.12% |

新的 MASE scale 中位数为：

| intensity | primary | fixed secondary |
|---:|---:|---:|
| I3 | 0.527 | 0.518 |
| I5 | 0.730 | 0.723 |

## 主表保持性

固定 L504 的 clean-primary MASE：

| model | before | after |
|---|---:|---:|
| timesfm2.5 | 0.546 | 0.547 |
| Chronos-2 | 0.633 | 0.634 |
| tirex2 | 0.963 | 0.966 |
| toto2.0 | 1.207 | 1.207 |

排序和机制指标均不变，说明修复没有实质改变 primary 主任务。

## Secondary 对比

固定 L504、相同 seed/intensity control：

| model | old secondary MASE | fixed MASE | old effect NRMSE | fixed NRMSE |
|---|---:|---:|---:|---:|
| Chronos-2 | 2.554 | 0.515 | 0.711 | 0.445 |
| timesfm2.5 | 3.505 | 0.290 | 1.334 | 0.244 |
| tirex2 | 2.788 | 0.629 | 0.627 | 0.573 |
| toto2.0 | 3.962 | 0.893 | 1.000 | 1.000 |

相对 clean-primary control 的新变化：

| model | MASE delta | mechanism delta |
|---|---:|---:|
| Chronos-2 | +19.8% | +33.8% |
| timesfm2.5 | +36.1% | +106.2% |
| tirex2 | -5.8% | -0.0% |
| toto2.0 | -5.2% | +0.0% |

## 解释

修复前的数倍退化主要是 family、效应尺度和 MASE denominator 三者混杂，不能解释为模型对非线性响应的真实敏感性。

修复后：

- timesfm2.5 仍有最好的 absolute MASE 和 effect recovery，但对响应律变化最敏感；
- Chronos-2 也能恢复 secondary effect，但相对 primary 有中等退化；
- tirex2 的 effect NRMSE 在两族间基本不变；
- toto2.0 当前适配器明确省略 unsupported covariates，因此两族 effect NRMSE 均为 1，符合预期。其 secondary MASE 小幅下降不是协变量能力改善。

现在 secondary 可以解释为“相同 driver、背景和 history effect dose 下，对非线性与 distributed-lag 响应律的 family sensitivity”，不再是 OOD driver/尺度混合压力测试。

## 验证

- 生成：688 clean masters，48 robustness masters
- 推理：每模型 2944 views，共 11776 predictions
- 缺失预测：0
- 生成验证：通过
- v8 定向单测：111 passed
- 未运行完整后端测试
