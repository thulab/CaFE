# Paper v8 common-factor / cross-series 稠密机制重测

## 范围

本轮只重测：

- `common_factor`
- `cross_series_dependence`

真实背景使用 GIFT-Eval `Electricity/H`，校准窗口为 256 个 L504 anchors。正式生成使用 64 个 master seeds、H48，并从同一母本切出 L96/L168/L336/L504。推理模型为：

- Chronos-2
- toto2.0
- tirex2
- timesfm2.5

TimesFM2.5 由适配器拆成独立单变量请求，其不能利用跨变量结构属于预期负控。

运行时产物：

```text
runtime/paper_exp/v8_test/dense_multivariate_retest/
  gift_electricity_h/
    01_calibration/
    02_generation/
    03_inference/seed_000000_000064/
    04_analysis/seed_000000_000064/
```

## covariate secondary 审计结论

当前 `covariate_response` secondary 不是与 primary 等难度的单一 family 替换：

- primary 是即时线性 `weather + event`；
- secondary 同时加入 `tanh`、weather 一阶滞后、事件三步 distributed lag；
- secondary 还额外放大作用尺度。

当前主校准坐标主要是 history 线性增量解释度。旧 pilot 中 primary 高强度目标超出 secondary 的可实现支持后，secondary 退回自己的相对 lambda 网格，因此不能解释为 matched-difficulty I3/I5 sensitivity。正式论文表在重新做尺度归一化前，不应把该 secondary 的分数差直接归因于 family 形状。

## 特征提取与生成参数映射

### common factor

- 主强度特征：history `pca_top1_explained`。
- 生成器实际支持经 12 个 calibration paths 扫描后，将 realized strength 等距放置为 I1–I5，再反解 lambda。
- 64-seed 实际均值：

| intensity | PCA top-1 share |
|---:|---:|
| I1 | 0.351 |
| I2 | 0.473 |
| I3 | 0.598 |
| I4 | 0.725 |
| I5 | 0.853 |

五档单调且间距没有在 I3–I5 饱和。

### cross-series dependence

旧 `cross_series_incremental_r2` 将所有变量、所有 lag 一次放进接近方阵的 ridge design；生成 lag 达到 48 以上时，真实边会被大量相关列稀释。v8 改为：

- 主强度坐标：`lead_lag_peak_abs`；
- incremental R²：保留为诊断，改成 compact own-history baseline 加单个 candidate source/lag 的 chronological holdout 搜索；
- 正确 driver、正确 lag、declared-edge holdout R² 和正控 effect recovery：作为独立 structural gate。

64-seed 实际强度均值：

| intensity | lead-lag peak abs |
|---:|---:|
| I1 | 0.376 |
| I2 | 0.546 |
| I3 | 0.711 |
| I4 | 0.853 |
| I5 | 0.999 |

这里 I5 接近结构上限，但 I1–I4 仍有清楚梯度。

## 调整后的生成器

### common factor

primary family：

```text
dense_dynamic_factor_with_joint_state_relay
```

- target dimension 从 3 增加到 5；
- 全路径包含确定性动态共同因子和异质局部确定性成分；
- 原 8–12 点短 code 改成 24/28/32 点平滑长观测块；
- 历史重复教授 `c = Bq`，响应是 rank-one loading 乘以由二维状态 `q` 决定的两基函数轨迹；
- strict pair 沿 protected channel 的 code row null direction 改变最终状态，所以 protected history 完全相同但 future 不同；
- 主表每 seed 只保留一个事实样本；strict pair 只在 sensitivity seeds 的 I5 生成。

### cross-series dependence

primary family：

```text
dense_delayed_linear_scm
```

- driver history 使用平滑 random-knot path 形成连续稠密激励，不再使用三个教学脉冲和边界第 4 个事件；
- responders 固定使用混合符号 `[+1,-1,...]`；
- `lag = horizon = 48`，future responder 的 48 点全部由最后 48 点可见 driver 覆盖；
- paired alternatives 来自相同稠密路径生成器，并匹配均值、标准差和光滑边缘，不是孤立符号翻转异常；
- 主表为事实预测，strict pair 只进入 I5 审计。

## 输入消融

主事实题之外新增 `multivariate_input_ablation`：

- common-factor：保留 protected target history，只用相同 intensity 的另一 seed 替换辅助通道 history，并做逐通道 affine 均值/标准差匹配；
- cross-series：保留 responder histories，只替换 causally forecast-covering 的最后 48 点 driver block，并匹配该段均值/标准差；
- future 始终使用原 clean latent；
- Oracle control 和 ablation 使用 control 选出的同一个 context。

因此消融只评分历史未改的 focal targets，不能用所有通道总 MASE。

## 生成回验

正式生成：

| artifact | master count |
|---|---:|
| clean | 736 |
| input ablation | 640 |
| observation-noise robustness | 48 |
| total | 1,424 |
| context views / model | 5,696 |

12 组抽样 I5 strict pairs 全部通过：

| gate | worst case |
|---|---:|
| common joint holdout R² | 1.000 |
| common best single-channel holdout R² | 0.208 |
| cross declared responder holdout R² | 0.996 |
| cross positive-control effect NRMSE | 0.0136 |

说明生成器内部结构可从 history 恢复，模型不响应不能归因于 future 不可预测。

## 模型结果

### 主事实表

Fixed L504：

| capability | model | MASE | factual mechanism error |
|---|---|---:|---:|
| common factor | toto2.0 | 0.456 | 0.499 |
| common factor | timesfm2.5 | 0.476 | 0.477 |
| common factor | tirex2 | 0.481 | 0.481 |
| common factor | Chronos-2 | 0.482 | 0.500 |
| cross-series | Chronos-2 | 0.423 | 0.756 |
| cross-series | tirex2 | 0.431 | 0.829 |
| cross-series | timesfm2.5 | 0.437 | 0.842 |
| cross-series | toto2.0 | 0.438 | 0.823 |

common factual mechanism error 是 `common_component_nmae`；cross 是 I5 `responder_normalized_mae`。common 的单变量负控也能预测动态共同成分，因此 factual reconstruction 不能单独证明模型使用了联合输入。

所有模型均非平线：

| capability | future curve correlation range | flat forecast rate |
|---|---:|---:|
| common factor | 0.785–0.803 | 0 |
| cross-series | 0.557–0.591 | 0 |

主表 MASE 明显优于 last/seasonal naive 的 0.80–1.07，但没有接近 0，难度处于可用区间。

### 跨变量输入消融

Fixed L504：

| capability | model | focal clean | focal ablated | relative increase |
|---|---|---:|---:|---:|
| common factor | Chronos-2 | 0.512 | 0.568 | +10.9% |
| common factor | toto2.0 | 0.475 | 0.478 | +0.4% |
| common factor | tirex2 | 0.507 | 0.509 | +0.3% |
| common factor | timesfm2.5 | 0.504 | 0.504 | 0.0% |
| cross-series | Chronos-2 | 0.509 | 0.567 | +11.5% |
| cross-series | toto2.0 | 0.544 | 0.545 | +0.3% |
| cross-series | tirex2 | 0.554 | 0.555 | +0.2% |
| cross-series | timesfm2.5 | 0.560 | 0.560 | 0.0% |

Oracle shared-context 下，Chronos 的增幅分别为 +20.8% 和 +12.2%；TimesFM 仍为 0。结果能区分明显使用、极弱使用和不可能使用跨变量输入的模型。

### I5 strict counterfactual audit

Fixed L504 effect NRMSE：

| capability | Chronos-2 | toto2.0 | tirex2 | timesfm2.5 |
|---|---:|---:|---:|---:|
| common factor | 0.990 | 1.001 | 0.999 | 1.000 |
| cross-series | 0.887 | 1.000 | 0.996 | 1.000 |

严格 state-relay / paired effect 对当前模型仍然过难。它可以报告“模型是否真的随反事实联合输入改变预测”，但不适合作为主能力分。

### 32-seed 批次稳定性

- common fixed-L504 accuracy：两个 batch 均由 toto2.0 排名第一；平均相对得分差 3.4%，tau-b 0.667。
- common factual mechanism：模型差异很小，batch 排名不稳定，tau-b -0.333；不能把它解释为跨变量能力排名。
- cross fixed-L504 accuracy：两个 batch 均由 Chronos-2 第一；平均相对得分差 1.6%，tau-b 0.667。
- cross factual mechanism：两个 batch 均由 Chronos-2 第一；平均相对得分差 1.7%，tau-b 1.0。
- input-ablation delta：两个 32-seed batch 中，Chronos 在 common/cross 都稳定第一；其他原生模型的 delta 接近 0，其细小次序没有机制意义。

## 结论

1. 新 common/cross 样本不再依赖孤立边界事件，强度可控，生成正控可恢复，模型曲线也不再一平如水。
2. common 的事实动态因子预测和“是否使用联合通道”必须分开报告；后者由 protected-target input ablation 提供。
3. cross 的 horizon-aligned dense SCM 已能让 Chronos 显著响应；Toto2/TiRex2 的响应接近 0 是本轮观察到的模型行为，不应继续通过模型特供型出题强行放大。
4. strict counterfactual 保留为 I5 子集审计，不进入主能力分。
5. v8 正式结果应同时展示事实误差、结构恢复指标和跨变量消融 delta，不合成任意单一总分。

## 验证

```text
105 passed
```

定向测试覆盖生成器、特征提取、v8 pipeline、输入消融、共享 Oracle context、tail shard 刷新和模型响应分析。未运行与 v8 无关的完整后端测试。
