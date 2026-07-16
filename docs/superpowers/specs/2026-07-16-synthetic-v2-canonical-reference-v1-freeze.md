# Synthetic v2 canonical reference paper-v1 冻结协议

日期：2026-07-16

## 目的

本文档记录 paper-v1 canonical intensity 的 development reference corpus、held-out corpus、窗口协议、能力资格审计与资产身份。数据集清单在模型评测前冻结；后续不得根据模型成绩反向增删 development profile，任何变更必须发布新的 `canonical_scale_id`。

## 冻结标尺身份

- `canonical_scale_id`: `synthetic-v2-paper-v1-frozen-2026-07-16`
- 五档默认分位：q20 / q35 / q50 / q70 / q90
- `nonlinear_persistence`：q55 / q62.5 / q70 / q80 / q90。低于真实中位数的 multi-lag gain 无法在所有 profile 中和季节性 nuisance 造成的估计器底噪区分，因此不作为强度档位。
- 聚合：每个 capability 内先求各 profile 的局部分位曲线，再对 profile 等权逐坐标取中位数。
- 分辨率：保留上述原始曲线的首尾端点；若相邻档小于原始 q-low--q-high range 的 10%，对中间档做最小间隔投影。artifact 同时保存投影前后的值，避免形式递增但实验上不可区分。
- 一个 dataset family 在同一 capability 中最多贡献一条曲线；同一数据集的额外 context/horizon 不重复加权。
- 新增 profile 只参与 canonical target 定义，不自动成为在线生成 bucket。在线 bucket 的 nuisance、feature gate 与 near-distance gate 仍独立拟合。

## Development reference corpus

### 5 个通用单变量 capability

下表只贡献 `trend`、`multi_seasonal`、`time_varying_seasonality`、`nonlinear_persistence` 和 `predictable_intermittency` 的 canonical 曲线。它们仍可作为在线 `regime_switching` conditioning profile，但不参与 regime target 拟合。

| Profile | Domain | Frequency | Context / horizon | Season | Source role |
| --- | --- | --- | ---: | ---: | --- |
| `m4_hourly_daily_168ctx` | Econ/Fin | H | 168 / 24 | 24 | existing development |
| `electricity_hourly_daily_168ctx` | Energy | H | 168 / 24 | 24 | existing development |
| `traffic_hourly_daily_168ctx` | Transport | H | 168 / 24 | 24 | existing development |
| `gift_hospital_monthly_60ctx_12h` | Healthcare | M | 60 / 12 | 12 | new canonical-only |
| `gift_jena_weather_hourly_168ctx_24h` | Nature | H | 168 / 24 | 24 | new canonical-only |
| `gift_bizitobs_l2c_hourly_168ctx_24h` | Web/CloudOps | H | 168 / 24 | 24 | new canonical-only |

### `regime_switching`

通用 forecast 数据并不天然包含“历史中重复出现且在预测期继续出现”的切换。以同一资格审计抽样扫描后，M4 Hourly、Electricity、Traffic 的合格率分别约为 2.3%、3.0%、0%，其余候选多为 0；因此它们不能决定 regime 标尺。

| Profile | Domain | Frequency | Context / horizon | Clock | Parameter / qualified | Source role |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `uci_hydraulic_eps1_420ctx_60h` | Industrial hydraulic test rig | 1 s | 420 / 60 | 60 s load cycle | 262 / 164 (62.6%) | canonical-only |
| `skchange_hvac_unit0_504ctx_144h` | Deployed HVAC vibration | 10 min | 504 / 144 | 2-day on/off clock | 78 / 33 (42.3%) | canonical-only |

UCI Hydraulic 的 `EPS1` 原始 100 Hz motor-power 信号按官方每个 60 秒 load cycle 分块均值为 1 Hz，再按 cycle 原顺序连接。skchange HVAC unit 0 来自 Soundsensing 提供的真实设备数据；规则化到 10 分钟网格时仅线性补齐 10 / 4320 个短缺口。来源说明见 [UCI Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition%20monitoring%20of%20hydraulic%20systems) 与 [skchange HVAC loader](https://github.com/NorskRegnesentral/skchange/blob/f209def94199607b11b1ae9b3108d80e3e87e624/skchange/datasets/_data_loaders.py)。

许可：UCI 页面标注 CC BY 4.0；skchange 仓库为 BSD-3-Clause。原始数据仅作为本地校准资产，不随 benchmark 仓库提交。

regime 窗口必须同时满足：候选 clock 只用 history 选择；history incremental R²、状态幅度、切换持续性与状态覆盖过门；预测期不重新拟合，并继续获得 MSE gain、双状态覆盖和至少一次切换。一次性 change point、稀疏 burst 和平滑季节调制均不得通过。canonical 分位数只统计通过该资格审计的窗口。

### `common_factor`

所有 profile 固定为 3-target，避免 PCA1 因 target dimension 不同产生系统偏移。

| Profile | Domain | Frequency | Context / horizon | Source role |
| --- | --- | --- | ---: | --- |
| `electricity_hourly_panel_168ctx` | Energy | H | 168 / 24 | existing development |
| `traffic_hourly_panel_168ctx` | Transport | H | 168 / 24 | existing development |
| `gift_jena_weather_hourly_panel_168ctx` | Nature | H | 168 / 24 | new canonical-only; native channels 0--2 |
| `gift_bizitobs_l2c_hourly_panel_168ctx` | Web/CloudOps | H | 168 / 24 | new canonical-only; native channels 0--2 |

### `hierarchical_coherence`

Nixtla profile 使用官方 `agg_mat.csv` 中恰含两个 bottom descendants 的父节点；两个 child 是官方 bottom series，parent 由二者严格相加。所选父节点的 bottom supports 在各数据集内互不重叠。

| Profile | Domain | Frequency | Context / horizon | Source role |
| --- | --- | --- | ---: | --- |
| `m5_daily_hierarchy_365ctx_28h` | Retail | D | 365 / 28 | existing development |
| `nixtla_labour_monthly_hierarchy_60ctx_12h` | Labour | M | 60 / 12 | new canonical-only |
| `nixtla_tourism_large_monthly_hierarchy_60ctx_12h` | Tourism | M | 60 / 12 | new canonical-only |

### `covariate_response`

| Profile | Domain | Frequency | Context / horizon | Known-future signal | Source role |
| --- | --- | --- | ---: | --- | --- |
| `m5_daily_covariate_365ctx_28h` | Retail | D | 365 / 28 | events, SNAP, price | existing development |
| `gefcom2014_load_hourly_covariate_168ctx_24h` | Energy load | H | 168 / 24 | temperature forecasts | existing development |
| `gefcom2014_solar_hourly_covariate_168ctx_24h` | Solar generation | H | 168 / 24 | 12 ECMWF NWP variables | new canonical-only |

## Held-out corpus

以下 dataset family 不参与 `T_{c,k}` 拟合：

- GIFT-Eval：`solar`、`covid_deaths`、`kdd_cup_2018_with_missing`、`restaurant`、`hierarchical_sales`、`LOOP_SEATTLE`、`SZ_TAXI`、`M_DENSE`、`ett1`、`ett2`、`bitbrains_fast_storage`、`bitbrains_rnd`、`bizitobs_application`、`bizitobs_service`，以及其余未列入 development 的 family。
- Nixtla hierarchy：`Traffic`、`Wiki2`、`TourismSmall`。
- GEFCom2014：Wind track。
- skchange HVAC：unit 1，仅作为独立诊断/held-out，不参与 scale fitting。

这些数据只用于 scale coverage、真实—合成能力对应性和最终模型实验。GIFT-Eval 的官方 test windows 不得进入任何生成器参数、feature gate、near-distance gate 或 canonical target 拟合。

对列入 development 的 GIFT-Eval family，也先按冻结代码的 short-term 协议计算
`windows = clip(ceil(0.1 × min_series_length / prediction_length), 1, 20)`，并在候选窗口构造前删除尾部 `prediction_length × windows`。本次实际截尾为 Hospital 12 点、Jena-H 912 点、BizITObs-L2C-H 288 点。截尾后三者的单变量 parameter windows 分别为 264、236、189；Jena/BizITObs panel 分别为 126、57。BizITObs panel 使用 stride 12，三路 split 之间保持 `context + horizon = 192` 点 embargo。

## 资产身份

| Asset | Frozen identity |
| --- | --- |
| GIFT-Eval Arrow (`dataset_info.json`, `data-*.arrow`, `state.json` 的有序内容 manifest) | SHA-256 `0f410dd0eadce583886e7141e556f3a40c069472ad6a1b6c3bd1663d5860c120` |
| GIFT-Eval protocol code | git `6fdb10df9c17411f0aef5ff862afbec23627c12f` |
| Nixtla `datasets.zip` | SHA-256 `6512d9aa80f111ee26480bc6f3f4eb3b5655d4ceecc384933100edc85adf704b` |
| M5 | SHA-256 `0349ba38a2efd30d0f5acc6394c1110e140e1a990c650d7b5ca44c5b25dd12f5` |
| GEFCom2014 | SHA-256 `d68d957270edd93b26a37d0f9b5e901f942abdf34c75eacbe14e417beb16e154` |
| M4 Hourly | SHA-256 `18085bd3c34e41cdc07441aa61c5610dac9e916b9489a6a381f8e89fd01c8a66` |
| Electricity Hourly | SHA-256 `eff447075dde68dca0105ab7e2851c5637967ae3bb21556fd8b931f196d5968c` |
| Traffic Hourly | SHA-256 `3db12ba866a9c9d3c8109b7b6d189a990c38d0e5002fa2617022157358d08299` |
| UCI Hydraulic Systems archive | SHA-256 `24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2` |
| skchange HVAC CSV | SHA-256 `1da08ee5922db6d4d6f4ab32a0e6a9666fc41680ed75dcffe53ac9e1819fff99` |
| skchange source | git `f209def94199607b11b1ae9b3108d80e3e87e624` |

## 冻结验收

1. 通用单变量 capability 至少有 6 个、structured capability 至少有 3 个 development profiles；regime 至少有 2 个不同物理来源且每个至少 30 个合格 parameter windows。
2. 每个 capability 的五档 canonical target 严格递增，且任意相邻档至少相隔原始 target range 的 10%。
3. 每个 calibration grid cell 用 2 组独立 seed bank 估计期望强度，连续反解 `structure_scale` 与 intensity lambda；所有 conditioning cells 再用第三组 256 样本验证，最大 normalized error 不超过 0.20。
4. 每个在线 `profile × capability × intensity` 至少 20 个独立生成 seed，first-pass acceptance rate 不低于 95%，且无 fail-open。
5. artifact 必须记录完整 reference profile 清单、资产哈希、协议代码版本、scale fingerprint 和 held-out family 清单。
6. 只有以上条件全部通过，才允许把新 artifact 标记为 paper-v1 frozen。

## 冻结结果

- `canonical_scale_fingerprint`: `a76b66924562be4f`。
- 9 个 capability 的五档 target 均严格递增并达到最小分辨率；29 个 conditioning cells 的独立 256-sample 校准验证全部通过，最大 normalized error 为 0.18265405，低于 0.20 门限。
- paper-v1 在线集合包含 8 个 profile、23 个 `profile × capability` cells 和 115 个 intensity cells。以根 seed `20260716` 按 `profile / capability / intensity / sample_index` 确定性派生，每个 intensity cell 使用 20 个 seed 走完整在线生成链路：2300 个样本全部在最多 2 次尝试内通过，2296 个首轮通过（99.83%）；单格最低首轮通过率为 95%，最大平均尝试数为 1.05，无 fail-open。
- `multi_seasonal` 与 `time_varying_seasonality` 的 feature-support controls 不再包含 target-coupled `noise_ratio`，而使用 `trend_strength`、`outlier_rate`、`spike_rate`；否则真实数据上的单周期残差会把第二周期或平滑调制误判为噪声并系统性拒绝高 intensity 样本。
- `electricity_hourly_daily_2048ctx_24h` 已完成强度反解及独立校准验证，但尚无 2048-context near-distance artifact，因此只标记为 `research_only_pending_near_distance_gate`，不属于 paper-v1 在线生成集合。它可用于后续输入窗口扫描；在单独冻结 2048 near-distance 标尺前，不计入上述在线验收结果。

| Capability | Primary realized feature | Intensity 1 / 2 / 3 / 4 / 5 |
| --- | --- | --- |
| `trend` | `trend_strength` | 0.05267095 / 0.11124118 / 0.18166790 / 0.36072477 / 0.63837322 |
| `multi_seasonal` | `multi_period_score` | 0.09115325 / 0.13190791 / 0.17819281 / 0.26251039 / 0.43837830 |
| `time_varying_seasonality` | `seasonal_amplitude_modulation` | 0.20458345 / 0.26774497 / 0.33387490 / 0.42103518 / 0.58615995 |
| `regime_switching` | `change_point_shift_energy` | 0.59986317 / 0.64333200 / 0.65754616 / 0.67176032 / 0.74200477 |
| `nonlinear_persistence` | `nonlinear_multi_lag_gain` | 0.06295584 / 0.07261687 / 0.08135004 / 0.09646402 / 0.11853709 |
| `predictable_intermittency` | `spike_rate` | 0 / 0.01308901 / 0.02465969 / 0.03926702 / 0.11570680 |
| `common_factor` | `pca_top1_explained` | 0.66643664 / 0.72857765 / 0.76861372 / 0.81026670 / 0.87516130 |
| `hierarchical_coherence` | `hierarchy_child_heterogeneity` | 0.23144562 / 0.31677785 / 0.37237839 / 0.39031792 / 0.41084095 |
| `covariate_response` | `covariate_incremental_r2` | 0.15764575 / 0.21118203 / 0.23302623 / 0.25841987 / 0.29686976 |
