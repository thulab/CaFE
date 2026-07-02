# Synthetic v2 能力维度指标曲线

日期：2026-07-02

结果指标：`mae`, `mase`。每张图横坐标为 difficulty，纵坐标为指标值，曲线为模型；`naive` 和 `seasonal_naive` 使用虚线。

## 输入数据

| Source | Capabilities |
| --- | --- |
| `runtime/research/synthetic-v2-univariate-capabilities-experiment/summary.json` | `trend`, `multi_seasonal`, `regime_switching`, `long_memory_nonlinear`, `intermittent_heteroskedastic` |
| `runtime/research/synthetic-v2-time-varying-seasonality-experiment/summary.json` | `time_varying_seasonality` |
| `runtime/research/synthetic-v2-multitarget-capabilities-experiment/summary.json` | `common_factor`, `lead_lag_coupling`, `coherent_regime_shift` |
| `runtime/research/synthetic-v2-hierarchical-coherence-experiment/summary.json` | `hierarchical_coherence` |
| `runtime/research/synthetic-v2-covariate-capabilities-experiment/summary.json` | `covariate_response` |

## 图表

### `trend` / `mae`

![trend mae](figures/synthetic-v2-capabilities/trend-mae.png)

### `multi_seasonal` / `mae`

![multi_seasonal mae](figures/synthetic-v2-capabilities/multi_seasonal-mae.png)

### `time_varying_seasonality` / `mae`

![time_varying_seasonality mae](figures/synthetic-v2-capabilities/time_varying_seasonality-mae.png)

### `regime_switching` / `mae`

![regime_switching mae](figures/synthetic-v2-capabilities/regime_switching-mae.png)

### `long_memory_nonlinear` / `mae`

![long_memory_nonlinear mae](figures/synthetic-v2-capabilities/long_memory_nonlinear-mae.png)

### `intermittent_heteroskedastic` / `mae`

![intermittent_heteroskedastic mae](figures/synthetic-v2-capabilities/intermittent_heteroskedastic-mae.png)

### `common_factor` / `mae`

![common_factor mae](figures/synthetic-v2-capabilities/common_factor-mae.png)

### `lead_lag_coupling` / `mae`

![lead_lag_coupling mae](figures/synthetic-v2-capabilities/lead_lag_coupling-mae.png)

### `coherent_regime_shift` / `mae`

![coherent_regime_shift mae](figures/synthetic-v2-capabilities/coherent_regime_shift-mae.png)

### `hierarchical_coherence` / `mae`

![hierarchical_coherence mae](figures/synthetic-v2-capabilities/hierarchical_coherence-mae.png)

### `covariate_response` / `mae`

![covariate_response mae](figures/synthetic-v2-capabilities/covariate_response-mae.png)

### `trend` / `mase`

![trend mase](figures/synthetic-v2-capabilities/trend-mase.png)

### `multi_seasonal` / `mase`

![multi_seasonal mase](figures/synthetic-v2-capabilities/multi_seasonal-mase.png)

### `time_varying_seasonality` / `mase`

![time_varying_seasonality mase](figures/synthetic-v2-capabilities/time_varying_seasonality-mase.png)

### `regime_switching` / `mase`

![regime_switching mase](figures/synthetic-v2-capabilities/regime_switching-mase.png)

### `long_memory_nonlinear` / `mase`

![long_memory_nonlinear mase](figures/synthetic-v2-capabilities/long_memory_nonlinear-mase.png)

### `intermittent_heteroskedastic` / `mase`

![intermittent_heteroskedastic mase](figures/synthetic-v2-capabilities/intermittent_heteroskedastic-mase.png)

### `common_factor` / `mase`

![common_factor mase](figures/synthetic-v2-capabilities/common_factor-mase.png)

### `lead_lag_coupling` / `mase`

![lead_lag_coupling mase](figures/synthetic-v2-capabilities/lead_lag_coupling-mase.png)

### `coherent_regime_shift` / `mase`

![coherent_regime_shift mase](figures/synthetic-v2-capabilities/coherent_regime_shift-mase.png)

### `hierarchical_coherence` / `mase`

![hierarchical_coherence mase](figures/synthetic-v2-capabilities/hierarchical_coherence-mase.png)

### `covariate_response` / `mase`

![covariate_response mase](figures/synthetic-v2-capabilities/covariate_response-mase.png)

### `hierarchical_coherence` / `coherence_mae`

![hierarchical_coherence coherence_mae](figures/synthetic-v2-capabilities/hierarchical_coherence-coherence_mae.png)
