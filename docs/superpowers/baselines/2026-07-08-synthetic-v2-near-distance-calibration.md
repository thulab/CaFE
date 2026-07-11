# Synthetic v2 Near-Distance Calibration

日期：2026-07-08

## Purpose

校准 DCR/NNDR 近距离污染风险阈值：用 real holdout 到 real train 的自然最近邻距离定 p01/p05 基线，并用 exact copy、jitter copy、normal synthetic 检查阈值是否能区分复制与正常生成。

## Design

- Buckets: 8 real profile buckets.
- Real windows per bucket cap: 600; splits: 5; synthetic controls per bucket: 48.
- Jitter copy scale: 0.02 on context-standardized target values.
- Raw distance is computed on context-standardized target windows. Feature distance uses robust-z explicit features fitted on each split's real train set.
- Near-constant real target windows are excluded before split calibration because zero-information windows can make p01 DCR thresholds collapse to zero.
- Strict risk: raw_mae_d1 <= real_holdout_p01 AND raw_l2_d1 <= real_holdout_p01.
- Combined risk: raw_mae_d1 <= p05 AND raw_l2_d1 <= p05 AND (feature_l2_d1 <= p01 OR raw_mae_nndr <= p01).

## Threshold Stability

| Bucket | real windows | raw MAE p01 mean/cv | raw L2 p01 mean/cv | feature L2 p01 mean/cv | NNDR p01 mean/cv |
| --- | ---: | ---: | ---: | ---: | ---: |
| `m4_hourly_daily_168ctx` | 600 | 0.0282/0.0188 | 0.0356/0.0299 | 0.0264/0.0834 | 0.2833/0.1116 |
| `electricity_hourly_daily_168ctx` | 597 | 0.1454/0.0683 | 0.1970/0.0858 | 0.1053/0.0850 | 0.7333/0.0665 |
| `traffic_hourly_daily_168ctx` | 600 | 0.0557/0.6894 | 0.0814/0.6612 | 0.0442/0.8323 | 0.3341/0.7151 |
| `electricity_hourly_panel_168ctx` | 600 | 0.1830/0.0383 | 0.2520/0.0120 | 0.1114/0.0450 | 0.7159/0.0328 |
| `traffic_hourly_panel_168ctx` | 600 | 0.2504/0.0575 | 0.4166/0.0147 | 0.1931/0.0499 | 0.6697/0.0680 |
| `m5_daily_covariate_365ctx_28h` | 494 | 0.2229/0.2307 | 1.035/0.0190 | 0.1927/0.0386 | 0.8389/0.0510 |
| `m5_daily_hierarchy_365ctx_28h` | 600 | 0.2532/0.0207 | 0.3531/0.0316 | 0.1439/0.1803 | 0.7763/0.0310 |
| `gefcom2014_load_hourly_covariate_168ctx_24h` | 600 | 0.1620/0.0228 | 0.2061/0.0380 | 0.0712/0.0859 | 0.7209/0.0407 |

## Positive/Negative Controls

| Bucket | holdout combined | exact strict/combined | jitter strict/combined | normal strict/combined |
| --- | ---: | ---: | ---: | ---: |
| `m4_hourly_daily_168ctx` | 0.0067 | 1.000/1.000 | 1.000/1.000 | 0/0 |
| `electricity_hourly_daily_168ctx` | 0.0050 | 1.000/1.000 | 1.000/1.000 | 0/0 |
| `traffic_hourly_daily_168ctx` | 0.0200 | 1.000/1.000 | 1.000/0.9833 | 0/0 |
| `electricity_hourly_panel_168ctx` | 0.0050 | 1.000/1.000 | 1.000/1.000 | 0/0 |
| `traffic_hourly_panel_168ctx` | 0.0150 | 1.000/1.000 | 1.000/1.000 | 0/0 |
| `m5_daily_covariate_365ctx_28h` | 0 | 1.000/1.000 | 1.000/1.000 | 0/0 |
| `m5_daily_hierarchy_365ctx_28h` | 0.0117 | 1.000/1.000 | 1.000/1.000 | 0/0 |
| `gefcom2014_load_hourly_covariate_168ctx_24h` | 0.0050 | 1.000/1.000 | 1.000/1.000 | 0/0 |

## Overall Checks

- Exact-copy strict-risk minimum across buckets: `1.000`.
- Jitter-copy combined-risk minimum across buckets: `0.9833`.
- Normal-synthetic combined-risk max across buckets: `0`.

## Bucket Flags

| Bucket | reason |
| --- | --- |
| `traffic_hourly_daily_168ctx` | raw MAE p01 CV=0.6894; feature L2 p01 CV=0.8323 |

## Notes

- This calibrates novelty thresholds and writes the online near-distance reference artifact used by generation acceptance.
- A good threshold should flag exact copies almost always, flag small jitter copies frequently, and keep normal synthetic combined risk near or below the paper tolerance target.
- If a bucket has high threshold CV or high normal-synthetic risk, rerun with a larger real-window cap and inspect that bucket before freezing paper thresholds.

Full JSON summary: `runtime/research/synthetic-v2-near-distance-calibration/summary.json`.
