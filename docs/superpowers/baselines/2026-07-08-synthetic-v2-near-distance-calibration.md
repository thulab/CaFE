# Synthetic v2 Near-Distance Calibration

日期：2026-07-15

## Purpose

校准 DCR/NNDR 近距离污染风险阈值：用 real holdout 到 real train 的自然最近邻距离定 p01/p05 基线，并用 exact copy、jitter copy、normal synthetic 检查阈值是否能区分复制与正常生成。

## Design

- Buckets: 8 real profile buckets.
- Real windows per bucket cap: 600; splits: 5; synthetic controls per bucket: 48.
- Jitter copy scale: 0.02 on context-standardized target values.
- Raw distance is computed on context-standardized target windows. Feature distance uses robust-z explicit features fitted on each split's real train set.
- Source series/panel groups never cross train/holdout. Single-series buckets use temporal blocks with a C+H non-overlap embargo.
- Full target-window and model-visible target-context DCR are both checked; the deployed threshold and reference rows come from the same fixed split.
- Near-constant real target windows are excluded before split calibration because zero-information windows can make p01 DCR thresholds collapse to zero.
- Scope: raw DCR covers target trajectories in the committed R_train reference. Known-future covariates enter feature DCR but are not concatenated into the raw vector; R_holdout and unknown pretraining corpora are not coverage claims.
- Strict risk: full-window OR context-only raw MAE/L2 DCR <= corresponding real-holdout p01.
- Combined risk: full-window combined rule OR context raw MAE/L2 <= p05 AND context NNDR <= p01.

## Threshold Stability

| Bucket | real windows | raw MAE p01 mean/cv | raw L2 p01 mean/cv | feature L2 p01 mean/cv | NNDR p01 mean/cv |
| --- | ---: | ---: | ---: | ---: | ---: |
| `m4_hourly_daily_168ctx` | 600 | 0.0345/0.1076 | 0.0429/0.0994 | 0.0382/0.1997 | 0.4264/0.2698 |
| `electricity_hourly_daily_168ctx` | 597 | 0.1602/0.0806 | 0.2193/0.0631 | 0.1358/0.0738 | 0.7569/0.0186 |
| `traffic_hourly_daily_168ctx` | 600 | 0.1475/0.2874 | 0.2291/0.3063 | 0.1143/0.2536 | 0.5982/0.2072 |
| `electricity_hourly_panel_168ctx` | 600 | 0.2309/0.0342 | 0.3053/0.0336 | 0.1545/0.1367 | 0.7969/0.0320 |
| `traffic_hourly_panel_168ctx` | 600 | 0.2798/0.0499 | 0.4473/0.0252 | 0.2509/0.1076 | 0.6916/0.0771 |
| `m5_daily_covariate_365ctx_28h` | 494 | 0.2676/0.0704 | 1.051/0.0105 | 0.2603/0.0556 | 0.8470/0.0218 |
| `m5_daily_hierarchy_365ctx_28h` | 600 | 0.2823/0.0720 | 0.3951/0.0904 | 0.3064/0.0601 | 0.7476/0.0524 |
| `gefcom2014_load_hourly_covariate_168ctx_24h` | 600 | 0.1716/0.0386 | 0.2237/0.0474 | 0.1110/0.0830 | 0.7080/0.0540 |

## Positive/Negative Controls

| Bucket | holdout combined | exact strict/combined | affine strict | context-copy strict | jitter strict/combined | normal strict/combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `m4_hourly_daily_168ctx` | 0.0100 | 1.000/1.000 | 1.000 | 1.000 | 1.000/1.000 | 0/0 |
| `electricity_hourly_daily_168ctx` | 0.0116 | 1.000/1.000 | 1.000 | 1.000 | 1.000/1.000 | 0/0 |
| `traffic_hourly_daily_168ctx` | 0.0267 | 1.000/1.000 | 1.000 | 1.000 | 1.000/1.000 | 0/0 |
| `electricity_hourly_panel_168ctx` | 0.0098 | 1.000/1.000 | 1.000 | 1.000 | 1.000/1.000 | 0/0 |
| `traffic_hourly_panel_168ctx` | 0.0133 | 1.000/1.000 | 1.000 | 1.000 | 1.000/1.000 | 0/0 |
| `m5_daily_covariate_365ctx_28h` | 0.0020 | 1.000/1.000 | 1.000 | 1.000 | 1.000/1.000 | 0/0 |
| `m5_daily_hierarchy_365ctx_28h` | 0.0117 | 1.000/1.000 | 1.000 | 1.000 | 1.000/1.000 | 0/0 |
| `gefcom2014_load_hourly_covariate_168ctx_24h` | 0.0133 | 1.000/1.000 | 1.000 | 1.000 | 1.000/1.000 | 0/0 |

## Overall Checks

- Exact-copy strict-risk minimum across buckets: `1.000`.
- Jitter-copy combined-risk minimum across buckets: `1.000`.
- Affine-copy strict-risk minimum across buckets: `1.000`.
- Context-copy strict-risk minimum across buckets: `1.000`.
- Normal-synthetic combined-risk max across buckets: `0`.

## Bucket Flags

| Bucket | reason |
| --- | --- |
| - | No bucket exceeded the current warning heuristics. |

## Notes

- This calibrates novelty thresholds and writes the online near-distance reference artifact used by generation acceptance.
- A good threshold should flag exact copies almost always, flag small jitter copies frequently, and keep normal synthetic combined risk near or below the paper tolerance target.
- If a bucket has high threshold CV or high normal-synthetic risk, rerun with a larger real-window cap and inspect that bucket before freezing paper thresholds.

Full JSON summary: `runtime/research/synthetic-v2-near-distance-calibration/summary.json`.
