# Synthetic v2 真实数据 Profile 烟测

日期：2026-06-29

## 目的

本次烟测验证 synthetic v2 第一版显式 feature profiler 路径：公开真实数据能否稳定转换成特征分位数 profile，以及目标特征是否能按固定倍数得到可解释上限。

## 输入

- US Births Dataset: https://zenodo.org/records/4656049/files/us_births_dataset.zip?download=1
- M4 Hourly Dataset: https://zenodo.org/records/4656589/files/m4_hourly_dataset.zip?download=1
- 本地数据缓存：`runtime/research`
- JSON profile 输出：`runtime/research/synthetic-v2-profile-smoke`
- 目标特征上限规则：`p95 * 1.5`；天然有界特征额外截断到 `1.0`。

## Profile 汇总

| Profile | 窗口数 | 序列数 | Trend p50/p95/cap | Seasonal p50/p95/cap | Slope p95/cap | Curvature p95/cap | Noise p95 |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| us_births_weekly | 20 | 1 | 0.2459/0.2938/0.4406 | 0.704/0.8309/1 | 0.2102/0.3606/0.541 | 0.3674/0.5142/0.7713 | 0.3567 |
| us_births_annual_diagnostic | 20 | 1 | 0.0733/0.1067/0.1601 | 0/0/0 | 0.2102/0.3606/0.541 | 0.3674/0.5142/0.7713 | 0.9962 |
| m4_hourly_daily_96ctx | 2000 | 414 | 0.0409/0.6437/0.9656 | 0.9431/0.9907/1 | 0.1468/0.4339/0.6508 | 0.0738/0.5878/0.8817 | 0.3611 |
| m4_hourly_daily_168ctx | 2000 | 414 | 0.1659/0.7714/1 | 0.9129/0.9961/1 | 0.1264/0.3543/0.5314 | 0.0711/0.6756/1.0135 | 0.3871 |
| m4_hourly_weekly | 1000 | 414 | 0.2812/0.9772/1 | 0.9758/0.9986/1 | 0.1242/0.3273/0.4909 | 0.0564/0.4355/0.6532 | 0.4087 |

## 观察

- profiler 现在可以读取带非 UTF-8 元数据的 Monash TSF zip，并且 TSF 输入的 `max_windows` 已按全数据集统一限流。
- US Births 适合作为小型日频 sanity check。周季节性有清晰信号；年季节性这里只作为诊断项，因为 `365+30` 窗口不足两个完整年周期，`seasonal_strength=0` 不代表真实数据没有年季节性。
- M4 Hourly 更适合作为第一版小时级 trend 和 seasonality anchor：它有数百条序列，日季节性强，并且更长 context 能暴露更强的趋势变化。
- cap multiplier 能避免目标特征增强无限偏离真实分布。对于 `trend_strength` / `seasonal_strength` 等天然 `[0, 1]` 特征，截断逻辑已经生效。

## 决策

- `m4_hourly_daily_168ctx` 作为 trend 和 multi-seasonal v2 pilot 的主小时级 anchor。
- `us_births_weekly` 保留为小型日频回归 / sanity anchor。
- 第一版 pilot 使用 `target_max_multiplier=1.5`，暂不放宽，因为多个真实 profile 的 p95 已经接近有界特征上限。

## 复现

```bash
python3 scripts/run_synthetic_v2_profile_smoke.py
```
