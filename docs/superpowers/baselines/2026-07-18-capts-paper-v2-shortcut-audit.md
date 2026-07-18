# CapTS paper-v2 shortcut-resistance baseline

日期：2026-07-18

## 冻结对象

- generator version：`capts-paper-v2`
- canonical scale：`synthetic-v2-paper-v2-shortcut-resistant-2026-07-18`
- scale fingerprint：`715a6bb980f0e4aa`
- conditioning artifact schema：`synthetic_v2_generator_conditioning_artifact.v3`
- conditioning cells：28，全部 `supported`
- 最大 canonical normalized error：0.165820
- unsupported cells：0

## 运行命令

```bash
cd backend
uv run python ../scripts/audit_synthetic_capability_shortcuts.py \
  --seed-count 128 \
  --intensities 1 3 5
```

审计使用脚本冻结的 profile 与 seed 派生规则。I=1/3/5 均生成和记录，
发布资格在 I=5 判定。单样本 future error 不参与生成 acceptance。

## 预注册判据

1. 对非季节能力，固定 `seasonal-naive(P)` 的平均 MAE 与 last-value
   平均 MAE 之比必须大于 0.85；比值过低表示存在统一周期捷径。
2. capability-aware 与 capability-blind 使用配对样本；样本数至少 24。
3. aware win rate 不低于 0.60。
4. 配对 loss difference 的 ratio-of-means 单侧 95% 下界不低于 0.02。
5. contrast 只做 seed-bank 聚合资格审计，不按单样本结果筛选 future。

## I=5 结果

| Capability | seasonal-naive / last | Mean loss gain | One-sided 95% LCB | Aware win rate | 通过 |
| --- | ---: | ---: | ---: | ---: | :---: |
| `trend` | 1.2041 | 0.2836 | 0.2222 | 0.7734 | 是 |
| `multi_seasonal` | 1.0613（季节能力，仅报告） | 0.9541 | 0.8712 | 1.0000 | 是 |
| `time_varying_seasonality` | 0.7689（季节能力，仅报告） | 0.6322 | 0.5513 | 0.8359 | 是 |
| `regime_switching` | 1.0131 | 0.6417 | 0.5892 | 0.9531 | 是 |
| `nonlinear_persistence` | 0.9817 | 0.0423 | 0.0252 | 0.6641 | 是 |
| `predictable_intermittency` | 1.3598 | 0.5415 | 0.5032 | 0.9688 | 是 |
| `common_factor` | 1.3252 | 0.1596 | 0.0985 | 0.6797 | 是 |
| `hierarchical_coherence` | 1.0871 | 0.1696 | 0.1247 | 0.7812 | 是 |
| `covariate_response` | 1.2143 | 0.1781 | 0.0956 | 0.6719 | 是 |

结论：9/9 capability contrast 通过；7/7 非季节能力通过固定
seasonal-naive shortcut 判据；整体审计通过。

## 非线性递推说明

非线性能力的 aware forecast 使用跨 profile 固定的 50% correction
shrinkage：

\[
\widehat y_{\mathrm{aware}}
=\widehat y_{\mathrm{blind}}
+0.5\left(
  \widehat y_{\mathrm{nonlinear}}-\widehat y_{\mathrm{blind}}
\right).
\]

该系数不按被评分 future 调整，用于抑制递归多步误差累积。另一个未参与
上述资格审计的 seed bank（3 个 168-context profiles，每个 256 seeds）
上，aware win rate 分别为 0.6719、0.6641、0.6328，单侧 95% LCB
分别为 0.0411、0.0389、0.0367。
