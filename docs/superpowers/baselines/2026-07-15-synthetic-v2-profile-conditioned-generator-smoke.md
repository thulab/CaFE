# Synthetic v2 profile-conditioned generator smoke

日期：2026-07-15

本记录只用于进入正式 E1 前的工程验收，不替代 500+ samples/cell、bootstrap CI、MMD/SWD、oracle headroom 和模型推理实验。

## 协议

- generator conditioning：`synthetic_v2_generator_conditioning_artifact.v1`
- 真实窗口三路拆分：generator parameter / feature-gate reference / feature-gate calibration
- 单元：9 个真实 profile 中可用的 29 个 `profile × capability` 组合，每个组合扫描 intensity 1--5
- 每个 intensity 使用相同的 32 个 sample seeds，保证 nuisance 配对
- feature-support：每 cell 32 个未拒绝候选，共 4640 个
- near-distance：有在线 reference 的组合每 cell 8 个未拒绝候选，共 920 个；2048-context 专用实验 bucket 不计入在线 artifact

## 结果

| 检查 | 结果 |
| --- | ---: |
| 预选 bucket feature-support 首次通过 | 4640 / 4640 |
| 所有兼容 bucket near-distance 首次通过 | 920 / 920 |
| 主 target feature 聚合均值方向不单调的 `profile × capability` 单元 | 0 / 29 |

拒绝采样仍保留为最多 32 次的工程故障保护，但本次 smoke 没有依赖它。正式 E1 应继续以 first-pass rate 为报告对象，要求每个 cell 至少 95%。

## Profile conditioning 可见性检查

固定 intensity=3、每个 case 128 个配对 seeds 时，部分生成分布中位数为：

| profile / capability | selected realized medians |
| --- | --- |
| M4 hourly / trend | seasonal strength 0.9553；noise ratio 0.0435；spike rate 0.0000 |
| Electricity hourly / trend | seasonal strength 0.9523；noise ratio 0.0469；spike rate 0.0000 |
| Traffic hourly / trend | seasonal strength 0.6305；noise ratio 0.3552；spike rate 0.0052 |
| GEFCom load / covariate response | residual ACF 0.1303；residual spike rate 0.0000；incremental R2 0.5104 |
| M5 retail / covariate response | residual ACF 0.0444；residual spike rate 0.0230；incremental R2 0.0083 |

这些数字只证明 conditioning 已实际改变生成 DGP，而不是仅改变 metadata。它们不表示合成分布必须逐特征复刻真实中位数；正式判断仍使用预注册 control-support、批量分布距离和 target dose response。
