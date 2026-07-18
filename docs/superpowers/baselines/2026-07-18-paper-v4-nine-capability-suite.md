# Paper v4：九能力四档 Profile 与合格样本验收

日期：2026-07-18

## 结论

九个能力维度已经全部具备 `H=48`、`L={96,168,336,504}` 的真实 profile、生成器
conditioning、controlled-feature support 与 near-distance 校准产物。层级数据使用
M5 Daily 的严格加和结构，在 H=48 下可用，因此没有退回 H=28。

独立种子验收覆盖：

```text
9 capabilities × 5 intensities × 8 master samples = 360 master samples
4 paired lookback views per master sample
```

结果为 `360/360` 母样本合格、`1,440/1,440` 视图同时通过，失败数为 0。每条母样本只
生成一次，四档 L 共享同一段原始 48 步 future。

## 能力结果

| 能力 | 任务结构 | 合格/期望 | 平均尝试次数 |
|---|---|---:|---:|
| trend | 单变量 | 40/40 | 1.0 |
| multi_seasonal | 单变量 | 40/40 | 1.0 |
| time_varying_seasonality | 单变量 | 40/40 | 1.0 |
| regime_switching | 单变量 | 40/40 | 1.0 |
| nonlinear_persistence | 单变量 | 40/40 | 1.0 |
| predictable_intermittency | 单变量 | 40/40 | 1.0 |
| common_factor | 三变量面板 | 40/40 | 1.0 |
| hierarchical_coherence | 三变量严格加和层级 | 40/40 | 1.0 |
| covariate_response | 单目标＋多外生变量 | 40/40 | 1.0 |

所有九个 conditioning 的五档主特征校准状态均为 `supported`，最大归一化校准误差为
0.1072，低于既定容差。

## Profile 与数据集对应

| 能力 | 真实 profile 数据集 |
|---|---|
| 六个单变量能力 | M4 Hourly；Electricity/H；Solar/H；ETT1/H；ETT2/H；Jena Weather/H；KDD Cup 2018/H；Loop Seattle/H；SZ-Taxi/H；M_DENSE/H；Bitbrains Fast Storage/H；Bitbrains RND/H；BizITObs L2C/H |
| common_factor | Electricity Hourly；Traffic Hourly；Jena Weather/H；BizITObs L2C/H |
| hierarchical_coherence | M5 Daily |
| covariate_response | GEFCom2014 Load；GEFCom2014 Solar |

单变量正式 source profile 为 `13×4=52` 个；结构能力 source profile 为
`(4+1+2)×4=28` 个；合计 80 个 source×L profile。在线生成按任务使用 4 个
L=504 master conditioning profiles，并使用 `4 tasks×4 L=16` 个 view-specific
feature/near-distance gate profiles。

## 关键校准修正

首轮审计没有把失败掩盖成重试通过，而是修正了两个机制问题：

1. 多数据集不能先全局 group split。否则 GEFCom Load 可能只在 reference、Solar 只在
   calibration，跨数据集距离会抬高 novelty 阈值。正式版在每个数据集内部先做三路无
   泄漏切分，再将三个 partition 分别按来源等权汇总。
2. GEFCom 长窗口真实 covariate residual outlier rate 的 P75 为 0.003623。Student-t
   残差会系统性超出 controlled-feature support，因此在 intensity 拟合之前依据真实
   control profile 选择 Gaussian residual；这属于生成机制预条件，不是放宽 gate。

层级 L=336 的早期边界失败来自 artifact 小数序列化。正式 feature artifact 对非空支持
阈值加入 `1.00001` 的纯数值 rounding guard，不改变 conformal coverage 目标。

## 产物

目录：

```text
runtime/paper_exp/v4/01_nine_capability_suite/
```

关键哈希：

```text
profile_suite.json                  350a0adc8d9f99f054e340f974c7274415bc96da847b509698e6e149141da375
generator_conditioning_artifact.json
                                    48a9504c61a7903dc71c3465d182af8c12b10f8d1c512726b693cbfef83bf9a9
feature_gate_artifact.json          67110e40a35f35cdba5864e5b6bab9922061c8511952e0f685daade11a190120
near_distance_artifact.json         3139b7a29036d2adf73463b2e79862eb093b8ca7ef23fb479f9bf6dae4c78392
qualification.json                  dc4cff303d5b4f5fca1f097d9112fc8ccc89f944763481e36c8c2038ab0adaf5
capability_dataset_mapping.csv      ef78a8277a7cf8fa3a595a5bd96d71a44dc22f635b4c18157884cb8470237a75
manifest.json                       fcd8f4b8b31324d5750f1336dc1bd0eaf53edfe49b4faf9ffff48612f3dc09b9
```

协议：

`docs/superpowers/specs/2026-07-18-paper-v4-nine-capability-profile-and-generation-protocol.md`
