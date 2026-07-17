# Paper v2：六个单变量能力 transfer freeze

日期：2026-07-17

## 结论

paper-v2 的六个单变量能力生成机制与真实分布迁移协议已经冻结。正式协议不再使用
`context=168`，而统一使用 `context=504`、`horizon=48`、`season_length=24`：
上下文覆盖 21 个完整日周期，canonical strength 在 `504+24=528` 的整周期前缀上测量，
完整 48 步未来仍用于门控和模型指标。

冻结的 9 个 held-out GIFT-Eval hourly profiles 为：

- `solar/H`
- `kdd_cup_2018_with_missing/H`
- `LOOP_SEATTLE/H`
- `SZ_TAXI/H`
- `M_DENSE/H`
- `ett1/H`
- `ett2/H`
- `bitbrains_fast_storage/H`
- `bitbrains_rnd/H`

这 9 个 profile 只使用官方 training prefix 拟合生成参数、feature support 与 near-distance；
紧邻 test 的 validation horizon 和全部官方 test windows 均未进入 artifact。

## Canonical scale

- scale ID：`synthetic-v2-paper-v2-504ctx-frozen-2026-07-17`
- fingerprint：`a9987cef66a2fd9b`
- 六能力：`trend`、`multi_seasonal`、`time_varying_seasonality`、
  `regime_switching`、`nonlinear_persistence`、`predictable_intermittency`
- canonical development families：M4 Hourly、Electricity、Traffic、
  Jena Weather/H、BizITObs-L2C/H

五个 development families 与 9 个 held-out transfer profiles 完全分离。每个能力先在最终
`504/48/24` shape 上形成 family-balanced 真实分位坐标，再做能力构造支持投影；held-out
profile 只能拟合从 frozen target 到本地生成参数的单调逆映射，不能重定义 intensity。

## 关键方法修正

- `nonlinear_persistence` 的主特征改为 `nonlinear_conditional_gain`。线性基线包含 lag-1、
  seasonal lag 与 raw nonlinear lag，增强模型只增加 `sin²(1.1×lag)`；v2 递推使用同一
  有界变换，最大 Lipschitz 稳定界为 0.975。
- `regime_switching` 的 canonical target 投影到 recurring-clock 构造支持
  `[0.56, 0.94]`，排除不可预测的一次性 change point。
- `nonlinear_persistence` 投影到 `[0.002, 0.025]`，避免有限样本估计器底噪。
- feature-support 只允许与目标机制不机械耦合的 observable controls。
  `regime_switching` 与 `predictable_intermittency` 会改变当前全部分解、spike/outlier
  摘要，因此 artifact 显式记录
  `not_applicable_no_independent_observable_controls`，不拟合伪独立门限。
- 上述两个 N/A 不代表取消真实性约束：train-only 参数支持、construction predictability、
  frozen canonical dose、target dose-response 与 near-distance 仍为强制项。

## 冻结质量

- 9 profiles × 6 capabilities = 54 个正式 transfer cells。
- 每个 profile 最多抽取 600 个 train-only windows；`SZ_TAXI/H` 可用 312 个，其余均为
  600 个。
- 各 cell 的 inverse-calibration normalized error 全部低于 0.20；最大值为
  `bitbrains_rnd/H × nonlinear_persistence` 的 0.1969。
- 正式 preflight 为 54/54 通过，最大 acceptance attempts 为 1。
- 后续 E2 planned-seed audit 又完整生成 21,600/21,600 条样本，无失败；平均
  acceptance attempts 为 1.0418，最大为 24。

## 封存信息

- 实现与 manifest commit：`08d752cbd8ed1418f70db63b310d2203d390cfca`
- 输出：`runtime/paper_exp/v2/00_transfer_protocol_freeze/`
- 大小：36,863,274 bytes
- manifest SHA-256：
  `15df2b046ddd5bf4d8c331ff9ea1b21ca25bf7315ad9f4cd42e01c033150d555`
- manifest 文件数：11

旧的中间 freeze 与失败 smoke/partial 均保留在 runtime 的 `superseded_*` 目录用于诊断，
不进入正式 E2/E3 manifest。

