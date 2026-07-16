# E1 — Synthetic method validity 正式结果

日期：2026-07-16

## 实验身份

- 协议与 runner commit：`a0f6843f632165b8f4466c5f51c854ca52f1bcf6`。
- canonical scale：`synthetic-v2-paper-v1-frozen-2026-07-16`，fingerprint `a76b66924562be4f`。
- 正式输出：`runtime/paper_exp/v1/E1_method_validity/`。
- 规模：8 个在线 profile、23 个 `profile × capability` cells、5 档 intensity、两轮独立 seed、每轮每格 64 个样本，共 14720 个样本。
- 未调用任何时序大模型。
- `manifest.json` 记录 3 个输入 artifact、runner 和 13 个输出文件的 SHA-256；复核时所有哈希一致，`samples.jsonl` 恰好 14720 行。

## 预注册判据结果

| Criterion | Result | Main statistic |
| --- | --- | --- |
| canonical dose-response | 未通过 | 22 / 23 profile-capability checks 通过 |
| realized control selectivity | 未通过 | 63 / 69 control-feature checks 通过 |
| construction predictability | 通过 | 23 / 23，14720 / 14720 samples validated |
| online control-feature support | 通过 | 115 / 115 intensity cells；首轮 99.82%，单格最低 96.88% |
| MMD / SWD | 通过 | closer-than-shifted-negative：95.65% / 100% |
| DCR / NNDR | 通过 | 115 / 115 cells，无 strict/combined risk |
| cross-round repetition | 通过 | 115 / 115 cells；三种 duplicate rate 均为 0 |
| naive / seasonal-naive / oracle | 通过 | 9 / 9 capabilities 的 oracle win rate ≥ 50% |

因此总体为 6 / 8 类预注册判据通过。不能把 E1 表述成“方法全部验收通过”；它同时给出了强生成链路证据和明确的 selectivity 缺陷。

## E1.1 Canonical dose-response

所有 23 个 profile/capability 的五档均值 Spearman 均为 1.0，说明剂量顺序稳定。22 个单元的最大 normalized error 不超过 0.25。

唯一失败为 `m4_hourly_daily_168ctx / regime_switching`：最大 normalized error 为 0.25597，出现在 intensity 1；五档 realized mean 为 0.56348、0.60778、0.62216、0.63646、0.70711，对应 target 为 0.59986、0.64333、0.65755、0.67176、0.74200。它表现为整条曲线近似平行下移，而不是剂量顺序错误。跨三个在线 univariate profiles 汇总后，regime 的最大 normalized error 为 0.0796。

| Capability | Profiles | Minimum Spearman | Worst profile normalized error | Failed profiles |
| --- | ---: | ---: | ---: | ---: |
| `trend` | 3 | 1.0000 | 0.0226 | 0 |
| `multi_seasonal` | 3 | 1.0000 | 0.0098 | 0 |
| `time_varying_seasonality` | 3 | 1.0000 | 0.0103 | 0 |
| `regime_switching` | 3 | 1.0000 | 0.2560 | 1 |
| `nonlinear_persistence` | 3 | 1.0000 | 0.1724 | 0 |
| `predictable_intermittency` | 3 | 1.0000 | 0.0463 | 0 |
| `common_factor` | 2 | 1.0000 | 0.0506 | 0 |
| `hierarchical_coherence` | 1 | 1.0000 | 0.0030 | 0 |
| `covariate_response` | 2 | 1.0000 | 0.0599 | 0 |

解释：结果强力支持五档排序和总体 absolute scale，但严格按预注册阈值，M4 regime 仍记为失败。不能因为只超出 0.00597 就事后改阈值；若要确认它属于有限样本波动，应另建不覆盖 v1 的 confirmatory run。

## E1.2 Selectivity

69 个 realized control-feature checks 中有 6 个失败：

| Profile / capability | Control | Median paired i5−i1 shift (real IQR) | Interpretation |
| --- | --- | ---: | --- |
| M4 / nonlinear | `spike_rate` | +0.5000002 | 数值上贴着 0.5 边界，受离散 spike count 量化影响 |
| Traffic / nonlinear | `seasonal_strength` | −1.0711 | recurrence 增强改变季节分解占比 |
| Traffic / nonlinear | `noise_ratio` | +1.1646 | target structure 与残差方差比耦合 |
| Traffic / intermittency | `seasonal_strength` | +1.7638 | 周期脉冲被 phase-seasonality estimator 吸收 |
| Traffic / intermittency | `noise_ratio` | −1.9428 | 脉冲增强扩大总方差，固定噪声的比例下降 |
| Traffic / trend | `noise_ratio` | −0.7582 | 趋势增强扩大总方差，固定噪声的比例下降 |

这些变化不是随机漂移：相关项的五档 Spearman 绝对值均接近 1。尤其 `noise_ratio` 是“残差方差 / 总方差”，在目标结构幅度变化时天然 target-coupled，不适合作为严格不变的 nuisance 证据。

完整 primary-feature response matrix 进一步显示：按各 feature 的 canonical range 归一 endpoint effect 后，只有 14 / 23 个 profile/capability 单元由对角目标 feature 占优。主要重叠为：

- `trend` 同时大幅提高 `change_point_shift_energy`；
- `predictable_intermittency` 同时大幅提高 `nonlinear_multi_lag_gain`；
- 部分 `multi_seasonal` 样本对 `change_point_shift_energy` 的归一响应略高于自身 target response。

这说明当前九种机制具备清晰的主目标剂量，但 realized feature 并非九个正交坐标。若论文要做强能力归因，这是 E1 发现的主要方法风险；若论文只主张多种结构 stress tests，则可以报告这种 overlap，但不能声称完全 disentangled。

## E1.3–E1.5 门控、分布与防复刻

- construction contracts：全部样本通过；没有通过重采样绕开配置级 predictability failure。
- 在线全链路首轮通过率为 0.99823，单格最低 0.96875，最大尝试数 2；最终 feature-support acceptance 为 100%。
- 22 / 23 个 profile/capability cells 的 MMD 和 SWD 同时优于 3-IQR shifted negative。唯一 cell 是 M4 nonlinear 的 MMD：synthetic-vs-real 1.0690，shifted-negative-vs-real 0.9936；同一 cell 的 SWD 仍通过。按预注册的至少 90% 聚合标准，MMD/SWD criterion 通过。
- 所有 115 个 intensity cells 的 strict-risk 与 combined-risk 都为 0。synthetic-to-real raw DCR/p05 的 cell-level q05 最低为 1.8376，context DCR/p05 最低为 1.8353。
- 两轮之间 float64 exact hash、六位小数 hash、MAE ≤ 1e-6 的重复率全部为 0。跨轮 DCR q01 的全局最小值为 0.07514。

这些结果支持：正式在线输出位于真实 control support 内，明显优于刻意 out-of-support 的 negative control，并且没有贴近真实 reference 或跨轮复刻。

## E1.6 简单预测响应

| Capability | Oracle win rate | Oracle / best baseline MAE |
| --- | ---: | ---: |
| `trend` | 0.8156 | 0.8233 |
| `multi_seasonal` | 1.0000 | 0.1227 |
| `time_varying_seasonality` | 0.9839 | 0.2755 |
| `regime_switching` | 0.9776 | 0.4769 |
| `nonlinear_persistence` | 0.9026 | 0.7615 |
| `predictable_intermittency` | 0.8797 | 0.7903 |
| `common_factor` | 1.0000 | 0.3794 |
| `hierarchical_coherence` | 0.9969 | 0.6948 |
| `covariate_response` | 0.7234 | 0.8325 |

九个 capability 均通过 sanity criterion，表明预测期不是纯随机噪声，历史结构与 known-future covariates 确实包含可利用信息。

错误不要求随 intensity 单调。实测中 seasonal-naive 在 multi-seasonal 上从 intensity 1 到 5 增长 2.15 倍，在 time-varying-seasonality 上增长 2.11 倍；但 common-factor 与 predictable-intermittency 的结构增强反而让简单预测更容易。这与 intensity 定义为 realized structure strength、而非统一模型难度一致。

## 结论与下一步建议

E1 已经足以支持 construction predictability、在线真实支持、防近距离复刻、跨轮多样性和结构可预测性。当前不建议立即进入大模型主实验，因为 selectivity 结果会削弱后续“某模型缺少某项独立能力”的归因力度。

建议先做一次不覆盖 E1 v1 的方法审议：

1. 把 construction-level nuisance invariance 与 realized-feature selectivity 分开；前者检查同 seed 下噪声、背景和日程参数不变，后者允许报告 estimator overlap。
2. 从 amplitude-changing capability 的严格 nuisance controls 中移除或替换 target-coupled ratio，例如 `noise_ratio`；任何 feature-gate 变化都重新校准 artifact。
3. 决定论文主张是“disentangled capabilities”还是“capability-focused、允许 overlap 的 stress tests”。若坚持前者，需要调整 trend/change-point、intermittency/nonlinear 等 feature 定义或生成机制，然后发布新实验版本并重跑 E1。
4. 对 M4 regime 的边界性 dose failure，只能用预注册的新 seed confirmatory run 检验，不能修改 v1 阈值或覆盖当前记录。
