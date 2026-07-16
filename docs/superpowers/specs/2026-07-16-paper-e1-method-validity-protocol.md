# E1：Synthetic method validity 实验协议

日期：2026-07-16

## 目的与边界

E1 验证 paper-v1 合成方法本身是否实现了预注册的强度、选择性、可预测性、真实控制特征支持与防复刻约束，并用简单预测器验证生成结构确实可利用。本实验不调用任何时序大模型，也不比较模型排名。

正式输出固定保存在 `runtime/paper_exp/v1/E1_method_validity/`。该目录视为不可变实验记录：runner 不覆盖任何已有文件；若方法、输入 artifact、样本量或判定标准改变，必须使用新的实验版本目录。目录保留配置、输入哈希、逐样本统计、聚合 CSV、报告和 manifest；完整曲线可由 artifact、seed 与代码重建，因此不重复保存。

## 冻结输入

- generator conditioning：`synthetic-v2-paper-v1-frozen-2026-07-16`，fingerprint `a76b66924562be4f`。
- 在线集合：8 个 profile、23 个 `profile × capability` cells；2048-context research-only profile 不进入 E1。
- intensity：1--5 全部运行。
- 两轮独立根 seed：`2026071601`、`2026071602`。
- 每轮每个 `profile × capability × intensity` 64 个样本，共 `23 × 5 × 64 × 2 = 14720` 个最终样本。
- 同一 `profile × capability × round × sample_index` 在五档 intensity 复用同一 sample seed，以配对检验 nuisance 漂移；两轮之间 seed 完全独立。
- 所有样本走正式 construction、feature-support 与 near-distance 全链路，不允许 fail-open。

## 分析与预注册判据

### E1.1 Canonical dose-response

对每个 profile/capability/intensity 统计 primary realized feature 的均值、标准差和 p05/p95，并与 artifact 中的 canonical target 比较。以该 capability 的 canonical range（下限 0.05）归一误差。

每个 profile/capability 同时满足以下条件才通过：五档均值对 intensity 的 Spearman 不低于 0.90；intensity 5 均值高于 intensity 1；五档最大 normalized absolute error 不超过 0.25。

### E1.2 非目标特征选择性

硬判据只使用每个 capability 预注册的 control features，避免把目标结构本身误称为 nuisance。对配对 seed 计算 intensity 5 减 intensity 1，并用真实 feature-gate IQR scale 标准化。每个 control feature 的 median absolute paired shift 不超过 1.0 IQR，且 absolute median signed shift 不超过 0.5 IQR。

另输出完整 primary-feature response matrix：行是被改变 intensity 的 capability，列是所有可计算的 capability primary features。对角与非对角 Spearman 作为诊断报告，不根据结果删改特征。

### E1.3 Construction predictability

记录每个样本的 capability-specific contract 与 evidence。正式样本必须 100% `construction_validated=true`；任何配置级失败都会终止实验，而不是重采样绕过。

### E1.4 Control-feature support 与 MMD/SWD

逐 intensity cell 的首轮完整链路通过率不得低于 95%，最终 feature-support acceptance 必须为 100%。

使用冻结 feature-gate 的同一真实 split 重建 gate-reference 与 gate-calibration control vectors，并按 artifact center/IQR 标准化。分别计算 real-vs-real、synthetic-vs-real，以及把 real calibration 整体平移 3 IQR 的 shifted negative control 的 fixed-bandwidth RBF-MMD 与 128-projection SWD。至少 90% 的 profile/capability cells 必须在 MMD 和 SWD 上都比 shifted negative 更接近真实 reference。real-vs-real 仅作采样基线；本实验不要求 synthetic 与真实分布不可区分。

### E1.5 DCR/NNDR 与跨轮重复

保存正式 near-distance gate 的 raw/context DCR、NNDR 及相对 p05 阈值。最终样本 strict-risk、combined-risk 必须均为 0，且 novelty acceptance 为 100%。

在相同 profile/capability/intensity 内计算第二轮到第一轮的最近标准化 MAE 与 NNDR，并检查 float64 精确哈希、六位小数哈希和 MAE ≤ `1e-6`。三种重复率都必须为 0。

### E1.6 简单预测响应

每个样本运行：last-value naive、按 profile season length 的 seasonal-naive、以及只使用 history、known-future covariates 和生成机制日程信息的 capability oracle。oracle 绝不读取 target future；测试会以替换 future 后预测不变来约束。

对每个 capability 汇总 oracle 相对两个 baseline 中较优者的 MAE win rate。预注册 sanity criterion 为每个 capability 的 win rate 不低于 50%。同时按 intensity 输出三类预测器 MAE，用于观察结构强度响应；该结果不作为大模型能力结论。

## 保留文件

- `config.json`：冻结配置与全部判据。
- `samples.jsonl`：逐样本 realized features、construction evidence、门控统计和三个简单预测器 MAE。
- 九张聚合 CSV：dose-response、selectivity、construction、support、MMD/SWD、DCR/NNDR、跨轮重复、baseline/oracle。
- `summary.json`、`report.md`：机器可读和人可读结论。
- `manifest.json`：代码 commit、runner 与三个输入 artifact 的 SHA-256、全部输出文件的 SHA-256 和大小。
