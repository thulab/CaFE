# E1：Synthetic method validity 实验协议

日期：2026-07-16

## 目的与边界

E1 验证逐 dataset 合成方法是否实现了预注册的相对强度、选择性、可预测性、真实控制
特征支持与防复刻约束，并用简单预测器验证生成结构确实可利用。本实验不调用时序大模型，
也不比较模型排名。

真实数据与合成测试的对应单位是：

```text
dataset_id × profile_id × capability_id
```

每个 dataset 独立拟合 nuisance、五档目标、generator conditioning、feature-support gate
和 near-distance gate。不同 dataset 的窗口与标定产物不得混用。

正式输出使用新的版本目录。runner 不覆盖已有文件；方法、输入 artifact、支持矩阵、样本量
或判据变化时必须创建新目录。旧跨数据集聚合强度实验记录只作历史审计，不进入本版结论。

## 冻结输入

- dataset profile suite：每个 dataset 独立形成 profile，不含跨数据集聚合 profile；
- generator conditioning：只接受 dataset-local real-bounded、generator-feasible
  relative-intensity schema；
- dataset capability support matrix：状态为 `supported` 的 cell 才进入 E1；
- feature-support 与 near-distance artifact：必须来自
  `runtime/paper_exp/v4/01_nine_capability_suite/`，并与相同
  `dataset_id/profile_id` 精确匹配；
- intensity：1--5 分别对应 parameter split 的
  `[q05, 1.2×q95]` 真实容忍区间与生成器响应区间交集内的
  `0/0.25/0.50/0.75/1.00` 相对位置；
- 每个 supported cell 运行两轮独立根 seed；
- 每轮每个 `dataset × profile × capability × intensity` 生成 64 个最终样本；
- 同一 `dataset × profile × capability × round × sample_index` 的五档复用同一 sample
  seed，以配对检验 nuisance 漂移；
- 两轮之间 seed 完全独立；
- 所有样本走 construction、feature-support 与 near-distance 全链路，不允许 fail-open。

若 dataset 缺少能力要求的变量结构、五档主特征没有足够间距、inverse calibration 失败，
或必要 gate 无法标定，该 cell 必须记录为 `unsupported`。它不生成样本、不计作实验失败，
也不进入通过率分母。

总样本量由冻结后的 supported cell 数 \(N_{\rm supported}\) 决定：

\[
N_{\rm sample}=N_{\rm supported}\times5\times64\times2.
\]

## 分析与预注册判据

### E1.1 Dataset-local dose-response

对每个 supported `dataset/profile/capability/intensity` 统计 primary realized feature 的
均值、标准差和 p05/p95，并与该 profile conditioning 中对应的
五个可行 target 比较。

目标误差使用该 cell 自己的 target span 归一化：

\[
e_{d,p,c,k}
=
\frac{|\bar f_{d,p,c,k}-T_{d,p,c,k}|}
{T_{d,p,c,5}-T_{d,p,c,1}}.
\]

target span 在进入 E1 前已经通过最小间距审计，因此分母不得在 E1 中临时替换或放宽。
每个 supported cell 同时满足：

- 五档 realized mean 对 intensity 的 Spearman 不低于 0.90；
- intensity 5 realized mean 高于 intensity 1；
- 五档最大 normalized absolute error 不超过 0.25。

Intensity 只表示该 dataset/profile 内从相对弱到相对强。不同 dataset 的 I3 不要求具有
相同 realized feature 数值。

### E1.2 非目标特征选择性

硬判据只使用 capability 预注册的 control features。对配对 seed 计算 intensity 5 减
intensity 1，并使用同一 dataset/profile feature gate 的 IQR 标准化。每个 control
feature 的 median absolute paired shift 不超过 1.0 IQR，absolute median signed shift
不超过 0.5 IQR。

另输出完整 primary-feature response matrix：行是被改变 intensity 的 capability，列是
该 dataset/task view 中所有可计算的 capability primary features。对角与非对角
Spearman 只作诊断，不根据结果删改特征。

### E1.3 Construction predictability

记录每个样本的 capability-specific contract 与 evidence。正式样本必须 100%
`construction_validated=true`；supported 配置的 contract 失败会终止实验，不通过重采样
绕过。

### E1.4 Dataset-local control-feature support

逐 intensity cell 的首轮完整链路通过率不得低于 95%，最终 feature-support acceptance
必须为 100%。

直接读取 feature-gate artifact 中随冻结 split 保存的 dataset-local gate-reference 与
gate-calibration control vectors。E1 不再从原始数据重建 split，避免代码或抽样参数变化
造成对照漂移。vectors 已按该 artifact 的 center/IQR 标准化。分别计算 real-vs-real、
synthetic-vs-real，以及将 real calibration 整体平移 3 IQR 的 shifted negative control
的 fixed-bandwidth RBF-MMD 与 128-projection SWD。

若某能力的构造会机械改变所有可计算观测量，因而没有合法 nuisance control，该项如实记录
为 `not_applicable_no_control_features`，不伪造零维距离，也不计入本小节分母。至少 90%
的可评估 supported cells 必须在 MMD 和 SWD 上都比 shifted negative 更接近其自身真实
reference；若没有任何可评估 cell，本小节不能通过。real-vs-real 只作采样基线；不要求
synthetic 与真实分布不可区分。

### E1.5 Dataset-local DCR/NNDR 与跨轮重复

保存同一 dataset/profile near-distance gate 的 raw/context DCR、NNDR 及相对 p05
阈值。最终样本 strict-risk、combined-risk 必须均为 0，novelty acceptance 为 100%。

在相同 `dataset/profile/capability/intensity` 内计算第二轮到第一轮的最近标准化 MAE
与 NNDR，并检查 float64 精确哈希、六位小数哈希和 MAE ≤ `1e-6`。三种重复率都必须为
0。不得跨 dataset 搜索最近邻后声称某个 dataset 通过。

### E1.6 简单预测响应

每个样本运行 last-value naive、按该 profile season length 的 seasonal-naive，以及只
使用 history、known-future covariates 和生成机制日程信息的 capability oracle。oracle
不得读取 target future；测试以替换 future 后预测不变来约束。

先在 `dataset/profile/capability` 内计算 oracle 相对两个 baseline 中较优者的 MAE win
rate，再跨 dataset 汇总。每个 supported cell 的 sanity criterion 为 win rate 不低于
50%。同时按 intensity 输出三类预测器 MAE，用于观察相对结构强度响应；该结果不作为
大模型能力结论。

## Unsupported 审计

E1 报告必须包含完整 support matrix：

- `dataset_id`、`profile_id`、task view、capability；
- `supported` 或 `unsupported`；
- 不支持的 reason codes；
- 若进入 calibration，保存 target spacing、单调性、目标误差和 gate 失败信息。

不支持的 cell 不得被填零、补最差结果或从报告中静默删除。论文中每项能力的 dataset
数量以该矩阵为准。

## 保留文件

- `config.json`：冻结配置、dataset/profile 列表、逐数据集九能力支持矩阵引用与全部判据；
- `samples.jsonl`：逐样本 dataset/profile 身份、相对分位档位、realized features、
  construction evidence、门控统计和简单预测器 MAE；
- 聚合 CSV：dose-response、selectivity、construction、support、MMD/SWD、DCR/NNDR、
  跨轮重复和 baseline/oracle；
- `dataset_capability_support_matrix.csv`：包含所有 supported/unsupported cells；
- `summary.json`、`report.md`：机器可读和人可读结论；
- `manifest.json`：代码 commit、runner、输入与输出文件的 SHA-256 和大小。
