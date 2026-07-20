# Paper v7 P0 数据集准备与准入记录

- 日期：2026-07-21
- 范围：结构化真实数据预处理与 loader 准入；不含模型结果
- 决策：`accepted for v7 calibration`

## 1. 冻结数据集

v7 首批新增 Swiss Hierarchical Demand 与 GEFCom2012 Load，并复用已有
ETT1、M5、GEFCom2014 Wind 资产补齐结构化 task views。准入在查看 v7 模型结果前
完成，不根据后续排名或 synthetic-real alignment 事后删数据。

| Dataset | Common factor | Strict hierarchy | Known-future covariates |
|---|---|---|---|
| ETT1/H | 原生同步 7 通道，canonical d=3 | — | — |
| M5 Daily | 同 store/dept 的 3 个 leaf siblings | 原有 canonical 3 节点 | weekday、event、SNAP |
| GEFCom2014 Wind | 10 zones，canonical d=3 | — | 单个官方 TaskExpVars release 的 4×3 NWP |
| Swiss Hierarchical Demand | 24 meters，canonical d=3 | `all=S1+S2` | 单一 issue-time 的 6 变量、24h NWP |
| GEFCom2012 Load | 20 zones，canonical d=3 | `total=subtotal_1_10+subtotal_11_20` | 6 个确定性日历/假日字段 |

M5 不使用 `sell_price` 或 `price_change`。GEFCom2012 不使用观测温度。
GEFCom2014 Wind 与 Swiss 的正式 benchmark horizon 都禁止拼接预测起点之后发布的
新天气预报。

## 2. 原始文件身份

原始文件保存在 ignored runtime，仓库只提交可复现处理和校验代码。

| File | Bytes | SHA-256 |
|---|---:|---|
| `swiss/power_data.p` | 179,139,915 | `635357fe9c2fb7a57b89cbe93c071f7e5ba6a61d4418f87571f178762fa42cde` |
| `swiss/nwp_data.h5` | 72,848,328 | `2d9c5a5a6d16e501acd5e510c7c2531e7aa08610d495ce441deeaf9ef3a0d2b6` |
| `gefcom2012/GEFCom2012.zip` | 11,630,887 | `028529ad01bd417648d11c4b99902d82a760a0b54c8597975833ae9ac061be9c` |

Swiss 来自 Zenodo record 3463137，许可证为 CC BY 4.0；两个发布文件的官方 MD5
也已通过。GEFCom2012 官方压缩包未包含明确许可证文件，本轮研究使用已由用户在
2026-07-21 明确确认；该确认不等同于授予重新分发原始压缩包的权利。

## 3. 预处理结果

处理脚本：

`scripts/prepare_paper_v7_p0_datasets.py`

所有输出 NPZ 仅含数值或 `datetime64` 数组，并实际以 `allow_pickle=False` 重新读取
验证。

| Dataset | Time rows | Arrays | Processed SHA-256 |
|---|---:|---|---|
| Swiss | 17,854 × 30min | 24 meters；7 aggregates；6×24 NWP cube 与显式 valid-time grid | `89f0f563a525c8eb3d60a9e4b2281dc8d9a26d8749834ef4875937acdd5683a5` |
| GEFCom2012 | 38,070 × hourly，9 个连续段 | 20 zones；total；3-node hierarchy；6 calendar covariates；segment IDs | `b09373a3a4b6412aee69745e8d1de5ae05b6af560f323a6dd5e9bd8205e79236` |

对应 metadata SHA-256 分别为 Swiss
`f8e3017e59cfde790e9611729cfac99fee89c8ae394685ef2eadc974b3d29978` 与
GEFCom2012
`833a44235cbf92bed03a9fd0b0bbeafb5de00a78796f3812d232f37d64e4f159`。

Swiss 只保留完整的三个 10 分钟观测组成的 30 分钟 mean bin；首个没有 prior NWP
的 bin 被删除。原始和聚合后层级最大绝对误差分别为 `4.55e-13` 和 `3.41e-13`。
每个目标时点同时保存最新 NWP 的 as-of timestamp 和 24 个 lead 的 valid-time grid。
历史最后一个 bin 观测完成后，forecast origin 是 `origin+1` 的半小时 bin start；
H48 只使用该 origin 的同一 24 小时 forecast cube，逐小时 lead 各映射到两个半小时
槽，并逐槽验证 valid time，不拼接后续发布。

GEFCom2012 的 target **只来自 `Load_history.csv`**；`Load_solution.csv` 从不回填
target，只用于审计 Zone 21 与 zones 1--20 之和的恒等式。所有任一 zone 缺失的
1,530 个小时（30,600 cells）均被排除，其中含 1,512 个官方隐藏评测小时和源文件末尾
不完整的 18 小时。剩余 38,070 小时冻结为 9 个严格连续 hourly segments，任何
profile、gate、real-source 或生成窗口都不得跨段。官方 solution 的 Zone 21 审计
误差和 canonical 两子节点层级误差均为 0。

## 4. Loader 准入审计

使用 `C=504`、loader `H=48+48 embargo`、`max_windows=120`：

| Dataset/task view | Master windows | L96 | L168 | L336 | L504 |
|---|---:|---:|---:|---:|---:|
| Swiss/common_factor | 120 | 120 | 120 | 120 | 120 |
| Swiss/hierarchy | 120 | 120 | 120 | 120 | 120 |
| Swiss/covariate | 120 | 120 | 120 | 120 | 120 |
| GEFCom2012/common_factor | 120 | 120 | 120 | 120 | 120 |
| GEFCom2012/hierarchy | 120 | 120 | 120 | 120 | 120 |
| GEFCom2012/covariate | 120 | 120 | 120 | 120 | 120 |

Swiss validation embargo 可以由 benchmark 之后的 as-of 行填充以完成数据三路拆分，
但这些值不会进入正式 H48 feature、生成或推理视图。所有 source rows 和 online
near-distance buckets 保存 target/covariate 列序、处理文件哈希、group/window 身份、
层级来源和 forecast issue-time 审计字段。

## 5. 已有资产修正

- ETT1 factor 不再把顺序读取的独立序列误当同步 panel；使用 Arrow 中同一 item 的
  原生 7 通道，canonical 取前三通道。
- M5 factor 只选择互不重叠的 leaf siblings；hierarchy、factor、covariate 共用
  canonical `dataset_id=m5_daily`，以 `task_view_id` 区分。
- GEFCom2014 Wind covariate 每个窗口只来自一个官方 monthly TaskExpVars release。
  H48 不完整、时间不连续或 target/NWP 含缺失的 release 自动 fail closed。

这些处理结果满足进入 v7 正式 dataset-local calibration 的数据准入条件。隔离的
`max_windows=120`、`calibration_samples=4` stage-build 也已完成：Swiss 与
GEFCom2012 的 `common_factor`、`hierarchical_coherence`、
`covariate_response` 共六个 cells 全部为 `supported`。每个 task view 的
L96/L168/L336/L504 online reference counts，Swiss 为 `44/43/43/42`，GEFCom2012
为 `44/44/43/43`；两个 L504 covariate buckets 的协变量 reference shape 分别为
`(42,552,6)` 与 `(43,552,6)`。GEFCom2012 的 120 个 source masters 覆盖全部
segments 0--8；三路时间分割后的 reference buckets 覆盖 segments 3--8。

小试产物位于
`runtime/paper_exp/v7/01_nine_capability_suite_p0_temporal_embargo_audit/`，support
matrix SHA-256 为
`24ac9be63e46445798b42981bd0d35d47f3c89ced862772432f7bbcb1716fbaf`。
正式 v7 suite 仍会用冻结的正式预算重跑校准，并写出最终 support matrix。
