# TSBenchmark 真实锚点与合成数据方法

本文面向需要理解、维护或在论文中描述 TSBenchmark 数据方法的读者，系统说明真实数据样本化、真实锚点画像、能力条件化合成、在线验收、离线距离验证以及统一评测链路。所有“当前实现”均以仓库代码为准；设计建议与尚未在线化的实验能力会单独标明。

## 1. 问题边界

TSBenchmark 同时使用真实数据和动态合成数据，但二者承担不同角色：

- 真实数据用于检验模型在实际观测分布上的预测表现。
- 真实锚点画像用于描述现实时间序列的统计范围。
- 合成数据用于隔离趋势、季节性、状态切换、跨变量依赖等具体预测能力。

系统的目标不是复制某条真实序列，而是实现：

\[
\text{真实分布约束}+\text{可解释能力机制}+\text{可复现随机生成}.
\]

完整链路为：

```text
真实 CSV / TsFile
        |
        +--> 解析、校验、滑窗 --> Real Shard -------------------+
        |                                                        |
        +--> 离线特征提取 --> Anchor Profile                    |
                                  |                              |
                                  v                              v
能力配置 + seed --> 候选生成 --> 特征验收 --> Synthetic Shard --> Track
                                                                    |
                                                                    v
                                                     模型推理、指标、报告、榜单
```

## 2. 统一符号

设原始规则时间序列为：

\[
\mathcal D=\{(t_i,\mathbf y_i,\mathbf x_i)\}_{i=0}^{R-1},
\]

其中：

- \(t_i\) 为时间戳；
- \(\mathbf y_i\in\mathbb R^{D_y}\) 为目标变量；
- \(\mathbf x_i\in\mathbb R^{D_x}\) 为协变量；
- \(R\) 为时间点数；
- \(L\) 为上下文长度；
- \(H\) 为预测长度；
- \(S\) 为滑窗步长；
- \(P\) 为季节周期。

一个完整预测样本为：

\[
S_k=(Y_k^{\mathrm{hist}},Y_k^{\mathrm{future}},X_k^{\mathrm{hist}},X_k^{\mathrm{future}}).
\]

模型只能看到历史目标、历史协变量、合法的未来已知协变量和预测长度，不能看到 \(Y_k^{\mathrm{future}}\)。

## 3. 真实数据评测链路

### 3.1 数据接入

上传接口探测 CSV 或 TsFile 的格式、编码、列名、列类型、行数和预览，原始文件保存在 `runtime/uploads/`。随后创建 `DatasetManifest`，记录数据源身份、文件位置、时间列、频率和时区等元数据。

目标列集合 \(\mathcal T\) 和协变量集合 \(\mathcal C\) 必须满足：

\[
\mathcal T\ne\varnothing,\qquad \mathcal T\cap\mathcal C=\varnothing.
\]

CSV 读取器要求选中值可转换为有限浮点数，时间戳唯一、严格递增、时区表示一致且等间隔。令：

\[
\Delta_i=t_i-t_{i-1},
\]

则规则时间轴满足：

\[
\Delta_1=\Delta_2=\cdots=\Delta_{R-1}.
\]

实现入口：

- `backend/app/api/routes/dataset_manifests.py`
- `backend/app/services/csv_dataset_reader.py`
- `backend/app/services/tsfile_dataset_reader.py`
- `backend/app/services/time_axis.py`

### 3.2 滑动窗口

第 \(k\) 个真实窗口起点为：

\[
b_k=kS.
\]

历史区间和未来区间分别为：

\[
I_k^{\mathrm{hist}}=[b_k,b_k+L-1],
\]

\[
I_k^{\mathrm{future}}=[b_k+L,b_k+L+H-1].
\]

窗口数为：

\[
N=\left\lfloor\frac{R-L-H}{S}\right\rfloor+1.
\]

例如小时负荷数据使用 \(L=168,H=24,S=24\)，表示用过去 7 天预测未来 1 天，每次向前移动 1 天。默认 \(S=H\)。如果限制 `max_samples`，系统沿时间轴近似等距抽样，而不是随机抽样。

### 3.3 指针化存储

系统不会为重叠窗口重复保存完整数值，而是使用：

```text
SeriesPoint：每个时间点保存一次
SampleIndex：保存 context / horizon 行号范围
```

第 \(i\) 个 `SeriesPoint` 可表示为：

\[
P_i=(\text{shard\_id},i,t_i,\{c:v_{i,c}\}).
\]

第 \(k\) 个 `SampleIndex` 保存：

\[
I_k=(\text{sample\_id},\text{shard\_id},k,[c_s,c_e],[h_s,h_e],\mathcal T,\mathcal C).
\]

读取时根据 `(shard_id, row_index)` 闭区间查询 `SeriesPoint`，重建 `sample.v1`。每个样本还有不包含随机 ID 的内容 SHA-256，可用于比较相同数据与切片配置下的样本内容。

实现入口：

- `backend/app/services/dataset_load_service.py`
- `backend/app/services/series_store.py`
- `backend/app/services/sample_store.py`

## 4. 真实锚点画像

真实锚点画像不是一条平均时间序列，而是大量真实窗口统计特征的经验分布。

### 4.1 分桶

画像应按频率、变量类型、领域和窗口配置分桶，例如：

```text
hourly_univariate_168ctx_24h
hourly_panel_168ctx_24h
daily_covariate_365ctx_28h
daily_hierarchy_365ctx_28h
```

不同频率、窗口长度或变量语义的数据不应无条件混合，否则特征分位数缺乏清晰含义。

### 4.2 窗口与稳健缩放

真实长序列先按照与目标合成任务匹配的 \(L,H,S\) 切成窗口 \(W_1,\ldots,W_N\)。对单个窗口计算中位数和四分位距：

\[
m=\operatorname{median}(W),\qquad q=Q_{0.75}(W)-Q_{0.25}(W),
\]

并进行稳健缩放：

\[
z_t=\frac{y_t-m}{q}.
\]

若 IQR 过小则退化为标准差。此步骤用于消除不同数据单位和量级对形状特征的影响。

### 4.3 窗口级特征

每个窗口得到特征向量：

\[
\boldsymbol\phi_i=\Phi(W_i).
\]

主要特征包括：

| 类别 | 特征 |
| --- | --- |
| 趋势 | `trend_strength`、`slope_abs`、`curvature_abs` |
| 季节 | `seasonal_strength`、`multi_period_score`、`seasonal_drift_score`、`seasonal_amplitude_cv` |
| 动态 | `acf1`、`acf_abs_mean`、`nonlinear_lag1_gain` |
| 切换 | `level_shift_strength`、`volatility_shift_strength`、`change_point_shift_energy` |
| 异常与噪声 | `noise_ratio`、`outlier_rate`、`spike_rate`、`burst_rate` |
| 多变量 | `avg_abs_target_corr`、`pca_top1_explained`、`pca_top2_explained`、`effective_factor_rank`、`lead_lag_peak_abs` |
| 协变量 | `avg_abs_covariate_target_corr`、`future_abs_covariate_target_corr`、`event_lift_abs` |
| 层级 | `hierarchy_residual_mean_abs` |

趋势与季节强度采用方差解释形式。若 \(z_t=T_t+S_t+R_t\)，则：

\[
F_T=\max\left(0,1-\frac{\operatorname{Var}(R_t)}{\operatorname{Var}(T_t+R_t)}\right),
\]

\[
F_S=\max\left(0,1-\frac{\operatorname{Var}(R_t)}{\operatorname{Var}(S_t+R_t)}\right).
\]

Lead-lag 峰值定义为：

\[
F_{\mathrm{lag}}=\max_{i\ne j}\max_{1\le\ell\le12}|\operatorname{Corr}(Y_{t,i},Y_{t+\ell,j})|.
\]

层级残差定义为：

\[
F_{\mathrm{hier}}=\frac1Q\sum_t\left|Y_{t,0}-\sum_{d=1}^{D-1}Y_{t,d}\right|.
\]

### 4.4 分位数汇总

对每个特征 \(j\)，将所有窗口的值汇总为：

\[
q_p^{(j)}=\operatorname{Quantile}_p(\phi_{1,j},\ldots,\phi_{N,j}),
\]

其中 \(p\in\{0.05,0.25,0.50,0.75,0.95\}\)。最终画像示例：

```json
{
  "trend_strength": {"p05": 0.08, "p50": 0.45, "p95": 0.87},
  "noise_ratio": {"p05": 0.05, "p50": 0.22, "p95": 0.58}
}
```

离线工具 `scripts/synthetic_feature_profile.py` 支持 CSV、Monash TSF、TSF ZIP、多变量 panel、协变量和层级数据。当前 Web 上传流程不会自动注册新画像，在线生成器使用 `synthetic_generation_service.py` 中预置的 `ANCHOR_FEATURE_QUANTILES` 与 `ANCHOR_PROFILE_BUCKETS`。

## 5. 合成数据生成

### 5.1 生成配置

生成任务表示为：

\[
G=(\mathcal C,L,H,N,I,D,s,f),
\]

其中 \(\mathcal C\) 是能力集合，\(N\) 是每个能力的样本数，\(I\in\{1,\ldots,5\}\) 是结构强度，\(s\) 是全局 seed。强度映射为：

\[
\lambda=\frac{I-1}{4}.
\]

`intensity` 表示目标统计结构强度，不表示模型误差必须单调增加。

一次选择多个能力时，系统为每个能力生成独立 Shard，而不是把多种机制混入同一条序列。

### 5.2 周期和维度

周期优先从匹配频率与能力的真实锚点周期中按权重采样；缺少画像时退化为经验默认值：小时为 24、日为 7、15 分钟为 96。

单变量能力强制 \(D_y=1\)；多变量能力使用 \(D_y=\max(2,D_{\mathrm{requested}})\)；协变量响应额外生成 `weather` 和 `event`。

### 5.3 可复现 seed

第 \(k\) 个样本的 seed 为：

\[
s_k=\operatorname{BLAKE2s}(s\Vert c\Vert k)\bmod(2^{32}-1).
\]

相同全局 seed、能力和样本序号产生相同随机流，不同能力互不干扰。

### 5.4 候选生成与标准化

每个样本直接生成 \(Q=L+H\) 个点。例如趋势能力：

\[
y_t=a u_t+b(u_t^2-0.35)+A\sin(2\pi t/P)+\varepsilon_t.
\]

完整样本随后只使用历史段计算均值和标准差：

\[
\mu=\frac1L\sum_{t=0}^{L-1}y_t,\qquad \sigma=\operatorname{std}(y_{0:L}),
\]

并统一变换历史和未来：

\[
\tilde y_t=\frac{y_t-\mu}{\sigma}.
\]

未来真值不参与标准化统计量计算。层级数据使用所有通道共享的尺度，避免破坏父子加总关系；连续协变量按历史段标准化，二元事件变量保持 0/1。

### 5.5 潜在参数与实际特征

生成器同时保存：

- `latent_params`：生成公式使用的斜率、曲率、切换点、耦合强度等参数；
- `realized_features`：对生成结果重新计算的趋势强度、噪声比例、相关性等统计特征。

二者必须区分：设置了非零趋势参数，不保证短序列最终呈现显著趋势，因此在线验收使用 `realized_features`。

### 5.6 有限次拒绝采样

若候选样本未通过在线硬阈值，第 \(a\) 次重试使用：

\[
s_{k,a}=(s_k+104729a)\bmod(2^{32}-1).
\]

最多尝试 12 次。若仍失败，当前实现保留最后候选，并在元数据中记录 `accepted=false` 和失败特征；它不是“未通过就绝不入库”的严格拒绝采样。

### 5.7 合成 Shard

每个样本长度为 \(Q=L+H\)，\(N\) 个独立样本按行拼接：

\[
Y^{\mathrm{all}}=Y_0\Vert Y_1\Vert\cdots\Vert Y_{N-1}.
\]

第 \(k\) 个样本的历史和未来位置为：

\[
[kQ,kQ+L-1],\qquad[kQ+L,(k+1)Q-1].
\]

拼接只是存储形式，样本间不存在自然连续关系。最终仍使用 `SeriesPoint + SampleIndex`，与真实数据共享相同下游接口。

## 6. 各能力在线验收关卡

硬上限通常由多个真实画像的最大 p95 放宽得到：

\[
U_j=m\max_kq_{0.95}^{(j,k)}.
\]

默认 \(m=1.5\)；天然位于 \([0,1]\) 的特征额外截断到 1；状态切换类使用 \(m=2.5\)；事件提升使用 \(m=5\)。

| 能力 | 生成机制 | 在线硬关卡 | 关卡数 |
| --- | --- | --- | ---: |
| 趋势 | 线性趋势、二次曲率、季节残差 | 趋势强度、斜率、曲率、噪声比例、尖峰率均不得超过上限 | 5 |
| 多季节性 | 主周期、次周期、可选三级周期 | 趋势强度、多周期分数、季节强度、噪声比例、尖峰率不得超过上限；季节强度不得低于 0.55 | 6 |
| 时变季节性 | 振幅和相位随时间漂移 | 季节漂移、振幅变异系数、噪声比例、尖峰率不得超过上限 | 4 |
| 状态切换 | 分段水平和波动率 | 变化点能量、水平切换、波动率切换、尖峰率不得超过放宽上限 | 4 |
| 长记忆非线性 | 高持久性 AR 与非线性延续项 | 非线性增益、平均绝对自相关、尖峰率不得超过上限；噪声比例不超过 1 | 4 |
| 间歇异方差 | 随机突发与时变噪声 | 突发率、尖峰率、异常率、趋势强度、季节强度不得超过上限；噪声比例不超过 1 | 6 |
| 公共因子 | 低秩潜在因子与随机载荷 | PCA1、有效因子秩、平均相关性、尖峰率不得超过上限；噪声比例不超过 0.9 | 5 |
| Lead-lag | 跨通道滞后影响 | Lead-lag 峰值、平均相关性、尖峰率不得超过上限；噪声比例不超过 0.9 | 4 |
| 协同状态切换 | 多通道共享变化点 | 水平切换、平均相关性、尖峰率不得超过放宽上限；噪声比例不超过 0.9 | 4 |
| 层级一致性 | 父节点等于子节点之和 | 层级残差、噪声比例、平均相关性不得超过上限 | 3 |
| 协变量响应 | weather/event 影响目标 | 总体与未来协变量相关性、噪声比例、尖峰率、事件提升不得超过上限 | 5 |

当前除多季节性的 `seasonal_strength >= 0.55` 外，大多数能力没有在线目标特征下限。因此当前系统依靠生成公式注入结构、上限防止结构失控，但不保证每个 `accepted` 样本的目标能力都足够显著。层级一致性的目标是残差趋近 0，仅使用上限是合理的。

## 7. 诊断区间与离线距离

### 7.1 真实区间诊断

对控制特征记录：

\[
q_{0.05}^{(j)}\le\phi_j\le q_{0.95}^{(j)}.
\]

结果写入 `inside_anchor_range`，目前通常不改变在线 `accepted` 状态。

### 7.2 单样本新颖性

对标准化后的合成样本 \(s\) 和真实窗口集合 \(\mathcal R\)，计算：

\[
\operatorname{DCR}(s)=\min_{r\in\mathcal R}d(s,r).
\]

仓库离线实验使用原始 MAE、原始 L2 和特征 L2 等距离。另计算最近邻距离比：

\[
\operatorname{NNDR}(s)=\frac{d_1(s)}{d_2(s)}.
\]

DCR 或 NNDR 过小意味着样本可能异常接近某条真实窗口。阈值通过 real holdout 到 real train 的自然最近邻距离低分位数校准，而不是直接设定固定常量。

### 7.3 批量分布距离

整批真实与合成特征分布使用 MMD 和 SWD 比较：

\[
\operatorname{MMD}(\Phi(\mathcal D_{\mathrm{real}}),\Phi(\mathcal D_{\mathrm{syn}})),
\]

\[
\operatorname{SWD}(\Phi(\mathcal D_{\mathrm{real}}),\Phi(\mathcal D_{\mathrm{syn}})).
\]

其目标是同时满足：

\[
\text{整批分布不过远}+\text{单个样本不过近}.
\]

当前 DCR、NNDR、MMD 和 SWD 均为离线验证项，不会触发 Web 在线生成的重采样或 Shard 回滚。

## 8. 真实与合成数据的统一评测

真实和合成数据最终都表示为：

```text
Shard
  +-- SeriesPoint
  +-- SampleIndex
```

读取后均得到 `sample.v1`。服务端完整样本包含未来真值，但模型输入构造函数会移除 `target_future`：

\[
\mathcal I_k=(Y_k^{\mathrm{hist}},X_k^{\mathrm{hist}},X_k^{\mathrm{future}},H).
\]

模型预测：

\[
\hat Y_k^{\mathrm{future}}=f_m(\mathcal I_k).
\]

服务端使用隐藏真值计算 MASE、MSE 和 MAE，并按 `Sample -> Shard -> Task -> Unit` 聚合。Shard 经 `CapabilityBlock` 组成 Track，因此可同时报告真实数据成绩和趋势、季节、状态切换等能力成绩。

## 9. 当前实现与论文表述边界

### 9.1 已实现

- 真实 CSV/TsFile 解析、规则时间轴校验和滑窗样本化；
- `SeriesPoint + SampleIndex` 指针化存储；
- 离线真实数据特征 profiler；
- 预置锚点分位数和周期画像；
- 11 类能力条件化生成器；
- 上下文条件标准化、实际特征计算和有限次在线验收；
- 生成参数、样本 seed、实际特征和验收结果追踪；
- 真实与合成数据统一推理、指标、报告和榜单链路。

### 9.2 尚未在线闭环

- Web 上传真实数据后自动生成并注册 Anchor Profile；
- 大多数能力的目标特征在线下限；
- 缺失或非有限必要特征的统一硬拒绝；
- 在线 DCR/NNDR 新颖性拒绝；
- 在线批量 MMD/SWD 发布门禁；
- 12 次验收失败后阻止不合格样本进入 Shard。

因此论文中宜使用以下表述：

> TSBenchmark 采用能力条件化参数随机过程生成合成样本，并利用预计算真实数据画像选择季节周期、构造特征上限以及记录生成后控制特征范围。在线生成执行有限次特征约束验收；DCR、NNDR、MMD 和 SWD 当前作为离线新颖性与分布验证项。

不宜写成：

> 系统已根据每次上传的真实数据自动训练生成器，并保证所有在线合成样本均通过新颖性和分布距离门禁。

## 10. 后续可执行改进

1. 为每个能力定义随 intensity 变化的目标特征下限 \(L_{c,j}(I)\)。
2. 将必要特征缺失或非有限统一视为验收失败。
3. 将 `accepted=false` 样本排除在可发布 Shard 之外，或将 Shard 标记为部分生成失败。
4. 将真实画像版本、数据来源、窗口配置和特征实现版本持久化。
5. 将 DCR/NNDR 接入样本级发布门禁，将 MMD/SWD 接入批次级门禁。
6. 使用时间块或数据集级拆分校准距离阈值，避免重叠窗口造成训练与 holdout 近重复。
7. 报告每个能力和 intensity 的接受率、尝试次数、特征分布及 naive 基线响应。

## 11. 代码与实验索引

| 内容 | 位置 |
| --- | --- |
| 合成能力、生成器、阈值与验收 | `backend/app/services/synthetic_generation_service.py` |
| 真实数据读取 | `backend/app/services/csv_dataset_reader.py`、`tsfile_dataset_reader.py` |
| 时间轴校验 | `backend/app/services/time_axis.py` |
| 窗口构造与真实 Shard 加载 | `backend/app/services/dataset_load_service.py` |
| 序列与样本存储 | `backend/app/services/series_store.py`、`sample_store.py` |
| 模型输入隔离 | `backend/app/services/model_input.py` |
| 离线真实画像 | `scripts/synthetic_feature_profile.py` |
| 在线验收 sweep | `scripts/run_synthetic_v2_acceptance_sweep.py` |
| DCR/NNDR 校准 | `scripts/run_synthetic_v2_near_distance_calibration.py` |
| 方法设计与能力契约 | `docs/superpowers/specs/2026-06-29-synthetic-v2-capability-contracts.md` |

