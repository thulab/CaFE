# CaFE 十能力真实路径锚定公式设计

> 状态：设计提案，非当前 canonical protocol  
> 日期：2026-08-12  
> 适用目标：把 CaFE 从“用真实特征校准合成机制”扩展为“在真实路径上执行可审计的能力反事实干预”

## 1. 结论摘要

十种能力不能机械地套用同一个合成模板。推荐分成三类：

| 类型 | 能力 | 反事实形式 |
|---|---|---|
| 单变量可加分量 | local nonlinear trend、independent multi-seasonality、carrier amplitude modulation、observed persistent level shift、predictable recurrent intermittency | `real path + scaled fitted component` |
| 结构可加分量 | forecastable common factor、zero-sum hierarchical contrast、directed cross-series predictive transfer、known-future conditional predictive response | `real structured path + scaled fitted structural component` |
| 递归参数干预 | nonlinear autoregressive persistence | 改变递归算子、共享 history innovation，再把两条 history-only rollout 的差加到真实 future |

六个尚未接入真实路径的能力，建议如下：

| 能力 | 是否应优化并接入 | 结论 |
|---|---|---|
| nonlinear persistence | 是，但单独开发 | 不能用静态加法冒充 persistence；需要递归 contract 和专用 validator，初期标记 experimental |
| predictable intermittency | 是，优先 | 可做真实事件 clock + empirical pulse template；clock predictability 必须成为硬门 |
| common factor | 条件接入 | 需要原生同步 panel、可预测 factor continuation，并强制配套 target-only / donor ablation |
| hierarchical coherence | 公式可做，主表暂缓 | zero-sum contrast 能严格保 hierarchy；但高剂量可能破坏非负计数 support，需先冻结协议取舍 |
| cross-series dependence | 条件接入 | 只能严谨地称 directed predictive transfer，不能把观察性 lag 关系写成 causal SCM |
| covariate response | 是，优先 | 只接受 adapter 明确声明的 known-future covariate；控制 response coefficient，不缩放 covariate 本身 |

现有四项也不是完全不需要调整。下一协议版本应先解决：

1. 每个真实 background 只冻结一次共享 full decomposition 和 component ownership；
2. independent secondary、carrier harmonic 和 AM sideband 互斥归属；
3. trend 增加 fixed-L168 可见强度门；
4. regime 增加 step-vs-ramp 与 joinpoint 稳定性门；
5. time-varying seasonality 使用受约束的纯 AM basis；若保留当前自由 sideband basis，应改名为 `carrier sideband variability`。

## 2. 统一研究语义

real-anchored 轨测量的是：

> 在保留真实数值路径、真实 residual 和真实 held-out nuisance realization 的前提下，只改变一个由历史可识别、并可由历史或合法 known-future input 外推的预测机制后，模型能否恢复声明的 forecast effect。

它不自动证明真实世界因果效应。尤其是：

- 单变量 nonlinear 只说明 autoregressive forecasting recurrence；
- cross-series 只说明 lagged predictive transfer；
- promotion/weather response 只说明 conditional predictive response；
- common factor 的原始 effect recovery 不足以证明模型使用了辅助通道，必须配套输入消融。

## 3. 统一记号和硬约束

令预测起点为 $o$：

\[
H_{fit}=X_{o-504:o},\qquad
H_{model}=X_{o-336:o},\qquad
Y^{real}=X_{o:o+48}.
\]

其中 $X_t$ 可以是单变量、同步 panel 或 hierarchy。所有 normalization 与 MASE reference 都从未修改的 $H_{model}$ 冻结。当前剂量网格沿用：

\[
\alpha\in\{1,1.2,1.4,1.6,1.8,2.0\}.
\]

### 3.1 可加分量的统一公式

若能力 $c$ 有一个从 L504 history 拟合并可合法外推的分量
$\widehat M_t^{(c)}$，则：

\[
\boxed{
X_t^{(c,\alpha)}
=X_t+(\alpha-1)\widehat M_t^{(c)}
}
\]

所以：

\[
X_t^{(c,1)}=X_t,
\qquad
\Delta Y_h^{(c,\alpha)}
=(\alpha-1)\widehat M_{o+h}^{(c)}.
\]

原真实 H48 没有被模型重建，而是作为 pair-shared nuisance 保留：

\[
Y_{o+h}^{(c,\alpha)}
=Y_{o+h}^{real}+\Delta Y_h^{(c,\alpha)}.
\]

### 3.2 所有能力的共同不变量

1. component、period、lag、joinpoint、loading、coefficient、event clock 和 availability 只由 target L504 history确定。
2. 只有 adapter 明确声明的 known-future covariate 可以参与 H48 component；target future 不能参与。
3. 任意改写 $Y^{real}$ 都不得改变 contract、availability 或 truth delta。
4. $\alpha=1$ 必须逐点返回真实 baseline。
5. 同一 background 的所有 member 共用 baseline L336 normalization 和 MASE，禁止逐 member z-score。
6. 单变量的统计单位是不同真实 background；panel 是不同 panel-origin；hierarchy 至少按 structural group-origin 聚类。
7. 至少四个 eligible 真实统计单位才允许一个 `dataset × capability` 进入该轨；不得用 seed 重复包装同一 background。
8. history、fixed-L168 可见区和 H48 的 controlled effect 都要有非退化强度。当前 1% baseline-scale gate 可作为统一下限，新增能力应先 qualification 再冻结更严格阈值。
9. 任何需要真实 H48 target 才能选择分量、剂量、方向或 clip 的方案均不合格。

## 4. 共享分解与 component ownership

单变量 background 建议一次性冻结：

\[
x_t=
L_t+T_t^{nl}+C_t+S_t^{sec}+M_t^{amp}
+R_t^{level}+E_t^{pred}+\varepsilon_t.
\]

各项含义为：

- $L_t$：level 与 local linear tangent；
- $T_t^{nl}$：最后 W96 的 nonlinear trend；
- $C_t$：固定 carrier；
- $S_t^{sec}$：独立 secondary seasonalities；
- $M_t^{amp}$：carrier amplitude modulation；
- $R_t^{level}$：已观察到的 persistent level shift；
- $E_t^{pred}$：可由稀疏 clock 预测的 recurrent event；
- $\varepsilon_t$：其余真实 residual。

这不是要求每条真实路径都必须拥有所有分量。每一项都要单独通过 history-only eligibility；没有证据就不创建该 component。

建议冻结一个 background-level ownership map：

```text
frequency / basis column
  -> carrier harmonic
  -> independent secondary
  -> symmetric AM sideband pair
  -> rejected / unresolved

time-local structure
  -> local nonlinear trend
  -> abrupt persistent step
  -> recurrent sparse event
  -> residual
```

该 map 必须互斥。其他已识别 component 在某项能力中只是固定 nuisance，不能因为当前被评能力不同而从 joint fit 中消失。

当前实现离这一目标还有三个具体差距：

- trend / multi fit 没有总是吸收已检测到的 modulation 与 regime nuisance；
- regime fit 没有总是吸收 modulation nuisance；
- modulation envelope 提取前尚未先去掉 independent secondary，beat 可能被误认成 AM。

## 5. 十能力公式

### 5.1 Local nonlinear trend continuation

兼容 capability id：`trend`。

令：

\[
u_t=\left[\frac{t-(o-96)}{96}\right]_+,
\qquad
\widehat T_t^{nl}=\sum_{k=2}^{K}\beta_k u_t^k.
\]

当前实现 $K=2$。若未来允许 cubic，必须预先冻结选择策略并用 chronological holdout，而不能按 H48 表现选择。

反事实为：

\[
\boxed{
x_t^{(\alpha)}
=x_t+(\alpha-1)\widehat T_t^{nl}
}
\]

因为 $u=0$ 时二阶及以上基的值和一阶导都为 0，干预不改变 $o-96$ 处的 level 与 linear tangent。它测的是 trailing-W96 local curvature continuation，不是“整条 L504 趋势强度”。

优化项：

- 增加 trailing-L168 controlled-component RMS；只用包含大量前缀零值的 L504 RMS 不足以说明模型可见；
- joint fit 固定 carrier、secondary、AM、step 和 event nuisance；
- 与 abrupt step、slope break、ramp 做 history-only 模型比较，避免 curvature 吸收结构突变；
- 保持 raw baseline、shared normalization 和 exact alpha proportionality。

成熟度：已实现，需小幅协议修订。

### 5.2 Independent multi-seasonality

兼容 capability id：`multi_seasonal`。

保留主 carrier $C_t$，只缩放彼此可辨识、且不属于 carrier harmonic 或 AM sideband 的 secondary：

\[
\widehat S_t^{sec}
=\sum_{j=1}^{J}\sum_{h=1}^{H_j}
\left[
a_{jh}\sin(2\pi ht/P_j)
+b_{jh}\cos(2\pi ht/P_j)
\right].
\]

\[
\boxed{
x_t^{(\alpha)}
=x_t+(\alpha-1)\widehat S_t^{sec}
}
\]

这比 $T+\alpha S+R$ 更准确：后者放大的是总季节性，不能保证 multi-seasonality 增强。这里主 carrier 始终固定。

优化项：

- carrier 本身必须通过 visibility gate，而不只是 period 在 L504 中够三周期；
- $P_j$ 至少在 L504 中出现三周期；
- 排除 carrier 的整数 harmonic；
- 先联合判别 symmetric AM sideband，再决定独立 secondary，避免 beat/AM 冒充第二季节；
- 保存每个 secondary 的独立 component 与总和，检查主 carrier 能量在 pair 间不变；
- fixed-L168 至少要暴露可辨识的 secondary history，不能只有 L504 前缀有信号。

成熟度：已实现基本公式，但 spectral ownership 需修订后再扩大主张。

### 5.3 Carrier amplitude modulation

兼容 capability id：`time_varying_seasonality`。

建议把能力收窄为振幅调制，不同时测试 phase modulation。令固定 carrier 为：

\[
C_t=A_0\cos(\omega_ct+\phi_0),
\]

慢包络为：

\[
m_t=u\cos(\omega_mt)+v\sin(\omega_mt),
\qquad \omega_m<\omega_c,
\]

受控 AM component 为：

\[
\widehat M_t^{amp}
=m_t\cos(\omega_ct+\phi_0).
\]

反事实：

\[
\boxed{
x_t^{(\alpha)}
=x_t+(\alpha-1)\widehat M_t^{amp}
}
\]

等价于固定 carrier，只把 envelope deviation 从 $m_t$ 放大到 $\alpha m_t$。

优化项：

- 上下 sideband 必须共享 carrier phase 并满足对称 AM 约束；
- envelope 拟合前移除 trend、independent secondary 和 persistent step；
- carrier、modulation 在 L504 中分别通过强度与最少周期门；
- alpha 最大剂量下 envelope 不应发生非预期的 carrier sign flip；
- 若继续使用当前四个自由 sideband 系数，它们可同时表示 amplitude 与 phase modulation，应把论文名称改成 `carrier sideband variability`。

成熟度：已实现自由 sideband 版本；纯 AM 公式需要修订。

### 5.4 Observed persistent level shift

兼容 capability id：`regime_switching`。

先在去除共同 nuisance 后，从 fixed-L168 可见区检测已发生的 joinpoint：

\[
\widehat\tau\in[o-144,o-24].
\]

受控 component：

\[
\widehat R_t^{level}
=\widehat\beta\,\mathbf 1[t\ge\widehat\tau].
\]

反事实：

\[
\boxed{
x_t^{(\alpha)}
=x_t+(\alpha-1)\widehat\beta
\mathbf 1[t\ge\widehat\tau]
}
\]

它只增强已观察到的 level jump，并把新水平常数延伸到 H48；不假定 future 会新发生一次 regime，也不控制 variance、slope 或 Markov switching rate。

优化项：

- abrupt step 必须优于 ramp、slope break 和局部 trend alternative；
- joinpoint 在 chronological history folds 中稳定；
- pre/post 局部段各自稳定且有足够长度；
- 已识别 carrier、secondary 和 AM 固定为 nuisance；
- 如论文仍使用 `regime switching` 标签，正文必须说明实际 estimand 是 persistent level-shift continuation。

成熟度：已实现，需补边界与稳定性 gate。

### 5.5 Nonlinear autoregressive persistence

兼容 capability id：`nonlinear_persistence`。

这是唯一不应套用固定 $(\alpha-1)M$ 曲线的能力。先去掉冻结的 deterministic nuisance，并用 baseline L336 scale 标准化：

\[
u_t=\frac{x_t-\widehat D_t}{s_{336}}.
\]

候选 nonlinear lag 使用预先声明集合，而不是对任意 lag 搜索：

\[
\mathcal L(P)=
\left\{
\operatorname{clip}(\operatorname{round}(Pq),2,32):
q\in\left\{\tfrac16,\tfrac15,\tfrac14,
\tfrac13,\tfrac12\right\}
\right\}.
\]

为避免多项式递归爆炸，使用 bounded nonlinear basis。示例：

\[
q(u)=\operatorname{clip}
\left(
\frac{u-\operatorname{median}(u_H)}
{\operatorname{IQR}(u_H)},-3,3
\right),
\]

\[
b_2(u)=\frac{q(u)^2}{1+q(u)^2},
\qquad
b_3(u)=\frac{q(u)^3}{1+|q(u)|^3}.
\]

把 $b_2,b_3$ 对 $(1,q)$ 残差化得到 $\phi_2,\phi_3$，以免受控项重新编码 level 或 linear persistence：

\[
g(u)=\theta_2\phi_2(u)+\theta_3\phi_3(u).
\]

递归算子为：

\[
F_\alpha(\mathbf u_t)
=a+\sum_p\varphi_pu_{t-p}
+\alpha g(u_{t-\ell}).
\]

#### History member

在真实 L504 history 上冻结 baseline one-step innovation：

\[
\widehat\varepsilon_t
=u_t-F_1(\mathbf u_t).
\]

从 model-history 起点 $t_s=o-336$ 递归：

\[
u_t^{(\alpha)}=u_t,\quad t<t_s,
\]

\[
u_t^{(\alpha)}
=F_\alpha(\mathbf u_t^{(\alpha)})
+\widehat\varepsilon_t,
\quad t_s\le t<o.
\]

对应的 model-visible history 为：

\[
x_t^{(\alpha)}
=\widehat D_t+s_{336}u_t^{(\alpha)}
=x_t+s_{336}\left(u_t^{(\alpha)}-u_t\right).
\]

由归纳可知 $\alpha=1$ 精确重构真实 history；其他剂量共享同一条真实 history innovation path。

#### Future truth

主定义不从真实 H48 反推 innovation。分别从 baseline 与 treatment 的终态做 zero-future-innovation rollout：

\[
v_{o+h}^{(a)}=F_a(\mathbf v_{o+h}^{(a)}),
\qquad a\in\{1,\alpha\}.
\]

其中 $v^{(1)}$ 从原始 baseline history 终态初始化，$v^{(\alpha)}$ 从对应 treatment history 终态初始化。

history-only structural effect：

\[
\Delta_h^{(\alpha)}
=s_{336}\left(v_{o+h}^{(\alpha)}-v_{o+h}^{(1)}\right).
\]

最终 target：

\[
\boxed{
y_{o+h}^{(\alpha)}
=y_{o+h}^{real}+\Delta_h^{(\alpha)}
}
\]

最后一段 history residual replay 可作为 sensitivity analysis，但不应替代 zero-innovation 主定义。

必要 gate：

- lag 与 nonlinear basis 由 blocked chronological validation 冻结；
- full model 相对 linear null 有稳定 out-of-fold incremental gain；
- linear state 与 nonlinear rollout 都稳定、有限、不依赖 clipping 才不发散；
- alpha 最大剂量没有明显离开历史支持；
- model-visible history 与 H48 effect 均非退化；
- validator 检查 contract replay、共享 innovation 与 exact identity，不检查 exact alpha proportionality。

成熟度：可行但 experimental。若不新增 dynamic contract 与专用 validator，应继续 unavailable。

### 5.6 Predictable recurrent intermittency

兼容 capability id：`predictable_intermittency`。

在 L504 history 上用 robust regression 拟合 smooth nuisance：

\[
r_t=x_t-\widehat D_t.
\]

从同方向的稀疏 residual peaks 拟合最多三相位的重复 interval motif：

\[
c_{k+1}=c_k+d_{k\bmod m},
\qquad m\in\{1,2,3\}.
\]

对历史事件片段对齐后，用 robust median 得到 phase-specific empirical template、width 与 amplitude：

\[
K_j(s)\ge0,
\qquad
K_j(s)=0\ \text{for}\ |s|>1,
\qquad
\max_sK_j(s)=1.
\]

可预测事件 component：

\[
\widehat E_t
=\sum_kA_{k\bmod m}
K_{k\bmod m}
\left(\frac{t-c_k}{w_{k\bmod m}}\right).
\]

反事实：

\[
\boxed{
x_t^{(\alpha)}
=x_t+(\alpha-1)\widehat E_t
}
\]

只缩放 history-clock 可预测的 pulse amplitude；event timing、rate、width、shape，以及未被 clock 解释的随机 burst 都保持不变。

必要 gate：

- L504 至少有足够的 motif repetition 和事件数；
- fixed-L168 中也必须看见多次 event，而不是只有较早 L504 前缀有证据；
- 用较早 history 拟合，在较晚 history block 验证 clock timing 与 waveform；
- timing F1、timing error、holdout clock \(R^2\)、polarity stability、event-to-background ratio 和 duty cycle 全部通过；
- H48 解析延伸中至少出现一个 predicted event window；否则当前 origin unavailable；
- 先去除 smooth carrier，防止把每天固定的宽峰重复计为 seasonality 与 intermittency。

当前 `event_positive_residual_energy_share` 只测 pulse prominence，不测 predictability；它只能做 intensity provenance，不能单独作为真实 contract 准入门。

成熟度：高可行，建议作为下一个单变量能力实现。初版只支持 positive pulse；negative trough 或 bipolar event 另行版本化。

### 5.7 Forecastable common factor

兼容 capability id：`common_factor`。

必须使用原生同步 $D\ge3$ panel。设 baseline L336 的逐通道 location 与 scale 为 $\mu,S=\operatorname{diag}(s_1,\ldots,s_D)$：

\[
Z_t=S^{-1}(X_t-\mu).
\]

在 L504 history 上拟合 rank-one factor：

\[
Z_t=\lambda f_t+e_t,
\qquad
f_t=\frac{\lambda^\top Z_t}{\lambda^\top\lambda}.
\]

再只用 factor history 拟合稳定的 AR / state-space extension，得到 $\widehat f_t$。raw-unit component：

\[
\widehat M_t^{CF}
=S\lambda
\begin{cases}
f_t,&t<o,\\
\widehat f_t,&t\ge o.
\end{cases}
\]

反事实：

\[
\boxed{
X_t^{(\alpha)}
=X_t+(\alpha-1)\widehat M_t^{CF}
}
\]

PCA loading 的整体符号不唯一不影响公式，因为 $\lambda f_t$ 的乘积不变。

必要 gate：

- 原生同步 panel，不能把独立 item 临时拼接；
- $D\ge3$，至少三个非退化 loading；
- top-factor share 显著高于 isotropic floor；
- loading direction 在 chronological folds 中稳定；
- factor continuation 优于 constant / seasonal baseline 且 state 稳定；
- history、fixed-L168 与 H48 factor component 非退化。

归因限制：单纯放大共同分量不能证明模型利用了横截面信息，因为受评通道自身也看见了该 factor。必须配套：

1. full synchronized panel；
2. target-only 或 auxiliary-channel donor replacement；
3. 单独报告 full-panel effect NRMSE 与 ablation degradation，不任意加权成一个总分。

可选的更严格 audit 是 protected-target 设计：保持受评 target history 不变，只改变 auxiliary factor evidence，再检查 protected target 的 forecast effect recovery。

成熟度：条件可行；需要 structural background 和输入消融链路。

### 5.8 Forecastable zero-sum hierarchical contrast

兼容 capability id：`hierarchical_coherence`。

只允许 adapter 声明的 additive hierarchy。设 parent 与 children 满足：

\[
p_t=\mathbf1^\top c_t.
\]

从 L504 history 估计长期 allocation weight：

\[
\mathbf1^\top w=1,
\qquad
q_t=c_t-wp_t,
\qquad
\mathbf1^\top q_t=0.
\]

在固定 zero-sum basis $B$ 上拟合 contrast state：

\[
q_t=Bg_t,
\qquad
\mathbf1^\top B=0,
\]

并由 history 外推 $\widehat q_t=B\widehat g_t$。反事实：

\[
\boxed{
p_t^{(\alpha)}=p_t,
\qquad
c_t^{(\alpha)}
=c_t+(\alpha-1)\widehat q_t
}
\]

因此对所有剂量：

\[
\mathbf1^\top c_t^{(\alpha)}=p_t.
\]

该能力控制的是可预测 child-allocation contrast，不是 coherence residual；coherence 自始至终都应为 0。

必要 gate：

- hierarchy、node ordering 和 aggregation matrix 必须由 adapter 声明；
- parent 必须等于 children sum，或明确记录为由真实 children 构造的 parent；
- contrast continuation 在 chronological holdout 有预测性；
- normalization 采用 coherent centering 与整棵局部 hierarchy 的共享 scale，禁止逐 child z-score；
- 每个 alpha 都通过 machine-precision aggregation identity；
- 分析按 structural group 聚类，多个 sibling pair 或 origin 不能冒充独立 hierarchy。

协议 blocker：计数数据的 additive contrast 在高 alpha 下可能产生负 child。两种选择不能静默混用：

1. 接受标准化实值域干预，并单独报告 raw-domain support violation；
2. 强制 raw-count 非负，改用 compositional / log-ratio 变换。

第二种通常需要真实 future share 才能逐点保持原路径并确保非负，从而违反严格 future-blind delta。该取舍冻结前，建议 hierarchy 只做 qualification，不进入 real-anchored 主 rank。

成熟度：公式可行，raw-support 政策待决。

### 5.9 Directed cross-series predictive transfer

兼容 capability id：`cross_series_dependence`。

正式名称不使用 causal SCM。对同步 panel 选择一个 driver $z_t$ 和至少两个 responder $y_{j,t}$。先控制共同因子、日历与各 responder 自身 persistence，再在 L504 history 上拟合：

\[
z_t=a_z+\sum_q\psi_qz_{t-q}+u_t,
\]

\[
y_{j,t}
=a_j+\sum_p\phi_{jp}y_{j,t-p}
+\sum_{\ell\in\mathcal L}\beta_{j\ell}z_{t-\ell}
+r_{j,t}.
\]

隔离由 driver 产生并经 responder state 传播的 transfer component：

\[
m_{j,t}
=\sum_p\phi_{jp}m_{j,t-p}
+\sum_{\ell\in\mathcal L}\beta_{j\ell}z_{t-\ell}.
\]

history 使用真实已观察 driver；future driver 必须来自 history-fitted stable extension $\widehat z_t$，不能读取真实 H48 driver。定义：

\[
\widehat M_t^{XS}=(0,m_{1,t},\ldots,m_{J,t}).
\]

反事实：

\[
\boxed{
z_t^{(\alpha)}=z_t,
\qquad
y_{j,t}^{(\alpha)}
=y_{j,t}+(\alpha-1)\widehat m_{j,t}
}
\]

在线性系统中，这等价于只改变：

\[
\beta_{j\ell}\longrightarrow\alpha\beta_{j\ell}.
\]

必要 gate：

- 原生同步 $D\ge3$ panel，一个 driver 对至少两个 responders；
- candidate lag 预声明并在 chronological folds 中稳定；
- forward incremental gain 超过 time-reverse、permutation 和 own-history null；
- 去掉 common-factor nuisance 后 directed gain 仍存在；
- 每个 responder 都有增量收益，不能靠平均值掩盖失败通道；
- driver extension 与 transfer state 稳定；
- 评分排除 truth effect 恒为 0 的 driver 通道；
- 配套 full-panel vs responder-only / donor ablation。

观察性 panel 不能排除未观测共同因素、反馈、measurement delay 或共同日历响应，因此结果只能解释为 predictive transfer。若论文坚持 causal edge，该能力的真实轨应 unavailable，causal estimand 只保留在 deterministic synthetic。

成熟度：条件可行；linear predictive 版本可保持 exact alpha proportionality。nonlinear feedback 版本应另建 recursive experimental contract。

### 5.10 Known-future conditional predictive response

兼容 capability id：`covariate_response`。

只接受 adapter 明确标记为 `known_future` 的 covariate $z_t$。联合拟合 target nuisance、own-history state 和 covariate response：

\[
y_t=n_t+\sum_p\phi_py_{t-p}
+\sum_{k=1}^{K}\sum_{\ell=0}^{L}
\beta_{k\ell}b_k(z_{k,t-\ell})+r_t.
\]

其中：

- $n_t$ 吸收 trend、carrier、secondary、AM、step 等 nuisance；
- continuous covariate 的 center/scale 只从 L504 history 冻结；
- binary / event covariate 保留 reference coding，不按全路径重新标准化；
- spline 或 nonlinear basis 一旦选择就写入 contract。

定义 covariate-driven response state：

\[
m_t
=\sum_p\phi_pm_{t-p}
+\sum_{k,\ell}\beta_{k\ell}b_k(z_{k,t-\ell}).
\]

由于 $z_{o:o+48}$ 是合法已知输入，可直接生成 H48 response，而不读取 target future：

\[
\boxed{
y_t^{(\alpha)}
=y_t+(\alpha-1)\widehat m_t,
\qquad
z_t^{(\alpha)}=z_t
}
\]

等价于只改变：

\[
\beta_{k\ell}\longrightarrow\alpha\beta_{k\ell}.
\]

这比直接把 binary promotion 从 1 乘到 1.2–2.0 更合理：dose 控制 target 对 covariate 的响应强度，而不是制造无语义的 covariate 值。

必要 gate：

- covariate provenance 明确、非 target 派生、完整覆盖 L504+H48 并与 target 对齐；
- continuous covariate 有足够 distinct support，binary/event 两种状态都有足够历史支持；
- chronological incremental gain 超过 time-shift / permutation null；
- coefficient sign 与量级跨 folds 稳定；
- 在控制 trend、seasonality 和 calendar 后响应仍存在；
- inference 必须把同一 covariate path 传给 baseline 与 treatment；
- H48 response component 非退化。

promotion 可能内生，天气 forecast 也可能带误差，因此只能称 conditional predictive response，不能称 causal lift。

成熟度：高可行，但依赖结构化 covariate background。

## 6. 真实数据与当前接入边界

当前 `RealSeriesRecord` 已能表达：

- 原生多通道 target；
- known-future covariates；
- hierarchy values / kind；
- structural group id。

但当前 real-anchored background builder 会把记录展开为单通道，并写死：

```text
target_dim = 1
covariates = None
hierarchy = None
```

所以四个结构能力目前并未真正接入 L504 real-anchored 链路。下一版应新增 background kind，而不是给现有 univariate tuple 简单加六个 capability id：

```text
univariate
synchronized_panel
additive_hierarchy
target_with_known_future_covariates
```

结构 background 至少冻结：

- `target[L552, D]` 与 L504/L336/H48 三段 hash；
- channel id、ordering、时间对齐与同步 missingness；
- `covariates[L552, K]`、name、kind、source hash；
- hierarchy summing matrix、parent/child index 与 node ordering；
- structural group id；
- per-channel normalization，或 hierarchy coherent shared normalization；
- H48 component source：`history_fitted_state_extension` 或 `declared_known_future_covariate_response`。

### 6.1 当前本地 GIFT 资产的实际候选

只读检查当前 `data/gift-eval`：

| 数据 | 原生结构 | 可候选能力 |
|---|---|---|
| ETT1 / ETT2 | 各 7 个同步 target channels | common factor、directed cross-series transfer |
| Jena Weather | 21 个同步 target channels | common factor、directed cross-series transfer |
| BizITObs L2C | 7 个同步 target channels | common factor、directed cross-series transfer |
| BizITObs Application / Service | 每 record 2 个 target channels | 不满足建议的 $D\ge3$ common/cross 主定义；可留作 sensitivity |
| Hierarchical Sales | 58 个不重叠 sibling-pair records、4 个 brand structural groups | hierarchy；全部 58 条还具有对齐的 known-future promotion，可候选 covariate response |

注意：

- Hierarchical Sales 的有效 hierarchy replication 至少应按 4 个 brand group 聚类，不能把 58 个 sibling pairs 当作 58 个完全独立体系；
- generic GIFT 的 `past_feat_dynamic_real` 是 past-only，当前 loader 也没有把它暴露成 known-future；不能用它伪造 covariate response；
- Electricity、Solar 等独立 item 不允许按相关性临时拼成“真实 panel”；
- FEV / M5 adapter 中已经声明的 panel、hierarchy 和 known-future 语义可以后续接入，但仍要排除官方 test tail 并通过相同 history-only gate。

## 7. Availability 与验证

### 7.1 共同验证

每个 contract 至少验证：

- source hashes 与 L504 history hash；
- 修改 target H48 后 contract/delta bitwise 不变；
- alpha=1 exact identity；
- shared normalization / MASE；
- baseline 与 treatment 只在声明字段不同；
- background 无放回分配和 effective background count；
- controlled component 的 history、fixed-L168 与 H48 RMS；
- 所有输出有限、无 silent clipping、无 shape/order 变化。

### 7.2 Additive 与 recursive validator 分开

除 nonlinear persistence 外，可加能力应验证：

\[
\frac{X^{(\alpha)}-X}{\alpha-1}
=\widehat M
\]

在数值精度内对所有 alpha 成立。

nonlinear persistence 不满足 exact dose proportionality。它应验证：

- 同一 $F_1,F_\alpha$、basis、lag 和 innovation replay contract；
- alpha=1 对 history 的逐点重构；
- future rollout 完全 target-future-blind；
- dose curve 有限，并保存每个 alpha 的 realized effect，而不是假设线性比例。

### 7.3 结构恒等式

- common factor：loading × factor 乘积与 sign convention 无关；
- hierarchy：每个时点和每个 alpha 都满足 aggregation matrix；
- cross-series：driver truth effect 严格为 0；
- covariate response：baseline/treatment covariate path 完全相同；
- common/cross：full-panel 与 ablation task 使用同一 target truth pair。

## 8. 评分与统计单位

对每个 pair：

\[
\Delta Y=Y^{treat}-Y^{base},
\qquad
\Delta\widehat Y
=\widehat Y^{treat}-\widehat Y^{base}.
\]

主机制量仍可使用：

\[
\operatorname{effect\ NRMSE}
=\frac{\lVert\Delta\widehat Y-\Delta Y\rVert_2}
{\max(\lVert\Delta Y\rVert_2,\epsilon)}.
\]

同时报告 effect correlation、amplitude ratio 和 shared-MASE-scaled effect MAE。

多通道规则：

- 先在一个 background 内对有非零 truth effect 的通道宏平均；
- driver、固定 parent 等零效应通道不进入 effect NRMSE 分母；
- 再对不同 background 等权；
- 同一 source item / structural group 的多个 origin 使用 group-aware bootstrap 或 cluster aggregation；
- common/cross 的 input-ablation degradation 独立报告，不与原始 effect NRMSE 任意加权；
- hierarchy 的 contrast recovery、coherence violation 与 raw-support violation分别报告。

## 9. 推荐实现顺序

1. **共享 decomposition v3**：统一 component ownership，先修 multi/AM 冲突、trend visible gate、regime step-vs-ramp。
2. **Predictable intermittency**：仍是单变量路径，工程改动最小，但要新增 clock/template qualification。
3. **Structural background contract**：保留 panel、covariate、hierarchy 与 group semantics，不再展开成独立单通道。
4. **Covariate response**：先接 Hierarchical Sales promotion，再接 adapter 明确声明的 FEV/M5 known-future covariate。
5. **Common factor**：同步实现 full-panel effect 与 mandatory input ablation。
6. **Directed cross-series transfer**：只先实现稳定 linear predictive version；禁止 causal claim。
7. **Nonlinear persistence**：独立 dynamic contract、rollout 与 validator，先作为 experimental 表。
8. **Hierarchy**：先冻结 raw-count positivity 政策，再决定是否进入正式 real-anchored rank。

## 10. 进入 canonical protocol 前必须冻结的决策

1. 当前 free-sideband TVS 是改成 constrained AM，还是改名为 sideband variability？
2. nonlinear future 主定义使用 zero innovation；history-residual replay 是否只放 sensitivity？
3. hierarchy 是否允许标准化实值域出现 raw negative child？若不允许，是否接受放宽 strict future-blind delta？
4. structural track 的最小 panel dimension 是否固定为 $D\ge3$？
5. common/cross 的 input ablation 是 headline attribution audit，还是 capability availability 的硬组成部分？本提案建议后者。
6. qualification thresholds 应基于独立 train/reference background bank 冻结，不能在最终 evaluation origins 上调参。

在这些决策和离线 qualification 通过前，本文件只定义下一版设计，不改变当前 canonical protocol，也不把尚未实现的六项标记为已支持。
