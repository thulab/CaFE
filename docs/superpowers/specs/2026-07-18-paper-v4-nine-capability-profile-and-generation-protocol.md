# Paper v4：九能力、四档 Lookback 的 Profile 与生成验收协议

日期：2026-07-18

## 1. 固定实验形状

- Prediction length：`H=48`
- Lookback：`L ∈ {96, 168, 336, 504}`
- 九个能力都使用四档 L；层级任务也使用 `H=48`，不再需要退回 `H=28`。

四档不是四次独立生成。每个任务先生成一条 `L=504,H=48` 母样本，再取历史后缀构成
`L=96/168/336/504` 四个视图。四个视图的 48 步 future 指向同一段原始值；按各自
lookback 重新标准化只改变坐标系，不改变预测任务。

## 2. Profile 的角色

Profile 是真实训练窗口的经验校准产物，不是合成样本：

1. source profile：保留数据集、配置、频率、维数、L/H、窗口数和真实特征分位数；
2. pooled task profile：按来源平衡后，用于生成器 nuisance/参数反校准；
3. feature-support artifact：用独立真实 reference/calibration 切分标定 controlled
   features 的联合支持域；
4. near-distance artifact：用真实窗口间距离标定 DCR/NNDR 防复制阈值。

真实窗口先预留官方测试尾部（GIFT）或内部验证尾部，再额外保留 48 步验证 embargo；
profile 与 gate 只接触其之前的窗口。

## 3. 数据集与能力映射

| 任务 | 能力 | 数据集 |
|---|---|---|
| 单变量 | trend、multi-seasonal、time-varying seasonality、regime switching、nonlinear persistence、predictable intermittency | 原 v4 小时级 13 配置（M4、Electricity、Solar、ETT1/2、Jena、KDD、Loop Seattle、SZ-Taxi、M_DENSE、Bitbrains Fast/RND、BizITObs L2C） |
| 多变量共同因子 | common factor | Electricity Hourly、Traffic Hourly、Jena Weather/H、BizITObs L2C/H |
| 严格层级 | hierarchical coherence | M5 Daily，父序列等于两个部门子序列逐点求和 |
| 外生协变量 | covariate response | GEFCom2014 Load、GEFCom2014 Solar |

单变量 13 配置全部保留为 source profile；其中数据稀疏或有效长窗不足的配置不强行进入
pooled gate 校准。进入 pooled profile 的来源等权，避免长数据集主导阈值。

## 4. 数据切分与防泄漏

每个 task×L 的真实窗口做三路切分：

- generator parameter：只用于 profile nuisance 和生成器反校准；
- gate reference：拟合 controlled-feature 联合支持和近邻参考库；
- gate calibration：只用于 conformal 阈值。

多序列按 group 隔离；单序列按时间切分，并至少 embargo `L+H`。多数据集任务必须先在
每个数据集内部独立完成三路切分，再将 parameter/reference/calibration 三个 partition
分别按来源等权汇总；禁止用 Load-only reference 对 Solar-only calibration 之类的跨
数据集距离标定 novelty 阈值。同一真实母窗口的四档视图只在同一档内部参与校准，不能
跨档充当彼此的独立样本。

## 5. Controlled features 的机制

合成不只依赖事后 gate：

1. 在 `L=504,H=48` 的 pooled real profile 上提取 nuisance 分布；
2. 对每个能力的五档 canonical target 反求 `structure_scale` 与
   `intensity_lambda[1..5]`；
3. 在强度拟合前按真实 control profile 选择 nuisance 机制；例如 GEFCom 长窗口
   covariate-residual outlier P75 不超过 0.01 时使用 Gaussian residual；
4. 用独立随机种子验证主特征的五档单调性与误差；
5. 生成后，四档视图分别进入 controlled-feature 联合支持 gate；
6. 四档分别进入真实标定的 near-distance gate。

无独立 controlled feature 的 regime switching 和 predictable intermittency 不伪造
控制量：它们依赖 construction predictability contract、主特征强度标定与
near-distance gate。

## 6. 最终合格判定

一个母样本只有同时满足以下条件才合格：

- 生成器的 capability-specific predictability construction 验证通过；
- `L=96/168/336/504` 四个视图的 feature-support gate 全部通过；
- 四个视图的 near-distance gate 全部通过；
- 四个视图共享相同的原始 future。

对九能力 × 五强度档使用独立种子，每个 cell 至少验收 8 个母样本。任何 cell 失败都不得
宣称九能力 profile 完成。

## 7. 复现命令

```bash
cd backend
uv run python ../scripts/build_paper_v4_nine_capability_suite.py
uv run pytest tests/unit/test_paper_v4_profile_suite_script.py \
  tests/unit/test_paper_v4_nine_capability_suite_script.py
```

输出目录：

```text
runtime/paper_exp/v4/01_nine_capability_suite/
```
