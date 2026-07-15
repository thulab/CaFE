# TSBenchmark 设计文档

![系统架构图](<TSBenchmark 设计文档.assets/TSBenchmark 设计文档-1_1.jpg>)

- 文档内容
  - 整体介绍：1段话 250-300 TSBenchmark

  - 功能介绍：一般用户、管理员分别一段话，能干什么
  - 应用场景：画饼，可给他们选型、各种方面提供一个科学支撑，模型不停留在人评估的范围上
  - 待确认问题：底下（有多少数据集、有哪些应用场景、未来的部署硬件形式）

## 确认

1. 数据集数量和情况
2. 数据来源
3. 硬件情况（模型部署）
4. 数据集的场景
5. 应用，落地形态

- 1个数据集
- 单变量
- 5个模型（Timer 3.5、Timer 3.0、Chronous 2，toto，timesfm2.5，5个）
- 指标：MASE、MSE、MAE（默认主榜指标为 MASE）
- 可视化页面（查看榜单，查看报告，进行测试）

## 目录结构

- 整体介绍
  - 工具是什么？
  - 解决什么问题？
  - 有什么创新点？
- 系统架构设计
  - 这里放我们绘制的图
- 核心功能（描述清楚有哪些功能，想明白每个功能是有哪些步骤，哪些分支
- ...

## 1. 核心功能

### 1.1 一般用户

- 查看榜单

  - 各个赛道榜单：
    - 可以选择一列做排序

| 模型 | 指标1 | 指标2 | 指标3 | 指标4 |
| --- | --- | --- | --- | --- |
| 模型1 |  |  |  |  |
| 模型2 |  |  |  |  |
| 模型3 |  |  |  |  |
| 模型4 |  |  |  |  |

- 查看报告：
  - 查看赛道所有模型的对比报告
  - 查看自选模型间的对比报告（低优先级）
- 申请上传模型：
  - 表单内容：（TODO）
- 申请上传数据集
  - 表单内容（TODO）

### 1.2 管理员

#### 1.2.1 数据集相关

- 加载本地数据集
- 查看测试数据集：
- 删除测试数据集（低优先级）
  - 协同删除哪些

#### 1.2.2 模型相关

- 查看可运行的模型有哪些
  - 能测的能力维度
  - 结果
- 加载模型
- 删除模型

#### 1.2.3 赛道相关

- 创建赛道
  - 赛道名字、能力维度（多选）、参数
  - 此处生成测试数据集
  - 能力维度中包含一个维度叫做真实数据
- 查看赛道
- 删除赛道（低优先级）

#### 1.2.4 评测相关

- 创建评测
  - 赛道、模型
- 查看评测（得有进度）
- 终止、删除评测

#### 1.2.5 报告相关

- 生成、查看报告
  - 赛道的报告
  - 评测的报告

## 2. 实体讨论

- model
  - model_id
  - name
  - method

- track
  - track_id
  - name
  - capability_ids
- capability block
  - capability_id
  - type（真实数据（也可以多维度）的参数集是对应真实数据的shard）
  - track_id

  - params -> shard_id
  - metrics
- shard
  - shard_id
  - capability_id
  - url: 生成数据集位置/真实数据集的位置

- Benchmarking(Eval)
  - benchmarking_id
  - track_id
  - unit_ids
- Unit
  - unit_id
  - benchmarking_id
  - model_id
  - task_ids
- Task
  - task_id
  - unit_id
  - capability_id
  - predicate_result 是否全存（全存/抽样存/只存统计需要再重跑）
  - metrics

- Report
  - report_id
  - unit_ids

- Rank
  - ranklist
  - report_id

## 3. 项目定位

TSBenchmark 是一项针对时间序列预测模型的动态评测平台。该平台不以单一总分作为唯一评价标准，而是侧重于提供数据生成、赛道组合与模型评测的自主可控性，支持用户从赛道、能力维度、样本及指标等多个层级立体化地理解和评估模型效能。项目关注的问题是：随着时间序列基础模型越来越多、预训练数据越来越广，测试数据集静态 benchmark 的可信度会下降。模型在固定的公开数据集上取得高分，不一定说明它具有真实泛化能力，也可能是因为预训练阶段见过相似数据。TSBenchmark 因此把评测数据设计成可动态生成、可复现、可追踪、可解释的对象，而不是一次性固定题库。TSBenchmark 的核心原则是：

- 动态生成：评测样本由不同的能力维度和参数配置依据统计机制动态生成。
- 能力拆解：一条赛道由多个能力维度组成，例如趋势、季节性、多变量交互、lead-lag 依赖、regime shift、协变量响应等。
- 可回溯评测：任何排行榜分数都能回溯到模型、赛道、能力测试块、task、sample、预测曲线和具体指标。

## 4. 平台要解决什么问题

从用户视角看，TSBenchmark 希望支持以下流程：

1. 创建或选择一条评测赛道。
2. 创建赛道时选择要测试的能力维度。
3. 按每个能力维度中默认或自定义参数生成测试数据。
4. 选择一个或多个模型参与评测。
5. 启动一次 benchmarking run。
6. 查看赛道榜单。
7. 下钻查看某个模型、某个能力测试块、某个样本或某个指标。
8. 对比某个 sample 上不同模型的预测曲线与真实未来值。

当前仓库已经实现了这个流程的 Web MVP 与 API 闭环：

- CSV / 单设备表模型 TsFile 输入；
- 单目标列摄入，目标列在 load job 阶段单选；
- SQLite `SeriesPoint` 逐点行存储作为样本真值源，`SampleIndex` 保存切片指针；
- 真实数据 shard、能力测试块与 track 生成；
- 通过 timer-rest-service 或本地桩执行模型评测；
- MASE / MSE / MAE 指标聚合、报告生成与榜单刷新；
- sample 级预测结果可视化。

后续平台工作的重点，是继续扩展多序列/面板数据、协变量/多目标、合成数据与 real-anchor 相关能力。

## 5. 核心实体

### 术语说明

| 设计术语 | 中文表达 | 含义 |
| --- | --- | --- |
| Capability | 能力维度 | 要考察的预测能力类型，如 lead-lag、common factor、regime shift。 |
| Capability Block | 能力测试块 | 某个能力维度下，带一组参数覆盖和样本集合的数据块。 |
| Track | 赛道 | 多个能力测试块组合成的一条评测赛道。 |
| Task | 任务 | 某模型在某次评测中针对某个能力测试块的结果集合。 |

## 6. 本地启停

当前 MVP 可以通过脚本统一启动、停止和查看前后端状态：

前置要求：已安装 `uv`、`npm`，且 Node.js 版本为 `20.19+` 或 `22.12+`。

```bash
./scripts/start-system.sh
./scripts/status-system.sh
./scripts/stop-system.sh
```

默认端口：

- 后端：`http://127.0.0.1:8000`
- 前端：`http://127.0.0.1:5173`

脚本会把 PID 和日志写到 `.tsbenchmark-system/`。可用环境变量覆盖默认行为：

```bash
TSBENCHMARK_BACKEND_PORT=8010 TSBENCHMARK_FRONTEND_PORT=5174 ./scripts/start-system.sh
TSBENCHMARK_SYSTEM_DIR=/tmp/tsbenchmark-system ./scripts/status-system.sh
```

## 6.x Docker 部署

前置：本机安装 Docker（含 `docker compose`）。仓库根目录已提供
`docker-compose.yml`、`backend/Dockerfile`、`frontend/Dockerfile`、`frontend/nginx.conf`
和 `.env.example`。

默认 Docker 编排假设推理服务 `timer-rest-service` 已在另一个容器或主机地址运行。
先复制环境变量模板并填写 JWT 密钥、管理员密码和推理服务地址：

```bash
cp .env.example .env
docker compose up -d --build
docker compose down             # 停止；想清掉 sqlite/上传文件加 -v
```

启动后访问：

- 前端（nginx 提供静态产物 + 反代 `/api/*`）：`http://localhost:5173`
- 后端 OpenAPI：`http://localhost:8000/docs`

默认两个服务：

- `backend`：FastAPI 后端；运行时数据写在命名卷 `tsbenchmark-runtime`（容器内 `/var/lib/tsbenchmark`），SQLite、上传文件、预测和报告都在里面。
- `frontend`：Vite 构建的静态资源，由 nginx 服务。

常见推理服务地址：

- 同 Docker 网络：`TSBENCHMARK_TIMER_SERVICE_BASE_URL=http://timer-service:10810`
- 推理服务发布在宿主机端口：`TSBENCHMARK_TIMER_SERVICE_BASE_URL=http://host.docker.internal:10810`

本地 smoke test 可以用仓库内可选桩服务：

```bash
TSBENCHMARK_TIMER_SERVICE_BASE_URL=http://stub:10810 \
docker compose --profile stub up -d --build
```

完整部署说明和环境变量表见
[`docs/developer/deployment.md`](docs/developer/deployment.md)。

发布镜像（按需）：

```bash
# 单平台
docker compose build
docker tag tsbenchmark-backend:latest  <registry>/tsbenchmark-backend:<tag>
docker tag tsbenchmark-frontend:latest <registry>/tsbenchmark-frontend:<tag>
docker push <registry>/tsbenchmark-backend:<tag>
docker push <registry>/tsbenchmark-frontend:<tag>

# 多平台（buildx）
docker buildx build --platform linux/amd64,linux/arm64 \
  -f backend/Dockerfile  -t <registry>/tsbenchmark-backend:<tag>  --push .
docker buildx build --platform linux/amd64,linux/arm64 \
  -f frontend/Dockerfile -t <registry>/tsbenchmark-frontend:<tag> --push .
```

### Shard

Shard 是可复用数据切片，表示一组参数完全固定的数据单元。它用于缓存、索引、复现和追踪；前端数据集页会把 shard 作为可选择资产展示，赛道可以复用同一 shard。一个 shard 包含多条 sample，这些 sample 共享同一组固定生成参数，例如：

- 能力维度 (capability)；
- difficulty（生成机制下预期的预测难度）；
- horizon ratio（预测长度比例）；
- season length（数据主周期）；
- target dimension（预测变量维度）；
- sample 数量（每组参数生成多少条时间序列样本）；
- seed（随机种子）；
- real anchor profile（真实数据分布特征来源），如果启用真实锚定。

### Sample

Sample 是模型实际看到的一道预测题，是最小模型输入单元。单变量、多变量和协变量任务统一使用以下 row-oriented schema：

```text
target_history: [context_length, target_dim]
target_future: [horizon, target_dim]
history_cov: [context_length, history_cov_dim]
future_cov: [horizon, future_cov_dim]
```

单变量任务就是 target_dim = 1 的特殊情况。多变量和协变量任务不需要另一套数据接口，因此 runner 和推理服务可以使用统一协议。每个 sample 还会记录：

- 生成配置（shard 的参数信息，用来复现 sample 的生成条件）；
- latent 参数（生成这条 sample 时采用的数学机制中的隐藏参数）；
- realized features（样本生成后实际测出来的统计特征）；
- real anchor 信息（样本参考了哪个真实数据结构来源）；
- baseline score（简单基线模型在这条 sample 上的分数，用来判断难度）；
- 所属 shard、能力测试块、track 与 benchmark version（说明它属于哪组数据、哪条赛道和哪个 benchmark 版本）。

### Capability Block

Capability Block，中文称为能力测试块，是用户可操作的数据块，表示某个能力维度下的一组测试数据。例子包括：

- common_factor：共享因子能力；
- hierarchical_coherence：层级一致性能力；
- covariate_response：已知未来协变量响应；
- 6 个单变量维度：trend、multi-seasonal、time-varying seasonality、regime switching、nonlinear persistence 和 predictable intermittency。

> 💡 直观理解：每个 Capability Block 都是一个“出题方向”，用于控制一种预测难点，让评测不只回答哪个模型平均误差低，还能回答模型到底擅长哪类时间序列结构。

| 能力测试块 | 直观场景 | 主要考察模型什么能力 |
| --- | --- | --- |
| common_factor（共享因子能力） | 多条序列表面上是不同通道，但背后受同一个或少数几个隐藏因素共同驱动。例如一组传感器会因为同一飞行阶段、同一负载变化或同一环境条件而一起升高、降低或呈现相似周期。 | 模型是否能利用跨通道的共同变化，而不是把每个通道当成互不相关的单变量序列。表现好的模型应能识别“大家一起动”的低维结构。 |
| hierarchical_coherence（层级一致性） | 多条序列组成父子层级，父节点在每个时刻都严格等于子节点之和。子节点的动态差异随 intensity 增强。 | 模型能否同时保持预测精度和加总一致性，而不是逐通道预测后破坏结构约束。 |
| covariate_response（已知未来协变量响应） | 预测目标不仅由自身历史决定，还会受到未来已知外部变量影响。例如未来计划量、事件标记、环境变量或控制输入已经提前知道，目标序列会随这些变量变化。 | 模型是否真正使用 `future_cov`。如果模型只根据历史 target 外推，而忽略未来协变量，它在这类任务上应明显落后于能利用协变量的模型。 |
| trend（趋势能力） | 单条序列存在上升、下降、衰减或趋势变点，趋势可能在中途改变方向或强度。 | 模型是否能区分短期波动和长期趋势，并在预测区间内合理外推趋势，而不是简单复制最近值。 |
| multi_seasonal（多周期季节性能力） | 单条序列同时包含多个周期，例如短周期波动叠加长周期变化，并且周期的幅度或相位可能缓慢漂移。文档里写的 seasonal 主要对应这一类。 | 模型是否能同时捕捉多个时间尺度的周期结构，并处理周期强度变化，而不是只记住一个固定周期模板。 |
| regime_switching（可预测状态切换） | 状态按固定驻留时长交替，历史中至少出现两次切换，预测区间内至少出现一次切换。 | 模型能否从历史识别切换时钟并预测下一状态，而不是应对无先兆随机冲击。 |
| nonlinear_persistence（非线性多滞后持久性） | 当前值同时依赖短滞后、季节滞后和非线性中程滞后，递推系数满足稳定性约束。 | 模型是否能利用多时间尺度依赖和非线性反馈；该维度不再声称严格的 fractional long memory。 |
| predictable_intermittency（可预测间歇性） | 稀疏脉冲按历史可识别的固定事件时钟重复出现，intensity 控制脉冲显著性。 | 模型能否识别稀疏事件规律并命中预测区间内的脉冲，而不是猜测随机 burst。 |

能力测试块本身不直接保存所有样本，而是索引若干 shard。默认情况下，一个能力测试块会展开一组参数网格，例如多个 difficulty、horizon ratio、season length、target dimension 和 sample 数量。用户也可以创建自定义能力测试块，只覆盖某个 difficulty 或某个输出长度。这使得用户操作层级足够直观，同时保留底层数据片的可复现性。

### Track

Track 是一条评测赛道，由多个能力测试块组成。它回答的问题是：这条评测想考察哪些能力？例子：

- multivariate_core ：common factor + lead-lag coupling + coherent regime shift；
- covariate_aware ：known-future covariate response；
- univariate_core ：多个单变量诊断能力维度；
- 用户自定义 track：从已有能力测试块中挑选组合。

评测和报告首先以 track 为粒度展开，然后允许用户下钻到能力测试块、task 和 sample。

### Benchmarking Run

Benchmarking run 表示一次评测执行。它记录：

- 选择了哪条 track；
- 选择了哪些模型；
- 使用哪个 benchmark version；
- 使用哪个数据版本；
- 运行时间；
- 每个模型的结果；
- 指标结果；
- 报告产物。

当前 CLI 里，一次 benchmarking run 由 run-eval 对某个 benchmark parquet 执行模型评测并写出 eval parquet 表示。最终平台里，它应该是一个有 run ID、状态和日志的后端任务。

### Unit

Unit 表示某个模型在一次 benchmarking run 中的完整评测结果。如果一次 benchmarking run 选择 5 个模型，那么它会产生 5 个 unit。每个 unit 聚合这个模型在当前 track 下所有能力测试块、task、sample 和 metric 的结果。Unit 是模型级下钻的自然入口：

- 模型元信息；
- track 总分；
- 各能力测试块表现；
- runtime / cost 信息；
- 失败样本；
- sample 级预测产物。

### Task

Task 表示一次 benchmarking run 中，某个模型针对某个能力测试块的评测结果。也就是：

```text
benchmarking run + model + capability block
```

例如，timer_rest:Chronos-2 在 common_factor 能力测试块上的一次评测，就是一个 task。它聚合该能力测试块下所有 sample 的预测和指标。Task 这一层用于回答能力维度问题：

- 哪个模型最擅长利用共享因子？
- 某个模型是否容易在层级一致性上失败？
- 某个模型是否只在短 horizon 上表现好？
- difficulty 增加后模型退化速度如何？

### Metric

Metric 定义 task 和 unit 如何被打分。当前指标包括 MASE、sMAPE、relative skill、runtime 等。未来可以扩展：

- 概率预测指标；
- calibration 指标；
- 吞吐、延迟、显存、GPU 时间等成本指标；
- real support distance 等合成数据有效性指标；

- 真实 probe track 与合成 anchor track 的排序一致性指标。

能力测试块可以声明哪些 metric 适用于该能力维度；task 则保存某个模型在该能力维度上的指标结果。

### Model

Model 是参与评测的预测后端。当前代码支持内置 baseline、本地 Python adapter 和 Timer REST Service adapter。最终平台中的 model 实体应包含：

- model id；
- 展示名称；
- 支持的 task type；
- 支持的输入格式；
- 是否原生支持多变量和协变量；
- 服务 endpoint 或执行环境；
- 模型版本；
- 硬件约束；
- 当前加载状态。

对于多变量 track，当前规则是保守的：只把原生支持多变量输入的模型放入多变量评测。

### Ranking List

Ranking list 是某条 track 的榜单。它聚合该 track 下每个模型的最新有效结果，展示所有相关模型在这条赛道上的成绩。它应支持：

- 总榜；
- 按 metric 排名；
- 按能力测试块排名；
- 每个模型的最新成绩；
- 必要时展示最佳成绩；
- 链接到所有历史 benchmarking run。

Ranking list 和 report 是两个概念。Ranking list 是持久化榜单对象；report 是对某次或若干次评测结果的解释性产物。

### Report

Report 是评测结果说明。

它消费 evaluation output，更新或支撑 ranking list，并提供更详细的分析视图。Report 应包含：

- 总体模型表；
- 能力维度级分数；
- difficulty 曲线；
- horizon 曲线；
- 雷达图或其他能力画像；
- sample 级预测曲线；
- 合成数据有效性诊断；
- 真实数据与合成数据对比；
- 后续可加入自动文字分析。

当前代码已经有聚合报告、静态图、真实/合成对比页面和 sample forecast viewer。平台版本需要把这些能力做成可交互页面。

## 6. 数据生成层级

TSBenchmark 的数据层级是：

```text
sample -> shard -> capability_block_shard -> capability_block -> track -> benchmark
```

用户主要操作能力测试块和 track：

- 选择能力维度；
- 选择默认或自定义参数覆盖；
- 生成能力测试块；
- 组合 track；
- 对 track 运行模型评测。

Shard 是可复用切片层级，用于保证固定参数数据单元可复现、可缓存、可索引；同一 shard 可以被多条赛道通过不同 capability block 复用。

## 7. 统一预测数据格式

TSBenchmark 使用统一格式表达单变量、多变量和协变量预测：

```text
target_history
target_future
history_cov
future_cov
```

这与当前本机推理服务的 target / history_cov / future_cov 形式一致。当前设计暂不引入 static covariate，优先覆盖现有模型服务可以直接消费的数据结构。该统一格式的好处是：

- 单变量模型、多变量模型、协变量模型可以共享 benchmark 表结构；
- runner 不需要为每种任务维护完全不同的数据协议；
- web 页面可以用同一套组件展示 history、future、covariate 和 forecast；
- 后续接入服务模型时，数据转换逻辑更稳定。

## 8. 真实数据如何被利用

TSBenchmark 使用真实数据的目的，不是把真实数据直接拿来当固定题库，也不是从真实数据中复制片段作为最终测试样本。真实数据在当前设计中承担的是“结构参照系”的角色：它帮助系统理解真实时间序列通常有哪些统计形态，再用这些统计形态约束和校验动态生成的数据。这背后的基本判断是：完全脱离真实数据的纯随机合成序列很容易失去现实意义；但直接使用公开真实数据又容易受到数据泄露和静态 benchmark 失效的影响。因此 TSBenchmark 采用“结构/分布对齐，而不是数据点对齐”的策略。整体流程如下：

```text
真实数据集
-> 数据集 manifest
-> 清洗并抽取目标列矩阵
-> 提取结构特征
-> 形成真实 anchor profile
-> 指导合成数据生成
-> 生成后再次提取特征并做有效性诊断
```

### 8.1 真实数据先被转化为结构特征

一个真实数据集进入系统时，会先通过 manifest 描述它的基本信息，例如：

- 数据集 ID；
- 数据领域；
- 时间频率；
- 文件路径；
- 时间列；

- target columns。

系统读取这些 target columns 后，会把它们整理为一个 [time, target_dim] 的目标矩阵。随后系统不会保存或复用某一段真实未来值作为测试答案，而是从这个矩阵中提取结构特征。当前多变量真实特征大致分为四组：

- 单通道时间结构：趋势强度、季节性强度、频谱熵、自相关半衰期、变点密度、方差漂移、间歇性、异常值比例。
- 跨通道同期关系：平均相关性、最大相关性、相关稀疏度、有效秩、第一主成分占比、通道尺度差异。
- 跨通道时滞关系：不同 lag 下通道之间的 lead-lag 强度，以及最大 lead-lag 强度对应的 lag。
- 多通道共同事件：多个通道是否会在相近时间同步发生均值或状态变化。

对于单变量数据，系统也会提取类似的单通道结构特征，例如趋势、季节性、谱熵、长记忆、变点、间歇性、异常值等。

### 8.2 Anchor Profile 保存什么

真实数据集被特征化后，会汇总成 anchor profile。Anchor profile 可以理解为真实数据结构分布的摘要。它主要保存：

- 每个结构特征的均值和标准差；
- 特征之间的协方差；
- 每个特征的 p05/p50/p95 支撑区间；
- 若干真实结构原型，即 prototype 或 medoid；
- 每个 prototype 所在簇的权重；
- 原始真实数据集的 manifest。

这些信息回答的是：“真实数据大致长什么样？”而不是“真实数据某个时刻的值是多少？”

### 8.3 Anchor Profile 如何指导生成

生成一个合成样本时，系统可以从 anchor profile 中采样一个真实结构 prototype。不同能力维度会读取其中不同的目标特征，并把这些目标特征映射到生成器的 latent 参数。例如：

- 共享因子能力会关注有效秩、第一主成分占比、通道相关强度，用于决定多通道是否由少数共同因子驱动。
- Lead-lag 能力会关注 lead-lag 强度、最大 lag、相关稀疏度，用于决定通道之间是否存在延迟影响。

- 同步状态切换能力会关注 coherent shift rate 和通道相关强度，用于决定有多少通道会共同经历状态变化。
- 协变量响应能力会参考真实数据中的相关结构强度，用于决定 target 对 future covariate 的响应强弱。

这里的“指导”不是硬约束。生成器仍然会产生新的序列、新的噪声、新的相位、新的局部形态。真实 anchor 只提供结构目标和合理范围。

### 8.4 Anchor 固定了什么，不固定什么

Real anchor 固定或约束的是：

- 目标维度的大致规模；
- 趋势、季节、噪声、长记忆等边际结构的合理范围；
- 多变量通道之间相关性、有效秩、尺度差异等结构；
- lead-lag 和同步状态变化等跨通道行为；
- 合成样本是否落在真实结构支撑域附近。

Real anchor 不固定：

- 原始真实数据点；
- 某个真实时间窗口；
- 真实时间戳；
- 真实未来值；
- 某个真实数据集中的具体曲线形状。

因此，TSBenchmark 希望得到的是“像真实数据，但不是那份真实数据”的评测样本。

### 8.5 生成后如何检查有效性

系统在生成前先按任务、频率、context、horizon 和 target dimension 选定精确 `anchor_profile_id`，再读取该 profile 的 generator-conditioning 参数生成。默认批次对兼容 profile 做确定性的均衡分层；研究脚本也可以按 capability 固定 profile。

生成完成后，系统重新提取 realized features，并执行三类硬检查：construction-level predictability、预选 profile 内的联合 control-feature support，以及相对所有兼容真实 profile 的 DCR/NNDR 近距离风险。novelty 校准按完整 series/panel group 切分；单序列使用带 `C+H` 非重叠区的时间块，且 full target window 与模型可见的 target context 都必须通过近复制检查。目标 capability feature 不作为逐样本拒绝条件，而在 `profile × capability × intensity` 批量上做 dose-response 验收。缺少精确校准组合或 artifact 特征口径不兼容时 fail closed。当前 raw DCR 的证据边界是已提交 `R_train` reference 内的目标轨迹 novelty；known-future covariates 只通过特征距离间接参与，不应表述为对未知预训练语料的全覆盖保证。

后续更完整的有效性验证还会引入 real probe track：让模型分别在真实 probe 数据和合成 anchor 数据上评测，观察模型排序是否保持大致一致。如果排序一致性较高，说明合成评测更可能反映真实预测能力。

## 9. 数据如何按能力维度生成

TSBenchmark 的数据生成不是一次性生成一个大混合数据集，而是按能力维度生成能力测试块。每个能力测试块覆盖一组参数组合，例如不同 difficulty、horizon ratio、season length、target dimension 和 sample 数量。每个固定参数组合对应一个 shard，shard 中包含多条 sample。通用生成流程如下：

```text
选择能力维度
-> 确定参数组合
-> 选择随机种子和可选 anchor
-> 生成完整时间序列矩阵
-> 切分 history / future / covariates
-> 标准化 target
-> 记录 latent_params 和 realized_features
-> 写入 sample / shard / capability block / track
```

其中 `intensity` 是目标结构的有序干预级别，不等同于预设难度，也不通过同时增加噪声来制造“更难”。噪声、季节残差等 nuisance 由预选真实 profile 决定，并在配对的 intensity 扫描中固定；主周期来自 profile bucket，旧请求中的 `season_length` 仅保留为兼容字段。

### 9.1 单变量数据如何生成

单变量任务是 target_dim = 1 的预测任务。它使用与多变量任务相同的 sample schema，只是 target 矩阵只有一列。当前单变量能力维度包括：

- trend ：考察模型对趋势、趋势衰减和趋势变点的处理能力。
- multi_seasonal ：考察模型对多周期、相位漂移和振幅变化的处理能力。
- regime_switching ：考察模型对状态切换和分布突变的处理能力。
- nonlinear_persistence：考察模型对稳定的多滞后和非线性反馈结构的处理能力。
- predictable_intermittency：考察模型能否从历史事件时钟预测稀疏脉冲。

单变量生成会参考单变量 anchor statistics。系统先从真实或 bootstrap 序列中提取单通道特征，形成 anchor prototype。生成某条样本时，会采样一个 prototype 作为结构参照，然后由具体能力维度生成一条完整序列。

例如：

- trend 能力会生成带趋势项、变点和季节扰动的序列；
- multi-seasonal 能力会叠加多个周期，并加入相位或振幅漂移；
- regime-switching 能力会让序列在不同状态之间随机切换；
- long-memory-nonlinear 能力会引入较长延迟反馈；
- intermittent-heteroskedastic 能力会生成大量低值、间歇性爆发和变化噪声。

生成后，系统会切出 context 和 horizon，形成 target_history 和 target_future ，并记录真实生成参数和测得的 realized features。

### 9.2 多变量数据如何生成

多变量任务是 target_dim >= 2 的预测任务。生成器先产生完整矩阵：

```text
[context_length + horizon, target_dim]
```

然后切分为：

```text
target_history: [context_length, target_dim]
target_future: [horizon, target_dim]
```

当前结构化能力维度包括：

- common_factor ：多个 target channel 由少数共享潜在因子共同驱动。它用于测试模型能否识别跨通道共同变化，而不是只把每个通道当成独立单变量处理。
- hierarchical_coherence：父节点严格等于子节点之和，用于同时测试预测精度与输出加总一致性。
- covariate_response：单目标未来依赖已知未来外生变量，用于测试模型是否真正利用 future covariates。

在启用 real anchor 时，结构化生成器会读取真实 profile 中的相关性、有效秩、层级残差和协变量响应等结构目标。生成器分别调整共享因子强度、子节点异质性或协变量效应强度。多目标生成结束后，系统会按 context 部分进行标准化；层级数据使用共同尺度，保证标准化后仍严格满足父子加总关系。

### 9.3 协变量数据如何生成

协变量任务仍然使用统一 schema，但额外包含：

```text
history_cov
future_cov
```

当前协变量能力维度是：

- covariate_response ：target 会受到已知未来协变量影响，例如天气型连续变量、事件型二值变量等。

该能力维度的重点不是只预测 target 本身，而是测试模型是否能正确利用 future covariate。生成时，系统会先生成一组 covariate，再让多个 target channel 对这些 covariate 产生不同强度、不同方向、可能带非线性的响应。在模型输入中：

- history_cov 给出历史阶段的协变量；
- future_cov 给出预测区间内已知的协变量；
- target_future 仍然是模型需要预测的答案。

这类任务可以用于区分两类模型：一类只是根据 target history 外推，另一类能够真正利用 future covariate 改善预测。

### 9.4 每条样本会保存什么

无论是单变量、多变量还是协变量样本，TSBenchmark 都会尽量保存两类信息。第一类是模型评测必需的信息：

- target_history
- target_future
- history_cov
- future_cov
- target_columns
- history_cov_columns
- future_cov_columns
- context_length
- horizon
- target_dim

第二类是分析和回溯需要的信息：

- latent_params：生成器使用的隐含参数，例如趋势斜率、周期、滞后强度、切换概率等。

- realized_features ：从最终样本中测得的统计结构。
- real_anchor_* ：使用了哪个真实 anchor profile、cluster 或 dataset。
- real_support_* ：合成样本相对真实支撑域的距离和越界情况。
- baseline_mase ：内置基线在该样本上的表现。

这使得 TSBenchmark 不只是生成一批预测题，还能解释每道题在考什么、实际生成出来是否符合预期、模型为什么在某些题上表现好或差。

## 10. 评测与报告流程

平台目标流程是：

```text
track
-> materialized benchmark
-> benchmarking run
-> unit per model
-> task per model-capability-block pair
-> sample predictions
-> metrics
-> report
-> ranking list
```

当前 CLI 对应关系是：

1. build-family-block ：生成能力测试块和 shard。当前 CLI 名称仍沿用早期的 family-block 命名。
2. compose-track ：组合 track 并物化 benchmark parquet。
3. run-eval ：评测一个模型并写出 eval parquet。
4. make-report ：聚合模型指标。
5. make-real-synth-viz ：对比真实数据和合成数据。
6. make-sample-forecast-viz ：查看 sample 级预测曲线。

后端平台应保留这些数据契约，但把它们封装成持久化实体和 API。

## 11. 用户界面目标

### Track Builder

用于创建赛道。

页面能力：

- 选择 task type；
- 选择能力测试块；
- 设置 difficulty、horizon、season length、target dimension；
- 设置 sample 数量；
- 选择 real anchor mode；
- 预览生成数据的统计特征。

### Benchmarking Run Page

用于选择 track 和模型并启动评测。页面能力：

- 展示选定 track；
- 展示参与模型；
- 展示评测任务状态；
- 按 model 和能力测试块展示进度；
- 展示失败、跳过和错误原因。

### Track Ranking Page

用于查看某条赛道榜单。页面能力：

- 展示模型总排名；
- 按 metric 切换榜单；
- 按能力测试块查看模型成绩；
- 按 difficulty 或 horizon 查看分布；
- 跳转到历史 benchmarking run 和 report。

### Report Page

用于解释某次评测结果。页面能力：

- 总体得分表；
- 能力维度级能力画像；
- difficulty 曲线；

- horizon 曲线；
- 真实/合成数据有效性可视化；
- sample 级预测结果查看。

### Sample Forecast Page

用于查看某个样本上模型的具体预测情况。页面筛选维度：

- 能力维度；
- 参数组合；
- sample id；
- target channel；
- model。

页面展示：

- history；
- ground truth；
- forecast；
- sample-level metric。

这部分对解释模型为什么赢或为什么输很重要。

## 12. 存储设计

当前实现使用文件系统产物：

```text
artifacts/family_blocks/
artifacts/tracks/
artifacts/eval/
artifacts/reports/
artifacts/viz/
```

这足够支撑本地实验和 CLI 闭环。最终平台需要持久化管理：

- dataset；
- shard；
- 能力测试块 manifest；

- track manifest；
- benchmarking run；
- unit；
- task；
- metric；
- model metadata；
- report artifact；
- ranking list snapshot。

从文件系统切换到 IoTDB、TsFile 或其他存储时，不应改变模型看到的 sample schema。存储系统应作为底层实现细节隐藏在相同逻辑实体后面。

## 13. 项目价值

TSBenchmark 的价值在于把三件事放在同一个系统里：

- 可控 benchmark 数据生成；
- 真实数据结构锚定；
- 模型结果解释与回放。

传统静态 benchmark 通常只能回答：哪个模型在固定数据集上平均分最高？TSBenchmark 希望回答更细的问题：

- 哪个模型更擅长 regime shift？
- 哪个模型能处理 lead-lag 关系？
- 哪个模型真正利用了 future covariate？
- 模型是否只在简单 difficulty 上好？
- 合成数据是否仍处于真实数据结构支撑域内？
- 动态生成不同批次后，模型排名是否稳定？
- 某个模型具体在哪些 sample 上失败？

因此，TSBenchmark 既是一个研究 benchmark，也可以发展为实用的模型评测平台。

## 14. 当前实现状态

已经实现：

- v2 统一 sample schema；
- 单变量、多变量、协变量能力维度生成；
- shard、能力测试块、track、benchmark manifest；

- 真实多变量 anchor profile；
- anchor-aware multivariate generation；
- Timer REST Service adapter；
- 多变量模型评测 runner；
- 报告聚合；
- 真实/合成数据可视化；
- sample forecast viewer。

仍待实现：

- web UI；
- 围绕 track、model、run、unit、task、report、ranking list 的后端 API；
- 更大的真实数据池；
- real probe track 与合成 anchor track 的排序一致性验证；
- 更完整的 report-level validity diagnostics；
- GPU 时间、显存、吞吐等成本指标；
- 大规模存储接入。

## 15. 总结

TSBenchmark 是一个面向时间序列预测模型的动态、可控、可解释 benchmark 平台。它的核心抽象是：评测数据由能力测试块生成，能力测试块组合成 track，track 被物化为 benchmark，模型通过 benchmarking run 参与评测，结果以 unit、task、metric 和 sample prediction 的形式保存，最后汇总成 report 和 ranking list。长期来看，TSBenchmark 应成为一个用户可以直观操作的平台：用户创建赛道、生成数据、选择模型、运行评测、查看榜单，并能从任意榜单分数回溯到产生该分数的具体样本和预测曲线。

## TODO（任务分解）

- 任务分解
- a. 数据生成方面
  - i. 单变量/多变量/协变量的基于真实数据集提取特征锚定以及能力维度数据生成机制
    1. 形式化说明（单变量初版已完成，多变量代码已完成；TODO：详细设计与文献支撑）
    2. 过程可视化（目前有静态样例；TODO：完整流程可视化）
    3. 有效性验证（TODO：通过模型在真实/合成数据上的预测结果对比）
  - ii. 大批量数据生成及存储

    1. 以 TimeBench_Tsfile 作为真实数据源
    2. 以 IoTDB 作为合成数据的存储方式
- b. 模型评测方面
  - i. 在 timer 中接入 toto 与 timesfm进行多变量推理，确定大批量评测的调用方式
  - ii. 确定赛道、评测单元的元数据存储形式（iotdb 建模）
  - iii. 后端实现从创建赛道到获取评测结果的完整流程接口
  - iv. 前端页面功能实现：
    1. 创建赛道、选择能力维度（capability）、生成对应数据集
    2. 选择赛道与参与模型进行一次评测
    3. 查看赛道榜单
      - a. 模型-能力维度得分雷达图（一维榜单可取各维度均值）
      - b. 选择某项指标，查看该指标下的模型得分榜单
      - c. 查看某样本下各模型的预测结果（曲线图、统计指标）（样本选定由 capability 选项、参数选项与 sample_id 选项组成）

(注：内容由 AI 生成，请谨慎参考）
