# TSBenchmark 平台功能定义设计

**日期：** 2026-05-15

**输入材料：** `README.md` 与 `TSBenchmark 设计文档.assets/TSBenchmark 设计文档-1_1.jpg`

**设计结论：** 本文档采用“双层定义”：先定义 TSBenchmark 的完整目标平台，再单独定义首期 `MVP` 范围。目标平台用于保留 README 中的长期架构与能力愿景；`MVP` 用于约束近期交付边界，避免首期实现范围失控。

## 1. 平台定义

TSBenchmark 是一个面向时间序列预测模型的动态、可控、可解释 `benchmark` 平台。它不只给模型一个单一总分，而是把预测能力拆解为多个 `Capability` 维度，通过可复现的数据生成、`Track` 组合、模型评测、报告分析和榜单维护，帮助用户从榜单分数一路回溯到 `Task`、`Sample`、预测曲线和具体指标。

平台的核心流程是：

```text
创建或选择 Track
-> 配置 Capability Block
-> 生成或选择 Shard 与 Sample
-> 选择参与评测的 Model
-> 启动 Benchmarking Run
-> 收集预测结果与 Metric
-> 生成 Report
-> 更新 Ranking List
-> 查看 Model、Task、Capability、Sample 与 Forecast 细节
```

因此，TSBenchmark 不是一个静态榜单网站，而是一个包含数据生成、模型执行、结果聚合、报告解释、榜单维护和样本级回放能力的评测系统。

## 2. 用户与角色

### 2.1 一般用户

一般用户主要消费评测结果并查看模型表现。他们可以：

- 查看某条 `Track` 的 `Ranking List`。
- 按 `Metric` 或 `Capability Block` 对榜单排序和筛选。
- 查看某条 `Track` 或某次 `Benchmarking Run` 的 `Report`。
- 在已有数据支持时，对比指定模型之间的表现。
- 查看 `Sample` 级预测曲线与指标。
- 提交模型或数据集上传申请。

一般用户不直接操作底层 `Shard`、执行 worker 或存储实现。

### 2.2 管理员

管理员负责管理评测资产和运行评测任务。他们可以：

- 注册、加载、查看和移除数据集。
- 注册、加载、查看和移除模型。
- 创建、查看和移除 `Track`。
- 选择一条 `Track` 和一个或多个模型，创建 `Benchmarking Run`。
- 按模型和 `Capability Block` 查看评测进度。
- 终止或删除评测任务。
- 生成和查看 `Report`。
- 发布或刷新 `Ranking List`。

### 2.3 评测系统

评测系统是 `REST API` 与 `CLI` 背后的执行层。它可以：

- 将 `Track` 物化为 benchmark 数据。
- 调用模型推理服务。
- 记录预测结果、指标、失败原因、日志和产物。
- 聚合 `Unit` 级和 `Task` 级结果。
- 生成 `Report` 产物和 `Ranking List` 快照。

## 3. 目标平台功能域

### 3.1 接入层

目标平台通过多种入口暴露同一套核心评测能力：

- `TSBenchmark Frontend`：浏览器前端，用于创建 `Track`、管理评测、查看榜单、查看报告和检查样本预测。
- `CLI`：命令行入口，用于脚本化、本地化和批处理操作。
- `SaaS + Gateway`：外部访问边界、鉴权边界和路由层。
- `Restful Module`：TSBenchmark 内部的 API 层，服务前端和网关请求。

设计口径：请求逻辑上由 `Frontend` 或 `CLI` 进入 `Gateway` / `REST API`，再进入 `Core Module`。架构图中 `Gateway` 与 `Restful Module` 的箭头按部署连接理解，不作为反向调用约束。

### 3.2 数据集与 Real Anchor 管理

数据集功能域负责管理真实数据，但真实数据在 TSBenchmark 中不是直接复制成固定公开题库，而是作为结构参照系使用。

目标功能包括：

- 注册数据集 `Manifest`，记录 dataset ID、数据领域、时间频率、文件路径、时间列和 target columns。
- 加载和查看本地或托管数据集。
- 从真实 target 矩阵中抽取时间序列结构特征。
- 构建 `Anchor Profile`，总结真实数据的结构分布。
- 使用 `Anchor Profile` 指导合成数据生成。
- 在合成数据生成后重新抽取 realized features。
- 计算有效性诊断指标，例如 support distance、prototype distance、violation rate 和 max violation。
- 后续支持 `real probe track`，用于检查 synthetic anchor 数据上的模型排序是否与真实 probe 数据上的排序大致一致。

这个功能域的目标是让评测样本在结构上“像真实数据”，但不复用真实数据中的具体未来值。

### 3.3 Capability 与数据生成

`Capability Block` 是用户可感知、可操作的 benchmark 数据单元。一个 `Capability Block` 表示某个预测能力维度下的一组参数覆盖，它索引底层生成的 `Shard`，而 `Shard` 内包含具体 `Sample`。

目标功能包括：

- 定义 `Capability` 维度。
- 配置 `Capability Block` 参数，包括 difficulty、horizon ratio、season length 或 dominant scale、target dimension、sample count、random seed 和 anchor mode。
- 为固定参数组合生成 `Shard`。
- 在统一预测数据格式下生成 `Sample`：

```text
target_history: [context_length, target_dim]
target_future: [horizon, target_dim]
history_cov: [context_length, history_cov_dim]
future_cov: [horizon, future_cov_dim]
```

目标平台支持的能力维度包括：

- 单变量：`trend`、`multi_seasonal`、`regime_switching`、`long_memory_nonlinear`、`intermittent_heteroskedastic`。
- 多变量：`common_factor`、`lead_lag_coupling`、`coherent_regime_shift`。
- 协变量：`covariate_response`。

每条 `Sample` 需要同时保存执行数据和解释数据：

- 执行数据：target history、target future、historical covariates、future covariates、列名、context length、horizon 和 target dimension。
- 解释数据：latent parameters、realized features、real anchor 引用、support diagnostics 和 baseline scores。

### 3.4 Track 与 Benchmark 物化

`Track` 是由一个或多个 `Capability Block` 组成的评测赛道。它回答的问题是：这条评测要考察哪些预测能力？

目标功能包括：

- 从选定的 `Capability Block` 创建 `Track`。
- 支持预置 `Track`，例如 univariate core、multivariate core 和 covariate-aware track。
- 支持管理员组装自定义 `Track`。
- 在模型运行前，将 `Track` 物化为 benchmark 数据。
- 保存 `Track` version、benchmark version、data version 和生成配置。

`Track` 创建应发生在 `Capability Block` 层级。`Shard` 保持为内部复现、缓存和索引单元。

### 3.5 Model 管理与推理服务

`Model` 功能域表示可以参与评测的预测后端。

目标功能包括：

- 注册模型元数据，包括 model ID、展示名称、版本、task type、输入格式、endpoint 或执行环境、硬件约束和加载状态。
- 声明模型是否支持单变量、多变量和协变量输入。
- 加载、卸载、查看和移除模型。
- 通过 adapter 边界路由推理请求。
- 记录 runtime、failure、skipped status 和 cost metrics。

设计口径：`Timer Service` 是首个具体模型推理服务实现。目标架构应暴露通用的模型推理服务抽象，使 Timer、Chronos、toto、TimesFM、baseline、本地 adapter 和未来服务都能用同一套 `Model` 语义表示。

### 3.6 Benchmarking Run 管理

`Benchmarking Run` 表示一次评测执行：在一条 `Track` 上运行一个或多个模型。

目标功能包括：

- 通过选择一条 `Track`、benchmark/data version 和一个或多个模型创建 run。
- 将 run 展开为每个模型一个 `Unit`。
- 将每个 `Unit` 展开为按模型和 `Capability Block` 划分的 `Task`。
- 对 `Sample` 执行模型推理。
- 记录预测结果、指标、日志、错误、跳过原因和 run 状态。
- 支持按模型和 `Capability Block` 查看进度。
- 支持终止和删除 run。

run 状态模型至少应支持：

```text
created -> queued -> running -> succeeded
                         |-> failed
                         |-> cancelled
```

部分失败应记录在 `Task` 或 `Sample` 层级。这样即使部分模型或任务失败，run 仍然可以产出可用报告。

### 3.7 Metric 与评分

`Metric` 定义 `Task` 和 `Unit` 如何被打分。

目标功能包括：

- 计算点预测指标，例如 MSE 和 MAE。
- 支持分析指标，例如 MASE、sMAPE、relative skill 和 runtime。
- 允许 `Capability Block` 声明适用的 `Metric`。
- 将 `Sample` 指标聚合到 `Task`、`Unit`、`Report` 和 `Ranking List` 视图。
- 后续支持成本指标，包括 GPU time、memory usage、latency 和 throughput。
- 后续支持合成数据有效性与 real-anchor alignment 指标。

设计口径：`MVP` 榜单使用 MSE 和 MAE，因为这是当前已确认的近期指标。目标平台保留 metric registry，使后续指标扩展不需要改变核心实体模型。

### 3.8 Report、Ranking 与解释

`Report` 用于解释评测结果。`Ranking List` 用于提供稳定的 `Track` 级榜单视图。

目标 `Report` 功能包括：

- 生成总体模型表。
- 展示能力维度级分数。
- 展示 difficulty 曲线和 horizon 曲线。
- 展示模型能力画像，例如雷达图。
- 展示合成数据有效性诊断。
- 展示真实数据与合成数据对比。
- 从聚合结果链接到 `Sample` 级预测视图。

目标 `Ranking List` 功能包括：

- 每条 `Track` 维护一个 `Ranking List`。
- 按指定 `Metric` 展示模型排名。
- 展示按 `Capability Block` 拆分的排名。
- 默认展示每个模型最新一次有效结果。
- 将历史最佳结果作为次级视图。
- 链接到历史 `Benchmarking Run` 和 `Report`。

设计口径：`Ranking List` 默认语义是“每个模型最新一次有效结果”；“最佳结果”是筛选项或次级 tab，不是主榜默认规则。

### 3.9 Sample Forecast 检查

`Sample` 级检查用于解释模型为什么赢或为什么输。

目标功能包括：

- 按 `Capability Block`、参数组合、sample ID、target channel、model 和 run 筛选。
- 展示 target history、ground truth、forecast 和 sample-level metric。
- 在同一条 `Sample` 上对比多个模型。
- 将 `Sample` 视图链接回 `Task`、`Unit`、run、`Report` 和 `Ranking List` 上下文。

## 4. 核心领域模型

目标平台使用以下核心实体。

| Entity | 定义 | 主要职责 |
| --- | --- | --- |
| Dataset Manifest | 真实数据源描述 | 记录数据源身份、领域、频率、路径、时间列和 target columns |
| Anchor Profile | 真实数据结构分布摘要 | 指导并验证生成的 benchmark 数据 |
| Capability | 预测能力类型 | 命名 benchmark 要测试的能力 |
| Capability Block | 某个能力及其参数覆盖下的用户可操作数据块 | 组织 `Shard` 和适用 `Metric` |
| Shard | 固定参数组合下生成的数据单元 | 缓存可复现的 `Sample` |
| Sample | 最小模型输入与答案单元 | 保存 history、future、covariates、latent parameters、realized features 和 metrics |
| Track | 多个 `Capability Block` 的组合 | 定义一条评测赛道 |
| Model | 预测后端 | 声明模型能力和推理 endpoint |
| Benchmarking Run | 一次在 `Track` 和模型集合上的执行 | 跟踪状态、版本、日志、Unit、Task、预测结果和产物 |
| Unit | 某模型在一次 run 中的完整结果 | 聚合该模型的全部 `Task` |
| Task | 某模型在某个 `Capability Block` 上的一次结果集合 | 聚合该能力维度下的样本预测和指标 |
| Metric | 评分定义 | 计算并聚合评测值 |
| Report | 解释性产物 | 解释 run 或 track 结果 |
| Ranking List | `Track` 级榜单 | 维护最新有效模型成绩和历史链接 |

基数口径：

- 一个 `Track` 包含一个或多个 `Capability Block`。
- 一个 `Capability Block` 索引一个或多个 `Shard`。
- 一个 `Shard` 包含一个或多个 `Sample`。
- 一个 `Benchmarking Run` 选择且只选择一条 `Track`，并选择一个或多个模型。
- 一个 `Benchmarking Run` 为每个被选模型产生一个 `Unit`。
- 一个 `Unit` 包含一个或多个 `Task`。
- 一个 `Task` 对应一次 `Benchmarking Run` 中的一个模型和一个 `Capability Block`。
- 一个 `Report` 可以消费一个 run，也可以消费一组被选 run。
- 一个 `Ranking List` 属于一条 `Track`。

## 5. 目标部署边界

架构图定义的目标部署形态是：

```text
Frontend / CLI
-> SaaS + Gateway
-> TSBenchmark Restful Module
-> TSBenchmark Core Module
-> Model Inference Service, first represented by Timer Service
```

在 GPU 物理机上，TSBenchmark 包含 `Restful Module` 和 `Core Module`。模型推理服务作为相邻服务部署，因为模型服务有独立的硬件需求和生命周期。

功能职责划分：

- `Frontend`：交互式用户流程和可视化。
- `CLI`：脚本化和本地化操作。
- `Gateway`：外部路由和访问控制。
- `REST module`：稳定 API 表面。
- `Core module`：领域编排、数据生成、run 管理、结果聚合和产物生成。
- `Timer/model service`：模型加载和推理执行。
- `Storage layer`：元数据、生成数据、预测结果、报告和榜单快照。

## 6. 首期 MVP 范围

`MVP` 实现的是能够展示完整评测闭环的最小平台版本。

### 6.1 MVP 范围内

`MVP` 功能范围：

- 一个受管数据集或一个数据源家族。
- 只支持单变量预测。
- 注册当前 README 确认范围中的 5 个模型。
- 以 MSE 和 MAE 作为主指标。
- 支持使用受限的单变量 `Capability Block` 创建 `Track`。
- 支持选择一条 `Track` 和一个或多个模型创建 `Benchmarking Run`。
- 支持查看 run 进度和结果状态。
- 提供某条 `Track` 的 `Ranking List` 页面。
- 提供 run 或 track 的 `Report` 页面。
- 提供 `Sample Forecast` 页面，展示 history、ground truth、forecast 和 sample metric。
- 提供围绕 track、model、run、unit、task、report、ranking list 的 `REST API`。

`MVP` 页面范围：

- `Track Builder`。
- `Benchmarking Run Page`。
- `Track Ranking Page`。
- `Report Page`。
- `Sample Forecast Page`。

`MVP` 后端范围：

- 持久化 track、capability block、model、benchmarking run、unit、task、metric、report 和 ranking list 元数据。
- 对生成数据、评测输出、报告和可视化使用现有文件产物结构，或增加一层轻量持久化封装。
- 通过稳定逻辑实体隐藏底层存储实现。

### 6.2 MVP 范围外

`MVP` 暂不包含：

- 面向产品使用的多变量 benchmark 生成与评测流程。
- 面向产品使用的协变量 benchmark 生成与评测流程。
- 完整的 real-anchor profile 管理 UI。
- `real probe track` 验证。
- 将大规模 IoTDB 或 TsFile 存储迁移作为首期强依赖。
- 除基础 runtime 之外的 GPU 成本指标。
- 公开自助式模型或数据集上传审批流程。
- 多租户计费、配额管理和高级 SaaS 管理功能。

这些排除项不否定目标平台能力，只用于明确首期实现边界。

## 7. 目标平台与 MVP 差异

| 领域 | 目标平台 | MVP |
| --- | --- | --- |
| 用户 | 一般用户、管理员、CLI 操作者、系统集成方 | 一般用户和管理员 |
| 接入 | Frontend、CLI、SaaS Gateway、REST API | Frontend 和 REST API；CLI 可以沿用现有本地能力 |
| 数据源 | 多数据集、Manifest、Anchor Profile、real probe track | 一个数据集或一个数据源家族 |
| 任务类型 | 单变量、多变量、协变量 | 单变量 |
| Capability Block | 完整能力目录 | 受限单变量能力集合 |
| 数据生成 | real anchor 指导的动态生成与有效性诊断 | 记录参数的单变量动态生成 |
| 模型 | 多模型服务 adapter 注册体系 | 当前确认范围中的 5 个模型 |
| Metric | MSE、MAE、MASE、sMAPE、relative skill、runtime、cost、validity metrics | MSE 和 MAE |
| Run 编排 | 队列、取消、部分失败、日志、产物管理 | 创建 run、查看进度、成功/失败状态、结果捕获 |
| Ranking | 默认最新有效结果，可切换最佳结果和历史记录 | 默认最新有效结果 |
| Report | 能力、difficulty、horizon、validity、sample 级完整分析 | 汇总表、指标表、基础样本下钻 |
| 存储 | 元数据存储 + 可扩展数据/产物存储，未来接入 IoTDB/TsFile | 文件产物或轻量持久化封装 |
| 模型服务 | 通用推理服务抽象，Timer 作为首个实现 | 面向 5 个 MVP 模型的 Timer/model service 集成 |

## 8. 功能验收标准

当设计能够支持以下端到端场景时，平台功能定义即满足要求。

### 8.1 MVP 场景

1. 管理员从可用 `Capability Block` 创建一条单变量 `Track`。
2. 系统将该 `Track` 物化为 benchmark 数据。
3. 管理员选择这条 `Track` 和 5 个已注册模型。
4. 系统创建一个 `Benchmarking Run`。
5. run 通过模型服务执行推理。
6. 系统记录预测结果并计算 MSE 和 MAE。
7. 系统生成 `Unit` 和 `Task` 结果。
8. 系统生成 `Report`。
9. 系统使用每个模型最新一次有效结果更新 `Track Ranking List`。
10. 一般用户打开榜单页面，查看排序结果，进入报告，并下钻到 `Sample Forecast`。

### 8.2 目标平台场景

1. 管理员注册一个真实数据集 `Manifest`。
2. 系统抽取结构特征并构建 `Anchor Profile`。
3. 管理员创建一条覆盖单变量、多变量和协变量 `Capability Block` 的 `Track`。
4. 系统生成 anchor-guided `Shard` 和 `Sample`。
5. 管理员选择多个声明能力兼容的模型。
6. 系统执行评测，记录预测结果、指标、runtime、错误和产物。
7. `Report` 按模型、能力维度、difficulty、horizon、sample 和有效性诊断解释结果。
8. `Ranking List` 更新最新有效成绩并链接到历史 run。
9. 用户检查某个模型为什么在某个能力和样本上表现好或差。

## 9. 架构约束

这份功能定义带来以下架构约束：

- `Capability Block`、`Track`、`Benchmarking Run`、`Unit`、`Task`、`Report` 和 `Ranking List` 应成为后端一等实体。
- 即使物理存储改变，`Shard` 和 `Sample` 也应保持为稳定的数据概念。
- 统一 `Sample` schema 必须在文件存储、`REST API` 和模型服务 adapter 之间保持稳定。
- 模型推理边界应基于 adapter，而不是在领域模型中绑定到 `Timer Service`。
- `Ranking List` 语义必须明确：默认使用最新有效结果，最佳结果作为可选视图。
- `MVP` 在单变量端到端闭环完成前，不应提前建设多变量、协变量、real-anchor UI 或大规模存储迁移。

## 10. 自查

- 文档没有遗留占位需求。
- 文档已明确区分目标平台和 `MVP` 范围。
- 来源材料中的歧义已转化为明确设计口径。
- 功能模型保留了 README 中的核心概念和架构图中的模块。
- `MVP` 范围足够小，可以作为下一步实施计划的输入。
