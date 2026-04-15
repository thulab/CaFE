- 测试工具的根目录为 `${root}`
- 当前范围聚焦零样本时序预测动态评测，不包含训练/微调流程
- 本文档定义目标领域模型，同时标注当前系统与目标模型的差异

# 总体结构
- `datas`: 管理数据源、动态数据集、数据画像、变换定义、校验与批次发布
- `models`: 管理模型元信息、运行时规格、依赖环境、加载入口与参数定义
- `tasks`: 管理赛道模板、赛道变体、评测计划、逻辑任务、实际执行、结果与资源采集
- `leaderboards`: 管理按指标生成的赛道榜、总榜与赛季榜
- `reports`: 管理模型报告、榜单报告与 bad case 报告

# 设计原则
- 原始数据文件不是最终评测数据；评测使用经过生成、变换或混合后的动态数据
- “赛道”不是“任务”；赛道描述输入输出结构、噪声模式与执行约束，任务描述一次模型评测行为
- 一个逻辑任务默认执行 3 次；稳定性通过多次实际执行聚合得到
- 榜单直接按指标出榜，不再维护综合分、等级分或积分分
- 文档优先描述稳定的领域边界，不直接迁就当前实现里的兼容字段

# 当前系统对齐总览
- `datas`: 已部分落地。数据源、数据集、赛道变体、结构化通道、输入长度/预测长度已进入后端；但变换定义、发布生命周期、细粒度校验报告还没独立成型
- `models`: 落地最少。当前更像“模型注册记录 + Hugging Face 适配配置”，还不是清晰分离的 `model + model_runtime_spec`
- `tasks`: 已部分落地。`task_run` 和重复执行已落地；但 `evaluation_plan`、`resource_usage`、`evaluation_result` 仍未独立成完整对象
- `leaderboards`: 已部分落地。按指标出榜已接入后端和前端；但总榜仍采用 `rank_sum`，且仍保留 `composite_score` 兼容语义
- `reports`: 初步落地。已有任务报告，但还不是独立的 `analysis_report` 体系，也没有榜单报告和 bad case 报告产品面

# datas 模块

## 模块职责
- 管理原始数据源登记
- 管理动态数据集生成、加载与持久化
- 管理数据画像、数据变换与数据校验
- 管理批次发布、冻结与赛道投放

## 目标模型

### dataset_source
- `source_id`: 唯一字符串，数据源编号，主键
- `source_name`: 数据源名称
- `source_type`: `csv` / `tsfile` / `synthetic_seed` / `mixed_source`
- `source_path`: 数据源存储位置；本地文件统一位于 `${root}/runtime/dataset_sources/{source_id}/`
- `source_schema`: 原始字段定义，至少包含时间列、序列标识列、目标列、协变量列
- `source_domain_tags`: 数据领域标签
- `source_desc`: 数据源说明
- `source_create_time`: 数据源登记时间

### dataset
- `dataset_id`: 唯一字符串，动态数据集编号，主键
- `dataset_name`: 数据集名称
- `dataset_kind`: `loaded` / `synthetic` / `transformed` / `mixed`
- `base_source_ids`: 基底数据源编号列表
- `storage_format`: `csv` / `tsfile`
- `dataset_path`: 数据集存储路径，统一位于 `${root}/runtime/datasets/{dataset_id}/`
- `time_column_name`: 时间列名
- `series_id_column_name`: 序列标识列名；单序列场景可为空
- `target_column_name`: 主要目标列名
- `covariate_column_names`: 协变量列名列表
- `freq`: 时间频率
- `prediction_length_range`: 支持的预测长度范围
- `dataset_feature_profile_id`: 数据画像编号
- `dataset_create_time`: 数据集生成时间

### dataset_feature_profile
- `profile_id`: 唯一字符串，画像编号，主键
- `dataset_id`: 对应数据集编号
- `trend_tags`: 趋势特征标签
- `seasonality_tags`: 周期/季节性特征标签
- `dominant_periods`: 主要周期集合
- `noise_level`: 噪声强度估计
- `missing_rate`: 缺失率
- `outlier_rate`: 异常值比例
- `feature_summary`: 供赛道映射与报告生成使用的结构化摘要

### transform_spec
- `transform_spec_id`: 唯一字符串，变换定义编号，主键
- `transform_type`: `synthetic_generation` / `augmentation` / `mixup` / `noise_injection` / `covariate_permutation`
- `input_dataset_ids`: 输入数据集编号列表
- `transform_params`: 变换参数
- `expected_feature_changes`: 预期增强或削弱的特征集合
- `random_seed`: 随机种子
- `output_constraints`: 输出约束；用于保证物理意义、范围边界与无明显缺陷
- `transform_create_time`: 变换定义创建时间

### validation_report
- `validation_report_id`: 唯一字符串，校验报告编号，主键
- `dataset_id`: 被校验的数据集编号
- `transform_spec_id`: 对应变换定义编号，可为空
- `validator_name`: 校验器名称
- `validation_status`: `passed` / `failed` / `needs_regenerate`
- `rule_results`: 各项规则的检查结果
- `issue_summary`: 失败原因或风险摘要
- `retry_count`: 当前数据集为通过校验已重试的次数
- `validation_time`: 校验时间

### dataset_batch
- `batch_id`: 唯一字符串，批次编号，主键
- `batch_name`: 批次名称
- `batch_period`: 批次周期标识，如月度批次
- `dataset_ids`: 本批次发布的数据集编号列表
- `track_variant_ids`: 本批次启用的赛道变体编号列表
- `release_time`: 批次发布时间
- `freeze_time`: 批次封板时间
- `batch_status`: `draft` / `published` / `closed` / `archived`
- `attempt_limit_per_model`: 同一模型在同一批次允许的最大逻辑任务提交次数
- `fairness_note`: 批次公平性说明，如是否参与月考、是否纳入赛季统计

## 当前实现映射
- 主要代码位于 `backend/app/datasets/`
- 已有 `DatasetSourceRecord`、`DatasetRecord`、`DatasetBatch`、`DatasetFeatureProfile`
- 已有结构化赛道定义：`TrackTemplateKind`、`NoiseMode`、`ExecutionConstraint`、`TrackSpec`
- `SeriesSample` 已同时支持兼容视图 `history/target/covariates` 和结构化通道视图
- `storage.py` 已能持久化 `dataset_sources`、`datasets`、`batches`

## 与目标模型的差异
- `transform_spec` 还没有独立实体与持久化目录，当前生成逻辑仍主要散落在 `synthetic.py` 和 loader/processor 流程中
- `validation_report` 目前只有简单的 `passed + issues`，没有 `validator_name`、`rule_results`、`retry_count` 等结构化字段
- `dataset_batch` 仍保留旧兼容字段 `track`、`context_length`、`horizon`，还没有完全收敛到 `track_variant_id + input_length + prediction_length`
- 批次生命周期还偏“生成即使用”，`release_time / freeze_time / batch_status / fairness_note` 没有形成完整发布流程
- `dataset_source_type` 目前只覆盖 `synthetic / csv / tsfile`，还没有明确的 `mixed_source`
- 当前 CSV loader 已支持结构化列映射，但 TsFile、混合源、跨数据集 mixup 仍未产品化

# models 模块

## 模块职责
- 管理模型注册与来源信息
- 管理运行时加载方式、推理入口与依赖环境
- 管理可配置参数定义与运行资源约束

## 目标模型

### model
- `model_id`: 唯一字符串，模型编号，主键
- `model_name`: 模型名称
- `model_alias_name`: 模型别名
- `model_source_type`: `huggingface` / `uploaded` / `self_hosted`
- `model_url`: 模型来源地址，如 Hugging Face URL
- `model_path`: 本地模型文件路径，统一位于 `${root}/runtime/models/{model_id}/`
- `model_card`: 模型说明与元信息摘要
- `model_task_scope`: 支持的任务范围，当前至少标明是否支持零样本预测
- `model_params_schema`: 模型可配置参数定义，包含名称、类型、默认值、取值范围
- `model_create_time`: 创建时间
- `model_update_time`: 更新时间

### model_runtime_spec
- `runtime_spec_id`: 唯一字符串，运行时规格编号，主键
- `model_id`: 对应模型编号
- `loader_entry`: 模型加载入口或调用方式
- `predict_entry`: 推理入口或调用方式
- `python_environment`: 依赖环境描述
- `hardware_requirements`: 运行硬件要求
- `resource_collect_policy`: 资源指标采集策略
- `token_count_policy`: token 统计策略；非 LLM 模型可为空
- `runtime_create_time`: 创建时间

## 当前实现映射
- 主要代码位于 `backend/app/models/`
- 当前核心对象是 `ModelRecord`
- Hugging Face 接入已经有 `HuggingFaceConfig`、任务推断、参数定义生成与加载状态管理
- 当前已经能表达一部分运行参数定义，如 `batch_size`、`context_length`、`use_covariates`

## 与目标模型的差异
- `model` 和 `model_runtime_spec` 还没有拆开；当前 `ModelRecord` 混合了静态元信息、来源信息、运行配置、运行状态
- 当前模型结构明显偏 Hugging Face 适配实现，`loader_entry / predict_entry / python_environment` 仍缺少统一抽象
- `model_params_schema` 目前只覆盖少量运行时参数定义，没有统一的默认值、取值范围、校验规则结构
- `model_task_scope` 仍然较弱，更多是 `capabilities` 字符串，而不是稳定的任务能力模型
- 当前实现仍大量沿用 `context_length` 术语，没有完全切换到与 plan 对齐的 `input_length`
- 资源采集策略还没有作为独立 runtime spec 管理；目前主要在任务执行和模型执行层临时计算

# tasks 模块

## 模块职责
- 管理基础赛道模板与带噪声赛道变体
- 管理评测计划、逻辑任务与多次实际执行
- 管理指标、资源消耗、结果聚合与稳定性统计

## 目标模型

### track_template
- `track_template_id`: 唯一字符串，基础赛道模板编号，主键
- `track_kind`: 固定为以下 5 类之一
  - `univariate_forecast`
  - `multivariate_forecast_all_to_all`
  - `multivariate_forecast_all_to_subset`
  - `multivariate_forecast_with_future_covariates`
  - `multivariate_forecast_via_univariate`
- `track_name`: 赛道模板名称，只表达基础赛道名
- `input_channels`: 历史输入通道定义
- `target_channels`: 预测目标通道定义
- `future_known_channels`: 未来已知协变量通道定义
- `execution_constraint`: 执行约束；例如 `joint_multivariate` 或 `per_channel_univariate`
- `track_desc`: 赛道模板说明

### track_variant
- `track_variant_id`: 唯一字符串，赛道变体编号，主键
- `track_template_id`: 对应基础赛道模板编号
- `noise_mode`: `clean` / `noisy`
- `transform_strategy`: 该赛道变体绑定的数据变换策略说明
- `metric_ids`: 该赛道变体使用的指标编号列表
- `variant_desc`: 赛道变体说明

### evaluation_plan
- `plan_id`: 唯一字符串，评测计划编号，主键
- `batch_id`: 所属数据批次编号
- `track_variant_id`: 所属赛道变体编号
- `dataset_selector`: 数据选择规则
- `input_length`: 计划配置的模型输入长度
- `prediction_length`: 计划配置的模型预测长度
- `metric_ids`: 本计划实际使用的指标编号列表
- `execution_repeat_count`: 每个逻辑任务的默认执行次数，默认值为 `3`
- `plan_create_time`: 评测计划创建时间

### evaluation_task
- `task_id`: 唯一字符串，逻辑任务编号，主键
- `plan_id`: 所属评测计划编号
- `model_id`: 参与评测的模型编号
- `runtime_spec_id`: 使用的运行时规格编号
- `dataset_id`: 实际执行所用数据集编号
- `task_status`: `created` / `queued` / `running` / `succeeded` / `failed`
- `attempt_no`: 当前模型在当前批次上的第几次逻辑任务提交
- `task_model_params`: 本次任务实际使用的模型参数
- `task_create_time`: 任务创建时间
- `task_start_time`: 任务开始时间
- `task_end_time`: 任务结束时间
- `failure_reason`: 失败原因

### task_run
- `run_id`: 唯一字符串，实际执行编号，主键
- `task_id`: 对应逻辑任务编号
- `run_no`: 该逻辑任务下的第几次执行
- `run_status`: `queued` / `running` / `succeeded` / `failed`
- `run_start_time`: 执行开始时间
- `run_end_time`: 执行结束时间
- `failure_reason`: 本次执行失败原因

### metric
- `metric_id`: 唯一字符串，指标编号，主键
- `metric_name`: 指标名称
- `metric_category`: `quality` / `latency` / `resource` / `stability`
- `metric_value_type`: 指标值类型
- `optimize_direction`: `min` / `max`
- `aggregation_method`: 多次执行或多样本聚合方式
- `metric_desc`: 指标说明

### resource_usage
- `usage_id`: 唯一字符串，资源消耗记录编号，主键
- `run_id`: 对应实际执行编号
- `data_load_time_ms`: 加载数据耗时
- `predict_time_ms`: 执行预测耗时
- `token_count`: 推理 token 数；非 LLM 模型可为空
- `gpu_time_seconds`: GPU 卡时
- `peak_gpu_memory_mb`: 峰值显存占用
- `peak_cpu_percent`: 峰值 CPU 占用
- `peak_memory_mb`: 峰值内存占用
- `usage_record_time`: 资源信息记录时间

### evaluation_result
- `result_id`: 唯一字符串，结果编号，主键
- `task_id`: 对应逻辑任务编号
- `run_ids`: 参与聚合的实际执行编号列表
- `metric_values`: 聚合后的指标结果
- `stability_stats`: 多次执行统计结果，如均值、标准差、最优值、最差值
- `prediction_artifact_path`: 预测输出及中间产物位置，统一位于 `${root}/runtime/results/{result_id}/`
- `bad_case_refs`: bad case 索引列表
- `result_summary`: 结果摘要
- `result_time`: 结果生成时间

## 当前实现映射
- 赛道相关定义目前主要落在 `backend/app/datasets/domain.py`
- 任务执行主要位于 `backend/app/tasks/`
- 已落地 `task_run` 与 `execution_repeat_count`
- 前后端已经支持 `track_variant_id`、`metric_id`、`execution_repeat_count`

## 与目标模型的差异
- `evaluation_plan` 还没有独立实体；当前任务更多是“直接从 batch 启动”
- `evaluation_task` 仍保留旧 `track` 字段，且没有真正的 `plan_id / runtime_spec_id / dataset_id / attempt_no`
- `evaluation_result` 还没有独立持久化对象；当前聚合结果仍直接挂在 `EvaluationTask.metrics` 和 `sample_outcomes`
- `resource_usage` 还没有独立对象与持久化目录，运行时资源指标仍以任务汇总字段为主
- 指标体系仍混有 `composite_score`，这和目标模型“榜单直接按指标出榜”不完全一致
- `metric` 还不是显式注册表，当前是代码里的指标字符串常量
- 结果产物、预测 artifact、bad case 索引、按 run 追溯接口仍不完整

# leaderboards 模块

## 模块职责
- 基于任务结果按指标生成赛道榜和总榜
- 管理榜单范围、赛季范围和榜单条目
- 管理榜单聚合规则与缺失赛道惩罚策略

## 目标模型

### season
- `season_id`: 唯一字符串，赛季编号，主键
- `season_name`: 赛季名称
- `start_time`: 赛季开始时间
- `end_time`: 赛季结束时间
- `batch_ids`: 纳入赛季统计的批次编号列表
- `season_status`: `planned` / `running` / `closed`
- `season_desc`: 赛季说明

### leaderboard
- `leaderboard_id`: 唯一字符串，榜单编号，主键
- `leaderboard_name`: 榜单名称
- `metric_id`: 榜单对应的单一指标编号
- `leaderboard_scope`: `track` / `overall`
- `track_variant_id`: 对应赛道变体编号；总榜可为空
- `noise_mode`: 榜单过滤的噪声模式，`clean` / `noisy`
- `batch_id`: 对应批次编号；批次榜时必填
- `season_id`: 对应赛季编号；赛季榜时必填
- `leaderboard_time`: 榜单生成时间

### leaderboard_entry
- `entry_id`: 唯一字符串，榜单条目编号，主键
- `leaderboard_id`: 所属榜单编号
- `model_id`: 模型编号
- `rank_no`: 排名
- `metric_value`: 该榜单指标下的直接结果值
- `metric_snapshot`: 附加指标快照
- `support_result_ids`: 支撑该条目的结果编号列表

## 当前实现映射
- 主要代码位于 `backend/app/leaderboards/`
- 已经按 `metric_id` 生成赛道榜和总榜
- 已经使用 `track_variant_id` 作为主要赛道标识
- 前端页面已支持切换 `metric_id`

## 与目标模型的差异
- `leaderboard` 和 `leaderboard_entry` 目前还是运行时视图，不是独立持久化实体
- `season` 还没有任何后端实体与统计入口
- 当前总榜仍使用 `rank_sum` 聚合，这是一种实现策略，不是完整的领域模型
- 兼容层仍暴露 `composite_score / mean_composite_score / track_scores` 等旧语义
- `noise_mode` 还主要隐含在 `track_variant_id` 中，未作为榜单过滤字段显式建模
- 当前 API 和 dashboard 仍更偏“当前视图查询”，不是可追溯、可归档的榜单对象管理

# reports 模块

## 模块职责
- 管理模型报告、榜单报告与 bad case 报告
- 管理报告输入引用、生成方式和可追溯输出

## 目标模型

### analysis_report
- `report_id`: 唯一字符串，报告编号，主键
- `report_type`: `model_report` / `leaderboard_report` / `bad_case_report`
- `target_id`: 报告目标编号；可对应模型、榜单或结果
- `input_refs`: 报告输入引用；可包含逻辑任务、实际执行、结果、榜单、数据画像
- `report_summary`: 报告摘要
- `report_file_path`: 报告文件路径，统一位于 `${root}/runtime/reports/{report_id}.pdf`
- `generator_info`: 报告生成方式说明，如规则模板或 Agent 分析
- `report_time`: 报告生成时间

## 当前实现映射
- 当前后端已有 `BenchmarkReport`
- 当前前端已有报告详情页，能展示任务摘要、风险、bad cases、`run_ids`

## 与目标模型的差异
- 当前报告本质上仍是任务执行报告，不是统一的 `analysis_report`
- 报告类型还没有扩展到 `leaderboard_report` 和 `bad_case_report`
- `input_refs`、`generator_info`、PDF 或其他产物路径还没有进入统一模型
- bad case 目前是文本摘要，不是可检索、可追踪的结果引用集合

# 跨模块关系
- `dataset_source -> dataset -> dataset_batch -> evaluation_plan -> evaluation_task -> task_run -> evaluation_result`
- `transform_spec` 与 `validation_report` 共同描述“如何生成动态考题”以及“是否允许发布”
- `track_template` 定义基础赛道结构，`track_variant` 通过 `noise_mode` 形成 clean/noisy 两种版本
- `track_variant` 由 `track_kind + noise_mode + execution_constraint` 共同确定
- `evaluation_task` 是最小逻辑评测单元，`task_run` 是该任务的一次实际执行
- `resource_usage` 记录单次执行资源消耗，`evaluation_result` 聚合逻辑任务下的多次执行结果
- `leaderboard` 基于单个 `metric_id` 聚合；赛道榜和总榜都不依赖综合分
- `analysis_report` 必须关联可追溯输入，而不只是最终展示文本

# 按模块的核心操作

## datas
- `register dataset source`: 导入或登记原始数据源
- `generate dynamic dataset`: 基于生成、变换或混合策略产出动态数据集
- `validate dataset`: 生成校验报告，决定是否进入批次
- `publish dataset batch`: 绑定批次、赛道变体、发布时间与冻结时间

## models
- `register model`: 录入模型静态元信息
- `register runtime spec`: 录入加载入口、依赖环境、硬件要求与资源采集策略
- `load model`: 根据运行时规格完成模型加载

## tasks
- `create evaluation plan`: 配置 `input_length`、`prediction_length`、`metric_ids`、`execution_repeat_count`
- `run evaluation task`: 为模型创建逻辑任务，并在任务内部生成 3 个 `task_run`
- `collect resource usage`: 对每个 `task_run` 记录资源消耗
- `aggregate evaluation result`: 将多次执行聚合为单个逻辑任务结果

## leaderboards
- `generate track leaderboard`: 针对单个 `metric_id` 生成赛道榜
- `generate overall leaderboard`: 针对单个 `metric_id` 生成总榜
- `generate season leaderboard`: 在赛季范围内聚合批次榜结果

## reports
- `generate model report`: 读取任务结果、稳定性统计与 bad case
- `generate leaderboard report`: 读取榜单和结果集合
- `generate bad case report`: 读取 bad case 引用与数据画像

# 后续改造优先级
- 第一优先级：重构 `models`，把当前 `ModelRecord` 拆成更清晰的 `model + model_runtime_spec`
- 第二优先级：在 `tasks` 中补齐 `evaluation_plan`、`evaluation_result`、`resource_usage`
- 第三优先级：让 `leaderboards` 持久化真实 `leaderboard` 与 `leaderboard_entry`，并移除 `composite_score` 兼容依赖
- 第四优先级：把 `reports` 扩展为统一 `analysis_report` 体系，并补齐榜单报告和 bad case 报告
