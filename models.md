- 测试工具的根目录为 `${root}`
- 当前范围聚焦零样本时序预测动态评测，不包含训练/微调流程
- 本文档只定义领域模型，不同步约束当前代码实现

# architecture
- `datas`: 管理数据源、动态数据集、批次发布、数据校验与数据画像
- `models`: 管理模型元信息、运行时约束、依赖环境与加载入口
- `tasks`: 管理赛道模板、赛道变体、评测计划、逻辑任务、实际执行、结果与资源消耗
- `leaderboards`: 管理按指标生成的赛道榜与总榜
- `reports`: 管理画像报告、榜单报告与 bad case 报告

# design principles
- 不直接把原始数据文件当作最终评测数据集；评测使用经过生成、变换或混合后的动态数据
- “赛道”不是“任务”；任务是一次逻辑评测，赛道定义的是输入输出结构、噪声模式与执行约束
- 一个逻辑任务默认执行 3 次；稳定性通过多次实际执行聚合得到
- 榜单直接按指标出榜，不再维护综合分、等级分或积分分

# core models

## dataset_source
- `source_id`: 唯一字符串，数据源编号，主键
- `source_name`: 数据源名称
- `source_type`: 数据源类型，`csv` / `tsfile` / `synthetic_seed` / `mixed_source`
- `source_path`: 原始数据源存储位置；本地文件统一落在 `${root}/runtime/dataset_sources/{source_id}/`
- `source_schema`: 原始字段定义，JSON 格式，至少包含时间列、目标列、协变量列、序列标识列
- `source_domain_tags`: 数据领域标签，如能源、交通、气象
- `source_desc`: 数据源说明
- `source_create_time`: 数据源登记时间

## dataset
- `dataset_id`: 唯一字符串，动态数据集编号，主键
- `dataset_name`: 数据集名称
- `dataset_kind`: 数据集类型，`loaded` / `synthetic` / `transformed` / `mixed`
- `base_source_ids`: 基底数据源编号列表，JSON 数组；纯生成数据可为空数组
- `storage_format`: 存储格式，`csv` / `tsfile`
- `dataset_path`: 数据集存储路径，统一位于 `${root}/runtime/datasets/{dataset_id}/`
- `time_column_name`: 时间列名
- `series_id_column_name`: 序列标识列名；单序列场景可为空
- `target_column_name`: 目标列名
- `covariate_column_names`: 协变量列名列表，JSON 数组
- `freq`: 时间频率
- `prediction_length_range`: 支持的预测长度范围，JSON 格式
- `dataset_feature_profile_id`: 数据画像编号
- `dataset_create_time`: 数据集生成时间

## dataset_feature_profile
- `profile_id`: 唯一字符串，数据画像编号，主键
- `dataset_id`: 对应数据集编号
- `trend_tags`: 趋势特征标签，JSON 数组
- `seasonality_tags`: 周期/季节性特征标签，JSON 数组
- `dominant_periods`: 主要周期集合，JSON 数组
- `noise_level`: 噪声强度估计
- `missing_rate`: 缺失率
- `outlier_rate`: 异常值比例
- `feature_summary`: 对数据特征的结构化摘要，供赛道映射和报告生成使用

## transform_spec
- `transform_spec_id`: 唯一字符串，变换定义编号，主键
- `transform_type`: 变换类型，`synthetic_generation` / `augmentation` / `mixup` / `noise_injection` / `covariate_permutation`
- `input_dataset_ids`: 输入数据集编号列表，JSON 数组
- `transform_params`: 变换参数，JSON 格式；例如周期、相位、噪声、协变量扰动强度、混合比例等
- `expected_feature_changes`: 预期增强或削弱的特征集合，JSON 格式
- `random_seed`: 随机种子
- `output_constraints`: 输出约束，JSON 格式；用于保证物理意义、范围边界与无明显缺陷
- `transform_create_time`: 变换定义创建时间

## validation_report
- `validation_report_id`: 唯一字符串，校验报告编号，主键
- `dataset_id`: 被校验的数据集编号
- `transform_spec_id`: 对应变换定义编号，可为空
- `validator_name`: 校验器名称
- `validation_status`: 校验结果，`passed` / `failed` / `needs_regenerate`
- `rule_results`: 各项规则的检查结果，JSON 格式
- `issue_summary`: 失败原因或风险摘要
- `retry_count`: 当前数据集为通过校验已重试的次数
- `validation_time`: 校验时间

## dataset_batch
- `batch_id`: 唯一字符串，批次编号，主键
- `batch_name`: 批次名称
- `batch_period`: 批次周期标识，如月度批次
- `dataset_ids`: 本批次发布的数据集编号列表，JSON 数组
- `track_variant_ids`: 本批次启用的赛道变体编号列表，JSON 数组
- `release_time`: 批次发布时间
- `freeze_time`: 批次封板时间
- `batch_status`: 批次状态，`draft` / `published` / `closed` / `archived`
- `attempt_limit_per_model`: 同一模型在同一批次允许的最大逻辑任务提交次数，用于防刷榜
- `fairness_note`: 批次公平性说明，如是否参与月考、是否纳入赛季统计

## model
- `model_id`: 唯一字符串，模型编号，主键
- `model_name`: 模型名称
- `model_alias_name`: 模型别名，自定义
- `model_source_type`: 模型来源类型，`huggingface` / `uploaded` / `self_hosted`
- `model_url`: 模型来源地址，如 Hugging Face URL
- `model_path`: 模型相关文件的相对存储路径，统一位于 `${root}/runtime/models/{model_id}/`
- `model_card`: 模型说明与元信息摘要
- `model_task_scope`: 支持的任务范围，当前至少标明是否支持零样本预测
- `model_params_schema`: 模型可配置参数定义，JSON 格式；包含名称、类型、默认值、取值范围
- `model_create_time`: 模型创建时间
- `model_update_time`: 模型更新时间

## model_runtime_spec
- `runtime_spec_id`: 唯一字符串，运行时规格编号，主键
- `model_id`: 对应模型编号
- `loader_entry`: 模型加载入口或调用方式
- `predict_entry`: 推理入口或调用方式
- `python_environment`: 依赖环境描述
- `hardware_requirements`: 运行硬件要求，JSON 格式
- `resource_collect_policy`: 资源指标采集策略，JSON 格式
- `token_count_policy`: token 统计策略；非 LLM 模型可为空
- `runtime_create_time`: 运行时规格创建时间

## track_template
- `track_template_id`: 唯一字符串，基础赛道模板编号，主键
- `track_kind`: 基础赛道类型，固定为以下 5 类之一
  - `univariate_forecast`: 单变量预测，单通道输入预测同一通道
  - `multivariate_forecast_all_to_all`: `1,2,3 -> 1,2,3`
  - `multivariate_forecast_all_to_subset`: `1,2,3,4,5 -> 4,5`
  - `multivariate_forecast_with_future_covariates`: `1,2,3,4,5 + future(1,2,3) -> 4,5`
  - `multivariate_forecast_via_univariate`: `1,2,3 -> 1,2,3`，但要求模型按单变量方式分别完成预测
- `track_name`: 赛道模板名称，只表达基础赛道名，不编码噪声版本
- `input_channels`: 历史输入通道定义，JSON 数组
- `target_channels`: 预测目标通道定义，JSON 数组
- `future_known_channels`: 未来已知协变量通道定义，JSON 数组
- `execution_constraint`: 执行约束说明；例如“joint_multivariate”或“per_channel_univariate”
- `track_desc`: 赛道模板说明

## track_variant
- `track_variant_id`: 唯一字符串，赛道变体编号，主键
- `track_template_id`: 对应基础赛道模板编号
- `noise_mode`: 噪声模式，`clean` / `noisy`
- `transform_strategy`: 该赛道变体绑定的数据变换策略说明
- `metric_ids`: 该赛道变体使用的指标编号列表，JSON 数组
- `variant_desc`: 赛道变体说明

## evaluation_plan
- `plan_id`: 唯一字符串，评测计划编号，主键
- `batch_id`: 所属数据批次编号
- `track_variant_id`: 所属赛道变体编号
- `dataset_selector`: 数据选择规则，JSON 格式
- `input_length`: 计划配置的模型输入长度
- `prediction_length`: 计划配置的模型预测长度
- `metric_ids`: 本计划实际使用的指标编号列表，JSON 数组
- `execution_repeat_count`: 每个逻辑任务的默认执行次数，默认值为 `3`
- `plan_create_time`: 评测计划创建时间

## evaluation_task
- `task_id`: 唯一字符串，逻辑任务编号，主键
- `plan_id`: 所属评测计划编号
- `model_id`: 参与评测的模型编号
- `runtime_spec_id`: 使用的运行时规格编号
- `dataset_id`: 实际执行所用数据集编号
- `task_status`: 逻辑任务状态，`created` / `queued` / `running` / `succeeded` / `failed`
- `attempt_no`: 当前模型在当前批次上的第几次逻辑任务提交
- `task_model_params`: 本次任务实际使用的模型参数，JSON 格式
- `task_create_time`: 任务创建时间
- `task_start_time`: 任务开始时间
- `task_end_time`: 任务结束时间
- `failure_reason`: 失败原因；成功时为空

## task_run
- `run_id`: 唯一字符串，实际执行编号，主键
- `task_id`: 对应逻辑任务编号
- `run_no`: 该逻辑任务下的第几次执行
- `run_status`: 执行状态，`queued` / `running` / `succeeded` / `failed`
- `run_start_time`: 执行开始时间
- `run_end_time`: 执行结束时间
- `failure_reason`: 本次执行失败原因；成功时为空

## metric
- `metric_id`: 唯一字符串，指标编号，主键
- `metric_name`: 指标名称
- `metric_category`: 指标类别，`quality` / `latency` / `resource` / `stability`
- `metric_value_type`: 指标值类型
- `optimize_direction`: 优化方向，`min` / `max`
- `aggregation_method`: 多次执行或多样本聚合方式
- `metric_desc`: 指标说明

## resource_usage
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

## evaluation_result
- `result_id`: 唯一字符串，结果编号，主键
- `task_id`: 对应逻辑任务编号
- `run_ids`: 参与聚合的实际执行编号列表，JSON 数组
- `metric_values`: 聚合后的指标结果，JSON 格式
- `stability_stats`: 基于多次执行的统计结果，JSON 格式；如均值、标准差、最优值、最差值、分布摘要
- `prediction_artifact_path`: 预测输出及中间产物位置，统一位于 `${root}/runtime/results/{result_id}/`
- `bad_case_refs`: Bad case 索引列表，JSON 数组
- `result_summary`: 结果摘要
- `result_time`: 结果生成时间

## season
- `season_id`: 唯一字符串，赛季编号，主键
- `season_name`: 赛季名称
- `start_time`: 赛季开始时间
- `end_time`: 赛季结束时间
- `batch_ids`: 纳入赛季统计的批次编号列表，JSON 数组
- `season_status`: 赛季状态，`planned` / `running` / `closed`
- `season_desc`: 赛季说明

## leaderboard
- `leaderboard_id`: 唯一字符串，榜单编号，主键
- `leaderboard_name`: 榜单名称
- `metric_id`: 榜单对应的单一指标编号
- `leaderboard_scope`: 榜单范围，`track` / `overall`
- `track_variant_id`: 对应赛道变体编号；总榜可为空
- `noise_mode`: 榜单过滤的噪声模式，`clean` / `noisy`
- `batch_id`: 对应批次编号；如果是批次榜则必填
- `season_id`: 对应赛季编号；如果是赛季榜则必填
- `leaderboard_time`: 榜单生成时间

## leaderboard_entry
- `entry_id`: 唯一字符串，榜单条目编号，主键
- `leaderboard_id`: 所属榜单编号
- `model_id`: 模型编号
- `rank_no`: 排名
- `metric_value`: 该榜单指标下的直接结果值
- `metric_snapshot`: 该模型的附加指标快照，JSON 格式
- `support_result_ids`: 支撑该条目的结果编号列表，JSON 数组

## analysis_report
- `report_id`: 唯一字符串，报告编号，主键
- `report_type`: 报告类型，`model_report` / `leaderboard_report` / `bad_case_report`
- `target_id`: 报告目标编号；可对应模型、榜单或结果
- `input_refs`: 报告输入引用，JSON 格式；可包含逻辑任务、实际执行、结果、榜单、数据画像
- `report_summary`: 报告摘要
- `report_file_path`: 报告文件路径，统一位于 `${root}/runtime/reports/{report_id}.pdf`
- `generator_info`: 报告生成方式说明，如规则模板或 Agent 分析
- `report_time`: 报告生成时间

# core relationships
- `dataset_source -> dataset -> dataset_batch -> evaluation_plan -> evaluation_task -> task_run -> evaluation_result`
- `transform_spec` 与 `validation_report` 共同描述“如何生成动态考题”以及“生成结果是否可发布”
- `track_template` 定义基础赛道结构，`track_variant` 通过 `noise_mode` 形成 clean/noisy 两种版本
- `track_variant` 由 `track_kind + noise_mode + execution_constraint` 共同确定
- `evaluation_task` 是最小逻辑评测单元，`task_run` 是该任务的一次实际执行
- `resource_usage` 记录单次执行资源消耗，`evaluation_result` 聚合单个逻辑任务下的多次执行结果
- `leaderboard` 基于某个 `metric_id` 的结果聚合；赛道榜和总榜都不依赖综合分
- `analysis_report` 必须关联可追溯输入，不能只保留最终 PDF 路径

# core operations

## register dataset source
- 导入或登记原始数据源
- 记录字段结构、领域标签、存储格式与来源路径
- 生成 `dataset_source`

## generate dynamic dataset
- 基于纯生成、真实数据变换或混合策略创建 `transform_spec`
- 产出动态 `dataset`
- 提取数据特征，生成 `dataset_feature_profile`
- 执行校验，生成 `validation_report`
- 校验通过后将数据集纳入某个 `dataset_batch`

## register model
- 录入模型静态信息，生成 `model`
- 录入加载入口、依赖环境、硬件要求与资源采集策略，生成 `model_runtime_spec`

## create evaluation plan
- 选择批次、赛道变体、数据选择规则、输入长度、预测长度与指标
- 配置 `execution_repeat_count`，默认执行 `3` 次
- 生成 `evaluation_plan`

## run evaluation task
- 选择模型与运行时规格
- 为每个待测模型创建 `evaluation_task`
- 按计划在任务内部生成 `3` 个 `task_run`
- 对每个 `task_run` 执行预测并采集资源消耗
- 按逻辑任务聚合多次执行结果，产出 `evaluation_result`

## generate leaderboard
- 按批次或赛季范围读取符合条件的 `evaluation_result`
- 针对每个 `metric_id` 分别生成赛道榜和总榜
- 生成 `leaderboard` 与 `leaderboard_entry`

## generate report
- 读取结果、榜单、数据画像与 bad case
- 生成模型画像报告、榜单报告或缺陷深挖报告
- 产出 `analysis_report`
