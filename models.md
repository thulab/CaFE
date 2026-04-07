- 测试工具的根目录为 ${root}

# architecture
- model_management: 模型管理器

# core models

## dataset
- dataset_id: 唯一字符串，数据集编号，主键
- dataset_name: 数据集名称
- dataset_path: 数据集存储位置，如果是生成的数据集则直接写入到${root}/runtime/datasets/{dataset_id}.csv；如果是直接上传的文件，则将上传的文件存储到dataset存储文件夹下的${root}/runtime/datasets/{dataset_id}.{format}
- dataset_time: 数据集生成/加载时间
- column_names: 列名，如果有多列则使用分号分隔开
- target_column_name: 目标列名

## model
- model_id: 唯一字符串，模型编号，主键
- model_name: 模型名称，自定义
- model_alias_name: 模型别名，自定义
- model_create_time: 模型创建时间
- model_update_time: 模型更新时间
- model_url: 模型的 hugging face 的 URL
- model_path: 模型相关文件的相对存储路径，不放绝对路径避免泄露存储位置，文件夹路径为${root}/runtime/models/{model_id}
- model_params: 模型所需的参数的名称、类型与取值范围，内部为json格式，每个参数需要包含参数名称、参数类型、默认值与取值范围
- model_environment: 安装本模型时需要同步安装的python环境内容

## metric
- metric_id: 唯一字符串，指标编号，主键
- metric_name: 指标名称
- metric_type: 指标数值类型

## task_type
- task_type_id: 唯一字符串，任务类型编号，主键
- task_type_name: 任务类型名称
- task_metric_ids: 任务关注的性能指标项目，json数组格式

## task
- task_id: 唯一字符串，任务编号，主键
- task_time: 任务生成时间
- task_type_id: 任务类型编号
- task_dataset_id: 任务使用的数据集编号
- task_model_id: 任务使用的模型编号
- task_model_params: 任务使用的模型的实际参数
- task_result: 测试结果，包括每个指标与起对应的值

## ranks
- rank_id: 唯一字符串，榜单编号，主键
- rank_metric_ids: 榜单的指标编号
- rank_task_type_id: 榜单的任务类型编号，总榜是一种特殊的榜单
- rank_time: 榜单生成时间
- list: 榜单排名，任务 task_id 列表

## reports
- report_id: 唯一字符串，报告编号，主键
- report_time: 报告生成时间
- report_file_path: 报告位置，位置为${root}/runtime/reports/{report_id}.pdf

# core operations

## create dataset
- 选择以哪种方式创建数据集
  - 直接生成数据集：生成方法，给出输入列名，输出列名等参数
  - 读取加载已有数据集：从已有的CSV文件中读取数据，要求给出必要参数
- 完成生成或加载，生成 dataset 实体

## create model
- 输入参数生成模型

## load model
- 触发加载模型
- 加载权重文件

## create task
- 选择数据集
- 选择模型
- 选择任务类型
- 创建任务
- 构建运行环境
- 执行模型预测
- 计算预测结果

## generate rank
- 选择 metric 指标
- 如果是单个任务类型的榜单
  - 直接 list 排序
- 如果是总榜
  - 将四个榜单的排名加和到一起后按照总排名排序

## generate report
- 根据榜单结果生成不同模型在不同维度的对比报告