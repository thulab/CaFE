# TSBenchmark 数据模型

## 核心模型

### SeriesSample（单条时序样本）

```python
sample_id: str                           # 样本唯一标识
history: list[float]                    # 输入上下文序列
target: list[float]                     # 预测目标序列
covariates: dict[str, list[float]]      # 协变量
input_channel_values: dict[str, list[float]]   # 输入通道值
target_channel_values: dict[str, list[float]]   # 目标通道值
future_known_channel_values: dict[str, list[float]]  # 未来已知通道
channel_layout: ChannelLayout           # 通道布局
track_tags: list[str]                   # 赛道标签
truth: SeriesTruth                      # Ground Truth
notes: dict[str, Any]                   # 附加信息
```

### SeriesTruth（样本真值）

```python
trend_type: str              # 趋势类型
periods: list[int]          # 周期列表
dominant_period: int        # 主周期
amplitude_mode: str         # 振幅模式
phase_shift: bool           # 相位偏移
noise_level: float          # 噪声级别
difficulty: str             # 难度等级
```

### DatasetBatch（数据集批次）

```python
batch_id: str
track: TrackKind                              # FORECAST_ACCURACY 等
track_variant_id: str
track_template_kind: TrackTemplateKind       # UNIVARIATE_FORECAST 等
noise_mode: NoiseMode
execution_constraint: ExecutionConstraint
input_channels: list[str]
target_channels: list[str]
future_known_channels: list[str]
policy: str
seed: int
source_type: DatasetSourceType               # SYNTHETIC / CSV / TSFILE
source_id: str
dataset_id: str
created_at: datetime
sample_count: int
input_length: int
prediction_length: int
context_length: int
horizon: int
samples: list[SeriesSample]                  # 完整样本数据
validation: ValidationReport
feature_profile: DatasetFeatureProfile
```

### DatasetFeatureProfile（特征画像）

```python
trend_tags: list[str]
seasonality_tags: list[str]
dominant_periods: list[int]
noise_level: float
missing_rate: float
outlier_rate: float
feature_summary: dict[str, Any]   # sample_count, difficulty_levels, channels
```

### ValidationReport（校验报告）

```python
passed: bool
issues: list[str]
```

## V1 Benchmark 模型

### V1 赛道（TrackKind）

```python
FORECAST_ACCURACY        # 预测精度（主赛道）
COVARIATE_ROBUSTNESS     # 协变量鲁棒性
NOISE_ROBUSTNESS         # 噪声鲁棒性
COST_INTENSIVE           # 成本密集型
```

### V1 诊断家族（5类）

| Family | 主测能力 |
|--------|---------|
| `trend` | 趋势外推与斜率保持 |
| `multi_seasonal` | 多周期组合与相位鲁棒性 |
| `regime_switching` | 分布突变与转移适应 |
| `long_memory_nonlinear` | 长依赖与非线性动力学 |
| `intermittent_heteroskedastic` | 稀疏需求与噪声鲁棒性 |

### V1 结构特征（8维）

```python
trend_strength       # 趋势强度
seasonal_strength   # 季节性强度
spectral_entropy    # 频谱熵/可预测性
acf_half_life       # 自相关半衰期
changepoint_density # 变点密度
variance_shift      # 方差漂移
intermittency       # 间歇性
outlier_rate        # 异常值率
```

### V1 生成参数

```python
horizon_ratio in {0.25, 0.5, 1.0}
H = clip(round(r * dominant_scale), 12, 96)    # 预测长度
context = min(8H, 512)                          # 上下文长度
burn_in = max(4 * dominant_scale, 200)          # 预热长度
```

### V1 数据配额

```
Anchor Track: 2000 条
Diagnostic Track: 5 family × 5 difficulty × 3 horizon_ratio × 100 = 7500 条
总计: 9500 条
```

### V1 评测指标

| 指标 | 说明 |
|------|------|
| MASE | 主指标 - Mean Absolute Scaled Error |
| sMAPE | 辅助指标 - Symmetric MAPE |
| Relative Skill | `1 - MASE_model / MASE_baseline` |

### V1 可用模型

**基线模型**：
- `last_value` - 简单基线
- `seasonal_naive` - 季节性朴素预测
- `auto_theta` - Theta 方法
- `ridge_ar` - 岭回归自回归

**Foundation Models**：
- `timesfm_2_5_200m`
- `chronos_bolt_base`
- `sundial_base_128m`
- `moirai_moe_base`
- `lag_llama`

## 枚举类型

### DatasetSourceType
```python
SYNTHETIC   # 合成数据
CSV          # CSV 文件导入
TSFILE       # TSFile 格式
```

### NoiseMode
```python
CLEAN        # 干净数据
NOISY        # 带噪声数据
```

### TrackTemplateKind
```python
UNIVARIATE_FORECAST
MULTIVARIATE_FORECAST_ALL_TO_ALL
MULTIVARIATE_FORECAST_ALL_TO_SUBSET
MULTIVARIATE_FORECAST_WITH_FUTURE_COVARIATES
MULTIVARIATE_FORECAST_VIA_UNIVARIATE
```
