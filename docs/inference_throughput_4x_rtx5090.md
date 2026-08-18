# 四卡 RTX 5090 推理吞吐调优

更新时间：2026-08-18

本文记录 CaFE 在 `timecho92` 本机访问 Timer REST Service 时的吞吐测试，
并给出按输入 token 成本动态控制 bulk 大小与 HTTP 并发的推荐配置。这里的
`view/s` 指一个原始 benchmark view 完成一次 H=48 预测；多变量 view 不按
通道重复计数，另以 `target-series/s` 报告通道吞吐。

## 测试环境

- GPU：4 × NVIDIA GeForce RTX 5090，单卡 32,607 MiB。
- 服务目录：`/data/xmy/timer-rest-service`。
- 服务管理：Supervisor，单实例监听 `127.0.0.1:10810`。
- Uvicorn workers：12。
- 客户端：`/data/xmy/CaFE`，通过 loopback MessagePack bulk API 请求。
- 预测长度：H=48。
- 服务端单个 GPU batch 输入预算：1,087,520 tokens。
- 每个 Uvicorn worker 的 L1 在途预算：4,350,080 tokens。
- `TIMER_ADMISSION_WAIT_MS=500`。修改前的默认值为 50ms；token预算未放大。

服务端 token 是输入数组元素数，而不是语言模型 token。一个同形 bulk 请求的
成本为：

\[
T_{req}=B\,[C(D+K)+HK],
\]

其中 `B` 是bulk中的view数，`C` 是模型实际可见context，`D` 是一次服务请求
中的target维数，`K` 是协变量维数。模型不支持原生多变量时，一个原始D维view
会先被拆成D个单变量child request，再按同形bulk发送。

## 测试方法

测试脚本为 `tools/benchmarking/timer_service_throughput.py`。基础吞吐扫描先在
四卡各加载一个replica；随后逐模型比较每卡1、2、4个replica，并扫描：

- context：96、512、2048、8192，以及模型声明的最大context；
- 原生target维数：1、7、21；无协变量模型额外测D=64；
- HTTP并发：2、4、8、16、32；
- H=48；支持known-future covariates的模型使用K=2；
- bulk大小同时受模型配置和`0.75 × 1,087,520`请求token目标约束。

编译敏感模型的最终case先warm up 8 个请求，使四个replica各至少收到一次该shape；
随后至少发送8个计时请求，并保证不少于一个完整并发wave。
推荐点只从零失败case中选择，并优先选择达到该shape最高稳定吞吐95%的较低并发。
原始结果位于服务器：

```text
/data/xmy/CaFE/runtime/throughput-benchmark-20260818/results.jsonl
/data/xmy/CaFE/runtime/throughput-benchmark-20260818-v3/results.jsonl
/data/xmy/CaFE/runtime/throughput-benchmark-20260818-tirex-b8-warm/results.jsonl
/data/xmy/CaFE/runtime/throughput-benchmark-20260818-v5-remaining/results.jsonl
/data/xmy/CaFE/runtime/throughput-benchmark-20260818-estimate/tirex-d1.jsonl
/data/xmy/CaFE/runtime/throughput-benchmark-20260818-estimate/toto-d1.jsonl
/data/xmy/CaFE/runtime/throughput-benchmark-20260818-true-batch/toto_fp32_high.ndjson
/data/xmy/CaFE/runtime/throughput-benchmark-20260818-true-batch/toto_bf16.ndjson
/data/xmy/CaFE/runtime/throughput-benchmark-20260818-true-batch/toto_fp32_final_smoke.ndjson
/data/xmy/CaFE/runtime/replica-benchmark-20260818/*.ndjson
```

早期 Toto 数据对应旧 adapter：REST bulk 虽然合并了传输，但 adapter 仍在
Python 中逐 item 调用 2.45B 模型，因此 bulk=4 与单条调用几乎等价。本文后面的
Toto 新数据来自修复后的真实 native batch：一个同形 bulk 只执行一次模型
`forecast`，并使用 dense MessagePack carrier 直达 worker，避免逐 item 字典解码、
stack 和重复 H2D。FlowState 中相同的逐 item forward 问题也已一并修复；其余 GPU
模型 adapter 已逐个核对，原本就保留了 native batch 或一次 pipeline batch 调用。

Timer Service 的 `max_group_rows=64` 表示单个任务的
`target variables + distinct covariates` 上限，不是bulk样本行数。CaFE已按这个
语义处理：若target本身不超过上限、只是协变量使总行数超限，则保留原生panel
并省略该模型无法接收的协变量；只有target本身超过上限时才转为模型级单变量
适配，因此不会发出必然422的原生panel请求。

扫描表列出的是用于确定调度边界的代表性零失败点，不是所有模型、shape与并发
的完整笛卡尔积。尤其Toto长上下文高维case在稳定边界已经明确后停止继续放大，
避免把大量时间消耗在已知低吞吐区域。

## 四卡推荐配置

普通模型的`maximum_request_input_tokens`为815,640。Toto 使用更保守的
131,072，因为真实 batch 后该值已经足以填满单卡，同时降低动态 shape 和超长
panel 对显存、尾延迟的压力。单个view自身超过上限时仍允许独占请求。下表是
当前代码中的四卡配置；“panel bulk”只在模型收到原生D>1请求时生效，模型级
单变量拆分仍走普通bulk。

| 模型 | 每卡replica | 基础HTTP并发 | 普通bulk | panel bulk | endpoint在途预算 | 额外成本规则 |
|---|---:|---:|---:|---:|---:|---|
| Timer-4.0 | 2 | 32 | 64 | 64 | 13,050,240 | 原生D>1按2倍token计 |
| Chronos-2 | 1 | 32 | 64 | 64 | 6,525,120 | 无 |
| timesfm2.5 | 2 | 32 | 64 | — | 6,525,120 | C>8192按2倍token计 |
| tirex2 | 2 | 32 | 64 | 8 | 6,525,120 | 原生panel通常并发2；C>512且D≥16时并发8 |
| moirai2 | 2 | 32 | 64 | — | 13,050,240 | 无 |
| Timer-3.5 | 1 | 32 | 64 | — | 6,525,120 | 无；双replica显存不足 |
| toto2.0 | 1 | 8 | 256 | 256 | 1,048,576 | request上限131,072；真实native batch |

### Replica扫描结论

Replica扫描使用四张卡、相同bulk和token预算，比较代表性的短上下文
`C=512`与长上下文；表中的吞吐为D=1的零失败case。多replica会复制模型权重，
因此只在实际吞吐改善足以覆盖显存、加载时间和长序列退化时采用。CaFE按模型
依次加载，选中的replica在该模型处理完所有数据集前保持驻留，不会按数据集反复
加载。

| 模型 | 1 replica短序列 | 候选多replica短序列 | 1 replica长序列 | 候选多replica长序列 | 采用 | 每卡显存MiB | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Timer-4.0 | 3,853 | 4,698（×2） | 454 | 371（×2） | 2 | 7,867 | short提升22%，长序列交给动态限流 |
| Chronos-2 | 4,237 | 3,901（×2） | 276 | 217（×2） | 1 | 5,486 | 双replica全形状退化 |
| timesfm2.5 | 2,531 | 3,468（×2） | 209 | 237（×2） | 2 | 10,791 | 短、长均提升 |
| tirex2 | 416 | 615（×2） | 428 | 560（×2） | 2 | 2,583 | D=1提升，panel仍由低并发保护 |
| moirai2 | 13,964 | 17,442（×2） | 3,047 | 2,896（×2） | 2 | 5,431 | short提升25%，长序列小幅退化 |
| Timer-3.5 | 1,938 | OOM（×2） | 687 | OOM（×2） | 1 | 21,696 | 单replica已接近单卡容量上限 |
| toto2.0 | 1,961 | 1,458（×2） | 140 | 134（×2） | 1 | 11,326 | 双replica复制大权重且更慢 |

Timer-4.0与Moirai2的选择针对本轮GIFT-Eval short-term工作负载；如果单独运行
medium/long实验，建议重新固定每卡1个replica的配置，而不是把short-term拓扑
视为所有长度的通用最优值。每卡4个replica只在少数短序列case继续小幅改善，
却明显增加加载时间、显存占用和长序列退化，因此未进入默认配置。

### 七模型端到端核验

使用`gift_restaurant_d`的8个官方实例运行generation、research validation、
inference和analysis四阶段，7个模型各产生248条预测，全部
`status=complete`且`failure_count=0`。manifest记录的实际worker数为：

| 模型 | 实际worker | 加载秒数 | 推理调用总秒数 |
|---|---:|---:|---:|
| Timer-4.0 | 8 | 27.17 | 31.30 |
| Chronos-2 | 4 | 9.03 | 11.53 |
| timesfm2.5 | 8 | 22.12 | 26.02 |
| tirex2 | 8 | 19.07 | 36.95 |
| moirai2 | 8 | 17.11 | 20.79 |
| Timer-3.5 | 4 | 28.15 | 31.44 |
| toto2.0 | 4 | 35.24 | 37.90 |

完整四阶段约199秒。该小实验主要用于验证加载拓扑、路由、预测落盘和分析闭环；
模型加载占比很高，不能用其端到端平均值外推全量数据吞吐。服务器产物位于：

```text
/data/xmy/CaFE/runtime/replica-topology-e2e-20260818/
```

### 单变量吞吐

每行选择零失败case中达到该shape稳定峰值95%的最低并发。TiRex只列已完成
warm-up的L96；其最大context为2048，长panel结果见下一表。

| 模型 | C | bulk | 推荐并发 | views/s | P95/s |
|---|---:|---:|---:|---:|---:|
| Timer-4.0 | 96 | 64 | 8 | 3,768.3 | 0.198 |
| Timer-4.0 | 2,048 | 64 | 8 | 1,391.8 | 0.494 |
| Timer-4.0 | 8,192 | 33 | 8 | 545.8 | 0.717 |
| Chronos-2 | 96 | 64 | 32 | 5,900.3 | 0.454 |
| Chronos-2 | 2,048 | 64 | 16 | 1,058.5 | 1.190 |
| Chronos-2 | 8,192 | 33 | 8 | 227.3 | 1.402 |
| timesfm2.5 | 96 | 64 | 32 | 2,601.5 | 0.733 |
| timesfm2.5 | 2,048 | 64 | 8 | 1,188.1 | 0.428 |
| timesfm2.5 | 15,360 | 17 | 4 | 57.1 | 1.435 |
| tirex2 | 96 | 64 | 32 | 1,466.4 | 1.289 |
| tirex2 | 512 | 64 | 8 | 1,636.4 | 0.463 |
| tirex2 | 2,048 | 64 | 16 | 1,426.6 | 0.879 |
| moirai2 | 96 | 64 | 32 | 12,221.7 | 0.147 |
| moirai2 | 2,048 | 64 | 32 | 10,482.9 | 0.174 |
| moirai2 | 16,384 | 49 | 16 | 866.4 | 0.790 |
| Timer-3.5 | 96 | 64 | 8 | 2,222.2 | 0.228 |
| Timer-3.5 | 2,048 | 64 | 32 | 1,424.3 | 1.263 |
| Timer-3.5 | 11,520 | 64 | 8 | 489.0 | 1.036 |
| toto2.0 | 512 | 256 | 8 | 2,874.5 | 0.707 |
| toto2.0 | 8,192 | 16 | 4 | 139.8 | — |
| toto2.0 | 16,384 | 8 | 8 | 99.2 | 0.638 |

### 原生多变量吞吐

| 模型 | C | D | bulk | 推荐并发 | views/s | target-series/s | P95/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Timer-4.0 | 96 | 7 | 64 | 32 | 2,784.2 | 19,489.1 | 0.948 |
| Timer-4.0 | 96 | 21 | 64 | 16 | 1,083.9 | 22,761.9 | 1.156 |
| Timer-4.0 | 8,192 | 21 | 4 | 8 | 76.1 | 1,598.1 | 0.611 |
| Chronos-2 | 96 | 7 | 64 | 32 | 2,618.1 | 18,326.7 | 1.042 |
| Chronos-2 | 96 | 21 | 64 | 16 | 1,294.2 | 27,178.2 | 0.965 |
| Chronos-2 | 2,048 | 21 | 17 | 4 | 111.9 | 2,349.9 | 0.640 |
| tirex2 | 96 | 7 | 8 | 2 | 84.2 | 589.4 | 0.276 |
| tirex2 | 96 | 21 | 8 | 2 | 55.4 | 1,163.4 | 0.362 |
| tirex2 | 2,048 | 21 | 8 | 8 | 42.4 | 889.5 | 1.449 |
| toto2.0 | 512 | 7 | 36 | 8 | 296.3 | 2,073.8 | 0.864 |
| toto2.0 | 512 | 21 | 12 | 8 | 99.4 | 2,086.4 | 0.863 |
| toto2.0 | 8,192 | 7 | 2 | 8 | 20.2 | 141.4 | — |
| toto2.0 | 8,192 | 21 | 1 | 4 | 9.8 | 206.6 | — |
| toto2.0 | 16,384 | 7 | 1 | 8 | 14.8 | 103.4 | 0.541 |
| toto2.0 | 16,384 | 21 | 1 | 8 | 3.3 | 70.2 | 2.391 |

同一shape的较高并发若出现503，不进入上述推荐点。压测工具自身不重试，因而
这些503可直接用来定位准入边界；正式CaFE请求会退避重试。单wave会有一定路由
抖动，配置采用跨shape的平滑规则，而不是逐格硬编码所有局部峰值。

## CaFE动态策略

CaFE使用两层成本约束：

1. 单请求限制：每个bulk不超过推荐的`maximum_request_input_tokens`；仅当单个
   benchmark view本身已超过该值时，允许它单独发送。
2. Endpoint限制：worker数仍由模型的基础HTTP并发控制，但实际请求还必须取得
   加权token额度，使该endpoint当前全部请求满足
   `sum(weight(T_req, C, D)) <= client_inflight_input_tokens`。权重只用于客户端
   调度；服务端仍按原始元素数准入。单个超大请求在endpoint空闲时可独占执行。
3. Shape限制：TiRex原生panel使用更小的bulk和并发；TimesFM对实测进入饱和区
   的超长context提高成本权重。Toto用较小的统一request token上限自然缩小长
   context bulk。单变量拆分不会误用原生panel权重。

因此，短小请求仍可使用模型的最高HTTP并发；长context、高维panel或大量
univariate children会自动降低有效并发。429/503采用1s、2s指数退避并加入
0–25%随机抖动，随后重新请求，不把瞬时准入拒绝直接记为模型失败。

服务端保留L1/L2/L3 backpressure。当前只把准入等待从50ms增加到500ms，未直接
扩大token预算：直接增大L1预算不会增加GPU算力，只会把更多排队压力转移到
ZMQ、GPU backlog和主机内存。

## Toto 优化与精度结论

本轮按 P0–P5 顺序处理了吞吐链路：

1. P0 验证了同卡双 replica 可把旧 adapter 的短序列吞吐从约45提升到约82
   views/s，但会复制2.45B权重，属于临时规避；真实 batch 上线后恢复每卡单
   replica。
2. P1 将整个 REST bulk 合并为一次 Toto native forecast，再按原始 item 边界拆
   回输出。最终FP32复测中C=512,D=1从旧链路约45提升到2,875 views/s，约64倍。
3. P2 让 bulk ndarray 通过一个 dense MessagePack carrier 传到 worker，并整批
   H2D，避免每行Python对象与重复device copy。
4. P3 正式配置保持FP32参数，启用`torch.set_float32_matmul_precision("high")`
   使用RTX 5090的TF32矩阵核。20个真实GIFT样本的BF16对FP32预测相对MAE平均
   0.370%、P95 0.862%、最大1.020%，所以`TIMER_TOTO_DTYPE=bfloat16`只保留为
   显式吞吐选项，不进入默认正式实验。TF32相对`highest`的固定输入相对预测
   MAE约0.02%–0.15%。
5. P4 CaFE按输入元素数动态决定bulk和endpoint在途额度；Toto当前使用
   request 131,072、endpoint 1,048,576、HTTP并发8。预测结果按返回bulk成批写
   Parquet，避免逐forecast Python float装箱。
6. P5 审计了所有GPU模型adapter；FlowState存在同型逐item forward并已改成一次
   batch forward。Timer-4.0、Chronos-2、Timer-3.5、Timer-3.0、Moirai2、
   TimesFM2.5、TiRex2原先已经走native batch。`torch.compile`/CUDA graph没有纳入
   正式配置：动态context、target维数和decode shape会扩大编译缓存，而真实batch
   已消除了主要瓶颈。

BF16吞吐作为上限参考：C=512,D=1在并发8时约3,315 views/s，C=8192,D=1约
247 views/s，C=16384,D=1约177 views/s。正式实验采用FP32+TF32，避免把精度模式
变化混入模型能力比较。
