# Paper v8 样本曲线浏览器

这是一个与评测平台前后端完全独立的只读小服务，用于浏览 Paper v8 的合成任务视图和推理结果。它不修改原始 JSONL，也不依赖 npm、FastAPI 或外部 CDN。默认从 `runtime/paper_exp/v8` 中自动选择最新一个含分析清单的完整实验；也可以显式指定实验目录。

## 启动

在仓库根目录运行：

```bash
./scripts/start-paper-sample-explorer.sh start
```

默认在后台监听 `0.0.0.0:8766`。本机访问 `http://127.0.0.1:8766/`，其他机器访问 `http://<服务器 IP>:8766/`。管理命令：

```bash
./scripts/start-paper-sample-explorer.sh status   # 查看进程与就绪状态
./scripts/start-paper-sample-explorer.sh logs     # 查看最近 80 行日志
./scripts/start-paper-sample-explorer.sh restart  # 后台重启
./scripts/start-paper-sample-explorer.sh stop     # 关闭服务
```

PID 与日志分别保存在 `.tsbenchmark-system/paper-sample-explorer.pid` 和 `.tsbenchmark-system/paper-sample-explorer.log`。后台进程会脱离当前终端，退出 SSH 会话后仍会运行。如果需要前台运行，可用 `./scripts/start-paper-sample-explorer.sh foreground`，按 `Ctrl+C` 停止。

显式选择某次 v8 实验：

```bash
./scripts/start-paper-sample-explorer.sh restart \
  --data-dir runtime/paper_exp/v8/<experiment_id>
```

可通过环境变量覆盖监听地址：

```bash
TSBENCHMARK_PAPER_EXPLORER_HOST=127.0.0.1 \
TSBENCHMARK_PAPER_EXPLORER_PORT=8877 \
  ./scripts/start-paper-sample-explorer.sh restart
```

服务没有登录鉴权。绑定 `0.0.0.0` 会向网络暴露实验曲线，请按实际环境配置主机防火墙或安全组。

## 索引

首次启动会顺序扫描各数据集的 `forecast_views.jsonl` 和每个模型的预测 JSONL，在实验根目录生成 `.sample-explorer-index.sqlite3`。索引只保存 JSONL 行的字节偏移和选择器元数据，不复制曲线；后续启动会校验所有输入文件的路径、大小和修改时间并直接复用。

需要显式刷新时运行：

```bash
./scripts/start-paper-sample-explorer.sh foreground \
  --data-dir runtime/paper_exp/v8/<experiment_id> \
  --rebuild-index --build-index-only
```

## 页面口径

- `Dataset`、`Capability` 和 `Sample` 定位一个 seed group；左右方向键或样本框两侧按钮可连续翻页。
- 页面只纳入 v8 主表的 `evaluation_table=main`、`generator_family_role=primary` 样本，不混入 secondary family、robustness、ablation 或 gate audit 任务。
- 五张图固定对应 intensity 1–5。同一档中，历史和真实未来是参考实线，不同模型预测以颜色和线型区分。
- `covariate_response` 等存在反事实成员的能力，会把同一 seed 下的每个 member 显示成独立 group，避免把两个成员错误合并为十条样本。
- `Context` 在 L96 / L168 / L336 / L504 的实际推理任务视图间切换。v8 的每个任务已经带有该 context 对应的标准化 `target`，浏览器直接读取它，不再自行重做 v7 的 suffix re-standardization。
- 多目标任务可通过 `Target` 切换通道；默认五图共享纵轴，以便直接比较强度变化，也可以关闭共享纵轴查看局部细节。
- 模型图例可逐项隐藏、全选或清空。图例数字是当前 seed group、当前 context 下五档 intensity 的平均 MASE；单样本 MAE/MASE 由任务中的 `mase_scale` 与预测重新计算。
- `01 / SELECT` 标注当前 dataset × capability 在分析产物中的 oracle-context、main-primary 平均 MASE 最佳真实模型。
- `03 / INTENSITY` 标注当前 seed group、当前 context 下 intensity=5 的 MASE 最低模型。
- 当前选择同步到 URL query，可以复制 URL 保留 dataset、capability、sample、context、target、缩放方式和模型显隐状态。

## 实现与兼容边界

入口是 `scripts/paper_sample_explorer.py`，静态页面位于 `scripts/paper_sample_explorer/`。v8 索引包含：

1. 各数据集主表 primary master sample 及四个 context task view 的文件偏移；
2. 每个能力按 seed 和可选 counterfactual member 划分的五强度 group；
3. 每个 dataset × model × task view 对应预测行的文件偏移。

一次页面切换只会通过 `os.pread` 读取当前 group 的 5 个 task 行和 5 × 模型数个 prediction 行。SQLite 和 JSONL 均按只读方式打开；索引是可安全删除和重建的派生产物。

查看器仍兼容提交 `5fd6a1a` 使用的 v7 `E2_dynamic_stability` 目录。需要时显式传入含 `samples.jsonl` 的旧目录即可；默认入口和页面口径以 v8 为准。
