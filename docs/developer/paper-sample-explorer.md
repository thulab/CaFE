# Paper v7 样本曲线浏览器

这是一个与评测平台前后端完全独立的只读小服务，用于浏览 `runtime/paper_exp/v7/E2_dynamic_stability` 中的合成样本和推理结果。它不修改原始 JSONL，也不依赖 npm、FastAPI 或外部 CDN。

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

可通过环境变量覆盖监听地址，也可以把 Python 服务参数放在 `start` 后面：

```bash
TSBENCHMARK_PAPER_EXPLORER_HOST=127.0.0.1 \
TSBENCHMARK_PAPER_EXPLORER_PORT=8877 \
  ./scripts/start-paper-sample-explorer.sh restart
```

服务没有登录鉴权。绑定 `0.0.0.0` 会向网络暴露实验曲线，请按实际环境配置主机防火墙或安全组。

首次启动需要顺序扫描约 20GB 产物，生成 `runtime/paper_exp/v7/E2_dynamic_stability/.sample-explorer-index.sqlite3`。该索引只保存 JSONL 行的字节偏移和选择器元数据，不复制曲线；后续启动会校验源文件大小与修改时间并直接复用。需要显式刷新时运行：

```bash
./scripts/start-paper-sample-explorer.sh foreground --rebuild-index --build-index-only
```

也可以浏览相同结构的其他实验目录：

```bash
./scripts/start-paper-sample-explorer.sh start \
  --data-dir runtime/paper_exp/v7/E2_dynamic_stability
```

## 页面交互

- `Dataset`、`Capability` 和 `Sample` 定位一个 paired group；左右方向键或样本框两侧按钮可连续翻页。
- 五张图固定对应 intensity 1–5。同一档中，历史和真实未来是参考实线，不同模型预测以颜色和线型区分。
- 每张图右上角可打开放大视图；按 `Esc`、点击遮罩或关闭按钮返回，并恢复到原放大按钮的键盘焦点。
- `Context` 在 L96 / L168 / L336 / L504 的推理视图间切换。历史和真实未来使用生成/推理时相同的 suffix re-standardization 规则。
- 多目标任务可通过 `Target` 切换通道；默认五图共享纵轴，以便直接比较强度变化，也可以关闭共享纵轴查看局部细节。
- 模型图例可逐项隐藏、全选或清空。图例数字是当前样本、当前 context 下五档 intensity 的平均 MASE，并高亮最低的真实模型；悬停曲线可看每个时间点的数值。
- `01 / SELECT` 标注当前 dataset × capability 的 oracle-context 最佳真实模型：每个模型先在每个 master sample 的四种 context 中取最低 MASE，再对五档 intensity 的全部 1,600 个样本求平均。
- `03 / INTENSITY` 标注当前样本、当前 context 下 intensity=5 的 MASE 最低真实模型。以上三个最佳模型口径均不让 naive / seasonal-naive 参与评选。
- 当前选择同步到 URL query，可以复制 URL 保留 dataset、capability、sample、context、target、缩放方式和模型显隐状态。

## 实现与数据边界

入口是 `scripts/paper_sample_explorer.py`，静态页面位于 `scripts/paper_sample_explorer/`。索引包含：

1. 88,000 个 master sample 的文件偏移；
2. 17,600 个五强度 paired group 的选择信息；
3. 每个模型、master sample、context view 对应 prediction 行的偏移。

一次页面切换只会通过 `os.pread` 读取当前 paired group 的 5 个 sample 行和 5 × 模型数个 prediction 行。SQLite 和 JSONL 均按只读方式打开；索引是可安全删除和重建的派生产物。
