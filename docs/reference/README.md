# 参考文档（外部同步）

本目录收录从飞书同步过来的**核心调用操作**与**数据结构服务**参考资料。
内容以飞书原文为准，本目录下的 `.md` 文件由同步脚本自动生成，**请勿手工编辑正文**。

## 文档清单

| 文件 | 主题 | 飞书原文 |
| --- | --- | --- |
| [`rest-api.md`](./rest-api.md) | 推理服务 REST API（推理 / 模型管理 / 数据集评估治理 / 健康探针 / 监控）——核心调用操作 | [docx](https://timechor.feishu.cn/docx/CwaMdyCmhovwIGxRbGicKuExnVh) |
| [`chronos-dump-report.md`](./chronos-dump-report.md) | Chronos 数据集 Arrow → TsFile 转储清单与字段映射——数据结构服务 | [docx](https://timechor.feishu.cn/docx/Dd2WdEvXKoLDdox5Ma9cEMb1nBf) |
| [`tsfile-dataframe-manual.md`](./tsfile-dataframe-manual.md) | TsFileDataFrame 用户手册（像 DataFrame 一样读取 TsFile：表/树模型、Timeseries 懒加载、`.loc` 对齐查询）——TsFile 接入 | [docx](https://apache-iotdb-project.feishu.cn/docx/SenJdxlbuoUS5Uxmq7jcOUzdnob) |

## 更新方案

同步清单与脚本：

- 清单：[`sources.json`](./sources.json) —— 每篇文档的飞书 `token`、标题、输出文件名。
- 脚本：[`../../scripts/sync-feishu-docs.py`](../../scripts/sync-feishu-docs.py) —— 仅依赖 Python 3 标准库与 `lark-cli`。

### 1. 准备 lark-cli 授权（仅首次）

脚本通过 [`lark-cli`](https://github.com/larksuite/cli) 读取飞书文档，需要 `docx:document:readonly` 权限：

```bash
lark-cli auth status                                   # 查看当前授权
lark-cli auth login --scope "docx:document:readonly"   # 缺权限时执行，按提示在浏览器完成授权
```

### 2. 同步（飞书原文有更新时执行）

```bash
python3 scripts/sync-feishu-docs.py
```

脚本会重新拉取清单内每篇文档，转换为 Markdown 并覆盖写入对应文件（含来源链接与同步日期的头部）。

### 3. 校验是否过期（可选，适合 CI / 提交前）

```bash
python3 scripts/sync-feishu-docs.py --check
```

只比对、不写入；若本地文件与飞书原文正文不一致，会打印 diff 并以退出码 `1` 结束（忽略“最后同步”日期行）。

### 新增 / 移除文档

编辑 [`sources.json`](./sources.json) 的 `documents` 数组（增删 `token` / `output` 条目），再运行同步脚本即可。

文档默认使用顶层 `host`（`timechor.feishu.cn`）；若某篇来自其它租户，在该条目里加 `host` 字段单独覆盖，例如 TsFile 手册来自 `apache-iotdb-project.feishu.cn`。
