# 协作 Learnings（与 SpriCoder）

> 下次合作从这里开始：以下是已确认的偏好与约束，无需重新教。

## 用户偏好
- **重视"真跑通/验证"胜过"声称完成"**。被问"你学会了嘛"时，期望的是：演示理解 + 诚实标注边界——哪些是文档原文、哪些是我自己推断的、哪些还没实跑过。下次交付知识或方案，主动区分「读懂文档」与「已验证」，不要笼统说"会了"。
- **倾向把外部资料纳入可重复的工程化流程，而非一次性手工落地**。接入新飞书文档时，认可"扩展 `sources.json` + 同步脚本、使其可一键重新同步"的做法，而不是手动贴一份 Markdown。

## 最佳实践（从返工中沉淀）
- **接外部文档前先看 URL 域名是否与现有默认 host 一致**。本次 TsFile 手册来自 `apache-iotdb-project.feishu.cn`，与 `sources.json` 默认的 `timechor.feishu.cn` 不是同一租户；原脚本只支持全局 `host`，需加「按文档 `host` 覆盖」（`host = doc.get("host", default_host)`）。下次遇到跨租户文档同理。
- **同步后用 `--check` 自验收尾**。生成 `.md` 后跑 `python3 scripts/sync-feishu-docs.py --check`，确认本地与飞书原文一致（退出码 1 = 过期）。把"读懂文档"推进到"流程跑通"。

## 项目关键约束与坑
- **飞书文档同步体系**：原文在飞书 → `scripts/sync-feishu-docs.py` 拉取转 Markdown 落到 `docs/reference/`；同步清单在 `docs/reference/sources.json`（每篇含 `token`/`title`/`output`/`note`，可选 `host` 覆盖默认租户）。生成的 `.md` 头部标注"自动生成请勿手工编辑"——**要改内容请改飞书原文再重跑脚本，别直接编辑 `.md` 正文**。
- **工具是 `lark-cli`（不是 `lark`）**，经 node 安装（`~/.nvm/.../bin/lark-cli`）。需 `docx:document:readonly` scope：`lark-cli auth status` 查授权，缺权限时 `lark-cli auth login --scope "docx:document:readonly"`。
- **拉取命令需带 `LARK_CLI_NO_PROXY=1`**：`LARK_CLI_NO_PROXY=1 lark-cli docs +fetch --api-version v2 --doc <url> --format pretty`。
- **项目正在接入 TsFile**，用 `TsFileDataFrame` Python API。要点已同步到 `docs/reference/tsfile-dataframe-manual.md`：
  - `TsFileDataFrame` 是**懒加载视图，不是 pandas.DataFrame**（初始化只扫元数据，按行号切片才触发 I/O）。
  - 表模型路径 `表名.标签.物理量` 与树模型 `root.设备段.物理量` **不可在同一目录/文件列表混用**（混了加载阶段抛异常）。
  - 只有 `list_timeseries_metadata()` 返回真 `pandas.DataFrame`（且是**元数据**）；数值读取（`ts[a:b]`、`AlignedTimeseries.values`）返回 `np.ndarray`。
- **项目用 `AGENTS.md` 作为 agent 指南**（仓库根有 `AGENTS.md`），根目录暂无 `CLAUDE.md` → 本 `LEARNINGS.md` 当前不会被自动加载（见文末待办）。

---
### 更新记录
- 2026-05-25：首次创建。沉淀飞书文档同步体系（lark-cli 用法、跨租户 host 覆盖、勿手工编辑生成的 .md）、TsFile/TsFileDataFrame 接入要点，以及"重视验证、诚实标注推断边界"的用户偏好。
