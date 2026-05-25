#!/usr/bin/env python3
"""把飞书（Lark）docx 文档同步到 docs/reference/，转换为 Markdown。

更新方案
--------
1. 在飞书里修改原文，或在 docs/reference/sources.json 增删条目。
2. 运行：python3 scripts/sync-feishu-docs.py
   每篇文档会被重新拉取并覆盖写入对应的 .md 文件。

依赖
----
- lark-cli（已登录且授予 docx:document:readonly 权限）
  首次缺权限时执行：lark-cli auth login --scope "docx:document:readonly"
- Python 3 标准库（无第三方依赖）

可选参数
--------
  --check   只拉取并转换，对比现有文件是否有差异，不写入（用于 CI / 校验是否过期）；
            有差异时以退出码 1 结束。
"""
from __future__ import annotations

import datetime
import difflib
import json
import os
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "docs" / "reference"
SOURCES = REF_DIR / "sources.json"

VOID = {"br", "col", "hr", "img", "input", "meta", "link"}


# --------------------------------------------------------------------------- #
# HTML → 轻量 DOM
# --------------------------------------------------------------------------- #
class Node:
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag, attrs=None, parent=None):
        self.tag = tag
        self.attrs = dict(attrs or {})
        self.children = []
        self.parent = parent


class TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.cur)
        self.cur.children.append(node)
        if tag not in VOID:
            self.cur = node

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, attrs, self.cur))

    def handle_endtag(self, tag):
        node = self.cur
        while node is not None and node.tag != tag:
            node = node.parent
        if node is not None and node.parent is not None:
            self.cur = node.parent

    def handle_data(self, data):
        if data:
            self.cur.children.append(data)


# --------------------------------------------------------------------------- #
# DOM → Markdown
# --------------------------------------------------------------------------- #
def text_of(node) -> str:
    if isinstance(node, str):
        return node
    return "".join(text_of(c) for c in node.children)


def render_inline(node) -> str:
    if isinstance(node, str):
        return node.replace("\n", " ")
    inner = "".join(render_inline(c) for c in node.children)
    tag = node.tag
    if tag in ("b", "strong"):
        s = inner.strip()
        return f"**{s}**" if s else ""
    if tag in ("i", "em"):
        s = inner.strip()
        return f"*{s}*" if s else ""
    if tag == "code":
        return f"`{text_of(node)}`"
    if tag == "br":
        return " "
    return inner


def find_child(node, tag):
    for c in node.children:
        if not isinstance(c, str) and c.tag == tag:
            return c
    return None


def render_pre(node) -> str:
    code = find_child(node, "code") or node
    lang = node.attrs.get("lang", "") or ""
    parts = []

    def walk(n):
        for c in n.children:
            if isinstance(c, str):
                parts.append(c)
            elif c.tag == "br":
                parts.append("\n")
            else:
                walk(c)

    walk(code)
    return f"```{lang}\n{''.join(parts)}\n```"


def render_list(node, kind, indent=0) -> str:
    lines = []
    idx = 1
    for li in node.children:
        if isinstance(li, str) or li.tag != "li":
            continue
        inline_parts, nested = [], []
        for c in li.children:
            if not isinstance(c, str) and c.tag in ("ul", "ol"):
                nested.append(c)
            elif isinstance(c, str):
                inline_parts.append(c)
            else:
                inline_parts.append(render_inline(c))
        bullet = f"{idx}." if kind == "ol" else "-"
        lines.append("  " * indent + f"{bullet} {''.join(inline_parts).strip()}")
        for nst in nested:
            lines.append(render_list(nst, nst.tag, indent + 1))
        idx += 1
    return "\n".join(lines)


def render_cell(cell) -> str:
    parts = []
    for c in cell.children:
        if isinstance(c, str):
            s = c.strip()
        else:
            s = render_inline(c).strip()
        if s:
            parts.append(s)
    return "<br>".join(parts).replace("|", "\\|")


def render_table(node) -> str:
    def cells_of(tr):
        return [
            render_cell(c)
            for c in tr.children
            if not isinstance(c, str) and c.tag in ("td", "th")
        ]

    header, body_trs = [], []
    thead, tbody = find_child(node, "thead"), find_child(node, "tbody")
    if thead:
        trs = [c for c in thead.children if not isinstance(c, str) and c.tag == "tr"]
        if trs:
            header = cells_of(trs[0])
    if tbody:
        body_trs = [c for c in tbody.children if not isinstance(c, str) and c.tag == "tr"]
    else:
        all_trs = [c for c in node.children if not isinstance(c, str) and c.tag == "tr"]
        if not header and all_trs:
            header, body_trs = cells_of(all_trs[0]), all_trs[1:]
        else:
            body_trs = all_trs

    rows = [cells_of(tr) for tr in body_trs]
    ncol = max([len(header)] + [len(r) for r in rows] + [0])
    if ncol == 0:
        return ""
    if not header:
        header = [""] * ncol

    def fmt(row):
        row = row + [""] * (ncol - len(row))
        return "| " + " | ".join(row) + " |"

    out = [fmt(header), "| " + " | ".join(["---"] * ncol) + " |"]
    out += [fmt(r) for r in rows]
    return "\n".join(out)


def render_block(node) -> str:
    blocks = []
    for c in node.children:
        if isinstance(c, str):
            s = c.strip()
            if s:
                blocks.append(s)
            continue
        tag = c.tag
        if tag == "title":
            txt = text_of(c).strip()
            if txt:
                blocks.append(f"# {txt}")
        elif re.fullmatch(r"h[1-6]", tag):
            blocks.append(f"{'#' * int(tag[1])} {render_inline(c).strip()}")
        elif tag == "p":
            txt = render_inline(c).strip()
            if txt:
                blocks.append(txt)
        elif tag == "hr":
            blocks.append("---")
        elif tag == "pre":
            blocks.append(render_pre(c))
        elif tag in ("ul", "ol"):
            blocks.append(render_list(c, tag))
        elif tag == "table":
            blocks.append(render_table(c))
        elif tag == "blockquote":
            inner = render_block(c).strip()
            blocks.append("\n".join((f"> {ln}" if ln else ">") for ln in inner.split("\n")))
        elif tag in ("div", "section", "article", "body", "html"):
            inner = render_block(c).strip()
            if inner:
                blocks.append(inner)
        else:
            txt = render_inline(c).strip()
            if txt:
                blocks.append(txt)
    return "\n\n".join(blocks)


def dedup_headings(md: str) -> str:
    """折叠相邻且完全相同的标题行（处理 <title> 与首个 <h1> 重复的情况）。"""
    lines = md.split("\n")
    out = []
    prev_heading = None
    for ln in lines:
        if ln.startswith("#"):
            if ln == prev_heading:
                continue
            prev_heading = ln
        elif ln.strip():
            prev_heading = None
        out.append(ln)
    return "\n".join(out)


def html_to_markdown(html: str) -> str:
    builder = TreeBuilder()
    builder.feed(html)
    md = render_block(builder.root)
    md = dedup_headings(md)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


# --------------------------------------------------------------------------- #
# 拉取
# --------------------------------------------------------------------------- #
def fetch_html(url: str) -> str:
    env = dict(os.environ, LARK_CLI_NO_PROXY="1")
    proc = subprocess.run(
        ["lark-cli", "docs", "+fetch", "--api-version", "v2", "--doc", url, "--format", "pretty"],
        capture_output=True,
        text=True,
        env=env,
    )
    out = proc.stdout.strip()
    if proc.returncode != 0 or out.startswith("{") and '"ok": false' in out:
        sys.exit(f"[sync] lark-cli 拉取失败：{url}\n{out or proc.stderr}")
    return out


def build_header(url: str, token: str, today: str) -> str:
    return (
        "<!-- 本文件由 scripts/sync-feishu-docs.py 自动生成，请勿手工编辑。 -->\n"
        "<!-- 内容更新：修改飞书原文后重新运行同步脚本。 -->\n\n"
        f"> **来源**：[飞书文档]({url})（docx token `{token}`）  \n"
        f"> **最后同步**：{today}  \n"
        "> **更新方式**：`python3 scripts/sync-feishu-docs.py`\n\n"
        "---\n\n"
    )


def main() -> int:
    check = "--check" in sys.argv[1:]
    cfg = json.loads(SOURCES.read_text(encoding="utf-8"))
    default_host = cfg.get("host", "timechor.feishu.cn")
    today = datetime.date.today().isoformat()

    stale = []
    for doc in cfg["documents"]:
        token = doc["token"]
        host = doc.get("host", default_host)  # 单篇文档可覆盖 host（跨租户文档）
        url = f"https://{host}/docx/{token}"
        md = html_to_markdown(fetch_html(url))
        content = build_header(url, token, today) + md + "\n"
        out_path = REF_DIR / doc["output"]

        if check:
            old = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
            # 比对时忽略“最后同步”这一行，仅关心正文是否变化。
            norm = lambda s: re.sub(r"> \*\*最后同步\*\*：.*", "", s)
            if norm(old) != norm(content):
                stale.append(doc["output"])
                print(f"✗ 已过期：{doc['output']}")
                sys.stdout.writelines(
                    difflib.unified_diff(
                        norm(old).splitlines(True),
                        norm(content).splitlines(True),
                        out_path.name + " (本地)",
                        out_path.name + " (飞书)",
                        n=1,
                    )
                )
            else:
                print(f"✓ 最新：{doc['output']}")
        else:
            out_path.write_text(content, encoding="utf-8")
            print(f"✓ 已写入 {doc['output']}  ←  {url}")

    if check and stale:
        print(f"\n{len(stale)} 篇文档与飞书原文不一致，运行 `python3 scripts/sync-feishu-docs.py` 更新。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
