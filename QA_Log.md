# 学习问答日志（SpriCoder）

## 2026-05-25 ｜ 我和你聊天的时候，有时候文字会变成表情或者乱码，如何处理

- **答案核心**：
  - 先诊断：实测 `locale` 为 `zh_CN.UTF-8`、`LC_CTYPE=zh_CN.UTF-8`，终端是 iTerm.app → **系统编码正常**，可排除经典编码问题。
  - "变表情"属**输入侧误触**：①macOS 地球键 🌐/fn 弹 Emoji 选择器（系统设置→键盘→"按下🌐键时"改为"不执行任何操作"）；②输入法的 emoji 联想候选；③macOS 文本替换。
  - "显示乱码/方块/错位"属 **iTerm2 字体渲染**：用 CJK+emoji 覆盖好的字体（Sarasa Mono SC / Nerd Font）、勾 "Use a different font for non-ASCII text"、勾 "Treat ambiguous-width characters as double width"、确认 Character Encoding=UTF-8。
  - 临场：若只是本地显示异常但字发对了，我读的是底层文本通常没事；若真发出乱码，重发或补一句澄清即可，我以后说为准。
- **相关引用**：本机实测 `locale` 输出；iTerm2 Settings→Profiles→Text / Terminal。
