# Claude Code 会话查看器

一个用于浏览 Claude Code/Codex 本地会话历史的 Web 工具。

## 为什么要写这个工具？

现在有了 Claude Code 这样的 AI 编程助手，写小工具变得非常简单。遇到需求时，直接让 AI 帮忙写代码，几分钟就能完成一个功能完整的工具，不用再花时间在网上找现成的了。

## 功能特性

- 📁 **项目管理**: 按项目组织和浏览会话
- 📄 **分页加载**: 支持大量会话和消息的分页浏览
- 🔗 **精准定位**: 支持消息 permalink，刷新或分享链接后可直接定位到对应消息
- 🤖 **多来源支持**: 同时浏览 Claude Code 和 Codex 会话
- 🎨 **现代界面**: 使用 Tailwind CSS 构建的清爽界面

## 技术栈

- **后端**: Python + FastAPI
- **前端**: 原生 JavaScript + Tailwind CSS
- **缓存**: SQLite（仅缓存会话元数据，不做全文索引）
- **Markdown**: Marked.js 渲染

## 快速开始

### 安装依赖

```bash
pip install fastapi uvicorn
```

### 运行

```bash
./run_claude_viewer.sh
```

或者直接运行：

```bash
uvicorn claude_viewer:app --reload
```

然后在浏览器中打开 http://localhost:8000

## 配置

可以通过环境变量自定义路径：

```bash
export CLAUDE_PROJECTS_DIR=~/.claude/projects  # Claude 项目目录
export CODEX_SESSIONS_DIR=~/.codex/sessions    # Codex sessions 目录
export CLAUDE_VIEWER_DB=~/.claude/claude_viewer.sqlite3  # 元数据缓存路径
```

## 升级说明

当前版本已移除全文搜索，只保留会话浏览所需的轻量元数据缓存。

如果你之前使用过带搜索的版本，旧的 `CLAUDE_VIEWER_DB` 里可能还残留较大的历史索引文件。最简单的清理方式是：

```bash
rm -f ~/.claude/claude_viewer.sqlite3
python3 rebuild_index.py
```

这样会重新生成一个更小的数据库。

## 项目结构

```
.
├── claude_viewer.py         # 后端 API 服务
├── claude_viewer.html       # 前端界面
├── favicon.svg              # 网站图标
├── rebuild_index.py         # 重建轻量元数据缓存
├── run_claude_viewer.sh     # 启动脚本
└── README.md                # 本文件
```

## License

MIT
