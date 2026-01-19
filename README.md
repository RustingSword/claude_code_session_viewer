# Claude Code 会话查看器

一个用于浏览和搜索 Claude Code 本地会话历史的 Web 工具。

## 为什么要写这个工具？

现在有了 Claude Code 这样的 AI 编程助手，写小工具变得非常简单。遇到需求时，直接让 AI 帮忙写代码，几分钟就能完成一个功能完整的工具，不用再花时间在网上找现成的了。

这个项目就是一个很好的例子——从零开始，通过与 Claude Code 对话，快速实现了一个功能丰富的会话查看器。

## 功能特性

- 📁 **项目管理**: 按项目组织和浏览会话
- 🔍 **全文搜索**: 支持中英文混合搜索，智能显示关键词上下文
- 📄 **分页加载**: 支持大量会话和搜索结果的分页浏览
- 🎯 **精准定位**: 从搜索结果直接跳转到会话中的具体消息
- 🎨 **现代界面**: 使用 Tailwind CSS 构建的清爽界面
- ⚡ **高性能**: 基于 SQLite FTS5 全文索引，搜索快速

## 技术栈

- **后端**: Python + FastAPI
- **前端**: 原生 JavaScript + Tailwind CSS
- **数据库**: SQLite + FTS5 全文搜索
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
export CLAUDE_VIEWER_DB=~/.claude/claude_viewer.sqlite3  # 数据库路径
```

## 中文搜索支持

项目实现了对中文搜索的完整支持：

- 在索引时对中文文本进行预处理（添加空格）
- 使用 SQLite FTS5 的 unicode61 tokenizer
- 搜索结果智能显示关键词上下文
- 自动高亮搜索关键词

## 项目结构

```
.
├── claude_viewer.py      # 后端 API 服务
├── claude_viewer.html    # 前端界面
├── favicon.svg           # 网站图标
├── run_claude_viewer.sh  # 启动脚本
└── README.md            # 本文件
```

## 开发体验

这个项目完全通过与 Claude Code 对话开发完成，展示了 AI 辅助编程的强大能力：

- ✅ 快速原型开发
- ✅ 实时问题解决
- ✅ 代码优化和重构
- ✅ 功能迭代和完善

## License

MIT
