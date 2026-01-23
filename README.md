# Claude Code 会话查看器

一个用于浏览和搜索 Claude Code/Codex 本地会话历史的 Web 工具。

## 为什么要写这个工具？

现在有了 Claude Code 这样的 AI 编程助手，写小工具变得非常简单。遇到需求时，直接让 AI 帮忙写代码，几分钟就能完成一个功能完整的工具，不用再花时间在网上找现成的了。

## 功能特性

- 📁 **项目管理**: 按项目组织和浏览会话
- 🔍 **智能搜索**: 使用 jieba 分词的中文全文搜索，支持词语级别的精准匹配
- 📄 **分页加载**: 支持大量会话和搜索结果的分页浏览
- 🎯 **精准定位**: 从搜索结果直接跳转到会话中的具体消息
- 🎨 **现代界面**: 使用 Tailwind CSS 构建的清爽界面

## 技术栈

- **后端**: Python + FastAPI
- **前端**: 原生 JavaScript + Tailwind CSS
- **数据库**: SQLite + FTS5 全文搜索
- **中文分词**: jieba（可选，自动回退到字符级分词）
- **Markdown**: Marked.js 渲染

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install fastapi uvicorn jieba
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
export CODEX_SESSIONS_DIR=~/.codex/sessions  # Codex sessions 目录
export CLAUDE_VIEWER_DB=~/.claude/claude_viewer.sqlite3  # 数据库路径
```

## 中文搜索支持

项目使用 **jieba 分词** 实现高质量的中文搜索：

### 搜索特性

- **词语级别匹配**: 使用 jieba 进行专业的中文分词
- **多粒度索引**: 搜索引擎模式生成多粒度的词，提高召回率
- **智能回退**: 如果 jieba 不可用，自动回退到字符级分词
- **中英文混合**: 完美支持中英文混合搜索
- **上下文显示**: 搜索结果智能显示关键词上下文

## 项目结构

```
.
├── claude_viewer.py         # 后端 API 服务
├── claude_viewer.html       # 前端界面
├── favicon.svg              # 网站图标
├── run_claude_viewer.sh     # 启动脚本
└── README.md               # 本文件
```

## License

MIT
