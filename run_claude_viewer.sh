#!/usr/bin/env bash
set -euo pipefail

# Claude Session Viewer 启动脚本

export CLAUDE_PROJECTS_DIR="${CLAUDE_PROJECTS_DIR:-$HOME/.claude/projects}"
export CLAUDE_VIEWER_DB="${CLAUDE_VIEWER_DB:-$HOME/.claude/claude_viewer.sqlite3}"

echo "Claude Session Viewer"
echo "===================="
echo "项目目录: $CLAUDE_PROJECTS_DIR"
echo "数据库: $CLAUDE_VIEWER_DB"
echo ""
echo "启动服务器..."
echo "访问地址: http://127.0.0.1:8000"
echo ""

python -m uvicorn claude_viewer:app \
  --host 127.0.0.1 \
  --port 8000
