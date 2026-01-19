"""Claude Code Session Viewer - Backend API

提供 Claude Code 会话历史的查看和搜索功能。
使用 SQLite FTS5 进行全文索引和搜索。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

# 配置路径
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = BASE_DIR / "claude_viewer.html"
FAVICON_PATH = BASE_DIR / "favicon.svg"

CLAUDE_PROJECTS_DIR = Path(
    os.environ.get("CLAUDE_PROJECTS_DIR", "~/.claude/projects")
).expanduser()
DB_PATH = Path(
    os.environ.get("CLAUDE_VIEWER_DB", "~/.claude/claude_viewer.sqlite3")
).expanduser()

INDEX_LOCK = threading.Lock()

# Precompiled regex for CJK character processing
CJK_CHAR_RE = re.compile(r"([\u4e00-\u9fff])")
CJK_SPACE_RE = re.compile(r"\s*([\u4e00-\u9fff])\s*")

app = FastAPI(title="Claude Session Viewer", version="1.0.0")


def get_db() -> sqlite3.Connection:
    """创建数据库连接"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_session() -> Generator[sqlite3.Connection, None, None]:
    """数据库连接上下文管理器"""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    """初始化数据库表结构"""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # 检查是否存在旧版本的 schema
    cursor = conn.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type='table' AND name='sessions'
        """
    )
    row = cursor.fetchone()
    needs_rebuild = False

    if row and "PRIMARY KEY (project, session_id)" not in row[0]:
        needs_rebuild = True

    # 检查 FTS 表是否使用了正确的 tokenizer
    cursor = conn.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type='table' AND name='messages_fts'
        """
    )
    fts_row = cursor.fetchone()
    if fts_row and "tokenize='unicode61'" not in fts_row[0]:
        needs_rebuild = True

    if needs_rebuild:
        # 旧版本 schema，需要重建
        print("检测到旧版本数据库 schema，正在重建...")
        conn.execute("DROP TABLE IF EXISTS sessions")
        conn.execute("DROP TABLE IF EXISTS files")
        conn.execute("DROP TABLE IF EXISTS messages_fts")

    # 会话元数据表
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT NOT NULL,
            project TEXT NOT NULL,
            file_path TEXT NOT NULL,
            started_at TEXT,
            updated_at TEXT,
            message_count INTEGER NOT NULL DEFAULT 0,
            summary TEXT,
            cwd TEXT,
            git_branch TEXT,
            PRIMARY KEY (project, session_id)
        )
        """
    )

    # 文件索引表（用于增量更新）
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            file_path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            session_id TEXT NOT NULL,
            project TEXT NOT NULL
        )
        """
    )

    # 全文搜索表 - 使用 unicode61 tokenizer
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
        USING fts5(
            session_id,
            project,
            role,
            type,
            content,
            timestamp,
            tokenize='unicode61'
        )
        """
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project)"
    )


def stringify(value: Any) -> str:
    """将任意值转换为字符串"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except Exception:
        return str(value)


def preprocess_for_fts(text: str) -> str:
    """预处理文本以支持中文FTS搜索

    在中文字符之间添加空格，使unicode61 tokenizer能够正确分词
    """
    if not text:
        return ""
    return CJK_CHAR_RE.sub(r" \1 ", text)


def restore_from_fts(text: str) -> str:
    """还原FTS预处理的文本，去掉中文字符周围的空格"""
    if not text:
        return ""
    return CJK_SPACE_RE.sub(r"\1", text)


def extract_excerpt_with_context(content: str, query: str, max_length: int = 400) -> str:
    """提取包含关键词上下文的摘要

    Args:
        content: 完整文本内容
        query: 搜索关键词
        max_length: 摘要最大长度

    Returns:
        包含关键词的摘要文本
    """
    if not content or not query:
        return content[:max_length] if content else ""

    # 查找关键词位置（不区分大小写）
    query_lower = query.lower()
    content_lower = content.lower()
    pos = content_lower.find(query_lower)

    if pos == -1:
        # 找不到关键词，返回开头
        return content[:max_length]

    # 计算摘要的起始和结束位置
    # 尽量让关键词居中，但如果关键词在开头或结尾，则调整
    half_length = max_length // 2
    start = max(0, pos - half_length)
    end = min(len(content), start + max_length)

    # 如果end到达末尾，调整start以显示更多内容
    if end == len(content) and len(content) > max_length:
        start = max(0, end - max_length)

    excerpt = content[start:end]

    # 添加省略号
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(content):
        excerpt = excerpt + "..."

    return excerpt


def normalize_event(raw: Dict[str, Any], fallback_session_id: str) -> Dict[str, str]:
    """标准化事件数据结构"""
    event_type = stringify(raw.get("type", "")).strip()
    timestamp = stringify(raw.get("timestamp", "")).strip()
    session_id = stringify(raw.get("sessionId") or raw.get("session_id") or "")
    session_id = session_id.strip() or fallback_session_id

    # 提取消息内容
    message = raw.get("message")
    role = ""
    content = None
    if isinstance(message, dict):
        role = stringify(message.get("role", "")).strip()
        content = message.get("content")
    elif message is not None:
        content = message

    if content is None:
        content = raw.get("content")

    text = stringify(content).strip()
    cwd = stringify(raw.get("cwd", "")).strip()
    git_branch = stringify(raw.get("gitBranch", "")).strip()

    return {
        "type": event_type,
        "role": role,
        "content": text,
        "timestamp": timestamp,
        "session_id": session_id,
        "cwd": cwd,
        "git_branch": git_branch,
    }


def parse_event_line(line: str, session_id: str) -> Optional[Dict[str, str]]:
    """解析单行事件数据"""
    if not line.strip():
        return None
    try:
        raw = json.loads(line)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return normalize_event(raw, session_id)


def scan_project_files() -> Iterable[tuple[str, Path]]:
    """扫描所有项目的会话文件"""
    if not CLAUDE_PROJECTS_DIR.exists():
        return []
    results = []
    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        for path in project_dir.glob("*.jsonl"):
            results.append((project_dir.name, path))
    return results


def remove_file_records(
    conn: sqlite3.Connection, file_path: str, session_id: str, project: str
) -> None:
    """删除文件相关的所有记录"""
    conn.execute(
        "DELETE FROM messages_fts WHERE session_id = ? AND project = ?",
        (session_id, project),
    )
    conn.execute(
        "DELETE FROM sessions WHERE session_id = ? AND project = ?",
        (session_id, project),
    )
    conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))


def index_file(conn: sqlite3.Connection, project: str, path: Path) -> None:
    """索引单个会话文件"""
    session_id = path.stem
    path_str = str(path)
    mtime = path.stat().st_mtime

    # 删除旧记录
    remove_file_records(conn, path_str, session_id, project)

    # 解析文件内容
    messages: List[tuple[str, str, str, str, str, str]] = []
    started_at = ""
    updated_at = ""
    message_count = 0
    summary = ""
    cwd = ""
    git_branch = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue

            event = normalize_event(raw, session_id)

            # 更新时间范围
            if event["timestamp"]:
                if not started_at or event["timestamp"] < started_at:
                    started_at = event["timestamp"]
                if not updated_at or event["timestamp"] > updated_at:
                    updated_at = event["timestamp"]

            # 统计消息数量
            is_message = event["type"] in {"user", "assistant"} or event["role"] in {
                "user",
                "assistant",
            }
            if is_message:
                message_count += 1
                if not summary and event["content"]:
                    summary = event["content"][:160]

            # 提取元数据
            if event["cwd"] and not cwd:
                cwd = event["cwd"]
            if event["git_branch"] and not git_branch:
                git_branch = event["git_branch"]

            # 添加到全文搜索索引
            if event["content"]:
                messages.append(
                    (
                        session_id,
                        project,
                        event["role"],
                        event["type"],
                        preprocess_for_fts(event["content"]),  # 预处理中文文本
                        event["timestamp"],
                    )
                )

    if not started_at:
        started_at = str(mtime)
    if not updated_at:
        updated_at = started_at

    # 插入会话元数据
    conn.execute(
        """
        INSERT INTO sessions (
            session_id,
            project,
            file_path,
            started_at,
            updated_at,
            message_count,
            summary,
            cwd,
            git_branch
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            project,
            path_str,
            started_at,
            updated_at,
            message_count,
            summary,
            cwd,
            git_branch,
        ),
    )

    # 插入消息到全文搜索表
    if messages:
        conn.executemany(
            """
            INSERT INTO messages_fts (
                session_id, project, role, type, content, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            messages,
        )

    # 记录文件索引状态
    conn.execute(
        """
        INSERT INTO files (file_path, mtime, session_id, project)
        VALUES (?, ?, ?, ?)
        """,
        (path_str, mtime, session_id, project),
    )


def ensure_index(conn: sqlite3.Connection) -> None:
    """确保索引是最新的（增量更新）"""
    with INDEX_LOCK:
        init_db(conn)

        # 获取已索引的文件
        existing = {
            row["file_path"]: row
            for row in conn.execute("SELECT * FROM files").fetchall()
        }

        # 扫描当前文件
        current_paths: set[str] = set()
        for project, path in scan_project_files():
            path_str = str(path)
            mtime = path.stat().st_mtime
            current_paths.add(path_str)

            # 如果文件是新的或已更新，重新索引
            row = existing.get(path_str)
            if row is None or row["mtime"] < mtime:
                index_file(conn, project, path)

        # 删除已不存在的文件记录
        for file_path, row in existing.items():
            if file_path not in current_paths:
                remove_file_records(
                    conn, file_path, row["session_id"], row["project"]
                )

        conn.commit()


@contextmanager
def indexed_db_session() -> Generator[sqlite3.Connection, None, None]:
    """数据库连接上下文管理器（确保索引最新）"""
    with db_session() as conn:
        ensure_index(conn)
        yield conn


def file_response_or_404(
    path: Path, detail: str, media_type: Optional[str] = None
) -> FileResponse:
    """返回静态文件或 404"""
    if not path.exists():
        raise HTTPException(status_code=404, detail=detail)
    if media_type:
        return FileResponse(path, media_type=media_type)
    return FileResponse(path)


# API 端点

@app.get("/")
def index() -> FileResponse:
    """返回前端页面"""
    return file_response_or_404(INDEX_HTML_PATH, "Index file not found")


@app.get("/favicon.svg")
def favicon() -> FileResponse:
    """返回 favicon 图标"""
    return file_response_or_404(
        FAVICON_PATH, "Favicon not found", media_type="image/svg+xml"
    )


@app.get("/api/projects")
def list_projects() -> Dict[str, Any]:
    """列出所有项目"""
    with indexed_db_session() as conn:
        rows = conn.execute(
            """
            SELECT project,
                   COUNT(*) AS session_count,
                   MAX(updated_at) AS updated_at
            FROM sessions
            GROUP BY project
            ORDER BY project
            """
        ).fetchall()
        return {"projects": [dict(row) for row in rows]}


@app.get("/api/projects/{project}/sessions")
def list_sessions(project: str) -> Dict[str, Any]:
    """列出项目的所有会话"""
    with indexed_db_session() as conn:
        rows = conn.execute(
            """
            SELECT session_id,
                   started_at,
                   updated_at,
                   message_count,
                   summary,
                   cwd,
                   git_branch
            FROM sessions
            WHERE project = ?
            ORDER BY updated_at DESC
            """,
            (project,),
        ).fetchall()
        return {"project": project, "sessions": [dict(row) for row in rows]}


def resolve_session_file(
    conn: sqlite3.Connection, session_id: str, project: Optional[str]
) -> Optional[str]:
    """解析会话文件路径"""
    query = "SELECT file_path FROM sessions WHERE session_id = ?"
    params: List[Any] = [session_id]
    if project:
        query += " AND project = ?"
        params.append(project)
    query += " ORDER BY updated_at DESC LIMIT 1"
    row = conn.execute(query, params).fetchone()
    return row["file_path"] if row else None


@app.get("/api/sessions/{session_id}")
def get_session(
    session_id: str,
    project: Optional[str] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
) -> Dict[str, Any]:
    """获取会话详情（支持分页）"""
    with indexed_db_session() as conn:
        file_path = resolve_session_file(conn, session_id, project)
        if not file_path:
            raise HTTPException(status_code=404, detail="Session not found")

        items: List[Dict[str, Any]] = []
        has_more = False
        lines_processed = 0

        with Path(file_path).open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if lines_processed < offset:
                    lines_processed += 1
                    continue

                if len(items) >= limit:
                    has_more = True
                    break

                lines_processed += 1

                event = parse_event_line(line, session_id)
                if not event:
                    continue

                items.append({
                    "type": event["type"],
                    "role": event["role"],
                    "content": event["content"],
                    "timestamp": event["timestamp"],
                    "sessionId": event["session_id"],
                    "cwd": event["cwd"],
                    "gitBranch": event["git_branch"],
                })

            if not has_more:
                has_more = any(line.strip() for line in handle)

        next_offset = lines_processed if has_more else None
        return {
            "session_id": session_id,
            "project": project,
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset,
            "items": items,
        }


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    project: Optional[str] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    """全文搜索"""
    query = preprocess_for_fts(q.strip()).strip()
    with indexed_db_session() as conn:
        params: List[Any] = [query]
        where = "messages_fts MATCH ?"
        if project:
            where += " AND project = ?"
            params.append(project)

        # 获取总数
        count_params = params.copy()
        try:
            count_row = conn.execute(
                f"""
                SELECT COUNT(*) as total
                FROM messages_fts
                WHERE {where}
                """,
                count_params,
            ).fetchone()
            total = count_row['total'] if count_row else 0
        except sqlite3.OperationalError:
            total = 0

        # 获取分页结果
        params.append(limit)
        params.append(offset)

        try:
            rows = conn.execute(
                f"""
                SELECT session_id,
                       project,
                       role,
                       type,
                       timestamp,
                       content
                FROM messages_fts
                WHERE {where}
                ORDER BY bm25(messages_fts)
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
            # 还原文本并提取包含关键词的摘要
            results = []
            for row in rows:
                row_dict = dict(row)
                # 还原完整文本
                full_content = restore_from_fts(row_dict['content'])
                # 提取包含关键词的摘要
                row_dict['excerpt'] = extract_excerpt_with_context(full_content, q, max_length=400)
                # 移除完整content，只保留excerpt
                del row_dict['content']
                results.append(row_dict)
        except sqlite3.OperationalError as e:
            raise HTTPException(status_code=400, detail=f"搜索语法错误: {str(e)}")

        has_more = (offset + len(results)) < total
        return {
            "query": q,
            "results": results,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": has_more,
        }
