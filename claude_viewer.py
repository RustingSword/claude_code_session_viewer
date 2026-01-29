"""Claude Code Session Viewer - Backend API

提供 Claude Code 会话历史的查看和搜索功能。
使用 SQLite FTS5 进行全文索引和搜索。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Literal, Optional, TypedDict, Union

# Python 3.9 compatibility: NotRequired is in typing (3.11+). Keep it runtime-safe for 3.9.
try:  # pragma: no cover
    from typing import NotRequired  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    try:
        from typing_extensions import NotRequired  # type: ignore
    except Exception:
        # Minimal fallback: only used for annotations; runtime behavior is irrelevant.
        def NotRequired(tp: Any) -> Any:  # type: ignore
            return tp

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

# 配置路径
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = BASE_DIR / "claude_viewer.html"
FAVICON_PATH = BASE_DIR / "favicon.svg"

CLAUDE_PROJECTS_DIR = Path(
    os.environ.get("CLAUDE_PROJECTS_DIR", "~/.claude/projects")
).expanduser()
CODEX_SESSIONS_DIR = Path(
    os.environ.get("CODEX_SESSIONS_DIR", "~/.codex/sessions")
).expanduser()
DB_PATH = Path(
    os.environ.get("CLAUDE_VIEWER_DB", "~/.claude/claude_viewer.sqlite3")
).expanduser()

INDEX_LOCK = threading.Lock()
LAST_INDEX_TIME = 0.0
INDEX_THROTTLE_SECONDS = 30.0

# Async indexing status (for UI feedback). Indexing work uses its own DB connection.
INDEX_STATUS_LOCK = threading.Lock()
INDEX_BG_THREAD: Optional[threading.Thread] = None
INDEX_STATUS: Dict[str, Any] = {
    "state": "idle",  # idle|running|ready|error
    "started_at": "",
    "finished_at": "",
    "total_files": 0,
    "processed_files": 0,
    "last_error": "",
}

# Precompiled regex for CJK character processing
CJK_CHAR_RE = re.compile(r"([\u4e00-\u9fff])")
CJK_SPACE_RE = re.compile(r"\s*([\u4e00-\u9fff])\s*")

app = FastAPI(title="Claude Session Viewer", version="1.0.0")

DEFAULT_SOURCE = "claude_code"


# ============================================================================
# 阶段 1: 统一数据模型
# ============================================================================
# 定义统一的事件和内容块结构，消除 Claude Code 和 Codex 的格式差异


class TextBlock(TypedDict):
    """文本内容块"""
    type: Literal["text"]
    text: str


class ThinkingBlock(TypedDict):
    """思考过程内容块（统一 Claude 的 thinking 和 Codex 的 reasoning）"""
    type: Literal["thinking"]
    thinking: str


class ToolUseBlock(TypedDict):
    """工具调用内容块"""
    type: Literal["tool_use"]
    name: str
    input: Any
    id: NotRequired[str]


class ToolResultBlock(TypedDict):
    """工具结果内容块"""
    type: Literal["tool_result"]
    content: Any
    id: NotRequired[str]
    is_error: NotRequired[bool]


class ImageBlock(TypedDict):
    """图片内容块"""
    type: Literal["image"]
    source: Dict[str, Any]


class UnknownBlock(TypedDict, total=False):
    """未知类型内容块（用于前向兼容）"""
    type: str


ContentBlock = Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock, ImageBlock, UnknownBlock]


class NormalizedEvent(TypedDict):
    """统一的事件格式

    新字段（推荐使用）：
    - id: 稳定的事件唯一标识
    - source: 数据源（claude_code 或 codex）
    - blocks: 结构化内容块列表
    - meta: 元数据字典

    旧字段（向后兼容）：
    - content: blocks 的 JSON 序列化
    - type: 推断的事件类型
    - cwd, git_branch: 从 meta 中提取
    """
    # 新的统一字段
    id: str
    source: str
    session_id: str
    timestamp: str
    role: str
    blocks: List[ContentBlock]
    meta: Dict[str, Any]

    # 向后兼容字段
    content: NotRequired[str]
    type: NotRequired[str]
    cwd: NotRequired[str]
    git_branch: NotRequired[str]


def generate_stable_event_id(source: str, session_id: str, raw_line: str) -> str:
    """生成稳定的事件 ID

    使用 blake2b 哈希确保：
    1. 同一事件总是生成相同的 ID（不受 Python hash() 随机化影响）
    2. 不同事件生成不同的 ID（避免碰撞）
    3. ID 长度适中（24 字符）
    """
    hasher = hashlib.blake2b(digest_size=12)
    hasher.update(source.encode("utf-8", "replace"))
    hasher.update(b"\0")
    hasher.update(session_id.encode("utf-8", "replace"))
    hasher.update(b"\0")
    hasher.update(raw_line.encode("utf-8", "replace"))
    return hasher.hexdigest()


def safe_json_dumps(value: Any) -> str:
    """安全的 JSON 序列化，永不失败"""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False, separators=(",", ":"))


def blocks_to_json_string(blocks: List[ContentBlock]) -> str:
    """将 blocks 序列化为 JSON 字符串（用于向后兼容的 content 字段）"""
    if not blocks:
        return ""
    return safe_json_dumps(blocks)


def blocks_to_plaintext(blocks: List[ContentBlock]) -> str:
    """将 blocks 转换为纯文本（用于全文索引和搜索）"""
    parts: List[str] = []

    for block in blocks:
        if not isinstance(block, dict):
            text = str(block).strip()
            if text:
                parts.append(text)
            continue

        block_type = str(block.get("type", "")).strip()

        if block_type == "text":
            text = str(block.get("text", "")).strip()
            if text:
                parts.append(text)

        elif block_type == "thinking":
            thinking = str(block.get("thinking", "")).strip()
            if thinking:
                parts.append(thinking)

        elif block_type == "tool_use":
            name = str(block.get("name", "")).strip()
            tool_input = block.get("input")
            if tool_input is not None and tool_input != "":
                input_str = safe_json_dumps(tool_input)
                parts.append(f"{name}\n{input_str}" if name else input_str)
            elif name:
                parts.append(name)

        elif block_type == "tool_result":
            content = block.get("content")
            if isinstance(content, str):
                text = content.strip()
            elif content is None:
                text = ""
            else:
                text = safe_json_dumps(content).strip()
            if text:
                parts.append(text)

        elif block_type == "image":
            # 图片不参与文本搜索，只保留占位符
            source = block.get("source")
            if isinstance(source, dict):
                media_type = str(source.get("media_type", "")).strip()
                parts.append(f"[image: {media_type}]" if media_type else "[image]")
            else:
                parts.append("[image]")

        else:
            # 未知类型：序列化为 JSON 以便索引
            parts.append(safe_json_dumps(block))

    return "\n\n".join(parts).strip()


def coerce_to_blocks(value: Any) -> List[ContentBlock]:
    """将任意值转换为标准的 ContentBlock 列表

    处理以下情况：
    1. None -> []
    2. 字符串 -> [{type: "text", text: "..."}]
    3. 字典（单个 block）-> [block]
    4. 列表（多个 blocks）-> blocks
    5. 其他类型 -> [{type: "text", text: str(value)}]
    """
    if value is None:
        return []

    if isinstance(value, list):
        blocks: List[ContentBlock] = []
        for item in value:
            if isinstance(item, dict):
                # 已有 type 字段的视为有效 block
                if "type" in item:
                    blocks.append(item)  # type: ignore[arg-type]
                else:
                    # 没有 type 的字典视为未知 block
                    blocks.append({"type": "unknown", **item})  # type: ignore[misc]
            elif isinstance(item, str):
                text = item.strip()
                if text:
                    blocks.append({"type": "text", "text": text})
            else:
                text = str(item).strip()
                if text:
                    blocks.append({"type": "text", "text": text})
        return blocks

    if isinstance(value, dict):
        # 单个 block 对象
        if "type" in value:
            return [value]  # type: ignore[list-item]
        # 没有 type 的字典转为文本
        return [{"type": "text", "text": safe_json_dumps(value)}]

    if isinstance(value, str):
        text = value.strip()
        return [{"type": "text", "text": text}] if text else []

    # 其他类型转为字符串
    text = str(value).strip()
    return [{"type": "text", "text": text}] if text else []


def infer_legacy_type(role: str, blocks: List[ContentBlock], fallback: str = "event") -> str:
    """推断旧的 type 字段（用于向后兼容）

    注意：不再使用 "reasoning"/"tool_call"/"tool_output" 这些事件级类型，
    因为它们已经统一为 blocks 中的 "thinking"/"tool_use"/"tool_result"
    """
    role = (role or "").strip()
    if role in {"user", "assistant", "tool"}:
        return role

    # 如果 role 为空，尝试从第一个 block 推断
    if blocks and isinstance(blocks[0], dict):
        block_type = str(blocks[0].get("type", "")).strip()
        if block_type:
            return block_type

    return fallback


# ============================================================================
# 阶段 2: 统一后端解析器
# ============================================================================
# 将解析逻辑抽象为类结构，便于扩展和维护


class EventParser(ABC):
    """事件解析器抽象基类

    职责：
    1. 定义统一的解析接口
    2. 提供公共的辅助方法（event_id 生成、结果组装）
    3. 消除重复的 meta/legacy 字段生成逻辑
    """

    def __init__(self, source: str) -> None:
        self.source = (source or DEFAULT_SOURCE).strip() or DEFAULT_SOURCE

    @abstractmethod
    def parse_raw(
        self,
        raw: Dict[str, Any],
        fallback_session_id: str,
        *,
        event_id: str = "",
    ) -> NormalizedEvent:
        """将原始事件字典解析为 NormalizedEvent

        Args:
            raw: 原始事件字典
            fallback_session_id: 当事件中没有 session_id 时使用的默认值
            event_id: 预生成的事件 ID（如果为空则自动生成）

        Returns:
            NormalizedEvent: 统一的事件格式
        """

    def _ensure_event_id(self, event_id: str, session_id: str, raw: Dict[str, Any]) -> str:
        """确保事件 ID 存在

        如果 event_id 已提供则直接返回，否则基于 raw 内容生成。
        注意：parse_event_line() 会基于原始行生成更稳定的 ID。
        """
        if event_id:
            return event_id
        return generate_stable_event_id(self.source, session_id, safe_json_dumps(raw))

    def _finalize(
        self,
        *,
        event_id: str,
        session_id: str,
        timestamp: str,
        role: str,
        blocks: List[ContentBlock],
        meta: Dict[str, Any],
        legacy_type_fallback: str,
    ) -> NormalizedEvent:
        """统一组装 NormalizedEvent（消除重复逻辑）

        这个方法将 blocks 和 meta 转换为完整的 NormalizedEvent，
        包括新字段和向后兼容的旧字段。
        """
        # 确保 meta 中有 cwd 和 git_branch
        cwd = str(meta.get("cwd", "") or "").strip()
        git_branch = str(meta.get("git_branch", "") or "").strip()
        meta_out = dict(meta)
        meta_out["cwd"] = cwd
        meta_out["git_branch"] = git_branch

        # 生成向后兼容的字段
        legacy_content = blocks_to_json_string(blocks)
        legacy_type = infer_legacy_type(role, blocks, fallback=legacy_type_fallback)

        return {
            # 新的统一字段
            "id": event_id,
            "source": self.source,
            "session_id": session_id,
            "timestamp": timestamp,
            "role": role,
            "blocks": blocks,
            "meta": meta_out,
            # 向后兼容的旧字段
            "content": legacy_content,
            "type": legacy_type,
            "cwd": cwd,
            "git_branch": git_branch,
        }


class ClaudeParser(EventParser):
    """Claude Code 事件解析器

    处理 Claude Code 格式的事件，这种格式已经使用 blocks 结构，
    主要工作是提取字段和规范化。
    """

    def parse_raw(
        self,
        raw: Dict[str, Any],
        fallback_session_id: str,
        *,
        event_id: str = "",
    ) -> NormalizedEvent:
        # 提取基本字段
        event_type = stringify(raw.get("type", "")).strip()
        timestamp = normalize_timestamp(raw.get("timestamp", ""))
        session_id = stringify(raw.get("sessionId") or raw.get("session_id") or "")
        session_id = session_id.strip() or fallback_session_id

        # 提取消息内容和角色
        message = raw.get("message")
        role = ""
        content: Any = None

        if isinstance(message, dict):
            role = stringify(message.get("role", "")).strip()
            content = message.get("content")
        elif message is not None:
            content = message

        if content is None:
            content = raw.get("content")

        # 提取元数据
        cwd = stringify(raw.get("cwd", "")).strip()
        git_branch = stringify(raw.get("gitBranch", "")).strip()

        # 转换为标准 blocks
        blocks = coerce_to_blocks(content)

        # 确保事件 ID
        event_id = self._ensure_event_id(event_id, session_id, raw)

        # 组装元数据
        meta: Dict[str, Any] = {
            "cwd": cwd,
            "git_branch": git_branch,
            "event_type": event_type,
        }

        return self._finalize(
            event_id=event_id,
            session_id=session_id,
            timestamp=timestamp,
            role=role,
            blocks=blocks,
            meta=meta,
            legacy_type_fallback=event_type or "event",
        )


class CodexParser(EventParser):
    """Codex 事件解析器

    处理 Codex 格式的事件，需要将各种事件类型转换为统一的 blocks 格式：
    - agent_reasoning → thinking block
    - function_call → tool_use block
    - function_call_output → tool_result block
    """

    def parse_raw(
        self,
        raw: Dict[str, Any],
        fallback_session_id: str,
        *,
        event_id: str = "",
    ) -> NormalizedEvent:
        # 提取基本字段
        event_type = stringify(raw.get("type", "")).strip()
        timestamp = normalize_timestamp(raw.get("timestamp") or raw.get("time"))
        session_id = stringify(
            raw.get("session_id")
            or raw.get("sessionId")
            or raw.get("session")
            or ""
        ).strip() or fallback_session_id

        # 提取 payload
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        payload_type = stringify(payload.get("type", "")).strip()

        # 过滤系统事件（如 token_count）
        if payload_type in ("token_count",):
            event_id = self._ensure_event_id(event_id, session_id, raw)
            return {
                "id": event_id,
                "source": self.source,
                "session_id": session_id,
                "timestamp": timestamp,
                "role": "",
                "blocks": [],
                "meta": {"event_type": event_type, "payload_type": payload_type},
                "content": "",
                "type": "",
                "cwd": "",
                "git_branch": "",
            }

        # 提取元数据
        cwd = stringify(payload.get("cwd") or raw.get("cwd") or "").strip()
        git_branch = stringify(
            payload.get("git_branch")
            or payload.get("gitBranch")
            or raw.get("git_branch")
            or raw.get("gitBranch")
            or ""
        ).strip()

        role = ""
        blocks: List[ContentBlock] = []

        # 处理 event_msg 类型
        if event_type == "event_msg":
            if payload_type == "user_message":
                role = "user"
                blocks = coerce_to_blocks(extract_codex_text(payload.get("message")))

            elif payload_type == "agent_message":
                role = "assistant"
                blocks = coerce_to_blocks(extract_codex_text(payload.get("message")))

            elif payload_type == "agent_reasoning":
                # 关键：将 Codex 的 reasoning 统一为 thinking block
                role = "assistant"
                thinking_text = extract_codex_text(payload.get("text"))
                if thinking_text.strip():
                    blocks = [{"type": "thinking", "thinking": thinking_text.strip()}]

        # 处理 response_item 类型
        elif event_type == "response_item":
            if payload_type == "message":
                if "role" in payload:
                    role = stringify(payload.get("role") or "").strip()
                    blocks = coerce_to_blocks(
                        extract_codex_text(payload.get("content") or payload.get("message"))
                    )
                else:
                    item = payload.get("item")
                    if isinstance(item, dict):
                        role = stringify(item.get("role") or "").strip()
                        blocks = coerce_to_blocks(
                            extract_codex_text(item.get("content") or item.get("message"))
                        )

            elif payload_type == "reasoning":
                # 加密的 reasoning，跳过
                blocks = []

            elif payload_type == "function_call":
                # 关键：将 Codex 的 function_call 统一为 tool_use block
                role = "assistant"
                tool_name = stringify(payload.get("name", "")).strip()
                call_id = stringify(payload.get("call_id", "")).strip()
                raw_args = payload.get("arguments")

                # 解析参数
                tool_input: Any = {}
                if isinstance(raw_args, (dict, list)):
                    tool_input = raw_args
                else:
                    arguments_str = stringify(raw_args or "").strip()
                    if arguments_str:
                        try:
                            tool_input = json.loads(arguments_str)
                        except Exception:
                            tool_input = {"_raw": arguments_str}

                tool_block: ToolUseBlock = {
                    "type": "tool_use",
                    "name": tool_name or "tool",
                    "input": tool_input,
                }
                if call_id:
                    tool_block["id"] = call_id
                blocks = [tool_block]

            elif payload_type == "custom_tool_call":
                # 关键：将 Codex 的 custom_tool_call 统一为 tool_use block
                role = "assistant"
                tool_name = stringify(payload.get("name", "")).strip()
                call_id = stringify(payload.get("call_id", "")).strip()
                raw_input = payload.get("input")

                # 解析参数（可能是 dict/list 或字符串）
                tool_input: Any = {}
                if isinstance(raw_input, (dict, list)):
                    tool_input = raw_input
                else:
                    input_str = stringify(raw_input or "").strip()
                    if input_str:
                        try:
                            tool_input = json.loads(input_str)
                        except Exception:
                            tool_input = {"_raw": input_str}

                tool_block: ToolUseBlock = {
                    "type": "tool_use",
                    "name": tool_name or "tool",
                    "input": tool_input,
                }
                if call_id:
                    tool_block["id"] = call_id
                blocks = [tool_block]

            elif payload_type in ("function_call_output", "custom_tool_call_output"):
                # 关键：将 Codex 的 function_call_output 统一为 tool_result block
                # 保留结构化内容（阶段 1 的修复）
                role = "tool"
                call_id = stringify(payload.get("call_id", "")).strip()
                raw_output = payload.get("output")

                # 保留结构化内容（如果是 dict/list）或字符串
                if isinstance(raw_output, (dict, list)):
                    output_content: Any = raw_output
                else:
                    output_str = stringify(raw_output or "").strip()
                    # 尝试解析为 JSON
                    if output_str:
                        try:
                            output_content = json.loads(output_str)
                        except Exception:
                            output_content = output_str
                    else:
                        output_content = ""

                result_block: ToolResultBlock = {
                    "type": "tool_result",
                    "content": output_content,
                }
                if call_id:
                    result_block["id"] = call_id
                blocks = [result_block]

        # Fallback：从其他字段提取内容
        if not blocks:
            fallback_text = extract_codex_text(
                payload.get("content") or payload.get("message") or raw.get("content")
            )
            blocks = coerce_to_blocks(fallback_text)

        # 启发式推断 role
        if not role:
            if event_type.startswith("request"):
                role = "user"
            elif event_type.startswith("response"):
                role = "assistant"

        # 确保事件 ID
        event_id = self._ensure_event_id(event_id, session_id, raw)

        # 组装元数据
        meta: Dict[str, Any] = {
            "cwd": cwd,
            "git_branch": git_branch,
            "event_type": event_type,
            "payload_type": payload_type,
        }

        return self._finalize(
            event_id=event_id,
            session_id=session_id,
            timestamp=timestamp,
            role=role,
            blocks=blocks,
            meta=meta,
            legacy_type_fallback=event_type or "event",
        )


# 解析器缓存（避免重复创建）
_PARSER_CACHE: Dict[str, EventParser] = {}


def get_parser(source: str) -> EventParser:
    """解析器工厂函数

    根据 source 返回对应的解析器实例。
    使用缓存避免重复创建解析器对象。

    Args:
        source: 数据源标识（"claude_code" 或 "codex"）

    Returns:
        EventParser: 对应的解析器实例
    """
    src = (source or DEFAULT_SOURCE).strip() or DEFAULT_SOURCE

    # 检查缓存
    cached = _PARSER_CACHE.get(src)
    if cached:
        return cached

    # 创建新解析器
    if src == "codex":
        parser: EventParser = CodexParser(src)
    else:
        # 默认使用 ClaudeParser（支持 claude_code 和其他未知 source）
        parser = ClaudeParser(src)

    # 缓存并返回
    _PARSER_CACHE[src] = parser
    return parser


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

    # 检查是否需要迁移
    cursor = conn.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type='table' AND name='sessions'
        """
    )
    row = cursor.fetchone()
    needs_migration = False
    needs_rebuild = False

    if row:
        schema_sql = row[0]
        # 检查是否有 source 字段
        if "source" not in schema_sql:
            needs_migration = True
        # 检查是否有 parent_session_id 字段
        elif "parent_session_id" not in schema_sql:
            needs_rebuild = True
        # 检查主键是否正确
        elif "PRIMARY KEY (source, project, session_id)" not in schema_sql:
            needs_rebuild = True

    # 检查 FTS 表
    cursor = conn.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type='table' AND name='messages_fts'
        """
    )
    fts_row = cursor.fetchone()
    if fts_row:
        fts_sql = fts_row[0]
        # tokenizer 不匹配需要重建；缺少 source 列优先走迁移（若 sessions 也缺 source）
        if "tokenize='unicode61'" not in fts_sql:
            needs_rebuild = True
        elif "source" not in fts_sql and not needs_migration:
            needs_rebuild = True
        # 精确跳转需要 event_id/line_no
        elif "event_id" not in fts_sql or "line_no" not in fts_sql:
            needs_rebuild = True

    if needs_rebuild:
        print("检测到旧版本数据库 schema，正在重建...")
        conn.execute("DROP TABLE IF EXISTS sessions")
        conn.execute("DROP TABLE IF EXISTS files")
        conn.execute("DROP TABLE IF EXISTS messages_fts")
        needs_migration = False
    elif needs_migration:
        print("检测到需要迁移数据库，正在执行平滑迁移...")
        _migrate_db_add_source(conn)
        return

    # 会话元数据表
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            source TEXT NOT NULL,
            session_id TEXT NOT NULL,
            project TEXT NOT NULL,
            file_path TEXT NOT NULL,
            started_at TEXT,
            updated_at TEXT,
            message_count INTEGER NOT NULL DEFAULT 0,
            summary TEXT,
            cwd TEXT,
            git_branch TEXT,
            parent_session_id TEXT,
            PRIMARY KEY (source, project, session_id)
        )
        """
    )

    # 文件索引表（用于增量更新）
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            file_path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            source TEXT NOT NULL,
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
            source UNINDEXED,
            session_id UNINDEXED,
            project UNINDEXED,
            event_id UNINDEXED,
            line_no UNINDEXED,
            role,
            type,
            content,
            timestamp UNINDEXED,
            tokenize='unicode61'
        )
        """
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_source_project ON sessions(source, project)"
    )


def _migrate_db_add_source(conn: sqlite3.Connection) -> None:
    """平滑迁移：为现有数据添加 source 字段"""
    try:
        conn.execute("BEGIN")

        # 迁移 sessions 表
        conn.execute(
            """
            CREATE TABLE sessions_new (
                source TEXT NOT NULL,
                session_id TEXT NOT NULL,
                project TEXT NOT NULL,
                file_path TEXT NOT NULL,
                started_at TEXT,
                updated_at TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                summary TEXT,
                cwd TEXT,
                git_branch TEXT,
                parent_session_id TEXT,
                PRIMARY KEY (source, project, session_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions_new (
                source, session_id, project, file_path, started_at, updated_at,
                message_count, summary, cwd, git_branch, parent_session_id
            )
            SELECT 'claude_code', session_id, project, file_path, started_at, updated_at,
                   message_count, summary, cwd, git_branch, NULL
            FROM sessions
            """
        )
        conn.execute("DROP TABLE sessions")
        conn.execute("ALTER TABLE sessions_new RENAME TO sessions")

        # 迁移 files 表
        conn.execute(
            """
            CREATE TABLE files_new (
                file_path TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                source TEXT NOT NULL,
                session_id TEXT NOT NULL,
                project TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO files_new (file_path, mtime, source, session_id, project)
            SELECT file_path, mtime, 'claude_code', session_id, project
            FROM files
            """
        )
        conn.execute("DROP TABLE files")
        conn.execute("ALTER TABLE files_new RENAME TO files")

        # 迁移 messages_fts 表
        # 注意：后续版本需要 event_id/line_no 做精确跳转，这里仅处理旧库加 source 的场景。
        # 如果缺少 event_id/line_no，将在 init_db() 中触发自动重建。
        conn.execute(
            """
            CREATE VIRTUAL TABLE messages_fts_new
            USING fts5(
                source UNINDEXED,
                session_id UNINDEXED,
                project UNINDEXED,
                event_id UNINDEXED,
                line_no UNINDEXED,
                role,
                type,
                content,
                timestamp UNINDEXED,
                tokenize='unicode61'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO messages_fts_new (
                source, session_id, project, event_id, line_no, role, type, content, timestamp
            )
            SELECT 'claude_code', session_id, project, '', 0, role, type, content, timestamp
            FROM messages_fts
            """
        )
        conn.execute("DROP TABLE messages_fts")
        conn.execute("ALTER TABLE messages_fts_new RENAME TO messages_fts")

        # 强制下一次 ensure_index() 全量重建索引：
        # - 老库没有 event_id/line_no，且历史数据无法补齐，只能通过重扫 jsonl 生成。
        # - 清空 files 表会让 ensure_index() 认为全部文件都需要重新索引。
        conn.execute("DELETE FROM files")

        # 创建索引
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_source_project ON sessions(source, project)"
        )

        conn.execute("COMMIT")
        print("数据库迁移完成")
    except Exception as e:
        conn.execute("ROLLBACK")
        print(f"迁移失败: {e}，将重建数据库")
        # 重建
        conn.execute("DROP TABLE IF EXISTS sessions")
        conn.execute("DROP TABLE IF EXISTS files")
        conn.execute("DROP TABLE IF EXISTS messages_fts")
        init_db(conn)


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

    优先使用 jieba 分词（如果可用），否则回退到字符级分词。

    jieba 方案优点：
    - 专业的中文分词，准确度高
    - 数据库更小（不需要在每个字符后加空格）
    - 支持词语级别的搜索

    字符级方案（回退）：
    - 无需额外依赖
    - 在每个中文字符后添加空格
    """
    if not text:
        return ""

    if JIEBA_AVAILABLE:
        # 使用 jieba 搜索引擎模式分词（生成多粒度的词）
        words = jieba.cut_for_search(text)
        # 过滤空白词
        words = [w.strip() for w in words if w.strip()]
        return " ".join(words)
    else:
        # 回退到字符级分词
        return CJK_CHAR_RE.sub(r"\1 ", text)


def restore_from_fts(text: str) -> str:
    """还原FTS预处理的文本

    jieba 模式：去除分词产生的空格
    字符级模式：去除中文字符后的空格
    """
    if not text:
        return ""

    if JIEBA_AVAILABLE:
        # jieba 分词后简单去除多余空格
        # 注意：无法完美还原，但搜索结果使用原始文本，所以影响不大
        return re.sub(r'\s+', '', text)
    else:
        # 去掉中文字符后的空格
        return re.sub(r"([\u4e00-\u9fff])\s+", r"\1", text)


def normalize_timestamp(value: Any) -> str:
    """规范化时间戳为 ISO8601 UTC 字符串

    接受 ISO 字符串、epoch 秒、epoch 毫秒和 float mtime。
    无法解析时返回原始字符串。
    """
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:  # 可能是毫秒
            ts /= 1000.0
        # 1e9 约等于 2001-09-09，大于此值视为 epoch 秒
        if ts >= 1e9:
            return (
                datetime.fromtimestamp(ts, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        return str(value)
    s = str(value).strip()
    if not s:
        return ""
    # 尝试解析为数字
    try:
        ts = float(s)
        if ts > 1e12:
            ts /= 1000.0
        if ts >= 1e9:
            return (
                datetime.fromtimestamp(ts, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
    except Exception:
        pass
    return s


def extract_codex_text(value: Any) -> str:
    """从 Codex session payload 中提取可读文本

    支持多种嵌套结构和内容块格式。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        # 常见结构: {"type":"input_text","text":"..."}, {"output_text":"..."}, {"text":"..."}
        if isinstance(value.get("text"), str):
            return value["text"].strip()
        for k in ("input_text", "output_text"):
            if isinstance(value.get(k), str):
                return value[k].strip()
        # 有时嵌套在其他字段中
        for k in ("content", "message", "item"):
            if k in value:
                txt = extract_codex_text(value.get(k))
                if txt:
                    return txt
        return ""
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                # 内容块: {"type":"input_text","text":"..."} / {"type":"output_text","text":"..."}
                txt = ""
                if isinstance(item.get("text"), str):
                    txt = item["text"]
                elif isinstance(item.get("input_text"), str):
                    txt = item["input_text"]
                elif isinstance(item.get("output_text"), str):
                    txt = item["output_text"]
                else:
                    txt = extract_codex_text(item.get("content"))
                txt = (txt or "").strip()
                if txt:
                    parts.append(txt)
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    return ""


def normalize_codex_event(
    raw: Dict[str, Any],
    fallback_session_id: str,
    *,
    event_id: str = "",
    source: str = "codex",
) -> NormalizedEvent:
    """规范化 Codex 事件（兼容入口）

    这个函数保留是为了向后兼容，实际工作委托给解析器工厂。
    新代码应该直接使用 parse_event_line() 或 get_parser(source).parse_raw()。
    """
    return get_parser(source).parse_raw(raw, fallback_session_id, event_id=event_id)


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


def normalize_event(
    raw: Dict[str, Any],
    fallback_session_id: str,
    *,
    source: str = "claude_code",
    event_id: str = "",
) -> NormalizedEvent:
    """规范化 Claude Code 事件（兼容入口）

    这个函数保留是为了向后兼容，实际工作委托给解析器工厂。
    新代码应该直接使用 parse_event_line() 或 get_parser(source).parse_raw()。
    """
    return get_parser(source).parse_raw(raw, fallback_session_id, event_id=event_id)


def parse_event_line(line: str, session_id: str, source: str) -> Optional[NormalizedEvent]:
    """解析单行事件数据（统一入口）

    这是唯一的事件解析入口，确保：
    1. index_file() 和 get_session() 使用相同的解析逻辑
    2. 生成稳定的事件 ID
    3. 返回统一的 NormalizedEvent 格式
    4. 使用解析器工厂模式（阶段 2）
    """
    if not line.strip():
        return None

    try:
        raw = json.loads(line)
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None

    # 生成稳定的事件 ID（基于原始行内容）
    event_id = generate_stable_event_id(source, session_id, line.rstrip("\n"))

    # 使用解析器工厂获取对应的解析器
    parser = get_parser(source)
    return parser.parse_raw(raw, session_id, event_id=event_id)


def scan_session_files() -> Iterable[tuple[str, str, Path, Optional[str]]]:
    """扫描 Claude Code 和 Codex session 文件

    返回 (source, project, path, parent_session_id) 四元组。
    - Claude Code: project = 目录名
    - Codex: project = YYYY-MM-DD（从路径提取）
    - parent_session_id: 父会话 ID（主会话为 None，子会话为父会话的 session_id）
    """
    # Claude Code sessions
    if CLAUDE_PROJECTS_DIR.exists():
        for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
            if not project_dir.is_dir():
                continue
            # 扫描主会话文件
            for path in project_dir.glob("*.jsonl"):
                yield ("claude_code", project_dir.name, path, None)

                # 检查是否有 subagents 子目录
                session_dir = project_dir / path.stem
                subagents_dir = session_dir / "subagents"
                if subagents_dir.exists() and subagents_dir.is_dir():
                    # 扫描子会话文件
                    for subagent_path in subagents_dir.glob("*.jsonl"):
                        yield ("claude_code", project_dir.name, subagent_path, path.stem)

    # Codex sessions
    if CODEX_SESSIONS_DIR.exists():
        for path in CODEX_SESSIONS_DIR.glob("*/*/*/*.jsonl"):
            try:
                day = path.parent.name
                month = path.parent.parent.name
                year = path.parent.parent.parent.name
                project = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            except Exception:
                # Fallback: 使用文件 mtime 的日期
                ts = path.stat().st_mtime
                project = (
                    datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
                )
            yield ("codex", project, path, None)


def remove_file_records(conn: sqlite3.Connection, file_path: str) -> None:
    """删除文件相关的所有记录"""
    rows = conn.execute(
        "SELECT source, project, session_id FROM sessions WHERE file_path = ?",
        (file_path,),
    ).fetchall()
    for row in rows:
        conn.execute(
            "DELETE FROM messages_fts WHERE source = ? AND project = ? AND session_id = ?",
            (row["source"], row["project"], row["session_id"]),
        )
    conn.execute("DELETE FROM sessions WHERE file_path = ?", (file_path,))
    conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))


def index_file(conn: sqlite3.Connection, source: str, project: str, path: Path, parent_session_id: Optional[str] = None) -> None:
    """索引单个会话文件"""
    session_id = path.stem
    path_str = str(path)
    mtime = path.stat().st_mtime

    # 删除旧记录
    remove_file_records(conn, path_str)

    # 解析文件内容
    # messages_fts: source, session_id, project, event_id, line_no, role, type, content, timestamp
    messages: List[tuple[str, str, str, str, int, str, str, str, str]] = []
    started_at = ""
    updated_at = ""
    message_count = 0
    summary = ""
    cwd = ""
    git_branch = ""

    # 用于基于内容的去重（针对系统消息）
    seen_content_hashes: set[str] = set()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle):
            # 使用统一的解析入口
            event = parse_event_line(line, session_id, source)
            if not event:
                continue

            # 提取 blocks
            blocks = event.get("blocks") or []

            # 更新时间范围（即使没有可索引内容也要更新）
            timestamp = event.get("timestamp") or ""
            if timestamp:
                if not started_at or timestamp < started_at:
                    started_at = timestamp
                if not updated_at or timestamp > updated_at:
                    updated_at = timestamp

            # 提取元数据（即使没有可索引内容也要提取）
            event_cwd = (event.get("cwd") or "").strip()
            event_git = (event.get("git_branch") or "").strip()
            if event_cwd and not cwd:
                cwd = event_cwd
            if event_git and not git_branch:
                git_branch = event_git

            # 转换为纯文本用于索引
            text_for_index = blocks_to_plaintext(blocks)
            if not text_for_index:
                # 没有可索引内容，跳过（但时间范围和元数据已更新）
                continue

            # Codex 去重：过滤 event_msg 类型的 user_message 和 agent_message
            # 因为它们的内容已经在对应的 response_item 事件中了
            if source == "codex":
                meta = event.get("meta") or {}
                event_type = meta.get("event_type", "")
                payload_type = meta.get("payload_type", "")
                if event_type == "event_msg" and payload_type in ("user_message", "agent_message"):
                    continue

            # 基于内容的去重（针对 Codex 系统消息）
            # 系统消息（如 AGENTS.md, environment_context）在每个 turn 中都会重复
            if source == "codex":
                role = (event.get("role") or "").strip()
                # 检测系统消息：role 是 user/developer 且内容以特定标记开头
                if role in ("user", "developer") and blocks:
                    first_block = blocks[0]
                    if isinstance(first_block, dict) and first_block.get("type") == "text":
                        text = first_block.get("text", "")
                        # 系统消息特征：以 <permissions、# AGENTS.md、<environment_context 开头
                        if text.startswith(("<permissions", "# AGENTS.md", "<environment_context")):
                            # 计算内容哈希
                            content_hash = hashlib.blake2b(
                                text.encode("utf-8", "replace"), digest_size=16
                            ).hexdigest()
                            if content_hash in seen_content_hashes:
                                # 重复的系统消息，跳过
                                continue
                            seen_content_hashes.add(content_hash)

            # 统计消息数量
            role = (event.get("role") or "").strip()
            is_message = role in {"user", "assistant"}
            if is_message:
                message_count += 1
                if not summary and text_for_index:
                    summary = text_for_index[:160]

            # 添加到全文搜索索引
            messages.append(
                (
                    source,
                    session_id,
                    project,
                    event["id"],
                    line_no,
                    role,
                    infer_legacy_type(role, blocks, fallback="event"),
                    preprocess_for_fts(text_for_index),  # 预处理中文文本
                    timestamp,
                )
            )

    if not started_at:
        started_at = normalize_timestamp(mtime)
    if not updated_at:
        updated_at = started_at

    # 插入会话元数据
    conn.execute(
        """
        INSERT INTO sessions (
            source,
            session_id,
            project,
            file_path,
            started_at,
            updated_at,
            message_count,
            summary,
            cwd,
            git_branch,
            parent_session_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source,
            session_id,
            project,
            path_str,
            started_at,
            updated_at,
            message_count,
            summary,
            cwd,
            git_branch,
            parent_session_id,
        ),
    )

    # 插入消息到全文搜索表
    if messages:
        conn.executemany(
            """
            INSERT INTO messages_fts (
                source, session_id, project, event_id, line_no, role, type, content, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            messages,
        )

    # 记录文件索引状态
    conn.execute(
        """
        INSERT INTO files (file_path, mtime, source, session_id, project)
        VALUES (?, ?, ?, ?, ?)
        """,
        (path_str, mtime, source, session_id, project),
    )


def ensure_index(conn: sqlite3.Connection, force: bool = False) -> None:
    """确保索引是最新的（增量更新，带节流）

    Args:
        conn: 数据库连接
        force: 是否强制刷新，跳过节流检查
    """
    global LAST_INDEX_TIME

    with INDEX_LOCK:
        # 索引节流：如果最近刚索引过，跳过（除非强制刷新）
        current_time = time.time()
        if not force and current_time - LAST_INDEX_TIME < INDEX_THROTTLE_SECONDS:
            return

        LAST_INDEX_TIME = current_time
        init_db(conn)

        # 获取已索引的文件
        existing = {
            row["file_path"]: row
            for row in conn.execute("SELECT * FROM files").fetchall()
        }

        # 扫描当前文件
        current_paths: set[str] = set()
        for source, project, path, parent_session_id in scan_session_files():
            path_str = str(path)
            mtime = path.stat().st_mtime
            current_paths.add(path_str)

            # 如果文件是新的或已更新，重新索引
            row = existing.get(path_str)
            if row is None or row["mtime"] < mtime:
                index_file(conn, source, project, path, parent_session_id)

        # 删除已不存在的文件记录
        for file_path, row in existing.items():
            if file_path not in current_paths:
                remove_file_records(conn, file_path)

        conn.commit()


def _set_index_status(**kwargs: Any) -> None:
    with INDEX_STATUS_LOCK:
        INDEX_STATUS.update(kwargs)


def get_index_status() -> Dict[str, Any]:
    with INDEX_STATUS_LOCK:
        return dict(INDEX_STATUS)


def _run_index_in_background(*, force: bool) -> None:
    """后台线程：执行增量索引并更新进度。"""
    try:
        # Keep indexing single-threaded inside this process (shared with ensure_index()).
        with INDEX_LOCK:
            conn = get_db()
            init_db(conn)

            # 获取已索引的文件
            existing = {
                row["file_path"]: row
                for row in conn.execute("SELECT * FROM files").fetchall()
            }

            # 预扫描以提供进度
            files = list(scan_session_files())
            _set_index_status(total_files=len(files), processed_files=0)

            current_paths: set[str] = set()
            processed = 0
            for source, project, path, parent_session_id in files:
                processed += 1
                try:
                    path_str = str(path)
                    mtime = path.stat().st_mtime
                    current_paths.add(path_str)
                    row = existing.get(path_str)
                    # force 仅用于跳过节流并立即扫描；不要无条件重建未变化的文件索引。
                    if row is None or row["mtime"] < mtime:
                        index_file(conn, source, project, path, parent_session_id)
                finally:
                    _set_index_status(processed_files=processed)

            # 删除已不存在的文件记录（force 也照做，保证一致）
            for file_path, row in existing.items():
                if file_path not in current_paths:
                    remove_file_records(conn, file_path)

            conn.commit()
            conn.close()
        _set_index_status(
            state="ready",
            finished_at=datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            last_error="",
        )
    except Exception as e:
        _set_index_status(
            state="error",
            finished_at=datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            last_error=str(e),
        )


def ensure_index_async(*, force: bool = False) -> None:
    """确保后台索引任务已启动（不会阻塞请求线程）。"""
    global LAST_INDEX_TIME, INDEX_BG_THREAD

    with INDEX_LOCK:
        status = get_index_status()
        if status.get("state") == "running" and INDEX_BG_THREAD and INDEX_BG_THREAD.is_alive():
            return
        # 失败状态不要被节流卡住
        if status.get("state") != "error":
            current_time = time.time()
            if not force and current_time - LAST_INDEX_TIME < INDEX_THROTTLE_SECONDS:
                return
            LAST_INDEX_TIME = current_time
        else:
            LAST_INDEX_TIME = time.time()

        _set_index_status(
            state="running",
            started_at=datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            finished_at="",
            total_files=0,
            processed_files=0,
            last_error="",
        )
        INDEX_BG_THREAD = threading.Thread(
            target=_run_index_in_background,
            kwargs={"force": force},
            daemon=True,
        )
        INDEX_BG_THREAD.start()


@contextmanager
def indexed_db_session(force_refresh: bool = False) -> Generator[sqlite3.Connection, None, None]:
    """数据库连接上下文管理器（确保索引最新）

    Args:
        force_refresh: 是否强制刷新索引，跳过节流检查
    """
    with db_session() as conn:
        # 先确保 schema 存在（避免首次启动时 queries 直接报错）
        init_db(conn)

        # 用户显式请求 force_refresh 时，直接在该请求内同步跑一遍增量索引：
        # - 语义更符合“刷新”
        # - 避免每次点击都走 503 轮询
        if force_refresh:
            _set_index_status(
                state="running",
                started_at=datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                finished_at="",
                total_files=0,
                processed_files=0,
                last_error="",
            )
            ensure_index(conn, force=True)
            _set_index_status(
                state="ready",
                finished_at=datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                last_error="",
            )
            yield conn
            return

        # 启动后台索引任务（避免阻塞 API 请求）
        ensure_index_async(force=False)

        # 如果数据库为空且索引仍在构建中，则返回 503 让前端轮询进度
        status = get_index_status()
        if status.get("state") == "running":
            has_any = conn.execute("SELECT 1 FROM sessions LIMIT 1").fetchone() is not None
            if not has_any:
                raise HTTPException(status_code=503, detail={"code": "indexing", "status": status})
        if status.get("state") == "error":
            has_any = conn.execute("SELECT 1 FROM sessions LIMIT 1").fetchone() is not None
            if not has_any:
                raise HTTPException(status_code=503, detail={"code": "index_error", "status": status})

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
            SELECT source,
                   project,
                   COUNT(*) AS session_count,
                   MAX(updated_at) AS updated_at
            FROM sessions
            GROUP BY source, project
            ORDER BY source, updated_at DESC
            """
        ).fetchall()
        return {"projects": [dict(row) for row in rows]}


@app.get("/api/index/status")
def index_status() -> Dict[str, Any]:
    """获取后台索引状态（用于前端进度显示）"""
    return {"status": get_index_status()}


@app.get("/api/projects/{project}/sessions")
def list_sessions(
    project: str,
    source: str = Query(DEFAULT_SOURCE),
    force_refresh: bool = Query(False)
) -> Dict[str, Any]:
    """列出项目的所有会话

    Args:
        project: 项目名称
        source: 数据源（claude_code 或 codex）
        force_refresh: 是否强制刷新索引（跳过30秒节流限制）
    """
    with indexed_db_session(force_refresh=force_refresh) as conn:
        rows = conn.execute(
            """
            SELECT session_id,
                   started_at,
                   updated_at,
                   message_count,
                   summary,
                   cwd,
                   git_branch,
                   parent_session_id
            FROM sessions
            WHERE source = ? AND project = ?
            ORDER BY updated_at DESC
            """,
            (source, project),
        ).fetchall()
        return {"source": source, "project": project, "sessions": [dict(row) for row in rows]}


@app.get("/api/sessions/{session_id}/subagents")
def list_subagents(
    session_id: str,
    project: Optional[str] = None,
    source: str = Query(DEFAULT_SOURCE),
) -> Dict[str, Any]:
    """列出会话的所有子会话"""
    with indexed_db_session() as conn:
        # 首先验证父会话存在
        parent_row = resolve_session_row(conn, session_id, project, source)
        if not parent_row:
            raise HTTPException(status_code=404, detail="Parent session not found")

        resolved_source = parent_row["source"]
        resolved_project = parent_row["project"]

        # 查询子会话
        rows = conn.execute(
            """
            SELECT session_id,
                   started_at,
                   updated_at,
                   message_count,
                   summary,
                   cwd,
                   git_branch,
                   parent_session_id
            FROM sessions
            WHERE source = ? AND project = ? AND parent_session_id = ?
            ORDER BY updated_at DESC
            """,
            (resolved_source, resolved_project, session_id),
        ).fetchall()

        return {
            "source": resolved_source,
            "project": resolved_project,
            "parent_session_id": session_id,
            "subagents": [dict(row) for row in rows],
        }


def resolve_session_row(
    conn: sqlite3.Connection,
    session_id: str,
    project: Optional[str],
    source: Optional[str],
) -> Optional[sqlite3.Row]:
    """解析会话行（含 file_path/source/project）"""
    query = "SELECT source, project, file_path FROM sessions WHERE session_id = ?"
    params: List[Any] = [session_id]
    if source:
        query += " AND source = ?"
        params.append(source)
    if project:
        query += " AND project = ?"
        params.append(project)
    query += " ORDER BY updated_at DESC LIMIT 1"
    return conn.execute(query, params).fetchone()


@app.get("/api/sessions/{session_id}")
def get_session(
    session_id: str,
    project: Optional[str] = None,
    source: str = Query(DEFAULT_SOURCE),
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
) -> Dict[str, Any]:
    """获取会话详情（支持分页）"""
    with indexed_db_session() as conn:
        row = resolve_session_row(conn, session_id, project, source)
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        file_path = row["file_path"]
        resolved_source = row["source"]
        resolved_project = row["project"]

        items: List[Dict[str, Any]] = []
        has_more = False
        lines_processed = 0

        # 用于去重：维护最近几条消息的事件 ID（滑动窗口）
        recent_event_ids: List[str] = []
        window_size = 50  # 窗口大小：检查最近 50 条消息

        # 用于基于内容的去重（针对系统消息）
        seen_content_hashes: set[str] = set()

        with Path(file_path).open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if lines_processed < offset:
                    lines_processed += 1
                    continue

                if len(items) >= limit:
                    has_more = True
                    break

                lines_processed += 1

                event = parse_event_line(line, session_id, resolved_source)
                if not event:
                    continue

                # 提取 blocks 和 content
                blocks = event.get("blocks") or []
                legacy_content = event.get("content") or blocks_to_json_string(blocks)

                # 只返回有内容的事件（过滤系统事件如 turn_context, session_meta）
                if not blocks or not legacy_content:
                    continue

                # Codex 去重：过滤 event_msg 类型的 user_message 和 agent_message
                # 因为它们的内容已经在对应的 response_item 事件中了
                if resolved_source == "codex":
                    meta = event.get("meta") or {}
                    event_type = meta.get("event_type", "")
                    payload_type = meta.get("payload_type", "")
                    if event_type == "event_msg" and payload_type in ("user_message", "agent_message"):
                        continue

                # 去重：使用稳定的事件 ID
                event_id = event["id"]
                if event_id in recent_event_ids:
                    # 重复消息，跳过
                    continue

                # 基于内容的去重（针对 Codex 系统消息）
                # 系统消息（如 AGENTS.md, environment_context）在每个 turn 中都会重复
                if resolved_source == "codex":
                    role = event.get("role", "")
                    # 检测系统消息：role 是 user/developer 且内容以特定标记开头
                    if role in ("user", "developer") and blocks:
                        first_block = blocks[0]
                        if isinstance(first_block, dict) and first_block.get("type") == "text":
                            text = first_block.get("text", "")
                            # 系统消息特征：以 <permissions、# AGENTS.md、<environment_context 开头
                            if text.startswith(("<permissions", "# AGENTS.md", "<environment_context")):
                                # 计算内容哈希
                                content_hash = hashlib.blake2b(
                                    text.encode("utf-8", "replace"), digest_size=16
                                ).hexdigest()
                                if content_hash in seen_content_hashes:
                                    # 重复的系统消息，跳过
                                    continue
                                seen_content_hashes.add(content_hash)

                # 添加到窗口
                recent_event_ids.append(event_id)
                # 保持窗口大小
                if len(recent_event_ids) > window_size:
                    recent_event_ids.pop(0)

                # 返回新旧两种格式的字段
                items.append({
                    # 新的统一字段（推荐前端使用）
                    "id": event["id"],
                    "source": resolved_source,
                    "session_id": event["session_id"],
                    "timestamp": event["timestamp"],
                    "role": event["role"],
                    "blocks": blocks,
                    "meta": event.get("meta") or {},

                    # 旧的向后兼容字段
                    "type": event.get("type") or infer_legacy_type(event.get("role") or "", blocks),
                    "content": legacy_content,
                    "sessionId": event["session_id"],
                    "cwd": event.get("cwd") or (event.get("meta") or {}).get("cwd", ""),
                    "gitBranch": event.get("git_branch") or (event.get("meta") or {}).get("git_branch", ""),
                })

            if not has_more:
                has_more = any(line.strip() for line in handle)

        next_offset = lines_processed if has_more else None
        return {
            "session_id": session_id,
            "source": resolved_source,
            "project": resolved_project,
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset,
            "items": items,
        }


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    project: Optional[str] = None,
    source: Optional[str] = None,
    session_id: Optional[str] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    """全文搜索"""
    query = preprocess_for_fts(q.strip()).strip()
    with indexed_db_session() as conn:
        params: List[Any] = [query]
        where_clauses = ["messages_fts MATCH ?"]
        if project:
            where_clauses.append("project = ?")
            params.append(project)
        if source:
            where_clauses.append("source = ?")
            params.append(source)
        if session_id:
            where_clauses.append("session_id = ?")
            params.append(session_id)
        where = " AND ".join(where_clauses)

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
                       source,
                       project,
                       event_id,
                       line_no,
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
