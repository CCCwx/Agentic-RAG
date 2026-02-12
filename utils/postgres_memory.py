"""
基于 PostgreSQL 的长期记忆：按 thread_id 存储/读取对话，供多轮与跨会话使用。
使用前：pip install psycopg2-binary，在 .env 中设置 DATABASE_URL。
"""
import os
from typing import List
from utils.logger_handler import logger

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_memory (
    id BIGSERIAL PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL,
    role VARCHAR(32) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_conversation_thread_id ON conversation_memory(thread_id);
CREATE INDEX IF NOT EXISTS idx_conversation_created_at ON conversation_memory(thread_id, created_at DESC);
"""


def _get_connection():
    import psycopg2
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set; cannot use PostgreSQL memory.")
    return psycopg2.connect(url)


def create_tables_if_not_exists() -> None:
    """创建 conversation_memory 表（若不存在）。"""
    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"[postgres_memory] create_tables failed: {e}")


def add_message(thread_id: str, role: str, content: str) -> None:
    """写入一条对话消息。"""
    if not (thread_id and role and content is not None):
        return
    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversation_memory (thread_id, role, content) VALUES (%s, %s, %s)",
                    (thread_id[:255], role[:32], content),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"[postgres_memory] add_message failed: {e}")


def get_recent_messages(thread_id: str, limit: int = 50) -> List[dict]:
    """按时间倒序取最近 limit 条消息，返回 [{role, content}, ...]（旧消息在前）。"""
    if not thread_id:
        return []
    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content FROM conversation_memory
                    WHERE thread_id = %s
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (thread_id, limit),
                )
                rows = cur.fetchall()
            return [{"role": r[0], "content": r[1] or ""} for r in rows]
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"[postgres_memory] get_recent_messages failed: {e}")
        return []


def is_available() -> bool:
    """是否已配置并可连接 PostgreSQL。"""
    if not os.environ.get("DATABASE_URL"):
        return False
    try:
        conn = _get_connection()
        conn.close()
        return True
    except Exception:
        return False
