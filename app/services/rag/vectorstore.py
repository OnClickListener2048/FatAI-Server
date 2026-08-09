"""向量存储与检索。

持有到 fat_ai.db 的专用 sqlite3 连接(不走 SQLAlchemy —— aiosqlite 的
load_extension 是协程,无法在 connect 事件里同步加载)。所有操作经
asyncio.to_thread 在连接所属线程内完成;每次操作打开独立连接,
WAL + busy_timeout 与主引擎共存。

两张表:
- document_chunks: chunk 元数据(作用域、来源、内容),所有模式下都需要
- chunks_vec0: sqlite-vec vec0 虚拟表,ANN 检索;扩展不可用时退化为
  BLOB + Python 余弦扫描(与 scripts/rag_demo.py 相同的朴素实现)
"""

import asyncio
import hashlib
import re
import sqlite3
import struct
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

_SQLITE_URL_PATTERN = re.compile(r"^sqlite(?:\+\w+)?:///(.*)$")

CHUNK_COLUMNS = (
    "id",
    "user_id",
    "workspace_id",
    "conversation_id",
    "source_type",
    "source_id",
    "chunk_index",
    "path",
    "title",
    "scope",
    "content",
    "content_hash",
)


@dataclass
class ChunkRecord:
    id: str
    user_id: str
    workspace_id: str | None
    conversation_id: str | None
    source_type: str  # memory | knowledge_document
    source_id: str
    chunk_index: int
    path: str
    title: str
    scope: str  # GLOBAL | WORKSPACE | CONVERSATION | KNOWLEDGE
    content: str
    content_hash: str


@dataclass
class ScoredChunk:
    chunk: ChunkRecord
    score: float


def sqlite_path_from_url(database_url: str) -> str | None:
    """从 SQLAlchemy 连接串提取 SQLite 文件路径;非 SQLite 返回 None。"""
    match = _SQLITE_URL_PATTERN.match(database_url)
    if not match:
        return None
    path = match.group(1)
    return str(Path(path).resolve())


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_embedding(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def unpack_embedding(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


class VectorStore:
    """chunk 元数据 + 向量检索的单一入口,接口对上层屏蔽 vec0/扫描差异。"""

    def __init__(self, database_url: str, dimensions: int) -> None:
        self._path = sqlite_path_from_url(database_url)
        self._dimensions = dimensions
        self._vec_available = False
        self._enabled = self._path is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def vec_available(self) -> bool:
        return self._vec_available

    def initialize(self) -> None:
        """建表并探测 sqlite-vec 是否可用。启动时调用一次。"""
        if not self._enabled:
            return
        with self._opened() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    workspace_id TEXT,
                    conversation_id TEXT,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    path TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    scope TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding BLOB
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON document_chunks (source_type, source_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_user ON document_chunks (user_id);
                """
            )
            try:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec0 USING vec0(
                        chunk_id TEXT PRIMARY KEY,
                        embedding float[{self._dimensions}],
                        user_id TEXT,
                        workspace_id TEXT
                    )
                    """
                )
                self._vec_available = True
            except Exception:
                # 扩展不可用(平台不支持/未安装)时退化为 BLOB 余弦扫描
                self._vec_available = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        # SQLite 扩展按连接生效: 仅 initialize() 时加载, 后续新连接会
        # 报 "no such module: vec0"。每个连接都要重新加载。
        if self._vec_available:
            try:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
            except Exception:
                # 扩展本次加载失败(极少见) → 本操作及后续退回扫描模式
                self._vec_available = False
        return conn

    @contextmanager
    def _opened(self):
        """连接 + 事务上下文: with 退出时提交/回滚, 且总是关闭连接。"""
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 写路径
    # ------------------------------------------------------------------
    async def replace_source(
        self,
        source_type: str,
        source_id: str,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        """删除某来源的全部旧 chunk 后整体写入(幂等重建)。"""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not self._enabled:
            return
        await asyncio.to_thread(self._replace_source_sync, source_type, source_id, chunks, embeddings)

    def _replace_source_sync(
        self,
        source_type: str,
        source_id: str,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        with self._opened() as conn:
            # 先删 vec0 再删 document_chunks: 子查询依赖 document_chunks 仍持有旧行
            if self._vec_available:
                conn.execute("DELETE FROM chunks_vec0 WHERE chunk_id IN (SELECT id FROM document_chunks WHERE source_type = ? AND source_id = ?)", (source_type, source_id))
            conn.execute("DELETE FROM document_chunks WHERE source_type = ? AND source_id = ?", (source_type, source_id))
            for chunk, vector in zip(chunks, embeddings):
                values = asdict(chunk)
                columns = ", ".join(CHUNK_COLUMNS)
                placeholders = ", ".join("?" for _ in CHUNK_COLUMNS)
                conn.execute(
                    f"INSERT INTO document_chunks ({columns}) VALUES ({placeholders})",
                    tuple(values[column] for column in CHUNK_COLUMNS),
                )
                if self._vec_available:
                    # vec0 的 metadata 列不接受 NULL: GLOBAL 记忆无 workspace,
                    # 归一为空串存储(作用域过滤在上层 Python 侧完成)。
                    conn.execute(
                        "INSERT INTO chunks_vec0 (chunk_id, embedding, user_id, workspace_id) VALUES (?, ?, ?, ?)",
                        (chunk.id, pack_embedding(vector), chunk.user_id, chunk.workspace_id or ""),
                    )
                else:
                    conn.execute(
                        "UPDATE document_chunks SET embedding = ? WHERE id = ?",
                        (pack_embedding(vector), chunk.id),
                    )

    async def delete_by_source(self, source_type: str, source_id: str) -> None:
        if not self._enabled:
            return
        await asyncio.to_thread(self._delete_by_source_sync, source_type, source_id)

    def _delete_by_source_sync(self, source_type: str, source_id: str) -> None:
        with self._opened() as conn:
            if self._vec_available:
                conn.execute(
                    "DELETE FROM chunks_vec0 WHERE chunk_id IN (SELECT id FROM document_chunks WHERE source_type = ? AND source_id = ?)",
                    (source_type, source_id),
                )
            conn.execute("DELETE FROM document_chunks WHERE source_type = ? AND source_id = ?", (source_type, source_id))

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    async def search(self, query_vec: list[float], user_id: str, k: int) -> list[ScoredChunk]:
        """按用户取相似度最高的 k 个 chunk(不做作用域过滤,由上层处理)。"""
        if not self._enabled:
            return []
        return await asyncio.to_thread(self._search_sync, query_vec, user_id, k)

    def _search_sync(self, query_vec: list[float], user_id: str, k: int) -> list[ScoredChunk]:
        with self._opened() as conn:
            if self._vec_available:
                rows = conn.execute(
                    "SELECT chunk_id, distance FROM chunks_vec0 WHERE embedding MATCH ? AND user_id = ? AND k = ?",
                    (pack_embedding(query_vec), user_id, k),
                ).fetchall()
                # vec0 默认 L2 距离; 向量已归一化, 故 L2² = 2(1 - cos),
                # 余弦相似度 = 1 - L2²/2 —— 不是 1 - L2
                scores = {chunk_id: 1.0 - (distance * distance) / 2.0 for chunk_id, distance in rows}
            else:
                candidates = conn.execute(
                    "SELECT id, embedding FROM document_chunks WHERE user_id = ? AND embedding IS NOT NULL",
                    (user_id,),
                ).fetchall()
                scores = {
                    chunk_id: cosine_similarity(query_vec, unpack_embedding(blob))
                    for chunk_id, blob in candidates
                    if blob is not None
                }
            if not scores:
                return []
            chunks = self._fetch_chunks(conn, list(scores.keys()))
            return sorted(
                (ScoredChunk(chunk=chunks[chunk_id], score=scores[chunk_id]) for chunk_id in scores if chunk_id in chunks),
                key=lambda item: item.score,
                reverse=True,
            )

    def _fetch_chunks(self, conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}
        placeholders = ", ".join("?" for _ in chunk_ids)
        rows = conn.execute(f"SELECT {', '.join(CHUNK_COLUMNS)} FROM document_chunks WHERE id IN ({placeholders})", chunk_ids).fetchall()
        return {row[0]: ChunkRecord(**dict(zip(CHUNK_COLUMNS, row))) for row in rows}

    async def count(self) -> int:
        if not self._enabled:
            return 0
        return await asyncio.to_thread(self._count_sync)

    def _count_sync(self) -> int:
        with self._opened() as conn:
            return conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]

    async def source_chunk_count(self, source_type: str, source_id: str) -> int:
        if not self._enabled:
            return 0
        return await asyncio.to_thread(self._source_chunk_count_sync, source_type, source_id)

    def _source_chunk_count_sync(self, source_type: str, source_id: str) -> int:
        with self._opened() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE source_type = ? AND source_id = ?",
                (source_type, source_id),
            ).fetchone()
            return int(row[0]) if row else 0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
