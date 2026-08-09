import asyncio
import tempfile
import unittest
from pathlib import Path

# 模块级临时目录仅供 VectorStoreTest/RetrievalScopeTest 共享向量库使用;
# 不在这里做 cleanup —— 由进程退出时的 atexit 兜底,避免类名排序
# (ContextRetrievalTest 排在 VectorStoreTest 之前)导致的提前删除。
_temp_directory = tempfile.TemporaryDirectory()
_VECTOR_DB_URL = f"sqlite+aiosqlite:///{(Path(_temp_directory.name) / 'rag_vec.db').as_posix()}"

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

from app.db import Base, MemoryEntry, User  # noqa: E402
from app.models import ChatMessageInput  # noqa: E402
from app.services.context import assemble_context  # noqa: E402
from app.services.rag.chunking import chunk_markdown, chunk_text  # noqa: E402
from app.services.rag.retrieval import RetrievalHit, RetrievalResult, RetrievalService  # noqa: E402
from app.services.rag.vectorstore import ChunkRecord, VectorStore, content_hash  # noqa: E402

QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class _FixedEmbedder:
    """固定向量嵌入器: 任何输入都返回 QUERY_VECTOR,检索分数恒为 1.0。"""

    async def embed_one(self, _text: str) -> list[float]:
        return list(QUERY_VECTOR)


def _chunk(
    user_id: str,
    source_type: str,
    source_id: str,
    content: str,
    scope: str,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
    title: str = "",
    index: int = 0,
) -> ChunkRecord:
    return ChunkRecord(
        id=f"{source_type}-{source_id}-{index}",
        user_id=user_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        source_type=source_type,
        source_id=source_id,
        chunk_index=index,
        path="",
        title=title,
        scope=scope,
        content=content,
        content_hash=content_hash(content),
    )


class ChunkerTest(unittest.TestCase):
    def test_heading_path_prefixes_chunks(self) -> None:
        md = "# 记忆系统\n\n记忆系统介绍。\n\n## 记忆提取\n\n提取协议内容。"
        chunks = chunk_markdown(md)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["path"], "记忆系统")
        self.assertIn("记忆系统介绍", chunks[0]["content"])
        self.assertEqual(chunks[1]["path"], "记忆系统 > 记忆提取")

    def test_long_section_split_with_overlap(self) -> None:
        text = "。".join(f"句子{i}" for i in range(200))
        chunks = chunk_text(text, max_chars=200, overlap_chars=40)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200 + 40)
        # 重叠区应包含上一块尾部的内容(句子边界截断,不完全相等但存在共享片段)
        self.assertNotEqual(chunks[0], chunks[1])
        self.assertIn(chunks[1][:10], text)

    def test_short_sections_merged(self) -> None:
        md = "# 标题\n\n短句。\n\n另一短句。"
        chunks = chunk_markdown(md, max_chars=800)
        self.assertEqual(len(chunks), 1)
        self.assertIn("短句", chunks[0]["content"])
        self.assertIn("另一短句", chunks[0]["content"])


class VectorStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = VectorStore(_VECTOR_DB_URL, dimensions=8)
        self.store.initialize()

    def tearDown(self) -> None:
        for source_id in ("mem-1", "mem-2", "mem-x", "mem-ortho", "mem-half"):
            asyncio.run(self.store.delete_by_source("memory", source_id))

    async def _seed(self) -> None:
        await self.store.replace_source(
            "memory",
            "mem-1",
            [_chunk("u1", "memory", "mem-1", "User prefers Chinese.", "GLOBAL")],
            [list(QUERY_VECTOR)],
        )
        await self.store.replace_source(
            "memory",
            "mem-2",
            [_chunk("u1", "memory", "mem-2", "User works at X.", "WORKSPACE", workspace_id="ws1")],
            [list(QUERY_VECTOR)],
        )

    def test_replace_and_search_top_k(self) -> None:
        async def run() -> None:
            await self._seed()
            results = await self.store.search(QUERY_VECTOR, "u1", k=5)
            self.assertEqual(len(results), 2)
            # 固定向量下分数相同,vec0 kNN 返回顺序不保证 —— 按集合断言
            self.assertEqual({r.chunk.source_id for r in results}, {"mem-1", "mem-2"})
            self.assertGreater(results[0].score, 0.99)
            # 重建幂等: 同一来源替换后行数不变
            await self.store.replace_source(
                "memory",
                "mem-1",
                [_chunk("u1", "memory", "mem-1", "Updated memory.", "GLOBAL")],
                [list(QUERY_VECTOR)],
            )
            self.assertEqual(await self.store.count(), 2)

        asyncio.run(run())

    def test_user_isolation_and_delete(self) -> None:
        async def run() -> None:
            await self._seed()
            other = [_chunk("u2", "memory", "mem-x", "Other user secret.", "GLOBAL")]
            await self.store.replace_source("memory", "mem-x", other, [list(QUERY_VECTOR)])
            results = await self.store.search(QUERY_VECTOR, "u1", k=10)
            self.assertEqual({r.chunk.source_id for r in results}, {"mem-1", "mem-2"})
            await self.store.delete_by_source("memory", "mem-1")
            self.assertEqual(await self.store.count(), 2)

        asyncio.run(run())

    def test_l2_distance_to_cosine_conversion(self) -> None:
        """vec0 返回 L2 距离, 换算为余弦: cos = 1 - L2²/2(向量已归一化)。"""
        async def run() -> None:
            ortho = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # 单位向量(范数必须为 1): 与 QUERY_VECTOR 的点积 = sqrt(0.5)
            half = [0.5 ** 0.5, 0.5 ** 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            await self.store.replace_source(
                "memory",
                "mem-ortho",
                [_chunk("u1", "memory", "mem-ortho", "Orthogonal memory.", "GLOBAL")],
                [ortho],
            )
            await self.store.replace_source(
                "memory",
                "mem-half",
                [_chunk("u1", "memory", "mem-half", "Half cosine memory.", "GLOBAL")],
                [half],
            )
            results = await self.store.search(QUERY_VECTOR, "u1", k=5)
            by_id = {r.chunk.source_id: r.score for r in results}
            self.assertAlmostEqual(by_id["mem-ortho"], 0.0, places=3)
            self.assertAlmostEqual(by_id["mem-half"], 0.5 ** 0.5, places=3)

        asyncio.run(run())

    def test_scan_fallback_mode(self) -> None:
        async def run() -> None:
            self.store._vec_available = False  # 模拟 sqlite-vec 不可用
            await self._seed()
            results = await self.store.search(QUERY_VECTOR, "u1", k=5)
            self.assertEqual(len(results), 2)
            self.assertGreater(results[0].score, 0.99)

        asyncio.run(run())

    def test_disabled_for_postgres_url(self) -> None:
        store = VectorStore("postgresql+asyncpg://user@host/db", dimensions=8)
        self.assertFalse(store.enabled)
        self.assertEqual(asyncio.run(store.search(QUERY_VECTOR, "u1", 5)), [])


class RetrievalScopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = VectorStore(_VECTOR_DB_URL, dimensions=8)
        self.store.initialize()
        self.service = RetrievalService(_FixedEmbedder(), self.store, min_score=0.5)

    def tearDown(self) -> None:
        for source_type, source_id in (
            ("memory", "g"),
            ("memory", "w1"),
            ("memory", "w2"),
            ("memory", "c"),
            ("knowledge_document", "doc1"),
            ("knowledge_document", "doc2"),
        ):
            asyncio.run(self.store.delete_by_source(source_type, source_id))

    async def _seed(self) -> None:
        seeds = [
            _chunk("u1", "memory", "g", "Global memory", "GLOBAL"),
            _chunk("u1", "memory", "w1", "Workspace memory", "WORKSPACE", workspace_id="ws1"),
            _chunk("u1", "memory", "w2", "Foreign workspace memory", "WORKSPACE", workspace_id="ws2"),
            _chunk("u1", "memory", "c", "Conversation memory", "CONVERSATION", workspace_id="ws1", conversation_id="conv1"),
            _chunk("u1", "knowledge_document", "doc1", "Knowledge doc in ws1", "KNOWLEDGE", workspace_id="ws1", title="doc1.md"),
            _chunk("u1", "knowledge_document", "doc2", "Knowledge doc in ws2", "KNOWLEDGE", workspace_id="ws2", title="doc2.md"),
        ]
        for chunk in seeds:
            await self.store.replace_source(
                chunk.source_type,
                chunk.source_id,
                [chunk],
                [list(QUERY_VECTOR)],
            )

    def test_scope_filtering(self) -> None:
        async def run() -> None:
            await self._seed()
            result = await self.service.search("query", "u1", "ws1", "conv1")
            self.assertEqual({hit.source_id for hit in result.memories}, {"g", "w1", "c"})
            self.assertEqual([hit.source_id for hit in result.documents], ["doc1"])
            self.assertEqual(
                sorted(result.sources, key=lambda source: (source["kind"], source["id"])),
                [
                    {"title": "doc1.md", "kind": "knowledge_document", "id": "doc1"},
                    {"title": "", "kind": "memory", "id": "c"},
                    {"title": "", "kind": "memory", "id": "g"},
                    {"title": "", "kind": "memory", "id": "w1"},
                ],
            )

        asyncio.run(run())

    def test_workspace_only_when_id_missing(self) -> None:
        async def run() -> None:
            await self._seed()
            result = await self.service.search("query", "u1", None, None)
            self.assertEqual([hit.source_id for hit in result.memories], ["g"])
            self.assertEqual(result.documents, [])

        asyncio.run(run())


class ContextRetrievalTest(unittest.TestCase):
    """独立于其他测试模块: 自建 engine + 建表, 不依赖模块级全局 engine。

    全局 `app.db.engine` 在 unittest discover 时由 test_model_configurations
    先 import 创建, 绑定的是另一个临时库 —— 复用它会拿到别人已清理的库。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        db_url = f"sqlite+aiosqlite:///{(Path(cls._directory.name) / 'context.db').as_posix()}"
        cls._engine = create_async_engine(db_url, future=True)
        asyncio.run(cls._create_tables())
        cls._session_factory = async_sessionmaker(cls._engine, expire_on_commit=False, class_=AsyncSession)

    @classmethod
    async def _create_tables(cls) -> None:
        async with cls._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    @classmethod
    def tearDownClass(cls) -> None:
        asyncio.run(cls._engine.dispose())
        cls._directory.cleanup()

    async def _seed_user(self, session) -> User:
        # 同一 engine 被本类多个用例共享,email 唯一约束要求按用例隔离
        user = User(
            id=f"rag-context-{self._testMethodName}",
            email=f"rag-{self._testMethodName}@device.fatai.local",
            display_name="RAG",
            password_hash="x",
        )
        session.add(user)
        await session.flush()
        return user

    def test_rag_blocks_injected_with_sources(self) -> None:
        async def run() -> None:
            async with self._session_factory() as session:
                user = await self._seed_user(session)
                await session.commit()

                class FakeRetriever:
                    async def search(self, *_args):
                        return RetrievalResult(
                            memories=[RetrievalHit("User prefers Chinese.", 0.9, "memory", "m1", "Fact memory", "")],
                            documents=[RetrievalHit("Koin 文档内容", 0.8, "knowledge_document", "d1", "arch.md", "架构")],
                        )

                messages, sources = await assemble_context(
                    session,
                    user,
                    "ws-1",
                    None,
                    [ChatMessageInput(role="user", content="怎么用 Koin?")],
                    "zh",
                    retriever=FakeRetriever(),
                )
                combined = "\n".join(message.content for message in messages)
                self.assertIn("Relevant memory reference", combined)
                self.assertIn("User prefers Chinese.", combined)
                self.assertIn("Retrieved knowledge reference", combined)
                self.assertIn("Koin 文档内容", combined)
                self.assertEqual(len(sources), 2)
                self.assertEqual(sources[0]["kind"], "memory")

        asyncio.run(run())

    def test_retrieval_failure_falls_back_to_recency(self) -> None:
        async def run() -> None:
            async with self._session_factory() as session:
                user = await self._seed_user(session)
                session.add(
                    MemoryEntry(
                        id="recency-mem",
                        user_id=user.id,
                        workspace_id="ws-1",
                        scope="WORKSPACE",
                        kind="FACT",
                        content="Recency fallback memory.",
                        is_archived=False,
                    )
                )
                await session.commit()

                class BrokenRetriever:
                    async def search(self, *_args):
                        raise RuntimeError("embedding down")

                messages, sources = await assemble_context(
                    session,
                    user,
                    "ws-1",
                    None,
                    [ChatMessageInput(role="user", content="Hello")],
                    "en",
                    retriever=BrokenRetriever(),
                )
                combined = "\n".join(message.content for message in messages)
                self.assertIn("Recency fallback memory.", combined)
                self.assertEqual(sources, [])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
