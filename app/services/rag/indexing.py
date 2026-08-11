"""索引服务: 把记忆与知识文档变成可检索的 chunk。

触发路径:
- 记忆: 客户端 sync upsert / REST 写操作提交后由路由 spawn 后台任务调用
  (archive/delete 同样走这里 —— 归档即删除其 chunk)
- 知识文档: 后台 worker 轮询 QUEUED, CAS 置 PROCESSING 后执行 docling 解析
  -> 分块 -> embedding -> 落库,完成置 READY,失败置 FAILED(不阻塞聊天流)
- 冷启动: backfill() 补索引已存在但无 chunk 的记忆
"""

import asyncio
import logging
import uuid
from pathlib import Path

from sqlalchemy import select

from app.db import FileAsset, KnowledgeDocument, MemoryEntry, SessionLocal
from app.services.documents import DoclingDocumentService
from app.services.rag.chunking import chunk_markdown, chunk_text
from app.services.rag.embedding import EmbeddingService
from app.services.rag.vectorstore import ChunkRecord, VectorStore, content_hash

logger = logging.getLogger("fatai.rag")

EMBED_BATCH = 8  # CPU 推理时一次嵌入条数


class IndexingService:
    def __init__(
        self,
        embedder: EmbeddingService,
        store: VectorStore,
        documents: DoclingDocumentService,
        chunk_chars: int = 800,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._documents = documents
        self._chunk_chars = chunk_chars

    # ------------------------------------------------------------------
    # 记忆
    # ------------------------------------------------------------------
    async def index_memory(self, memory_id: str) -> None:
        async with SessionLocal() as session:
            memory = await session.get(MemoryEntry, memory_id)
            if memory is None:
                return
            if memory.is_archived:
                await self._store.delete_by_source("memory", memory.id)
                return
            chunks = [
                {"path": "", "content": piece}
                for piece in chunk_text(memory.content, self._chunk_chars)
            ]
        records, vectors = await self._embed_chunks("memory", memory, chunks)
        await self._store.replace_source("memory", memory.id, records, vectors)
        logger.info("[RAG] indexed memory %s (%d chunks)", memory.id, len(records))

    async def delete_memory(self, memory_id: str) -> None:
        await self._store.delete_by_source("memory", memory_id)

    async def backfill_memories(self) -> None:
        """冷启动补索引: 已有但无 chunk 的记忆。失败仅记日志,不阻断启动。"""
        try:
            async with SessionLocal() as session:
                memory_ids = list(
                    await session.scalars(
                        select(MemoryEntry.id).where(MemoryEntry.is_archived.is_(False))
                    )
                )
            for memory_id in memory_ids:
                if await self._store.source_chunk_count("memory", memory_id) == 0:
                    await self.index_memory(memory_id)
        except Exception:
            logger.exception("[RAG] memory backfill failed")

    # ------------------------------------------------------------------
    # 知识文档
    # ------------------------------------------------------------------
    async def process_pending_documents(self) -> None:
        """处理所有 QUEUED 的知识文档(CAS 防多进程重复处理)。"""
        async with SessionLocal() as session:
            pending = list(
                await session.scalars(
                    select(KnowledgeDocument)
                    .where(KnowledgeDocument.status == "QUEUED")
                    .limit(5)
                )
            )
            for document in pending:
                claim = await session.execute(
                    KnowledgeDocument.__table__.update()
                    .where(KnowledgeDocument.id == document.id, KnowledgeDocument.status == "QUEUED")
                    .values(status="PROCESSING")
                )
                if claim.rowcount != 1:
                    continue
                await session.commit()
                await self.index_knowledge_document(document.id)

    async def index_knowledge_document(self, document_id: str) -> None:
        try:
            async with SessionLocal() as session:
                document = await session.get(KnowledgeDocument, document_id)
                if document is None:
                    return
                asset = await session.get(FileAsset, document.file_asset_id)
                if asset is None:
                    await self._mark_failed(document, "File asset missing.")
                    return
                path = Path(asset.storage_path)
                if not path.is_file():
                    await self._mark_failed(document, f"Stored file not found: {asset.storage_path}")
                    return
                content_bytes = path.read_bytes()
                display_name, mime_type = asset.display_name, asset.mime_type

            result = await self._documents.read(display_name, mime_type, content_bytes)
            markdown = result.markdown
            # 语义分块内部会调用 embedder 做句子级嵌入
            chunks = await chunk_markdown(markdown, self._embedder, self._chunk_chars)

            async with SessionLocal() as session:
                document = await session.get(KnowledgeDocument, document_id)
                if document is None:
                    return
                document.extracted_markdown = markdown
                document.status = "PROCESSING"
                await session.commit()

            records, vectors = await self._embed_chunks("knowledge_document", asset, chunks)
            await self._store.replace_source("knowledge_document", document_id, records, vectors)

            async with SessionLocal() as session:
                document = await session.get(KnowledgeDocument, document_id)
                if document is not None:
                    document.status = "READY"
                    await session.commit()
            logger.info("[RAG] indexed knowledge document %s (%d chunks)", document_id, len(records))
        except Exception as error:  # noqa: BLE001 —— 索引失败不得影响聊天流
            logger.exception("[RAG] knowledge document %s indexing failed", document_id)
            await self._mark_failed(await self._get_document(document_id), str(error)[:500])

    async def _mark_failed(self, document: KnowledgeDocument | None, message: str) -> None:
        if document is None:
            return
        async with SessionLocal() as session:
            record = await session.get(KnowledgeDocument, document.id)
            if record is not None and record.status == "PROCESSING":
                record.status = "FAILED"
                await session.commit()

    async def _get_document(self, document_id: str) -> KnowledgeDocument | None:
        async with SessionLocal() as session:
            return await session.get(KnowledgeDocument, document_id)

    # ------------------------------------------------------------------
    # 公共: 分块 + 批量嵌入
    # ------------------------------------------------------------------
    async def _embed_chunks(
        self,
        source_type: str,
        source,
        chunks: list[dict],
    ) -> tuple[list[ChunkRecord], list[list[float]]]:
        """把 chunks 嵌入并组装 ChunkRecord。source 需含 user_id/workspace_id/scope 相关字段。"""
        workspace_id = getattr(source, "workspace_id", None)
        conversation_id = getattr(source, "conversation_id", None)
        scope = getattr(source, "scope", "KNOWLEDGE")
        title = getattr(source, "display_name", "") or getattr(source, "kind", "") or ""
        if source_type == "memory" and not title:
            kind = getattr(source, "kind", "")
            title = {"FACT": "Fact memory", "SUMMARY": "Conversation summary"}.get(kind, "Memory")

        vectors: list[list[float]] = []
        for index in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[index : index + EMBED_BATCH]
            vectors.extend(await self._embedder.embed([chunk["content"] for chunk in batch]))

        records = [
            ChunkRecord(
                id=str(uuid.uuid4()),
                user_id=source.user_id,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                source_type=source_type,
                source_id=str(source.id),
                chunk_index=index,
                path=chunk["path"],
                title=title,
                scope=scope,
                content=chunk["content"],
                content_hash=content_hash(chunk["content"]),
            )
            for index, (chunk, _vector) in enumerate(zip(chunks, vectors))
        ]
        return records, vectors


async def knowledge_document_worker(indexer: IndexingService, sweep_seconds: float) -> None:
    """周期轮询 QUEUED 知识文档的后台任务,由 lifespan 启动/取消。"""
    while True:
        try:
            await indexer.process_pending_documents()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[RAG] knowledge document worker error")
        await asyncio.sleep(sweep_seconds)
