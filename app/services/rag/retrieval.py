"""检索服务: 查询向量化 -> vec0 top-k -> 作用域过滤 -> 阈值截断。

作用域语义与 context.py 的旧实现一致:
- 记忆: GLOBAL 全部可见;WORKSPACE 仅当前工作区;CONVERSATION 仅当前会话
- 知识文档: 仅当前工作区
"""

import logging
from dataclasses import dataclass

from app.services.rag.embedding import EmbeddingService
from app.services.rag.vectorstore import VectorStore

logger = logging.getLogger("fatai.rag")

# 先取多一点再在内存里做作用域 OR 过滤(vec0 辅助列只支持 AND 等值过滤)
_RAW_TOP_K = 30


@dataclass
class RetrievalHit:
    content: str
    score: float
    source_type: str  # memory | knowledge_document
    source_id: str
    title: str
    path: str


@dataclass
class RetrievalResult:
    memories: list[RetrievalHit]
    documents: list[RetrievalHit]

    @property
    def sources(self) -> list[dict]:
        return [
            {"title": hit.title, "kind": hit.source_type, "id": hit.source_id}
            for hit in [*self.memories, *self.documents]
        ]

    @property
    def empty(self) -> bool:
        return not self.memories and not self.documents


class RetrievalService:
    def __init__(
        self,
        embedder: EmbeddingService,
        store: VectorStore,
        top_k_memory: int = 8,
        top_k_document: int = 5,
        min_score: float = 0.45,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k_memory = top_k_memory
        self._top_k_document = top_k_document
        self._min_score = min_score

    async def search(
        self,
        query: str,
        user_id: str,
        workspace_id: str | None,
        conversation_id: str | None,
    ) -> RetrievalResult:
        if not self._store.enabled:
            return RetrievalResult([], [])
        query_vec = await self._embedder.embed_one(query)
        raw = await self._store.search(query_vec, user_id, k=max(_RAW_TOP_K, self._top_k_memory + self._top_k_document))

        memories: list[RetrievalHit] = []
        documents: list[RetrievalHit] = []
        for scored in raw:
            chunk = scored.chunk
            if scored.score < self._min_score:
                continue
            hit = RetrievalHit(
                content=chunk.content,
                score=scored.score,
                source_type=chunk.source_type,
                source_id=chunk.source_id,
                title=chunk.title,
                path=chunk.path,
            )
            if chunk.source_type == "memory" and _memory_in_scope(chunk.scope, chunk.workspace_id, chunk.conversation_id, workspace_id, conversation_id):
                memories.append(hit)
            elif chunk.source_type == "knowledge_document" and workspace_id is not None and chunk.workspace_id == workspace_id:
                documents.append(hit)

        memories.sort(key=lambda hit: hit.score, reverse=True)
        documents.sort(key=lambda hit: hit.score, reverse=True)
        logger.info(
            "[RAG] query=%r user=%s memories=%d documents=%d",
            query[:60],
            user_id,
            len(memories),
            len(documents),
        )
        return RetrievalResult(memories[: self._top_k_memory], documents[: self._top_k_document])


def _memory_in_scope(
    scope: str,
    chunk_workspace_id: str | None,
    chunk_conversation_id: str | None,
    workspace_id: str | None,
    conversation_id: str | None,
) -> bool:
    if scope == "GLOBAL":
        return True
    if scope == "WORKSPACE":
        return workspace_id is not None and chunk_workspace_id == workspace_id
    if scope == "CONVERSATION":
        return conversation_id is not None and chunk_conversation_id == conversation_id
    return False
