"""检索服务: 混合检索(向量 + BM25) -> RRF 融合 -> 作用域过滤 -> 截断。

当前主流 RAG 检索架构:
    1. 稠密路: 查询向量化 -> vec0 top-k(低于 min_score 的直接丢弃 ——
       语义质量底线, 过滤噪声)
    2. 稀疏路: jieba 分词 -> FTS5 BM25 top-k(关键词精确匹配, 不受阈值
       约束 —— 数字、ID、专有名词等向量路召回差的内容由它补上)
    3. RRF(Reciprocal Rank Fusion)融合两路排名: score = Σ 1/(k + rank),
       k=60 为业界标准常量, 对两路尺度差异不敏感
    4. 融合排序后做作用域过滤, 再按 top_k 截断

作用域语义与 context.py 的旧实现一致:
- 记忆: GLOBAL 全部可见;WORKSPACE 仅当前工作区;CONVERSATION 仅当前会话
- 知识文档: 仅当前工作区
"""

import logging
from dataclasses import dataclass

from app.services.rag.embedding import EmbeddingService
from app.services.rag.vectorstore import VectorStore

logger = logging.getLogger("fatai.rag")

# 融合前每路召回量: 足够大以保证 RRF 能看见两路各自的合理候选
_RAW_TOP_K = 50
# RRF 常量(业界标准, Cornack et al. 2009)
_RRF_K = 60.0


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
        vec_scored = await self._store.search(query_vec, user_id, k=_RAW_TOP_K)
        fts_scored = await self._store.bm25_search(query, user_id, k=_RAW_TOP_K)

        # 稠密路先做语义阈值过滤; 稀疏路不过滤(关键词精确匹配也是信号)
        fused: dict[str, float] = {}
        for rank, scored in enumerate(vec_scored, start=1):
            if scored.score >= self._min_score:
                _rrf_add(fused, scored.chunk.id, rank)
        for rank, scored in enumerate(fts_scored, start=1):
            _rrf_add(fused, scored.chunk.id, rank)

        if not fused:
            logger.info("[RAG] query=%r user=%s no hits (vec=%d fts=%d)", query[:60], user_id, len(vec_scored), len(fts_scored))
            return RetrievalResult([], [])

        by_id = {scored.chunk.id: scored for scored in [*vec_scored, *fts_scored]}
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)

        memories: list[RetrievalHit] = []
        documents: list[RetrievalHit] = []
        for chunk_id, rrf_score in ranked:
            chunk = by_id[chunk_id].chunk
            hit = RetrievalHit(
                content=chunk.content,
                score=rrf_score,
                source_type=chunk.source_type,
                source_id=chunk.source_id,
                title=chunk.title,
                path=chunk.path,
            )
            if chunk.source_type == "memory" and _memory_in_scope(chunk.scope, chunk.workspace_id, chunk.conversation_id, workspace_id, conversation_id):
                memories.append(hit)
            elif chunk.source_type == "knowledge_document" and workspace_id is not None and chunk.workspace_id == workspace_id:
                documents.append(hit)
            if len(memories) >= self._top_k_memory and len(documents) >= self._top_k_document:
                break

        logger.info(
            "[RAG] query=%r user=%s vec=%d fts=%d memories=%d documents=%d",
            query[:60],
            user_id,
            len(vec_scored),
            len(fts_scored),
            len(memories),
            len(documents),
        )
        return RetrievalResult(memories[: self._top_k_memory], documents[: self._top_k_document])


def _rrf_add(fused: dict[str, float], chunk_id: str, rank: int, k: float = _RRF_K) -> None:
    """把一条命中的 RRF 贡献累加进去: 1/(k + rank), rank 为该路内 1-based 排名。"""
    fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)


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
