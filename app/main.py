from contextlib import asynccontextmanager
import asyncio
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.db import initialize_database
from app.models import ApiError, HealthResponse
from app.api.domain_routes import router as domain_router
from app.services.documents import DoclingDocumentService
from app.services.errors import ServiceError
from app.services.rag.embedding import EmbeddingService
from app.services.rag.indexing import IndexingService, knowledge_document_worker
from app.services.rag.retrieval import RetrievalService
from app.services.rag.vectorstore import VectorStore
from app.services.search import BingRssSearchService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await initialize_database()
    client = httpx.AsyncClient(timeout=settings.request_timeout_seconds, follow_redirects=True)
    app.state.search_service = BingRssSearchService(client)
    # Docling 转换单独使用长超时 client: 通用超时(20s)会掐断本地 CPU 上的
    # 大文件转换(OCR/多页 PDF 常见 30s+), 表现为误报 DOCLING_UNAVAILABLE。
    docling_client = httpx.AsyncClient(timeout=settings.docling_timeout_seconds, follow_redirects=True)
    app.state.document_service = DoclingDocumentService(
        client=docling_client,
        server_url=str(settings.docling_server_url),
        max_size_bytes=settings.max_document_size_bytes,
    )
    # RAG: embedding -> 向量存储 -> 索引/检索。向量库不可用时检索自动降级为空,
    # 聊天流随后按旧逻辑(时间倒序记忆)继续。
    embedder = EmbeddingService(
        settings.embedding_base_url,
        settings.embedding_api_key,
        settings.embedding_model,
        settings.embedding_dimensions,
    )
    vector_store = VectorStore(settings.database_url, settings.embedding_dimensions)
    vector_store.initialize()
    indexer = IndexingService(
        embedder,
        vector_store,
        app.state.document_service,
        chunk_chars=settings.rag_chunk_chars,
    )
    app.state.rag_vector_store = vector_store
    app.state.rag_indexer = indexer
    app.state.rag_retriever = RetrievalService(
        embedder,
        vector_store,
        top_k_memory=settings.rag_top_k_memory,
        top_k_document=settings.rag_top_k_document,
        min_score=settings.rag_min_score,
    )
    app.state.rag_worker = asyncio.create_task(
        knowledge_document_worker(indexer, settings.rag_sweep_seconds)
    )
    asyncio.create_task(indexer.backfill_memories())
    yield
    app.state.rag_worker.cancel()
    await docling_client.aclose()
    await client.aclose()


def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    settings = get_settings()
    app = FastAPI(title="FatAI Server", version="0.1.0", lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.cors_origins],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(ServiceError)
    async def service_error_handler(_: Request, error: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=ApiError(code=error.code, message=error.message).model_dump(),
        )

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse(service=settings.service_name)

    app.include_router(router)
    app.include_router(domain_router)
    return app


app = create_app()
