from contextlib import asynccontextmanager

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
from app.services.search import BingRssSearchService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await initialize_database()
    client = httpx.AsyncClient(timeout=settings.request_timeout_seconds, follow_redirects=True)
    app.state.search_service = BingRssSearchService(client)
    app.state.document_service = DoclingDocumentService(
        client=client,
        server_url=str(settings.docling_server_url),
        max_size_bytes=settings.max_document_size_bytes,
    )
    yield
    await client.aclose()


def create_app() -> FastAPI:
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
