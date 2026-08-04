import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db import User, get_session
from app.models import (
    ChatStreamRequest,
    DocumentReadResponse,
    WeatherRequest,
    WeatherResponse,
    WebSearchRequest,
    WebSearchResponse,
)
from app.services.chat import LangChainChatService, ServerToolExecutor
from app.services.documents import DoclingDocumentService
from app.services.model_configurations import get_user_model_credentials
from app.services.search import DuckDuckGoSearchService, WeatherService
from app.security import get_current_user

router = APIRouter(prefix="/v1")


def get_search_service(request: Request) -> DuckDuckGoSearchService:
    return request.app.state.search_service


def get_document_service(request: Request) -> DoclingDocumentService:
    return request.app.state.document_service


@router.post("/tools/search", response_model=WebSearchResponse)
async def search(
    payload: WebSearchRequest,
    service: DuckDuckGoSearchService = Depends(get_search_service),
) -> WebSearchResponse:
    query = payload.query.strip()
    return WebSearchResponse(query=query, results=await service.search(query, payload.max_results))


@router.post("/tools/weather", response_model=WeatherResponse)
async def weather(
    payload: WeatherRequest,
    search_service: DuckDuckGoSearchService = Depends(get_search_service),
) -> WeatherResponse:
    location = payload.location.strip()
    service = WeatherService(search_service)
    return WeatherResponse(location=location, results=await service.weather(location, payload.max_results))


@router.post("/tools/document-read", response_model=DocumentReadResponse)
async def document_read(request: Request, service: DoclingDocumentService = Depends(get_document_service)) -> DocumentReadResponse:
    """Read a selected document.

    Multipart uploads are the production interface. The legacy JSON `localPath` contract remains
    available only for a co-located desktop client during the staged Kotlin-to-Python migration.
    Disable it with `ALLOW_LOCAL_DOCUMENT_PATHS=false` before exposing this server remotely.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        if not get_settings().allow_local_document_paths:
            raise HTTPException(410, "Local-path document reads are disabled; upload the file instead.")
        payload = await request.json()
        local_path = Path(str(payload.get("localPath", ""))).expanduser().resolve()
        if not local_path.is_file():
            raise HTTPException(400, "The selected file does not exist or is not a regular file.")
        return await service.read(
            display_name=str(payload.get("displayName", "")).strip() or local_path.name,
            mime_type=str(payload.get("mimeType", "application/octet-stream")),
            content=local_path.read_bytes(),
        )

    form = await request.form()
    file = form.get("file")
    if not isinstance(file, UploadFile) and not hasattr(file, "read"):
        raise HTTPException(422, "multipart/form-data must include a file part.")
    return await service.read(
        display_name=file.filename or "document",
        mime_type=file.content_type or "application/octet-stream",
        content=await file.read(),
    )


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatStreamRequest,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    credentials = await get_user_model_credentials(
        session, user.id, payload.model_configuration_id, get_settings()
    )
    service = LangChainChatService(
        credentials, ServerToolExecutor(get_search_service(request))
    )
    service.ensure_configured()

    async def events() -> AsyncIterator[str]:
        async for content, tool_calls in service.stream(payload):
            if content:
                yield f"event: message\ndata: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
            for tool_call in tool_calls:
                yield f"event: tool_call\ndata: {json.dumps(tool_call, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
