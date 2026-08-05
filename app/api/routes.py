import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db import Conversation, Message, SessionLocal, User, get_session
from app.models import (
    ChatStreamRequest,
    DocumentReadResponse,
    WeatherRequest,
    WeatherResponse,
    WebSearchRequest,
    WebSearchResponse,
)
from app.api.domain_routes import entity_payload, record_change
from app.services.chat import LangChainChatService, ServerToolExecutor
from app.services.context import assemble_context
from app.services.documents import DoclingDocumentService
from app.services.model_configurations import get_user_model_credentials
from app.services.search import BingRssSearchService, WeatherService
from app.services.titles import generate_conversation_title
from app.security import get_current_user

router = APIRouter(prefix="/v1")

# Keeps detached background tasks alive; asyncio otherwise garbage-collects unreferenced tasks.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def get_search_service(request: Request) -> BingRssSearchService:
    return request.app.state.search_service


def get_document_service(request: Request) -> DoclingDocumentService:
    return request.app.state.document_service


@router.post("/tools/search", response_model=WebSearchResponse)
async def search(
    payload: WebSearchRequest,
    service: BingRssSearchService = Depends(get_search_service),
) -> WebSearchResponse:
    query = payload.query.strip()
    return WebSearchResponse(query=query, results=await service.search(query, payload.max_results))


@router.post("/tools/weather", response_model=WeatherResponse)
async def weather(
    payload: WeatherRequest,
    search_service: BingRssSearchService = Depends(get_search_service),
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

    context = await assemble_context(
        session,
        user,
        payload.workspace_id,
        payload.conversation_id,
        payload.messages,
        payload.response_language_tag,
        payload.tool_results,
        payload.include_contextual_references,
    )

    async def events() -> AsyncIterator[str]:
        answer = ""
        persisted = False
        try:
            async for content, tool_calls in service.stream(payload, context=context):
                answer += content
                if content:
                    yield f"event: message\ndata: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                for tool_call in tool_calls:
                    yield f"event: tool_call\ndata: {json.dumps(tool_call, ensure_ascii=False)}\n\n"
            await persist_chat_turn(session, user, payload, answer)
            persisted = True
            if payload.conversation_id:
                _spawn_background(generate_title_in_background(user.id, payload.conversation_id))
        finally:
            # A disconnected client (stop generation) still leaves its partial answer behind.
            # shield keeps the persistence running even though the streaming task was cancelled.
            if not persisted and answer:
                try:
                    await asyncio.shield(persist_chat_turn(session, user, payload, answer))
                except Exception:
                    pass
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


async def persist_chat_turn(
    session: AsyncSession,
    user: User,
    payload: ChatStreamRequest,
    answer: str,
) -> None:
    """Save a chat turn directly on the server so clients no longer sync chat messages.

    Only conversation chats opt in (memory extraction and other internal calls pass no
    conversation id). The conversation is created on the fly when the client's own sync
    operation has not arrived yet; the client's later upsert carries richer data and, with
    delete-wins semantics, any delete still applies.
    """
    if not payload.conversation_id or not payload.assistant_message_id:
        return
    conversation = await session.get(Conversation, payload.conversation_id)
    if conversation is None:
        conversation = Conversation(
            id=payload.conversation_id,
            user_id=user.id,
            workspace_id=payload.workspace_id,
            title="New conversation",
            provider_type="OpenAI",
            model=payload.model or "",
        )
        session.add(conversation)
        await session.flush()
        await record_change(session, user, "conversation", conversation.id, "UPSERT", entity_payload(conversation))
    elif conversation.user_id != user.id:
        return

    user_turn = next((message for message in reversed(payload.messages) if message.role == "user"), None)
    if user_turn is not None:
        await persist_message(
            session,
            user,
            payload.user_message_id,
            payload.conversation_id,
            "user",
            user_turn.content,
            "",
            "Text",
        )
    await persist_message(
        session,
        user,
        payload.assistant_message_id,
        payload.conversation_id,
        "assistant",
        answer,
        "",
        "Markdown",
    )
    await session.commit()


async def persist_message(
    session: AsyncSession,
    user: User,
    message_id: str | None,
    conversation_id: str,
    role: str,
    content: str,
    reasoning_content: str,
    content_type: str,
) -> None:
    if not message_id or not content:
        return
    record = await session.get(Message, message_id)
    if record is None:
        record = Message(
            id=message_id,
            user_id=user.id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            reasoning_content=reasoning_content,
            content_type=content_type,
        )
        session.add(record)
    else:
        record.content = content
        record.role = role
        record.reasoning_content = reasoning_content
        record.content_type = content_type
    await record_change(session, user, "message", message_id, "UPSERT", entity_payload(record))


async def generate_title_in_background(user_id: str, conversation_id: str) -> None:
    """Titles a brand-new conversation with a model call, then syncs it via the change stream.

    Runs detached from the streaming request so the answer is never delayed. A title is only
    generated for the first turn; failures are swallowed because the title is cosmetic.
    """
    try:
        async with SessionLocal() as session:
            user = await session.get(User, user_id)
            conversation = await session.get(Conversation, conversation_id)
            if user is None or conversation is None or conversation.user_id != user_id:
                return
            messages = list(
                await session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at)
                )
            )
            if len(messages) > 2 or not messages or messages[0].role != "user":
                return
            credentials = await get_user_model_credentials(session, user_id, None, get_settings())
            model = ChatOpenAI(
                api_key=credentials.api_key,
                base_url=credentials.base_url or None,
                model=credentials.model,
            )
            title = await generate_conversation_title(model, messages[0].content)
            if not title:
                return
            conversation.title = title
            await record_change(session, user, "conversation", conversation_id, "UPSERT", entity_payload(conversation))
            await session.commit()
    except Exception:
        pass
