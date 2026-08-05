from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import logging
import time

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.models import ChatMessageInput, ChatStreamRequest, ToolDefinitionInput, WebSearchResult
from app.services.errors import ServiceError
from app.services.model_configurations import UserModelCredentials
from app.services.search import BingRssSearchService, WeatherService

logger = logging.getLogger("fatai.perf")

MAX_TOOL_OUTPUT_CHARACTERS = 24_000
MAX_TOOL_ROUNDS = 2
# Trail of text held back per round so short tool-round narration can be dropped.
NARRATION_BUFFER_CHARS = 200


@dataclass
class ToolOutcome:
    """Formatted tool result plus the structured, citable sources for the client UI."""

    content: str
    sources: list[dict[str, str]] = field(default_factory=list)


class ServerToolExecutor:
    """Executes model-requested tools inside the chat stream.

    Tool execution lives on the server so one streaming call covers the full model-plus-tool
    loop; the client only renders content and tool-call provenance. Only tools the server can
    back are bound, so advertised-but-unavailable tools never reach the model.
    """

    SUPPORTED = frozenset({"web_search", "weather"})

    def __init__(self, search_service: BingRssSearchService) -> None:
        self._search_service = search_service
        self._weather_service = WeatherService(search_service)

    def bindable(self, requested: list[ToolDefinitionInput]) -> list[dict[str, object]]:
        return [to_openai_tool(tool) for tool in requested if tool.name in self.SUPPORTED]

    async def execute(self, name: str, arguments: dict[str, object]) -> ToolOutcome:
        try:
            if name == "web_search":
                return await self._web_search(arguments)
            if name == "weather":
                return await self._weather(arguments)
            return ToolOutcome(f"Tool failed (TOOL_NOT_AVAILABLE): {name} is not available on the server.")
        except ServiceError as error:
            return ToolOutcome(f"Tool failed ({error.code}): {error.message}")
        except Exception:
            return ToolOutcome(f"Tool failed (TOOL_EXECUTION_FAILED): {name} execution failed.")

    async def _web_search(self, arguments: dict[str, object]) -> ToolOutcome:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolOutcome("Tool failed (INVALID_ARGUMENT): query must not be blank.")
        max_results = max(1, min(int(arguments.get("max_results", 5)), 10))
        results = await self._search_service.search(query, max_results)
        if not results:
            return ToolOutcome(f"No web results were found for: {query}")
        return ToolOutcome(
            bounded(format_results(f"Web results for: {query}", results)),
            [{"title": result.title, "url": result.url} for result in results],
        )

    async def _weather(self, arguments: dict[str, object]) -> ToolOutcome:
        location = str(arguments.get("location", "")).strip()
        if not location:
            return ToolOutcome("Tool failed (INVALID_ARGUMENT): location must not be blank.")
        results = await self._weather_service.weather(location, 10)
        if not results:
            return ToolOutcome(f"No weather sources were found for: {location}")
        return ToolOutcome(
            bounded(format_results(f"Weather references for: {location}", results)),
            [{"title": result.title, "url": result.url} for result in results],
        )


def format_results(header: str, results: list[WebSearchResult]) -> str:
    lines = [header]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}")
        if result.snippet:
            lines.append(result.snippet)
        lines.append(f"Source: {result.url}")
    return "\n".join(lines)


def bounded(content: str) -> str:
    if len(content) <= MAX_TOOL_OUTPUT_CHARACTERS:
        return content
    return content[:MAX_TOOL_OUTPUT_CHARACTERS] + "\n[Output truncated]"


class LangChainChatService:
    """Provider-neutral chat boundary built with LangChain's OpenAI-compatible adapter."""

    def __init__(self, credentials: UserModelCredentials, tools: ServerToolExecutor | None = None) -> None:
        self._credentials = credentials
        self._tools = tools

    def ensure_configured(self) -> None:
        if not self._credentials.api_key:
            raise ServiceError("MODEL_NOT_CONFIGURED", "The selected model provider is not configured.", 503)

    async def stream(
        self,
        request: ChatStreamRequest,
        context: list[ChatMessageInput] | None = None,
    ) -> AsyncIterator[tuple[str, list[dict[str, object]]]]:
        self.ensure_configured()

        model = ChatOpenAI(
            api_key=self._credentials.api_key,
            base_url=self._credentials.base_url or None,
            model=request.model or self._credentials.model,
            temperature=request.temperature,
            streaming=True,
        )
        messages = [self._to_langchain_message(message) for message in (context or request.messages)]
        tool_definitions = self._tools.bindable(request.tools) if self._tools else []
        if tool_definitions:
            model = model.bind_tools(tool_definitions)

        # Sources are deduplicated across the whole request (the model may search several
        # rounds and hit the same page twice), not per round.
        seen_source_urls: set[str] = set()
        stream_started_at = time.perf_counter()
        round_started_at = time.perf_counter()
        for round_index in range(MAX_TOOL_ROUNDS):
            combined_chunk = None
            # Keep a trailing buffer of the round's text: tool-round narration ("let me
            # search...") is short and gets dropped if the round ends in tool calls, while
            # a real answer longer than the buffer streams through with a tiny delay.
            buffered_content: list[str] = []
            buffered_chars = 0
            async for chunk in model.astream(messages):
                combined_chunk = chunk if combined_chunk is None else combined_chunk + chunk
                content = chunk.content
                if isinstance(content, str) and content:
                    buffered_content.append(content)
                    buffered_chars += len(content)
                    while buffered_chars > NARRATION_BUFFER_CHARS and buffered_content:
                        piece = buffered_content.pop(0)
                        buffered_chars -= len(piece)
                        yield piece, []
            if combined_chunk is None:
                return
            tool_calls = combined_chunk.tool_calls
            if not tool_calls:
                for content in buffered_content:
                    yield content, []
                logger.info(
                    "[PERF] round=%d model=%.2fs total=%.2fs final_answer=%d chars",
                    round_index + 1,
                    time.perf_counter() - round_started_at,
                    time.perf_counter() - stream_started_at,
                    sum(len(c) for c in buffered_content),
                )
                return
            logger.info(
                "[PERF] round=%d model=%.2fs tool_calls=%d (accumulated %.2fs)",
                round_index + 1,
                time.perf_counter() - round_started_at,
                len(tool_calls),
                time.perf_counter() - stream_started_at,
            )
            # AIMessageChunk has no to_message() in this langchain-core version; rebuild the
            # assistant turn from the normalized tool calls so the next round sees them.
            messages.append(
                AIMessage(content=combined_chunk.content or "", tool_calls=combined_chunk.tool_calls)
            )
            outcomes: list[ToolOutcome] = []

            async def execute_call(call: dict[str, object]) -> ToolOutcome:
                raw_arguments = call.get("args", {})
                arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
                return await self._tools.execute(str(call["name"]), arguments)

            # Run the round's tool calls concurrently (the model often issues two searches),
            # which roughly halves the searching time.
            tools_started_at = time.perf_counter()
            outcomes = list(await asyncio.gather(*(execute_call(call) for call in tool_calls)))
            logger.info(
                "[PERF] round=%d tools=%.2fs (%d concurrent)",
                round_index + 1,
                time.perf_counter() - tools_started_at,
                len(outcomes),
            )
            for call, outcome in zip(tool_calls, outcomes):
                messages.append(ToolMessage(content=outcome.content, tool_call_id=str(call.get("id", ""))))
            calls = []
            for call, outcome in zip(tool_calls, outcomes):
                sources = []
                for source in outcome.sources:
                    url = source.get("url")
                    if url and url in seen_source_urls:
                        continue
                    if url:
                        seen_source_urls.add(url)
                    sources.append(source)
                calls.append(to_provider_tool_call(call, sources))
            yield "", calls
            if round_index == MAX_TOOL_ROUNDS - 1:
                # The model kept requesting tools; force a final answer without tool access so
                # the user always receives a response built from the results already gathered.
                model = ChatOpenAI(
                    api_key=self._credentials.api_key,
                    base_url=self._credentials.base_url or None,
                    model=request.model or self._credentials.model,
                    temperature=request.temperature,
                    streaming=True,
                )
            round_started_at = time.perf_counter()

        forced_started_at = time.perf_counter()
        async for chunk in model.astream(messages):
            content = chunk.content
            if isinstance(content, str) and content:
                yield content, []
        logger.info(
            "[PERF] final model=%.2fs total=%.2fs (forced answer round)",
            time.perf_counter() - forced_started_at,
            time.perf_counter() - stream_started_at,
        )

    @staticmethod
    def _to_langchain_message(message: ChatMessageInput) -> SystemMessage | HumanMessage | AIMessage:
        if message.role == "system":
            return SystemMessage(content=message.content)
        if message.role == "assistant":
            return AIMessage(content=message.content)
        return HumanMessage(content=message.content)


def to_openai_tool(tool: ToolDefinitionInput) -> dict[str, object]:
    properties = {
        parameter.name: {
            "type": "string",
            "description": parameter.description,
            **({"enum": parameter.allowed_values} if parameter.allowed_values else {}),
        }
        for parameter in tool.parameters
    }
    required = [parameter.name for parameter in tool.parameters if parameter.required]
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def to_provider_tool_call(call: dict[str, object], sources: list[dict[str, str]] | None = None) -> dict[str, object]:
    return {
        "id": call.get("id"),
        "name": str(call["name"]),
        "arguments": {key: str(value) for key, value in dict(call.get("args", {})).items()},
        "sources": sources or [],
    }
