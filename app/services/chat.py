from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import asyncio
import json
import logging
import time

import httpx

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
        # Token usage accumulated across every provider call of the current turn (including
        # tool rounds). None-y semantics: usage_totals() returns None when the provider never
        # reported usage, which callers must distinguish from a genuinely zero-token turn.
        self._usage: dict[str, int] = {"prompt": 0, "completion": 0}

    def usage_totals(self) -> dict[str, int] | None:
        totals = dict(self._usage)
        if not totals["prompt"] and not totals["completion"]:
            return None
        return totals

    @staticmethod
    def _capture_usage(accumulated: dict[str, int], chunk) -> None:
        metadata = getattr(chunk, "usage_metadata", None) or {}
        accumulated["prompt"] += int(metadata.get("input_tokens") or 0)
        accumulated["completion"] += int(metadata.get("output_tokens") or 0)

    def ensure_configured(self) -> None:
        if not self._credentials.api_key:
            raise ServiceError("MODEL_NOT_CONFIGURED", "The selected model provider is not configured.", 503)

    async def stream(
        self,
        request: ChatStreamRequest,
        context: list[ChatMessageInput] | None = None,
    ) -> AsyncIterator[tuple[str, list[dict[str, object]], str]]:
        """Streams a chat turn.

        Yields ``(content, tool_calls, reasoning_content)`` tuples. In thinking mode the
        provider stream is forwarded directly (LangChain drops reasoning_content), in
        non-thinking mode the LangChain tool loop is used.
        """
        self.ensure_configured()
        self._usage = {"prompt": 0, "completion": 0}
        if request.thinking:
            async for content, tool_calls, reasoning in self._stream_direct(request, context):
                yield content, tool_calls, reasoning
            return

        model = ChatOpenAI(
            api_key=self._credentials.api_key,
            base_url=self._credentials.base_url or None,
            model=request.model or self._credentials.model,
            temperature=request.temperature,
            streaming=True,
            extra_body={"thinking": {"type": "disabled"}},
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
                self._capture_usage(self._usage, chunk)
                content = chunk.content
                if isinstance(content, str) and content:
                    buffered_content.append(content)
                    buffered_chars += len(content)
                    while buffered_chars > NARRATION_BUFFER_CHARS and buffered_content:
                        piece = buffered_content.pop(0)
                        buffered_chars -= len(piece)
                        yield piece, [], ""
            if combined_chunk is None:
                return
            tool_calls = combined_chunk.tool_calls
            if not tool_calls:
                for content in buffered_content:
                    yield content, [], ""
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
            yield "", calls, ""
            if round_index == MAX_TOOL_ROUNDS - 1:
                # The model kept requesting tools; force a final answer without tool access so
                # the user always receives a response built from the results already gathered.
                model = ChatOpenAI(
                    api_key=self._credentials.api_key,
                    base_url=self._credentials.base_url or None,
                    model=request.model or self._credentials.model,
                    temperature=request.temperature,
                    streaming=True,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            round_started_at = time.perf_counter()

        forced_started_at = time.perf_counter()
        async for chunk in model.astream(messages):
            self._capture_usage(self._usage, chunk)
            content = chunk.content
            if isinstance(content, str) and content:
                yield content, [], ""
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

    async def _stream_direct(
        self,
        request: ChatStreamRequest,
        context: list[ChatMessageInput] | None = None,
    ) -> AsyncIterator[tuple[str, list[dict[str, object]], str]]:
        """Thinking-mode streaming straight from the provider's SSE feed.

        LangChain's ChatOpenAI discards ``reasoning_content``, so thinking mode bypasses it:
        messages go to the provider via httpx and both content and reasoning are forwarded.
        Tools still run as a loop (advertise, execute, feed back) when the model asks for them.
        """
        base_url = (self._credentials.base_url or "https://api.deepseek.com").rstrip("/")
        messages = [{"role": message.role, "content": message.content} for message in (context or request.messages)]
        tool_definitions = self._tools.bindable(request.tools) if self._tools else []
        seen_source_urls: set[str] = set()
        stream_started_at = time.perf_counter()
        round_started_at = time.perf_counter()
        for round_index in range(MAX_TOOL_ROUNDS):
            combined_chunk = None
            buffered_content: list[str] = []
            buffered_chars = 0
            async for delta in self._direct_stream_once(
                base_url, messages, request, tool_definitions if round_index < MAX_TOOL_ROUNDS - 1 else []
            ):
                combined_chunk = delta if combined_chunk is None else _merge_delta(combined_chunk, delta)
                reasoning = delta.get("reasoning_content") or ""
                if reasoning:
                    yield "", [], reasoning
                content = delta.get("content") or ""
                if content:
                    buffered_content.append(content)
                    buffered_chars += len(content)
                    while buffered_chars > NARRATION_BUFFER_CHARS and buffered_content:
                        piece = buffered_content.pop(0)
                        buffered_chars -= len(piece)
                        yield piece, [], ""
            if combined_chunk is None:
                return
            tool_calls = _normalized_tool_calls(combined_chunk.get("tool_calls") or [])
            if not tool_calls:
                for content in buffered_content:
                    yield content, [], ""
                logger.info(
                    "[PERF] thinking round=%d model=%.2fs total=%.2fs final_answer=%d chars",
                    round_index + 1,
                    time.perf_counter() - round_started_at,
                    time.perf_counter() - stream_started_at,
                    sum(len(c) for c in buffered_content),
                )
                return
            logger.info(
                "[PERF] thinking round=%d model=%.2fs tool_calls=%d (accumulated %.2fs)",
                round_index + 1,
                time.perf_counter() - round_started_at,
                len(tool_calls),
                time.perf_counter() - stream_started_at,
            )
            assistant_turn: dict[str, object] = {
                "role": "assistant",
                "content": "".join(buffered_content),
            }
            raw_calls = combined_chunk.get("tool_calls") or []
            if raw_calls:
                assistant_turn["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call["args"], ensure_ascii=False)},
                    }
                    for call in tool_calls
                ]
            messages.append(assistant_turn)

            async def execute_call(call: dict[str, object]) -> ToolOutcome:
                arguments = call.get("args") if isinstance(call.get("args"), dict) else {}
                return await self._tools.execute(str(call["name"]), arguments)

            tools_started_at = time.perf_counter()
            outcomes = list(await asyncio.gather(*(execute_call(call) for call in tool_calls)))
            logger.info(
                "[PERF] thinking round=%d tools=%.2fs (%d concurrent)",
                round_index + 1,
                time.perf_counter() - tools_started_at,
                len(outcomes),
            )
            for call, outcome in zip(tool_calls, outcomes):
                messages.append({"role": "tool", "tool_call_id": str(call.get("id", "")), "content": outcome.content})
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
            yield "", calls, ""
            round_started_at = time.perf_counter()

    async def _direct_stream_once(
        self,
        base_url: str,
        messages: list[dict[str, object]],
        request: ChatStreamRequest,
        tool_definitions: list[dict[str, object]],
    ) -> AsyncIterator[dict[str, object]]:
        """One raw SSE pass against the provider's chat/completions endpoint."""
        body: dict[str, object] = {
            "model": request.model or self._credentials.model,
            "messages": messages,
            "stream": True,
            "temperature": request.temperature,
            "thinking": {"type": "enabled"},
            # OpenAI/DeepSeek only include usage in the stream when explicitly requested.
            "stream_options": {"include_usage": True},
        }
        if tool_definitions:
            body["tools"] = tool_definitions
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._credentials.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                json=body,
            ) as response:
                if response.status_code != 200:
                    error_body = (await response.aread()).decode("utf-8", "replace")
                    raise ServiceError(
                        "PROVIDER_ERROR",
                        error_body[:500] or f"Provider returned {response.status_code}.",
                        response.status_code,
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # The final chunk carries top-level usage instead of a delta; capture it
                    # so the turn total survives for the done event.
                    if "usage" in payload:
                        usage = payload["usage"] or {}
                        self._usage["prompt"] += int(usage.get("prompt_tokens") or 0)
                        self._usage["completion"] += int(usage.get("completion_tokens") or 0)
                        continue
                    try:
                        delta = payload["choices"][0]["delta"]
                    except (KeyError, IndexError):
                        continue
                    yield dict(delta)


def _merge_delta(combined: dict[str, object], delta: dict[str, object]) -> dict[str, object]:
    """Accumulates streamed tool calls (by index) into the round's combined chunk."""
    merged = dict(combined)
    calls = list(merged.get("tool_calls") or [])
    for tool_call in delta.get("tool_calls") or []:
        index = int(tool_call.get("index", 0))
        while len(calls) <= index:
            calls.append({"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
        accumulated = calls[index]
        if tool_call.get("id"):
            accumulated["id"] = tool_call["id"]
        function = tool_call.get("function") or {}
        if function.get("name"):
            accumulated["function"]["name"] = function["name"]
        if function.get("arguments"):
            accumulated["function"]["arguments"] += function["arguments"]
    merged["tool_calls"] = calls
    return merged


def _normalized_tool_calls(raw_calls: list[dict[str, object]]) -> list[dict[str, object]]:
    """Normalizes accumulated streamed tool calls into {id, name, args} dicts."""
    calls = []
    for call in raw_calls:
        function = call.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        arguments: dict[str, object] = {}
        raw = function.get("arguments") or ""
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    arguments = parsed
            except json.JSONDecodeError:
                arguments = {}
        calls.append({"id": call.get("id"), "name": name, "args": arguments})
    return calls


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
