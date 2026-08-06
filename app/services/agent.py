import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.models import ChatMessageInput, ToolDefinitionInput
from app.services.chat import (
    LangChainChatService,
    ServerToolExecutor,
    to_openai_tool,
    to_provider_tool_call,
)
from app.services.errors import ServiceError
from app.services.model_configurations import UserModelCredentials

logger = logging.getLogger("fatai.agent")

MAX_TOOL_ROUNDS = 3
NARRATION_BUFFER_CHARS = 200


class AgentState(TypedDict):
    messages: list[BaseMessage]
    tool_rounds: int


class LangGraphAgentService:
    """LangGraph agent with tool execution, conditional routing, and streaming.

    Builds a graph: model -> (tools | END).  The model is called with tools
    bound; when it requests tool calls the tools node executes them
    concurrently and routes back to the model.  An optional approval mode
    pauses execution and yields ``approval_required`` events so the caller
    can prompt the user before tools run.
    """

    def __init__(
        self,
        credentials: UserModelCredentials,
        tools: ServerToolExecutor | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self._credentials = credentials
        self._tools = tools
        self._max_tool_rounds = max_tool_rounds

    # ------------------------------------------------------------------
    # Synchronous single-shot entry point (kept for backward compat)
    # ------------------------------------------------------------------
    async def run(
        self, messages: list[ChatMessageInput], model_name: str | None = None
    ) -> str:
        if not self._credentials.api_key:
            raise ServiceError(
                "MODEL_NOT_CONFIGURED",
                "The selected model provider is not configured.",
                503,
            )
        model = ChatOpenAI(
            api_key=self._credentials.api_key,
            base_url=self._credentials.base_url or None,
            model=model_name or self._credentials.model,
        )

        async def invoke_model(state: AgentState) -> dict[str, object]:
            return {"messages": [await model.ainvoke(state["messages"])]}

        graph = StateGraph(AgentState)
        graph.add_node("model", invoke_model)
        graph.add_edge(START, "model")
        graph.add_edge("model", END)
        initial: AgentState = {"messages": [], "tool_rounds": 0}
        initial["messages"] = [
            LangChainChatService._to_langchain_message(m) for m in messages
        ]
        result = await graph.compile().ainvoke(initial)
        response = result["messages"][-1]
        if not isinstance(response, AIMessage) or not isinstance(response.content, str):
            raise ServiceError(
                "AGENT_FAILED", "The agent returned an unsupported response.", 502
            )
        return response.content

    # ------------------------------------------------------------------
    # Streaming agent entry point
    # ------------------------------------------------------------------
    async def stream(
        self,
        messages: list[ChatMessageInput],
        model_name: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        tool_definitions: list[ToolDefinitionInput] | None = None,
        require_approval: bool = False,
    ) -> AsyncIterator[dict[str, object]]:
        """Run the agent and yield SSE-style events.

        Events yielded:

        * ``{"event": "message", "content": "..."}``
        * ``{"event": "tool_call", "id": "...", "name": "...", "arguments": {...}, "sources": [...]}``
        * ``{"event": "approval_required", "id": "...", "name": "...", "arguments": {...}}``
        * ``{"event": "done"}``
        """
        if not self._credentials.api_key:
            raise ServiceError(
                "MODEL_NOT_CONFIGURED",
                "The selected model provider is not configured.",
                503,
            )

        lc_messages: list[BaseMessage] = []
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))
        lc_messages.extend(
            LangChainChatService._to_langchain_message(m) for m in messages
        )

        model_name = model_name or self._credentials.model
        bindable_tools = (
            self._tools.bindable(tool_definitions or []) if self._tools else []
        )

        stream_started_at = time.perf_counter()

        round_index = 0
        while round_index < self._max_tool_rounds:
            round_started_at = time.perf_counter()
            has_tools = round_index < self._max_tool_rounds - 1 and bindable_tools
            model = self._make_model(
                model_name,
                temperature,
                bindable_tools if has_tools else [],
            )

            combined_chunk = None
            buffered: list[str] = []
            buffered_chars = 0

            async for chunk in model.astream(lc_messages):
                combined_chunk = (
                    chunk if combined_chunk is None else combined_chunk + chunk
                )
                content = chunk.content
                if isinstance(content, str) and content:
                    buffered.append(content)
                    buffered_chars += len(content)
                    while buffered_chars > NARRATION_BUFFER_CHARS and buffered:
                        yield {"event": "message", "content": buffered.pop(0)}

            if combined_chunk is None:
                yield {"event": "done"}
                return

            tool_calls = combined_chunk.tool_calls
            if not tool_calls or not self._tools:
                for piece in buffered:
                    yield {"event": "message", "content": piece}
                logger.info(
                    "[AGENT] round=%d model=%.2fs total=%.2fs done",
                    round_index + 1,
                    time.perf_counter() - round_started_at,
                    time.perf_counter() - stream_started_at,
                )
                yield {"event": "done"}
                return

            logger.info(
                "[AGENT] round=%d model=%.2fs tool_calls=%d",
                round_index + 1,
                time.perf_counter() - round_started_at,
                len(tool_calls),
            )

            lc_messages.append(
                AIMessage(
                    content=combined_chunk.content or "",
                    tool_calls=combined_chunk.tool_calls,
                )
            )

            # Approval gate
            if require_approval:
                for call in tool_calls:
                    yield {
                        "event": "approval_required",
                        "id": str(call.get("id", "")),
                        "name": str(call["name"]),
                        "arguments": _extract_args(call),
                    }
                yield {"event": "done"}
                return

            # Execute tools concurrently
            outcomes, events = await self._execute_tools(tool_calls)
            for evt in events:
                yield evt

            for call, outcome in zip(tool_calls, outcomes):
                lc_messages.append(
                    ToolMessage(
                        content=outcome.content,
                        tool_call_id=str(call.get("id", "")),
                    )
                )

            round_index += 1

        # Forced final answer (out of tool rounds)
        model = self._make_model(model_name, temperature, [])
        async for chunk in model.astream(lc_messages):
            content = chunk.content
            if isinstance(content, str) and content:
                yield {"event": "message", "content": content}
        logger.info(
            "[AGENT] forced-answer total=%.2fs",
            time.perf_counter() - stream_started_at,
        )
        yield {"event": "done"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_model(
        self,
        model_name: str,
        temperature: float | None,
        tool_definitions: list[dict[str, object]],
    ) -> ChatOpenAI:
        model = ChatOpenAI(
            api_key=self._credentials.api_key,
            base_url=self._credentials.base_url or None,
            model=model_name,
            temperature=temperature or 0.7,
            streaming=True,
        )
        if tool_definitions:
            model = model.bind_tools(tool_definitions)
        return model

    async def _execute_tools(
        self, tool_calls: list[dict[str, object]]
    ) -> tuple[list[object], list[dict[str, object]]]:
        seen_source_urls: set[str] = set()

        async def execute_call(call: dict[str, object]):
            raw = call.get("args", {})
            arguments = raw if isinstance(raw, dict) else {}
            return await self._tools.execute(str(call["name"]), arguments)

        outcomes = list(await asyncio.gather(*(execute_call(c) for c in tool_calls)))

        events: list[dict[str, object]] = []
        for call, outcome in zip(tool_calls, outcomes):
            sources: list[dict[str, str]] = []
            for source in outcome.sources:
                url = source.get("url")
                if url and url in seen_source_urls:
                    continue
                if url:
                    seen_source_urls.add(url)
                sources.append(source)
            entry = to_provider_tool_call(call, sources)
            entry["event"] = "tool_call"
            events.append(entry)

        return outcomes, events


def _extract_args(call: dict[str, object]) -> dict[str, object]:
    raw = call.get("args", {})
    if isinstance(raw, dict):
        return {k: str(v) for k, v in raw.items()}
    return {}
