from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.models import ChatMessageInput, ChatStreamRequest, ToolDefinitionInput, WebSearchResult
from app.services.errors import ServiceError
from app.services.model_configurations import UserModelCredentials
from app.services.search import DuckDuckGoSearchService, WeatherService

MAX_TOOL_OUTPUT_CHARACTERS = 24_000
MAX_TOOL_ROUNDS = 5


class ServerToolExecutor:
    """Executes model-requested tools inside the chat stream.

    Tool execution lives on the server so one streaming call covers the full model-plus-tool
    loop; the client only renders content and tool-call provenance. Only tools the server can
    back are bound, so advertised-but-unavailable tools never reach the model.
    """

    SUPPORTED = frozenset({"web_search", "weather"})

    def __init__(self, search_service: DuckDuckGoSearchService) -> None:
        self._search_service = search_service
        self._weather_service = WeatherService(search_service)

    def bindable(self, requested: list[ToolDefinitionInput]) -> list[dict[str, object]]:
        return [to_openai_tool(tool) for tool in requested if tool.name in self.SUPPORTED]

    async def execute(self, name: str, arguments: dict[str, object]) -> str:
        try:
            if name == "web_search":
                return await self._web_search(arguments)
            if name == "weather":
                return await self._weather(arguments)
            return f"Tool failed (TOOL_NOT_AVAILABLE): {name} is not available on the server."
        except ServiceError as error:
            return f"Tool failed ({error.code}): {error.message}"
        except Exception:
            return f"Tool failed (TOOL_EXECUTION_FAILED): {name} execution failed."

    async def _web_search(self, arguments: dict[str, object]) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "Tool failed (INVALID_ARGUMENT): query must not be blank."
        max_results = max(1, min(int(arguments.get("max_results", 5)), 10))
        results = await self._search_service.search(query, max_results)
        if not results:
            return f"No web results were found for: {query}"
        return bounded(format_results(f"Web results for: {query}", results))

    async def _weather(self, arguments: dict[str, object]) -> str:
        location = str(arguments.get("location", "")).strip()
        if not location:
            return "Tool failed (INVALID_ARGUMENT): location must not be blank."
        results = await self._weather_service.weather(location, 10)
        if not results:
            return f"No weather sources were found for: {location}"
        return bounded(format_results(f"Weather references for: {location}", results))


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

    async def stream(self, request: ChatStreamRequest) -> AsyncIterator[tuple[str, list[dict[str, object]]]]:
        self.ensure_configured()

        model = ChatOpenAI(
            api_key=self._credentials.api_key,
            base_url=self._credentials.base_url or None,
            model=request.model or self._credentials.model,
            temperature=request.temperature,
            streaming=True,
        )
        messages = [self._to_langchain_message(message) for message in request.messages]
        tool_definitions = self._tools.bindable(request.tools) if self._tools else []
        if tool_definitions:
            model = model.bind_tools(tool_definitions)

        for _ in range(MAX_TOOL_ROUNDS):
            combined_chunk = None
            async for chunk in model.astream(messages):
                combined_chunk = chunk if combined_chunk is None else combined_chunk + chunk
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content, []
            if combined_chunk is None:
                return
            tool_calls = combined_chunk.tool_calls
            if not tool_calls:
                return
            yield "", [to_provider_tool_call(call) for call in tool_calls]
            messages.append(combined_chunk.to_message())
            for call in tool_calls:
                raw_arguments = call.get("args", {})
                arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
                result = await self._tools.execute(str(call["name"]), arguments)
                messages.append(ToolMessage(content=result, tool_call_id=str(call.get("id", ""))))
        yield "I could not complete this request because the tool loop exceeded its limit.", []

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


def to_provider_tool_call(call: dict[str, object]) -> dict[str, object]:
    return {
        "id": call.get("id"),
        "name": str(call["name"]),
        "arguments": {key: str(value) for key, value in dict(call.get("args", {})).items()},
    }
