from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.models import ChatMessageInput, ChatStreamRequest, ToolDefinitionInput
from app.services.errors import ServiceError
from app.services.model_configurations import UserModelCredentials


class LangChainChatService:
    """Provider-neutral chat boundary built with LangChain's OpenAI-compatible adapter."""

    def __init__(self, credentials: UserModelCredentials) -> None:
        self._credentials = credentials

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
        streaming_model = model.bind_tools([self._to_openai_tool(tool) for tool in request.tools]) if request.tools else model
        combined_chunk = None
        async for chunk in streaming_model.astream(messages):
            # Tool-enabled requests must use astream too. The previous ainvoke call buffered the
            # entire answer and produced one SSE event at the end of the response.
            combined_chunk = chunk if combined_chunk is None else combined_chunk + chunk
            content = chunk.content
            if isinstance(content, str) and content:
                yield content, []

        if combined_chunk is not None and combined_chunk.tool_calls:
            yield "", self._to_provider_tool_calls(combined_chunk.tool_calls)

    @staticmethod
    def _to_provider_tool_calls(tool_calls: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {
                "id": call.get("id"),
                "name": str(call["name"]),
                "arguments": {key: str(value) for key, value in dict(call.get("args", {})).items()},
            }
            for call in tool_calls
        ]

    @staticmethod
    def _to_openai_tool(tool: ToolDefinitionInput) -> dict[str, object]:
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

    @staticmethod
    def _to_langchain_message(message: ChatMessageInput) -> SystemMessage | HumanMessage | AIMessage:
        if message.role == "system":
            return SystemMessage(content=message.content)
        if message.role == "assistant":
            return AIMessage(content=message.content)
        return HumanMessage(content=message.content)
