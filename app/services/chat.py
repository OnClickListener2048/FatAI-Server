from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.models import ChatMessageInput, ChatStreamRequest, ToolDefinitionInput
from app.services.errors import ServiceError


class LangChainChatService:
    """Provider-neutral chat boundary built with LangChain's OpenAI-compatible adapter."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def ensure_configured(self) -> None:
        if not self._settings.openai_api_key:
            raise ServiceError("MODEL_NOT_CONFIGURED", "The server model provider is not configured.", 503)

    async def stream(self, request: ChatStreamRequest) -> AsyncIterator[tuple[str, list[dict[str, object]]]]:
        self.ensure_configured()

        model = ChatOpenAI(
            api_key=self._settings.openai_api_key,
            base_url=str(self._settings.openai_base_url) if self._settings.openai_base_url else None,
            model=request.model or self._settings.default_chat_model,
            temperature=request.temperature,
            streaming=True,
        )
        messages = [self._to_langchain_message(message) for message in request.messages]
        if request.tools:
            response = await model.bind_tools([self._to_openai_tool(tool) for tool in request.tools]).ainvoke(messages)
            tool_calls = [
                {
                    "id": call.get("id"),
                    "name": str(call["name"]),
                    "arguments": {key: str(value) for key, value in dict(call.get("args", {})).items()},
                }
                for call in response.tool_calls
            ]
            if tool_calls:
                yield "", tool_calls
                return
            if isinstance(response.content, str) and response.content:
                yield response.content, []
            return

        async for chunk in model.astream(messages):
            content = chunk.content
            if isinstance(content, str) and content:
                yield content, []

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
