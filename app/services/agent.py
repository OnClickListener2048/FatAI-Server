from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.models import ChatMessageInput
from app.services.chat import LangChainChatService
from app.services.errors import ServiceError


class AgentState(TypedDict):
    messages: list[BaseMessage]


class LangGraphAgentService:
    """LangGraph boundary for agents; later tool and approval nodes attach here."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, messages: list[ChatMessageInput], model_name: str | None = None) -> str:
        if not self._settings.openai_api_key:
            raise ServiceError("MODEL_NOT_CONFIGURED", "The server model provider is not configured.", 503)
        model = ChatOpenAI(
            api_key=self._settings.openai_api_key,
            base_url=str(self._settings.openai_base_url) if self._settings.openai_base_url else None,
            model=model_name or self._settings.default_chat_model,
        )

        async def invoke_model(state: AgentState) -> dict[str, list[BaseMessage]]:
            return {"messages": [await model.ainvoke(state["messages"])]}

        graph = StateGraph(AgentState)
        graph.add_node("model", invoke_model)
        graph.add_edge(START, "model")
        graph.add_edge("model", END)
        result = await graph.compile().ainvoke(
            {"messages": [LangChainChatService._to_langchain_message(message) for message in messages]}
        )
        response = result["messages"][-1]
        if not isinstance(response, AIMessage) or not isinstance(response.content, str):
            raise ServiceError("AGENT_FAILED", "The agent returned an unsupported response.", 502)
        return response.content
