import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_temp_directory = tempfile.TemporaryDirectory()
_database_path = Path(_temp_directory.name) / "verification.db"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_database_path.as_posix()}"

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db import ModelConfiguration, SessionLocal, engine
from app.main import create_app
from app.models import ChatMessageInput, ChatStreamRequest, ToolDefinitionInput
from app.services.chat import LangChainChatService
from app.services.model_configurations import UserModelCredentials, decrypt_api_key


class ModelConfigurationTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        asyncio.run(engine.dispose())
        _temp_directory.cleanup()

    def setUp(self) -> None:
        self.client = TestClient(create_app())
        self.client.__enter__()
        response = self.client.post(
            "/v1/auth/device",
            json={
                "device_id": "123e4567-e89b-12d3-a456-426614174000",
                "display_name": "Test device",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.headers = {"Authorization": f"Bearer {response.json()['access_token']}"}

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)

    def test_user_configuration_is_encrypted_and_required_for_chat(self) -> None:
        chat_response = self.client.post(
            "/v1/chat/stream",
            headers=self.headers,
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        self.assertEqual(chat_response.status_code, 503, chat_response.text)
        self.assertEqual(chat_response.json()["code"], "MODEL_NOT_CONFIGURED")

        configuration_id = "123e4567-e89b-12d3-a456-426614174001"
        secret = "verification-secret"
        response = self.client.post(
            "/v1/model-configurations",
            headers=self.headers,
            json={
                "id": configuration_id,
                "name": "Test OpenAI",
                "provider_type": "OpenAI",
                "api_key": secret,
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "is_active": True,
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertNotIn("api_key", response.json())

        async def verify() -> None:
            async with SessionLocal() as session:
                record = await session.get(ModelConfiguration, configuration_id)
                self.assertIsNotNone(record)
                assert record is not None
                self.assertNotEqual(record.api_key_ciphertext, secret)
                self.assertEqual(decrypt_api_key(record.api_key_ciphertext, get_settings()), secret)

        asyncio.run(verify())

        async def model_stream(*_args, **_kwargs):
            yield "hello", []

        with patch("app.api.routes.LangChainChatService.stream", model_stream):
            stream_response = self.client.post(
                "/v1/chat/stream",
                headers=self.headers,
                json={
                    "model_configuration_id": configuration_id,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        self.assertEqual(stream_response.status_code, 200, stream_response.text)
        self.assertTrue(stream_response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(stream_response.text, 'event: message\ndata: {"content": "hello"}\n\nevent: done\ndata: {}\n\n')

    def test_tool_enabled_chat_streams_model_chunks(self) -> None:
        class Chunk:
            def __init__(self, content: str) -> None:
                self.content = content
                self.tool_calls: list[dict] = []

            def __add__(self, other: "Chunk") -> "Chunk":
                return Chunk(self.content + other.content)

        class StreamingModel:
            def bind_tools(self, _tools):
                return self

            async def astream(self, _messages):
                yield Chunk("stream")
                yield Chunk("ed")

        model = StreamingModel()
        request = ChatStreamRequest(
            messages=[ChatMessageInput(role="user", content="Hello")],
            tools=[ToolDefinitionInput(name="weather", description="Get weather")],
        )

        async def collect() -> list[tuple[str, list[dict[str, object]]]]:
            service = LangChainChatService(
                UserModelCredentials("test-key", "https://example.test/v1", "test-model")
            )
            with patch("app.services.chat.ChatOpenAI", return_value=model):
                return [event async for event in service.stream(request)]

        self.assertEqual(asyncio.run(collect()), [("stream", []), ("ed", [])])

    def test_server_executes_tools_inside_stream(self) -> None:
        from langchain_core.messages import AIMessage, ToolMessage

        class Chunk:
            def __init__(self, content: str, tool_calls: list[dict] | None = None) -> None:
                self.content = content
                self.tool_calls = tool_calls or []

            def __add__(self, other: "Chunk") -> "Chunk":
                return Chunk(self.content + other.content, self.tool_calls + other.tool_calls)

            def to_message(self) -> AIMessage:
                return AIMessage(content=self.content, tool_calls=self.tool_calls)

        class StreamingModel:
            def __init__(self) -> None:
                self.rounds = 0
                self.recorded_messages: list[list[object]] = []

            def bind_tools(self, tools):
                self.bound_tools = tools
                return self

            async def astream(self, messages):
                self.recorded_messages.append(list(messages))
                if self.rounds == 0:
                    self.rounds += 1
                    yield Chunk("", [{"id": "call_1", "name": "web_search", "args": {"query": "test"}}])
                else:
                    yield Chunk("answer")

        class RecordingExecutor:
            def __init__(self) -> None:
                self.executed: list[tuple[str, dict]] = []

            def bindable(self, requested):
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {"type": "object", "properties": {}, "required": []},
                        },
                    }
                    for tool in requested
                ]

            async def execute(self, name: str, arguments: dict):
                self.executed.append((name, arguments))
                return "result"

        model = StreamingModel()
        executor = RecordingExecutor()
        request = ChatStreamRequest(
            messages=[ChatMessageInput(role="user", content="Hello")],
            tools=[ToolDefinitionInput(name="web_search", description="Search")],
        )

        async def collect() -> list[tuple[str, list[dict[str, object]]]]:
            service = LangChainChatService(
                UserModelCredentials("test-key", "https://example.test/v1", "test-model"),
                executor,
            )
            with patch("app.services.chat.ChatOpenAI", return_value=model):
                return [event async for event in service.stream(request)]

        events = asyncio.run(collect())
        self.assertEqual(
            events,
            [
                ("", [{"id": "call_1", "name": "web_search", "arguments": {"query": "test"}}]),
                ("answer", []),
            ],
        )
        self.assertEqual(executor.executed, [("web_search", {"query": "test"})])
        self.assertEqual(len(model.recorded_messages), 2)
        self.assertEqual(len(model.recorded_messages[1]), 3)
        self.assertIsInstance(model.recorded_messages[1][-1], ToolMessage)
        self.assertEqual(model.recorded_messages[1][-1].content, "result")

    def test_server_assembles_context_from_database(self) -> None:
        async def seed_and_verify() -> None:
            from app.db import MemoryEntry, PromptTemplate, User, Workspace

            async with SessionLocal() as session:
                user_id = "context-test-user"
                workspace_id = "context-workspace"
                session.add(User(id=user_id, email="context@device.fatai.local", display_name="Context", password_hash="x"))
                session.add_all(
                    [
                        Workspace(id=workspace_id, user_id=user_id, name="Research", system_prompt="Be precise."),
                        PromptTemplate(
                            id="tpl-1",
                            user_id=user_id,
                            workspace_id=workspace_id,
                            name="Style",
                            content="Answer in short bullets.",
                            priority=100,
                            is_enabled=True,
                        ),
                        MemoryEntry(
                            id="mem-1",
                            user_id=user_id,
                            workspace_id=workspace_id,
                            scope="WORKSPACE",
                            kind="FACT",
                            content="User prefers English.",
                            is_archived=False,
                        ),
                    ]
                )
                await session.commit()

                from app.services.context import assemble_context

                user = await session.get(User, user_id)
                history = [
                    ChatMessageInput(role="user", content="What is the weather?"),
                    ChatMessageInput(role="assistant", content="Let me check."),
                ]
                messages = await assemble_context(
                    session,
                    user,
                    workspace_id,
                    None,
                    history,
                    "zh",
                    tool_results=["Document: report.pdf\nExtracted content."],
                )
                self.assertEqual(len(messages), 7)
                self.assertEqual(messages[0].role, "system")
                self.assertIn("The active application language is zh", messages[0].content)
                self.assertIn("User-configured application instruction (Style)", messages[1].content)
                self.assertIn("Current workspace: Research", messages[2].content)
                self.assertIn("Be precise.", messages[2].content)
                self.assertIn("User prefers English.", messages[3].content)
                self.assertEqual(messages[4].content, "What is the weather?")
                self.assertEqual(messages[5].content, "Let me check.")
                self.assertEqual(messages[6].role, "system")
                self.assertIn("Document: report.pdf", messages[6].content)

                standalone = await assemble_context(
                    session,
                    user,
                    None,
                    None,
                    [ChatMessageInput(role="user", content="Hello")],
                    "en",
                )
                self.assertEqual(len(standalone), 2)
                self.assertEqual(standalone[0].role, "system")
                self.assertIn("The active application language is en", standalone[0].content)
                self.assertEqual(standalone[1].content, "Hello")

        asyncio.run(seed_and_verify())

    def test_delete_applies_even_with_stale_sequence(self) -> None:
        response = self.client.post(
            "/v1/sync/operations",
            headers=self.headers,
            json={
                "operation_id": "op-upsert-1",
                "entity_type": "message",
                "entity_id": "msg-stale-delete",
                "operation": "UPSERT",
                "sequence": 5,
                "schema_version": 1,
                "payload": {"id": "msg-stale-delete", "conversation_id": "any-conversation", "role": "user", "content": "hello"},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["applied"])

        delete_response = self.client.post(
            "/v1/sync/operations",
            headers=self.headers,
            json={
                "operation_id": "op-delete-1",
                "entity_type": "message",
                "entity_id": "msg-stale-delete",
                "operation": "DELETE",
                "sequence": 1,
                "schema_version": 1,
                "payload": {},
            },
        )
        self.assertEqual(delete_response.status_code, 200, delete_response.text)
        self.assertTrue(delete_response.json()["applied"])

        snapshot = self.client.get("/v1/sync/snapshot", headers=self.headers).json()
        self.assertNotIn(
            "msg-stale-delete",
            [entity["entity_id"] for entity in snapshot["entities"] if entity["entity_type"] == "message"],
        )

    def test_chat_turn_persists_messages_and_ensures_conversation(self) -> None:
        async def verify() -> None:
            from app.api.routes import persist_chat_turn
            from app.db import Conversation, Message, SyncChange, User, Workspace
            from app.models import ChatStreamRequest
            from sqlalchemy import select

            async with SessionLocal() as session:
                session.add(User(id="chat-persist-user", email="chat@device.fatai.local", display_name="Chat", password_hash="x"))
                session.add(Workspace(id="chat-persist-workspace", user_id="chat-persist-user", name="W"))
                await session.commit()
                user = await session.get(User, "chat-persist-user")
                payload = ChatStreamRequest(
                    messages=[ChatMessageInput(role="user", content="What is the weather?")],
                    conversation_id="chat-saved-conversation",
                    workspace_id="chat-persist-workspace",
                    user_message_id="chat-user-msg",
                    assistant_message_id="chat-assistant-msg",
                    model="test-model",
                )
                await persist_chat_turn(session, user, payload, "Sunny.")

                conversation = await session.get(Conversation, "chat-saved-conversation")
                self.assertIsNotNone(conversation)
                self.assertEqual(conversation.model, "test-model")

                user_message = await session.get(Message, "chat-user-msg")
                assistant_message = await session.get(Message, "chat-assistant-msg")
                self.assertEqual(user_message.content, "What is the weather?")
                self.assertEqual(assistant_message.content, "Sunny.")
                self.assertEqual(assistant_message.conversation_id, "chat-saved-conversation")

                changes = list(
                    await session.scalars(
                        select(SyncChange).where(SyncChange.entity_id.in_(["chat-user-msg", "chat-assistant-msg"]))
                    )
                )
                self.assertEqual(len(changes), 2)

        asyncio.run(verify())
