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
