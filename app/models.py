from typing import Any, Literal

from pydantic import BaseModel, Field


class ApiError(BaseModel):
    code: str
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str


class WebSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    max_results: int = Field(default=5, ge=1, le=10)


class WebSearchResult(BaseModel):
    title: str
    snippet: str = ""
    url: str
    source: str


class WebSearchResponse(BaseModel):
    query: str
    results: list[WebSearchResult]


class WeatherRequest(BaseModel):
    location: str = Field(min_length=1, max_length=256)
    max_results: int = Field(default=3, ge=1, le=5)


class WeatherResponse(BaseModel):
    location: str
    results: list[WebSearchResult]


class DocumentReadResponse(BaseModel):
    displayName: str
    markdown: str


class ChatMessageInput(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ToolParameterInput(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str
    required: bool = False
    allowed_values: list[str] = Field(default_factory=list)


class ToolDefinitionInput(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str
    parameters: list[ToolParameterInput] = Field(default_factory=list)


class ChatStreamRequest(BaseModel):
    messages: list[ChatMessageInput] = Field(min_length=1)
    model: str | None = Field(default=None, max_length=256)
    model_configuration_id: str | None = Field(default=None, min_length=1, max_length=64)
    temperature: float = Field(default=0.7, ge=0, le=2)
    thinking: bool = False
    tools: list[ToolDefinitionInput] = Field(default_factory=list)
    workspace_id: str | None = Field(default=None, max_length=64)
    conversation_id: str | None = Field(default=None, max_length=64)
    response_language_tag: str = Field(default="en", max_length=32)
    tool_results: list[str] = Field(default_factory=list)
    include_contextual_references: bool = True
    user_message_id: str | None = Field(default=None, min_length=1, max_length=64)
    assistant_message_id: str | None = Field(default=None, min_length=1, max_length=64)


class SyncOperationInput(BaseModel):
    operation_id: str = Field(min_length=1, max_length=64)
    entity_type: Literal[
        "workspace",
        "conversation",
        "message",
        "memory",
        "prompt_template",
        "model_configuration",
        "file_asset",
        "setting",
    ]
    entity_id: str = Field(min_length=1, max_length=64)
    operation: Literal["UPSERT", "DELETE"]
    sequence: int = Field(ge=1)
    schema_version: int = Field(default=1, ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class SyncOperationResponse(BaseModel):
    operation_id: str
    entity_type: str
    entity_id: str
    sequence: int
    applied: bool
    cursor: int | None = None


class SyncChangeResponse(BaseModel):
    cursor: int
    operation_id: str
    entity_type: str
    entity_id: str
    operation: Literal["UPSERT", "DELETE"]
    sequence: int
    schema_version: int = 1
    payload: dict[str, Any] = Field(default_factory=dict)


class SyncChangesResponse(BaseModel):
    changes: list[SyncChangeResponse]
    next_cursor: int
    has_more: bool


class SyncSnapshotResponse(BaseModel):
    entities: list[SyncChangeResponse]
    cursor: int
