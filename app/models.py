from typing import Literal

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
    temperature: float = Field(default=0.7, ge=0, le=2)
    tools: list[ToolDefinitionInput] = Field(default_factory=list)
