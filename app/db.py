import uuid
from datetime import datetime

import sqlalchemy
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


class IdTimestampMixin:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class User(IdTimestampMixin, Base):
    __tablename__ = "users"
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(512))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Workspace(IdTimestampMixin, Base):
    __tablename__ = "workspaces"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)


class Conversation(IdTimestampMixin, Base):
    __tablename__ = "conversations"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    title: Mapped[str] = mapped_column(String(240), default="New conversation")
    provider_type: Mapped[str] = mapped_column(String(64), default="OpenAI")
    model: Mapped[str] = mapped_column(String(256))
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)


class ModelConfiguration(IdTimestampMixin, Base):
    __tablename__ = "model_configurations"
    __table_args__ = (UniqueConstraint("user_id", "id", name="uq_model_configurations_user_id"),)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    provider_type: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(160))
    api_key_ciphertext: Mapped[str] = mapped_column(Text)
    base_url: Mapped[str] = mapped_column(String(1024))
    model: Mapped[str] = mapped_column(String(256))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Message(IdTimestampMixin, Base):
    __tablename__ = "messages"
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    reasoning_content: Mapped[str] = mapped_column(Text, default="")
    content_type: Mapped[str] = mapped_column(String(32), default="Markdown")


class MemoryEntry(IdTimestampMixin, Base):
    __tablename__ = "memory_entries"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    workspace_id: Mapped[str | None] = mapped_column(ForeignKey("workspaces.id"), nullable=True, index=True)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    scope: Mapped[str] = mapped_column(String(32))
    kind: Mapped[str] = mapped_column(String(32), default="FACT")
    content: Mapped[str] = mapped_column(Text)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)


class PromptTemplate(IdTimestampMixin, Base):
    __tablename__ = "prompt_templates"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    workspace_id: Mapped[str | None] = mapped_column(ForeignKey("workspaces.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(160))
    content: Mapped[str] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, default=100)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class FileAsset(IdTimestampMixin, Base):
    __tablename__ = "file_assets"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    workspace_id: Mapped[str | None] = mapped_column(ForeignKey("workspaces.id"), nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    message_id: Mapped[str | None] = mapped_column(ForeignKey("messages.id"), nullable=True)
    display_name: Mapped[str] = mapped_column(String(512))
    mime_type: Mapped[str] = mapped_column(String(160))
    size_bytes: Mapped[int] = mapped_column(Integer)
    storage_path: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(32), default="UPLOADED")


class KnowledgeDocument(IdTimestampMixin, Base):
    __tablename__ = "knowledge_documents"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    workspace_id: Mapped[str | None] = mapped_column(ForeignKey("workspaces.id"), nullable=True, index=True)
    file_asset_id: Mapped[str] = mapped_column(ForeignKey("file_assets.id"), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")
    extracted_markdown: Mapped[str] = mapped_column(Text, default="")


class AppSetting(Base):
    __tablename__ = "app_settings"
    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_app_settings_user_key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    key: Mapped[str] = mapped_column(String(120))
    value: Mapped[str] = mapped_column(Text)


class SyncOperation(Base):
    """Durable idempotency record for a client mutation."""

    __tablename__ = "sync_operations"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str] = mapped_column(String(64))
    operation: Mapped[str] = mapped_column(String(16))
    sequence: Mapped[int] = mapped_column(Integer)
    applied: Mapped[bool] = mapped_column(Boolean, default=True)
    cursor: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SyncEntityState(Base):
    """Latest accepted sequence for each user-owned entity."""

    __tablename__ = "sync_entity_states"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SyncChange(Base):
    """Append-only change feed consumed by other client devices."""

    __tablename__ = "sync_changes"
    cursor: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    operation_id: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str] = mapped_column(String(64))
    operation: Mapped[str] = mapped_column(String(16))
    sequence: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


engine = create_async_engine(
    get_settings().database_url,
    future=True,
    connect_args={
        # WAL mode allows concurrent reads and writes, avoiding "database is locked".
        "statement_cache_size": 0,
    },
)


@sqlalchemy.event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record):
    """Enable WAL journal and a 5 s busy timeout on every new SQLite connection."""
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        cursor.close()
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def initialize_database() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def get_session():
    async with SessionLocal() as session:
        yield session
