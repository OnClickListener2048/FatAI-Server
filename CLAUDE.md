# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FatAI Server is a FastAPI backend for a local, user-owned AI assistant. It provides SSE streaming chat with BYOK (bring-your-own-key) OpenAI-compatible models, server-side tool execution (web search, weather), JWT auth, and an offline-first durable sync protocol for workspaces/conversations/messages/memories/prompts. See `README.md` (Chinese) and `AGENTS.md` (English contributor guidelines) for the basics.

Python 3.12+, managed with [uv](https://docs.astral.sh/uv/).

## Commands

```bash
uv sync                          # Install dependencies (committed uv.lock)
uv run python main.py            # Start dev server on :8080 with hot reload
uv run python -m unittest discover tests   # Run all tests
uv run python -m unittest tests.test_model_configurations   # Run the single test file
```

Open `http://127.0.0.1:8080/docs` for the interactive OpenAPI schema.

## Architecture: Two-Router Split

The API is served by two routers mounted at `/v1` in `app/main.py`:

- **`app/api/routes.py`** — Compatibility/tool endpoints (`/tools/search`, `/tools/weather`, `/tools/document-read`) and the main SSE chat endpoint (`/chat/stream`). These were the original routes and are largely self-contained.
- **`app/api/domain_routes.py`** — Everything else: auth (`/auth/register`, `/auth/login`, `/auth/device`), CRUD for workspaces/conversations/messages/memories/prompts/files/settings, model configurations, agent run, sync protocol, and the server-assembled `/conversations/{id}/generate`.

A key shared dependency: `routes.py` imports `record_change` and `entity_payload` from `domain_routes.py` so chat-turn persistence also writes to the sync change stream.

## SSE Streaming Architecture

`POST /v1/chat/stream` (`routes.py`) and `POST /v1/conversations/{id}/generate` (`domain_routes.py`) both return `text/event-stream`. The core service is `LangChainChatService` (`app/services/chat.py`).

### Two streaming paths

1. **Normal mode** (`thinking=False`): Uses LangChain's `ChatOpenAI` with tool binding. A tool-calling loop runs up to `MAX_TOOL_ROUNDS` (2) rounds. Between rounds, content is buffered with `NARRATION_BUFFER_CHARS` (200) to suppress short tool-round narration ("let me search...") that would otherwise flash briefly then disappear when tool calls follow. After the last tool round, an unbounded final answer pass streams through.

2. **Thinking mode** (`thinking=True`): Bypasses LangChain entirely — `ChatOpenAI` discards `reasoning_content` from provider SSE. Instead, `_stream_direct` calls the provider's `/chat/completions` directly via httpx and yields both `reasoning` and `content` deltas. Tool execution still runs in a loop. The base URL defaults to `https://api.deepseek.com` when none is configured (DeepSeek is the primary thinking-mode provider).

### Tool execution

`ServerToolExecutor` runs in the chat loop. Supported tools: `web_search`, `weather`. Each round's tool calls execute **concurrently** (`asyncio.gather`). Source URLs are deduplicated across all rounds. Tool output is capped at `MAX_TOOL_OUTPUT_CHARACTERS` (24,000).

### SSE events

Three event types: `message` (incremental content, can include `reasoning_content`), `tool_call` (with structured `sources`), and `done`.

## Server-Side Context Assembly

`assemble_context()` in `app/services/context.py` is the single place where the system prompt and reference data are layered. The fixed order is:

1. Core policy (`SYSTEM_PROMPT` — provider-neutral instructions, response language, instruction priority)
2. Enabled prompt templates (scoped to workspace or global, ordered by priority desc)
3. Workspace instruction (name + system_prompt)
4. Memories (GLOBAL + WORKSPACE + CONVERSATION scope, last 20 by updated_at)
5. Last 20 history turns from the client
6. Tool results (appended as system messages, marked as reference data only)

The policy explicitly prevents reference data (memories, tool results, history) from overriding core instructions. The `{responseLanguageTag}` placeholder is replaced with the client's language tag.

## BYOK Model: Credential Encryption

Users configure their own model providers via `POST /v1/model-configurations`. The API key is encrypted at rest:

- A Fernet cipher key is derived from `SHA-256(JWT_SECRET)` (`app/services/model_configurations.py:_cipher`)
- Only one configuration can be `is_active` at a time; activating one deactivates others
- `get_user_model_credentials()` resolves credentials: by `model_configuration_id` if given, otherwise the active configuration
- The ciphertext is **never** returned by any API; sync payloads and snapshots strip `api_key_ciphertext`
- **Rotating `JWT_SECRET` invalidates all stored API keys** — this coupling is intentional to keep provider keys out of environment variables

## Durable Sync Protocol

Three tables implement an offline-first sync in `app/db.py`:

| Table | Purpose |
|-------|---------|
| `sync_operations` | Idempotency: each `operation_id` is applied at most once. Racing duplicates return the same stored response. |
| `sync_entity_states` | Per-entity `sequence` counter (user_id, entity_type, entity_id). UPSERTs apply only when `payload.sequence > state.sequence`. |
| `sync_changes` | Append-only feed with auto-increment `cursor`. Other devices consume `/v1/sync/changes?cursor=N`. |

**Delete-wins**: DELETEs always apply regardless of sequence. This is necessary because chat messages saved server-side get server-assigned sequences the client cannot predict, so a client-initiated delete must still take effect.

`record_change()` (`domain_routes.py`) is the server-side hook for REST-originated writes to enter the change stream — it assigns a sequence and writes a `SyncChange` row. This ensures all devices converge whether mutations come from sync or REST.

## Chat Persistence Flow

When `POST /v1/chat/stream` includes `conversation_id` and `assistant_message_id`, `persist_chat_turn()` (`routes.py`) saves the turn directly:

1. Upserts the conversation (creates if missing, checks ownership)
2. Saves the user message and assistant answer as `Message` rows
3. Writes both to the sync change stream via `record_change()`
4. Wrapped in `asyncio.shield` in the finally block so a client disconnect still persists partial output
5. After the stream, a detached background task calls the model to generate a title (only if the conversation still has the default title) and syncs it

This means the client no longer syncs chat messages — the server owns the persistence and the client receives updates through the change stream.

## Database

SQLAlchemy 2.0 async with `aiosqlite` (default) or `asyncpg` (PostgreSQL). **No migrations** — `Base.metadata.create_all` runs at startup. Tables: `users`, `workspaces`, `conversations`, `model_configurations`, `messages`, `memory_entries`, `prompt_templates`, `file_assets`, `knowledge_documents`, `app_settings`, `sync_operations`, `sync_entity_states`, `sync_changes`.

All domain models use `IdTimestampMixin` (UUID PK, created_at, updated_at). Database sessions are obtained via FastAPI dependency (`get_session` → `SessionLocal`).

## Key Design Decisions

- **Chat tool execution is server-side only** — clients advertise tool definitions but never execute them. This keeps the tool loop simple and means the model always sees consistent tool output formatting.
- **Thinking mode uses raw httpx SSE, not LangChain** — because LangChain's `ChatOpenAI` drops `reasoning_content` chunks. The raw path manually parses SSE `data:` lines and merges streaming tool-call deltas by index.
- **`JWT_SECRET` is dual-purpose** — it signs access tokens AND derives the Fernet key for API key encryption. This is a deliberate trade-off: no separate encryption key to manage, but rotation requires re-entering all provider keys.
- **No Alembic** — schema changes require manual handling. The DB is auto-created at startup for development convenience.
- **`ENABLE_CHAT_TOOLS` and `ENABLE_CHAT_PERSIST` toggles** — referenced in recent commits as passthrough testing controls; check current state in `app/api/routes.py` before assuming tools/persistence are active.
