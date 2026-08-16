# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FatAI Server is a FastAPI backend for a local, user-owned AI assistant. It provides SSE streaming chat with BYOK (bring-your-own-key) OpenAI-compatible models, server-side tool execution (web search, weather), JWT auth, and an offline-first durable sync protocol for workspaces/conversations/messages/memories/prompts. See `README.md` (Chinese) and `AGENTS.md` (English contributor guidelines) for the basics.

Python 3.12+, managed with [uv](https://docs.astral.sh/uv/).

### Local needle2 engine (Windows-only, pinned via uv sources)

`app/services/local_needle.py` uses `cactus-needle` for on-device conversation titles.
PyPI's `cactus-needle` (2.0.5+) ships only a universal wheel whose first run downloads the
platform engine from `huggingface.co` (unreachable from this network). This repo therefore
pins the verified `win_amd64` wheel in `wheels/` (engine + model embedded in
`libneedle.dll`, no runtime download) via `[tool.uv.sources]`:

```toml
[tool.uv.sources]
cactus-needle = { path = "wheels/cactus_needle-2.0.1-py3-none-win_amd64.whl" }
```

A plain `uv sync` installs from the pinned wheel; on non-Windows machines the install
fails, which is fine — imports are lazy and `generate_title_local` returns `None` when
`needle` is absent, so the server falls back to the cloud title path.

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

**Invoke-markup recovery**: deepseek-chat sometimes emits its official-app web-search invocation (`<invoke name="web_search"><parameter name="query">…</parameter></invoke>`) as plain text instead of structured `tool_calls`. Both streaming paths scan the full round content for that markup when no structured calls arrived (`_extract_invoke_calls` in `app/services/chat.py`) and execute it as a real tool call, restricted to the tool names actually bound this request. The markup is stripped from the assistant turn content.

### Tool execution

`ServerToolExecutor` runs in the chat loop. Supported tools: `web_search`, `weather`. Each round's tool calls execute **concurrently** (`asyncio.gather`). Source URLs are deduplicated across all rounds. Tool output is capped at `MAX_TOOL_OUTPUT_CHARACTERS` (24,000).

### SSE events

Three event types: `message` (incremental content, can include `reasoning_content`), `tool_call` (with structured `sources`), and `done`. The `done` event carries `{"sources": [...]}` — the RAG references injected into the context as `{title, kind, id}` (`kind` is `memory` or `knowledge_document`); empty array when nothing was injected.

## Server-Side Context Assembly

`assemble_context()` in `app/services/context.py` is the single place where the system prompt and reference data are layered. The fixed order is:

1. Core policy (`SYSTEM_PROMPT` — provider-neutral instructions, response language, instruction priority)
2. Enabled prompt templates (scoped to workspace or global, ordered by priority desc)
3. Workspace instruction (name + system_prompt)
4. RAG-recalled memories and knowledge documents (when the retriever is available; see below)
5. Memories fallback (last 20 by `updated_at`) when retrieval is unavailable or returns nothing
6. Last 20 history turns from the client
7. Tool results (appended as system messages, marked as reference data only)

The policy explicitly prevents reference data (memories, tool results, history) from overriding core instructions. The `{responseLanguageTag}` placeholder is replaced with the client's language tag.

## RAG (Server-Side Retrieval)

`app/services/rag/` implements retrieval over memories and knowledge documents:

- **`embedding.py`** — OpenAI-compatible `POST {base_url}/embeddings` client (default: local Ollama `bge-m3`, 1024 dims). Responses are sorted by `index`; vectors are normalized so cosine == dot product.
- **`chunking.py`** — semantic chunking + structure awareness (current mainstream approach): heading stack → `path > path` prefixes; overlong sections are embedded sentence-by-sentence (one extra batch call) and split at low-similarity boundaries (LlamaIndex SemanticSplitterNodeParser style; boundaries above `BREAK_THRESHOLD` 0.5 never hard-split); target chunk size = 70% of `RAG_CHUNK_CHARS`, so chunks may slightly exceed the cap instead of breaking mid-thought; uniform high-similarity text falls back to a sentence sliding window; chunks < 60 chars merge with the previous same-path chunk. Memories use plain sentence-window chunking (`chunk_text`, no overlap).
- **`vectorstore.py`** — chunk metadata + vectors in the SQLite database. `chunks_vec0` (sqlite-vec vec0 virtual table) provides ANN search; when the extension is unavailable, `document_chunks.embedding` BLOBs fall back to a Python cosine scan. `chunk_fts` (FTS5) holds jieba-tokenized content for BM25; when FTS5 is unavailable, `bm25_search` returns empty and retrieval degrades to vector-only. **The extension is loaded per connection** (SQLite extensions are per-connection); vec0 returns L2 distance, converted via `cos = 1 - L2²/2` (normalized vectors). vec0 metadata columns reject NULL — absent `workspace_id` is stored as `""`. FTS5 deletes only work by `rowid` — `_delete_fts_rows` resolves rowids first; deletion order is vec0/FTS before `document_chunks`. Writes go through a dedicated sync `sqlite3` connection via `asyncio.to_thread`, never SQLAlchemy (aiosqlite's `load_extension` is a coroutine).
- **`indexing.py`** — `index_memory()` hooks on memory create/update/archive (REST + sync paths via `domain_routes._reindex_memory`); `backfill_memories()` indexes memories with zero chunks on startup. `knowledge_document_worker()` polls `QUEUED` documents every `RAG_SWEEP_SECONDS`, CAS-claims rows (`PROCESSING`), converts via Docling, chunks, embeds, and marks `READY`/`FAILED`.
- **`retrieval.py`** — hybrid retrieval with RRF fusion. Two recall paths (each `_RAW_TOP_K`=50): dense (`vec0` ANN; hits below `RAG_MIN_SCORE` are dropped as a semantic quality floor) and sparse (jieba-tokenized FTS5 BM25; keyword-exact matches pass regardless of vector score — numbers, IDs, proper nouns). RRF merges both by rank: `score = Σ 1/(k + rank)`, k=60 (Cornack et al. 2009). Fused ranking is scope post-filtered (vec0 only supports equality aux-column filters): GLOBAL always, WORKSPACE/CONVERSATION by the request's IDs, knowledge docs only within the request's workspace. Empty/failed retrieval falls back to recent memories so chat never breaks.

The retriever is optional — `assemble_context(..., retriever=None)` disables RAG entirely. `app/main.py` constructs the services in lifespan and stores them on `app.state` (`rag_retriever`, `rag_worker`).

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

**Workspace id reclaim**: workspace ids are client-chosen constants (the default is literally `"inbox"`), so after a device identity change or DB restore the same id can still be owned by an orphaned user row. `apply_sync_payload` looks up by `(id, user_id)`, and when the row exists under another user it is adopted (reassigned to the current user) instead of crashing the INSERT on the global unique `workspaces.id` constraint. Only workspaces use fixed ids — every other entity type uses client-generated uuids, so cross-user collisions there are never reclaimed.

`record_change()` (`domain_routes.py`) is the server-side hook for REST-originated writes to enter the change stream — it assigns a sequence and writes a `SyncChange` row. This ensures all devices converge whether mutations come from sync or REST.

## Chat Persistence Flow

When `POST /v1/chat/stream` includes `conversation_id` and `assistant_message_id`, `persist_chat_turn()` (`routes.py`) saves the turn directly:

1. Upserts the conversation (creates if missing, checks ownership)
2. Saves the user message and assistant answer as `Message` rows
3. Writes both to the sync change stream via `record_change()`
4. Wrapped in `asyncio.shield` in the finally block so a client disconnect still persists partial output
5. After the stream, a detached background task titles the conversation (only if it still has the default title) and syncs it: the local needle2 engine (`app/services/local_needle.py`, Windows `cactus-needle` wheel with the engine+model embedded in `libneedle.dll`) goes first — it declares a single `set_title` tool and takes `function_calls[0].arguments.title`; a refusal or a missing wheel falls back to the cloud `generate_conversation_title` (`titles.py`), which summarizes the opening exchange (first 4 user/assistant turns, rendered by `transcript_for_title`) into a ≤20-character title rather than echoing the first message. Needle titles cost no cloud tokens, so `record_token_usage(..., "title", ...)` is only called for cloud titles

This means the client no longer syncs chat messages — the server owns the persistence and the client receives updates through the change stream.

## Database

SQLAlchemy 2.0 async with `aiosqlite` (default) or `asyncpg` (PostgreSQL). **No migrations** — `Base.metadata.create_all` runs at startup. Tables: `users`, `workspaces`, `conversations`, `model_configurations`, `messages`, `memory_entries`, `prompt_templates`, `file_assets`, `knowledge_documents`, `app_settings`, `sync_operations`, `sync_entity_states`, `sync_changes`.

The RAG vector store lives **inside the same SQLite file**: the `document_chunks` table (metadata + scan-mode embedding BLOB), the `chunks_vec0` vec0 virtual table, and the `chunk_fts` FTS5 table (jieba-tokenized content for BM25) are created by `VectorStore.initialize()` at startup. On PostgreSQL (`DATABASE_URL` non-SQLite) the vector store is disabled and retrieval falls back to recency — embeddings are not stored in Postgres.

All domain models use `IdTimestampMixin` (UUID PK, created_at, updated_at). Database sessions are obtained via FastAPI dependency (`get_session` → `SessionLocal`).

## Key Design Decisions

- **Chat tool execution is server-side only** — clients advertise tool definitions but never execute them. This keeps the tool loop simple and means the model always sees consistent tool output formatting.
- **Thinking mode uses raw httpx SSE, not LangChain** — because LangChain's `ChatOpenAI` drops `reasoning_content` chunks. The raw path manually parses SSE `data:` lines and merges streaming tool-call deltas by index.
- **`JWT_SECRET` is dual-purpose** — it signs access tokens AND derives the Fernet key for API key encryption. This is a deliberate trade-off: no separate encryption key to manage, but rotation requires re-entering all provider keys.
- **No Alembic** — schema changes require manual handling. The DB is auto-created at startup for development convenience.
- **`ENABLE_CHAT_TOOLS` and `ENABLE_CHAT_PERSIST` toggles** — referenced in recent commits as passthrough testing controls; check current state in `app/api/routes.py` before assuming tools/persistence are active.
