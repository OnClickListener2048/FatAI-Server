# FatAI Server API

Base URL: `http://127.0.0.1:8080` by default. The interactive OpenAPI specification is available
at `/docs`, and the machine-readable specification is `/openapi.json`.

Unless marked otherwise, domain APIs require `Authorization: Bearer <access_token>`. JSON bodies
use `snake_case`; timestamps are ISO 8601 values. Error responses use:

```json
{"code":"MODEL_NOT_CONFIGURED","message":"Configure and select a model provider before starting a chat."}
```

## Authentication and model configuration

| Method | Path | Authentication | Purpose |
| --- | --- | --- | --- |
| `POST` | `/v1/auth/register` | No | Create an email/password account. |
| `POST` | `/v1/auth/login` | No | Exchange email/password for a bearer token. |
| `POST` | `/v1/auth/device` | No | Create or resume a local device account. |
| `GET` | `/v1/users/me` | Yes | Return the current user. |
| `POST` | `/v1/model-configurations` | Yes | Create or update a user-owned provider configuration. |
| `POST` | `/v1/model-configurations/{id}/activate` | Yes | Make one configuration active. |
| `DELETE` | `/v1/model-configurations/{id}` | Yes | Delete a configuration. |

`POST /v1/model-configurations` accepts:

```json
{
  "id": "client-generated-uuid",
  "name": "My DeepSeek",
  "provider_type": "DeepSeek",
  "api_key": "provider-secret",
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash",
  "is_active": true
}
```

The API key is encrypted at rest and is never returned. Clients retain only the configuration
metadata after this request succeeds.

## Chat streaming

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/stream` | Stream a response to client-supplied conversation turns. |
| `POST` | `/v1/conversations/{id}/generate` | Rebuild stored context and stream an assistant response. |
| `POST` | `/v1/agents/run` | Run the current user's basic agent graph. |

`POST /v1/chat/stream` accepts `messages` (raw conversation turns, not pre-assembled), optional
`model`, optional `model_configuration_id`, `temperature` (`0`–`2`), optional function `tools`,
optional `workspace_id` and `conversation_id`, optional `response_language_tag` (default `en`),
optional `tool_results` (string array of transient client-side tool output), optional
`include_contextual_references` (default `true`), and optional `user_message_id` /
`assistant_message_id` (the client-owned ids under which the turn is persisted).

Context is assembled server-side (`app/services/context.py`): the core policy, enabled prompt
templates, the workspace instruction, recalled memories, and the history limit (last 20 turns)
are layered in a fixed order; `tool_results` are appended after history. Clients therefore only
send their own turns and never influence the instruction layers.

Conversation chats (with `conversation_id` and `assistant_message_id`) are persisted directly:
the server upserts the conversation when missing, saves the user turn and the assistant answer,
and records both in the change stream, so the client no longer re-syncs chat messages. A
disconnected stream still saves whatever was streamed so far.

It responds with `Content-Type: text/event-stream`:

```text
event: message
data: {"content":"Hello"}

event: tool_call
data: {"id":"call_1","name":"web_search","arguments":{"query":"iPhone 17 price"},"sources":[{"title":"Apple","url":"https://www.apple.com/iphone/"}]}

event: done
data: {}
```

`message` events are emitted incrementally, including when tool definitions are supplied.
`tool_call` events carry the structured `sources` (title and URL) produced by server-side tool
execution, deduplicated by URL across the whole request.

## Workspaces, conversations, and messages

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`, `POST` | `/v1/workspaces` | List or create workspaces. |
| `PATCH` | `/v1/workspaces/{id}` | Update a workspace. |
| `GET`, `POST` | `/v1/conversations` | List or create conversations. |
| `GET`, `POST` | `/v1/conversations/{id}/messages` | List or create messages. |

Conversation creation requires `workspace_id`, `model`, and optionally `id`, `title`, and
`provider_type`. Message creation requires `role` (`user`, `assistant`, `system`, or `tool`) and
`content`; `id`, `reasoning_content`, and `content_type` are optional.

## Memory, prompts, files, and settings

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`, `POST` | `/v1/memories` | List or create memories. |
| `POST` | `/v1/memories/{id}/archive` | Archive a memory. |
| `GET`, `POST` | `/v1/prompt-templates` | List or create prompt templates. |
| `POST` | `/v1/files` | Upload a file as `multipart/form-data` with a `file` part. |
| `POST` | `/v1/knowledge/documents/{file_id}` | Queue an uploaded file for knowledge processing. |
| `GET`, `PUT` | `/v1/settings/{key}` | Read or set a user setting (`{"value":"..."}`). |

Memory creation accepts `scope` (`GLOBAL`, `WORKSPACE`, or `CONVERSATION`) and `content`;
workspace/conversation IDs and `kind` (`FACT` or `SUMMARY`) are optional. Prompt templates accept
`name`, `content`, optional `workspace_id`, `priority`, and `is_enabled`.

## Durable synchronization

These authenticated endpoints support an offline-first client cache. Each client mutation
contains a client-generated `operation_id`, an `entity_type`/`entity_id` pair, and a monotonically
increasing `sequence` for that entity. Repeating the same `operation_id` is idempotent. Older
sequences are acknowledged but not applied, so late retries cannot overwrite newer state.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/sync/operations` | Apply one ordered, idempotent mutation. |
| `GET` | `/v1/sync/snapshot` | Return all current user-owned entities and the current change cursor to rebuild an empty client database. |
| `GET` | `/v1/sync/changes?cursor=<n>&limit=<n>` | Return changes after a cursor for incremental synchronization. |

`POST /v1/sync/operations` accepts `operation_id`, `entity_type`, `entity_id`, `operation`
(`UPSERT` or `DELETE`), `sequence`, `schema_version`, and the entity payload. Snapshot and change
responses never include provider API keys; secrets remain encrypted on the server. Clients should
apply snapshot entities first, persist the returned cursor, then consume changes in ascending
cursor order and advance the cursor only after each local batch is committed.

UPSERTs apply only when the payload `sequence` is higher than the entity's recorded sequence.
DELETEs always apply (delete-wins): chat messages are saved server-side with a sequence the
client cannot predict, so delete-wins keeps client-initiated deletes and regenerates working.

## Utility endpoints

| Method | Path | Authentication | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health` | No | Service readiness check. |
| `POST` | `/v1/tools/search` | No | Search with `query` and optional `max_results`. |
| `POST` | `/v1/tools/weather` | No | Weather lookup with `location` and optional `max_results`. |
| `POST` | `/v1/tools/document-read` | No | Read an uploaded document; local JSON paths are development-only. |
