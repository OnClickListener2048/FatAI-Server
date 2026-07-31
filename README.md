# FatAI Server

FastAPI backend for FatAI. It replaces the former Ktor tool server and exposes compatible search
and weather endpoints, plus a safer multipart document-read endpoint and a LangChain streaming
chat boundary. It owns user authentication, synced workspace/chat state, prompts, memories,
attachments, knowledge-document queue records, and account settings.

## Start

```powershell
python -m pip install -e .
python main.py
```

The server listens on `http://127.0.0.1:8080`; its generated API documentation is at `/docs`.

## Endpoints

- `GET /health`
- `POST /v1/tools/search` — preserves the Kotlin server request/response contract.
- `POST /v1/tools/weather` — preserves the Kotlin server request/response contract.
- `POST /v1/tools/document-read` — accepts `multipart/form-data` with a `file` part and forwards it
  to Docling. During desktop migration it also accepts the legacy JSON `localPath` request when
  `ALLOW_LOCAL_DOCUMENT_PATHS=true`; set it to `false` before remote deployment.
- `POST /v1/chat/stream` — Server-Sent Events through LangChain's OpenAI-compatible adapter.

Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) before using chat streaming. LangGraph is
included for the next migration stage: durable agent workflows should be built in a separate
`app/agents` package, with tool permissions and persisted execution state.
