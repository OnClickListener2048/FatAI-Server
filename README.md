# FatAI Server

FatAI 的 FastAPI 后端，提供工具调用、流式 AI 对话、用户认证，以及工作区与聊天数据同步能力。默认使用 SQLite；数据库表会在启动时自动创建。

## 功能

- 搜索、天气查询和文档转 Markdown 工具接口；文档解析通过 Docling 服务完成。
- 用户自带密钥（BYOK）的 OpenAI 兼容模型 Server-Sent Events（SSE）流式聊天，以及基于 LangGraph 的基础 Agent 运行入口。
- JWT 用户认证，支持邮箱注册/登录和桌面端迁移期间的设备账号初始化。
- 按用户隔离的工作区、会话、消息、记忆、提示词模板与应用设置同步。
- 文件上传与知识库文档处理：入队后由后台 worker 调用 Docling 解析为 Markdown，语义分块并建立索引。
- 服务端 RAG：记忆与知识文档在写入/变更时异步索引（本地 bge-m3 嵌入；语义分块按句子相似度断句），聊天时混合检索（sqlite-vec 向量 + jieba/FTS5 BM25 关键词，RRF 融合，扩展不可用自动降级）召回相关记忆与文档片段注入上下文，并在 `done` 事件中返回引用来源。

## 快速开始

需要 Python 3.12+。推荐使用 [uv](https://docs.astral.sh/uv/)（仓库已提交 `uv.lock`）：

```powershell
uv sync
Copy-Item .env.example .env
uv run python main.py
```

也可使用 pip 安装依赖后执行 `python main.py`。服务默认监听 `http://127.0.0.1:8080`，OpenAPI 文档位于 `http://127.0.0.1:8080/docs`，健康检查为 `GET /health`。

完整的静态接口说明见 [docs/API.md](docs/API.md)；它包含鉴权、用户模型配置、SSE 事件与主要领域接口。

## 配置

从 `.env.example` 复制 `.env` 后按需设置：

| 变量 | 说明 |
| --- | --- |
| 用户模型密钥 | 在客户端的“模型提供商”中按用户添加；服务端以 JWT 认证的用户 ID 隔离保存，并用 `JWT_SECRET` 派生的密钥加密后存储。 |
| `JWT_SECRET` | 访问令牌签名密钥，同时用于加密用户模型密钥；部署时必须替换默认值并保持稳定。 |
| `DOCLING_SERVER_URL` | 文档解析服务地址，默认 `http://127.0.0.1:5001`。 |
| `DOCLING_TIMEOUT_SECONDS` | Docling 转换超时（默认 300 秒）。大图 OCR/多页 PDF 在本地 CPU 上常超过通用 20s 请求超时，使用独立的长超时 client。 |
| `DATABASE_URL` | SQLAlchemy 异步连接串，默认 `sqlite+aiosqlite:///./fat_ai.db`；可使用 `postgresql+asyncpg://...`。 |
| `JWT_SECRET` / `JWT_EXPIRATION_MINUTES` | 访问令牌签名密钥与有效期；生产环境必须替换默认密钥。 |
| `CORS_ORIGINS` | 允许的来源 JSON 数组，例如 `["http://localhost:3000"]`。 |
| `UPLOAD_DIRECTORY` / `MAX_DOCUMENT_SIZE_BYTES` | 上传文件目录和大小上限（默认 50 MiB）。 |
| `ALLOW_LOCAL_DOCUMENT_PATHS` | 仅为本地桌面端迁移保留 JSON `localPath` 读取；部署到远程环境前设为 `false`。 |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | 嵌入服务（OpenAI 兼容 `POST /embeddings`），默认本地 Ollama `http://127.0.0.1:11434/v1` + `bge-m3`（1024 维）。 |
| `RAG_TOP_K_MEMORY` / `RAG_TOP_K_DOCUMENT` / `RAG_MIN_SCORE` | 检索召回数与向量路相似度阈值（默认 8 / 5 / 0.55；bge-m3 对表格行/数字给分偏高，0.45 会漏进无关命中）。 |
| `RAG_CHUNK_CHARS` | 语义分块的最大块大小（默认 800 字符；目标块约为其 70%）。 |
| `RAG_SWEEP_SECONDS` | 知识文档 worker 轮询间隔（默认 5 秒）。 |

## 依赖与职责

以下为 `pyproject.toml` 中声明的直接依赖及其上游仓库（或官方源码项目）。其余由这些包带入的传递依赖以 `uv.lock` 为准。

| 依赖仓库 | 本项目中的作用 |
| --- | --- |
| [FastAPI](https://github.com/fastapi/fastapi) | 提供 HTTP 路由、依赖注入、文件上传、CORS、中间件和 OpenAPI 文档。 |
| [Uvicorn](https://github.com/Kludex/uvicorn) | 在 `main.py` 中运行 ASGI 应用并提供开发期热重载。 |
| [HTTPX](https://github.com/encode/httpx) | 异步访问 DuckDuckGo、天气来源和 Docling 文档解析服务。 |
| [Beautiful Soup](https://code.launchpad.net/beautifulsoup) | 解析搜索和天气页面的 HTML 内容。 |
| [LangChain](https://github.com/langchain-ai/langchain) 与 [langchain-openai](https://github.com/langchain-ai/langchain/tree/master/libs/partners/openai) | 通过 OpenAI 兼容接口调用模型，并把输出转换为 SSE 聊天事件。 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | 实现 `/v1/agents/run` 的基础状态图 Agent 节点。 |
| [Pydantic Settings](https://github.com/pydantic/pydantic-settings) | 从环境变量和 `.env` 加载、校验运行配置；Pydantic 模型由 FastAPI 的依赖链提供。 |
| [email-validator](https://github.com/JoshData/python-email-validator) | 校验注册与登录请求中的邮箱地址。 |
| [pwdlib](https://github.com/frankie567/pwdlib) 与 [PyJWT](https://github.com/jpadilla/pyjwt) | 分别处理密码哈希和 JWT 的签发、验证。 |
| [python-multipart](https://github.com/Kludex/python-multipart) | 支持文档读取和文件上传接口的 `multipart/form-data` 请求。 |
| [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) | 定义用户、会话、消息、文件等模型，并提供异步数据库会话。 |
| [aiosqlite](https://github.com/omnilib/aiosqlite) 与 [asyncpg](https://github.com/MagicStack/asyncpg) | 分别作为默认 SQLite 和可选 PostgreSQL 的 SQLAlchemy 异步驱动。 |
| [sqlite-vec](https://github.com/asg017/sqlite-vec) | vec0 虚拟表提供 ANN 向量检索；扩展不可用时自动退化为 BLOB 余弦扫描。 |
| [jieba](https://github.com/fxsjy/jieba) | 中文分词后写入 FTS5，为 BM25 稀疏检索提供关键词索引（FTS5 本身不支持中文分词）。 |

## API 概览

除认证和健康检查外，领域接口都需要 `Authorization: Bearer <token>`。先创建账号：

```powershell
curl.exe -X POST http://127.0.0.1:8080/v1/auth/register `
  -H "Content-Type: application/json" `
  -d '{"email":"me@example.com","password":"password123","display_name":"Me"}'
```

| 分组           | 接口                                                                                                                                   |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------|
| 认证           | `POST /v1/auth/register`、`/auth/login`、`/auth/device`；`GET /v1/users/me`                                                            |
| 工具与即时对话 | `POST /v1/tools/search`、`/tools/weather`、`/tools/document-read`、`/chat/stream`                                                      |
| 工作区和会话   | `GET`/`POST /v1/workspaces`、`PATCH /v1/workspaces/{id}`；`GET`/`POST /v1/conversations`；`GET`/`POST /v1/conversations/{id}/messages` |
| 持久化对话     | `POST /v1/conversations/{id}/generate`：服务端组合提示词、记忆和已保存消息后，以 SSE 返回回复并保存助手消息。                          |
| Agent          | `POST /v1/agents/run`：运行当前基础 LangGraph 模型节点。                                                                               |
| 记忆与提示词   | `GET`/`POST /v1/memories`、`POST /v1/memories/{id}/archive`、`GET`/`POST /v1/prompt-templates`                                         |
| 文件与知识库   | `POST /v1/files` 上传文件；`POST /v1/knowledge/documents/{file_id}` 入队处理；`GET /v1/knowledge/documents/{file_id}` 查询状态；`POST /v1/knowledge/documents/{file_id}/retry` 重新入队失败的文档。 |
| 设置           | `GET`/`PUT /v1/settings/{key}`                                                                                                         |

`POST /v1/chat/stream` 与会话生成接口均以 `text/event-stream` 返回 `message`、可选 `tool_call` 和最终 `done` 事件。`done` 携带 `{"sources": [...]}`，即本次回答引用的 RAG 来源（`{title, kind, id}`，`kind` 为 `memory` 或 `knowledge_document`）。详尽的请求体、响应模型和可交互调试入口请使用 `/docs`。

## 开发说明

核心 HTTP 路由位于 `app/api/`，业务和外部集成位于 `app/services/`，配置在 `app/core/config.py`，数据库模型与初始化逻辑在 `app/db.py`。运行生成的 `.env`、`data/`、`fat_ai.db` 不应提交。

当前仓库尚未配置自动化测试或代码格式化命令。新增功能时请至少通过 `/docs` 或相应的 HTTP 请求验证，并为外部依赖（OpenAI、Docling、搜索）使用可控的测试替身。

## 客户端迁移状态

Compose 客户端已通过 SSE 使用服务端模型网关，并通过设备会话把工作区、会话、消息、记忆和提示词模板的新增写入镜像到本服务。服务端已实现异步 RAG 索引（记忆与知识文档）与聊天引用来源回传；客户端 SQLDelight 目前仍负责读取与离线缓存；服务端回拉、双向冲突处理、文件对象存储及完整的 LangGraph 工作流仍是后续工作。
