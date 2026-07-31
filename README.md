# FatAI Server

FatAI 的 FastAPI 后端，提供工具调用、流式 AI 对话、用户认证，以及工作区与聊天数据同步能力。默认使用 SQLite；数据库表会在启动时自动创建。

## 功能

- 搜索、天气查询和文档转 Markdown 工具接口；文档解析通过 Docling 服务完成。
- OpenAI 兼容模型的 Server-Sent Events（SSE）流式聊天，以及基于 LangGraph 的基础 Agent 运行入口。
- JWT 用户认证，支持邮箱注册/登录和桌面端迁移期间的设备账号初始化。
- 按用户隔离的工作区、会话、消息、记忆、提示词模板与应用设置同步。
- 文件上传及知识库文档入队记录；当前知识库接口只负责入队和状态保存，不执行异步解析或检索。

## 快速开始

需要 Python 3.12+。推荐使用 [uv](https://docs.astral.sh/uv/)（仓库已提交 `uv.lock`）：

```powershell
uv sync
Copy-Item .env.example .env
uv run python main.py
```

也可使用 pip 安装依赖后执行 `python main.py`。服务默认监听 `http://127.0.0.1:8080`，OpenAPI 文档位于 `http://127.0.0.1:8080/docs`，健康检查为 `GET /health`。

## 配置

从 `.env.example` 复制 `.env` 后按需设置：

| 变量 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 流式聊天和 Agent 所需的 API Key 与可选兼容服务地址。 |
| `DEFAULT_CHAT_MODEL` | 未指定模型时使用的模型，默认 `gpt-4o-mini`。 |
| `DOCLING_SERVER_URL` | 文档解析服务地址，默认 `http://127.0.0.1:5001`。 |
| `DATABASE_URL` | SQLAlchemy 异步连接串，默认 `sqlite+aiosqlite:///./fat_ai.db`；可使用 `postgresql+asyncpg://...`。 |
| `JWT_SECRET` / `JWT_EXPIRATION_MINUTES` | 访问令牌签名密钥与有效期；生产环境必须替换默认密钥。 |
| `CORS_ORIGINS` | 允许的来源 JSON 数组，例如 `["http://localhost:3000"]`。 |
| `UPLOAD_DIRECTORY` / `MAX_DOCUMENT_SIZE_BYTES` | 上传文件目录和大小上限（默认 50 MiB）。 |
| `ALLOW_LOCAL_DOCUMENT_PATHS` | 仅为本地桌面端迁移保留 JSON `localPath` 读取；部署到远程环境前设为 `false`。 |

## API 概览

除认证和健康检查外，领域接口都需要 `Authorization: Bearer <token>`。先创建账号：

```powershell
curl.exe -X POST http://127.0.0.1:8080/v1/auth/register `
  -H "Content-Type: application/json" `
  -d '{"email":"me@example.com","password":"password123","display_name":"Me"}'
```

| 分组 | 接口 |
| --- | --- |
| 认证 | `POST /v1/auth/register`、`/auth/login`、`/auth/device`；`GET /v1/users/me` |
| 工具与即时对话 | `POST /v1/tools/search`、`/tools/weather`、`/tools/document-read`、`/chat/stream` |
| 工作区和会话 | `GET`/`POST /v1/workspaces`、`PATCH /v1/workspaces/{id}`；`GET`/`POST /v1/conversations`；`GET`/`POST /v1/conversations/{id}/messages` |
| 持久化对话 | `POST /v1/conversations/{id}/generate`：服务端组合提示词、记忆和已保存消息后，以 SSE 返回回复并保存助手消息。 |
| Agent | `POST /v1/agents/run`：运行当前基础 LangGraph 模型节点。 |
| 记忆与提示词 | `GET`/`POST /v1/memories`、`POST /v1/memories/{id}/archive`、`GET`/`POST /v1/prompt-templates` |
| 文件与知识库 | `POST /v1/files` 上传文件；`POST /v1/knowledge/documents/{file_id}` 创建知识库处理队列记录。 |
| 设置 | `GET`/`PUT /v1/settings/{key}` |

`POST /v1/chat/stream` 与会话生成接口均以 `text/event-stream` 返回 `message`、可选 `tool_call` 和最终 `done` 事件。详尽的请求体、响应模型和可交互调试入口请使用 `/docs`。

## 开发说明

核心 HTTP 路由位于 `app/api/`，业务和外部集成位于 `app/services/`，配置在 `app/core/config.py`，数据库模型与初始化逻辑在 `app/db.py`。运行生成的 `.env`、`data/`、`fat_ai.db` 不应提交。

当前仓库尚未配置自动化测试或代码格式化命令。新增功能时请至少通过 `/docs` 或相应的 HTTP 请求验证，并为外部依赖（OpenAI、Docling、搜索）使用可控的测试替身。
