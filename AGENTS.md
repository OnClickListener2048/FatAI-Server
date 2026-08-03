# Repository Guidelines

## Project Structure & Module Organization

The application entry point is `main.py`, which starts `app.main:app` with Uvicorn. Keep HTTP concerns in `app/api/`: `routes.py` contains the compatibility/tool endpoints and `domain_routes.py` contains application-domain endpoints. Put request and response Pydantic models in `app/models.py`; add business or integration logic under `app/services/`; and keep settings in `app/core/config.py`. Database setup belongs in `app/db.py`, while authentication helpers belong in `app/security.py`.

Runtime-only files are intentionally excluded from Git: `.env`, `data/`, `fat_ai.db`, virtual environments, and bytecode. Use `.env.example` as the starting point for local configuration.

## Build, Test, and Development Commands

Requires Python 3.12+. `uv.lock` records the resolved dependency set.

```powershell
python -m pip install -e .  # install the project and its dependencies
python main.py              # run the reload-enabled server on port 8080
```

Open `http://127.0.0.1:8080/docs` to exercise the generated OpenAPI interface, or call `GET /health` for a quick service check. Chat streaming requires `OPENAI_API_KEY`; document reading also expects a reachable `DOCLING_SERVER_URL`.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, standard-library imports first, then third-party and local imports. Use `snake_case` for modules, functions, fields, and variables; `PascalCase` for classes and Pydantic models; and descriptive route/service names such as `chat_stream` and `DuckDuckGoSearchService`. Type annotate public functions and async boundaries. Keep route handlers thin and return typed response models; place remote calls and reusable logic in services.

No formatter, linter, or static type checker is configured yet. Do not add unrelated tool configuration in a focused change; match nearby code and run an import/startup check before submitting.

## Testing Guidelines

There are currently no committed test files or configured test runner. Add new tests under `tests/` using filenames such as `test_search.py` and functions named `test_<behavior>`. Prefer FastAPI `TestClient` or async HTTP tests, and mock external services (OpenAI, Docling, and search) so tests are deterministic. When introducing a test framework, add its dependency and document the exact command in the same change.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional Commit-style subjects, for example `feat: add device session sync support`. Use `feat:`, `fix:`, `docs:`, `refactor:`, or `test:` followed by a brief imperative summary. Keep each commit scoped to one change.

Pull requests should state the behavior changed, list configuration or migration impacts, link the relevant issue when available, and include request/response examples or screenshots for API-visible changes. Call out any security-sensitive settings, especially local-path document access and JWT configuration.

## API & Database Documentation

Every change to an HTTP endpoint, request/response model, authentication or sync protocol, database model, migration, or persistence behavior must update `docs/API.md` in the same commit. New server interfaces must follow the existing authenticated `/v1` route structure, typed Pydantic models, ownership checks, and documented sync/idempotency conventions.
