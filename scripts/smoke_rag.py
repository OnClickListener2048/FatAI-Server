"""RAG 集成端到端冒烟: auth → 记忆建索引 → 文档入队 → 检索验证。

用法: uv run python scripts/smoke_rag.py
需要: server 运行在 8080, Ollama(bge-m3)运行在 11434。Docling 未启动时
知识文档会预期性失败(FAILED),记忆链路不受影响。
"""

import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8080"
DEVICE_ID = "rag-smoke-device"


async def main() -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. 设备认证
        auth = await client.post(
            f"{BASE}/v1/auth/device",
            json={"device_id": DEVICE_ID, "display_name": "RAG smoke"},
        )
        auth.raise_for_status()
        token = auth.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("1. auth OK")

        # 2. 建记忆(触发索引钩子); 随机 id 保证脚本可重复运行
        import uuid

        memory_id = f"rag-smoke-{uuid.uuid4().hex[:8]}"
        memory = await client.post(
            f"{BASE}/v1/memories",
            headers=headers,
            json={
                "id": memory_id,
                "scope": "GLOBAL",
                "kind": "FACT",
                "content": "用户每天早上九点开始工作,中午需要午休半小时,晚上十一点前入睡。",
            },
        )
        memory.raise_for_status()
        print("2. memory created OK")

        # 3. 上传知识文档并入队
        sample = "scripts/sample_knowledge.md"
        with open(sample, "rb") as fh:
            upload = await client.post(
                f"{BASE}/v1/files",
                headers=headers,
                files={"file": ("sample_knowledge.md", fh, "text/markdown")},
            )
        upload.raise_for_status()
        file_id = upload.json()["id"]
        enqueue = await client.post(f"{BASE}/v1/knowledge/documents/{file_id}", headers=headers)
        enqueue.raise_for_status()
        print(f"3. knowledge doc enqueued file_id={file_id}")

        # 4. 轮询处理状态(worker 5s sweep; docling 未启动则预期 FAILED)
        status = "QUEUED"
        for _ in range(10):
            await asyncio.sleep(3)
            doc = await client.get(f"{BASE}/v1/knowledge/documents/{file_id}", headers=headers)
            status = doc.json()["status"]
            if status in ("READY", "FAILED"):
                break
        print(f"4. knowledge doc status = {status}")

        # 5. 检索验证: 真实 embedding + 真实向量库
        from app.core.config import get_settings
        from app.services.rag.embedding import EmbeddingService
        from app.services.rag.retrieval import RetrievalService
        from app.services.rag.vectorstore import VectorStore

        settings = get_settings()
        store = VectorStore(settings.database_url, settings.embedding_dimensions)
        store.initialize()
        embedder = EmbeddingService(
            settings.embedding_base_url,
            settings.embedding_api_key,
            settings.embedding_model,
            settings.embedding_dimensions,
        )
        retriever = RetrievalService(embedder, store, min_score=settings.rag_min_score)

        print(f"5. vector store: enabled={store.enabled} vec0={store.vec_available}")
        print(f"   chunks indexed (memory) = {await store.source_chunk_count('memory', memory_id)}")

        # 5a. 命中查询(应命中记忆); user_id 即设备认证生成的 user id
        hits = await retriever.search("我通常几点开始工作?", DEVICE_ID, None, None)
        print(f"5a. hit query  -> memories={[h.source_id for h in hits.memories]} docs={[h.source_id for h in hits.documents]}")
        for hit in hits.memories[:2]:
            print(f"     score={hit.score:.3f} title={hit.title!r} content={hit.content[:40]!r}")

        # 5b. 负向查询(应低于阈值,不出结果)
        misses = await retriever.search("晚饭吃什么比较好?", DEVICE_ID, None, None)
        print(f"5b. noise query -> memories={[h.source_id for h in misses.memories]} docs={[h.source_id for h in misses.documents]}")

        print("\nSMOKE DONE")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:  # noqa: BLE001
        print(f"SMOKE FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
