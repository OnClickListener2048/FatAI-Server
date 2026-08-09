"""OpenAI 兼容 embedding 客户端(bge-m3 本地 Ollama / 托管 BYOK)。

与 scripts/explore_embedding.py 学到的协议保持一致:POST {base_url}/embeddings,
响应 data 数组按 index 排序(API 不保证顺序)。向量归一化后余弦相似度退化为点积,
这也是向量库检索加速的前提。
"""

import httpx

from app.services.errors import ServiceError


class EmbeddingService:
    def __init__(self, base_url: str, api_key: str, model: str, dimensions: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入并归一化。返回与输入一一对应的向量列表。"""
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self._model, "input": texts},
                )
                response.raise_for_status()
                data = response.json().get("data", [])
        except httpx.HTTPError as error:
            raise ServiceError(
                "EMBEDDING_UNAVAILABLE",
                f"Embedding service is unavailable ({self._model}).",
                503,
            ) from error

        vectors = [item["embedding"] for item in sorted(data, key=lambda item: item.get("index", 0))]
        if len(vectors) != len(texts):
            raise ServiceError("EMBEDDING_FAILED", "Embedding response count mismatch.", 502)
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise ServiceError(
                    "EMBEDDING_MISMATCH",
                    f"Embedding model {self._model} returned {len(vector)} dimensions, "
                    f"expected {self._dimensions}.",
                    502,
                )
        return [normalize(vector) for vector in vectors]

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]


def normalize(vector: list[float]) -> list[float]:
    """单位化向量,使余弦相似度 == 点积(向量库检索的前提)。"""
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]
