from __future__ import annotations

import hashlib
import json

from openai import AsyncOpenAI
from redis.asyncio import Redis


class EmbeddingService:
    """Embedding 适配器：批量调用 + Redis 缓存。

    生产注意：
    - 文档侧和查询侧必须使用同一个 embedding 模型和归一化方式；
    - 批量大小受模型服务限制，默认 32；
    - 缓存 key 带模型名，换模型后不会命中旧向量。
    """

    def __init__(self, base_url: str, api_key: str, model: str, dim: int, redis_url: str) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.dim = dim
        self.redis = Redis.from_url(redis_url, decode_responses=True)

    async def close(self) -> None:
        await self.client.close()
        await self.redis.aclose()

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            cached = await self.redis.get(self._cache_key(text))
            if cached:
                results[index] = json.loads(cached)
            else:
                missing.append((index, text))
        for start in range(0, len(missing), 32):
            batch = missing[start : start + 32]
            response = await self.client.embeddings.create(
                model=self.model,
                input=[text[:6000] for _, text in batch],
            )
            for (index, text), item in zip(batch, response.data):
                vector = item.embedding
                results[index] = vector
                await self.redis.set(self._cache_key(text), json.dumps(vector), ex=86400)
        if any(vector is None for vector in results):
            raise RuntimeError("embedding failed for some inputs")
        return [vector for vector in results if vector is not None]

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha1(f"{self.model}:{text}".encode("utf-8")).hexdigest()
        return f"embed:{digest}"
