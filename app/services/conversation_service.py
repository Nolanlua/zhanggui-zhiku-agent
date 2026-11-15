from __future__ import annotations

import json

from redis.asyncio import Redis


class ConversationService:
    """用 Redis 保存最近几轮对话，供 Query 改写和生成使用。

    生产可替换为 PostgreSQL + 摘要压缩；这里只保留最近 10 轮，避免上下文无限增长。
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 7 * 86400) -> None:
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_seconds

    def _key(self, tenant_id: str, conversation_id: str) -> str:
        return f"chat:{tenant_id}:{conversation_id}"

    async def get_history(self, tenant_id: str, conversation_id: str) -> list[dict[str, str]]:
        values = await self.redis.lrange(self._key(tenant_id, conversation_id), -20, -1)
        return [json.loads(item) for item in values]

    async def append(self, tenant_id: str, conversation_id: str, role: str, content: str) -> None:
        key = self._key(tenant_id, conversation_id)
        await self.redis.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
        await self.redis.ltrim(key, -20, -1)
        await self.redis.expire(key, self.ttl)

    async def close(self) -> None:
        await self.redis.aclose()
