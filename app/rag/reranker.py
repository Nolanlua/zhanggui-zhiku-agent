from __future__ import annotations

import logging

import httpx

from app.rag.types import RetrievalHit

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder 精排适配器。

    Rerank 只处理召回后的 TopN，不参与全量检索；失败时降级为原始排序，
    保证主链路可用。
    """

    def __init__(self, url: str = "", timeout_seconds: float = 3.0) -> None:
        self.url = url
        self.timeout = timeout_seconds

    async def rerank(self, query: str, hits: list[RetrievalHit], top_n: int = 20) -> list[RetrievalHit]:
        if not hits or not self.url:
            return hits[:top_n]
        candidates = hits[:top_n]
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.url,
                    json={"query": query, "documents": [hit.text for hit in candidates], "top_n": top_n},
                )
                response.raise_for_status()
                payload = response.json()
            ordered = []
            for item in payload.get("results", []):
                index = int(item["index"])
                hit = candidates[index]
                hit.score = float(item.get("relevance_score", hit.score))
                hit.source = "rerank"
                ordered.append(hit)
            return ordered[:top_n] if ordered else candidates
        except Exception as exc:  # noqa: BLE001
            logger.warning("rerank failed, fallback to fusion order: %s", exc)
            return candidates
