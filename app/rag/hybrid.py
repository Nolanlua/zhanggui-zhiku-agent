from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.rag.bm25 import BM25Index
from app.rag.embedder import EmbeddingService
from app.rag.reranker import Reranker
from app.rag.types import RetrievalHit
from app.rag.vector_store import MilvusVectorStore

if TYPE_CHECKING:
    from app.api.schemas import UserContext
    from app.services.permission_service import PermissionService


def rrf_fuse(result_lists: list[list[RetrievalHit]], k: int = 60, top_k: int = 20) -> list[RetrievalHit]:
    """Reciprocal Rank Fusion：只用排名，不直接加不同来源的分数。"""
    scores: dict[str, float] = {}
    best: dict[str, RetrievalHit] = {}
    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            if hit.chunk_id not in best or hit.score > best[hit.chunk_id].score:
                best[hit.chunk_id] = hit
    ordered = sorted(best.values(), key=lambda item: scores[item.chunk_id], reverse=True)
    for hit in ordered:
        hit.score = scores[hit.chunk_id]
        hit.source = "rrf"
    return ordered[:top_k]


class HybridRetriever:
    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: MilvusVectorStore,
        bm25: BM25Index,
        reranker: Reranker,
        permission: "PermissionService",
        rrf_k: int = 60,
        vector_top_k: int = 40,
        bm25_top_k: int = 40,
        rerank_top_n: int = 20,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.permission = permission
        self.rrf_k = rrf_k
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rerank_top_n = rerank_top_n

    async def retrieve(self, query: str, user: "UserContext", top_k: int = 8) -> list[RetrievalHit]:
        vector = await self.embedder.embed_query(query)
        milvus_filter = self.permission.milvus_filter(user)
        es_filters = self.permission.es_filters(user)
        vector_hits, bm25_hits = await asyncio.gather(
            self.vector_store.search(vector, milvus_filter, self.vector_top_k),
            self.bm25.search(query, es_filters, self.bm25_top_k),
        )
        fused = rrf_fuse([vector_hits, bm25_hits], k=self.rrf_k, top_k=self.rerank_top_n)
        return await self.reranker.rerank(query, fused, top_n=top_k)

    async def retrieve_many(self, queries: list[str], user: "UserContext", top_k: int = 8) -> list[RetrievalHit]:
        if not queries:
            return []
        all_hits = await asyncio.gather(*(self.retrieve(query, user, top_k=self.rerank_top_n) for query in queries))
        fused = rrf_fuse(list(all_hits), k=self.rrf_k, top_k=top_k)
        return await self.reranker.rerank(queries[0], fused, top_n=top_k)
