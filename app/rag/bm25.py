from __future__ import annotations

from typing import Any

from app.rag.types import Chunk, RetrievalHit


class BM25Index:
    """Elasticsearch BM25 检索。中文生产环境建议把 analyzer 换成 IK/smartcn。"""

    def __init__(self, url: str, index: str) -> None:
        from elasticsearch import AsyncElasticsearch

        self.client = AsyncElasticsearch(hosts=[url])
        self.index = index

    async def ensure_index(self) -> None:
        if await self.client.indices.exists(index=self.index):
            return
        mappings = {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "section": {"type": "text"},
                "text": {"type": "text"},
                "tenant_id": {"type": "keyword"},
                "visibility": {"type": "keyword"},
                "department_id": {"type": "keyword"},
                "owner_user_id": {"type": "keyword"},
                "page": {"type": "integer"},
            }
        }
        await self.client.indices.create(index=self.index, mappings=mappings)

    async def index_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            await self.client.index(
                index=self.index,
                id=chunk.chunk_id,
                document={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "text": chunk.text,
                    "tenant_id": chunk.tenant_id,
                    "visibility": chunk.visibility,
                    "department_id": chunk.department_id or "",
                    "owner_user_id": chunk.owner_user_id or "",
                    "page": chunk.page,
                },
                refresh=False,
            )

    async def search(self, query: str, filters: list[dict[str, Any]], top_k: int) -> list[RetrievalHit]:
        bool_query: dict[str, Any] = {
            "must": [{"multi_match": {"query": query, "fields": ["text", "title^2", "section^1.5"]}}]
        }
        if filters:
            bool_query["filter"] = filters
        response = await self.client.search(
            index=self.index,
            query={"bool": bool_query},
            size=top_k,
            _source=["chunk_id", "doc_id", "title", "section", "page", "text"],
        )
        hits: list[RetrievalHit] = []
        for item in response["hits"]["hits"]:
            source = item["_source"]
            hits.append(
                RetrievalHit(
                    chunk_id=source["chunk_id"],
                    doc_id=source["doc_id"],
                    text=source["text"],
                    score=float(item["_score"] or 0.0),
                    source="bm25",
                    title=source.get("title", ""),
                    section=source.get("section") or None,
                    page=source.get("page") or None,
                )
            )
        return hits

    async def delete_by_doc(self, doc_id: str, tenant_id: str) -> None:
        await self.client.delete_by_query(
            index=self.index,
            query={"bool": {"filter": [{"term": {"doc_id": doc_id}}, {"term": {"tenant_id": tenant_id}}]}},
            refresh=True,
        )

    async def close(self) -> None:
        await self.client.close()
