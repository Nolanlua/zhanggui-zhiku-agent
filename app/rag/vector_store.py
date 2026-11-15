from __future__ import annotations

import json
import logging
from typing import Any

from app.rag.types import Chunk, RetrievalHit

logger = logging.getLogger(__name__)


def escape_filter_value(value: str) -> str:
    """用 JSON 编码转义引号和反斜杠，避免把用户输入直接拼进 Milvus 表达式。"""
    return json.dumps(value, ensure_ascii=False)[1:-1]


class MilvusVectorStore:
    """Milvus 向量库适配器，检索前按 tenant + ACL 做 metadata filter。"""

    def __init__(self, uri: str, collection: str, dim: int) -> None:
        from pymilvus import MilvusClient

        self.client = MilvusClient(uri=uri)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from pymilvus import DataType, MilvusClient

        if self.client.has_collection(self.collection):
            return
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("section", DataType.VARCHAR, max_length=512)
        schema.add_field("page", DataType.INT64)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=128)
        schema.add_field("visibility", DataType.VARCHAR, max_length=32)
        schema.add_field("department_id", DataType.VARCHAR, max_length=128)
        schema.add_field("owner_user_id", DataType.VARCHAR, max_length=128)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(collection_name=self.collection, schema=schema, index_params=index_params)

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        rows: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors):
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title[:500],
                    "section": (chunk.section or "")[:500],
                    "page": chunk.page or 0,
                    "tenant_id": chunk.tenant_id,
                    "visibility": chunk.visibility,
                    "department_id": chunk.department_id or "",
                    "owner_user_id": chunk.owner_user_id or "",
                    "text": chunk.text[:65000],
                    "vector": vector,
                }
            )
        self.client.upsert(collection_name=self.collection, data=rows)

    async def search(self, vector: list[float], filter_expr: str, top_k: int) -> list[RetrievalHit]:
        results = self.client.search(
            collection_name=self.collection,
            data=[vector],
            filter=filter_expr,
            limit=top_k,
            output_fields=["chunk_id", "doc_id", "title", "section", "page", "text"],
        )
        hits: list[RetrievalHit] = []
        for item in results[0]:
            entity = item.get("entity", item)
            hits.append(
                RetrievalHit(
                    chunk_id=str(entity.get("chunk_id", item.get("id", ""))),
                    doc_id=str(entity.get("doc_id", "")),
                    text=str(entity.get("text", "")),
                    score=float(item.get("distance", 0.0)),
                    source="vector",
                    title=str(entity.get("title", "")),
                    section=str(entity.get("section", "")) or None,
                    page=int(entity.get("page", 0)) or None,
                )
            )
        return hits

    async def delete_by_doc(self, doc_id: str, tenant_id: str) -> None:
        expr = f'doc_id == "{escape_filter_value(doc_id)}" and tenant_id == "{escape_filter_value(tenant_id)}"'
        self.client.delete(collection_name=self.collection, filter=expr)
