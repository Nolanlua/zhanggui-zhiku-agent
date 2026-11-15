from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.schemas import UserContext
from app.db.models import Document
from app.rag.bm25 import BM25Index
from app.rag.chunker import StructureAwareChunker
from app.rag.embedder import EmbeddingService
from app.rag.parser import DocumentParser
from app.rag.vector_store import MilvusVectorStore

logger = logging.getLogger(__name__)


class KnowledgeService:
    """文档入库编排：解析 -> 分块 -> Embedding -> 向量库/ES -> 更新状态。

    这里用 asyncio.create_task 做异步索引；生产建议换成任务队列（Celery/Arq），
    避免 API 进程重启导致任务丢失。
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        parser: DocumentParser,
        chunker: StructureAwareChunker,
        embedder: EmbeddingService,
        vector_store: MilvusVectorStore,
        bm25: BM25Index,
    ) -> None:
        self.session_factory = session_factory
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self._tasks: set[asyncio.Task] = set()

    async def submit(
        self,
        file_name: str,
        content: bytes,
        user: UserContext,
        title: str | None = None,
        visibility: str = "public",
        department_id: str | None = None,
    ) -> str:
        doc_id = f"doc_{uuid.uuid4().hex[:16]}"
        suffix = Path(file_name).suffix.lower() or ".txt"
        upload_dir = Path(tempfile.gettempdir()) / "zhiku-uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / f"{doc_id}{suffix}"
        path.write_bytes(content)

        async with self.session_factory() as session:
            document = Document(
                id=doc_id,
                tenant_id=user.tenant_id,
                title=title or Path(file_name).stem,
                source_uri=f"upload://{file_name}",
                status="pending",
                visibility=visibility,
                department_id=department_id,
                owner_user_id=user.user_id,
            )
            session.add(document)
            await session.commit()

        task = asyncio.create_task(self._index(doc_id, path, user, visibility, department_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return doc_id

    async def _index(
        self,
        doc_id: str,
        path: Path,
        user: UserContext,
        visibility: str,
        department_id: str | None,
    ) -> None:
        try:
            await self._set_status(doc_id, status="indexing")
            parsed = self.parser.parse(path, doc_id=doc_id)
            chunks = self.chunker.chunk_document(parsed, tenant_id=user.tenant_id)
            for chunk in chunks:
                chunk.visibility = visibility
                chunk.department_id = department_id
                chunk.owner_user_id = user.user_id
            vectors = await self.embedder.embed_texts([chunk.text for chunk in chunks])
            await self.vector_store.upsert(chunks, vectors)
            await self.bm25.index_chunks(chunks)
            await self._set_status(doc_id, status="indexed", chunk_count=len(chunks))
            logger.info("document indexed", extra={"doc_id": doc_id, "chunks": len(chunks)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("document indexing failed", extra={"doc_id": doc_id})
            await self._set_status(doc_id, status="failed", error=str(exc)[:2000])
        finally:
            path.unlink(missing_ok=True)

    async def _set_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> None:
        async with self.session_factory() as session:
            document = await session.get(Document, doc_id)
            if document is None:
                return
            document.status = status
            if chunk_count is not None:
                document.chunk_count = chunk_count
            if error is not None:
                document.error = error
            await session.commit()

    async def get_document(self, doc_id: str, tenant_id: str) -> Document | None:
        async with self.session_factory() as session:
            result = await session.execute(
                select(Document).where(Document.id == doc_id, Document.tenant_id == tenant_id)
            )
            return result.scalar_one_or_none()

    async def delete_document(self, doc: Document) -> None:
        await self.vector_store.delete_by_doc(doc.id, doc.tenant_id)
        await self.bm25.delete_by_doc(doc.id, doc.tenant_id)
        async with self.session_factory() as session:
            stored = await session.get(Document, doc.id)
            if stored is not None:
                await session.delete(stored)
                await session.commit()
