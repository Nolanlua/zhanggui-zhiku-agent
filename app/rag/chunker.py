from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.rag.types import Chunk, ParsedBlock, ParsedDocument

SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])")


@dataclass(slots=True)
class ChunkerConfig:
    max_chars: int = 700
    min_chars: int = 120
    overlap_chars: int = 120


class StructureAwareChunker:
    """按标题层级优先、语义窗口兜底的分块器。

    设计取舍：
    - 标题/章节是业务文档最强的结构信号，优先按结构切；
    - 单个段落过长时再按句子切，并保留 overlap，避免答案被截断；
    - 表格整块保留，并在文本里带表头，避免出现只有数字没有列名的 chunk。
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk_document(self, doc: ParsedDocument, tenant_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for block in doc.blocks:
            for piece in self._chunk_block(block):
                chunks.append(
                    Chunk(
                        chunk_id=self._chunk_id(doc.doc_id, len(chunks), piece),
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=piece,
                        section=block.section,
                        page=block.page,
                        chunk_index=len(chunks),
                        tenant_id=tenant_id,
                        metadata={"source_type": doc.source_type, "kind": block.kind},
                    )
                )
        return chunks

    def _chunk_block(self, block: ParsedBlock) -> list[str]:
        text = block.text.strip()
        if not text:
            return []
        if block.kind == "table":
            return [text] if len(text) <= self.config.max_chars * 2 else self._split_long(text)
        if len(text) <= self.config.max_chars:
            return [text]
        return self._split_long(text)

    def _split_long(self, text: str) -> list[str]:
        sentences = [part.strip() for part in SENTENCE_SPLIT.split(text) if part.strip()]
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if not current:
                current = sentence
            elif len(current) + len(sentence) <= self.config.max_chars:
                current += sentence
            else:
                pieces.append(current)
                overlap = current[-self.config.overlap_chars :] if self.config.overlap_chars else ""
                current = overlap + sentence
                if len(current) > self.config.max_chars:
                    pieces.append(current[: self.config.max_chars])
                    current = current[self.config.max_chars - self.config.overlap_chars :]
        if current.strip():
            pieces.append(current.strip())
        return [piece for piece in pieces if len(piece) >= self.config.min_chars or len(pieces) == 1]

    @staticmethod
    def _chunk_id(doc_id: str, index: int, text: str) -> str:
        digest = hashlib.sha1(f"{doc_id}:{index}:{text[:120]}".encode("utf-8")).hexdigest()
        return f"chunk_{digest[:20]}"
