from __future__ import annotations

import re
from dataclasses import dataclass

from app.api.schemas import Citation
from app.rag.types import RetrievalHit

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(slots=True)
class CitationReport:
    valid_indexes: list[int]
    invalid_indexes: list[int]
    citation_count: int

    @property
    def is_valid(self) -> bool:
        return self.citation_count > 0 and not self.invalid_indexes


class CitationBuilder:
    """组装上下文并校验回答中的引用。

    这里不是“让模型自己保证引用正确”，而是在后处理阶段检查：
    - 引用编号是否来自本次上下文；
    - 没有引用的回答是否需要降级/拒答。
    """

    def build(self, hits: list[RetrievalHit], max_chars: int = 6000) -> tuple[str, list[Citation]]:
        lines: list[str] = []
        citations: list[Citation] = []
        used = 0
        for hit in hits:
            header = f"[{len(citations) + 1}] {hit.title}"
            if hit.section:
                header += f" / {hit.section}"
            if hit.page:
                header += f" / 第 {hit.page} 页"
            snippet = hit.text[:900].replace("\n", " ")
            block = f"{header}\n{snippet}"
            if used + len(block) > max_chars and citations:
                break
            lines.append(block)
            citations.append(
                Citation(
                    index=len(citations) + 1,
                    doc_id=hit.doc_id,
                    chunk_id=hit.chunk_id,
                    title=hit.title,
                    section=hit.section,
                    page=hit.page,
                    snippet=snippet,
                )
            )
            used += len(block)
        return "\n\n".join(lines), citations

    def validate(self, answer: str, citations: list[Citation]) -> CitationReport:
        indexes = [int(item) for item in CITATION_PATTERN.findall(answer)]
        valid = sorted({index for index in indexes if 1 <= index <= len(citations)})
        invalid = sorted({index for index in indexes if index < 1 or index > len(citations)})
        return CitationReport(valid_indexes=valid, invalid_indexes=invalid, citation_count=len(indexes))

    @staticmethod
    def fallback_answer(citations: list[Citation]) -> str:
        if not citations:
            return "我没有在知识库中找到足够可靠的依据，建议补充文档或转人工确认。"
        return "我暂时无法生成有依据的回答，请查看右侧引用资料，或补充更具体的问题。"
