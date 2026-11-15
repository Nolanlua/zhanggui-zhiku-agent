from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class ParsedBlock:
    text: str
    page: int | None = None
    section: str | None = None
    kind: Literal["text", "table", "image_text"] = "text"


@dataclass(slots=True)
class ParsedDocument:
    doc_id: str
    title: str
    source_type: str
    blocks: list[ParsedBlock]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    section: str | None
    page: int | None
    chunk_index: int
    tenant_id: str
    visibility: str = "public"
    department_id: str | None = None
    owner_user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str
    title: str = ""
    section: str | None = None
    page: int | None = None
