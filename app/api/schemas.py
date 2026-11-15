from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class UserContext(BaseModel):
    """请求用户上下文；生产环境应从 JWT/网关解析，而不是直接信任 Header。"""

    user_id: str
    tenant_id: str
    roles: list[str] = Field(default_factory=list)
    department_ids: list[str] = Field(default_factory=list)
    store_ids: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    index: int
    doc_id: str
    chunk_id: str
    title: str
    section: str | None = None
    page: int | None = None
    snippet: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str = Field(default="default", max_length=128)
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_tools: bool = True


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    rewritten_queries: list[str]
    latency_ms: int
    trace_id: str


class DocumentUploadResponse(BaseModel):
    doc_id: str
    status: Literal["pending", "indexed", "failed"]
    message: str


class DocumentStatusResponse(BaseModel):
    doc_id: str
    title: str
    status: str
    chunk_count: int
    created_at: datetime
    error: str | None = None


class EvalRequest(BaseModel):
    dataset_path: str = "app/evaluation/golden_set.jsonl"
    top_k: int = Field(default=5, ge=1, le=20)


class EvalResponse(BaseModel):
    total: int
    recall_at_k: float
    top3_recall: float
    mrr: float
    citation_accuracy: float
    details: list[dict[str, Any]] = Field(default_factory=list)
