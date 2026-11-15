from __future__ import annotations

from typing import Any, TypedDict

from app.api.schemas import Citation, UserContext
from app.rag.types import RetrievalHit


class AgentState(TypedDict, total=False):
    question: str
    user: UserContext
    history: list[dict[str, str]]
    conversation_id: str
    trace_id: str
    use_tools: bool
    top_k: int
    rewritten_queries: list[str]
    hits: list[RetrievalHit]
    context: str
    citations: list[Citation]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    answer: str
    step: int
    validation: dict[str, Any]
    error: str | None
