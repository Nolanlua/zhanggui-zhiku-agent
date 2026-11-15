from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from app.agent.prompts import ANSWER_SYSTEM, build_answer_prompt
from app.agent.state import AgentState
from app.agent.tools import ToolPlanner, ToolRegistry
from app.api.schemas import Citation, UserContext
from app.rag.citation import CitationBuilder
from app.rag.hybrid import HybridRetriever
from app.rag.query_rewrite import QueryRewriter
from app.rag.types import RetrievalHit

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentDependencies:
    rewriter: QueryRewriter
    retriever: HybridRetriever
    tool_planner: ToolPlanner
    tool_registry: ToolRegistry
    llm: AsyncOpenAI
    citation_builder: CitationBuilder
    chat_model: str
    top_k: int = 8


class KnowledgeNodes:
    def __init__(self, deps: AgentDependencies) -> None:
        self.deps = deps

    async def rewrite(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        try:
            queries = await self.deps.rewriter.rewrite(question, state.get("history"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("rewrite node failed: %s", exc)
            queries = [question]
        return {"rewritten_queries": queries}

    async def retrieve(self, state: AgentState) -> dict[str, Any]:
        queries = state.get("rewritten_queries") or [state["question"]]
        user: UserContext = state["user"]
        top_k = state.get("top_k") or self.deps.top_k
        try:
            hits: list[RetrievalHit] = await self.deps.retriever.retrieve_many(queries[:3], user, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            logger.exception("retrieve node failed")
            return {"hits": [], "context": "", "citations": [], "error": str(exc)}
        context, citations = self.deps.citation_builder.build(hits, max_chars=6000)
        return {"hits": hits, "context": context, "citations": citations}

    async def tools(self, state: AgentState) -> dict[str, Any]:
        if not state.get("use_tools", True):
            return {"tool_calls": [], "tool_results": []}
        try:
            calls, results = await self.deps.tool_planner.plan(
                state["question"], state["user"], self.deps.tool_registry
            )
            return {"tool_calls": calls, "tool_results": results}
        except Exception as exc:  # noqa: BLE001
            logger.warning("tool node failed: %s", exc)
            return {"tool_calls": [], "tool_results": []}

    async def generate(self, state: AgentState) -> dict[str, Any]:
        context = state.get("context", "")
        tool_results = state.get("tool_results", [])
        if not context and not tool_results:
            return {
                "answer": "知识库中没有找到足够可靠的依据，建议补充相关文档，或把问题转给人工确认。",
                "citations": [],
            }
        prompt = build_answer_prompt(
            question=state["question"],
            context=context,
            tool_results=tool_results,
            history=state.get("history", []),
            retry_hint=state.get("step", 0) > 0,
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": ANSWER_SYSTEM}]
        for item in state.get("history", [])[-6:]:
            if item.get("role") in ("user", "assistant") and item.get("content"):
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": prompt})
        try:
            response = await self.deps.llm.chat.completions.create(
                model=self.deps.chat_model,
                temperature=0.1,
                max_tokens=900,
                messages=messages,
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("generate node failed")
            answer = self.deps.citation_builder.fallback_answer(state.get("citations", []))
        return {"answer": answer}

    async def verify(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("answer", "")
        citations: list[Citation] = state.get("citations", [])
        step = state.get("step", 0)
        if not state.get("context") and not state.get("tool_results"):
            return {
                "validation": {"valid": True, "reason": "no_evidence", "citation_count": 0},
                "step": step + 1,
            }
        report = self.deps.citation_builder.validate(answer, citations)
        if report.is_valid:
            return {
                "validation": {
                    "valid": True,
                    "reason": "ok",
                    "citation_count": report.citation_count,
                    "invalid_indexes": report.invalid_indexes,
                },
                "step": step + 1,
            }
        if step >= 1:
            return {
                "answer": self.deps.citation_builder.fallback_answer(citations),
                "validation": {"valid": True, "reason": "fallback", "citation_count": report.citation_count},
                "step": step + 1,
            }
        return {
            "validation": {
                "valid": False,
                "reason": "missing_or_invalid_citation",
                "citation_count": report.citation_count,
                "invalid_indexes": report.invalid_indexes,
            },
            "step": step + 1,
        }


def route_after_verify(state: AgentState) -> str:
    validation = state.get("validation", {})
    return "end" if validation.get("valid") else "retry"
