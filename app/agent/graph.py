from __future__ import annotations

import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agent.nodes import AgentDependencies, KnowledgeNodes, route_after_verify
from app.agent.state import AgentState
from app.api.schemas import UserContext


class KnowledgeAgent:
    """知识库 Agent 入口：封装 LangGraph 的构建与调用。"""

    def __init__(self, deps: AgentDependencies) -> None:
        self.nodes = KnowledgeNodes(deps)
        self.graph = self._build()

    def _build(self):
        builder = StateGraph(AgentState)
        builder.add_node("rewrite", self.nodes.rewrite)
        builder.add_node("retrieve", self.nodes.retrieve)
        builder.add_node("tools", self.nodes.tools)
        builder.add_node("generate", self.nodes.generate)
        builder.add_node("verify", self.nodes.verify)
        builder.set_entry_point("rewrite")
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "tools")
        builder.add_edge("tools", "generate")
        builder.add_edge("generate", "verify")
        builder.add_conditional_edges("verify", route_after_verify, {"retry": "generate", "end": END})
        return builder.compile(checkpointer=MemorySaver())

    async def answer(
        self,
        question: str,
        user: UserContext,
        conversation_id: str,
        history: list[dict[str, str]],
        top_k: int | None = None,
        use_tools: bool = True,
    ) -> dict[str, Any]:
        state: AgentState = {
            "question": question,
            "user": user,
            "history": history,
            "conversation_id": conversation_id,
            "trace_id": uuid.uuid4().hex,
            "use_tools": use_tools,
            "top_k": top_k or 8,
            "step": 0,
        }
        config = {
            "configurable": {"thread_id": f"{user.tenant_id}:{user.user_id}:{conversation_id}"},
            "recursion_limit": 20,
        }
        return await self.graph.ainvoke(state, config=config)
