from __future__ import annotations

import json
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_container, get_user
from app.api.schemas import ChatRequest, ChatResponse, UserContext

router = APIRouter()


def _to_response(result: dict, latency_ms: int) -> dict:
    citations = result.get("citations", [])
    return {
        "answer": result.get("answer", ""),
        "citations": [item.model_dump() if hasattr(item, "model_dump") else item for item in citations],
        "rewritten_queries": result.get("rewritten_queries", []),
        "latency_ms": latency_ms,
        "trace_id": result.get("trace_id", ""),
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    user: UserContext = Depends(get_user),
    container=Depends(get_container),
) -> ChatResponse:
    start = time.perf_counter()
    history = await container.conversation.get_history(user.tenant_id, payload.conversation_id)
    result = await container.agent.answer(
        question=payload.question,
        user=user,
        conversation_id=payload.conversation_id,
        history=history,
        top_k=payload.top_k,
        use_tools=payload.use_tools,
    )
    await container.conversation.append(user.tenant_id, payload.conversation_id, "user", payload.question)
    await container.conversation.append(user.tenant_id, payload.conversation_id, "assistant", result.get("answer", ""))
    latency_ms = int((time.perf_counter() - start) * 1000)
    data = _to_response(result, latency_ms)
    return ChatResponse(**data)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    user: UserContext = Depends(get_user),
    container=Depends(get_container),
) -> StreamingResponse:
    """SSE 示例：先返回状态事件，再返回最终答案。

    生产可以换成 graph.astream_events，把 token 和节点状态实时推给前端。
    """

    async def event_generator() -> AsyncIterator[str]:
        yield "event: status\ndata: {\"stage\":\"started\"}\n\n"
        start = time.perf_counter()
        history = await container.conversation.get_history(user.tenant_id, payload.conversation_id)
        result = await container.agent.answer(
            question=payload.question,
            user=user,
            conversation_id=payload.conversation_id,
            history=history,
            top_k=payload.top_k,
            use_tools=payload.use_tools,
        )
        await container.conversation.append(user.tenant_id, payload.conversation_id, "user", payload.question)
        await container.conversation.append(user.tenant_id, payload.conversation_id, "assistant", result.get("answer", ""))
        latency_ms = int((time.perf_counter() - start) * 1000)
        payload_data = _to_response(result, latency_ms)
        yield "event: answer\ndata: " + json.dumps(payload_data, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
