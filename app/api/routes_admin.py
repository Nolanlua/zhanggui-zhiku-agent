from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_container, get_user
from app.api.schemas import EvalRequest, EvalResponse, UserContext

router = APIRouter()


@router.post("/admin/eval", response_model=EvalResponse)
async def run_eval(
    payload: EvalRequest,
    user: UserContext = Depends(get_user),
    container=Depends(get_container),
) -> EvalResponse:
    result = await container.evaluation.run(payload.dataset_path, user, top_k=payload.top_k)
    return EvalResponse(**result)
