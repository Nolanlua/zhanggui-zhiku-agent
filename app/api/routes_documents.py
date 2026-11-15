from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_container, get_user
from app.api.schemas import DocumentStatusResponse, DocumentUploadResponse, UserContext

router = APIRouter()
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    visibility: str = Form(default="public"),
    department_id: str | None = Form(default=None),
    user: UserContext = Depends(get_user),
    container=Depends(get_container),
) -> DocumentUploadResponse:
    if visibility not in ("public", "department", "private"):
        raise HTTPException(status_code=400, detail="invalid visibility")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")
    doc_id = await container.knowledge.submit(
        file_name=file.filename or "upload.txt",
        content=content,
        user=user,
        title=title,
        visibility=visibility,
        department_id=department_id,
    )
    return DocumentUploadResponse(doc_id=doc_id, status="pending", message="indexing scheduled")


@router.get("/documents/{doc_id}", response_model=DocumentStatusResponse)
async def get_document(
    doc_id: str,
    user: UserContext = Depends(get_user),
    container=Depends(get_container),
) -> DocumentStatusResponse:
    document = await container.knowledge.get_document(doc_id, user.tenant_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentStatusResponse(
        doc_id=document.id,
        title=document.title,
        status=document.status,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        error=document.error,
    )


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    user: UserContext = Depends(get_user),
    container=Depends(get_container),
) -> dict[str, str]:
    document = await container.knowledge.get_document(doc_id, user.tenant_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    if not container.permission.can_manage_document(user, document.owner_user_id, document.department_id):
        raise HTTPException(status_code=403, detail="permission denied")
    await container.knowledge.delete_document(document)
    return {"status": "deleted", "doc_id": doc_id}
