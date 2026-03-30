# =====================================================
# routers/ingestion.py
# API endpoint that triggers document ingestion.
#
# Endpoints:
#   POST /ingest/{document_id}  → trigger ingestion
#   GET  /ingest/status/{id}    → check ingestion status
# =====================================================

from fastapi import APIRouter, HTTPException, status, Depends
from loguru import logger
from pydantic import BaseModel

from ..core.supabase import supabase_admin
from ..middleware.rbac import get_current_user, require_employee
from ..models.schemas import UserProfile
from ..ingestion.pipeline import ingest_document

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    success: bool
    error: str | None
    duration_seconds: float


@router.post(
    "/{document_id}",
    response_model=IngestionResponse,
    summary="Trigger ingestion for an uploaded document"
)
async def trigger_ingestion(
    document_id: str,
    current_user: UserProfile = Depends(require_employee),
):
    """
    Starts the ingestion pipeline for a document.
    Only the uploader or admin can trigger ingestion.

    In production this would be an async background task.
    For simplicity here it runs synchronously.
    (We'll add async queuing as an enhancement later.)
    """
    # Verify document exists and user has permission
    doc_result = supabase_admin\
        .table("documents")\
        .select("uploaded_by, filename, status")\
        .eq("id", document_id)\
        .single()\
        .execute()

    if not doc_result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    doc = doc_result.data

    # Check ownership
    is_owner = doc["uploaded_by"] == str(current_user.id)
    is_admin = current_user.role.value == "admin"

    if not (is_owner or is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the uploader or admin can ingest this document."
        )

    # Prevent re-ingestion of already processed docs
    if doc["status"] == "processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is already being processed."
        )

    logger.info(
        f"Ingestion triggered | doc={document_id} "
        f"by={current_user.email}"
    )

    # Run ingestion pipeline
    result = await ingest_document(document_id)

    return IngestionResponse(
        document_id=result.document_id,
        filename=result.filename,
        chunks_created=result.chunks_created,
        success=result.success,
        error=result.error,
        duration_seconds=result.duration_seconds,
    )


@router.get(
    "/status/{document_id}",
    summary="Check ingestion status of a document"
)
async def get_ingestion_status(
    document_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """Returns the current processing status of a document."""
    result = supabase_admin\
        .table("documents")\
        .select("id, filename, status, chunk_count")\
        .eq("id", document_id)\
        .single()\
        .execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    return result.data