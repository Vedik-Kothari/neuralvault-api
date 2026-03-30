# =====================================================
# routers/documents.py
# Document management endpoints.
#
# Endpoints:
#   POST /documents/upload  → upload a file
#   GET  /documents/        → list accessible documents
#   GET  /documents/{id}    → get one document's metadata
#   DELETE /documents/{id}  → delete a document
#
# NOTE: Actual chunking + embedding happens in Phase 3.
# For now, upload stores the file and creates a DB record.
# =====================================================

from fastapi import (
    APIRouter, HTTPException, UploadFile, File,
    Form, Depends, status
)
from loguru import logger
from typing import Optional
import uuid

from ..core.supabase import supabase_admin
from ..middleware.rbac import (
    get_current_user, require_employee, require_admin
)
from ..models.schemas import (
    UserProfile, DocumentResponse, DocumentListResponse,
    RoleType, DocumentMetadata
)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Allowed file types and their max sizes
ALLOWED_TYPES = {"pdf", "docx", "txt"}
MAX_FILE_SIZE_MB = 10
MAX_FILES_PER_USER = 10     


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document for ingestion"
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, DOCX, or TXT file"),
    role_access: str = Form(...,
        description="Comma-separated roles: intern,employee,manager,admin"
    ),
    department: str = Form(default="general"),
    current_user: UserProfile = Depends(require_employee),
):
    """
    Upload a document and register it for processing.
    """

    # --- Check upload limit (FIXED: moved inside function) ---
    existing = supabase_admin.table("documents")\
        .select("id", count="exact")\
        .eq("uploaded_by", str(current_user.id))\
        .execute()

    if (existing.count or 0) >= MAX_FILES_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Upload limit reached ({MAX_FILES_PER_USER} files max). "
                   f"Delete existing files to upload more."
        )

    # --- Validate file type ---
    filename = file.filename or "unknown"
    extension = filename.rsplit(".", 1)[-1].lower()

    if extension not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '.{extension}' not supported. "
                   f"Allowed: {', '.join(ALLOWED_TYPES)}"
        )

    # --- Validate file size ---
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB"
        )

    # --- Parse and validate role_access ---
    try:
        parsed_roles = [
            RoleType(r.strip())
            for r in role_access.split(",")
            if r.strip()
        ]
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role in role_access: {e}"
        )

    user_priority_map = {
        RoleType.intern: 1, RoleType.employee: 2,
        RoleType.manager: 3, RoleType.admin: 4
    }
    user_priority = user_priority_map[current_user.role]

    min_role_in_access = min(
        user_priority_map[role] for role in parsed_roles
    )

    if min_role_in_access > user_priority:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"role_access must include at least one role "
                   f"at or below your own role '{current_user.role}'. "
                   f"You cannot create documents only accessible "
                   f"to roles higher than yours."
        )

    # --- Upload file to Supabase Storage ---
    storage_path = f"{current_user.department}/{uuid.uuid4()}/{filename}"

    try:
        supabase_admin.storage\
            .from_("documents")\
            .upload(storage_path, content)
    except Exception as e:
        logger.error(f"Storage upload failed for {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File storage failed. Please try again."
        )

    # --- Create document record in DB ---
    try:
        doc_data = {
            "uploaded_by": str(current_user.id),
            "filename": filename,
            "file_type": extension,
            "role_access": [r.value for r in parsed_roles],
            "department": department,
            "storage_path": storage_path,
            "status": "pending",
            "chunk_count": 0,
        }

        result = supabase_admin.table("documents")\
            .insert(doc_data)\
            .execute()

        doc = result.data[0]
        logger.info(
            f"Document uploaded | file={filename} "
            f"user={current_user.email} roles={parsed_roles}"
        )

        return DocumentResponse(**doc)

    except Exception as e:
        logger.error(f"DB insert failed for {filename}: {e}")
        supabase_admin.storage.from_("documents").remove([storage_path])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register document. Please try again."
        )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List documents accessible to current user"
)
async def list_documents(
    current_user: UserProfile = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0,
):
    try:
        result = supabase_admin.table("documents")\
            .select("*", count="exact")\
            .contains("role_access", [current_user.role.value])\
            .in_("department", [current_user.department, "general"])\
            .order("created_at", desc=True)\
            .range(offset, offset + limit - 1)\
            .execute()

        return DocumentListResponse(
            documents=[DocumentResponse(**d) for d in result.data],
            total=result.count or 0
        )

    except Exception as e:
        logger.error(f"List documents error for {current_user.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve documents."
        )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document (owner or admin only)"
)
async def delete_document(
    document_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    try:
        result = supabase_admin.table("documents")\
            .select("*")\
            .eq("id", document_id)\
            .single()\
            .execute()

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found."
            )

        doc = result.data

        is_owner = doc["uploaded_by"] == str(current_user.id)
        is_admin = current_user.role == RoleType.admin

        if not (is_owner or is_admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the uploader or admin can delete this document."
            )

        if doc.get("storage_path"):
            supabase_admin.storage\
                .from_("documents")\
                .remove([doc["storage_path"]])

        supabase_admin.table("documents")\
            .delete()\
            .eq("id", document_id)\
            .execute()

        logger.info(
            f"Document deleted | id={document_id} "
            f"by={current_user.email}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error for {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Delete failed."
        )