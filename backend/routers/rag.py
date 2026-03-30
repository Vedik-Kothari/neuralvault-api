# =====================================================
# routers/rag.py
# The main chat endpoint — what the frontend calls.
#
# Endpoints:
#   POST /rag/chat    → full RAG pipeline
#   GET  /rag/history → user's query history
# =====================================================

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from loguru import logger

from ..middleware.rbac import get_current_user
from ..models.schemas import UserProfile
from ..rag.pipeline import run_rag_pipeline

router = APIRouter(prefix="/rag", tags=["RAG"])


class ChatRequest(BaseModel):
    query: str
    max_chunks: int = 5

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        if len(v) > 2000:
            raise ValueError("Query too long (max 2000 chars)")
        return v


class ChatResponse(BaseModel):
    answer: str
    access_granted: bool
    chunks_count: int
    injection_detected: bool
    latency_ms: float
    sources: list[str]      # filenames used to answer


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question — full RAG pipeline with RBAC"
)
async def chat(
    request: ChatRequest,
    current_user: UserProfile = Depends(get_current_user),
):
    """
    Main chat endpoint.
    - Defends against injection
    - Retrieves only authorized chunks
    - Generates grounded LLM answer
    - Logs everything for audit
    """
    result = await run_rag_pipeline(
        query=request.query,
        user=current_user,
        max_chunks=request.max_chunks,
    )

    # Extract unique source filenames for citation
    sources = list(set(
        c.get("source", "Unknown")
        for c in result.chunks_used
    ))

    return ChatResponse(
        answer=result.answer,
        access_granted=result.access_granted,
        chunks_count=result.chunks_count,
        injection_detected=result.injection_detected,
        latency_ms=round(result.latency_ms, 2),
        sources=sources,
    )


@router.get(
    "/history",
    summary="Get current user's query history"
)
async def get_history(
    current_user: UserProfile = Depends(get_current_user),
    limit: int = 20,
):
    """Returns the last N queries made by this user."""
    from ..core.supabase import supabase_admin

    result = supabase_admin\
        .table("audit_logs")\
        .select(
            "query, access_granted, chunks_retrieved, "
            "injection_detected, latency_ms, created_at"
        )\
        .eq("user_id", str(current_user.id))\
        .order("created_at", desc=True)\
        .limit(limit)\
        .execute()

    return {
        "history": result.data,
        "total": len(result.data)
    }