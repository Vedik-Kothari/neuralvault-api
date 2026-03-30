# =====================================================
# routers/query.py
# Query endpoint — secure retrieval API.
#
# This is a PREVIEW endpoint for Phase 5.
# Right now it returns raw chunks.
# In Phase 5, we'll pipe these chunks into the LLM
# to generate a natural language answer.
#
# Endpoints:
#   POST /query/search    → retrieve relevant chunks
#   POST /query/compare   → admin: compare across roles
# =====================================================

from fastapi import APIRouter, HTTPException, status, Depends
from loguru import logger
from pydantic import BaseModel
import time

from ..middleware.rbac import get_current_user, require_admin
from ..models.schemas import UserProfile, QueryRequest
from ..retriever.retriever import retrieve_chunks, retrieve_for_roles_comparison

router = APIRouter(prefix="/query", tags=["Query"])


class SearchResponse(BaseModel):
    """Response for a secure chunk retrieval."""
    chunks: list[dict]
    total_returned: int
    access_granted: bool
    message: str


class RoleCompareRequest(BaseModel):
    query: str
    roles: list[str] = ["intern", "employee", "manager", "admin"]
    department: str = "general"


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieve relevant chunks for a query (RBAC enforced)"
)
async def search_chunks(
    request: QueryRequest,
    current_user: UserProfile = Depends(get_current_user),
):
    """
    Retrieve document chunks relevant to the query.
    Only returns chunks the current user is authorized to see.

    In Phase 5, these chunks are fed to the LLM to
    generate a natural language answer.
    """
    start = time.time()

    result = retrieve_chunks(
        query=request.query,
        user=current_user,
        max_chunks=request.max_chunks,
    )

    latency_ms = (time.time() - start) * 1000

    if not result.access_granted:
        return SearchResponse(
            chunks=[],
            total_returned=0,
            access_granted=False,
            message=(
                "No relevant information found for your query. "
                "This may be because: (1) no documents match your query, "
                "or (2) matching documents are restricted to higher roles."
            )
        )

    # Format chunks for response (remove embedding vectors — too large)
    clean_chunks = [
        {
            "id":          c.get("id"),
            "content":     c.get("content"),
            "source":      c.get("source"),
            "similarity":  round(c.get("similarity", 0), 4),
            "role_access": c.get("role_access"),
            "department":  c.get("department"),
        }
        for c in result.chunks
    ]

    logger.info(
        f"Search complete | user={current_user.email} "
        f"chunks={len(clean_chunks)} latency={latency_ms:.0f}ms"
    )

    return SearchResponse(
        chunks=clean_chunks,
        total_returned=len(clean_chunks),
        access_granted=True,
        message=f"Found {len(clean_chunks)} relevant chunk(s)."
    )


@router.post(
    "/compare",
    summary="ADMIN: Compare chunk visibility across roles"
)
async def compare_roles(
    request: RoleCompareRequest,
    current_user: UserProfile = Depends(require_admin),
):
    """
    Admin-only endpoint: shows how many chunks each role
    would see for the same query.

    Useful for:
    - Verifying RBAC is working correctly
    - Demonstrating the system in interviews/demos
    - Debugging access control issues
    """
    results = retrieve_for_roles_comparison(
        query=request.query,
        roles=request.roles,
        department=request.department,
    )

    # Return summary (not full chunks — just counts + previews)
    summary = {}
    for role, chunks in results.items():
        summary[role] = {
            "chunks_visible": len(chunks),
            "previews": [
                c.get("content", "")[:80] + "..."
                for c in chunks[:2]   # first 2 previews only
            ]
        }

    return {
        "query": request.query,
        "role_comparison": summary,
        "message": "Higher roles see all lower role content plus their own."
    }