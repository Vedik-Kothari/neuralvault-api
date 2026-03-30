# =====================================================
# retriever/retriever.py
# The core secure retriever — the heart of Phase 4.
#
# This is where vector search meets RBAC.
# It calls the match_chunks() function we wrote in Phase 1
# which has BOTH vector similarity AND role filtering
# baked into the SQL.
#
# DUAL SECURITY FLOW:
# 1. Python builds query with user's role + department
# 2. Supabase RLS applies to the SQL function call
# 3. Python validator re-checks every returned chunk
# 4. Only chunks passing BOTH checks reach the LLM
# =====================================================

from loguru import logger
from dataclasses import dataclass

from ..core.supabase import supabase_admin
from ..core.config import settings
from ..models.schemas import UserProfile
from .embedder import embed_query
from .validator import validate_chunks


@dataclass
class RetrievalResult:
    """
    Result from a secure retrieval operation.
    Contains both the authorized chunks and security metadata.
    """
    chunks: list[dict]           # Authorized chunks to send to LLM
    blocked_count: int           # How many chunks were blocked
    query_embedding: list[float] # The embedded query vector
    access_granted: bool         # Whether ANY chunks were returned
    total_found: int             # Total before filtering


def retrieve_chunks(
    query: str,
    user: UserProfile,
    max_chunks: int = None,
    similarity_threshold: float = None,
) -> RetrievalResult:
    """
    Securely retrieve document chunks relevant to a query.

    This is the main function called by the RAG pipeline.
    It handles everything: embedding, searching, filtering.

    Args:
        query:               User's natural language question
        user:                Authenticated user (role + dept extracted)
        max_chunks:          Max chunks to return (default from config)
        similarity_threshold: Min similarity score (default from config)

    Returns:
        RetrievalResult with authorized chunks only

    SECURITY GUARANTEE:
    A user with role X will NEVER receive chunks that require
    role Y where Y > X, regardless of query content.
    This is enforced at both app and DB level.
    """
    max_chunks = max_chunks or settings.max_retrieval_chunks
    similarity_threshold = similarity_threshold or settings.similarity_threshold

    logger.info(
        f"Retrieval started | user={user.email} "
        f"role={user.role} dept={user.department} "
        f"query='{query[:60]}...'"
    )

    # ── Step 1: Embed the query ───────────────────────────
    query_embedding = embed_query(query)

    # ── Step 2: Call match_chunks() in Supabase ───────────
    # This SQL function (defined in Phase 1) does:
    # - Cosine similarity search against all embeddings
    # - Filters by role_access >= user's role
    # - Filters by department
    # - Returns top N results by similarity
    try:
        result = supabase_admin.rpc(
            "match_chunks",
            {
                "query_embedding":  query_embedding,
                "match_threshold":  similarity_threshold,
                "match_count":      max_chunks * 2,  # fetch 2x, filter down
                "filter_role":      user.role.value,
                "filter_dept":      user.department,
            }
        ).execute()

        raw_chunks = result.data or []
        logger.info(f"DB returned {len(raw_chunks)} raw chunks")

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        # Return empty result — never expose error details to user
        return RetrievalResult(
            chunks=[],
            blocked_count=0,
            query_embedding=query_embedding,
            access_granted=False,
            total_found=0,
        )

    # ── Step 3: Application-level validation ─────────────
    # Re-verify every chunk independently (defense in depth)
    authorized, blocked = validate_chunks(
        chunks=raw_chunks,
        user=user,
        min_similarity=similarity_threshold,
    )

    # ── Step 4: Trim to max_chunks ────────────────────────
    # Already sorted by similarity from DB, just trim
    final_chunks = authorized[:max_chunks]

    # ── Step 5: Log retrieval for audit ───────────────────
    _log_retrieval(
        user=user,
        query=query,
        authorized=final_chunks,
        blocked=blocked,
    )

    access_granted = len(final_chunks) > 0

    logger.info(
        f"Retrieval complete | "
        f"returned={len(final_chunks)} "
        f"blocked={len(blocked)} "
        f"access={'granted' if access_granted else 'denied'}"
    )

    return RetrievalResult(
        chunks=final_chunks,
        blocked_count=len(blocked),
        query_embedding=query_embedding,
        access_granted=access_granted,
        total_found=len(raw_chunks),
    )


def _log_retrieval(
    user: UserProfile,
    query: str,
    authorized: list[dict],
    blocked: list[dict],
) -> None:
    """
    Write retrieval audit log to database.
    Stored in audit_logs table for admin dashboard.

    WHY LOG EVERYTHING including denials?
    - Denials reveal attempted unauthorized access
    - Patterns of denials = potential security threat
    - Required for compliance (SOC2, GDPR, HIPAA)
    """
    try:
        log_entry = {
            "user_id":            str(user.id),
            "query":              query[:500],  # truncate long queries
            "user_role":          user.role.value,
            "user_department":    user.department,
            "access_granted":     len(authorized) > 0,
            "chunks_retrieved":   len(authorized),
            "retrieved_chunk_ids": [c.get("id") for c in authorized],
            "injection_detected": False,  # updated in Phase 5
        }

        supabase_admin.table("audit_logs").insert(log_entry).execute()

    except Exception as e:
        # Never let logging failure affect the user's request
        logger.error(f"Audit log failed (non-fatal): {e}")


def retrieve_for_roles_comparison(
    query: str,
    roles: list[str],
    department: str = "general",       # ← add this parameter
) -> dict[str, list[dict]]:
    """
    ADMIN UTILITY: Retrieve chunks for multiple roles at once.
    """
    from ..models.schemas import RoleType

    results = {}

    for role_name in roles:
        try:
            role = RoleType(role_name)
        except ValueError:
            continue

        embedding = embed_query(query)

        result = supabase_admin.rpc(
            "match_chunks",
            {
                "query_embedding": embedding,
                "match_threshold": 0.1,
                "match_count":     10,
                "filter_role":     role_name,
                "filter_dept":     department,    # ← use parameter
            }
        ).execute()

        results[role_name] = result.data or []
        logger.info(
            f"Role comparison | role={role_name} "
            f"chunks_visible={len(results[role_name])}"
        )

    return results