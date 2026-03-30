# =====================================================
# retriever/validator.py
# Post-retrieval chunk validation.
#
# WHY validate AFTER retrieval if RLS already filtered?
# Defense in depth — this is Layer 1 checking Layer 2's work.
#
# REAL SCENARIO WHERE THIS MATTERS:
# Imagine a bug in the RLS policy (SQL mistake).
# Without this validator, unauthorized chunks reach the LLM.
# With this validator, they're caught BEFORE the LLM sees them.
#
# This validator also catches:
# - Department mismatches (RLS might allow, but dept doesn't match)
# - Similarity scores that are too low (irrelevant chunks)
# - Empty or malformed chunks
# =====================================================

from loguru import logger
from ..models.schemas import UserProfile, RoleType


# Mirror of ROLE_PRIORITY in rbac.py
ROLE_PRIORITY = {
    "intern":   1,
    "employee": 2,
    "manager":  3,
    "admin":    4,
}


def validate_chunk_access(
    chunk: dict,
    user: UserProfile,
) -> bool:
    """
    Final application-level check: can this user see this chunk?

    This runs AFTER the database returns results.
    It re-verifies every chunk independently.

    Args:
        chunk: A chunk dict returned from the DB
        user:  The authenticated user

    Returns:
        True if the user can access this chunk, False otherwise.
    """
    user_priority = ROLE_PRIORITY.get(user.role.value, 0)

    # --- Check 1: Role hierarchy ---
    chunk_roles = chunk.get("role_access", [])

    if not chunk_roles:
        # No role_access defined = block by default
        logger.warning(f"Chunk {chunk.get('id')} has no role_access — blocking")
        return False

    # User must have >= priority of at least one allowed role
    role_allowed = any(
        ROLE_PRIORITY.get(r, 99) <= user_priority
        for r in chunk_roles
    )

    if not role_allowed:
        logger.warning(
            f"Role check failed | chunk_roles={chunk_roles} "
            f"user_role={user.role} user_priority={user_priority}"
        )
        return False

    # --- Check 2: Department isolation ---
    chunk_dept = chunk.get("department", "general")
    user_dept  = user.department

    dept_allowed = (
        chunk_dept == "general"
        or chunk_dept == user_dept
    )

    if not dept_allowed:
        logger.warning(
            f"Department check failed | "
            f"chunk_dept={chunk_dept} user_dept={user_dept}"
        )
        return False

    return True


def validate_chunks(
    chunks: list[dict],
    user: UserProfile,
    min_similarity: float = 0.3,
) -> tuple[list[dict], list[dict]]:
    """
    Filter a list of retrieved chunks — keep only authorized ones.

    Returns:
        (authorized_chunks, blocked_chunks)
        Both lists are returned so we can log the blocked ones.
    """
    authorized = []
    blocked    = []

    for chunk in chunks:
        # Check similarity threshold
        similarity = chunk.get("similarity", 0)
        if similarity < min_similarity:
            logger.debug(
                f"Chunk filtered: low similarity "
                f"{similarity:.3f} < {min_similarity}"
            )
            blocked.append(chunk)
            continue

        # Check access
        if validate_chunk_access(chunk, user):
            authorized.append(chunk)
        else:
            blocked.append(chunk)
            # Log every blocked chunk — important for audit
            logger.warning(
                f"SECURITY: Chunk blocked post-retrieval | "
                f"chunk_id={chunk.get('id')} "
                f"user={user.email} role={user.role}"
            )

    logger.info(
        f"Chunk validation complete | "
        f"authorized={len(authorized)} blocked={len(blocked)}"
    )

    return authorized, blocked