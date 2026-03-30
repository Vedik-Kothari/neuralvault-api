# =====================================================
# rag/pipeline.py
# Full RAG pipeline — orchestrates everything.
#
# This is the "main function" of the entire system.
# It connects: injection defense → retriever → LLM
#
# LCEL-INSPIRED DESIGN:
# LangChain Expression Language (LCEL) chains steps like:
#   retriever | prompt | llm | output_parser
#
# We implement the same concept manually in Python
# (without importing all of LangChain) so you can see
# exactly what each step does — better for learning
# and debugging.
#
# FLOW:
# query → defend → retrieve → build_prompt → llm → filter → respond
# =====================================================

import time
from loguru import logger
from dataclasses import dataclass

from ..models.schemas import UserProfile
from ..retriever.retriever import retrieve_chunks
from ..rag.injection_defender import analyze_query, filter_response
from ..rag.prompt_builder import build_messages, build_no_access_response
from ..rag.llm import generate_answer
from ..core.supabase import supabase_admin


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    answer: str
    access_granted: bool
    chunks_used: list[dict]
    chunks_count: int
    injection_detected: bool
    latency_ms: float
    threat_level: str


async def run_rag_pipeline(
    query: str,
    user: UserProfile,
    max_chunks: int = 5,
) -> RAGResponse:
    """
    Execute the full RAG pipeline securely.

    Steps:
    1. Defend against injection
    2. Retrieve authorized chunks
    3. Handle no-access case
    4. Build secure prompt
    5. Generate LLM answer
    6. Filter output
    7. Log audit trail
    8. Return response

    Args:
        query:      User's natural language question
        user:       Authenticated user with role + department
        max_chunks: Max context chunks to use

    Returns:
        RAGResponse with answer and metadata
    """
    start_time = time.time()
    injection_detected = False

    logger.info(
        f"RAG pipeline started | "
        f"user={user.email} role={user.role} "
        f"query='{query[:60]}'"
    )

    # ── Step 1: Injection Defense ─────────────────────────
    defense = analyze_query(query)

    if not defense.is_safe:
        injection_detected = True
        latency_ms = (time.time() - start_time) * 1000

        logger.warning(
            f"INJECTION BLOCKED | "
            f"user={user.email} "
            f"threat={defense.threat_level} "
            f"patterns={defense.detected_patterns}"
        )

        # Log the blocked attempt
        await _write_audit_log(
            user=user,
            query=query,
            response="BLOCKED: Injection attempt",
            chunks=[],
            access_granted=False,
            injection_detected=True,
            latency_ms=latency_ms,
        )

        return RAGResponse(
            answer=f"Request blocked: {defense.block_reason}",
            access_granted=False,
            chunks_used=[],
            chunks_count=0,
            injection_detected=True,
            latency_ms=latency_ms,
            threat_level=defense.threat_level,
        )

    # Use sanitized version of query from here on
    safe_query = defense.sanitized_query

    # ── Step 2: Retrieve Authorized Chunks ────────────────
    retrieval = retrieve_chunks(
        query=safe_query,
        user=user,
        max_chunks=max_chunks,
    )

    # ── Step 3: Handle No-Access Case ────────────────────
    if not retrieval.access_granted or not retrieval.chunks:
        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"No authorized chunks found | "
            f"user={user.email} query='{safe_query[:40]}'"
        )

        await _write_audit_log(
            user=user,
            query=safe_query,
            response="Access Restricted or Not Available",
            chunks=[],
            access_granted=False,
            injection_detected=False,
            latency_ms=latency_ms,
        )

        return RAGResponse(
            answer=build_no_access_response(),
            access_granted=False,
            chunks_used=[],
            chunks_count=0,
            injection_detected=False,
            latency_ms=latency_ms,
            threat_level="none",
        )

    # ── Step 4: Build Secure Prompt ───────────────────────
    messages = build_messages(
        query=safe_query,
        chunks=retrieval.chunks,
        user=user,
    )

    logger.info(
        f"Prompt built | "
        f"chunks={len(retrieval.chunks)} "
        f"messages={len(messages)}"
    )

    # ── Step 5: Generate LLM Answer ───────────────────────
    try:
        raw_answer = generate_answer(
            messages=messages,
            temperature=0.1,   # low = factual, consistent
            max_tokens=1024,
        )
        if not isinstance(raw_answer, str):
         raw_answer = str(raw_answer)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        latency_ms = (time.time() - start_time) * 1000

        return RAGResponse(
            answer="The AI service is temporarily unavailable. Please try again.",
            access_granted=True,
            chunks_used=retrieval.chunks,
            chunks_count=len(retrieval.chunks),
            injection_detected=False,
            latency_ms=latency_ms,
            threat_level="none",
        )

    # ── Step 6: Filter Output ─────────────────────────────
    output_safe, final_answer = filter_response(raw_answer)

    if not output_safe:
        logger.warning(
            f"Output filtered | user={user.email}"
        )

    # ── Step 7: Audit Log ─────────────────────────────────
    latency_ms = (time.time() - start_time) * 1000

    await _write_audit_log(
        user=user,
        query=safe_query,
        response=final_answer[:500],
        chunks=retrieval.chunks,
        access_granted=True,
        injection_detected=injection_detected,
        latency_ms=latency_ms,
    )

    logger.info(
        f"RAG pipeline complete | "
        f"latency={latency_ms:.0f}ms "
        f"chunks={len(retrieval.chunks)}"
    )

    return RAGResponse(
        answer=final_answer,
        access_granted=True,
        chunks_used=retrieval.chunks,
        chunks_count=len(retrieval.chunks),
        injection_detected=injection_detected,
        latency_ms=latency_ms,
        threat_level=defense.threat_level,
    )


async def _write_audit_log(
    user: UserProfile,
    query: str,
    response: str,
    chunks: list[dict],
    access_granted: bool,
    injection_detected: bool,
    latency_ms: float,
) -> None:
    """Write complete audit log entry to database."""
    try:
        supabase_admin.table("audit_logs").insert({
            "user_id":             str(user.id),
            "query":               query[:500],
            "user_role":           user.role.value,
            "user_department":     user.department,
            "access_granted":      access_granted,
            "chunks_retrieved":    len(chunks),
            "retrieved_chunk_ids": [c.get("id") for c in chunks],
            "response_preview":    response[:300],
            "injection_detected":  injection_detected,
            "latency_ms":          latency_ms,
        }).execute()

    except Exception as e:
        logger.error(f"Audit log failed (non-fatal): {e}")