# =====================================================
# ingestion/pipeline.py
# Orchestrates the full ingestion process.
#
# This is the "conductor" — it calls parser, chunker,
# embedder in order and stores everything in Supabase.
#
# FLOW:
# document_id → fetch file from storage → parse text
#   → chunk → embed → store chunks → update doc status
# =====================================================

import time
from loguru import logger
from dataclasses import dataclass

from ..core.supabase import supabase_admin
from ..core.config import settings
from .parser import parse_document
from .chunker import chunk_text
from .embedder import embed_chunks


@dataclass
class IngestionResult:
    """Result returned after ingestion completes."""
    document_id: str
    filename: str
    chunks_created: int
    success: bool
    error: str | None = None
    duration_seconds: float = 0.0


async def ingest_document(document_id: str) -> IngestionResult:
    """
    Full ingestion pipeline for a single document.

    Steps:
    1. Fetch document metadata from DB
    2. Download file from Supabase Storage
    3. Parse text based on file type
    4. Split into chunks
    5. Generate embeddings for all chunks
    6. Store chunks + embeddings in document_chunks table
    7. Update document status to 'completed'

    SECURITY:
    - role_access is copied from the document to EVERY chunk
    - This means access control is enforced at the chunk level
    - Even if someone queries chunks directly, RLS applies
    """
    start_time = time.time()

    # ── Step 1: Fetch document metadata ──────────────────
    logger.info(f"Starting ingestion | document_id={document_id}")

    try:
        doc_result = supabase_admin\
            .table("documents")\
            .select("*")\
            .eq("id", document_id)\
            .single()\
            .execute()

        if not doc_result.data:
            return IngestionResult(
                document_id=document_id,
                filename="unknown",
                chunks_created=0,
                success=False,
                error="Document not found in database"
            )

        doc = doc_result.data
        filename = doc["filename"]
        file_type = doc["file_type"]
        storage_path = doc["storage_path"]
        role_access = doc["role_access"]      # e.g. ["employee", "manager"]
        department = doc["department"]

        logger.info(f"Document found | filename={filename} type={file_type}")

    except Exception as e:
        logger.error(f"Failed to fetch document {document_id}: {e}")
        return IngestionResult(
            document_id=document_id,
            filename="unknown",
            chunks_created=0,
            success=False,
            error=f"Database fetch failed: {str(e)}"
        )

    # ── Step 2: Update status to 'processing' ────────────
    supabase_admin.table("documents")\
        .update({"status": "processing"})\
        .eq("id", document_id)\
        .execute()

    try:
        # ── Step 3: Download file from Supabase Storage ──
        logger.info(f"Downloading file from storage | path={storage_path}")

        file_response = supabase_admin\
            .storage\
            .from_("documents")\
            .download(storage_path)

        # file_response is raw bytes
        file_bytes = file_response
        logger.info(f"File downloaded | size={len(file_bytes)/1024:.1f}KB")

        # ── Step 4: Parse text ────────────────────────────
        logger.info(f"Parsing {file_type} document...")
        text = parse_document(file_bytes, file_type)
        logger.info(f"Text extracted | chars={len(text)}")

        # ── Step 5: Chunk text ────────────────────────────
        logger.info("Splitting text into chunks...")
        chunks = chunk_text(
            text=text,
            source=filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        logger.info(f"Chunks created | count={len(chunks)}")

        # ── Step 6: Generate embeddings ───────────────────
        logger.info("Generating embeddings...")
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = embed_chunks(chunk_texts)
        logger.info(f"Embeddings generated | count={len(embeddings)}")

        # ── Step 7: Store chunks in database ──────────────
        logger.info("Storing chunks in database...")

        # Build rows for bulk insert
        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            doc_metadata = doc.get("metadata") or {}

            rows.append({
                "document_id":  document_id,
                "content":      chunk.content,
                "embedding":    embedding,
                "chunk_index":  chunk.chunk_index,
                "role_access":  role_access,
                "department":   department,
                "source":       filename,
                "metadata": {
                    "char_start":    chunk.char_start,
        "char_end":      chunk.char_end,
        "filename":      filename,
        # Propagate auto-extracted metadata to chunks
        "document_type": doc_metadata.get("document_type", "other"),
        "topics":        doc_metadata.get("topics", []),
        "sensitivity":   doc_metadata.get("sensitivity", "internal"),
        "summary":       doc_metadata.get("summary", ""),
    }
})

        # Insert in batches of 50 (Supabase has request size limits)
        batch_size = 50
        total_inserted = 0

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            result = supabase_admin\
                .table("document_chunks")\
                .insert(batch)\
                .execute()
            total_inserted += len(result.data)
            logger.info(f"Inserted batch {i//batch_size + 1} | {total_inserted}/{len(rows)} chunks")

        # ── Step 8: Update document status ────────────────
        supabase_admin.table("documents")\
            .update({
                "status":      "completed",
                "chunk_count": total_inserted,
            })\
            .eq("id", document_id)\
            .execute()

        duration = time.time() - start_time
        logger.info(
            f"Ingestion complete | document={filename} "
            f"chunks={total_inserted} duration={duration:.1f}s"
        )

        return IngestionResult(
            document_id=document_id,
            filename=filename,
            chunks_created=total_inserted,
            success=True,
            duration_seconds=duration,
        )

    except Exception as e:
        # ── On failure: mark document as failed ───────────
        logger.error(f"Ingestion failed for {document_id}: {e}")

        supabase_admin.table("documents")\
            .update({"status": "failed"})\
            .eq("id", document_id)\
            .execute()

        return IngestionResult(
            document_id=document_id,
            filename=doc.get("filename", "unknown"),
            chunks_created=0,
            success=False,
            error=str(e),
            duration_seconds=time.time() - start_time,
        )