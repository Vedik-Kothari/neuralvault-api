# =====================================================
# ingestion/embedder.py
# Converts text chunks into numerical vectors.
#
# WHY EMBEDDINGS?
# Computers can't compare meaning of text directly.
# Embeddings convert text → 384 numbers (a vector).
# Similar texts get similar vectors.
# We can then find relevant chunks by comparing vectors.
#
# EXAMPLE:
# "What is the salary policy?" → [0.2, -0.5, 0.8, ...]
# "Employee compensation rules" → [0.19, -0.48, 0.79, ...]
#   ↑ very similar vectors = similar meaning ✓
# "The cat sat on the mat"    → [-0.6, 0.3, -0.1, ...]
#   ↑ very different vectors = unrelated ✓
#
# MODEL: all-MiniLM-L6-v2
# - FREE (runs locally, no API key)
# - Fast (designed for semantic search)
# - 384 dimensions (small but effective)
# - Downloads automatically on first use (~90MB)
# =====================================================

from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np


# Global model instance — loaded ONCE when module is imported
# Loading takes ~2-3 seconds, so we don't want to do it per request
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Returns the embedding model, loading it if not already loaded.
    Singleton pattern — only one instance in memory.
    """
    global _model

    if _model is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2...")
        logger.info("(First load downloads ~90MB — please wait)")

        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        logger.info("Embedding model loaded successfully!")

    return _model


def embed_text(text: str) -> list[float]:
    """
    Convert a single text string into a 384-dim embedding vector.

    Used at QUERY TIME to embed the user's question,
    so we can find similar chunks in the database.

    Returns:
        List of 384 floats
    """
    model = get_embedding_model()

    # encode() returns a numpy array — convert to Python list
    # for JSON serialization and Supabase storage
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_chunks(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Convert multiple text chunks into embeddings efficiently.

    WHY BATCH?
    Processing texts one-by-one is slow.
    Batching lets the GPU/CPU process multiple texts in parallel.
    batch_size=32 is a safe default for most machines.

    Used at INGESTION TIME to embed all document chunks.

    Args:
        texts:      List of text strings to embed
        batch_size: How many to process at once

    Returns:
        List of embeddings (each is a list of 384 floats)
    """
    if not texts:
        return []

    model = get_embedding_model()

    logger.info(f"Embedding {len(texts)} chunks in batches of {batch_size}...")

    # sentence-transformers handles batching internally
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,    # normalize to unit length
                                      # makes cosine similarity = dot product
                                      # slightly faster search
        show_progress_bar=len(texts) > 10,  # show progress for large batches
    )

    logger.info(f"Embedding complete | shape={embeddings.shape}")

    # Convert numpy array to list of lists for JSON/Supabase
    return embeddings.tolist()