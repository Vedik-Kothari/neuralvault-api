# =====================================================
# retriever/embedder.py
# Embeds the user's query at search time.
#
# WHY a separate file from ingestion/embedder.py?
# At INGESTION time: we embed many chunks in batches
# At QUERY time: we embed ONE query string quickly
# Same model, different usage patterns.
# Keeping them separate makes each file easier to read.
#
# IMPORTANT: Must use the EXACT same model as ingestion.
# If ingestion used MiniLM and retrieval uses a different
# model, the vectors are in completely different spaces
# and similarity search returns garbage results.
# =====================================================

from loguru import logger

# Import the singleton model from ingestion embedder
# This ensures we load the model only ONCE across the app
from ..ingestion.embedder import get_embedding_model


def embed_query(query: str) -> list[float]:
    """
    Convert a user's search query into a 384-dim vector.

    This vector is then compared against all stored chunk
    vectors using cosine similarity to find the most
    semantically relevant chunks.

    Args:
        query: The user's natural language question

    Returns:
        List of 384 floats representing the query meaning
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    model = get_embedding_model()

    # encode() with normalize=True means we can use
    # cosine similarity = dot product (faster search)
    embedding = model.encode(
        query.strip(),
        normalize_embeddings=True,
    )

    logger.debug(f"Query embedded | query='{query[:50]}...' dim={len(embedding)}")

    return embedding.tolist()