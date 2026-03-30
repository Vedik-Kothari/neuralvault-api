# =====================================================
# ingestion/chunker.py
# Splits long text into smaller overlapping chunks.
#
# WHY CHUNK AT ALL?
# LLMs have a "context window" — a max number of tokens
# they can process at once. A 50-page document has ~25,000
# tokens — way too much. We split it into ~500-char pieces
# and only send the RELEVANT pieces to the LLM.
#
# WHY OVERLAP?
# If a sentence is cut across two chunks, neither chunk
# has the full context. Overlapping (100 chars) means
# important sentences appear in at least one complete chunk.
#
# Chunk 1: "...The revenue was $2M in Q3. The board..."
# Chunk 2: "...Q3. The board approved a 10% raise..."
#           ↑ overlap ensures "Q3" context isn't lost
# =====================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from dataclasses import dataclass


@dataclass
class TextChunk:
    """
    A single chunk of text with its metadata.
    We use a dataclass (not dict) so fields are typed and
    autocomplete works in your editor.
    """
    content: str          # The actual text
    chunk_index: int      # Position in the original document
    source: str           # Filename (for citations)
    char_start: int       # Character offset in original text
    char_end: int         # Character offset end


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter.

    WHY RecursiveCharacterTextSplitter?
    It tries to split on natural boundaries in this order:
      1. Paragraph breaks (\n\n)
      2. Line breaks (\n)
      3. Sentences (. )
      4. Words ( )
      5. Characters (last resort)

    This means chunks end at natural breakpoints rather than
    cutting mid-sentence — better for coherent embeddings.

    Args:
        text:          Full document text
        source:        Filename (stored with each chunk for citation)
        chunk_size:    Max characters per chunk (default 500)
        chunk_overlap: Characters shared between adjacent chunks

    Returns:
        List of TextChunk objects
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Try these separators in order
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    # Split text — returns list of strings
    raw_chunks = splitter.split_text(text)

    # Filter out chunks that are too short to be meaningful
    # (e.g. a page header like "Page 3" alone)
    meaningful_chunks = [c for c in raw_chunks if len(c.strip()) > 50]

    # Build TextChunk objects with position tracking
    chunks = []
    search_start = 0

    for idx, chunk_text_content in enumerate(meaningful_chunks):
        # Find where this chunk appears in the original text
        char_start = text.find(chunk_text_content[:50], search_start)
        char_end = char_start + len(chunk_text_content)

        chunks.append(TextChunk(
            content=chunk_text_content.strip(),
            chunk_index=idx,
            source=source,
            char_start=max(0, char_start),
            char_end=char_end,
        ))

        # Move search forward (with overlap allowance)
        search_start = max(0, char_end - chunk_overlap - 10)

    logger.info(
        f"Chunking complete | source={source} "
        f"total_chars={len(text)} chunks={len(chunks)} "
        f"avg_size={sum(len(c.content) for c in chunks)//max(len(chunks),1)}"
    )

    return chunks