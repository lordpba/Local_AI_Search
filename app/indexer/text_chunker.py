"""
TextChunker — Splits text into overlapping chunks for embedding.
Recursive character splitting with smart separators.
"""

import logging
from dataclasses import dataclass

from app.config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS

logger = logging.getLogger(__name__)

# Separators in priority order: try to split at paragraph boundaries first,
# then sentences, then commas, then words.
SEPARATORS = ["\n\n", "\n", ". ", ", ", " "]


@dataclass
class TextChunk:
    """A chunk of text with inherited metadata."""
    text: str
    metadata: dict


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Recursively split text using hierarchical separators.
    Tries paragraph breaks first, falls back to smaller separators.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try each separator in priority order
    for sep in SEPARATORS:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""

            for part in parts:
                candidate = current + sep + part if current else part

                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # If single part is too long, split recursively
                    if len(part) > chunk_size:
                        sub_chunks = _split_text(part, chunk_size, chunk_overlap)
                        chunks.extend(sub_chunks)
                        current = ""
                    else:
                        current = part

            if current:
                chunks.append(current)

            # Apply overlap: prepend end of previous chunk to current
            if chunk_overlap > 0 and len(chunks) > 1:
                overlapped = [chunks[0]]
                for i in range(1, len(chunks)):
                    prev_text = chunks[i - 1]
                    overlap_text = prev_text[-chunk_overlap:] if len(prev_text) > chunk_overlap else prev_text
                    # Only prepend overlap if it doesn't make chunk too large
                    combined = overlap_text + sep + chunks[i]
                    if len(combined) <= chunk_size * 1.2:  # Allow 20% overflow for overlap
                        overlapped.append(combined)
                    else:
                        overlapped.append(chunks[i])
                return overlapped

            return chunks

    # Fallback: hard split at chunk_size boundaries
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _is_table(text: str) -> bool:
    """Detect if text chunk contains a table (markdown pipes or aligned columns)."""
    lines = text.strip().split("\n")
    pipe_lines = sum(1 for line in lines if "|" in line and line.count("|") >= 2)
    return pipe_lines >= 3


def chunk_documents(
    documents: list,
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
    progress_callback=None,
) -> list[TextChunk]:
    """
    Split documents into chunks for embedding.

    Args:
        documents: List of Document objects
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between consecutive chunks in characters
        progress_callback: Optional callback(progress_float, status_str)

    Returns:
        List of TextChunk objects with inherited metadata
    """
    all_chunks = []
    total = len(documents)
    global_chunk_idx = 0  # file-global counter to guarantee unique chunk_index

    for i, doc in enumerate(documents):
        if progress_callback:
            progress_callback(
                i / total,
                f"Suddivisione testo: {doc.metadata.get('filename', '?')}..."
            )

        if doc.is_empty:
            continue

        text = doc.text.strip()

        # Special handling for tables: keep them intact if possible
        if _is_table(text) and len(text) <= chunk_size * 1.5:
            all_chunks.append(TextChunk(
                text=text,
                metadata={**doc.metadata, "chunk_index": global_chunk_idx, "is_table": True}
            ))
            global_chunk_idx += 1
            continue

        # Split text into chunks
        text_chunks = _split_text(text, chunk_size, chunk_overlap)

        for j, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                all_chunks.append(TextChunk(
                    text=chunk_text.strip(),
                    metadata={**doc.metadata, "chunk_index": global_chunk_idx}
                ))
                global_chunk_idx += 1

    if progress_callback:
        progress_callback(1.0, f"Suddivisione completata: {len(all_chunks)} sezioni create")

    logger.info(f"Chunked {total} documents into {len(all_chunks)} chunks")
    return all_chunks
