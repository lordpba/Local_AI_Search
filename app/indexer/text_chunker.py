"""
TextChunker — Splits text into overlapping chunks for embedding.
Recursive character splitting with smart separators.
Includes regex-based metadata extraction for structured data (codici fiscali, etc.).
"""

import logging
import re
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


# ─── Structured metadata extraction ──────────────────────────────────────────

# Italian codice fiscale: 6 letters + 2 digits + 1 letter + 2 digits + 1 letter + 3 digits + 1 letter
_RE_CODICE_FISCALE = re.compile(r"\b[A-Z]{6}\d{2}[A-EHLMPR-T]\d{2}[A-Z]\d{3}[A-Z]\b")

# Italian Partita IVA: 11 digits
_RE_PARTITA_IVA = re.compile(r"\b\d{11}\b")

# IBAN (Italian format: IT + 2 digits + letter + 5 digits + 5 digits + 12 alphanums)
_RE_IBAN = re.compile(r"\bIT\s?\d{2}\s?[A-Z]\s?(?:\d{5}\s?){2}[\dA-Z]{12}\b", re.IGNORECASE)

# Date patterns (dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy)
_RE_DATE = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")


def _extract_structured_metadata(text: str) -> dict:
    """
    Extract structured data from text using regex.
    Returns a dict of found entities to store as chunk metadata.
    """
    meta = {}

    codici_fiscali = _RE_CODICE_FISCALE.findall(text)
    if codici_fiscali:
        meta["codici_fiscali"] = ",".join(set(codici_fiscali))

    iban_matches = _RE_IBAN.findall(text)
    if iban_matches:
        meta["iban"] = ",".join(set(iban_matches))

    dates = _RE_DATE.findall(text)
    if dates:
        meta["dates"] = ",".join(dates[:10])  # Cap to avoid huge metadata

    return meta


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

        # Prepend filename + page as header for better search & LLM context
        filename = doc.metadata.get("filename", "")
        page = doc.metadata.get("page", "")
        header = f"--- Documento: {filename}"
        if page:
            header += f", pagina {page}"
        header += " ---\n\n"
        text_with_header = header + text

        # OCR documents: keep entire page as single chunk (forms are structured,
        # splitting loses field-value associations)
        extraction = doc.metadata.get("extraction", "")
        if extraction in ("ocr-vision", "native") and len(text_with_header) <= chunk_size * 2:
            extracted = _extract_structured_metadata(text)
            all_chunks.append(TextChunk(
                text=text_with_header,
                metadata={**doc.metadata, "chunk_index": global_chunk_idx, **extracted}
            ))
            global_chunk_idx += 1
            continue

        # Special handling for tables: keep them intact if possible
        if _is_table(text) and len(text_with_header) <= chunk_size * 1.5:
            extracted = _extract_structured_metadata(text)
            all_chunks.append(TextChunk(
                text=text_with_header,
                metadata={**doc.metadata, "chunk_index": global_chunk_idx, "is_table": True, **extracted}
            ))
            global_chunk_idx += 1
            continue

        # Split text into chunks (for very long documents)
        text_chunks = _split_text(text_with_header, chunk_size, chunk_overlap)

        for j, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                extracted = _extract_structured_metadata(chunk_text)
                all_chunks.append(TextChunk(
                    text=chunk_text.strip(),
                    metadata={**doc.metadata, "chunk_index": global_chunk_idx, **extracted}
                ))
                global_chunk_idx += 1

    if progress_callback:
        progress_callback(1.0, f"Suddivisione completata: {len(all_chunks)} sezioni create")

    logger.info(f"Chunked {total} documents into {len(all_chunks)} chunks")
    return all_chunks
