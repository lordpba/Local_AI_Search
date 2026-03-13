"""
Embedder — Generates vector embeddings via Ollama bge-m3 API.
Handles batch embedding for indexing and single embedding for queries.
"""

import logging
import hashlib

import httpx

from app.config import OLLAMA_HOST, EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using bge-m3 via Ollama API."""

    def __init__(self):
        self.model = EMBEDDING_MODEL

    @property
    def api_url(self) -> str:
        from app.config import OLLAMA_HOST
        return f"{OLLAMA_HOST}/api/embed"

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats (embedding vector)
        """
        try:
            response = httpx.post(
                self.api_url,
                json={"model": self.model, "input": text},
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()

            embeddings = result.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            else:
                logger.error("Empty embedding response")
                return [0.0] * EMBEDDING_DIM

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return [0.0] * EMBEDDING_DIM

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        progress_callback=None,
    ) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts per API call
            progress_callback: Optional callback(progress_float, status_str)

        Returns:
            List of embedding vectors (same order as input)
        """
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            if progress_callback:
                progress_callback(
                    i / total,
                    f"Embedding: batch {batch_num}/{total_batches} ({i + len(batch)}/{total})"
                )

            try:
                response = httpx.post(
                    self.api_url,
                    json={"model": self.model, "input": batch},
                    timeout=120.0,
                )
                response.raise_for_status()
                result = response.json()

                embeddings = result.get("embeddings", [])
                if len(embeddings) == len(batch):
                    all_embeddings.extend(embeddings)
                else:
                    # Fallback: embed one by one
                    logger.warning(f"Batch returned {len(embeddings)} embeddings for {len(batch)} texts, falling back to single")
                    for text in batch:
                        all_embeddings.append(self.embed_text(text))

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}, falling back to single")
                for text in batch:
                    all_embeddings.append(self.embed_text(text))

        if progress_callback:
            progress_callback(1.0, f"Embedding completato: {len(all_embeddings)} vettori")

        logger.info(f"Embedded {len(all_embeddings)} texts")
        return all_embeddings

    def is_available(self) -> bool:
        """Check if embedding model is available on Ollama."""
        try:
            from app.config import OLLAMA_HOST
            r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return any(m == EMBEDDING_MODEL or m == f"{EMBEDDING_MODEL}:latest" or (":" not in EMBEDDING_MODEL and m.startswith(f"{EMBEDDING_MODEL}:")) for m in models)
        except Exception:
            return False


def chunk_id(filepath: str, chunk_index) -> str:
    """Generate deterministic chunk ID to prevent duplicates in ChromaDB."""
    raw = f"{filepath}::chunk::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
