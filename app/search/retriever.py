"""
Retriever — Handles the query pipeline: embed question → search → rerank → return.
"""

import logging

from app.config import TOP_K, SIMILARITY_THRESHOLD
from app.indexer.embedder import Embedder
from app.search.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Query pipeline: embed → search → filter → format."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: float = SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: User's natural language question
            top_k: Max number of chunks to return
            min_score: Minimum similarity score threshold

        Returns:
            List of dicts with: text, metadata, score — sorted by score descending
        """
        # 1. Embed the query
        query_embedding = self.embedder.embed_text(query)

        # 2. Search vector store
        results = self.store.search(query_embedding, top_k=top_k * 2)  # Get more, filter later

        # 3. Filter by minimum score
        filtered = [r for r in results if r["score"] >= min_score]

        # 4. Deduplicate: if multiple chunks from same file are very similar, keep best
        deduped = self._deduplicate(filtered)

        # 5. Return top_k
        return deduped[:top_k]

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        """
        Remove near-duplicate chunks from same file.
        Keep the highest-scoring chunk per (filepath, page) combination,
        but allow multiple chunks from same file if they are from different pages.
        """
        seen = {}  # (filepath, page) -> best result
        unique = []

        for r in results:
            key = (r["metadata"].get("filepath", ""), r["metadata"].get("page", 0))

            if key not in seen:
                seen[key] = r
                unique.append(r)
            elif r["score"] > seen[key]["score"]:
                # Replace with higher score
                unique = [x for x in unique if x is not seen[key]]
                seen[key] = r
                unique.append(r)

        # Re-sort by score
        unique.sort(key=lambda x: x["score"], reverse=True)
        return unique

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks as context for the LLM prompt.

        Returns:
            Formatted string with source citations.
        """
        if not results:
            return "(Nessun documento rilevante trovato)"

        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            filename = meta.get("filename", "sconosciuto")
            page = meta.get("page", "")
            page_str = f", pag. {page}" if page else ""
            score = r["score"]

            parts.append(
                f"[Fonte {i}: {filename}{page_str} — rilevanza: {score:.0%}]\n"
                f"{r['text']}"
            )

        return "\n\n---\n\n".join(parts)

    def format_sources(self, results: list[dict]) -> list[dict]:
        """
        Format sources for display in the UI.

        Returns:
            List of dicts with: filename, page, score, preview
        """
        sources = []
        for r in results:
            meta = r["metadata"]
            sources.append({
                "filename": meta.get("filename", "sconosciuto"),
                "filepath": meta.get("filepath", ""),
                "page": meta.get("page", ""),
                "score": r["score"],
                "preview": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                "extraction": meta.get("extraction", "native"),
            })
        return sources
