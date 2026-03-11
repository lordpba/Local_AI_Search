"""
Retriever — Hybrid search pipeline: semantic (ChromaDB) + keyword (BM25) + RRF fusion.
"""

import logging
import re
from collections import defaultdict

from app.config import TOP_K, TOP_K_AGGREGATION, SIMILARITY_THRESHOLD, RRF_K
from app.indexer.embedder import Embedder
from app.search.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ─── Aggregation intent detection ─────────────────────────────────────────────

_AGGREGATION_PATTERNS = [
    r"\btutti\b", r"\btutto\b", r"\belenca\b", r"\blista\b", r"\blistare\b",
    r"\bogni\b", r"\bciascun[oa]?\b", r"\bquanti\b", r"\bquante\b",
    r"\bcompleto\b", r"\bcompleta\b", r"\briepilogo\b", r"\bsommario\b",
    r"\btotale\b", r"\btutte\b", r"\bognuno\b", r"\braccolta\b",
    r"\ball\b", r"\bevery\b", r"\blist\b", r"\beach\b",
]
_AGGREGATION_RE = re.compile("|".join(_AGGREGATION_PATTERNS), re.IGNORECASE)

# Patterns that suggest structured data extraction
_EXTRACTION_PATTERNS = [
    r"codic[ei]\s*fiscal[ei]", r"\biban\b", r"\bcf\b", r"\bp\.?\s*iva\b",
    r"partita\s*iva", r"\bnome\b.*\bcognome\b", r"\bindirizzo\b",
    r"data\s*di\s*nascita", r"\bcomun[ei]\b", r"\bprovincia\b",
]
_EXTRACTION_RE = re.compile("|".join(_EXTRACTION_PATTERNS), re.IGNORECASE)


def detect_query_type(query: str) -> str:
    """
    Classify query intent.

    Returns:
        'aggregation' — user wants data from ALL documents (e.g. "elenca tutti i comuni")
        'extraction'  — user wants specific structured data (e.g. "codice fiscale")
        'normal'      — standard Q&A
    """
    has_agg = bool(_AGGREGATION_RE.search(query))
    has_ext = bool(_EXTRACTION_RE.search(query))

    if has_agg:
        return "aggregation"
    if has_ext:
        return "extraction"
    return "normal"


class Retriever:
    """Hybrid query pipeline: embed → semantic search + keyword search → RRF fusion → return."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float = SIMILARITY_THRESHOLD,
        query_type: str | None = None,
    ) -> list[dict]:
        """
        Hybrid retrieve: BM25 keyword + semantic search with RRF fusion.

        Args:
            query: User's natural language question
            top_k: Max chunks to return (auto-selected if None)
            min_score: Minimum similarity threshold for semantic results
            query_type: Override auto-detected query type

        Returns:
            List of dicts with: text, metadata, score — sorted by fused score desc
        """
        if query_type is None:
            query_type = detect_query_type(query)

        if top_k is None:
            top_k = TOP_K_AGGREGATION if query_type == "aggregation" else TOP_K

        # How many candidates to fetch from each engine
        fetch_k = top_k * 3

        # 1. Semantic search (ChromaDB)
        query_embedding = self.embedder.embed_text(query)
        semantic_results = self.store.search(query_embedding, top_k=fetch_k)

        # 2. Keyword search (BM25)
        bm25_results = self.store.keyword_search(query, top_k=fetch_k)

        # 3. Fuse with Reciprocal Rank Fusion
        fused = self._rrf_fuse(semantic_results, bm25_results, min_score=min_score)

        logger.info(
            f"Hybrid retrieve: query_type={query_type}, "
            f"semantic={len(semantic_results)}, bm25={len(bm25_results)}, "
            f"fused={len(fused)}, returning top {top_k}"
        )

        return fused[:top_k]

    def retrieve_all_chunks(self) -> list[dict]:
        """
        Retrieve ALL indexed chunks (for map-reduce aggregation).
        Groups them by file.

        Returns:
            Dict mapping filepath → list of chunk dicts
        """
        all_chunks = self.store.get_all_chunks()
        by_file = defaultdict(list)
        for chunk in all_chunks:
            fp = chunk.get("metadata", {}).get("filepath", "unknown")
            by_file[fp].append(chunk)
        return dict(by_file)

    def _rrf_fuse(
        self,
        semantic_results: list[dict],
        bm25_results: list[dict],
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion of semantic and BM25 results.

        RRF score = 1/(k + rank_semantic) + 1/(k + rank_bm25)
        """
        k = RRF_K

        # Build unique key → result mapping using (text snippet)
        result_map = {}  # key → result dict
        rrf_scores = defaultdict(float)

        # Process semantic results
        for rank, r in enumerate(semantic_results):
            if r["score"] < min_score:
                continue
            key = self._result_key(r)
            rrf_scores[key] += 1.0 / (k + rank + 1)
            if key not in result_map:
                result_map[key] = r

        # Process BM25 results
        for rank, r in enumerate(bm25_results):
            key = self._result_key(r)
            rrf_scores[key] += 1.0 / (k + rank + 1)
            if key not in result_map:
                # BM25 results don't have a similarity score; set a baseline
                result_map[key] = {**r, "score": 0.5}

        # Sort by fused score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        fused = []
        for key in sorted_keys:
            result = result_map[key].copy()
            result["rrf_score"] = round(rrf_scores[key], 6)
            fused.append(result)

        return fused

    @staticmethod
    def _result_key(r: dict) -> str:
        """Generate a unique key for a result based on filepath + chunk_index."""
        meta = r.get("metadata", {})
        return f"{meta.get('filepath', '')}::{meta.get('chunk_index', '')}::{meta.get('page', '')}"

    def format_context(self, results: list[dict]) -> str:
        """Format retrieved chunks as context for the LLM prompt."""
        if not results:
            return "(Nessun documento rilevante trovato)"

        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            filename = meta.get("filename", "sconosciuto")
            page = meta.get("page", "")
            page_str = f", pag. {page}" if page else ""
            score = r.get("score", 0)

            parts.append(
                f"[Fonte {i}: {filename}{page_str} — rilevanza: {score:.0%}]\n"
                f"{r['text']}"
            )

        return "\n\n---\n\n".join(parts)

    def format_sources(self, results: list[dict]) -> list[dict]:
        """Format sources for display in the UI."""
        sources = []
        for r in results:
            meta = r["metadata"]
            sources.append({
                "filename": meta.get("filename", "sconosciuto"),
                "filepath": meta.get("filepath", ""),
                "page": meta.get("page", ""),
                "score": r.get("score", 0),
                "preview": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                "extraction": meta.get("extraction", "native"),
            })
        return sources
