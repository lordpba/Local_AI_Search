"""
VectorStore — ChromaDB + BM25 hybrid index with anti-duplicate protection.
Handles collection management, incremental updates, keyword + semantic search.
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.config import CHROMA_DIR, MANIFEST_FILE, EMBEDDING_DIM, BM25_INDEX_FILE
from app.indexer.embedder import chunk_id

logger = logging.getLogger(__name__)


# ─── BM25 keyword index ──────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25: lowercase, split on non-alphanum, drop short."""
    tokens = re.findall(r"[a-zA-Z0-9àèéìòùÀÈÉÌÒÙ]{2,}", text.lower())
    return tokens


class BM25Index:
    """Persistent BM25 keyword index that lives alongside ChromaDB."""

    def __init__(self, path: Path = BM25_INDEX_FILE):
        self.path = path
        # Internal storage: parallel lists
        self.doc_ids: list[str] = []          # chunk IDs
        self.doc_texts: list[str] = []        # raw text per chunk
        self.doc_metadatas: list[dict] = []   # metadata per chunk
        self.tokenized: list[list[str]] = []  # tokenized text per chunk
        self.bm25: Optional[BM25Okapi] = None
        self._load()

    # ─── Persistence ──────────────────────────────────────────

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    data = pickle.load(f)
                self.doc_ids = data["ids"]
                self.doc_texts = data["texts"]
                self.doc_metadatas = data["metadatas"]
                self.tokenized = data["tokenized"]
                self._rebuild_bm25()
                logger.info(f"BM25 index loaded: {len(self.doc_ids)} chunks")
            except Exception as e:
                logger.warning(f"BM25 load failed, rebuilding: {e}")
                self._reset()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({
                "ids": self.doc_ids,
                "texts": self.doc_texts,
                "metadatas": self.doc_metadatas,
                "tokenized": self.tokenized,
            }, f)

    def _rebuild_bm25(self):
        if self.tokenized:
            self.bm25 = BM25Okapi(self.tokenized)
        else:
            self.bm25 = None

    def _reset(self):
        self.doc_ids = []
        self.doc_texts = []
        self.doc_metadatas = []
        self.tokenized = []
        self.bm25 = None

    # ─── Add / Remove ─────────────────────────────────────────

    def add_chunks(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        """Add chunks to the BM25 index."""
        for cid, text, meta in zip(ids, texts, metadatas):
            self.doc_ids.append(cid)
            self.doc_texts.append(text)
            self.doc_metadatas.append(meta)
            self.tokenized.append(_tokenize(text))
        self._rebuild_bm25()
        self._save()

    def remove_by_ids(self, ids_to_remove: set[str]):
        """Remove chunks by their IDs."""
        if not ids_to_remove:
            return
        keep = [(i, cid) for i, cid in enumerate(self.doc_ids) if cid not in ids_to_remove]
        if not keep:
            self._reset()
            self._save()
            return
        indices = [i for i, _ in keep]
        self.doc_ids = [self.doc_ids[i] for i in indices]
        self.doc_texts = [self.doc_texts[i] for i in indices]
        self.doc_metadatas = [self.doc_metadatas[i] for i in indices]
        self.tokenized = [self.tokenized[i] for i in indices]
        self._rebuild_bm25()
        self._save()

    def clear(self):
        self._reset()
        self._save()

    # ─── Search ───────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        BM25 keyword search.

        Returns:
            List of dicts: {id, text, metadata, score} sorted by score desc
        """
        if not self.bm25 or not self.doc_ids:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Get top results
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scored[:top_k]:
            if score > 0:
                results.append({
                    "id": self.doc_ids[idx],
                    "text": self.doc_texts[idx],
                    "metadata": self.doc_metadatas[idx],
                    "score": float(score),
                })
        return results

    def get_all_chunks(self) -> list[dict]:
        """Return ALL indexed chunks (for map-reduce aggregation)."""
        return [
            {"id": cid, "text": text, "metadata": meta}
            for cid, text, meta in zip(self.doc_ids, self.doc_texts, self.doc_metadatas)
        ]

    @property
    def count(self) -> int:
        return len(self.doc_ids)


class VectorStore:
    """ChromaDB + BM25 hybrid vector store with manifest tracking."""

    COLLECTION_NAME = "private_search"

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25 = BM25Index()
        self.manifest = self._load_manifest()

    # ─── Manifest management ─────────────────────────────────────────────

    def _load_manifest(self) -> dict:
        """Load index manifest from disk."""
        if MANIFEST_FILE.exists():
            try:
                with open(MANIFEST_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, TypeError):
                pass
        return {"files": {}}

    def _save_manifest(self):
        """Save index manifest to disk."""
        MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_FILE, "w") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    # ─── Change detection ─────────────────────────────────────────────────

    def detect_changes(self, file_inventory: list[dict]) -> dict:
        """
        Compare current file inventory with manifest to detect changes.

        Returns:
            {
                "new": [file_info, ...],
                "modified": [file_info, ...],
                "deleted": [filepath, ...],
                "unchanged": [file_info, ...],
            }
        """
        indexed_files = self.manifest.get("files", {})
        current_paths = {f["path"] for f in file_inventory}
        indexed_paths = set(indexed_files.keys())

        changes = {"new": [], "modified": [], "deleted": [], "unchanged": []}

        for file_info in file_inventory:
            path = file_info["path"]
            if path not in indexed_paths:
                changes["new"].append(file_info)
            elif indexed_files[path]["hash"] != file_info["hash"]:
                changes["modified"].append(file_info)
            else:
                changes["unchanged"].append(file_info)

        # Files in manifest but no longer on disk
        for path in indexed_paths - current_paths:
            changes["deleted"].append(path)

        return changes

    def has_changes(self, file_inventory: list[dict]) -> bool:
        """Quick check if there are any changes."""
        changes = self.detect_changes(file_inventory)
        return bool(changes["new"] or changes["modified"] or changes["deleted"])

    # ─── Add / Remove / Update ────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list,
        embeddings: list[list[float]],
        file_info: dict,
    ):
        """
        Add chunks for a single file to the collection.
        Uses deterministic IDs to prevent duplicates.

        Args:
            chunks: List of TextChunk objects
            embeddings: Corresponding embedding vectors
            file_info: File info dict (path, hash, name, etc.)
        """
        if not chunks or not embeddings:
            return

        ids = []
        documents = []
        metadatas = []
        seen_ids = set()

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Use enumeration index i as fallback to guarantee uniqueness
            cid = chunk_id(file_info["path"], chunk.metadata.get("chunk_index", i))
            # Safety: if duplicate ID detected, append suffix to make it unique
            if cid in seen_ids:
                cid = chunk_id(file_info["path"], f"{chunk.metadata.get('chunk_index', i)}_{i}")
            seen_ids.add(cid)
            ids.append(cid)
            documents.append(chunk.text)
            meta = {
                "filepath": chunk.metadata.get("filepath", file_info["path"]),
                "filename": chunk.metadata.get("filename", file_info["name"]),
                "file_type": chunk.metadata.get("file_type", ""),
                "page": chunk.metadata.get("page", 0),
                "chunk_index": chunk.metadata.get("chunk_index", i),
                "extraction": chunk.metadata.get("extraction", "native"),
            }
            # Include extracted structured metadata (codici_fiscali, iban, dates)
            for extra_key in ("codici_fiscali", "iban", "dates"):
                if extra_key in chunk.metadata:
                    meta[extra_key] = chunk.metadata[extra_key]
            metadatas.append(meta)

        # Upsert to handle any edge cases with existing IDs
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # Also index in BM25 for keyword search
        self.bm25.add_chunks(ids, documents, metadatas)

        # Update manifest
        self.manifest["files"][file_info["path"]] = {
            "hash": file_info["hash"],
            "name": file_info["name"],
            "chunk_count": len(ids),
            "chunk_ids": ids,
        }
        self._save_manifest()

        logger.info(f"Added {len(ids)} chunks for {file_info['name']}")

    def remove_file(self, filepath: str):
        """
        Remove all chunks for a file from both ChromaDB and BM25.
        Uses manifest to know exactly which IDs to delete.
        """
        file_entry = self.manifest.get("files", {}).get(filepath)
        if file_entry and file_entry.get("chunk_ids"):
            chunk_ids = file_entry["chunk_ids"]
            try:
                self.collection.delete(ids=chunk_ids)
                logger.info(f"Removed {len(chunk_ids)} chunks from ChromaDB for {filepath}")
            except Exception as e:
                logger.warning(f"Error removing chunks by ID for {filepath}: {e}")
                try:
                    self.collection.delete(where={"filepath": filepath})
                except Exception as e2:
                    logger.error(f"Fallback delete also failed for {filepath}: {e2}")

            # Also remove from BM25
            self.bm25.remove_by_ids(set(chunk_ids))

        # Remove from manifest
        self.manifest.get("files", {}).pop(filepath, None)
        self._save_manifest()

    def update_file(
        self,
        chunks: list,
        embeddings: list[list[float]],
        file_info: dict,
    ):
        """Remove old chunks then add new ones — atomic file update."""
        self.remove_file(file_info["path"])
        self.add_chunks(chunks, embeddings, file_info)

    # ─── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search in the vector store.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            where: Optional ChromaDB metadata filter

        Returns:
            List of dicts with: text, metadata, score
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        items = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity: 1 - (distance / 2)
                similarity = 1.0 - (dist / 2.0)
                items.append({
                    "text": doc,
                    "metadata": meta,
                    "score": round(similarity, 4),
                })

        return items

    # ─── Utilities ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get collection statistics."""
        count = self.collection.count()
        file_count = len(self.manifest.get("files", {}))
        return {
            "total_chunks": count,
            "total_files": file_count,
            "files": list(self.manifest.get("files", {}).keys()),
        }

    def clear(self):
        """Delete entire collection, BM25 index, and manifest."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25.clear()
        self.manifest = {"files": {}}
        self._save_manifest()
        logger.info("Vector store + BM25 cleared")

    def keyword_search(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 keyword search. Returns [{text, metadata, score}, ...]."""
        bm25_results = self.bm25.search(query, top_k=top_k)
        return [
            {"text": r["text"], "metadata": r["metadata"], "score": r["score"]}
            for r in bm25_results
        ]

    def get_all_chunks(self) -> list[dict]:
        """Return ALL indexed chunks (for map-reduce). Uses BM25 store."""
        return self.bm25.get_all_chunks()

    @property
    def is_empty(self) -> bool:
        return self.collection.count() == 0
