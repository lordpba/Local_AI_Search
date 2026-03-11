"""
VectorStore — ChromaDB wrapper with anti-duplicate protection.
Handles collection management, incremental updates, and search.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import CHROMA_DIR, MANIFEST_FILE, EMBEDDING_DIM
from app.indexer.embedder import chunk_id

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store with manifest tracking."""

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
            metadatas.append({
                "filepath": chunk.metadata.get("filepath", file_info["path"]),
                "filename": chunk.metadata.get("filename", file_info["name"]),
                "file_type": chunk.metadata.get("file_type", ""),
                "page": chunk.metadata.get("page", 0),
                "chunk_index": chunk.metadata.get("chunk_index", i),
                "extraction": chunk.metadata.get("extraction", "native"),
            })

        # Upsert to handle any edge cases with existing IDs
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

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
        Remove all chunks for a file from the collection.
        Uses manifest to know exactly which IDs to delete.
        """
        file_entry = self.manifest.get("files", {}).get(filepath)
        if file_entry and file_entry.get("chunk_ids"):
            try:
                self.collection.delete(ids=file_entry["chunk_ids"])
                logger.info(f"Removed {len(file_entry['chunk_ids'])} chunks for {filepath}")
            except Exception as e:
                logger.warning(f"Error removing chunks by ID for {filepath}: {e}")
                # Fallback: delete by metadata filter
                try:
                    self.collection.delete(where={"filepath": filepath})
                except Exception as e2:
                    logger.error(f"Fallback delete also failed for {filepath}: {e2}")

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
        """Delete entire collection and manifest."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.manifest = {"files": {}}
        self._save_manifest()
        logger.info("Vector store cleared")

    @property
    def is_empty(self) -> bool:
        return self.collection.count() == 0
