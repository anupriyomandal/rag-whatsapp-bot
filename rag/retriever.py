"""FAISS retriever for semantic search over precomputed book chunks."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a retrieved chunk plus retrieval metadata."""

    chunk_id: int
    text: str
    score: float


class FaissRetriever:
    """Loads an on-disk FAISS index and chunk metadata for retrieval."""

    def __init__(self, index_path: Path, chunks_path: Path) -> None:
        self._index_path = index_path
        self._chunks_path = chunks_path
        self._index = None
        self._chunks: List[Any] = []

    def load(self) -> None:
        """Load FAISS index and chunks from disk once at startup."""
        if not self._index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {self._index_path}")
        if not self._chunks_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {self._chunks_path}")

        self._index = faiss.read_index(str(self._index_path))
        with self._chunks_path.open("rb") as file:
            self._chunks = pickle.load(file)

        if not isinstance(self._chunks, list):
            raise ValueError("chunks.pkl must contain a list")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]:
        """Run vector similarity search and return top-k text chunks."""
        if self._index is None:
            raise RuntimeError("Retriever not initialized. Call load() before search().")

        if query_embedding.ndim != 1:
            raise ValueError("Expected a 1D embedding vector")

        query_vector = np.asarray([query_embedding], dtype=np.float32)
        distances, indices = self._index.search(query_vector, top_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue

            text = self._extract_chunk_text(self._chunks[idx])
            if text:
                results.append(RetrievedChunk(chunk_id=int(idx), text=text, score=float(score)))

        return results

    @staticmethod
    def _extract_chunk_text(raw_chunk: Any) -> str:
        """Normalize chunk representations into plain text for prompting."""
        if isinstance(raw_chunk, str):
            return raw_chunk.strip()
        if isinstance(raw_chunk, dict):
            for key in ("text", "chunk", "content"):
                value = raw_chunk.get(key)
                if isinstance(value, str):
                    return value.strip()
        return ""
