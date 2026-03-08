"""Embedding client for converting user queries into dense vectors."""

from __future__ import annotations

from typing import List

import numpy as np
from openai import OpenAI


class OpenAIEmbedder:
    """Wraps OpenAI embeddings API for query vector generation."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def embed_query(self, query: str) -> np.ndarray:
        """Return a float32 embedding vector for a user query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        response = self._client.embeddings.create(
            model=self._model,
            input=query.strip(),
        )
        vector: List[float] = response.data[0].embedding
        return np.asarray(vector, dtype=np.float32)
