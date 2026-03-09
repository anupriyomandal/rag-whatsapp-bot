"""Application settings loaded from environment variables.

This module centralizes runtime configuration for model names,
file paths, and API behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the RAG backend."""

    openai_api_key: str
    embedding_model: str
    llm_model: str
    faiss_index_path: Path
    chunks_path: Path
    prompt_path: Path
    top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables with production defaults."""
        project_root = Path(__file__).resolve().parents[1]

        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")

        return cls(
            openai_api_key=openai_api_key,
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            faiss_index_path=Path(
                os.getenv(
                    "FAISS_INDEX_PATH",
                    str(project_root / "data" / "embeddings" / "molecule_index.faiss"),
                )
            ),
            chunks_path=Path(
                os.getenv(
                    "CHUNKS_PATH",
                    str(project_root / "data" / "embeddings" / "chunks.pkl"),
                )
            ),
            prompt_path=Path(
                os.getenv(
                    "PROMPT_PATH",
                    str(project_root / "prompts" / "book_prompt.txt"),
                )
            ),
            top_k=int(os.getenv("TOP_K", "5")),
        )
