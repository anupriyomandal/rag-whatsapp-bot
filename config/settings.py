"""Application settings loaded from environment variables.

This module centralizes runtime configuration for model names,
file paths, and API behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_bool(value: str, default: bool) -> bool:
    """Parse common truthy/falsey strings into a boolean."""
    normalized = (value or "").strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


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
    require_api_key: bool
    backend_api_key: str
    api_key_header: str

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables with production defaults."""
        project_root = Path(__file__).resolve().parents[1]

        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")

        require_api_key = _parse_bool(os.getenv("REQUIRE_API_KEY", "true"), default=True)
        backend_api_key = os.getenv("BACKEND_API_KEY", "").strip()
        if require_api_key and not backend_api_key:
            raise ValueError("BACKEND_API_KEY is required when REQUIRE_API_KEY=true")

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
            require_api_key=require_api_key,
            backend_api_key=backend_api_key,
            api_key_header=os.getenv("API_KEY_HEADER", "X-API-Key").strip() or "X-API-Key",
        )
