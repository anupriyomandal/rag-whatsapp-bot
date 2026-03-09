"""FastAPI application exposing the WhatsApp RAG AI endpoint."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from config.settings import Settings
from rag.pipeline import RagPipeline, create_pipeline


class AskResponse(BaseModel):
    """API response model for /ask endpoint."""

    answer: str


app = FastAPI(
    title="Molecule of More - WhatsApp RAG Backend",
    version="1.0.0",
)

_pipeline: RagPipeline | None = None


@app.on_event("startup")
def startup_event() -> None:
    """Initialize pipeline once so FAISS index is loaded only at startup."""
    global _pipeline
    settings = Settings.from_env()
    _pipeline = create_pipeline(settings)


@app.get("/health")
def health() -> dict:
    """Lightweight health check for orchestration and deployment probes."""
    return {"status": "ok"}


@app.get("/ask", response_model=AskResponse)
def ask(q: str = Query(..., min_length=1, description="User question")) -> AskResponse:
    """Answer a user query using retrieval-augmented generation."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        answer = _pipeline.ask(q)
        return AskResponse(answer=answer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}") from exc
