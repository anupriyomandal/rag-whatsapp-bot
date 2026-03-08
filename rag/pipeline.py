"""Orchestration layer for end-to-end RAG query handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from openai import OpenAI

from config.settings import Settings
from rag.embedder import OpenAIEmbedder
from rag.generator import AnswerGenerator
from rag.retriever import FaissRetriever


@dataclass
class RagPipeline:
    """Coordinates embedding, retrieval, and generation steps."""

    embedder: OpenAIEmbedder
    retriever: FaissRetriever
    generator: AnswerGenerator
    top_k: int = 5

    def ask(self, question: str) -> str:
        """Process a user question and return a grounded answer."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty")

        query_embedding = self.embedder.embed_query(normalized_question)
        retrieved = self.retriever.search(query_embedding=query_embedding, top_k=self.top_k)
        context_chunks: List[str] = [chunk.text for chunk in retrieved]

        if not context_chunks:
            context_chunks = [
                "No relevant excerpts were found in the indexed content. "
                "Answer cautiously and acknowledge uncertainty."
            ]

        return self.generator.generate(
            question=normalized_question,
            context_chunks=context_chunks,
        )


def create_pipeline(settings: Settings) -> RagPipeline:
    """Factory to initialize and wire all RAG components."""
    client = OpenAI(api_key=settings.openai_api_key)

    retriever = FaissRetriever(
        index_path=settings.faiss_index_path,
        chunks_path=settings.chunks_path,
    )
    retriever.load()

    embedder = OpenAIEmbedder(client=client, model=settings.embedding_model)
    generator = AnswerGenerator(
        client=client,
        model=settings.llm_model,
        prompt_path=settings.prompt_path,
    )

    return RagPipeline(
        embedder=embedder,
        retriever=retriever,
        generator=generator,
        top_k=settings.top_k,
    )
