"""Build FAISS index and chunk metadata from a book file.

Usage examples:
  python3 ingest.py
  python3 ingest.py --input /path/to/the_molecule_of_more.pdf
  python3 ingest.py --input /path/to/book.txt --chunk-size 1200 --chunk-overlap 200

Outputs:
  data/embeddings/molecule_index.faiss
  data/embeddings/chunks.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from openai import OpenAI

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

try:
    from pypdf import PdfReader
except Exception:  # noqa: BLE001
    PdfReader = None


@dataclass(frozen=True)
class IngestConfig:
    """Configuration for chunking and embedding workflow."""

    input_path: Path
    output_index_path: Path
    output_chunks_path: Path
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    batch_size: int = 64


def parse_args() -> IngestConfig:
    """Parse CLI arguments into an immutable config object."""
    parser = argparse.ArgumentParser(description="Ingest a book into FAISS for RAG")
    parser.add_argument(
        "--input",
        default="data/book",
        help="Path to source file (.txt/.md/.pdf) or directory containing exactly one source file",
    )
    parser.add_argument(
        "--index-out",
        default="data/embeddings/molecule_index.faiss",
        help="Path to output FAISS index",
    )
    parser.add_argument(
        "--chunks-out",
        default="data/embeddings/chunks.pkl",
        help="Path to output chunks pickle",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model",
    )
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap in characters between adjacent chunks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size",
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be >= 0")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be less than --chunk-size")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    return IngestConfig(
        input_path=resolve_input_path(Path(args.input)),
        output_index_path=Path(args.index_out),
        output_chunks_path=Path(args.chunks_out),
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )


def resolve_input_path(input_path: Path) -> Path:
    """Resolve an input file path from a file or a source directory."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported input type. Use .txt, .md, or .pdf")
        return input_path

    if input_path.is_dir():
        candidates = sorted(
            p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not candidates:
            raise FileNotFoundError(
                f"No supported files found in directory: {input_path}. "
                "Add one .txt, .md, or .pdf file."
            )
        if len(candidates) > 1:
            names = ", ".join(p.name for p in candidates)
            raise ValueError(
                f"Multiple supported files found in {input_path}: {names}. "
                "Keep one file in the folder or pass --input /path/to/file."
            )
        return candidates[0]

    raise ValueError(f"Unsupported input path: {input_path}")


def read_source_text(path: Path) -> str:
    """Read text content from txt/md/pdf files."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError(
                "PDF support requires pypdf. Install dependencies from requirements.txt."
            )
        reader = PdfReader(str(path))
        pages: List[str] = []
        for page in reader.pages:
            pages.append((page.extract_text() or "").strip())
        return "\n\n".join(pages)

    raise ValueError("Unsupported input type. Use .txt, .md, or .pdf")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]

    normalized_lines: List[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 1:
                normalized_lines.append("")
            continue
        blank_run = 0
        normalized_lines.append(" ".join(line.split()))

    return "\n".join(normalized_lines).strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split source text into overlapping character-based chunks."""
    if not text.strip():
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            split_at = text.rfind("\n\n", start, end)
            if split_at == -1:
                split_at = text.rfind(". ", start, end)
            if split_at != -1 and split_at > start + (chunk_size // 2):
                end = split_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = max(0, end - chunk_overlap)

    return chunks


def embed_chunks(client: OpenAI, chunks: List[str], model: str, batch_size: int) -> np.ndarray:
    """Embed chunks in batches and return float32 matrix [n_chunks, dim]."""
    vectors: List[List[float]] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)

        # API returns one embedding per input in-order.
        vectors.extend(item.embedding for item in response.data)
        print(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    return np.asarray(vectors, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS L2 index from embedding matrix."""
    if embeddings.ndim != 2:
        raise ValueError("Expected 2D embeddings matrix")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_outputs(index: faiss.Index, chunks: List[str], index_path: Path, chunks_path: Path) -> None:
    """Persist FAISS index and chunk metadata to disk."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    chunk_records = [{"id": i, "text": text} for i, text in enumerate(chunks)]
    with chunks_path.open("wb") as file:
        pickle.dump(chunk_records, file)


def main() -> None:
    """Run ingestion pipeline end-to-end."""
    config = parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")

    print(f"Reading source file: {config.input_path}")
    raw_text = read_source_text(config.input_path)
    text = normalize_text(raw_text)

    chunks = chunk_text(
        text=text,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    if not chunks:
        raise ValueError("No chunks generated from input file")

    print(f"Generated {len(chunks)} chunks")
    client = OpenAI(api_key=api_key)
    embeddings = embed_chunks(
        client=client,
        chunks=chunks,
        model=config.embedding_model,
        batch_size=config.batch_size,
    )

    index = build_faiss_index(embeddings)
    save_outputs(
        index=index,
        chunks=chunks,
        index_path=config.output_index_path,
        chunks_path=config.output_chunks_path,
    )

    print(f"Saved FAISS index to: {config.output_index_path}")
    print(f"Saved chunks to: {config.output_chunks_path}")


if __name__ == "__main__":
    main()
