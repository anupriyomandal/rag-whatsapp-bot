# RAG-Based WhatsApp Bot

A **Retrieval-Augmented Generation (RAG)** backend that answers user questions grounded in the content of a book (originally configured for *The Molecule of More*). The backend exposes a clean HTTP API designed to be called from a WhatsApp bot gateway or any other client.

---

## How It Works

```
User Question
      │
      ▼
┌─────────────────┐
│   FastAPI /ask  │  ← HTTP GET ?q=<question>
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OpenAI Embedder│  ← Converts question → float32 vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Retriever│  ← Finds top-K most similar book chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Answer Generator│  ← GPT-4o-mini, grounded by retrieved chunks
└────────┬────────┘
         │
         ▼
    JSON Answer
```

There are two phases:

### 1. Ingestion (offline, run once)
`ingest.py` reads a book file, splits it into overlapping text chunks, embeds them via the OpenAI Embeddings API, and saves a **FAISS flat L2 index** plus a **chunks pickle** to disk.

### 2. Serving (online, always running)
The FastAPI server loads the FAISS index and chunk metadata at startup into memory. On each `/ask` request, it embeds the question, retrieves the `top_k` most semantically similar chunks, and feeds them as context to a chat LLM to produce a grounded answer.

---

## Project Structure

```
rag-based-whatsapp-bot/
├── ingest.py                  # Offline pipeline: read → chunk → embed → FAISS index
├── api/
│   └── app.py                 # FastAPI app: /health and /ask endpoints
├── rag/
│   ├── pipeline.py            # Orchestrates embedder + retriever + generator
│   ├── embedder.py            # Wraps OpenAI Embeddings API for query vectors
│   ├── retriever.py           # FAISS index loader and semantic search
│   └── generator.py          # Prompt builder + GPT chat completion caller
├── config/
│   └── settings.py            # All runtime config loaded from environment variables
├── prompts/
│   └── book_prompt.txt        # System prompt template with {context} and {question} slots
├── data/
│   ├── book/                  # Drop your book file here (.txt, .md, or .pdf)
│   └── embeddings/
│       ├── molecule_index.faiss   # Built by ingest.py (FAISS index)
│       └── chunks.pkl             # Built by ingest.py (text chunk records)
├── Dockerfile                 # Production container (Python 3.11-slim + uvicorn)
└── requirements.txt           # Python dependencies
```

---

## Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A book file in `.txt`, `.md`, or `.pdf` format

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone git@github.com:anupriyomandal/rag-whatsapp-bot.git
cd rag-whatsapp-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables

Copy and fill in the required values:

```bash
export OPENAI_API_KEY=sk-...          # Required
export BACKEND_API_KEY=your-secret    # Required (used to protect the /ask endpoint)
```

Optional overrides (with defaults shown):

```bash
export EMBEDDING_MODEL=text-embedding-3-small
export LLM_MODEL=gpt-4o-mini
export TOP_K=5
export REQUIRE_API_KEY=true
export API_KEY_HEADER=X-API-Key
export FAISS_INDEX_PATH=data/embeddings/molecule_index.faiss
export CHUNKS_PATH=data/embeddings/chunks.pkl
export PROMPT_PATH=prompts/book_prompt.txt
```

### 3. Add your book

Place your book file inside `data/book/`. The ingestion script auto-discovers a single `.txt`, `.md`, or `.pdf` file in that directory:

```bash
cp ~/Downloads/my-book.pdf data/book/
```

### 4. Run ingestion

```bash
python3 ingest.py
```

This reads the book, splits it into overlapping chunks (default: 1200 chars with 200-char overlap), embeds them in batches of 64, and writes the FAISS index and chunks pickle to `data/embeddings/`.

**Advanced options:**

```bash
python3 ingest.py \
  --input data/book/my-book.pdf \
  --chunk-size 1000 \
  --chunk-overlap 150 \
  --batch-size 32 \
  --embedding-model text-embedding-3-small
```

---

## Running the Server

### Locally

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### With Docker

```bash
docker build -t rag-whatsapp-bot .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e BACKEND_API_KEY=your-secret \
  rag-whatsapp-bot
```

---

## API Reference

### `GET /health`

Lightweight liveness check. No authentication required.

**Response:**
```json
{ "status": "ok" }
```

### `GET /ask?q=<question>`

Answer a question grounded in the indexed book content.

**Headers:**
```
X-API-Key: your-secret
```

**Query Parameters:**
| Parameter | Type   | Description           |
|-----------|--------|-----------------------|
| `q`       | string | The user's question   |

**Response:**
```json
{ "answer": "Dopamine is a neurotransmitter associated with motivation and desire..." }
```

**Error Responses:**
| Status | Meaning                              |
|--------|--------------------------------------|
| 400    | Empty or invalid question            |
| 401    | Missing or invalid `X-API-Key`       |
| 503    | Server not yet initialized           |
| 500    | Unexpected error during generation   |

---

## Connecting to WhatsApp

This backend is designed to sit behind a WhatsApp messaging gateway (e.g. [Twilio WhatsApp API](https://www.twilio.com/whatsapp), [Meta Cloud API](https://developers.facebook.com/docs/whatsapp/cloud-api), or a self-hosted [whatsapp-web.js](https://github.com/pedroslopez/whatsapp-web.js) bridge).

Your WhatsApp webhook handler should:
1. Receive the incoming message text from the user.
2. Forward it as a `GET /ask?q=<message>` request to this server (with the `X-API-Key` header).
3. Send the `answer` field from the JSON response back to the user on WhatsApp.

---

## Customising the Prompt

Edit `prompts/book_prompt.txt` to change how the LLM responds. The template must contain two placeholders:

- `{context}` — injected retrieved book excerpts
- `{question}` — the user's original question

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| FAISS `IndexFlatL2` | Exact nearest-neighbour search; fast enough for book-sized corpora without approximate search |
| Overlapping chunks | 200-char overlap prevents answer context from being split across chunk boundaries |
| Load index at startup | Index is loaded once into memory; each request only pays for embedding + search |
| `gpt-4o-mini` default | Low cost per token while maintaining good instruction-following for RAG |
| Static API key auth | Simple and portable; no OAuth dependency for a single-user or internal deployment |
| `temperature=0.2` | Keeps answers factual and grounded; reduces hallucination in RAG setting |

---

## Dependencies

| Package      | Purpose                                  |
|--------------|------------------------------------------|
| `fastapi`    | HTTP API framework                       |
| `uvicorn`    | ASGI server                              |
| `openai`     | Embeddings + chat completions            |
| `faiss-cpu`  | Vector similarity search                 |
| `numpy`      | Embedding matrix operations              |
| `pydantic`   | Request/response validation              |
| `pypdf`      | PDF text extraction during ingestion     |

---

## License

MIT
