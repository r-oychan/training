# Local RAG with Ollama + SQLite

A hands-on demo showing how RAG (Retrieval-Augmented Generation) works — just a local LLM, a PDF, and SQLite. No cloud services needed.

## Prerequisites

- [Ollama](https://ollama.com) running locally with `phi3:mini` pulled
- [uv](https://docs.astral.sh/uv/) installed

```bash
# Pull the model (if not already done)
ollama pull phi3:mini
```

## How It Works

```
PDF ──→ [01_ingest] ──→ SQLite (plain text chunks)
                             │
                        [02_embed] ──→ SQLite (+ embedding vectors)
                             │
         Question ──→ [03_query] ──→ vector search ──→ LLM answer
```

The database (`rag.db`) is the single source of truth throughout the pipeline. No intermediate JSON files.

## Step by Step

### 1. Install dependencies

```bash
cd elc/session2/rag
uv sync
```

### 2. Add a PDF

Place a PDF in the `docs/` folder. Or download the sample:

```bash
curl -L -o docs/attention.pdf https://arxiv.org/pdf/1706.03762
```

### 3. Ingest — PDF to chunks in SQLite

```bash
uv run python 01_ingest.py
```

Extracts text from the PDF page by page, splits it into 500-character overlapping chunks, and stores them in `rag.db`.

### 4. Embed — Create vectors and store in SQLite

```bash
uv run python 02_embed.py
```

Reads chunks from the database, calls Ollama `/api/embed` to get a vector for each chunk, and stores the vectors in a `sqlite-vec` virtual table — all in the same `rag.db`.

### 5. Query — Ask a question

```bash
uv run python 03_query.py "What is attention?"
```

Embeds your question, finds the most similar chunks via vector search, then sends them as context to the LLM to generate an answer.

## Stack

| Component | Tool |
|-----------|------|
| LLM + Embeddings | Ollama `phi3:mini` |
| PDF parsing | `pymupdf` |
| Vector DB | `sqlite-vec` (SQLite extension) |
| HTTP | `requests` (no wrapper libraries) |
| Display | `rich` (pretty terminal tables) |
