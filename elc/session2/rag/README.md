# Local RAG with Ollama + SQLite

A hands-on demo showing how RAG (Retrieval-Augmented Generation) works — just an LLM, a PDF, and SQLite.

Supports **local** (Ollama) or **cloud** (Azure AI Foundry) as the LLM/embedding provider.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- **Local mode**: [Ollama](https://ollama.com) running with `phi3:mini` pulled
- **Azure mode**: An Azure AI Foundry project with chat + embedding deployments

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

### 2. Configure provider

Copy the sample env file and fill in your settings:

```bash
cp .env.sample .env
```

**For local (Ollama)** — just pull the model, defaults work out of the box:

```bash
ollama pull phi3:mini
```

**For Azure AI Foundry** — edit `.env` and set:

```env
PROVIDER=azure
AZURE_API_KEY=<your-azure-api-key>
AZURE_CHAT_BASE_URL=https://<resource-name>.openai.azure.com/openai/v1
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_EMBED_BASE_URL=https://<resource-name>.openai.azure.com/openai/v1
AZURE_EMBED_DEPLOYMENT=text-embedding-ada-002
```

### 3. Add a PDF

Place a PDF in the `docs/` folder. Or download the sample:

```bash
curl -L -o docs/attention.pdf https://arxiv.org/pdf/1706.03762
```

### 4. Ingest — PDF to chunks in SQLite

```bash
uv run python 01_ingest.py
```

Extracts text from the PDF page by page, splits it into 500-character overlapping chunks, and stores them in `rag.db`.

### 5. Embed — Create vectors and store in SQLite

```bash
uv run python 02_embed.py
```

Reads chunks from the database, embeds each one, and stores the vectors in a `sqlite-vec` virtual table — all in the same `rag.db`.

### 6. Query — Ask a question

```bash
uv run python 03_query.py "What is attention?"
```

Embeds your question, finds the most similar chunks via vector search, then sends them as context to the LLM to generate an answer.

## Stack

| Component | Local | Azure |
|-----------|-------|-------|
| LLM | Ollama `phi3:mini` | Azure `gpt-4o` / `gpt-5.2` |
| Embeddings | Ollama `phi3:mini` | Azure `text-embedding-ada-002` |
| PDF parsing | `pymupdf` | `pymupdf` |
| Vector DB | `sqlite-vec` | `sqlite-vec` |
| HTTP | `requests` | `requests` |
| Display | `rich` | `rich` |
