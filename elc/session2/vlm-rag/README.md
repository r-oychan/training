# VLM RAG — Vision Language Model RAG with Ollama + SQLite

Same RAG pattern, but using page **images** instead of extracted text for the answer step. This preserves tables, figures, and formatting that text extraction would lose.

Supports **local** (Ollama) or **cloud** (Azure AI Foundry) as the LLM/embedding provider.

## LLM RAG vs VLM RAG

| | LLM RAG (`rag/`) | VLM RAG (`vlm-rag/`) |
|---|---|---|
| **Chunk unit** | 500-char text snippets | Whole pages |
| **Text extraction** | `pymupdf` `get_text()` | `liteparse` (better spatial reading order) |
| **Stored in DB** | Plain text | Text + VLM description + page image (base64 PNG) |
| **Retrieval** | Text embedding search | Text + VLM description embedding search |
| **Answer generation** | LLM reads text context | VLM reads page images |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- [liteparse](https://github.com/run-llama/liteparse) installed: `npm i -g @llamaindex/liteparse`
- **Local mode**: [Ollama](https://ollama.com) running with models pulled
- **Azure mode**: An Azure AI Foundry project with chat, VLM, and embedding deployments

## How It Works

```
PDF ──→ [01_ingest] ──→ liteparse (text) + pymupdf (images) + VLM (description)
                             │
                        stores in SQLite: text + description + base64 image
                             │
        [02_embed] ──→ embed (text + description separately) ──→ SQLite (+ vectors)
                             │
         Question ──→ [03_query] ──→ vector search ──→ page images ──→ VLM answer
                      [04_query_rerank] ──→ + reranking step before VLM
```

## Step by Step

### 1. Install dependencies

```bash
cd elc/session2/vlm-rag
uv sync
```

### 2. Configure provider

Copy the sample env file and fill in your settings:

```bash
cp .env.sample .env
```

**For local (Ollama)** — pull the required models:

```bash
ollama pull phi3:mini      # embeddings + reranking
ollama pull qwen3.5:4b     # vision (page description + answering)
```

**For Azure AI Foundry** — edit `.env` and set:

```env
PROVIDER=azure
AZURE_API_KEY=<your-azure-api-key>

# Chat (text-only, used for reranking)
AZURE_CHAT_BASE_URL=https://<resource-name>.openai.azure.com/openai/v1
AZURE_CHAT_DEPLOYMENT=gpt-4o

# VLM (vision-capable, used for page description + answering)
AZURE_VLM_BASE_URL=https://<resource-name>.openai.azure.com/openai/v1
AZURE_VLM_DEPLOYMENT=gpt-4o

# Embeddings
AZURE_EMBED_BASE_URL=https://<resource-name>.openai.azure.com/openai/v1
AZURE_EMBED_DEPLOYMENT=text-embedding-ada-002
```

### 3. Add a PDF

Place a PDF in the `docs/` folder.

### 4. Ingest — PDF pages to SQLite (text + images + VLM descriptions)

```bash
uv run python 01_ingest.py
```

For each page:
- Extracts text via **liteparse** (better spatial reading order than pymupdf)
- Renders the page as a PNG image via **pymupdf**
- Asks the **VLM** to describe the image — extracting diagram titles, chart descriptions, table summaries

All three are stored in `rag.db`. One row per page.

### 5. Embed — Create vectors from page text + VLM description

```bash
uv run python 02_embed.py
```

Creates **two embeddings per page** (text and description separately) so chart titles and figure labels get their own focused vectors for better retrieval.

### 6. Query — Ask a question (images sent to VLM)

```bash
uv run python 03_query.py "At which point does the vacancies-to-unemployment ratio break even?"
```

1. Embeds your question
2. Finds the most similar pages via vector search
3. Sends the retrieved page **images** + descriptions to the VLM
4. Prints the full prompt and the answer

### 7. Query with reranking (optional)

```bash
uv run python 04_query_rerank.py "At which point does the vacancies-to-unemployment ratio break even?"
```

Same as step 6 but adds a **reranking** step: retrieves top 6 candidates, scores each for relevance using the LLM, then keeps the top 3. Shows before/after comparison.

## Stack

| Component | Local | Azure |
|-----------|-------|-------|
| Embeddings | Ollama `phi3:mini` | Azure `text-embedding-ada-002` |
| Chat / Rerank | Ollama `phi3:mini` | Azure `gpt-4o` / `gpt-5.2` |
| Vision LLM | Ollama `qwen3.5:4b` | Azure `gpt-4o` / `gpt-5.2` |
| Text extraction | `liteparse` | `liteparse` |
| Page rendering | `pymupdf` | `pymupdf` |
| Vector DB | `sqlite-vec` | `sqlite-vec` |
| HTTP | `requests` | `requests` |
| Display | `rich` | `rich` |
