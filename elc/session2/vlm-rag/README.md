# VLM RAG — Vision Language Model RAG with Ollama + SQLite

Same RAG pattern, but using page **images** instead of extracted text for the answer step. This preserves tables, figures, and formatting that text extraction would lose.

## LLM RAG vs VLM RAG

| | LLM RAG (`rag/`) | VLM RAG (`vlm-rag/`) |
|---|---|---|
| **Chunk unit** | 500-char text snippets | Whole pages |
| **Text extraction** | `pymupdf` `get_text()` | `liteparse` (better spatial reading order) |
| **Stored in DB** | Plain text | Text + VLM description + page image (base64 PNG) |
| **Retrieval** | Text embedding search | Text + VLM description embedding search |
| **Answer generation** | LLM reads text context | VLM reads page images |
| **Model** | `phi3:mini` only | `phi3:mini` (embed) + `qwen3.5:4b` (vision) |

## Prerequisites

- [Ollama](https://ollama.com) running locally
- Models pulled:
  ```bash
  ollama pull phi3:mini      # for embeddings
  ollama pull qwen3.5:4b     # for vision answering
  ```
- [uv](https://docs.astral.sh/uv/) installed
- [liteparse](https://github.com/run-llama/liteparse) installed:
  ```bash
  npm i -g @llamaindex/liteparse
  ```

## How It Works

```
PDF ──→ [01_ingest] ──→ liteparse (text) + pymupdf (images) + qwen3.5 (description)
                             │
                        stores in SQLite: text + description + base64 image
                             │
        [02_embed] ──→ embed (text + description) ──→ SQLite (+ vectors)
                             │
         Question ──→ [03_query] ──→ vector search ──→ page images ──→ VLM answer
```

## Step by Step

### 1. Install dependencies

```bash
cd elc/session2/vlm-rag
uv sync
```

### 2. Add a PDF

Place a PDF in the `docs/` folder.

### 3. Ingest — PDF pages to SQLite (text + images + VLM descriptions)

```bash
uv run python 01_ingest.py
```

For each page:
- Extracts text via **liteparse** (better spatial reading order than pymupdf)
- Renders the page as a PNG image via **pymupdf**
- Asks **qwen3.5:4b** to describe the image — extracting diagram titles, chart descriptions, table summaries

All three are stored in `rag.db`. One row per page.

### 4. Embed — Create vectors from page text + VLM description

```bash
uv run python 02_embed.py
```

Embeds the **combined** extracted text + VLM description via Ollama `phi3:mini`. The VLM description enriches the embedding with info about figures, tables, and layout that raw text extraction misses.

### 5. Query — Ask a question (images sent to VLM)

```bash
uv run python 03_query.py "At which point does the vacancies-to-unemployment ratio break even?"
```

1. Embeds your question (using `phi3:mini`)
2. Finds the most similar pages via vector search
3. Sends the retrieved page **images** to `qwen3.5:4b` (VLM)
4. Prints the full prompt and the answer

## Stack

| Component | Tool |
|-----------|------|
| Embeddings | Ollama `phi3:mini` |
| Vision LLM | Ollama `qwen3.5:4b` |
| Text extraction | `liteparse` (spatial reading order) |
| Page rendering | `pymupdf` |
| Vector DB | `sqlite-vec` (SQLite extension) |
| HTTP | `requests` (no wrapper libraries) |
| Display | `rich` (pretty terminal tables) |
