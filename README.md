
# AI Training Repository (Multi-Session)

Hands-on training materials for practical AI workflows across multiple sessions.

This repository currently includes **Session 2** labs for:
- Chat completion with API calls
- Text RAG (Retrieval-Augmented Generation)
- Vision RAG (VLM + RAG)

---

## Repository Structure

```
elc/
	session2/
		chat_completion.sh
		rag/
			01_ingest.py
			02_embed.py
			03_query.py
			docs/
		vlm-rag/
			01_ingest.py
			02_embed.py
			03_query.py
			04_query_rerank.py
			docs/
```

---

## Prerequisites

### Required for all tracks
- Git
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package/environment manager)

### Required for local RAG / VLM-RAG tracks
- [Ollama](https://ollama.com/) running locally
- Ollama model(s):
	- `phi3:mini` (embeddings + local text model)
	- `qwen3.5:4b` (for VLM track)

### Required for VLM-RAG only
- Node.js + npm
- [liteparse CLI](https://github.com/run-llama/liteparse): `npm i -g @llamaindex/liteparse`

---

## Installation Guide

### 1) Install core tools

#### macOS (Homebrew)
```bash
brew update
brew install git python@3.11 uv ollama node
```

#### Windows 11 (winget, PowerShell)
```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
winget install --id Astral-sh.uv -e
winget install --id Ollama.Ollama -e
winget install --id OpenJS.NodeJS.LTS -e
```

### 2) Clone the repository
```bash
git clone <YOUR_REPO_URL>
cd training
```

### 3) Pull required Ollama models
```bash
ollama pull phi3:mini
ollama pull qwen3.5:4b
```

### 4) Install liteparse (VLM-RAG track)
```bash
npm i -g @llamaindex/liteparse
```

### 5) Sync Python dependencies per lab

For text RAG:
```bash
cd elc/session2/rag
uv sync
```

For VLM-RAG:
```bash
cd elc/session2/vlm-rag
uv sync
```

---

## Session 2 Labs

### A) Chat Completion API Script
Location: `elc/session2/chat_completion.sh`

1. Create a `.env` file in `elc/session2/`:
```bash
OPENAI_API_KEY=your_api_key_here
# Optional overrides:
# OPENAI_ENDPOINT=https://api.openai.com/v1/chat/completions
# OPENAI_MODEL=gpt-4.1
# OPENAI_SEED=42
# OPENAI_TEMPERATURE=0
```

2. Run:
```bash
cd elc/session2
chmod +x chat_completion.sh
./chat_completion.sh "Explain RAG in 3 bullet points"
```

---

### B) Local Text RAG (Ollama + SQLite)
Location: `elc/session2/rag`

1. Add a PDF to `elc/session2/rag/docs/`
2. Run pipeline:
```bash
cd elc/session2/rag
uv run python 01_ingest.py
uv run python 02_embed.py
uv run python 03_query.py "What is attention?"
```

---

### C) VLM-RAG (Vision + Retrieval)
Location: `elc/session2/vlm-rag`

1. Add a PDF to `elc/session2/vlm-rag/docs/`
2. Run pipeline:
```bash
cd elc/session2/vlm-rag
uv run python 01_ingest.py
uv run python 02_embed.py
uv run python 03_query.py "Summarize the key chart on unemployment ratio"
```

Optional reranking flow (if your lab uses it):
```bash
uv run python 04_query_rerank.py "Your question"
```

---

## Suggested Training Flow

1. Start with `chat_completion.sh` for API fundamentals.
2. Move to `rag/` for text-based retrieval.
3. Finish with `vlm-rag/` to handle visual-heavy documents (tables/charts/diagrams).

---

## Troubleshooting

- If `ollama` commands fail, start Ollama app/service first.
- If `uv` is not found, restart terminal after installation.
- If Python version is too old, ensure `python --version` is 3.11+.
- If `liteparse` is not found, verify npm global bin is on your PATH.
- If `chat_completion.sh` fails, confirm `.env` exists and `OPENAI_API_KEY` is set.

---

## Notes

- Each lab folder has its own focused README with deeper implementation details.
- Keep PDFs small during training runs for faster ingestion and query cycles.
