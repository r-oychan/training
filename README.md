
# AI Training Repository (Multi-Session)

Hands-on training materials for practical AI workflows across multiple sessions.

This repository includes **Session 2** labs for:
- Chat completion with API calls (seed-based reproducibility)
- Text RAG (Retrieval-Augmented Generation)
- Vision RAG (VLM + RAG)
- **Interactive Jupyter notebooks** with Gradio Q&A interface

Everything is designed to **run locally** using [Ollama](https://ollama.com/) — no cloud API keys required. Azure AI Foundry is supported as an optional alternative.

---

## Repository Structure

```
elc/
    session2/
        chat_completion.sh          # Shell script — API call demo
        chat_completion.ps1         # PowerShell equivalent
        rag/                        # Text RAG (Python scripts)
            01_ingest.py
            02_embed.py
            03_query.py
            docs/                   # Place PDFs here
        vlm-rag/                    # Vision RAG (Python scripts)
            01_ingest.py
            02_embed.py
            03_query.py
            04_query_rerank.py
            docs/                   # Place PDFs here
        notebooks/                  # Interactive Jupyter notebooks
            01_chat_completion_seed.ipynb
            02_rag_pipeline.ipynb        # includes Gradio Q&A
            03_vlm_rag_pipeline.ipynb    # includes Gradio Q&A
deploy/                             # Docker + Azure deployment
    Dockerfile
    docker-compose.yml
    requirements.txt
    pulumi/                         # Infrastructure as Code
        __main__.py
        Pulumi.yaml
        Pulumi.dev.yaml
        requirements.txt
.github/
    workflows/
        deploy.yml                  # CI/CD pipeline
```

---

## Quick Start (Run Locally)

### 1) Install prerequisites

| Tool | macOS (Homebrew) | Windows (winget) |
|------|-----------------|-------------------|
| Git | `brew install git` | `winget install Git.Git` |
| Python 3.11+ | `brew install python@3.11` | `winget install Python.Python.3.11` |
| uv | `brew install uv` | `winget install Astral-sh.uv` |
| Ollama | `brew install ollama` | `winget install Ollama.Ollama` |
| Node.js (VLM only) | `brew install node` | `winget install OpenJS.NodeJS.LTS` |

### 2) Clone and enter the repo

```bash
git clone <YOUR_REPO_URL>
cd training
```

### 3) Start Ollama and pull models

```bash
# Start Ollama (if not already running)
ollama serve &

# Pull required models
ollama pull phi3:mini        # embeddings + text chat
ollama pull qwen3.5:4b       # vision model (VLM track)
```

### 4) Choose your path

#### Option A: Jupyter Notebooks (recommended for learning)

```bash
cd elc/session2/notebooks
uv sync                    # creates .venv with all dependencies (including JupyterLab)
uv run jupyter lab         # launch JupyterLab using the managed venv
```

Open the notebooks in order:
1. `01_chat_completion_seed.ipynb` — seed-based reproducibility
2. `02_rag_pipeline.ipynb` — text RAG end-to-end + Gradio Q&A
3. `03_vlm_rag_pipeline.ipynb` — vision RAG end-to-end + Gradio Q&A

Dependencies are managed by `uv` via `pyproject.toml` — no manual `pip install` needed.

#### Option B: Python Scripts (CLI)

```bash
# Text RAG
cd elc/session2/rag
cp .env.sample .env          # defaults to local Ollama
uv sync
uv run python 01_ingest.py
uv run python 02_embed.py
uv run python 03_query.py "What is attention?"

# VLM RAG
cd elc/session2/vlm-rag
cp .env.sample .env
uv sync
uv run python 01_ingest.py
uv run python 02_embed.py
uv run python 03_query.py "Summarize the key chart"
```

#### Option C: Chat Completion Script

```bash
cd elc/session2
# Create .env with your OpenAI API key:
echo 'OPENAI_API_KEY=your_key' > .env
chmod +x chat_completion.sh
./chat_completion.sh "Explain RAG in 3 bullet points"
```

---

## Don't Want to Run Locally?

We provide a Docker image that bundles Ollama + JupyterLab + all notebooks. You can either:

### Run with Docker locally

```bash
docker compose -f deploy/docker-compose.yml up --build
# Open http://localhost:8888
```

### Deploy to Azure

The `deploy/` folder contains everything to deploy to Azure Container Instances:

1. **Infrastructure**: Pulumi IaC in `deploy/pulumi/`
2. **CI/CD**: GitHub Actions in `.github/workflows/deploy.yml`

See [deploy/README.md](#deployment) below for details.

---

## Session 2 Labs

### A) Chat Completion with Seed Control

Demonstrates how `seed` + `temperature=0` produces deterministic outputs.

- **Script**: `elc/session2/chat_completion.sh` (Bash) / `.ps1` (PowerShell)
- **Notebook**: `elc/session2/notebooks/01_chat_completion_seed.ipynb`

### B) Text RAG (Ollama + SQLite)

Full RAG pipeline: PDF → chunks → embeddings → vector search → LLM answer.

- **Scripts**: `elc/session2/rag/01_ingest.py` → `02_embed.py` → `03_query.py`
- **Notebook**: `elc/session2/notebooks/02_rag_pipeline.ipynb`

### C) VLM RAG (Vision + Retrieval)

Vision-aware RAG: page images → VLM descriptions → dual embeddings → reranking → VLM answer.

- **Scripts**: `elc/session2/vlm-rag/01_ingest.py` → `02_embed.py` → `03_query.py` → `04_query_rerank.py`
- **Notebook**: `elc/session2/notebooks/03_vlm_rag_pipeline.ipynb`

Each RAG notebook includes a **Gradio Q&A** section at the end that launches an interactive web interface for querying your documents.

---

## Deployment

### Docker

```bash
# Build and run (includes Ollama + JupyterLab)
docker compose -f deploy/docker-compose.yml up --build

# Access at http://localhost:8888
```

### Azure (Pulumi + GitHub Actions)

Prerequisites:
- [Pulumi CLI](https://www.pulumi.com/docs/install/) installed
- Azure subscription with contributor access
- GitHub repository secrets configured:
  - `AZURE_CREDENTIALS` — Azure service principal JSON
  - `PULUMI_ACCESS_TOKEN` — Pulumi access token

Manual deployment:
```bash
cd deploy/pulumi
pip install -r requirements.txt
pulumi stack init dev
pulumi up
```

The GitHub Actions workflow (`.github/workflows/deploy.yml`) triggers automatically on pushes to `main` that modify `deploy/` or `elc/`.

---

## Configuration

All labs default to **local Ollama** (`PROVIDER=local`). To use Azure AI Foundry instead:

1. Copy `.env.sample` to `.env` in the relevant lab folder
2. Set `PROVIDER=azure`
3. Fill in your Azure endpoint URLs and API key

See `.env.sample` in each lab folder for all available options.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ollama` commands fail | Start Ollama: `ollama serve` or open the Ollama app |
| `uv` not found | Restart terminal after installing uv |
| Python version too old | Ensure `python --version` shows 3.11+ |
| Notebook kernel not found | Run `pip install ipykernel` and restart JupyterLab |
| Gradio app won't launch | Ensure port 7860 is free, or set `demo.launch(server_port=7861)` |
| Docker build fails | Ensure Docker Desktop is running and has at least 8GB RAM allocated |
| `rag.db not found` in Gradio | Run the ingest + embed steps first (notebooks 02/03 or the scripts) |

---

## Notes

- Each lab folder has its own focused README with deeper implementation details.
- Keep PDFs small during training runs for faster ingestion and query cycles.
- The notebooks create separate `.db` files (e.g., `rag_notebook.db`) to avoid conflicts with the CLI scripts.
