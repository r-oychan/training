"""
Shared config for VLM RAG demo — supports local (Ollama) and Azure AI Foundry.

Reads from .env in the same directory. Switch PROVIDER between "local" and "azure".

Azure uses OpenAI-compatible endpoints (same format as the OpenAI Python client):
  base_url + /chat/completions, /embeddings
  Authorization: Bearer <key>
"""

import os
from pathlib import Path

import requests

# Load .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

PROVIDER = os.environ.get("PROVIDER", "local")

# --- Ollama ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "phi3:mini")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "phi3:mini")
OLLAMA_VLM_MODEL = os.environ.get("OLLAMA_VLM_MODEL", "qwen3.5:4b")

# --- Azure AI Foundry (OpenAI-compatible) ---
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")
AZURE_CHAT_BASE_URL = os.environ.get("AZURE_CHAT_BASE_URL", "").rstrip("/")
AZURE_CHAT_DEPLOYMENT = os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-4o")
AZURE_VLM_BASE_URL = os.environ.get("AZURE_VLM_BASE_URL", "").rstrip("/")
AZURE_VLM_DEPLOYMENT = os.environ.get("AZURE_VLM_DEPLOYMENT", "gpt-4o")
AZURE_EMBED_BASE_URL = os.environ.get("AZURE_EMBED_BASE_URL", "").rstrip("/")
AZURE_EMBED_DEPLOYMENT = os.environ.get("AZURE_EMBED_DEPLOYMENT", "text-embedding-ada-002")


def _azure_headers() -> dict:
    return {
        "Authorization": f"Bearer {AZURE_API_KEY}",
        "Content-Type": "application/json",
    }


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from configured provider."""
    if PROVIDER == "azure":
        response = requests.post(
            f"{AZURE_EMBED_BASE_URL}/embeddings",
            headers=_azure_headers(),
            json={"model": AZURE_EMBED_DEPLOYMENT, "input": text},
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    else:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": text},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


def chat(messages: list[dict], temperature: float = 0, max_tokens: int = 1024) -> str:
    """Send chat completion to configured provider (text-only, for reranking etc)."""
    if PROVIDER == "azure":
        response = requests.post(
            f"{AZURE_CHAT_BASE_URL}/chat/completions",
            headers=_azure_headers(),
            json={
                "model": AZURE_CHAT_DEPLOYMENT,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            },
        )
    else:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": OLLAMA_CHAT_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            },
        )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def vlm_chat(messages: list[dict], temperature: float = 0, max_tokens: int = 1024) -> str:
    """Send chat completion to VLM (vision model, supports image content parts)."""
    if PROVIDER == "azure":
        response = requests.post(
            f"{AZURE_VLM_BASE_URL}/chat/completions",
            headers=_azure_headers(),
            json={
                "model": AZURE_VLM_DEPLOYMENT,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            },
        )
    else:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": OLLAMA_VLM_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            },
        )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def embed_label() -> str:
    """Display label for embedding model."""
    if PROVIDER == "azure":
        return f"Azure ({AZURE_EMBED_DEPLOYMENT})"
    return f"Ollama ({OLLAMA_EMBED_MODEL})"


def chat_label() -> str:
    """Display label for chat model."""
    if PROVIDER == "azure":
        return f"Azure ({AZURE_CHAT_DEPLOYMENT})"
    return f"Ollama ({OLLAMA_CHAT_MODEL})"


def vlm_label() -> str:
    """Display label for VLM model."""
    if PROVIDER == "azure":
        return f"Azure ({AZURE_VLM_DEPLOYMENT})"
    return f"Ollama ({OLLAMA_VLM_MODEL})"
