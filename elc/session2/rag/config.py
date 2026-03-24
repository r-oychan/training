"""
Shared config for RAG demo — supports local (Ollama) and Azure AI Foundry.

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
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")

# --- Azure AI Foundry (OpenAI-compatible) ---
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")
AZURE_CHAT_BASE_URL = os.environ.get("AZURE_CHAT_BASE_URL", "").rstrip("/")
AZURE_CHAT_DEPLOYMENT = os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-4o")
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
            json={"model": OLLAMA_MODEL, "input": text},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


def chat(messages: list[dict], temperature: float = 0, max_tokens: int = 1024) -> str:
    """Send chat completion to configured provider."""
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
                "model": OLLAMA_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            },
        )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def provider_label() -> str:
    """Display label for chat model."""
    if PROVIDER == "azure":
        return f"Azure ({AZURE_CHAT_DEPLOYMENT})"
    return f"Ollama ({OLLAMA_MODEL})"


def embed_label() -> str:
    """Display label for embedding model."""
    if PROVIDER == "azure":
        return f"Azure ({AZURE_EMBED_DEPLOYMENT})"
    return f"Ollama ({OLLAMA_MODEL})"
