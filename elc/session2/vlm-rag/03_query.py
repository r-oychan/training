"""
Step 3: Search + Answer — The VLM RAG Pipeline

Key difference from text RAG:
  - We retrieve pages by text embedding (same as before)
  - But we send the page IMAGES to a Vision LLM for the answer

Teaching point: "The VLM sees the actual page — tables, figures, layout —
not lossy extracted text"
"""

import base64
import sqlite3
import struct
import sys
from pathlib import Path

import requests
import sqlite_vec
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# --- Config ---
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "phi3:mini"           # for embedding the question
VLM_MODEL = "qwen3.5:4b"           # vision model for answering
DB_FILE = Path(__file__).parent / "rag.db"
TOP_K = 3


def serialize_float32(vector: list[float]) -> bytes:
    """Serialize a list of floats into a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def search_similar(db: sqlite3.Connection, query_vec: list[float], top_k: int) -> list[dict]:
    """Search for the most similar pages using vector distance.

    We search across both text and description embeddings, then deduplicate
    by page — keeping the best (closest) match per page.
    """
    # Fetch more candidates to account for dedup (2 embeddings per page)
    rows = db.execute(
        """
        SELECT v.embed_id, v.distance, e.page_id, e.source_type,
               p.text, p.description, p.image_b64, p.source, p.page
        FROM vec_pages v
        JOIN embed_index e ON e.embed_id = v.embed_id
        JOIN pages p ON p.id = e.page_id
        WHERE v.embedding MATCH ?
            AND v.k = ?
        ORDER BY v.distance
        """,
        (serialize_float32(query_vec), top_k * 2),
    ).fetchall()

    # Deduplicate by page — keep the best match per page
    seen_pages = set()
    results = []
    for r in rows:
        page_id = r[2]
        if page_id in seen_pages:
            continue
        seen_pages.add(page_id)
        results.append({
            "id": page_id, "distance": r[1], "matched_on": r[3],
            "text": r[4], "description": r[5],
            "image_b64": r[6], "source": r[7], "page": r[8],
        })
        if len(results) >= top_k:
            break

    return results


def build_messages(question: str, results: list[dict]) -> list[dict]:
    """Build messages with page images for the VLM."""
    # Build user content with images
    content_parts = []

    # Add each retrieved page image + its VLM description
    for r in results:
        content_parts.append({
            "type": "text",
            "text": f"[Page {r['page']} from {r['source']}]\nDescription: {r['description']}",
        })
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{r['image_b64']}",
            },
        })

    # Add the question
    content_parts.append({
        "type": "text",
        "text": f"\n---\n\nQuestion: {question}",
    })

    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question based on "
                "the provided page images ONLY, not internal knowledge. If the images "
                "don't contain enough information, say so. "
                "Cite which page the information comes from when possible."
            ),
        },
        {
            "role": "user",
            "content": content_parts,
        },
    ]


def chat(messages: list[dict]) -> str:
    """Send messages with images to VLM via Ollama."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/v1/chat/completions",
        json={
            "model": VLM_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_completion_tokens": 1024,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage:[/red] uv run python 03_query.py \"your question here\"")
        sys.exit(1)

    question = sys.argv[1]

    if not DB_FILE.exists():
        console.print("[red]rag.db not found.[/red] Run 01_ingest.py and 02_embed.py first.")
        sys.exit(1)

    console.print(Panel(f"[bold]Question:[/bold] {question}"))

    # Step 1: Embed the question (using text embedding model)
    console.print(f"\n[bold]1. Embedding question (using {EMBED_MODEL})...[/bold]")
    query_vec = get_embedding(question)
    console.print(f"   Vector dimension: {len(query_vec)}")

    # Step 2: Vector search
    console.print(f"\n[bold]2. Searching for top-{TOP_K} similar pages...[/bold]")
    db = sqlite3.connect(str(DB_FILE))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    results = search_similar(db, query_vec, TOP_K)
    db.close()

    table = Table(title="Retrieved Pages", show_lines=True)
    table.add_column("Rank", style="cyan", width=5)
    table.add_column("Dist", style="yellow", width=8)
    table.add_column("Page", style="green", width=5)
    table.add_column("Matched", style="magenta", width=12)
    table.add_column("VLM Description (first 120 chars)", style="white")

    for i, r in enumerate(results):
        table.add_row(
            str(i + 1),
            f"{r['distance']:.4f}",
            str(r["page"]),
            r["matched_on"],
            r["description"][:120].replace("\n", " ") + "...",
        )

    console.print(table)

    # Step 3: Show what we're sending to the VLM
    console.print(f"\n[bold]3. Prompt sent to VLM ({VLM_MODEL}):[/bold]\n")
    messages = build_messages(question, results)

    # Show system message
    console.print(Panel(
        messages[0]["content"],
        title="[bold cyan]system[/bold cyan]",
        border_style="dim",
    ))

    # Show user message (text parts only, summarize images)
    user_parts = messages[1]["content"]
    prompt_preview = ""
    for part in user_parts:
        if part["type"] == "text":
            prompt_preview += part["text"] + "\n"
        else:
            prompt_preview += "  [attached page image]\n"

    console.print(Panel(
        prompt_preview.strip(),
        title="[bold yellow]user[/bold yellow]",
        border_style="dim",
    ))

    # Step 4: Get the answer from VLM
    console.print(f"\n[bold]4. VLM response ({VLM_MODEL}):[/bold]\n")
    answer = chat(messages)

    console.print(Panel(answer, title="[bold green]Answer[/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
