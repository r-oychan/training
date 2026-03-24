"""
Step 4: Search + Rerank + Answer — VLM RAG with Reranking

Same as 03_query.py but adds a reranking step:
  1. Embed question → vector search (wide net: top 6 candidates)
  2. Rerank candidates using LLM scoring (cross-encoder style)
  3. Keep top 3 after reranking
  4. Send page images to VLM for answer

Teaching point: "Vector search is fast but imprecise — it matches by embedding
distance, which can miss nuance. A reranker reads each candidate carefully and
scores how relevant it actually is to your question."
"""

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
RERANK_MODEL = "phi3:mini"          # for scoring relevance (cross-encoder style)
VLM_MODEL = "qwen3.5:4b"           # vision model for answering
DB_FILE = Path(__file__).parent / "rag.db"
INITIAL_K = 6                       # wider net for initial retrieval
FINAL_K = 3                         # keep top 3 after reranking


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
    """Search for the most similar pages using vector distance."""
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

    # Deduplicate by page
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


def rerank(question: str, candidates: list[dict]) -> list[dict]:
    """Rerank candidates using LLM as a cross-encoder scorer.

    For each candidate, ask the LLM to score how relevant the page content
    is to the question on a scale of 0-10.
    """
    scored = []
    for c in candidates:
        # Use description (structured summary) for scoring — more informative
        context = c["description"]

        response = requests.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": RERANK_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Rate relevance of the document to the question. "
                            "Reply with ONLY a single integer 0-10.\n\n"
                            f"Question: {question}\n\n"
                            f"Document: {context}\n\n"
                            "Relevance score (0-10):"
                        ),
                    },
                ],
                "temperature": 0,
                "max_completion_tokens": 4,
            },
        )
        response.raise_for_status()
        score_text = response.json()["choices"][0]["message"]["content"].strip()

        # Parse score — extract first number, clamp to 0-10
        score = 0.0
        for token in score_text.replace("/", " ").split():
            try:
                val = float(token)
                if 0 <= val <= 10:
                    score = val
                    break
            except ValueError:
                continue

        scored.append({**c, "rerank_score": score})

    # Sort by rerank score (highest first)
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored


def build_messages(question: str, results: list[dict]) -> list[dict]:
    """Build messages with page images for the VLM."""
    content_parts = []

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
        console.print("[red]Usage:[/red] uv run python 04_query_rerank.py \"your question here\"")
        sys.exit(1)

    question = sys.argv[1]

    if not DB_FILE.exists():
        console.print("[red]rag.db not found.[/red] Run 01_ingest.py and 02_embed.py first.")
        sys.exit(1)

    console.print(Panel(f"[bold]Question:[/bold] {question}"))

    # Step 1: Embed the question
    console.print(f"\n[bold]1. Embedding question (using {EMBED_MODEL})...[/bold]")
    query_vec = get_embedding(question)
    console.print(f"   Vector dimension: {len(query_vec)}")

    # Step 2: Vector search — wider net
    console.print(f"\n[bold]2. Vector search: top-{INITIAL_K} candidates...[/bold]")
    db = sqlite3.connect(str(DB_FILE))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    candidates = search_similar(db, query_vec, INITIAL_K)
    db.close()

    table = Table(title=f"Initial Candidates (vector search, top {INITIAL_K})", show_lines=True)
    table.add_column("Rank", style="cyan", width=5)
    table.add_column("Dist", style="yellow", width=8)
    table.add_column("Page", style="green", width=5)
    table.add_column("Matched", style="magenta", width=12)
    table.add_column("Description (first 100 chars)", style="white")

    for i, r in enumerate(candidates):
        table.add_row(
            str(i + 1),
            f"{r['distance']:.4f}",
            str(r["page"]),
            r["matched_on"],
            r["description"][:100].replace("\n", " ") + "...",
        )

    console.print(table)

    # Step 3: Rerank
    console.print(f"\n[bold]3. Reranking with {RERANK_MODEL} (scoring relevance 0-10)...[/bold]")
    reranked = rerank(question, candidates)

    table2 = Table(title="After Reranking", show_lines=True)
    table2.add_column("Rank", style="cyan", width=5)
    table2.add_column("Score", style="yellow", width=7)
    table2.add_column("Page", style="green", width=5)
    table2.add_column("Was #", style="dim", width=6)
    table2.add_column("Description (first 100 chars)", style="white")

    # Show old rank for comparison
    old_ranks = {r["page"]: i + 1 for i, r in enumerate(candidates)}
    for i, r in enumerate(reranked):
        style = "bold" if i < FINAL_K else "dim"
        marker = " *" if i < FINAL_K else ""
        table2.add_row(
            f"[{style}]{i + 1}{marker}[/{style}]",
            f"[{style}]{r['rerank_score']:.0f}[/{style}]",
            f"[{style}]{r['page']}[/{style}]",
            f"[{style}]{old_ranks[r['page']]}[/{style}]",
            f"[{style}]{r['description'][:100].replace(chr(10), ' ')}...[/{style}]",
        )

    console.print(table2)
    console.print(f"  [dim]* = selected for VLM (top {FINAL_K} after reranking)[/dim]")

    results = reranked[:FINAL_K]

    # Step 4: Show prompt
    console.print(f"\n[bold]4. Prompt sent to VLM ({VLM_MODEL}):[/bold]\n")
    messages = build_messages(question, results)

    console.print(Panel(
        messages[0]["content"],
        title="[bold cyan]system[/bold cyan]",
        border_style="dim",
    ))

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

    # Step 5: Get the answer
    console.print(f"\n[bold]5. VLM response ({VLM_MODEL}):[/bold]\n")
    answer = chat(messages)

    console.print(Panel(answer, title="[bold green]Answer[/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
