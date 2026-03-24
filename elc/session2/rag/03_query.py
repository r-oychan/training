"""
Step 3: Search + Answer — The RAG Pipeline

Tie it all together:
  1. Embed the user's question
  2. Search SQLite for similar chunks (vector search)
  3. Feed retrieved context + question to the LLM
  4. Print the answer

Teaching point: "RAG = retrieve relevant context + feed it to the LLM"
"""

import sqlite3
import struct
import sys
from pathlib import Path

import sqlite_vec
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import get_embedding, embed_label, provider_label
from config import chat as llm_chat

console = Console()

# --- Config ---
DB_FILE = Path(__file__).parent / "rag.db"
TOP_K = 3


def serialize_float32(vector: list[float]) -> bytes:
    """Serialize a list of floats into a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def search_similar(db: sqlite3.Connection, query_vec: list[float], top_k: int) -> list[dict]:
    """Search for the most similar chunks using vector distance."""
    rows = db.execute(
        """
        SELECT v.id, v.distance, c.text, c.source, c.page
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.id
        WHERE v.embedding MATCH ?
            AND v.k = ?
        ORDER BY v.distance
        """,
        (serialize_float32(query_vec), top_k),
    ).fetchall()

    return [
        {"id": r[0], "distance": r[1], "text": r[2], "source": r[3], "page": r[4]}
        for r in rows
    ]


def build_messages(question: str, context_chunks: list[dict]) -> list[dict]:
    """Build the messages array that will be sent to the LLM."""
    context_text = "\n\n---\n\n".join(
        f"[From {c['source']}, page {c['page']}]\n{c['text']}"
        for c in context_chunks
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question based on "
                "the provided context ONLY, not internal knowledge. If the context doesn't contain enough information, "
                "say so. Cite which page the information comes from when possible."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context_text}\n\n"
                f"---\n\n"
                f"Question: {question}"
            ),
        },
    ]




def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage:[/red] uv run python 03_query.py \"your question here\"")
        sys.exit(1)

    question = sys.argv[1]

    if not DB_FILE.exists():
        console.print("[red]rag.db not found.[/red] Run 01_ingest.py and 02_embed.py first.")
        sys.exit(1)

    console.print(Panel(f"[bold]Question:[/bold] {question}"))

    # Step 1: Embed the question
    console.print(f"\n[bold]1. Embedding question ({embed_label()})...[/bold]")
    query_vec = get_embedding(question)
    console.print(f"   Vector dimension: {len(query_vec)}")

    # Step 2: Vector search
    console.print(f"\n[bold]2. Searching for top-{TOP_K} similar chunks...[/bold]")
    db = sqlite3.connect(str(DB_FILE))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    results = search_similar(db, query_vec, TOP_K)
    db.close()

    table = Table(title="Retrieved Chunks", show_lines=True)
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Distance", style="yellow", width=10)
    table.add_column("Source", style="magenta", width=16)
    table.add_column("Page", style="green", width=6)
    table.add_column("Text (first 100 chars)", style="white")

    for i, r in enumerate(results):
        table.add_row(
            str(i + 1),
            f"{r['distance']:.4f}",
            r["source"],
            str(r["page"]),
            r["text"][:100].replace("\n", " ") + "...",
        )

    console.print(table)

    # Step 3: Build prompt and show what we're sending to the LLM
    console.print("\n[bold]3. Prompt sent to LLM:[/bold]\n")
    messages = build_messages(question, results)

    for msg in messages:
        role_style = "bold cyan" if msg["role"] == "system" else "bold yellow"
        console.print(Panel(
            msg["content"],
            title=f"[{role_style}]{msg['role']}[/{role_style}]",
            border_style="dim",
        ))

    # Step 4: Get the answer
    console.print(f"\n[bold]4. LLM response ({provider_label()}):[/bold]\n")
    answer = llm_chat(messages)

    console.print(Panel(answer, title="[bold green]Answer[/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
