"""
Step 2: Chunks → Embeddings (read from DB, write vectors back to DB)

Read plain-text chunks from SQLite, embed each one via Ollama, and store
the vectors in a sqlite-vec virtual table — all in the same database.

Teaching point: "Each chunk becomes a high-dimensional vector —
text turned into numbers that capture meaning.
No Pinecone, no Weaviate — just SQLite with a vector extension."
"""

import sqlite3
import struct
import sys
from pathlib import Path

import sqlite_vec
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from config import get_embedding, embed_label

console = Console()

# --- Config ---
DB_FILE = Path(__file__).parent / "rag.db"


def serialize_float32(vector: list[float]) -> bytes:
    """Serialize a list of floats into a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def main():
    if not DB_FILE.exists():
        console.print("[red]rag.db not found.[/red] Run 01_ingest.py first.")
        sys.exit(1)

    db = sqlite3.connect(str(DB_FILE))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    # Read chunks from DB
    chunks = db.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    console.print(Panel(f"[bold]Embedding {len(chunks)} chunks using {embed_label()}[/bold]"))

    # Embed first chunk to discover dimension
    first_vec = get_embedding(chunks[0][1])
    dim = len(first_vec)

    # Create vec table (drop if re-running)
    db.execute("DROP TABLE IF EXISTS vec_chunks")
    db.execute(f"""
        CREATE VIRTUAL TABLE vec_chunks USING vec0(
            id INTEGER PRIMARY KEY,
            embedding FLOAT[{dim}]
        )
    """)

    # Insert first embedding, then the rest
    db.execute(
        "INSERT INTO vec_chunks (id, embedding) VALUES (?, ?)",
        (chunks[0][0], serialize_float32(first_vec)),
    )

    for chunk_id, text in track(chunks[1:], description="Embedding chunks..."):
        vector = get_embedding(text)
        db.execute(
            "INSERT INTO vec_chunks (id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_float32(vector)),
        )

    db.commit()

    # Show what embeddings look like
    console.print(f"\n[bold green]Embedding dimension:[/bold green] {dim}")
    console.print(
        f"[dim]Each chunk is now a vector of {dim} floating-point numbers[/dim]\n"
    )

    table = Table(title="Sample Embeddings (first 8 dimensions)", show_lines=True)
    table.add_column("Chunk ID", style="cyan", width=10)
    table.add_column("Text (first 60 chars)", style="white", width=40)
    table.add_column("Vector (first 8 dims)", style="yellow")

    # Re-embed a few to display (cheaper than storing raw vectors in memory)
    for chunk_id, text in chunks[:5]:
        vector = get_embedding(text)
        preview = [f"{v:.4f}" for v in vector[:8]]
        table.add_row(
            str(chunk_id),
            text[:60].replace("\n", " ") + "...",
            "[" + ", ".join(preview) + ", ...]",
        )

    console.print(table)

    # DB stats
    vec_count = db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    chunk_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    stats = Table(title="Database Stats")
    stats.add_column("Table", style="cyan")
    stats.add_column("Rows", style="yellow")
    stats.add_column("Description", style="white")
    stats.add_row("chunks", str(chunk_count), "Plain text + metadata")
    stats.add_row("vec_chunks", str(vec_count), f"Embedding vectors ({dim}d)")
    console.print(stats)

    db.close()
    console.print(f"\n[bold green]Embeddings stored in {DB_FILE.name}[/bold green]")


if __name__ == "__main__":
    main()
