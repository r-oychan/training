"""
Step 2: Page text + VLM description → Embeddings (stored back in the same DB)

We create TWO embeddings per page:
  1. One from the extracted text
  2. One from the VLM description (chart titles, axis labels, figure summaries)

Both point back to the same page. At query time we search both vectors,
so a question about "vacancies-to-unemployment ratio" can match the description
embedding (which has the chart title) even if the raw text doesn't mention it prominently.

Teaching point: "Small embedding models lose signal in long text.
Splitting into focused embeddings — text vs description — gives better retrieval."
"""

import sqlite3
import struct
import sys
from pathlib import Path

import requests
import sqlite_vec
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

console = Console()

# --- Config ---
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "phi3:mini"
DB_FILE = Path(__file__).parent / "rag.db"


def serialize_float32(vector: list[float]) -> bytes:
    """Serialize a list of floats into a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama /api/embed endpoint."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def main():
    if not DB_FILE.exists():
        console.print("[red]rag.db not found.[/red] Run 01_ingest.py first.")
        sys.exit(1)

    db = sqlite3.connect(str(DB_FILE))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    pages = db.execute("SELECT id, text, description FROM pages ORDER BY id").fetchall()
    console.print(Panel(f"[bold]Embedding {len(pages)} pages using {EMBED_MODEL}[/bold]"))
    console.print("[dim]Two embeddings per page: extracted text + VLM description (separate)[/dim]\n")

    # Discover dimension
    first_vec = get_embedding(pages[0][1])
    dim = len(first_vec)

    # Create embedding index table — maps each embedding back to a page
    db.execute("DROP TABLE IF EXISTS embed_index")
    db.execute("""
        CREATE TABLE embed_index (
            embed_id INTEGER PRIMARY KEY,
            page_id INTEGER NOT NULL,
            source_type TEXT NOT NULL
        )
    """)

    db.execute("DROP TABLE IF EXISTS vec_pages")
    db.execute(f"""
        CREATE VIRTUAL TABLE vec_pages USING vec0(
            embed_id INTEGER PRIMARY KEY,
            embedding FLOAT[{dim}]
        )
    """)

    embed_id = 0
    skipped = 0
    for page_id, text, description in track(pages, description="Embedding pages..."):
        # Embedding 1: extracted text
        text_vec = get_embedding(text)
        db.execute(
            "INSERT INTO embed_index (embed_id, page_id, source_type) VALUES (?, ?, ?)",
            (embed_id, page_id, "text"),
        )
        db.execute(
            "INSERT INTO vec_pages (embed_id, embedding) VALUES (?, ?)",
            (embed_id, serialize_float32(text_vec)),
        )
        embed_id += 1

        # Embedding 2: VLM description — only if it has visual content
        # Skip generic "no diagrams/charts" descriptions, they add noise
        desc_lower = description.lower()
        has_visuals = not (
            "no diagrams" in desc_lower
            or "no charts" in desc_lower
            or "there are no" in desc_lower
        )
        if has_visuals:
            desc_vec = get_embedding(description)
            db.execute(
                "INSERT INTO embed_index (embed_id, page_id, source_type) VALUES (?, ?, ?)",
                (embed_id, page_id, "description"),
            )
            db.execute(
                "INSERT INTO vec_pages (embed_id, embedding) VALUES (?, ?)",
                (embed_id, serialize_float32(desc_vec)),
            )
            embed_id += 1
        else:
            skipped += 1

    db.commit()

    # Show results
    console.print(f"\n[bold green]Embedding dimension:[/bold green] {dim}")
    console.print(f"[dim]{embed_id} total embeddings ({len(pages)} text + {embed_id - len(pages)} description, skipped {skipped} text-only pages)[/dim]\n")

    table = Table(title="Sample Embeddings", show_lines=True)
    table.add_column("Page", style="cyan", width=6)
    table.add_column("Type", style="magenta", width=12)
    table.add_column("Content (first 60 chars)", style="white", width=40)
    table.add_column("Vector (first 6 dims)", style="yellow")

    for page_id, text, description in pages[:3]:
        for label, content in [("text", text), ("description", description)]:
            vector = get_embedding(content)
            preview = [f"{v:.4f}" for v in vector[:6]]
            table.add_row(
                str(page_id),
                label,
                content[:60].replace("\n", " ") + "...",
                "[" + ", ".join(preview) + ", ...]",
            )

    console.print(table)

    stats = Table(title="Database Stats")
    stats.add_column("Table", style="cyan")
    stats.add_column("Rows", style="yellow")
    stats.add_column("Description", style="white")
    stats.add_row("pages", str(len(pages)), "Text + VLM description + image + metadata")
    stats.add_row("embed_index", str(embed_id), "Maps embeddings → pages")
    stats.add_row("vec_pages", str(embed_id), f"Embedding vectors ({dim}d)")
    console.print(stats)

    db.close()
    console.print(f"\n[bold green]Embeddings stored in {DB_FILE.name}[/bold green]")


if __name__ == "__main__":
    main()
