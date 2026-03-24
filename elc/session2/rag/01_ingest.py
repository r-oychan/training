"""
Step 1: PDF → Chunks → SQLite

Load a PDF, extract text page by page, chunk it, and store directly in SQLite.
No intermediate files — the database is the single source of truth.

Teaching point: "This is how we break documents into digestible pieces for the LLM"
"""

import sqlite3
import sys
from pathlib import Path

import fitz  # pymupdf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# --- Config ---
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks
DOCS_DIR = Path(__file__).parent / "docs"
DB_FILE = Path(__file__).parent / "rag.db"


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from each page of a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks of fixed character size."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def main():
    # Find PDFs in docs/
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        console.print("[red]No PDF files found in docs/ folder.[/red]")
        console.print("Place a PDF in the docs/ directory and run again.")
        console.print(
            "\nExample — download a sample PDF:\n"
            "  curl -L -o docs/attention.pdf "
            "https://arxiv.org/pdf/1706.03762"
        )
        sys.exit(1)

    # Fresh DB each run
    if DB_FILE.exists():
        DB_FILE.unlink()

    db = sqlite3.connect(str(DB_FILE))
    db.execute("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            page INTEGER NOT NULL
        )
    """)

    total = 0
    for pdf_path in pdf_files:
        console.print(Panel(f"[bold]Processing:[/bold] {pdf_path.name}"))

        pages = extract_text_from_pdf(pdf_path)
        console.print(f"  Extracted text from [cyan]{len(pages)}[/cyan] pages\n")

        for page_info in pages:
            chunks = chunk_text(page_info["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk in chunks:
                db.execute(
                    "INSERT INTO chunks (text, source, page) VALUES (?, ?, ?)",
                    (chunk, pdf_path.name, page_info["page"]),
                )
                total += 1

    db.commit()

    # Display sample chunks from DB
    console.print(f"\n[bold green]Total chunks stored:[/bold green] {total}\n")

    table = Table(title="Sample Chunks (from SQLite)", show_lines=True)
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Source", style="magenta", width=20)
    table.add_column("Page", style="yellow", width=6)
    table.add_column("Text (first 120 chars)", style="white")

    rows = db.execute("SELECT id, source, page, text FROM chunks LIMIT 10").fetchall()
    for row in rows:
        table.add_row(
            str(row[0]),
            row[1],
            str(row[2]),
            row[3][:120].replace("\n", " ") + "...",
        )
    console.print(table)

    if total > 10:
        console.print(f"\n  ... and {total - 10} more chunks\n")

    db.close()
    console.print(f"[bold green]Saved {total} chunks to {DB_FILE.name}[/bold green]")


if __name__ == "__main__":
    main()
