"""
Step 1: PDF → Page images + text + VLM description → SQLite

For each page we store:
  - The page rendered as a base64 PNG image (what the VLM will see at query time)
  - The extracted text via liteparse (better spatial extraction than pymupdf)
  - A VLM-generated description of the page image (captures figures, tables, layout)

The description + extracted text together are used for embedding in step 2.

Teaching point: "Text extraction misses figures and tables.
Asking a VLM to describe the page captures what text extraction cannot."
"""

import base64
import json
import platform
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # pymupdf — used only for rendering page images
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from config import vlm_chat, vlm_label

console = Console()

# --- Config ---
DOCS_DIR = Path(__file__).parent / "docs"
DB_FILE = Path(__file__).parent / "rag.db"
DPI = 150  # render resolution for page images


def extract_text_with_liteparse(pdf_path: Path) -> dict[int, str]:
    """Extract text per page using liteparse (better spatial reading order)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    # shell=True needed on Windows so subprocess can find npm .cmd wrappers (lit.cmd)
    subprocess.run(
        ["lit", "parse", str(pdf_path), "--format", "json", "--no-ocr", "-o", out_path],
        check=True, capture_output=True,
        shell=(platform.system() == "Windows"),
    )

    with open(out_path) as f:
        data = json.load(f)

    Path(out_path).unlink()

    return {page["page"]: page["text"] for page in data["pages"] if page["text"].strip()}


def render_page_to_base64(page: fitz.Page) -> str:
    """Render a PDF page to a base64-encoded PNG string."""
    pix = page.get_pixmap(dpi=DPI)
    return base64.b64encode(pix.tobytes("png")).decode("ascii")


def describe_image(image_b64: str, page_num: int) -> str:
    """Ask the VLM to describe a page image."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"This is page {page_num} of a document. Describe this page concisely:\n"
                        "1. If there are diagrams or figures: state the exact title/label (e.g. 'Figure 1: ...') and briefly describe what the diagram shows.\n"
                        "2. If there are charts or graphs: state the title, axis labels, and what trend or comparison is shown.\n"
                        "3. If there are tables: state the title and what the columns/rows represent.\n"
                        "4. If there are equations: write them out.\n"
                        "5. Summarize the main text content in 2-3 sentences.\n"
                        "Focus on extracting structured information, not narrating layout."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ],
        },
    ]
    return vlm_chat(messages, max_tokens=512)


def main():
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
        CREATE TABLE pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            description TEXT NOT NULL,
            image_b64 TEXT NOT NULL,
            source TEXT NOT NULL,
            page INTEGER NOT NULL
        )
    """)

    all_pages = []
    for pdf_path in pdf_files:
        console.print(Panel(f"[bold]Processing:[/bold] {pdf_path.name}"))

        # Extract text with liteparse (better than pymupdf for reading order)
        console.print("  Extracting text with [cyan]liteparse[/cyan]...")
        page_texts = extract_text_with_liteparse(pdf_path)
        console.print(f"  Extracted text from [cyan]{len(page_texts)}[/cyan] pages")

        # Render page images with pymupdf
        console.print("  Rendering page images with [cyan]pymupdf[/cyan]...")
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            if (page_num + 1) not in page_texts:
                continue
            image_b64 = render_page_to_base64(doc[page_num])
            all_pages.append((
                pdf_path.name,
                page_num + 1,
                page_texts[page_num + 1],
                image_b64,
            ))

        doc.close()
        console.print(f"  Rendered [cyan]{len(all_pages)}[/cyan] page images\n")

    console.print(f"[bold]Describing pages with {vlm_label()}...[/bold]\n")

    for source, page_num, text, image_b64 in track(all_pages, description="Describing pages..."):
        description = describe_image(image_b64, page_num)
        db.execute(
            "INSERT INTO pages (text, description, image_b64, source, page) VALUES (?, ?, ?, ?, ?)",
            (text, description, image_b64, source, page_num),
        )

    db.commit()

    # Display summary
    console.print(f"\n[bold green]Total pages stored:[/bold green] {len(all_pages)}\n")

    table = Table(title="Pages in SQLite", show_lines=True)
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Page", style="yellow", width=5)
    table.add_column("Image", style="green", width=8)
    table.add_column("VLM Description (first 120 chars)", style="white")

    rows = db.execute(
        "SELECT id, page, LENGTH(image_b64), description FROM pages"
    ).fetchall()
    for row in rows:
        table.add_row(
            str(row[0]),
            str(row[1]),
            f"{row[2] // 1024} KB",
            row[3][:120].replace("\n", " ") + "...",
        )

    console.print(table)

    db.close()
    console.print(f"\n[bold green]Saved {len(all_pages)} pages to {DB_FILE.name}[/bold green]")


if __name__ == "__main__":
    main()
