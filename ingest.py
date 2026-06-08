"""
Milestone 3 — Document ingestion, cleaning, and chunking.

Pipeline stage: Document Ingestion -> Chunking
- Loads every .txt document in data/
- Cleans each document (strip HTML tags/entities, normalize whitespace/quotes)
- Splits each document into overlapping chunks using a recursive
  character splitter that respects paragraph -> sentence -> word boundaries
- Attaches source metadata (filename + position) to every chunk
- Saves the chunks to chunks.json for the embedding stage (Milestone 4)

Run:  python ingest.py
"""

import html
import json
import os
import re
import sys

DATA_DIR = "data"
OUTPUT_FILE = "chunks.json"

# --- Chunking parameters (see planning.md "Chunking Strategy") ----------------
# planning.md originally specified 400 tokens (~1,400 chars). Inspecting that
# output, a 1,400-char chunk merged 5-6 FAQ Q&A pairs into one embedding and
# produced only 26 chunks across 10 docs (below the 50-chunk guideline), so
# specific queries (e.g. dietary accommodations, weekend hours) matched coarsely.
# We reduced to ~225 tokens / 800 chars with ~40 tokens / 150 chars overlap so
# each chunk isolates one or two FAQ pairs / discussion points while keeping
# narrative paragraphs intact. planning.md has been updated to reflect this.
CHUNK_SIZE = 800       # characters (~225 tokens)
CHUNK_OVERLAP = 150    # characters (~40 tokens, ~19% overlap)
MIN_CHUNK_LEN = 50     # drop fragments shorter than this many characters


def clean_text(text: str) -> str:
    """Remove boilerplate/markup and normalize whitespace.

    Our documents are already hand-collected plain text, so this is light:
    strip any stray HTML tags, decode HTML entities (&amp;, &nbsp;, &#39;),
    normalize smart quotes, and collapse runaway whitespace while preserving
    paragraph breaks (blank lines) so the chunker can split on them.
    """
    # Strip HTML tags if any slipped in.
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities (&amp; -> &, &#39; -> ', &nbsp; -> space).
    text = html.unescape(text)
    # Normalize smart quotes / dashes to ASCII for consistent embeddings.
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-")
                .replace(" ", " "))
    # Strip trailing whitespace on each line.
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Collapse 3+ newlines into a paragraph break (2 newlines).
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _split_recursive(text: str, separators) -> list[str]:
    """Split text on the first separator that actually divides it, recursing
    into pieces still larger than CHUNK_SIZE. Mirrors LangChain's
    RecursiveCharacterTextSplitter behavior without the dependency."""
    if len(text) <= CHUNK_SIZE or not separators:
        return [text]

    sep = separators[0]
    rest = separators[1:]
    parts = text.split(sep) if sep else list(text)

    pieces = []
    for part in parts:
        if len(part) <= CHUNK_SIZE:
            pieces.append(part)
        else:
            pieces.extend(_split_recursive(part, rest))
    return pieces


def chunk_text(text: str) -> list[str]:
    """Produce overlapping chunks that respect content boundaries.

    Strategy: split into paragraphs, then greedily pack paragraphs into chunks
    up to CHUNK_SIZE. Paragraphs longer than CHUNK_SIZE are recursively split on
    sentence then word boundaries. Consecutive chunks share CHUNK_OVERLAP
    characters so context spanning a boundary is preserved.
    """
    # First break into atomic pieces no larger than CHUNK_SIZE.
    separators = ["\n\n", "\n", ". ", " "]
    atoms = _split_recursive(text, separators)
    atoms = [a.strip() for a in atoms if a.strip()]

    chunks: list[str] = []
    current = ""
    for atom in atoms:
        candidate = (current + " " + atom).strip() if current else atom
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Start the next chunk with overlap tail of the previous chunk.
            if current and CHUNK_OVERLAP > 0:
                tail = current[-CHUNK_OVERLAP:]
                # Begin overlap at a word boundary for readability.
                space = tail.find(" ")
                tail = tail[space + 1:] if space != -1 else tail
                current = (tail + " " + atom).strip()
            else:
                current = atom
            # If a single atom still exceeds CHUNK_SIZE, hard-split it.
            while len(current) > CHUNK_SIZE:
                chunks.append(current[:CHUNK_SIZE])
                current = current[CHUNK_SIZE - CHUNK_OVERLAP:]
    if current:
        chunks.append(current)

    # Drop empty / fragment chunks.
    return [c.strip() for c in chunks if len(c.strip()) >= MIN_CHUNK_LEN]


def load_documents(data_dir: str) -> dict[str, str]:
    """Load every .txt file in data_dir as {filename: raw_text}."""
    if not os.path.isdir(data_dir):
        sys.exit(f"ERROR: data directory '{data_dir}' not found.")
    docs = {}
    for name in sorted(os.listdir(data_dir)):
        if name.lower().endswith(".txt"):
            with open(os.path.join(data_dir, name), encoding="utf-8") as f:
                docs[name] = f.read()
    if not docs:
        sys.exit(f"ERROR: no .txt files found in '{data_dir}'.")
    return docs


def build_chunks(docs: dict[str, str]) -> list[dict]:
    """Clean and chunk every document, attaching source metadata."""
    records = []
    for source, raw in docs.items():
        cleaned = clean_text(raw)
        for i, chunk in enumerate(chunk_text(cleaned)):
            records.append({
                "id": f"{source}::chunk_{i}",
                "text": chunk,
                "source": source,
                "chunk_index": i,
            })
    return records


def main():
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents from '{DATA_DIR}/':")
    for name, raw in docs.items():
        print(f"  - {name:30s} {len(raw):>6,} chars raw")

    # Show one cleaned document so we can eyeball the cleaning step.
    sample_name = next(iter(docs))
    print(f"\n{'='*70}\nCLEANED PREVIEW: {sample_name}\n{'='*70}")
    print(clean_text(docs[sample_name])[:600], "...\n")

    records = build_chunks(docs)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # --- Stats -----------------------------------------------------------------
    lengths = [len(r["text"]) for r in records]
    print(f"{'='*70}\nCHUNK STATS\n{'='*70}")
    print(f"Total chunks:        {len(records)}")
    print(f"Chunk size target:   {CHUNK_SIZE} chars  (overlap {CHUNK_OVERLAP})")
    print(f"Min / max / avg len: {min(lengths)} / {max(lengths)} / "
          f"{sum(lengths)//len(lengths)} chars")
    empties = sum(1 for r in records if not r["text"].strip())
    print(f"Empty chunks:        {empties}")
    print(f"Saved to:            {OUTPUT_FILE}")

    # --- Inspect 5 representative chunks ---------------------------------------
    # Spread across the corpus rather than all from one document.
    step = max(1, len(records) // 5)
    sample_idxs = list(range(0, len(records), step))[:5]
    print(f"\n{'='*70}\n5 REPRESENTATIVE CHUNKS (inspect for standalone meaning)\n{'='*70}")
    for n, idx in enumerate(sample_idxs, 1):
        r = records[idx]
        print(f"\n--- Chunk {n}  [{r['source']} #{r['chunk_index']}]  "
              f"({len(r['text'])} chars) ---")
        print(r["text"])


if __name__ == "__main__":
    main()
