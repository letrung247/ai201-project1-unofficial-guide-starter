"""
Milestone 4 — Embedding and retrieval.

Pipeline stages: Embedding + Vector Store -> Retrieval
- Loads the chunks produced by ingest.py (chunks.json)
- Embeds each chunk locally with sentence-transformers all-MiniLM-L6-v2
- Stores embeddings + source metadata in a persistent ChromaDB collection
- Exposes retrieve(query, k) for the generation stage (Milestone 5)

Build the index:   python embed.py            (re-embeds everything)
Test retrieval:    python embed.py --test     (runs the eval-plan queries)

ChromaDB notes (for the exercise):
- We use a PersistentClient so the index survives between runs (stored in
  chroma_db/, which is git-ignored).
- We pass our OWN embeddings to collection.add()/query() rather than letting
  Chroma pick a default embedding function, so the same all-MiniLM-L6-v2 model
  embeds both documents and queries.
- The collection is created with {"hnsw:space": "cosine"} so query() returns
  cosine distances (0 = identical, ~1 = unrelated), which is what the distance
  thresholds in planning.md / the milestone assume.
"""

import json
import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "augie_dining"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 6  # planning.md Retrieval Approach

# Cache the model and collection so repeated retrieve() calls don't reload them.
_model = None
_collection = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model '{EMBED_MODEL}'...", file=sys.stderr)
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_collection():
    """Return the persistent Chroma collection (read-only access)."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def build_index():
    """Embed every chunk and (re)load it into ChromaDB."""
    if not os.path.exists(CHUNKS_FILE):
        sys.exit(f"ERROR: {CHUNKS_FILE} not found. Run `python ingest.py` first.")

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} chunks from {CHUNKS_FILE}.")

    model = get_model()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Start clean so re-running doesn't duplicate or stale the index.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    texts = [r["text"] for r in records]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")
    embeddings = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[r["id"] for r in records],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"source": r["source"], "chunk_index": r["chunk_index"]}
            for r in records
        ],
    )
    print(f"Stored {collection.count()} chunks in ChromaDB collection "
          f"'{COLLECTION_NAME}' at {CHROMA_DIR}/.")


def retrieve(query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return the top-k chunks most similar to `query`.

    Each result: {text, source, chunk_index, distance}. Lower distance = closer.
    """
    model = get_model()
    collection = get_collection()
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        results.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })
    return results


# Subset of the planning.md evaluation-plan queries used to sanity-check retrieval.
TEST_QUERIES = [
    "What are the dining hall hours on weekends?",
    "What do students say about food quality in the dining halls?",
    "What accommodations are available for vegetarian or gluten-free meals?",
]


def test_retrieval():
    if get_collection().count() == 0:
        sys.exit("ERROR: index is empty. Run `python embed.py` first.")
    for q in TEST_QUERIES:
        print(f"\n{'='*72}\nQUERY: {q}\n{'='*72}")
        for rank, r in enumerate(retrieve(q), 1):
            preview = r["text"].replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240] + "..."
            print(f"\n[{rank}] distance={r['distance']:.3f}  "
                  f"source={r['source']} #{r['chunk_index']}")
            print("    " + preview.encode("ascii", "replace").decode())


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_retrieval()
    else:
        build_index()
