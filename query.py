"""
Milestone 5 — Grounded generation.

Pipeline stage: Retrieval -> Generation
- retrieve() (from embed.py) pulls the top-k chunks for a question
- We build a context block that labels each chunk with its source filename
- We send it to Groq's llama-3.3-70b-versatile with a system prompt that
  enforces grounding: answer ONLY from the provided context, and say so when
  the context doesn't cover the question
- Source attribution is guaranteed programmatically: the returned `sources`
  list is built from the retrieved chunks' metadata, not from the LLM's text

ask(question) -> {"answer": str, "sources": [filenames], "chunks": [...]}

Quick test:  python query.py
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from embed import retrieve, DEFAULT_TOP_K

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

# Chunks whose cosine distance exceeds this are treated as too weak to ground an
# answer. In-domain matches here sit around 0.30-0.55; this gate mainly catches
# out-of-domain questions where every chunk is a poor match.
RELEVANCE_CUTOFF = 0.85

NO_ANSWER = "I don't have enough information on that."

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about dining at "
    "Augustana College. You must answer using ONLY the information in the "
    "numbered context passages provided in the user message. Follow these "
    "rules strictly:\n"
    "1. Do NOT use any outside or prior knowledge. If the context does not "
    "contain enough information to answer, reply with exactly: "
    f'"{NO_ANSWER}"\n'
    "2. When the context does answer the question, cite the source filename(s) "
    "you used inline, e.g. (source: dining_faq.txt).\n"
    "3. Distinguish official information (hours, meal plans, policies) from "
    "student opinions (Reddit/blog) when both appear, since opinions may be "
    "outdated or subjective.\n"
    "4. Be concise and specific. Do not invent hours, prices, or policies that "
    "are not in the context."
)

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key or key == "your_key_here":
            sys.exit("ERROR: GROQ_API_KEY not set. Copy .env.example to .env "
                     "and add your Groq key.")
        _client = Groq(api_key=key)
    return _client


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered, source-labeled context block."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[Passage {i}] (source: {c['source']})\n{c['text'].strip()}"
        )
    return "\n\n".join(blocks)


def ask(question: str, k: int = DEFAULT_TOP_K) -> dict:
    """Retrieve, ground, and generate an answer for `question`."""
    retrieved = retrieve(question, k=k)

    # Keep only chunks that are relevant enough to ground an answer.
    relevant = [c for c in retrieved if c["distance"] <= RELEVANCE_CUTOFF]

    # No chunk is close enough: decline deterministically, before calling the LLM.
    if not relevant:
        return {"answer": NO_ANSWER, "sources": [], "chunks": retrieved}

    context = build_context(relevant)
    user_msg = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. If it is insufficient, say "
        f'"{NO_ANSWER}"'
    )

    completion = get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,  # low temperature: stay close to the source text
    )
    answer = completion.choices[0].message.content.strip()

    # Programmatic source attribution: unique sources of the chunks we passed,
    # ordered by retrieval rank (best first). Guaranteed regardless of LLM text.
    sources = []
    for c in relevant:
        if c["source"] not in sources:
            sources.append(c["source"])

    # If the model itself declined, don't attach sources (nothing was used).
    if answer.strip().rstrip(".").lower() == NO_ANSWER.rstrip(".").lower():
        sources = []

    return {"answer": answer, "sources": sources, "chunks": relevant}


if __name__ == "__main__":
    demo_questions = [
        "What are the dining hall hours on weekends?",
        "What do students say about food quality in the dining halls?",
        "Who is the current president of the United States?",  # out-of-domain
    ]
    for q in demo_questions:
        print(f"\n{'='*72}\nQ: {q}\n{'='*72}")
        result = ask(q)
        print("A:", result["answer"].encode("ascii", "replace").decode())
        print("Sources:", result["sources"])
