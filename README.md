# The Unofficial Guide — Augustana College Dining

A retrieval-augmented (RAG) question-answering system that answers questions
about dining at Augustana College — hours, meal plans, food quality, dietary
options, and campus dining locations — using a collection of official pages,
menus, FAQs, a blog post, and student discussions. Answers are grounded in the
collected documents and cite their sources.

**Run it:**

```bash
pip install -r requirements.txt
cp .env.example .env          # add your free Groq API key
python ingest.py              # load + clean + chunk documents -> chunks.json
python embed.py               # embed chunks into ChromaDB
python app.py                 # launch the Gradio UI at http://localhost:7860
```

`python embed.py --test` and `python evaluate.py` reproduce the retrieval and
end-to-end evaluation results below.

---

## Domain

This system covers **dining at Augustana College**: where to eat on campus,
when dining locations are open, what meal plans cost and include, what dietary
accommodations exist, and what students actually think of the food.

This knowledge is valuable but scattered. Official pages list hours and meal
plans but say nothing about whether the food is any good; student opinions live
in Reddit threads and a blog post; menus live in a separate JavaScript app.
A prospective or current student deciding on a meal plan has to stitch together
four or five different sources. This system consolidates them and answers in one
place, while keeping official facts (hours, prices, policies) distinguishable
from subjective student opinion.

---

## Document Sources

Ten documents were collected as plain `.txt` files in `data/`. JavaScript-rendered
sources (the menu app, Reddit) were captured by copying/summarizing their text
content, since they could not be scraped directly.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Augustana Dining (main page) | Official page | `data/augustana_dining.txt` — augustana.edu/.../dining |
| 2 | Dining Locations and Hours | Official page | `data/dining_locations.txt` — augustana.edu/.../dining/locations |
| 3 | Campus Food Options (locations + hours detail) | Official page | `data/campus_food_options.txt` — augustana.edu/.../dining/locations |
| 4 | Dining FAQ | Official FAQ | `data/dining_faq.txt` — augustana.edu/.../dining/faq |
| 5 | Meal Plans 2025–26 | Official page | `data/meal_plans.txt` — augustana.edu/.../dining/plans-25-26 |
| 6 | Dining Menus (menu app description) | Menu system | `data/dining_menus.txt` — augustana.net/csldining/new_app/ |
| 7 | "What's good to eat at Augie?" | Blog post | `data/augie_food_blog.txt` — augustana.edu/blog/whats-good-eat-augie |
| 8 | Dining Services overview | Official page | `data/dining_services.txt` — augustana.edu/.../dining |
| 9 | Reddit — Augustana dining discussion | Student forum | `data/reddit_augustana_1.txt` — r/QuadCities thread |
| 10 | Reddit — "any red flags?" discussion | Student forum | `data/reddit_augustana_2.txt` — r/IntltoUSA thread |

---

## Chunking Strategy

**Chunk size:** ~225 tokens / **800 characters** (revised down from the original
400-token plan).

**Overlap:** ~40 tokens / **150 characters** (~19%).

**Preprocessing before chunking** (`clean_text()` in `ingest.py`): strip any HTML
tags, decode HTML entities (`&amp;`, `&#39;`, `&nbsp;`), normalize smart quotes
and dashes to ASCII, strip trailing whitespace, collapse 3+ blank lines to a
single paragraph break, and collapse runs of spaces. A cleaned document is
printed during ingestion so the cleaning step can be eyeballed.

**Chunking method:** a boundary-respecting recursive character splitter
(paragraph → sentence → word) that greedily packs pieces up to 800 chars and
carries a 150-char overlap tail into the next chunk. Chunks shorter than 50
characters are dropped, so there are no empty/fragment chunks.

**Why these choices fit the documents:** The corpus mixes structured content
(FAQ Q&A pairs, hour tables, meal-plan descriptions) with narrative content
(blog, Reddit). The original 400-token size was too coarse for these short,
dense documents — a single chunk merged 5–6 distinct FAQ Q&A pairs into one
embedding, and the whole corpus produced only **26 chunks**, below the 50-chunk
guideline. At ~225 tokens each chunk isolates one or two FAQ pairs / discussion
points while keeping narrative paragraphs intact, which sharpens retrieval on
specific queries (e.g. "weekend hours", "dietary options").

**Final chunk count:** **46 chunks** across 10 documents (avg 646 chars, range
291–798, 0 empty).

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, run locally (no
API key, no rate limits). Stored in a persistent **ChromaDB** collection with
cosine distance and source metadata (`source` filename + `chunk_index`). The
same model embeds both documents and queries.

**Production tradeoff reflection:** all-MiniLM-L6-v2 (384-dim) is fast,
lightweight, and accurate enough for this prototype's dining Q&A. If I were
deploying this for real users and cost weren't a constraint, I'd evaluate larger
retrieval models such as **BGE-base/large** or **E5-large**, which generally
score higher on domain-specific retrieval and would better distinguish subtle
distinctions (e.g. meal-plan tiers, "Any 15" vs "Any 19"). E5 models also offer
stronger multilingual support, which would matter for the international students
who appear in the Reddit sources. The tradeoffs are latency and memory: larger
models are slower per query and need more RAM/GPU, and an API-hosted embedder
adds per-call cost and a network dependency. For a small, single-domain corpus
like this, MiniLM's speed and zero cost outweigh the marginal accuracy gain.

---

## Grounded Generation

**LLM:** Groq `llama-3.3-70b-versatile` (free tier, OpenAI-compatible),
temperature 0.1.

**System prompt grounding instruction** (`SYSTEM_PROMPT` in `query.py`): the
model is told to answer **using ONLY the numbered context passages**, to use no
outside or prior knowledge, and — when the context is insufficient — to reply
*verbatim* with `"I don't have enough information on that."` It is also told to
cite the source filename(s) inline and to distinguish official information
(hours, plans, policies) from student opinion (Reddit/blog), since opinions may
be outdated or subjective.

**Structural grounding choices** (not left to the prompt alone):
- Retrieved chunks are formatted into a **numbered, source-labeled context
  block** (`[Passage N] (source: <file>)`) so the model can attribute claims.
- A **relevance gate** drops chunks whose cosine distance exceeds 0.85; if no
  chunk clears the gate, the system returns the "not enough information" answer
  **deterministically, without calling the LLM**.
- A **decline detector** strips the source list when the model itself declines,
  so a refusal never carries spurious citations.

**How source attribution is surfaced:** the response's `sources` list is built
**programmatically** from the metadata of the chunks actually passed to the
model (ordered by retrieval rank), not parsed from the LLM's text. The Gradio UI
shows it in a separate "Retrieved from" panel, so attribution is guaranteed even
if the model forgets to cite inline.

---

## Evaluation Report

Run with `python evaluate.py`. Distance = cosine distance of the top retrieved
chunk (lower = closer).

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What are the dining hall hours on weekends? | Specific Saturday/Sunday hours | Sat: continuous 9 a.m.–7 p.m., brunch 10–1:30, dinner 4:30–7; Sun: continuous 9 a.m.–8 p.m., brunch 10–1:30, dinner 4:30–8 (source: campus_food_options.txt) | Relevant (0.350) | **Accurate** |
| 2 | What do students say about food quality in the dining halls? | Summary of student opinion | "Practical, consistent, and sufficient" but not "exceptional or gourmet"; not "restaurant-quality" but reliable (source: reddit_augustana_1.txt) | Relevant (0.300) | **Accurate** |
| 3 | What meal plan options are available for 2025–26? | Full list of plans (Gerber Unlimited, Any 19/15/12/10, Any 80/100) | Returned Any 15 and Gerber Unlimited; explicitly said the full list "is not provided" (source: meal_plans.txt) | Partially relevant (0.360) | **Partially accurate** |
| 4 | Which dining location has the shortest wait times per student reviews? | A specific location with reasoning | "I don't have enough information on that." | Off-target (no doc covers wait times) | **Accurate (correct refusal)** |
| 5 | What accommodations exist for vegetarian/gluten-free meals? | Special meal options + how to request | Gluten-free, vegetarian, vegan daily; Wild Thymes station; menu app allergen filters (VGT/GF/V); Halal; contact Dining Services (sources: dining_faq, campus_food_options, dining_menus, augustana_dining) | Relevant (0.421) | **Accurate** |

**Summary:** 3 accurate, 1 correct refusal, 1 partially accurate. Four of five
top-result distances are below 0.5; the system never hallucinated, and it
correctly declined the one question (Q4) the corpus cannot support.

---

## Failure Case Analysis

**Question that failed:** Q3 — "What meal plan options are available for the
2025–26 academic year?"

**What the system returned:** It named only the *Any 15* plan and the *Gerber
Unlimited* plan and stated that "the full list of meal plan options is not
provided" — even though `data/meal_plans.txt` explicitly lists **Gerber
Unlimited, Any 19, Any 15, Any 12, Any 10, Any 80, and Any 100**.

**Root cause (tied to the chunking + embedding/retrieval stages):** The full
enumeration of plan names sits in `meal_plans.txt` chunk #2, whose surrounding
text is mostly about *where* plans can be used ("…Westerlin Market, and Gus's
Snack Bar. However, the Gerber Unlimited plan is restricted to meal swipes used
only in the Gerber Dining Center…"). Because the chunk's overall topic is usage
locations and restrictions, its embedding lands far from the query "what meal
plan **options** are available" — it does not appear even in the top 10 results
(its distance is worse than 0.53), while the question-framing intro chunk (#0,
"What are the meal plans for 2025–26?") ranks first at 0.360. So the chunk that
actually contains the answer was split away from its framing header and never
reached the generation stage, and the model honestly reported it couldn't see
the full list. This is a chunking-boundary problem amplified by embedding
dilution, not a generation failure — the model behaved correctly given what it
was handed.

**What I would change to fix it:** Chunk this document so the plan enumeration
stays attached to its framing ("There are multiple plan types…: Any 19, Any 15,
…"), e.g. heading-aware or list-aware splitting, or a smaller chunk size for
list-structured documents. A query-side fix (HyDE / query expansion that turns
"what options are available" into the plan names) or a hybrid keyword+vector
retriever would also surface the enumeration chunk. Simply raising k does **not**
fix it here, which is what makes the root cause a representation problem rather
than a cutoff problem.

*(Secondary observation — Q4: the system declined because no document covers
wait times. This is correct behavior, but it exposes a **coverage gap** in the
ingestion stage: the evaluation plan assumed wait-time data the corpus does not
actually contain.)*

---

## Spec Reflection

**One way the spec helped you during implementation:** Writing the Chunking
Strategy, Retrieval Approach, and architecture diagram in `planning.md` *before*
coding gave me concrete, unambiguous targets to build against — exact chunk
size/overlap, `all-MiniLM-L6-v2`, top-k = 6, cosine distance, and a clear
five-stage pipeline. When I prompted an AI tool to generate each stage, I could
hand it those specifics, and reviewing the output became a simple matter of
checking it against the spec ("does this produce 400-token chunks with overlap?
does it store source metadata?") rather than guessing what I wanted. The diagram
in particular kept the stages decoupled, so `embed.py` cleanly consumed
`ingest.py`'s output and `query.py` consumed `embed.py`'s.

**One way your implementation diverged from the spec, and why:** The plan
specified 400-token chunks and LangChain's `RecursiveCharacterTextSplitter`.
After building the pipeline and **inspecting the actual chunks**, I cut the chunk
size to ~225 tokens (800 chars): at 400 tokens a single chunk merged 5–6 FAQ
Q&A pairs and the corpus produced only 26 chunks, too coarse for the short, dense
documents. I also implemented the splitter as a small custom recursive character
splitter rather than pulling in LangChain, to avoid an extra dependency while
reproducing the same paragraph→sentence→word boundary behavior. Both changes are
documented in `planning.md`. This is exactly the "verify each stage before
relying on it" loop the milestone asked for — the spec was the starting point,
and inspection corrected it.

---

## AI Usage

**Instance 1 — Ingestion and chunking**

- *What I gave the AI:* The Documents and Chunking Strategy sections of
  `planning.md` (file types, 400-token / 75-token overlap target) plus the
  pipeline diagram, and asked it to implement a script that loads, cleans, and
  chunks the documents with source metadata.
- *What it produced:* `ingest.py` with a `clean_text()` function and a recursive
  character splitter sized to the 400-token spec, which yielded 26 chunks.
- *What I changed or overrode:* After printing and inspecting the chunks, I
  directed it to reduce the chunk size to 800 chars / 150 overlap because the
  400-token chunks merged multiple unrelated FAQ pairs and produced too few
  chunks. I also had it print 5 representative and 5 random chunks for the
  checkpoint and update `planning.md` to record why the numbers changed.

**Instance 2 — Grounded generation**

- *What I gave the AI:* My grounding requirement (answer from retrieved context
  only, decline otherwise, cite sources), the desired output format
  (answer + source list), and the retrieval function from `embed.py`.
- *What it produced:* `query.py` with a system prompt instructing the model to
  use only the context, and source citation requested in the prompt.
- *What I changed or overrode:* I tightened grounding so it doesn't rely on the
  prompt alone: I made source attribution **programmatic** (built from chunk
  metadata, not parsed from the LLM's text), added a **relevance gate** that
  returns the "not enough information" answer deterministically when no chunk is
  close enough, and added a **decline detector** that removes citations when the
  model itself declines. I verified this by asking an out-of-domain question
  ("Who is the current president of the United States?"), which the system
  correctly refused.
