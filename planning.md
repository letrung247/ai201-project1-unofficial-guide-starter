# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? --> 
This project creates an unofficial guide to dining at Augustana College. It combines information from dining websites, menus, FAQs, and student discussions to help students find answers about food quality, meal plans, dining hours, and campus dining options. This information is often scattered across multiple sources and can be difficult to find quickly.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Augustana Dining| Main dining information page| https://www.augustana.edu/student-life/residential-life/dining |
| 2 |Dining Locations |Campus dining locations and hours | https://www.augustana.edu/student-life/residential-life/dining/locations|
| 3 | Dining FAQ|Meal plans and dining questions | https://www.augustana.edu/student-life/residential-life/dining/faq|
| 4 |Meal Plans | Student meal plan information| https://www.augustana.edu/student-life/residential-life/dining/plans-25-26|
| 5 |Dining Menus |Available food and menus | https://augustana.net/csldining/new_app/|
| 6 |What's Good to Eat at Augie? |Blog about dining options | https://www.augustana.edu/blog/whats-good-eat-augie|
| 7 |Reddit Discussion 1 | Student opinions about Augustana| https://www.reddit.com/r/QuadCities/comments/1jo9isz/hey_guys_just_had_some_questions_about_augustana/|
| 8 | Reddit Discussion 2| Student experiences and advice|https://www.reddit.com/r/IntltoUSA/comments/12tsngz/enrolling_at_augustana_college_any_red_flags/ |
| 9 |Dining Services Page |Additional dining information | https://www.augustana.edu/student-life/residential-life/dining|
| 10 | Campus Dining Locations| Food options on campus| https://www.augustana.edu/student-life/residential-life/dining/locations|

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 225 tokens (~800 characters) — *revised down from 400 tokens during implementation*

**Overlap:** 40 tokens (~150 characters, ~19% overlap) — *revised down from 75 tokens*

**Reasoning:** Your corpus mixes structured (FAQs, menus, official pages) and unstructured (Reddit discussions, blog posts) content. The original plan called for 400-token chunks, but the documents turned out to be short and topically dense (1–4 KB each). At 400 tokens, a single chunk merged 5–6 distinct FAQ Q&A pairs into one embedding and the whole corpus produced only 26 chunks across 10 documents — below the 50-chunk guideline — so specific queries (e.g. "dietary accommodations", "weekend hours") matched coarsely against chunks covering many topics. Reducing to ~225 tokens / 800 characters isolates one or two FAQ pairs / discussion points per chunk while keeping narrative paragraphs intact, and yields 46 chunks (avg 646 chars, range 291–798). The ~19% overlap (40 tokens) preserves context when a meal-plan detail or hours block spans a boundary. A character-based recursive splitter (paragraph → sentence → word) is used in place of the LangChain RecursiveCharacterTextSplitter from the diagram, avoiding an extra dependency while reproducing the same boundary-respecting behavior.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** all-MiniLM-L6-v2 (via sentence-transformers)

**Top-k:** 6 

**Production tradeoff reflection:** For a real system, I'd evaluate larger models like BGE-base (384-dim) or E5-large for better domain-specific accuracy, especially to distinguish dining quality nuances and meal plan details. However, all-MiniLM-L6-v2 is practical for this prototype: fast inference, small memory footprint, and sufficient for dining Q&A. If budget allowed, E5 models offer better multilingual support (useful if supporting international students) and superior retrieval accuracy on domain-specific queries. For a 6-chunk retrieval, this balances coverage without overwhelming the generation stage.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What are the dining hall hours on weekends?| Specific hours for weekend dining (e.g., breakfast/lunch/dinner times)|
| 2 |What do students say about food quality in the dining halls? |Summary of student opinions from Reddit or blog sources (positive/negative feedback about variety, taste, options) |
| 3 |What meal plan options are available for the 2025-26 academic year? | List of available meal plans and their descriptions/pricing|
| 4 |Which dining location has the shortest wait times according to student reviews? | Specific dining hall name with reasoning from student discussions|
| 5 |What accommodations are available for dietary restrictions like vegetarian or gluten-free meals? |Information about special meal options and how to request them |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Inconsistent and outdated student reviews: Reddit discussions and blog posts may reflect experiences from different semesters or years. Menu changes, staffing changes, and meal plan updates may make student opinions obsolete. The system could confidently retrieve outdated negative feedback about a dining option that has since improved.

2. Missing source attribution in generation: When combining information from official pages (FAQs, dining hours) and student opinions (Reddit, blogs), the system may not clearly distinguish fact from opinion. A student's casual comment ("the food is bad") could be presented as fact alongside official meal plan information, potentially misleading users about actual dining policies.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     ↓                    ↓                    ↓                    ↓            ↓
  (URLs, HTML,    (LangChain's        (sentence-transformers  (Retrieve    (Claude/
   Reddit, blogs)  RecursiveText-      all-MiniLM-L6-v2 +      top-6        LLM
                     Splitter)           FAISS/Chroma)           chunks)      prompt)

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
