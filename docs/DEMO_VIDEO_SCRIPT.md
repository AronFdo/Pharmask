# Pharmask — Demo video script (compulsory structure)

**Target length:** ~8–12 minutes (adjust pacing; practise once with a timer).

**Before recording:** Start the app (`uvicorn app.main:app --host 0.0.0.0 --port 8000`), open the browser UI, zoom to ~110%, hide bookmarks bar, close unrelated tabs.

**On-screen identity (optional):** Small text overlay: *Pharmask — Hybrid Pharmaceutical RAG (FYP)*.

---

## 1️⃣ Problem & research gap (~2–3 min)

### What to show
- Title slide or browser on project README / proposal title (no API keys visible).
- Optional: one bullet slide (PowerPoint / Google Slides) — keep it minimal.

### What to say (narration script)

**Problem**

> Pharmaceutical information is split across **unstructured text** (papers, labels, narratives) and **structured tables** (dosages, indications, interactions). Users often ask questions that need **one or both**: for example, “what is this product indicated for?” is naturally **structured**, while “how does this mechanism work in the literature?” is **narrative**. A plain chat model can hallucinate; a plain semantic search can miss **exact tabular facts** or fail to combine **evidence types**.

**Research gap**

> Many demos use **vector-only RAG**. That gap matters here because **label and database-style answers** should be grounded in **relational evidence**, not only similar sentences. There is a need for a **routed hybrid pipeline**: classify the question, retrieve from **ChromaDB** for text and **SQLite** for structured rows, then **synthesize** with clear sources — with a **two-tier LLM** design to balance cost (cheap routing) and quality (strong synthesis).

**Justification (one sentence)**

> This project implements that hybrid design and evaluates it against a **vector-only** baseline so we can show **when structured routing actually changes behaviour**.

### Cut / transition
> “Next I’ll show how the system is built, then I’ll run live tests.”

---

## 2️⃣ Solution & technical design (~3–4 min)

### A. Architecture (diagram — ~90 seconds)

**What to show**
- Your **sequence diagram** (Mermaid export or slide): user → FastAPI → orchestrator → Tier-1 classify → **HybridRetriever** → Chroma / SQLite → Tier-2 synthesize.

**What to say**

> The flow is: **Tier-1** classifies the query as **text**, **sql**, or **hybrid**. **HybridRetriever** then runs **vector search** and/or **SQL retrieval**. For SQL, a **Tier-1 model generates read-only SQL** from the schema; results are executed on **SQLite**. **Tier-2** synthesizes the final answer with **citations**. Ingestion loads **DailyMed / PMC-style** content into **both** stores.

### B. Code walkthrough (~2 min — pick 2 files max)

**What to show**
- IDE: `app/services/orchestrator.py` — highlight `process_query`: classify → retrieve → synthesize.
- IDE: `app/services/retrieval/hybrid_retriever.py` — show the `if text / sql / hybrid` branches.

**What to say**

> “Orchestration is **explicit Python** — not the umbrella LangChain agent framework — here in `RAGOrchestrator`. Retrieval is **branching logic** in `HybridRetriever`: text-only hits Chroma; SQL-only generates and runs SQL; hybrid does **both**.”

**Do not** scroll through long files or paste API keys.

### Cut / transition
> “Now I’ll demonstrate the running system with intentional **positive** and **negative** cases.”

---

## 3️⃣ System demonstration (~4–5 min)

### Setup (5 seconds on screen)

**What to show**
- Terminal: `uvicorn` running (or `GET /health` in browser tab showing `healthy`).

**What to say**
> “The app is running locally; the UI posts to `/query`.”

---

### ✅ Positive test cases (show clear success)

**Goal:** Show **classification**, **sources**, and a **grounded** answer.

| # | Type | Example query (use UI chips or paste) | What to point out |
|---|------|----------------------------------------|-------------------|
| P1 | SQL | *What is 911 Stress and Anxiousness indicated for according to the database?* | Badge **sql**; **table** sources; answer matches rows. |
| P2 | TEXT | *What side effects and safety warnings are commonly discussed for prescription medicines in the retrieved literature?* | Badge **text**; **text** sources; coherent paragraphs. |
| P3 | HYBRID | *What is Good Mood Enhancer indicated for in the database, and what does the biomedical text add about mood, stress, or nervousness?* | Badge **hybrid**; both **text** and **table** sources if data exists. |

**What to say (between cases)**

> “Positive case: the retriever returns evidence and the answer stays within context.”

---

### ❌ Negative test cases (show clear behaviour — not “broken”, but **honest limits**)

**Goal:** Show **empty retrieval**, **refusal / uncertainty**, or **no data** — not a crash.

| # | Type | Example query | Expected behaviour to narrate |
|---|------|---------------|--------------------------------|
| N1 | SQL / data gap | *What is the recommended dose of COMPLETELY_FAKE_DRUG_XYZ for adults according to the database?* | Classifier may still be **sql**; **no or sparse rows**; answer should say **not enough information** or similar — **no hallucinated dose**. |
| N2 | Nonsense / out of domain | *What is the weather in Paris tomorrow?* | Irrelevant; expect **weak or empty** evidence and a **safe** refusal or “not in knowledge base”. |
| N3 | Limitation / aggregation | *What is the single best treatment across all diseases in the corpus?* | Should **not** invent a single drug; ideally **uncertainty** — corpus cannot support a universal ranking. |

**What to say**

> “Negative tests show the system under **missing data** or **unanswerable** questions. The important behaviour is **not** a perfect score — it’s **transparency**: no evidence, no fabricated facts.”

---

### Closing (~30 seconds)

**What to say**

> “To summarise: we addressed a **gap** between unstructured and structured pharmaceutical evidence, implemented a **hybrid RAG** pipeline with **evaluation** against a baselines, and demonstrated **live** behaviour on **positive** and **negative** queries. Thank you.”

---

## Checklist (compulsory coverage)

| Requirement | Where in video |
|-------------|----------------|
| Problem explained | Section 1 |
| Research gap + justification | Section 1 |
| Architecture / algorithm / model | Section 2A |
| Code walkthrough | Section 2B |
| Positive tests | Section 3 — P1–P3 |
| Negative tests | Section 3 — N1–N3 |
| Behaviour visible | Zoom UI; show classification badge + sources + answer text |

---

## Recording tips

- **Record audio** clearly; if using Zoom/Teams, record **separate** system audio + mic if needed.
- **One take per test**; edit in silence between sections.
- **Blur** `.env` if it appears in IDE.
