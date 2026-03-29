# Detailed script: `orchestrator.py` & `hybrid_retriever.py`

Use this as a **spoken walkthrough** for demos, viva, or thesis defence. Open the two files side-by-side in the IDE while you talk. Approx. **6–9 minutes** if you cover everything; skip “Optional depth” for a shorter take.

---

## Big picture (30 seconds)

**Say:**

> The orchestrator is the **single entry point** for one user question. It wires three services in order: **classify**, **retrieve**, **synthesize**. It does **not** implement retrieval itself—that lives in `HybridRetriever`, which decides whether to call **Chroma** (vector), **SQLite** (SQL), or **both**, based on the classification.

**Show:** `RAGOrchestrator.process_query` in `app/services/orchestrator.py` (the method body).

---

## Part A — `app/services/orchestrator.py`

### Imports and dependencies (lines 1–10)

**What the code does:** Pulls in the domain models and the three pipeline services plus cost calculation.

**Say:**

> We import `QueryClassifier` for Tier-1 routing, `HybridRetriever` for evidence gathering, and `AnswerSynthesizer` for Tier-2 answers. `calculate_cost` turns token counts into dollar estimates for the API response.

**Point at:** Lines 7–10 — each service has one responsibility.

---

### Class docstring (lines 15–21)

**Say:**

> The docstring states the **three stages** explicitly: Tier-1 classification, retrieval from vector and/or SQL, then Tier-2 synthesis. That’s the architecture in one place.

---

### `__init__` (lines 23–27)

**Say:**

> On construction we instantiate **one** classifier, **one** hybrid retriever, and **one** synthesizer. The orchestrator is a **thin coordinator**—it doesn’t hold business logic for SQL or embeddings; it **delegates**.

**Optional depth:** Mention that each call to `process_query` could reuse these instances (same as FastAPI creating one orchestrator per request in `main.py` today—if you ever move to a singleton, say so in thesis).

---

### `process_query` — token accounting (lines 39–40)

**Say:**

> We track **Tier-1** and **Tier-2** tokens separately. Tier-1 includes **classification** and any **SQL-generation** calls inside retrieval; Tier-2 is **answer synthesis** only. That split supports the cost panel in the UI.

---

### Step 1 — Classification (lines 42–48)

**Code:** `classification, t1_classify_tokens = await self.classifier.classify(query)`

**Say:**

> First we call the **QueryClassifier**. It uses the **Tier-1** LLM to output a structured label: `text`, `sql`, or `hybrid`, plus confidence and reasoning. The returned token count is added to **Tier-1** totals. This step decides **which retrieval paths** run next—we don’t retrieve everything for every question.

**Point at:** `classification.query_type` — this value is passed straight into the retriever.

---

### Step 2 — Retrieval (lines 50–57)

**Code:** `retrieval_result, t1_retrieve_tokens = await self.retriever.retrieve(query, classification)`

**Say:**

> Second, we pass the **same user query** and the **classification** into `HybridRetriever`. The retriever is responsible for filling `RetrievalResult`: **text chunks** from Chroma, **SQL rows** from SQLite, and a unified **sources** list. Extra Tier-1 tokens here usually come from **natural-language-to-SQL** inside `SQLRetriever`.

**Point at:** Log line counting chunks and rows — useful when debugging “not enough context”.

---

### Step 3 — Synthesis (lines 59–61)

**Code:** `answer, sources, t2_tokens = await self.synthesizer.synthesize(query, retrieval_result)`

**Say:**

> Third, the **Tier-2** model receives the **original question** and the **full retrieval bundle**. It must ground the answer in the provided text and SQL context and return citations. Tier-2 token count is tracked separately for cost.

---

### Source merging (lines 63–64)

**Code:** `all_sources = self._merge_sources(retrieval_sources, synthesis_sources)`

**Say:**

> Retrieval already produced sources; synthesis may attach or refine sources. `_merge_sources` **deduplicates** by `(source_type, source_id)` so the UI doesn’t show the same chunk twice.

**Open:** `_merge_sources` (lines 89–113).

**Say:**

> We iterate retrieval sources first, then synthesis sources, and skip duplicates. Order is stable: first-seen wins.

---

### Cost and response (lines 66–87)

**Say:**

> `calculate_cost` maps Tier-1 and Tier-2 tokens to USD using the pricing helper. We build a **QueryResponse** with query, answer, classification, merged sources, token counts, and cost. **Latency** is set to zero here because the **FastAPI endpoint** measures wall-clock time and overwrites `latency_ms` after `process_query` returns.

**Point at:** Line 75 — `latency_ms=0` comment.

---

## Part B — `app/services/retrieval/hybrid_retriever.py`

### Role of the class (lines 13–17)

**Say:**

> `HybridRetriever` is the **policy layer** for retrieval. It reads `classification.query_type` and runs **only** the retrieval engines that match. That’s how we avoid always doing expensive vector + SQL for every query.

---

### `__init__` (lines 19–22)

**Say:**

> We compose two engines: **VectorRetriever** talks to Chroma; **SQLRetriever** generates SQL and runs it via **SQLClient** on SQLite. Hybrid retrieval is **composition**, not inheritance.

---

### `retrieve` — inputs and outputs (lines 24–42)

**Say:**

> The method takes the **raw query string** and the **QueryClassification** object. It returns a **RetrievalResult** and an **integer** of extra Tier-1 tokens used (mainly SQL generation). We initialise empty lists for chunks, rows, and sources.

---

### Branch: `text` (lines 46–50)

**Say:**

> If the classifier says **text**, we only call **vector retrieval**. Chroma returns semantically similar **chunks** from ingested papers and label text. **No SQL** is run—so we don’t waste SQL generation tokens on purely narrative questions.

**Point at:** `vector_result.get("chunks", [])` — defensive default if keys are missing.

---

### Branch: `sql` (lines 52–57)

**Say:**

> If the label is **sql**, we only call **SQLRetriever**. That path uses the Tier-1 model to **generate a read-only SELECT** from the schema, executes it, and maps rows to **table-type** sources. **No vector search**—exact lookups and joins come from the database.

**Point at:** `tokens_used += tokens` — SQL path can consume Tier-1 tokens.

---

### Branch: `hybrid` (lines 59–68)

**Say:**

> Otherwise we treat the query as **hybrid**. We run **vector retrieval first**, then **SQL retrieval**, and concatenate sources from both. Both **text chunks** and **SQL rows** are passed to the synthesizer so answers can combine **narrative evidence** with **structured facts**—for example indication text plus dosage rows.

**Optional depth:** Mention order is **vector then SQL**; if you ever need parallel execution, that would be an optimisation here.

---

### Deduplication and result (lines 70–84)

**Say:**

> We deduplicate **sources** by `(source_type, source_id)` so hybrid runs don’t list the same reference twice. We then pack everything into **RetrievalResult** and log a one-line summary: chunk count, row count, and query type.

**Open:** `_deduplicate_sources` (lines 86–97) — same pattern as orchestrator merge, but only for retrieval sources.

---

## How the two files connect (30 seconds)

**Say:**

> **Orchestrator** owns the **pipeline order** and **response assembly**. **HybridRetriever** owns **which stores** are queried given `query_type`. Classification errors can still route to the wrong branch—that’s why evaluation measures **classification accuracy** and **DB-type correctness** separately.

**Diagram (say aloud):**

> User query → **Orchestrator** → **Classifier** → `query_type` → **HybridRetriever** branches → Chroma / SQLite → **RetrievalResult** → **Synthesizer** → **QueryResponse**.

---

## Suggested IDE tour order (for video)

1. `orchestrator.py` — full `process_query` from line 29 to 87 in one scroll.
2. `hybrid_retriever.py` — `retrieve` from line 24 to 84, pause on each `if` branch.
3. Jump to `vector_retriever.py` or `sql_retriever.py` only if asked “what happens inside?”

---

## Quick Q&A cues

| Question | Answer |
|----------|--------|
| Why not LangChain agents here? | Orchestration is **explicit Python** for clarity; LangChain is used inside classifier / SQL / synthesis for **LLM bindings**, not for the control flow. |
| Where is hybrid “smart”? | In **Tier-1 classification** + **HybridRetriever** branching; not in the orchestrator itself. |
| What if both retrievers return empty? | Synthesizer still runs; `AnswerSynthesizer` should return an **insufficient context** style message (see synthesis module). |

---

## File references

```29:87:app/services/orchestrator.py
    async def process_query(self, query: str) -> QueryResponse:
        ...
        classification, t1_classify_tokens = await self.classifier.classify(query)
        ...
        retrieval_result, t1_retrieve_tokens = await self.retriever.retrieve(query, classification)
        ...
        answer, sources, t2_tokens = await self.synthesizer.synthesize(query, retrieval_result)
        ...
        return response
```

```24:84:app/services/retrieval/hybrid_retriever.py
    async def retrieve(
        self,
        query: str,
        classification: QueryClassification,
    ) -> Tuple[RetrievalResult, int]:
        ...
        if query_type == "text":
            ...
        elif query_type == "sql":
            ...
        else:  # hybrid
            ...
        return result, tokens_used
```
