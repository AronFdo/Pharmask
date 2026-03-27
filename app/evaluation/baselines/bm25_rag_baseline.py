"""BM25-only RAG baseline for pharmaceutical Q&A.

This uses the BM25Retriever over the ingested Chroma corpus to fetch text
chunks, then passes them to the existing AnswerSynthesizer (Tier-2 LLM only)
to generate answers. It is intended as a lexical-retrieval counterpart to the
full Hybrid RAG system for pre-evaluation comparisons.
"""

import time
import logging
from typing import Optional

from app.evaluation.schemas import EvalQuestion, BaselineResponse
from app.evaluation.baselines.base import BaseSystem
from app.services.retrieval import BM25Retriever
from app.services.synthesis import AnswerSynthesizer
from app.models import RetrievalResult
from app.services.cost_calculator import calculate_cost

logger = logging.getLogger(__name__)


class BM25RAGBaseline(BaseSystem):
    """BM25-only RAG baseline (BM25 retrieval + Tier-2 synthesis)."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.retriever = BM25Retriever(top_k=top_k)
        self.synthesizer = AnswerSynthesizer()
        self._initialized = False

    @property
    def name(self) -> str:
        return "BM25 RAG (Text Only)"

    async def initialize(self) -> None:
        """Build the BM25 index from the existing vector store."""
        if not self._initialized:
            await self.retriever.initialize()
            self._initialized = True

    async def answer(self, question: EvalQuestion) -> BaselineResponse:
        """
        Answer using BM25 retrieval over the Chroma-backed corpus + Tier-2 LLM.

        For PubMedQA-style tasks, we ignore `question.context` and instead
        search the global pharmaceutical text corpus being evaluated.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Step 1: retrieve chunks via BM25
        retrieval_raw = await self.retriever.retrieve(question.question, top_k=self.top_k)
        chunks = retrieval_raw.get("chunks", [])
        sources = retrieval_raw.get("sources", [])

        retrieval_result = RetrievalResult(
            text_chunks=chunks,
            sql_rows=[],
            sources=sources,
        )

        # Step 2: synthesize answer with Tier-2 model
        answer_text, synth_sources, tier2_tokens = await self.synthesizer.synthesize(
            question.question,
            retrieval_result,
        )

        # Merge source IDs for logging
        all_sources = {s.source_id for s in sources}
        all_sources.update(s.source_id for s in synth_sources)

        latency_ms = (time.time() - start_time) * 1000

        # For PubMedQA, we still need a yes/no/maybe label
        predicted_label = self._extract_label(answer_text)

        # Cost: no Tier-1, only Tier-2
        cost = calculate_cost(
            tier1_tokens=0,
            tier2_tokens=tier2_tokens,
        )

        return BaselineResponse(
            answer=answer_text,
            predicted_label=predicted_label,
            latency_ms=latency_ms,
            tokens_used=tier2_tokens,
            cost_usd=cost.total_cost_usd,
            sources=list(all_sources),
        )

    def _extract_label(self, answer: str) -> str:
        """Extract yes/no/maybe label from answer text (mirrors other baselines)."""
        answer_lower = answer.lower()

        if "final answer:" in answer_lower:
            final_part = answer_lower.split("final answer:")[-1].strip()[:20]
            if "yes" in final_part:
                return "yes"
            if "no" in final_part:
                return "no"
            if "maybe" in final_part:
                return "maybe"

        first_sentence = answer_lower.split(".")[0]
        if "yes" in first_sentence and "no" not in first_sentence:
            return "yes"
        if "no" in first_sentence and "yes" not in first_sentence:
            return "no"

        yes_count = answer_lower.count(" yes")
        no_count = answer_lower.count(" no ")
        maybe_count = answer_lower.count("maybe") + answer_lower.count("uncertain") + answer_lower.count("unclear")

        if yes_count > no_count and yes_count > maybe_count:
            return "yes"
        if no_count > yes_count and no_count > maybe_count:
            return "no"
        if maybe_count > 0:
            return "maybe"

        return "maybe"

