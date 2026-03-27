"""SQL-only RAG baseline for pharmaceutical Q&A.

This uses the SQLRetriever to query the structured pharma SQLite database and
feeds the resulting rows into the AnswerSynthesizer. It allows us to evaluate
how far a tables-only approach can go compared to text-only and hybrid RAG.
"""

import time
import logging

from app.evaluation.schemas import EvalQuestion, BaselineResponse
from app.evaluation.baselines.base import BaseSystem
from app.services.retrieval import SQLRetriever
from app.services.synthesis import AnswerSynthesizer
from app.models import RetrievalResult, QueryClassification
from app.services.cost_calculator import calculate_cost

logger = logging.getLogger(__name__)


class SQLRAGBaseline(BaseSystem):
    """SQL-only RAG baseline (SQL retrieval + Tier-2 synthesis)."""

    def __init__(self):
        self.retriever = SQLRetriever()
        self.synthesizer = AnswerSynthesizer()

    @property
    def name(self) -> str:
        return "SQL RAG (Tables Only)"

    async def answer(self, question: EvalQuestion) -> BaselineResponse:
        """
        Answer using SQL-based retrieval over pharma.db + Tier-2 synthesis.
        """
        start_time = time.time()

        sql_result, tier1_tokens = await self.retriever.retrieve(question.question)
        rows = sql_result.get("rows", [])
        sources = sql_result.get("sources", [])

        retrieval_result = RetrievalResult(
            text_chunks=[],
            sql_rows=rows,
            sources=sources,
        )

        # Synthesize answer using Tier-2 model; this is similar to the hybrid pipeline
        answer_text, synth_sources, tier2_tokens = await self.synthesizer.synthesize(
            question.question,
            retrieval_result,
        )

        all_sources = {s.source_id for s in sources}
        all_sources.update(s.source_id for s in synth_sources)

        latency_ms = (time.time() - start_time) * 1000
        predicted_label = self._extract_label(answer_text)

        cost = calculate_cost(
            tier1_tokens=tier1_tokens,
            tier2_tokens=tier2_tokens,
        )

        return BaselineResponse(
            answer=answer_text,
            predicted_label=predicted_label,
            latency_ms=latency_ms,
            tokens_used=tier1_tokens + tier2_tokens,
            cost_usd=cost.total_cost_usd,
            sources=list(all_sources),
        )

    def _extract_label(self, answer: str) -> str:
        """Extract yes/no/maybe label from answer text."""
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

