"""Context-Provided RAG baseline for fair evaluation.

This baseline uses the provided PubMedQA context (like BM25 does)
but processes it through the RAG synthesis pipeline.
This creates a fair comparison of synthesis quality.
"""

import time
import logging

from app.evaluation.schemas import EvalQuestion, BaselineResponse
from app.evaluation.baselines.base import BaseSystem
from app.services.synthesis import AnswerSynthesizer
from app.services.classifier import QueryClassifier
from app.services.cost_calculator import calculate_cost
from app.models import RetrievalResult, SourceReference

logger = logging.getLogger(__name__)


class ContextProvidedRAG(BaseSystem):
    """
    RAG system with context provided (not retrieved).
    
    Uses the same context as BM25 baseline for fair comparison,
    but processes through Tier-1 classification and Tier-2 synthesis.
    """
    
    def __init__(self):
        """Initialize the adapter."""
        self.classifier = None
        self.synthesizer = None
    
    @property
    def name(self) -> str:
        return "Context-Provided RAG"
    
    async def initialize(self) -> None:
        """Initialize components."""
        if self.classifier is None:
            self.classifier = QueryClassifier()
        if self.synthesizer is None:
            self.synthesizer = AnswerSynthesizer()
        logger.info("Context-Provided RAG initialized")
    
    async def answer(self, question: EvalQuestion) -> BaselineResponse:
        """
        Answer using provided context + RAG synthesis.
        
        Args:
            question: The evaluation question (includes context)
            
        Returns:
            BaselineResponse with answer and metrics
        """
        if self.synthesizer is None:
            await self.initialize()
        
        start_time = time.time()
        tier1_tokens = 0
        tier2_tokens = 0
        
        try:
            # Step 1: Classify the query (still use Tier-1)
            classification, t1_tokens = await self.classifier.classify(question.question)
            tier1_tokens += t1_tokens
            
            # Step 2: Use the provided context (same as BM25)
            # This makes the comparison fair
            context = question.context
            
            # Create retrieval result with provided context
            retrieval_result = RetrievalResult(
                text_chunks=[{
                    "text": context,
                    "metadata": {
                        "source": "pubmedqa_provided",
                        "pubmed_id": question.id,
                    }
                }],
                sql_rows=[],
                sources=[
                    SourceReference(
                        source_type="text",
                        source_id=f"pubmedqa_{question.id}",
                        title="PubMedQA Abstract",
                        snippet=context[:200] + "..." if len(context) > 200 else context
                    )
                ]
            )
            
            # Step 3: Synthesize answer using Tier-2
            answer_text, sources, t2_tokens = await self.synthesizer.synthesize(
                question.question,
                retrieval_result
            )
            tier2_tokens += t2_tokens
            
            # Extract sources
            source_ids = [s.source_id for s in sources]
            
        except Exception as e:
            logger.error(f"Context-Provided RAG error: {e}")
            answer_text = f"Error: {str(e)}"
            source_ids = []
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract label
        predicted_label = self._extract_label(answer_text)
        
        # Calculate cost
        cost = calculate_cost(tier1_tokens, tier2_tokens)
        
        return BaselineResponse(
            answer=answer_text,
            predicted_label=predicted_label,
            latency_ms=latency_ms,
            tokens_used=tier1_tokens + tier2_tokens,
            cost_usd=cost.total_cost_usd,
            sources=source_ids,
            raw_response={"classification": classification.query_type}
        )
    
    def _extract_label(self, answer: str) -> str:
        """Extract yes/no/maybe label from answer text."""
        answer_lower = answer.lower()
        
        # Look for explicit answers
        if "final answer:" in answer_lower:
            final_part = answer_lower.split("final answer:")[-1].strip()[:20]
            if "yes" in final_part:
                return "yes"
            if "no" in final_part:
                return "no"
            if "maybe" in final_part:
                return "maybe"
        
        # Check first sentence
        first_sentence = answer_lower.split('.')[0]
        if "yes" in first_sentence and "no" not in first_sentence:
            return "yes"
        if "no" in first_sentence and "yes" not in first_sentence:
            return "no"
        
        # Count occurrences
        yes_count = answer_lower.count(" yes")
        no_count = answer_lower.count(" no ")
        maybe_count = answer_lower.count("maybe") + answer_lower.count("uncertain")
        
        if yes_count > no_count and yes_count > maybe_count:
            return "yes"
        if no_count > yes_count and no_count > maybe_count:
            return "no"
        
        return "maybe"
