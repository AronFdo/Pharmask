"""Adapter to wrap the Hybrid RAG system for evaluation."""

import time
import logging

from app.evaluation.schemas import EvalQuestion, BaselineResponse
from app.evaluation.baselines.base import BaseSystem
from app.services.orchestrator import RAGOrchestrator

logger = logging.getLogger(__name__)


class HybridRAGAdapter(BaseSystem):
    """
    Adapter that wraps the main Hybrid RAG system for evaluation.
    
    This allows the existing RAG system to be tested alongside
    the baseline systems using the same evaluation framework.
    """
    
    def __init__(self):
        """Initialize the adapter."""
        self.orchestrator = None
    
    @property
    def name(self) -> str:
        return "Hybrid RAG"
    
    async def initialize(self) -> None:
        """Initialize the RAG orchestrator."""
        if self.orchestrator is None:
            self.orchestrator = RAGOrchestrator()
            logger.info("Hybrid RAG adapter initialized")
    
    async def answer(self, question: EvalQuestion) -> BaselineResponse:
        """
        Answer using the full Hybrid RAG pipeline.
        
        Args:
            question: The evaluation question
            
        Returns:
            BaselineResponse with answer and metrics
        """
        if self.orchestrator is None:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Use the full RAG pipeline
            response = await self.orchestrator.process_query(question.question)
            
            answer_text = response.answer
            tokens_used = response.tier1_tokens + response.tier2_tokens
            cost_usd = response.cost.total_cost_usd if response.cost else 0.0
            
            # Extract sources
            sources = [s.source_id for s in response.sources]
            
        except Exception as e:
            logger.error(f"Hybrid RAG error: {e}")
            answer_text = f"Error: {str(e)}"
            tokens_used = 0
            cost_usd = 0.0
            sources = []
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract label
        predicted_label = self._extract_label(answer_text)
        
        return BaselineResponse(
            answer=answer_text,
            predicted_label=predicted_label,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            sources=sources,
            raw_response={"classification": response.classification.query_type if hasattr(response, 'classification') else None}
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
        maybe_count = answer_lower.count("maybe") + answer_lower.count("uncertain") + answer_lower.count("unclear")
        
        if yes_count > no_count and yes_count > maybe_count:
            return "yes"
        if no_count > yes_count and no_count > maybe_count:
            return "no"
        
        return "maybe"
