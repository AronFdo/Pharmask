"""Direct LLM baseline (no retrieval) for evaluation."""

import time
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.evaluation.schemas import EvalQuestion, BaselineResponse
from app.evaluation.baselines.base import BaseSystem
from app.services.cost_calculator import calculate_cost, get_model_pricing

logger = logging.getLogger(__name__)


class DirectLLMBaseline(BaseSystem):
    """
    Direct LLM baseline without any retrieval.
    
    Sends the question directly to the Tier-2 LLM,
    relying only on the model's parametric knowledge.
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize Direct LLM baseline.
        
        Args:
            provider: LLM provider ('openai' or 'anthropic')
            model: Model name
            temperature: Generation temperature
        """
        self.provider = provider or settings.tier2_provider
        self.model = model or settings.tier2_model
        self.temperature = temperature
        self.llm = None
    
    @property
    def name(self) -> str:
        return "Direct LLM (No Retrieval)"
    
    async def initialize(self) -> None:
        """Initialize the LLM client."""
        if self.llm is not None:
            return
        
        if self.provider == "openai":
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=settings.openai_api_key,
            )
        elif self.provider == "anthropic":
            self.llm = ChatAnthropic(
                model=self.model,
                temperature=self.temperature,
                api_key=settings.anthropic_api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        logger.info(f"Direct LLM baseline initialized with {self.provider}/{self.model}")
    
    async def answer(self, question: EvalQuestion) -> BaselineResponse:
        """
        Answer using direct LLM without retrieval.
        
        Args:
            question: The evaluation question
            
        Returns:
            BaselineResponse with answer and metrics
        """
        if self.llm is None:
            await self.initialize()
        
        start_time = time.time()
        
        # Build prompt for PubMedQA task
        system_prompt = """You are a biomedical expert answering questions based on your knowledge.
For each question, provide a clear answer and conclude with either "yes", "no", or "maybe".

Format your response as:
1. Brief explanation of your reasoning
2. Final answer: [yes/no/maybe]"""

        user_prompt = f"""Question: {question.question}

Based on your biomedical knowledge, please answer this question.
Remember to conclude with a clear "yes", "no", or "maybe" answer."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            answer_text = response.content
            
            # Estimate tokens (rough approximation)
            input_tokens = len(system_prompt.split()) + len(user_prompt.split())
            output_tokens = len(answer_text.split())
            total_tokens = int((input_tokens + output_tokens) * 1.3)  # Rough token estimation
            
        except Exception as e:
            logger.error(f"Direct LLM error: {e}")
            answer_text = f"Error: {str(e)}"
            total_tokens = 0
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract label
        predicted_label = self._extract_label(answer_text)
        
        # Calculate cost (Tier-2 only, no retrieval)
        cost = calculate_cost(
            tier1_tokens=0,
            tier2_tokens=total_tokens
        )
        
        return BaselineResponse(
            answer=answer_text,
            predicted_label=predicted_label,
            latency_ms=latency_ms,
            tokens_used=total_tokens,
            cost_usd=cost.total_cost_usd,
            sources=[],  # No retrieval
        )
    
    def _extract_label(self, answer: str) -> str:
        """Extract yes/no/maybe label from answer text."""
        answer_lower = answer.lower()
        
        # Look for "Final answer: X" pattern
        if "final answer:" in answer_lower:
            final_part = answer_lower.split("final answer:")[-1].strip()[:20]
            if "yes" in final_part:
                return "yes"
            if "no" in final_part:
                return "no"
            if "maybe" in final_part:
                return "maybe"
        
        # Look for explicit patterns at the end
        last_100 = answer_lower[-100:]
        if "the answer is yes" in last_100 or "answer: yes" in last_100:
            return "yes"
        if "the answer is no" in last_100 or "answer: no" in last_100:
            return "no"
        if "the answer is maybe" in last_100 or "answer: maybe" in last_100:
            return "maybe"
        
        # Count occurrences
        yes_count = answer_lower.count(" yes")
        no_count = answer_lower.count(" no ")
        maybe_count = answer_lower.count("maybe") + answer_lower.count("uncertain")
        
        if yes_count > no_count and yes_count > maybe_count:
            return "yes"
        if no_count > yes_count and no_count > maybe_count:
            return "no"
        
        return "maybe"
