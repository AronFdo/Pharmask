"""Base class for evaluation systems."""

from abc import ABC, abstractmethod
from app.evaluation.schemas import EvalQuestion, BaselineResponse


class BaseSystem(ABC):
    """
    Abstract base class for all systems under evaluation.
    
    All systems (Hybrid RAG, BM25, Direct LLM) must implement this interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the system name for reporting."""
        pass
    
    @abstractmethod
    async def answer(self, question: EvalQuestion) -> BaselineResponse:
        """
        Generate an answer for the given question.
        
        Args:
            question: The evaluation question to answer
            
        Returns:
            BaselineResponse with answer, predicted label, and metrics
        """
        pass
    
    async def initialize(self) -> None:
        """
        Optional initialization hook.
        Override if the system needs setup before answering questions.
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Optional cleanup hook.
        Override if the system needs cleanup after evaluation.
        """
        pass
