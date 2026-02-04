"""BM25 keyword search baseline for evaluation."""

import time
import logging
from typing import Optional

from rank_bm25 import BM25Okapi

from app.evaluation.schemas import EvalQuestion, BaselineResponse
from app.evaluation.baselines.base import BaseSystem
from app.services.synthesis import AnswerSynthesizer
from app.services.cost_calculator import calculate_cost
from app.models import RetrievalResult, SourceReference

logger = logging.getLogger(__name__)


class BM25Baseline(BaseSystem):
    """
    BM25 keyword search baseline.
    
    Uses BM25 for retrieval instead of vector similarity,
    then passes to Tier-2 LLM for synthesis.
    """
    
    def __init__(
        self,
        documents: Optional[list[str]] = None,
        top_k: int = 5,
    ):
        """
        Initialize BM25 baseline.
        
        Args:
            documents: List of documents to index. If None, uses question context.
            top_k: Number of documents to retrieve
        """
        self.documents = documents or []
        self.top_k = top_k
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_docs: list[list[str]] = []
        self.synthesizer = AnswerSynthesizer()
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "BM25 Baseline"
    
    async def initialize(self) -> None:
        """Initialize the BM25 index if documents are provided."""
        if self.documents and not self._initialized:
            self._build_index(self.documents)
            self._initialized = True
    
    def _build_index(self, documents: list[str]) -> None:
        """Build BM25 index from documents."""
        self.documents = documents
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info(f"BM25 index built with {len(documents)} documents")
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        # Basic tokenization - could be improved with NLTK or spaCy
        return text.lower().split()
    
    def _search(self, query: str, top_k: int = None) -> list[tuple[str, float]]:
        """
        Search the BM25 index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.bm25:
            return []
        
        k = top_k or self.top_k
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], scores[idx]))
        
        return results
    
    async def answer(self, question: EvalQuestion) -> BaselineResponse:
        """
        Answer using BM25 retrieval + Tier-2 synthesis.
        
        For PubMedQA, we use the question's context as the document corpus.
        """
        start_time = time.time()
        
        # For PubMedQA, the context is the relevant abstract
        # We split it into paragraphs and use BM25 to find relevant parts
        context_parts = question.context.split("\n\n")
        context_parts = [p.strip() for p in context_parts if p.strip()]
        
        if not context_parts:
            context_parts = [question.context]
        
        # Build temporary BM25 index for this question's context
        if len(context_parts) > 1:
            self._build_index(context_parts)
            search_results = self._search(question.question, top_k=self.top_k)
            retrieved_context = "\n\n".join([doc for doc, _ in search_results])
        else:
            # If only one paragraph, use it directly
            retrieved_context = question.context
        
        # Create retrieval result for synthesizer
        retrieval_result = RetrievalResult(
            text_chunks=[{
                "text": retrieved_context,
                "metadata": {"source": "bm25_retrieval"}
            }],
            sql_rows=[],
            sources=[
                SourceReference(
                    source_type="text",
                    source_id="bm25_context",
                    title="BM25 Retrieved Context",
                    snippet=retrieved_context[:200] + "..." if len(retrieved_context) > 200 else retrieved_context
                )
            ]
        )
        
        # Synthesize answer using Tier-2 LLM
        answer_text, sources, tokens_used = await self.synthesizer.synthesize(
            question.question,
            retrieval_result
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract yes/no/maybe label from answer
        predicted_label = self._extract_label(answer_text)
        
        # Calculate cost (Tier-2 only for BM25)
        cost = calculate_cost(
            tier1_tokens=0,
            tier2_tokens=tokens_used
        )
        
        return BaselineResponse(
            answer=answer_text,
            predicted_label=predicted_label,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_usd=cost.total_cost_usd,
            sources=["bm25_retrieval"],
        )
    
    def _extract_label(self, answer: str) -> str:
        """Extract yes/no/maybe label from answer text."""
        answer_lower = answer.lower()
        
        # Look for explicit answers
        if "the answer is yes" in answer_lower or "yes," in answer_lower[:50]:
            return "yes"
        if "the answer is no" in answer_lower or "no," in answer_lower[:50]:
            return "no"
        if "the answer is maybe" in answer_lower or "maybe" in answer_lower[:50]:
            return "maybe"
        
        # Check for presence of keywords
        yes_count = answer_lower.count("yes")
        no_count = answer_lower.count("no")
        maybe_count = answer_lower.count("maybe") + answer_lower.count("uncertain") + answer_lower.count("unclear")
        
        # If clear majority, use that
        if yes_count > no_count and yes_count > maybe_count:
            return "yes"
        if no_count > yes_count and no_count > maybe_count:
            return "no"
        if maybe_count > 0:
            return "maybe"
        
        # Default to maybe if ambiguous
        return "maybe"
