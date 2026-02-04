"""RAG Orchestrator that coordinates the full query pipeline."""

import logging
from typing import Optional

from app.models import QueryRequest, QueryResponse, QueryClassification, SourceReference
from app.services.classifier import QueryClassifier
from app.services.retrieval import HybridRetriever
from app.services.synthesis import AnswerSynthesizer
from app.services.cost_calculator import calculate_cost

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Orchestrates the complete RAG pipeline:
    1. Tier-1: Classify query (text/sql/hybrid)
    2. Retrieve context from Vector DB and/or SQL DB
    3. Tier-2: Synthesize final answer
    """
    
    def __init__(self):
        """Initialize the orchestrator with all required services."""
        self.classifier = QueryClassifier()
        self.retriever = HybridRetriever()
        self.synthesizer = AnswerSynthesizer()
    
    async def process_query(self, query: str) -> QueryResponse:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: The user's natural language query
            
        Returns:
            QueryResponse with answer, sources, and metadata
        """
        tier1_tokens = 0
        tier2_tokens = 0
        
        # Step 1: Classify the query using Tier-1 model
        logger.info(f"Processing query: {query[:100]}...")
        
        classification, t1_classify_tokens = await self.classifier.classify(query)
        tier1_tokens += t1_classify_tokens
        
        logger.info(f"Query classified as: {classification.query_type} (confidence: {classification.confidence})")
        
        # Step 2: Retrieve relevant context based on classification
        retrieval_result, t1_retrieve_tokens = await self.retriever.retrieve(query, classification)
        tier1_tokens += t1_retrieve_tokens
        
        logger.info(
            f"Retrieved: {len(retrieval_result.text_chunks)} text chunks, "
            f"{len(retrieval_result.sql_rows)} SQL rows"
        )
        
        # Step 3: Synthesize answer using Tier-2 model
        answer, sources, t2_tokens = await self.synthesizer.synthesize(query, retrieval_result)
        tier2_tokens += t2_tokens
        
        # Combine sources from retrieval and synthesis
        all_sources = self._merge_sources(retrieval_result.sources, sources)
        
        # Calculate cost
        cost = calculate_cost(tier1_tokens, tier2_tokens)
        
        # Build response
        response = QueryResponse(
            query=query,
            answer=answer,
            classification=classification,
            sources=all_sources,
            latency_ms=0,  # Will be set by the endpoint
            tier1_tokens=tier1_tokens,
            tier2_tokens=tier2_tokens,
            cost=cost,
        )
        
        logger.info(
            f"Response generated. Tier-1: {tier1_tokens} tokens (${cost.tier1_cost_usd:.6f}), "
            f"Tier-2: {tier2_tokens} tokens (${cost.tier2_cost_usd:.6f}), "
            f"Total: ${cost.total_cost_usd:.6f}"
        )
        
        return response
    
    def _merge_sources(
        self,
        retrieval_sources: list[SourceReference],
        synthesis_sources: list[SourceReference],
    ) -> list[SourceReference]:
        """
        Merge and deduplicate sources from retrieval and synthesis.
        
        Args:
            retrieval_sources: Sources from the retrieval step
            synthesis_sources: Sources identified during synthesis
            
        Returns:
            Merged list of unique sources
        """
        seen = set()
        merged = []
        
        for source in retrieval_sources + synthesis_sources:
            key = (source.source_type, source.source_id)
            if key not in seen:
                seen.add(key)
                merged.append(source)
        
        return merged
