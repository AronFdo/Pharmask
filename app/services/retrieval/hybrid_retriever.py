"""Hybrid retrieval combining Vector and SQL retrieval based on query classification."""

import logging
from typing import Tuple

from app.models import QueryClassification, RetrievalResult, SourceReference
from .vector_retriever import VectorRetriever
from .sql_retriever import SQLRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever that routes queries to Vector and/or SQL retrieval
    based on the Tier-1 classification.
    """
    
    def __init__(self):
        """Initialize hybrid retriever with both retrieval engines."""
        self.vector_retriever = VectorRetriever()
        self.sql_retriever = SQLRetriever()
    
    async def retrieve(
        self,
        query: str,
        classification: QueryClassification,
    ) -> Tuple[RetrievalResult, int]:
        """
        Retrieve relevant context based on query classification.
        
        Args:
            query: The user's natural language query
            classification: The Tier-1 classification result
            
        Returns:
            Tuple of (RetrievalResult, tokens_used)
        """
        text_chunks = []
        sql_rows = []
        sources = []
        tokens_used = 0
        
        query_type = classification.query_type
        
        if query_type == "text":
            # Text-only retrieval
            vector_result = await self.vector_retriever.retrieve(query)
            text_chunks = vector_result.get("chunks", [])
            sources.extend(vector_result.get("sources", []))
            
        elif query_type == "sql":
            # SQL-only retrieval
            sql_result, tokens = await self.sql_retriever.retrieve(query)
            sql_rows = sql_result.get("rows", [])
            sources.extend(sql_result.get("sources", []))
            tokens_used += tokens
            
        else:  # hybrid
            # Both retrieval types
            vector_result = await self.vector_retriever.retrieve(query)
            text_chunks = vector_result.get("chunks", [])
            sources.extend(vector_result.get("sources", []))
            
            sql_result, tokens = await self.sql_retriever.retrieve(query)
            sql_rows = sql_result.get("rows", [])
            sources.extend(sql_result.get("sources", []))
            tokens_used += tokens
        
        # Deduplicate sources
        sources = self._deduplicate_sources(sources)
        
        result = RetrievalResult(
            text_chunks=text_chunks,
            sql_rows=sql_rows,
            sources=sources,
        )
        
        logger.info(
            f"Hybrid retrieval ({query_type}): "
            f"{len(text_chunks)} text chunks, {len(sql_rows)} SQL rows"
        )
        
        return result, tokens_used
    
    def _deduplicate_sources(self, sources: list[SourceReference]) -> list[SourceReference]:
        """Remove duplicate source references."""
        seen = set()
        unique = []
        
        for source in sources:
            key = (source.source_type, source.source_id)
            if key not in seen:
                seen.add(key)
                unique.append(source)
        
        return unique
