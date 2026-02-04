"""Vector retrieval service for semantic search over text chunks."""

import logging
from typing import Optional

from app.db import VectorClient
from app.config import settings
from app.models import SourceReference

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retriever for semantic search over pharmaceutical text chunks."""
    
    def __init__(self, top_k: int = None):
        """
        Initialize the vector retriever.
        
        Args:
            top_k: Number of results to return (default from settings)
        """
        self.client = VectorClient()
        self.top_k = top_k or settings.vector_top_k
    
    async def retrieve(
        self,
        query: str,
        filter_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Retrieve relevant text chunks for a query.
        
        Args:
            query: The search query
            filter_metadata: Optional metadata filter
            
        Returns:
            Dict with 'chunks' list and 'sources' list
        """
        try:
            # Search the vector database
            results = self.client.search(
                query=query,
                n_results=self.top_k,
                where=filter_metadata,
            )
            
            # Format results
            chunks = []
            sources = []
            
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", []),
                results.get("distances", []),
            )):
                # Create chunk dict
                chunk = {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "relevance_score": 1 - distance if distance else 1.0,  # Convert distance to score
                    "rank": i + 1,
                }
                chunks.append(chunk)
                
                # Create source reference
                source = SourceReference(
                    source_type="text",
                    source_id=doc_id,
                    title=metadata.get("title", ""),
                    snippet=document[:200] + "..." if len(document) > 200 else document,
                    metadata={
                        "section": metadata.get("section", ""),
                        "source_doc": metadata.get("source_doc", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                    }
                )
                sources.append(source)
            
            logger.info(f"Vector retrieval found {len(chunks)} chunks for query")
            
            return {
                "chunks": chunks,
                "sources": sources,
            }
            
        except Exception as e:
            logger.error(f"Vector retrieval error: {e}")
            return {"chunks": [], "sources": []}
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store."""
        return self.client.get_document_count()
