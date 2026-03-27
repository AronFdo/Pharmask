"""BM25 retriever over ingested pharmaceutical text chunks.

This provides a lexical-retrieval counterpart to the existing vector-based
retriever, built on top of the same Chroma-backed corpus so that IDs and
metadata remain consistent across dense, hybrid, and BM25-only baselines.
"""

import logging
import time
from typing import Optional, List, Dict

from rank_bm25 import BM25Okapi

from app.db import VectorClient
from app.config import settings
from app.models import SourceReference

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Retriever for BM25 keyword search over pharmaceutical text chunks."""

    def __init__(self, top_k: int | None = None):
        """
        Initialize the BM25 retriever.

        Args:
            top_k: Number of results to return (default from settings)
        """
        self.client = VectorClient()
        self.top_k = top_k or settings.vector_top_k

        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        self._documents: List[str] = []
        self._metadatas: List[Dict] = []
        self._initialized = False

    async def initialize(self) -> None:
        """
        Lazily build the BM25 index from the existing vector store.

        This assumes that all relevant text chunks have already been ingested
        into Chroma via the ingestion pipeline.
        """
        if self._initialized:
            return

        start = time.time()
        total = self.client.get_document_count()
        if total == 0:
            logger.warning("BM25Retriever: vector store is empty; nothing to index")
            self._initialized = True
            return

        # For the current project scale (few thousand chunks), we can fetch
        # everything in a single call. If this grows substantially, we can
        # extend this to fetch in batches using offsets.
        results = self.client.get_all_documents(limit=None)

        self._doc_ids = results.get("ids", []) or []
        self._documents = results.get("documents", []) or []
        self._metadatas = results.get("metadatas", []) or []

        if not self._documents:
            logger.warning("BM25Retriever: no documents returned from vector store")
            self._initialized = True
            return

        tokenized_docs = [self._tokenize(d) for d in self._documents]
        self._bm25 = BM25Okapi(tokenized_docs)
        self._initialized = True

        logger.info(
            "BM25Retriever: built BM25 index over %d chunks in %.2fs",
            len(self._documents),
            time.time() - start,
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return (text or "").lower().split()

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> dict:
        """
        Retrieve relevant text chunks for a query using BM25.

        Args:
            query: The search query
            top_k: Optional override for number of results to return

        Returns:
            Dict with 'chunks' list and 'sources' list,
            mirroring the shape of VectorRetriever.
        """
        if not self._initialized:
            await self.initialize()

        if not self._bm25 or not self._documents:
            logger.warning("BM25Retriever: index not available; returning empty result")
            return {"chunks": [], "sources": []}

        k = top_k or self.top_k
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices by score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        chunks: List[Dict] = []
        sources: List[SourceReference] = []

        for rank, idx in enumerate(ranked_indices, start=1):
            score = scores[idx]
            if score <= 0:
                # Skip non-informative matches
                continue

            doc_id = self._doc_ids[idx] if idx < len(self._doc_ids) else str(idx)
            text = self._documents[idx]
            metadata = self._metadatas[idx] if idx < len(self._metadatas) else {}

            chunk = {
                "id": doc_id,
                "text": text,
                "metadata": metadata,
                "relevance_score": float(score),
                "rank": rank,
            }
            chunks.append(chunk)

            source = SourceReference(
                source_type="text",
                source_id=doc_id,
                title=metadata.get("title", ""),
                snippet=text[:200] + "..." if len(text) > 200 else text,
                metadata={
                    "section": metadata.get("section", ""),
                    "source_doc": metadata.get("source_doc", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "source_file": metadata.get("source_file", ""),
                    "retriever": "bm25",
                },
            )
            sources.append(source)

        logger.info("BM25Retriever: found %d chunks for query", len(chunks))

        return {
            "chunks": chunks,
            "sources": sources,
        }

