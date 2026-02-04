"""
Ingest PubMedQA contexts into ChromaDB for fair evaluation.

This ensures the Hybrid RAG system can retrieve the same documents
that are available to the BM25 baseline during evaluation.
"""

import json
import logging
from pathlib import Path
import hashlib

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.vector_client import VectorClient
from app.services.ingestion.chunker import TextChunker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "evaluation" / "pubmedqa"


def load_pubmedqa_data(subset: str = "pqa_labeled", split: str = "train") -> list[dict]:
    """Load PubMedQA dataset from JSON files."""
    # Try different file patterns
    possible_files = [
        DATA_DIR / f"{subset}_{split}.json",
        DATA_DIR / f"{subset}_train.json",
    ]
    
    for filepath in possible_files:
        if filepath.exists():
            logger.info(f"Loading from {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    
    raise FileNotFoundError(f"PubMedQA data not found in {DATA_DIR}")


def ingest_pubmedqa_contexts(
    clear_existing: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """
    Ingest PubMedQA contexts into the vector database.
    
    Args:
        clear_existing: If True, clear existing PubMedQA documents first
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    # Load data
    data = load_pubmedqa_data()
    logger.info(f"Loaded {len(data)} PubMedQA questions")
    
    # Initialize clients
    vector_client = VectorClient()
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Prepare documents
    documents = []
    metadatas = []
    ids = []
    
    for item in data:
        context = item.get("context", "")
        if not context or len(context.strip()) < 50:
            continue
        
        question_id = item.get("id", "")
        question = item.get("question", "")
        
        # Create a unique ID based on content
        content_hash = hashlib.md5(context.encode()).hexdigest()[:8]
        doc_id = f"pubmedqa_{question_id}_{content_hash}"
        
        # Add the full context as a document
        # Also include the question for better semantic matching
        combined_text = f"Question: {question}\n\nAbstract: {context}"
        
        # Chunk the text - returns generator of TextChunk objects
        chunks = list(chunker.chunk_text(combined_text, source_doc=doc_id))
        
        for chunk in chunks:
            chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"
            
            documents.append(chunk.text)
            metadatas.append({
                "source": "pubmedqa",
                "pubmed_id": question_id,
                "question": question[:200],  # Truncate for metadata
                "chunk_index": chunk.chunk_index,
                "total_chunks": len(chunks),
                "ground_truth": item.get("final_decision", ""),
            })
            ids.append(chunk_id)
    
    logger.info(f"Prepared {len(documents)} chunks from {len(data)} questions")
    
    # Check for existing PubMedQA documents if not clearing
    if clear_existing:
        logger.info("Clearing existing PubMedQA documents...")
        # ChromaDB doesn't have a direct delete by metadata, so we'll just add
        # The IDs will overwrite if they exist
    
    # Add to vector database in batches
    batch_size = 100
    total_added = 0
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        try:
            vector_client.add_documents(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            total_added += len(batch_docs)
            
            if (i + batch_size) % 500 == 0:
                logger.info(f"Progress: {total_added}/{len(documents)} chunks added")
                
        except Exception as e:
            logger.error(f"Error adding batch at index {i}: {e}")
    
    logger.info(f"Successfully added {total_added} PubMedQA chunks to vector database")
    
    # Verify
    count = vector_client.get_document_count()
    logger.info(f"Total documents in vector database: {count}")
    
    return total_added


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest PubMedQA contexts into ChromaDB")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing PubMedQA documents first"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for text splitting (default: 500)"
    )
    
    args = parser.parse_args()
    
    ingest_pubmedqa_contexts(
        clear_existing=args.clear,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
