"""ChromaDB client for vector storage and retrieval."""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional
from pathlib import Path

from app.config import settings


class VectorClient:
    """Client for ChromaDB vector database operations."""
    
    _instance: Optional["VectorClient"] = None
    _client: Optional[chromadb.ClientAPI] = None
    _collection: Optional[chromadb.Collection] = None
    
    COLLECTION_NAME = "pharmaceutical_texts"
    
    def __new__(cls):
        """Singleton pattern for vector client."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ChromaDB client."""
        if self._client is None:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the ChromaDB client and collection."""
        # Ensure persist directory exists
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Pharmaceutical text chunks from PMC-OA and DailyMed"}
        )
    
    @property
    def collection(self) -> chromadb.Collection:
        """Get the ChromaDB collection."""
        if self._collection is None:
            self._initialize_client()
        return self._collection
    
    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of text chunks to embed and store
            metadatas: List of metadata dicts for each document
            ids: List of unique IDs for each document
        """
        if not documents:
            return
            
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Search for similar documents.
        
        Args:
            query: Query text to search for
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'distances'
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        
        # Flatten results (query returns nested lists)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }
    
    def get_document_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def delete_all(self) -> None:
        """Delete all documents from the collection."""
        # Get all IDs and delete them
        all_docs = self.collection.get()
        if all_docs["ids"]:
            self.collection.delete(ids=all_docs["ids"])
    
    def reset(self) -> None:
        """Reset the collection (delete and recreate)."""
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Pharmaceutical text chunks from PMC-OA and DailyMed"}
        )
