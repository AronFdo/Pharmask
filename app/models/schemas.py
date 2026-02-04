"""Pydantic schemas for the Pharmaceutical RAG Agent."""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for user queries."""
    query: str = Field(..., description="Natural language query about pharmaceutical data")


class QueryClassification(BaseModel):
    """Classification result from Tier-1 model."""
    query_type: Literal["text", "sql", "hybrid"] = Field(
        ..., description="Type of query: text-only, sql-only, or hybrid"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score of classification"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Brief explanation of classification"
    )


class SourceReference(BaseModel):
    """Reference to a source document or table."""
    source_type: Literal["text", "table"] = Field(..., description="Type of source")
    source_id: str = Field(..., description="Unique identifier of the source")
    title: Optional[str] = Field(default=None, description="Title or name of the source")
    snippet: Optional[str] = Field(default=None, description="Relevant snippet from source")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


class RetrievalResult(BaseModel):
    """Result from retrieval engines."""
    text_chunks: list[dict] = Field(default_factory=list, description="Retrieved text chunks")
    sql_rows: list[dict] = Field(default_factory=list, description="Retrieved SQL rows")
    sources: list[SourceReference] = Field(default_factory=list, description="Source references")


class CostBreakdown(BaseModel):
    """Cost breakdown for a query."""
    tier1_tokens: int = Field(default=0, description="Tokens used by Tier-1 model")
    tier2_tokens: int = Field(default=0, description="Tokens used by Tier-2 model")
    tier1_cost_usd: float = Field(default=0.0, description="Cost of Tier-1 in USD")
    tier2_cost_usd: float = Field(default=0.0, description="Cost of Tier-2 in USD")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    tier1_model: str = Field(default="", description="Tier-1 model used")
    tier2_model: str = Field(default="", description="Tier-2 model used")
    savings_vs_tier2_only: float = Field(default=0.0, description="Savings compared to using Tier-2 for everything")


class QueryResponse(BaseModel):
    """Response model for user queries."""
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Synthesized answer")
    classification: QueryClassification = Field(..., description="Query classification")
    sources: list[SourceReference] = Field(default_factory=list, description="Source references")
    latency_ms: float = Field(..., description="Total processing time in milliseconds")
    tier1_tokens: int = Field(default=0, description="Tokens used by Tier-1 model")
    tier2_tokens: int = Field(default=0, description="Tokens used by Tier-2 model")
    cost: Optional[CostBreakdown] = Field(default=None, description="Cost breakdown for this query")


class IngestionResult(BaseModel):
    """Result from document ingestion."""
    documents_processed: int = Field(default=0, description="Number of documents processed")
    text_chunks_created: int = Field(default=0, description="Number of text chunks created")
    tables_extracted: int = Field(default=0, description="Number of tables extracted")
    errors: list[str] = Field(default_factory=list, description="Any errors during ingestion")
    timestamp: datetime = Field(default_factory=datetime.now, description="Ingestion timestamp")
