"""Tier-2 Answer Synthesizer using GPT-4o or Claude for final answer generation."""

import json
import logging
from typing import Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.models import RetrievalResult, SourceReference

logger = logging.getLogger(__name__)


# Synthesis prompt template
SYNTHESIS_PROMPT = """You are a pharmaceutical expert assistant. Your task is to synthesize a comprehensive, accurate answer to the user's question using the provided context.

## Context:

### Text Context (from research papers and drug documentation):
{text_context}

### Structured Data (from database):
{sql_context}

## Guidelines:
1. Answer the question directly and comprehensively
2. Use ONLY information from the provided context - do not make up facts
3. If the context doesn't contain enough information, say so clearly
4. Cite your sources using the format [Source: <source_id>] after relevant facts
5. For numerical data (dosages, frequencies), be precise and cite the source
6. If there are conflicting data points, mention both and note the discrepancy
7. Use clear, professional medical language
8. Structure your answer with appropriate formatting (paragraphs, lists) for readability

## User Question:
{query}

## Your Answer:"""


class AnswerSynthesizer:
    """Tier-2 synthesizer for generating comprehensive answers using GPT-4o or Claude."""
    
    def __init__(self):
        """Initialize the answer synthesizer."""
        self._llm: Optional[BaseChatModel] = None
        self._prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
    
    def _get_llm(self) -> BaseChatModel:
        """Get or create the LLM instance for Tier-2 synthesis."""
        if self._llm is not None:
            return self._llm
        
        provider = settings.tier2_provider
        model = settings.tier2_model
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=model,
                temperature=0.2,  # Slight creativity for natural responses
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(
                api_key=settings.anthropic_api_key,
                model=model,
                temperature=0.2,
            )
        else:
            raise ValueError(f"Unknown Tier-2 provider: {provider}")
        
        logger.info(f"Initialized Tier-2 synthesizer with {provider}/{model}")
        return self._llm
    
    async def synthesize(
        self,
        query: str,
        retrieval_result: RetrievalResult,
    ) -> Tuple[str, list[SourceReference], int]:
        """
        Synthesize an answer from retrieved context.
        
        Args:
            query: The user's original question
            retrieval_result: Retrieved text chunks and SQL rows
            
        Returns:
            Tuple of (answer string, sources list, tokens_used)
        """
        try:
            # Format context for the prompt
            text_context = self._format_text_context(retrieval_result.text_chunks)
            sql_context = self._format_sql_context(retrieval_result.sql_rows)
            
            # Handle empty context
            if not text_context and not sql_context:
                return (
                    "I don't have enough information in my knowledge base to answer this question. "
                    "Please try rephrasing your question or ask about a different pharmaceutical topic.",
                    [],
                    0
                )
            
            # Get LLM and create chain
            llm = self._get_llm()
            chain = self._prompt | llm
            
            # Generate answer
            response = await chain.ainvoke({
                "query": query,
                "text_context": text_context or "No text context available.",
                "sql_context": sql_context or "No structured data available.",
            })
            
            answer = response.content
            
            # Estimate token usage
            prompt_tokens = (
                len(SYNTHESIS_PROMPT.split()) +
                len(query.split()) +
                len(text_context.split()) +
                len(sql_context.split())
            )
            response_tokens = len(answer.split())
            tokens_used = prompt_tokens + response_tokens
            
            # Extract and enhance sources from the answer
            sources = self._enhance_sources(answer, retrieval_result.sources)
            
            logger.info(f"Synthesized answer with {len(sources)} sources, ~{tokens_used} tokens")
            
            return answer, sources, tokens_used
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return (
                f"I encountered an error while generating your answer: {str(e)}. "
                "Please try again.",
                [],
                0
            )
    
    def _format_text_context(self, chunks: list[dict]) -> str:
        """Format text chunks for the prompt."""
        if not chunks:
            return ""
        
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            source_doc = metadata.get("source_doc", "unknown")
            section = metadata.get("section", "")
            
            header = f"[Source {i}: {source_doc}"
            if section:
                header += f", Section: {section}"
            header += "]"
            
            formatted.append(f"{header}\n{text}")
        
        return "\n\n---\n\n".join(formatted)
    
    def _format_sql_context(self, rows: list[dict]) -> str:
        """Format SQL rows for the prompt."""
        if not rows:
            return ""
        
        # Group rows by source if possible
        formatted = []
        for i, row in enumerate(rows, 1):
            source = row.get("source_doc", "database")
            
            # Format row as key-value pairs
            row_str = ", ".join(
                f"{k}: {v}" for k, v in row.items()
                if v and k != "id"  # Skip empty values and IDs
            )
            
            formatted.append(f"[DB Record {i}, Source: {source}]\n{row_str}")
        
        return "\n\n".join(formatted)
    
    def _enhance_sources(
        self,
        answer: str,
        original_sources: list[SourceReference],
    ) -> list[SourceReference]:
        """
        Enhance source list based on which sources were actually cited in the answer.
        
        Args:
            answer: The generated answer text
            original_sources: Original sources from retrieval
            
        Returns:
            Filtered/enhanced list of sources that were used
        """
        # For now, return all sources
        # In a more sophisticated version, we could parse the answer
        # to find which sources were actually cited
        return original_sources
