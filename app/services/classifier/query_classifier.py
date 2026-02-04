"""Tier-1 Query Classifier using cloud-hosted LLM (Groq, Gemini, or OpenAI)."""

import json
import logging
from typing import Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.models import QueryClassification

logger = logging.getLogger(__name__)


# Classification prompt template
CLASSIFICATION_PROMPT = """You are a pharmaceutical query classifier. Your job is to classify user queries into one of three categories based on what type of data retrieval is needed.

## Categories:

1. **text**: The query requires searching through unstructured text (research papers, drug descriptions, clinical narratives). 
   - Examples: "What are the side effects of aspirin?", "Explain how beta blockers work", "What does the research say about metformin and diabetes?"

2. **sql**: The query requires precise factual lookup from structured data (drug names, dosages, NDC codes, specific values).
   - Examples: "What is the recommended dose of ibuprofen?", "List all formulations of lisinopril", "What drugs interact with warfarin?"

3. **hybrid**: The query requires both structured data AND unstructured text to fully answer.
   - Examples: "What is the standard dose of metformin and why?", "Compare the side effect profiles of different SSRIs", "What are the indications for atorvastatin and how effective is it?"

## Instructions:
- Analyze the query carefully
- Consider what type of data would best answer it
- If in doubt between text and hybrid, prefer hybrid
- If in doubt between sql and hybrid, prefer hybrid
- Pharmaceutical queries about mechanisms, research, or explanations are usually "text"
- Pharmaceutical queries about specific values, lists, or lookups are usually "sql"
- Pharmaceutical queries asking "why" or requiring context alongside data are usually "hybrid"

## Output Format:
Return a JSON object with these fields:
- "query_type": one of "text", "sql", or "hybrid"
- "confidence": a number between 0 and 1 indicating confidence
- "reasoning": a brief explanation of your classification

## Query to classify:
{query}

## Your classification (JSON only):"""


class QueryClassifier:
    """Tier-1 classifier for routing pharmaceutical queries."""
    
    def __init__(self):
        """Initialize the classifier with the configured Tier-1 model."""
        self._llm: Optional[BaseChatModel] = None
        self._prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
        self._parser = JsonOutputParser()
        self._token_count = 0
    
    def _get_llm(self) -> BaseChatModel:
        """Get or create the LLM instance based on configuration."""
        if self._llm is not None:
            return self._llm
        
        provider = settings.tier1_provider
        model = settings.tier1_model
        
        if provider == "groq":
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                api_key=settings.groq_api_key,
                model=model,
                temperature=0,
            )
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                google_api_key=settings.google_api_key,
                model=model,
                temperature=0,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=model,
                temperature=0,
            )
        else:
            raise ValueError(f"Unknown Tier-1 provider: {provider}")
        
        logger.info(f"Initialized Tier-1 classifier with {provider}/{model}")
        return self._llm
    
    async def classify(self, query: str) -> Tuple[QueryClassification, int]:
        """
        Classify a user query.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Tuple of (QueryClassification, tokens_used)
        """
        try:
            llm = self._get_llm()
            
            # Create the chain
            chain = self._prompt | llm
            
            # Run classification
            response = await chain.ainvoke({"query": query})
            
            # Parse the response
            content = response.content
            
            # Extract JSON from response
            classification_data = self._parse_json_response(content)
            
            # Estimate token usage (approximate)
            prompt_tokens = len(CLASSIFICATION_PROMPT.split()) + len(query.split())
            response_tokens = len(content.split()) if isinstance(content, str) else 50
            tokens_used = prompt_tokens + response_tokens
            
            # Create classification object
            classification = QueryClassification(
                query_type=classification_data.get("query_type", "hybrid"),
                confidence=float(classification_data.get("confidence", 0.8)),
                reasoning=classification_data.get("reasoning", ""),
            )
            
            logger.info(f"Classified query as '{classification.query_type}' (confidence: {classification.confidence})")
            
            return classification, tokens_used
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            # Default to hybrid on error (safest choice)
            return QueryClassification(
                query_type="hybrid",
                confidence=0.5,
                reasoning=f"Classification failed, defaulting to hybrid: {str(e)}",
            ), 0
    
    def _parse_json_response(self, content: str) -> dict:
        """Parse JSON from LLM response, handling various formats."""
        if not isinstance(content, str):
            content = str(content)
        
        # Try to extract JSON from the response
        content = content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (```json and ```)
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            content = content.strip()
        
        # Try to find JSON object
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to parse entire content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from response: {content[:200]}")
            return {"query_type": "hybrid", "confidence": 0.5, "reasoning": "Parse error"}
    
    def classify_sync(self, query: str) -> Tuple[QueryClassification, int]:
        """
        Synchronous version of classify for testing.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Tuple of (QueryClassification, tokens_used)
        """
        import asyncio
        return asyncio.run(self.classify(query))
