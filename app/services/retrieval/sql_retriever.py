"""SQL retrieval service for structured pharmaceutical data."""

import logging
import re
from typing import Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

from app.db import SQLClient
from app.config import settings
from app.models import SourceReference

logger = logging.getLogger(__name__)


# SQL generation prompt
SQL_GENERATION_PROMPT = """You are a SQL query generator for a pharmaceutical database. Generate a safe, read-only SQL query to answer the user's question.

## Available Tables and Columns:

1. **drugs** - Drug information
   - id (TEXT, primary key)
   - name (TEXT)
   - generic_name (TEXT)
   - brand_name (TEXT)
   - manufacturer (TEXT)
   - ndc_code (TEXT)
   - description (TEXT)
   - source_doc (TEXT)

2. **indications** - Drug indications/uses
   - id (INTEGER, primary key)
   - drug_id (TEXT, foreign key to drugs.id)
   - indication (TEXT)
   - source_doc (TEXT)

3. **dosages** - Drug dosage information
   - id (INTEGER, primary key)
   - drug_id (TEXT, foreign key to drugs.id)
   - form (TEXT) - e.g., tablet, capsule, injection
   - strength (TEXT) - e.g., 10mg, 500mg
   - route (TEXT) - e.g., oral, IV, topical
   - frequency (TEXT)
   - source_doc (TEXT)

4. **adverse_reactions** - Side effects and adverse reactions
   - id (INTEGER, primary key)
   - drug_id (TEXT, foreign key to drugs.id)
   - reaction (TEXT)
   - frequency (TEXT)
   - severity (TEXT)
   - source_doc (TEXT)

5. **interactions** - Drug interactions
   - id (INTEGER, primary key)
   - drug_id (TEXT, foreign key to drugs.id)
   - interacting_drug (TEXT)
   - interaction_type (TEXT)
   - description (TEXT)
   - severity (TEXT)
   - source_doc (TEXT)

## Rules:
1. Generate ONLY SELECT statements (read-only)
2. Use LIKE with % wildcards for text matching (case-insensitive searches)
3. Limit results to 20 rows maximum
4. Join tables when needed to get complete information
5. Always select source_doc for traceability
6. If the query cannot be answered from the database, return "NO_QUERY"

## User Question:
{query}

## SQL Query (just the query, no explanation):"""


class SQLRetriever:
    """Retriever for structured SQL queries over pharmaceutical data."""
    
    def __init__(self):
        """Initialize the SQL retriever."""
        self.client = SQLClient()
        self._llm: Optional[BaseChatModel] = None
        self._prompt = ChatPromptTemplate.from_template(SQL_GENERATION_PROMPT)
    
    def _get_llm(self) -> BaseChatModel:
        """Get or create the LLM for SQL generation (uses Tier-1 model)."""
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
        
        return self._llm
    
    async def retrieve(self, query: str) -> Tuple[dict, int]:
        """
        Retrieve relevant rows for a query.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Tuple of (result dict with 'rows' and 'sources', tokens_used)
        """
        try:
            # Generate SQL query
            sql_query, tokens = await self._generate_sql(query)
            
            if not sql_query or sql_query == "NO_QUERY":
                logger.info("No SQL query generated for this question")
                return {"rows": [], "sources": [], "sql_query": None}, tokens
            
            # Validate and execute query
            if not self._is_safe_query(sql_query):
                logger.warning(f"Unsafe SQL query rejected: {sql_query}")
                return {"rows": [], "sources": [], "sql_query": None}, tokens
            
            # Execute the query
            rows = self.client.execute_query(sql_query)
            
            # Format results
            sources = self._extract_sources(rows)
            
            logger.info(f"SQL retrieval found {len(rows)} rows")
            
            return {
                "rows": rows,
                "sources": sources,
                "sql_query": sql_query,
            }, tokens
            
        except Exception as e:
            logger.error(f"SQL retrieval error: {e}")
            return {"rows": [], "sources": [], "sql_query": None}, 0
    
    async def _generate_sql(self, query: str) -> Tuple[str, int]:
        """
        Generate SQL query from natural language.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (SQL query string, tokens_used)
        """
        try:
            llm = self._get_llm()
            chain = self._prompt | llm
            
            response = await chain.ainvoke({"query": query})
            content = response.content.strip()
            
            # Extract SQL from response
            sql_query = self._extract_sql(content)
            
            # Estimate tokens
            tokens = len(SQL_GENERATION_PROMPT.split()) + len(query.split()) + len(content.split())
            
            return sql_query, tokens
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return "", 0
    
    def _extract_sql(self, content: str) -> str:
        """Extract SQL query from LLM response."""
        content = content.strip()
        
        # Check for NO_QUERY
        if "NO_QUERY" in content.upper():
            return "NO_QUERY"
        
        # Remove markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            content = content.strip()
        
        # Remove "sql" language tag if present
        if content.lower().startswith("sql"):
            content = content[3:].strip()
        
        # Find SELECT statement
        match = re.search(r"(SELECT\s+.+?;?)\s*$", content, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1).strip()
            # Ensure it ends with semicolon or remove trailing one for execution
            return sql.rstrip(";")
        
        # If content looks like SQL, return it
        if content.upper().startswith("SELECT"):
            return content.rstrip(";")
        
        return ""
    
    def _is_safe_query(self, sql_query: str) -> bool:
        """
        Validate that a SQL query is safe to execute.
        
        Args:
            sql_query: The SQL query to validate
            
        Returns:
            True if safe, False otherwise
        """
        sql_upper = sql_query.upper()
        
        # Must be a SELECT query
        if not sql_upper.strip().startswith("SELECT"):
            return False
        
        # Reject dangerous keywords
        dangerous_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
            "TRUNCATE", "EXEC", "EXECUTE", "GRANT", "REVOKE",
            "UNION", "--", "/*", "*/"
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False
        
        return True
    
    def _extract_sources(self, rows: list[dict]) -> list[SourceReference]:
        """Extract source references from result rows."""
        sources = []
        seen_sources = set()
        
        for row in rows:
            source_doc = row.get("source_doc", "")
            if source_doc and source_doc not in seen_sources:
                seen_sources.add(source_doc)
                
                # Determine source type based on table
                source = SourceReference(
                    source_type="table",
                    source_id=source_doc,
                    title=row.get("name", row.get("drug_id", "")),
                    snippet=str(row)[:200],
                    metadata={"table": self._infer_table_from_row(row)},
                )
                sources.append(source)
        
        return sources
    
    def _infer_table_from_row(self, row: dict) -> str:
        """Infer which table a row came from based on its columns."""
        if "reaction" in row:
            return "adverse_reactions"
        elif "interacting_drug" in row:
            return "interactions"
        elif "indication" in row:
            return "indications"
        elif "form" in row or "strength" in row:
            return "dosages"
        else:
            return "drugs"
    
    def get_table_schemas(self) -> dict:
        """Get schemas for all tables."""
        tables = self.client.get_all_tables()
        schemas = {}
        for table in tables:
            schemas[table] = self.client.get_table_schema(table)
        return schemas
