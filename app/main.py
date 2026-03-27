"""FastAPI application entry point for Pharmaceutical RAG Agent."""

import logging
import time
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.config import settings
from app.models import QueryRequest, QueryResponse, IngestionResult

logger = logging.getLogger(__name__)


def _check_api_keys() -> list[str]:
    """Return list of human-readable issues with API key configuration."""
    issues = []
    t1_key = settings.get_tier1_api_key()
    t2_key = settings.get_tier2_api_key()
    if not (t1_key and t1_key.strip()):
        issues.append(
            f"Tier-1 ({settings.tier1_provider}): no API key set. "
            f"Add {settings.tier1_provider.upper()}_API_KEY to .env (e.g. GROQ_API_KEY for groq)."
        )
    elif "your_" in t1_key.lower() or "here" in t1_key.lower():
        issues.append(
            f"Tier-1 ({settings.tier1_provider}): .env still has placeholder value. "
            f"Replace with a real API key from the provider's dashboard."
        )
    if not (t2_key and t2_key.strip()):
        issues.append(
            f"Tier-2 ({settings.tier2_provider}): no API key set. "
            f"Add {settings.tier2_provider.upper()}_API_KEY to .env (e.g. OPENAI_API_KEY for openai)."
        )
    elif "your_" in t2_key.lower() or "here" in t2_key.lower():
        issues.append(
            f"Tier-2 ({settings.tier2_provider}): .env still has placeholder value. "
            f"Replace with a real API key from the provider's dashboard."
        )
    return issues


# Initialize FastAPI app
app = FastAPI(
    title="Pharmaceutical RAG Agent",
    description="Hybrid RAG system for answering pharmaceutical questions using text and table data",
    version="0.1.0",
)


@app.on_event("startup")
def startup_validate_api_keys():
    """Log clear errors at startup if API keys are missing or placeholders."""
    issues = _check_api_keys()
    if issues:
        for msg in issues:
            logger.warning(msg)
        logger.warning(
            "API key issues detected. Queries will fail with 'invalid API key' until you fix .env. "
            "See README for setup: copy .env.example to .env and set real keys."
        )

# Setup templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the minimal UI for the prototype."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint. Includes API key config status (no keys are exposed)."""
    key_issues = _check_api_keys()
    return {
        "status": "healthy",
        "tier1_provider": settings.tier1_provider,
        "tier2_provider": settings.tier2_provider,
        "api_keys_ok": len(key_issues) == 0,
        "api_key_issues": key_issues if key_issues else None,
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a natural language query through the RAG pipeline.
    
    1. Tier-1: Classify query as text/sql/hybrid
    2. Retrieve relevant context from Vector DB and/or SQL DB
    3. Tier-2: Synthesize final answer with sources
    """
    start_time = time.time()
    
    try:
        # Import services here to avoid circular imports
        from app.services.orchestrator import RAGOrchestrator
        
        orchestrator = RAGOrchestrator()
        response = await orchestrator.process_query(request.query)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        response.latency_ms = latency_ms
        
        return response
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        detail = str(e)
        # Give a clearer hint when the provider reports invalid/missing API key
        if any(
            phrase in detail.lower()
            for phrase in ("invalid", "api key", "authentication", "incorrect api key", "invalid_api_key")
        ):
            key_issues = _check_api_keys()
            if key_issues:
                detail = f"{detail} — Config: {'; '.join(key_issues)}"
            else:
                detail = (
                    f"{detail} — If the key is correct, check for extra spaces/quotes in .env "
                    "or that the key is active in the provider dashboard."
                )
        raise HTTPException(status_code=500, detail=detail)


@app.post("/ingest", response_model=IngestionResult)
async def ingest_documents(data_dir: str = "./data/documents"):
    """
    Ingest documents from the specified directory.
    
    Processes XML/PDF files, extracts text and tables,
    loads text chunks to Vector DB and tables to SQL DB.
    """
    try:
        from app.services.ingestion.worker import IngestionWorker
        
        worker = IngestionWorker()
        result = await worker.ingest_directory(Path(data_dir))
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during ingestion: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.debug)
