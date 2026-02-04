"""FastAPI application entry point for Pharmaceutical RAG Agent."""

import time
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.config import settings
from app.models import QueryRequest, QueryResponse, IngestionResult

# Initialize FastAPI app
app = FastAPI(
    title="Pharmaceutical RAG Agent",
    description="Hybrid RAG system for answering pharmaceutical questions using text and table data",
    version="0.1.0",
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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "tier1_provider": settings.tier1_provider,
        "tier2_provider": settings.tier2_provider,
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
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


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
