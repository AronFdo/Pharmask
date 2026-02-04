# Pharmaceutical RAG Agent

A hybrid Retrieval-Augmented Generation (RAG) system for answering pharmaceutical questions using both unstructured text and structured table data.

## Features

- **Hybrid Data Ingestion**: Process XML (PMC-OA, DailyMed) and PDF pharmaceutical documents
- **Two-Tier Model Architecture**:
  - **Tier-1**: Cheap/free cloud-hosted model (Groq, Gemini, gpt-4o-mini) for query classification
  - **Tier-2**: Powerful model (GPT-4o, Claude 3.5 Sonnet) for answer synthesis
- **Dual Retrieval**:
  - Vector search for unstructured text (ChromaDB)
  - SQL queries for structured data (SQLite)
- **Query Classification**: Automatically routes queries to appropriate retrieval engine(s)
- **Source Citations**: Every answer includes traceable source references
- **Minimal UI**: Simple web interface for querying

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Tier-1: Query Classifier                        │
│              (Groq / Gemini / gpt-4o-mini)                       │
│                                                                  │
│     Classifies query as: "text" | "sql" | "hybrid"              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │  Text    │   │   SQL    │   │    Both      │
    │  Only    │   │   Only   │   │   (Hybrid)   │
    └────┬─────┘   └────┬─────┘   └──────┬───────┘
         │              │                │
         ▼              ▼                ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ Vector Search  │ │  SQL Query     │ │ Vector + SQL   │
│ (ChromaDB)     │ │  (SQLite)      │ │   Combined     │
└────────┬───────┘ └────────┬───────┘ └────────┬───────┘
         │              │                │
         └──────────────┴────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Tier-2: Answer Synthesis                        │
│                (GPT-4o / Claude 3.5 Sonnet)                      │
│                                                                  │
│     Synthesizes final answer with source citations              │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Response + Sources                           │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone and Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Tier-1: Choose one (Groq recommended - free tier)
GROQ_API_KEY=your_groq_api_key

# Tier-2: For answer synthesis
OPENAI_API_KEY=your_openai_api_key

# Model configuration
TIER1_PROVIDER=groq
TIER1_MODEL=llama-3.1-8b-instant
TIER2_PROVIDER=openai
TIER2_MODEL=gpt-4o
```

### 3. Download and Ingest Data

#### Option A: Quick Setup (Recommended)

Download PMC-OA dataset from Hugging Face and run ingestion:

```bash
# Download 100 PMC articles (for quick testing)
python scripts/download_pmc_dataset.py --limit 100

# Or download more (1000 articles)
python scripts/download_pmc_dataset.py --limit 1000

# Or download all (WARNING: very large dataset, ~3M articles)
python scripts/download_pmc_dataset.py
```

#### Option B: Full Setup with DailyMed

If you have downloaded DailyMed data separately:

```bash
# Setup with both PMC and DailyMed
python scripts/setup_data.py --pmc-limit 500 --dailymed-path /path/to/dailymed/download

# Or just DailyMed (skip PMC)
python scripts/setup_data.py --skip-pmc --dailymed-path /path/to/dailymed/download
```

#### Option C: Manual Ingestion

If you already have XML/PDF files:

```bash
# Place files in data/documents/
mkdir -p data/documents
# Copy your files there...

# Then run ingestion via API
curl -X POST "http://localhost:8000/ingest?data_dir=./data/documents"
```

#### Using PMC-OA Dataset in Python

```python
from datasets import load_dataset

# Load the dataset (commercial subset)
ds = load_dataset("TomTBT/pmc_open_access_xml", "commercial", streaming=True)

# Access articles
for article in ds["train"]:
    pmcid = article["pmcid"]
    xml_content = article["xml"]
    # Process...
```

### 4. Run the Application

```bash
# Start the server
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

## API Endpoints

### POST /query
Process a pharmaceutical query.

**Request:**
```json
{
  "query": "What are the side effects of aspirin?"
}
```

**Response:**
```json
{
  "query": "What are the side effects of aspirin?",
  "answer": "Aspirin can cause several side effects...",
  "classification": {
    "query_type": "text",
    "confidence": 0.92,
    "reasoning": "Query asks about side effects, which requires text search"
  },
  "sources": [
    {
      "source_type": "text",
      "source_id": "PMC12345",
      "title": "Aspirin Safety Profile",
      "snippet": "Common side effects include..."
    }
  ],
  "latency_ms": 2340.5,
  "tier1_tokens": 150,
  "tier2_tokens": 850
}
```

### POST /ingest
Ingest documents from a directory.

**Request:**
```
POST /ingest?data_dir=./data/documents
```

### GET /health
Health check endpoint.

## Project Structure

```
app/
├── main.py                 # FastAPI application
├── config.py               # Configuration settings
├── models/                 # Pydantic schemas
│   └── schemas.py
├── db/                     # Database clients
│   ├── vector_client.py    # ChromaDB client
│   └── sql_client.py       # SQLite client
├── services/
│   ├── ingestion/          # Document processing
│   │   ├── worker.py       # Main ingestion worker
│   │   ├── xml_parser.py   # PMC-OA/DailyMed XML parser
│   │   ├── pdf_parser.py   # PDF document parser
│   │   └── chunker.py      # Text chunking utilities
│   ├── classifier/         # Tier-1 classification
│   │   └── query_classifier.py
│   ├── retrieval/          # Dual retrieval engines
│   │   ├── vector_retriever.py
│   │   ├── sql_retriever.py
│   │   └── hybrid_retriever.py
│   ├── synthesis/          # Tier-2 answer generation
│   │   └── answer_synthesizer.py
│   └── orchestrator.py     # Pipeline orchestration
└── templates/
    └── index.html          # Minimal UI
```

## Tier-1 Provider Options

| Provider | Model | Cost | Speed |
|----------|-------|------|-------|
| **Groq** (recommended) | llama-3.1-8b-instant | Free tier | Very fast |
| Google Gemini | gemini-1.5-flash | Free tier | Fast |
| OpenAI | gpt-4o-mini | ~$0.15/1M tokens | Fast |
| Anthropic | claude-3-haiku | ~$0.25/1M tokens | Fast |

## Success Metrics (KPIs)

- **Query Classification Accuracy**: >90% F1-score
- **Latency**: Full hybrid synthesis <15 seconds
- **Cost-Efficiency Ratio (CER)**: Significantly lower than GPT-4-only baseline

## Testing

```bash
# Run tests
pytest tests/

# Run classifier tests
pytest tests/test_classifier.py -v
```

## License

MIT
