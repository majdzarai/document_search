# RAG Engine - Project Summary

## What This Project Is

A **production-grade RAG (Retrieval-Augmented Generation) API** that allows you to:
1. **Ingest documents** (PDF, TXT, HTML, DOCX) into a vector database
2. **Ask questions** and get AI-generated answers grounded in your documents
3. **Get source citations** so you know where the answer came from

## Quick Overview

```
YOUR APP  ──HTTP──>  RAG ENGINE API  ──HTTP──>  OpenRouter (LLM/Embeddings)
                          │
                          ▼
                    Qdrant (Vector DB)
```

## Implementation Status

| Component | Status | What It Does |
|-----------|--------|--------------|
| **API Layer** | DONE | FastAPI endpoints for query, ingest, health |
| **Embeddings** | DONE | OpenRouter + 4 alternative providers |
| **Vector Store** | DONE | Qdrant + 3 alternative providers |
| **LLM** | DONE | OpenRouter + 4 alternative providers |
| **Ingestion** | DONE | Loads PDF/TXT/HTML/DOCX, chunks, stores |
| **Reranking** | DONE | LLM-based + 3 alternative rerankers |
| **Evaluation** | DONE | Retrieval metrics, generation metrics, RAGAS integration |
| **Query Understanding** | DONE | Query analysis, expansion, intent classification |
| **Generation Enhancement** | DONE | Prompt building, context building, citations, streaming |
| **Security** | STRUCTURE | API key/RBAC structure exists |
| **Observability** | STRUCTURE | Logging/tracing/metrics structure exists |

## How It Works

### Document Ingestion Flow
```
Upload File → Load Text → Chunk → Embed → Store in Qdrant
```

### Query Flow
```
Question → Embed → Search Qdrant → Rerank → Generate Answer → Return with Sources
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key
export OPENROUTER_API_KEY=your-key

# Run
uvicorn src.main:app --reload

# Test
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Ask a question, get AI answer |
| `/api/v1/ingest` | POST | Upload a document |
| `/health` | GET | Health check |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI)                     │
│         /query         /ingest         /health              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       RAG PIPELINE                           │
│  Embed → Search → Rerank (opt) → Generate → Evaluate (opt) │
└─────────────────────────────────────────────────────────────┘
                              │
      ┌───────────────┬───────┼───────┬───────────────┐
      ▼               ▼       ▼       ▼               ▼
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐
│ EMBEDDINGS│  │VECTOR STORE│  │    LLM   │  │  EVALUATION   │
│(OpenRouter│  │  (Qdrant)  │  │(OpenRouter│  │  (Metrics)    │
│ +4 others)│  │ +3 others) │  │ +4 others)│  │               │
└───────────┘  └───────────┘  └───────────┘  └───────────────┘
```

## Key Configuration

```bash
# Required
OPENROUTER_API_KEY=sk-or-v1-...

# Optional - Chunking
CHUNKING_STRATEGY=recursive  # or: sentence, semantic
MAX_CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Optional - Reranking
ENABLE_RERANKING=true
RERANKER_TOP_K=5

# Optional - Production (Qdrant Cloud)
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-key

# Optional - Evaluation
ENABLE_EVALUATION=true
EVALUATION_DEFAULT_K=5
ENABLE_RAGAS=false
```

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | Application entry point |
| `src/api/routes/query.py` | Query endpoint |
| `src/api/routes/ingest.py` | Ingestion endpoint |
| `src/rag/pipeline.py` | Main RAG orchestrator |
| `src/ingestion/service.py` | Document processing |
| `src/evaluation/evaluator.py` | RAG quality metrics |
| `src/core/providers.py` | Shared provider instances |
| `src/core/config.py` | Configuration management |

## Design Patterns Used

1. **Factory Pattern** - Provider creation
2. **Strategy Pattern** - Swappable providers
3. **Abstract Base Class** - Provider interfaces
4. **Singleton** - Shared providers (ingestion + query use same instances)
5. **Facade** - `RAGPipeline.run()` hides complexity

## Important Notes

1. **Shared Providers**: Ingestion and queries MUST use the same vector store instance (handled by `src/core/providers.py`)

2. **In-Memory Default**: Qdrant runs in-memory by default - data lost on restart. Configure `QDRANT_HOST`/`QDRANT_PORT` for persistence.

3. **Same Embedding Model**: Must use the same embedding model for ingestion and queries (vectors must be comparable)

4. **API Key Required**: OpenRouter API key is required for embeddings and LLM generation

## Documentation

- **CLAUDE.md** - Comprehensive project documentation
- **README.md** - User-facing quick start guide
- **summary.md** - This file (brief overview)

## Resources

- OpenRouter: https://openrouter.ai
- Qdrant: https://qdrant.tech
- FastAPI: https://fastapi.tiangolo.com
