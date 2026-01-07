# RAG Engine - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [What is RAG?](#what-is-rag)
3. [Architecture Overview](#architecture-overview)
4. [Complete Workflow](#complete-workflow)
5. [How to Run](#how-to-run)
6. [Client Integration Guide](#client-integration-guide)
7. [API Reference](#api-reference)
8. [Component Deep Dive](#component-deep-dive)
9. [Configuration Reference](#configuration-reference)
10. [Folder Structure](#folder-structure)
11. [Design Patterns](#design-patterns)
12. [Data Flow Diagrams](#data-flow-diagrams)

---

## Project Overview

This is a **production-grade RAG (Retrieval-Augmented Generation) Engine** built as an external API service. The system is designed to:

- **Be used by multiple client applications** via REST API
- **Model-agnostic**: Supports multiple LLM, embedding, and vector store providers
- **Extensible**: Easy to add new providers through abstract base classes
- **Production-ready**: Includes ingestion, reranking, and proper configuration management

### Current Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| API Layer | DONE | FastAPI with query, ingest, and health endpoints |
| Embeddings | DONE | OpenRouter provider with abstraction layer |
| Vector Store | DONE | Qdrant provider with in-memory/cloud modes |
| LLM Integration | DONE | OpenRouter LLM provider for answer generation |
| Ingestion Pipeline | DONE | PDF, TXT, HTML, DOCX loaders + chunking strategies |
| Reranking | DONE | Simple LLM-based reranker |
| Evaluation | DONE | Retrieval metrics, generation metrics, RAGAS integration |
| Observability | STRUCTURE | Logging/tracing structure exists |
| Security | STRUCTURE | API key/RBAC structure exists |

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that enhances Large Language Models by providing them with relevant context from your own documents.

### The RAG Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG WORKFLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. USER QUESTION                                                            │
│     "What are the benefits of using Python for AI?"                          │
│                            │                                                 │
│                            ▼                                                 │
│  2. EMBED QUESTION                                                           │
│     Convert to vector: [0.12, 0.45, 0.78, ...] (1536 numbers)               │
│                            │                                                 │
│                            ▼                                                 │
│  3. VECTOR SEARCH                                                            │
│     Find similar document chunks in the vector database                      │
│     Returns: Top K most relevant chunks with similarity scores               │
│                            │                                                 │
│                            ▼                                                 │
│  4. RERANK (Optional)                                                        │
│     Re-score documents for better relevance                                  │
│     Uses LLM or cross-encoder to compare query + document together           │
│                            │                                                 │
│                            ▼                                                 │
│  5. GENERATE ANSWER                                                          │
│     Send context + question to LLM                                           │
│     LLM generates answer grounded in your documents                          │
│                            │                                                 │
│                            ▼                                                 │
│  6. RETURN RESPONSE                                                          │
│     {"answer": "Python offers...", "sources": ["doc1.pdf", ...]}            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why RAG Instead of Fine-Tuning?

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Data Updates | Instant (add new docs) | Requires retraining |
| Cost | Lower (no training) | Higher (GPU time) |
| Transparency | Shows source documents | Black box |
| Accuracy | Grounded in facts | May hallucinate |
| Best For | Dynamic knowledge bases | Consistent style/behavior |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLIENT APPLICATIONS                                  │
│         (Web Apps, Mobile Apps, CLI Tools, Other Services)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP/REST
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (FastAPI)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ POST /query │  │ POST /ingest│  │ GET /health │  │ GET /collections    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAG PIPELINE                                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │ Embedding │──│ Retrieval │──│ Reranking │──│ Context   │──│Generation │ │
│  │           │  │           │  │ (Optional)│  │ Building  │  │   (LLM)   │ │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
        ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
        │   EMBEDDING   │  │  VECTOR STORE │  │      LLM      │
        │   PROVIDER    │  │   PROVIDER    │  │   PROVIDER    │
        │  (OpenRouter) │  │   (Qdrant)    │  │  (OpenRouter) │
        └───────────────┘  └───────────────┘  └───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                                    │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │  Document │──│  Chunking │──│ Metadata  │──│ Embedding │──│  Storage  │ │
│  │  Loaders  │  │ Strategies│  │Enrichment │  │           │  │           │ │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘  └───────────┘ │
│   PDF, TXT,      Recursive,     Extract &      Same model     Same vector  │
│   HTML, DOCX     Sentence,      enrich info    as queries     store        │
│                  Semantic                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Workflow

### Workflow 1: Document Ingestion (Adding Documents)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DOCUMENT INGESTION WORKFLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: CLIENT UPLOADS FILE                                                 │
│          POST /api/v1/ingest                                                 │
│          Body: { file: report.pdf }                                          │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 2: SELECT LOADER                                                       │
│          .pdf → PDFLoader                                                    │
│          .txt → TextLoader                                                   │
│          .html → HTMLLoader                                                  │
│          .docx → DOCXLoader                                                  │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 3: EXTRACT TEXT                                                        │
│          PDFLoader extracts text from all pages                              │
│          Preserves page numbers in metadata                                  │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 4: CHUNK TEXT                                                          │
│          Split into smaller pieces (default: 512 chars)                      │
│          Strategies: recursive, sentence, or semantic                        │
│          Overlap ensures context continuity (default: 50 chars)              │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 5: EMBED CHUNKS                                                        │
│          Each chunk → OpenRouter API → 1536-dim vector                       │
│          All chunks processed in batch for efficiency                        │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 6: STORE IN VECTOR DB                                                  │
│          Qdrant stores: vector + text + metadata                             │
│          Metadata includes: source, page, chunk_id, timestamp                │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 7: RETURN RESULT                                                       │
│          { success: true, chunks: 45, document_id: "abc123" }               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Workflow 2: Query Processing (Asking Questions)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       QUERY PROCESSING WORKFLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: CLIENT SENDS QUESTION                                               │
│          POST /api/v1/query                                                  │
│          Body: { "question": "What are Python's AI benefits?" }             │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 2: EMBED QUESTION                                                      │
│          Question → OpenRouter API → 1536-dim vector                         │
│          CRITICAL: Same embedding model as ingestion!                        │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 3: VECTOR SIMILARITY SEARCH                                            │
│          Qdrant searches for similar document vectors                        │
│          Uses cosine similarity (0-1 score)                                  │
│          Returns top K results (default: 5)                                  │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 4: RERANKING (if enabled)                                              │
│          LLM scores each document's relevance to query                       │
│          More accurate than pure vector similarity                           │
│          May filter low-relevance documents                                  │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 5: BUILD CONTEXT                                                       │
│          Format retrieved documents for LLM:                                 │
│          "Context: [Doc 1] Python offers... [Doc 2] AI libraries..."        │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 6: LLM GENERATION                                                      │
│          Send to OpenRouter LLM:                                             │
│          - System prompt: "Answer based on context"                          │
│          - Context + Question                                                │
│          LLM generates grounded answer                                       │
│                            │                                                 │
│                            ▼                                                 │
│  STEP 7: RETURN RESPONSE                                                     │
│          {                                                                   │
│            "answer": "Python offers several benefits for AI...",            │
│            "sources": ["python_guide.pdf", "ai_handbook.pdf"]               │
│          }                                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How to Run

### Prerequisites

- Python 3.9+
- OpenRouter API key (get one at https://openrouter.ai/keys)

### Quick Start

```bash
# 1. Clone/download the project
cd rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
# Windows CMD:
set OPENROUTER_API_KEY=your-api-key-here

# Windows PowerShell:
$env:OPENROUTER_API_KEY="your-api-key-here"

# Linux/Mac:
export OPENROUTER_API_KEY=your-api-key-here

# 4. Start the server
uvicorn src.main:app --reload

# 5. Server is now running at http://localhost:8000
```

### Verify It's Working

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy"}
```

### Test a Query

```bash
# Send a test query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

---

## Client Integration Guide

### How Clients Use This API

This RAG Engine is designed as a **backend service** that client applications call via HTTP.

#### Integration Pattern

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  YOUR APP       │  HTTP   │   RAG ENGINE    │  HTTP   │   OPENROUTER    │
│  (Frontend/     │ ──────> │   (This API)    │ ──────> │   (LLM/Embed)   │
│   Backend)      │         │                 │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

### Example Integrations

#### Python Integration

```python
import requests

RAG_API_URL = "http://localhost:8000"

# 1. Upload a document
def ingest_document(file_path: str):
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{RAG_API_URL}/api/v1/ingest",
            files={"file": f}
        )
    return response.json()

# 2. Ask a question
def ask_question(question: str):
    response = requests.post(
        f"{RAG_API_URL}/api/v1/query",
        json={"question": question}
    )
    return response.json()

# Usage
result = ingest_document("company_handbook.pdf")
print(f"Ingested {result['chunk_count']} chunks")

answer = ask_question("What is our vacation policy?")
print(f"Answer: {answer['answer']}")
print(f"Sources: {answer['sources']}")
```

#### JavaScript/TypeScript Integration

```typescript
const RAG_API_URL = "http://localhost:8000";

// Ask a question
async function askQuestion(question: string) {
  const response = await fetch(`${RAG_API_URL}/api/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return response.json();
}

// Usage
const result = await askQuestion("What are the key features?");
console.log(result.answer);
console.log(result.sources);
```

#### cURL Integration

```bash
# Query the RAG system
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does authentication work?"}'

# Upload a document
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@./document.pdf"
```

### Typical Use Cases

1. **Customer Support Chatbot**: Connect your chat UI to the `/query` endpoint
2. **Internal Knowledge Base**: Ingest company documents, let employees ask questions
3. **Documentation Search**: Better than keyword search - understands meaning
4. **Research Assistant**: Query academic papers, reports, or technical docs

---

## API Reference

### `POST /api/v1/query`

Query the RAG system with a question.

**Request:**
```json
{
  "question": "What are the benefits of RAG?",
  "top_k": 5,           // Optional: number of documents to retrieve
  "enable_reranking": true  // Optional: enable reranking
}
```

**Response:**
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) offers several benefits...",
  "sources": ["rag_guide.pdf", "ai_handbook.pdf"],
  "metadata": {
    "documents_retrieved": 5,
    "reranking_enabled": true
  }
}
```

### `POST /api/v1/ingest`

Ingest a document into the RAG system.

**Request:** Multipart form with file upload

**Response:**
```json
{
  "success": true,
  "document_name": "report.pdf",
  "chunk_count": 45,
  "document_id": "doc_abc123"
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

---

## Component Deep Dive

### 1. Embedding Layer (`src/embeddings/`)

**Purpose:** Convert text into numerical vectors (embeddings)

**How it works:**
- Text → OpenRouter API → 1536-dimensional vector
- Similar meanings = similar vectors
- Used for both ingestion and queries

**Key Files:**
- `base.py`: Abstract interface (`EmbeddingProvider`)
- `factory.py`: Creates provider instances
- `providers/openrouter.py`: OpenRouter implementation

### 2. Vector Store Layer (`src/vectorstore/`)

**Purpose:** Store and search document embeddings

**How it works:**
- Documents stored with: vector + text + metadata
- Cosine similarity finds similar documents
- Qdrant handles the heavy lifting

**Key Files:**
- `base.py`: Abstract interface (`VectorStoreProvider`)
- `factory.py`: Creates provider instances
- `providers/qdrant.py`: Qdrant implementation

**Qdrant Modes:**
- **In-memory** (default): No persistence, good for testing
- **Local server**: Persistent, requires Qdrant running
- **Cloud**: Production-ready, fully managed

### 3. LLM Layer (`src/llm/`)

**Purpose:** Generate natural language answers

**How it works:**
- Takes context (retrieved docs) + question
- Sends to OpenRouter API
- Returns generated answer

**Key Files:**
- `base.py`: Abstract interface (`LLMProvider`)
- `factory.py`: Creates provider instances
- `providers/openrouter.py`: OpenRouter implementation

### 4. Ingestion Pipeline (`src/ingestion/`)

**Purpose:** Process documents for storage in vector DB

**Components:**
- **Loaders**: Extract text from files (PDF, TXT, HTML, DOCX)
- **Chunkers**: Split text into manageable pieces
- **Metadata**: Extract and enrich document info

**Chunking Strategies:**
| Strategy | Best For | How it Works |
|----------|----------|--------------|
| Recursive | General docs | Splits by separators (paragraph, sentence, word) |
| Sentence | Articles/news | Respects sentence boundaries |
| Semantic | Technical docs | Groups by topic similarity (uses embeddings) |

### 5. Reranker Layer (`src/reranker/`)

**Purpose:** Improve retrieval quality by re-scoring documents

**Why Reranking?**
- Vector search is fast but imperfect
- Reranker compares query + document together
- Catches nuances that vector search misses

**Current Implementation:**
- LLM-based reranker (uses OpenRouter)
- Asks LLM to score document relevance 0-10
- More accurate than pure vector similarity

### 6. RAG Pipeline (`src/rag/pipeline.py`)

**Purpose:** Orchestrates the entire RAG flow

**Flow:**
1. Initialize providers (embedding, vector store, LLM, reranker)
2. On query: embed → search → rerank → generate
3. Return answer with sources

**Key Methods:**
- `run(question)`: Main entry point
- `_embed_question()`: Convert question to vector
- `_retrieve_documents()`: Search vector store
- `_rerank_documents()`: Apply reranking
- `_generate_answer()`: Call LLM

### 7. Shared Providers (`src/core/providers.py`)

**Purpose:** Ensure ingestion and queries use the SAME providers

**Why This Matters:**
- Ingestion stores documents in vector store A
- Queries search vector store A
- If they used different instances, queries would find nothing!

**Key Functions:**
- `get_embedding_provider()`: Returns shared embedding provider
- `get_vector_store()`: Returns shared vector store

### 8. Evaluation Layer (`src/evaluation/`)

**Purpose:** Measure RAG quality without affecting responses

**Why Evaluation?**
- Know if retrieval is finding relevant documents
- Detect if answers are grounded in context (vs hallucinated)
- Compare different configurations objectively
- Monitor quality in production

**IMPORTANT - Non-Blocking:**
Evaluation runs AFTER the response is built and NEVER modifies the answer.
If evaluation fails, the response is still returned.

**Components:**

1. **Retrieval Metrics** (`metrics/retrieval_metrics.py`):
   - `Precision@K`: Fraction of retrieved docs that are relevant
   - `Recall@K`: Fraction of relevant docs that were retrieved
   - `MRR`: How high relevant docs are ranked
   - `Hit Rate`: Did we find at least one relevant doc?
   - `Basic Stats`: Score distributions when no ground truth

2. **Generation Metrics** (`metrics/generation_metrics.py`):
   - `Faithfulness`: Is the answer grounded in context?
   - `Context Coverage`: How much context was used?
   - `Hallucination Risk`: Estimate of unsupported content

3. **RAGAS Adapter** (`metrics/ragas_adapter.py`):
   - Optional integration with RAGAS library
   - LLM-based metrics (requires `pip install ragas`)
   - Advanced faithfulness and relevancy scoring

4. **RAGEvaluator** (`evaluator.py`):
   - Main orchestrator class
   - Combines all metrics
   - Handles configuration and logging
   - Optional history storage for analysis

**Usage Example:**
```python
from src.evaluation import get_evaluator

# Get shared evaluator (respects config)
evaluator = get_evaluator()

# Evaluate a RAG response
result = evaluator.evaluate(
    question="What is RAG?",
    answer="RAG combines retrieval with generation...",
    retrieved_documents=[{"id": "doc1", "content": "..."}],
    ground_truth_ids=["doc1", "doc2"]  # Optional
)

# Access metrics
print(result.get_summary())
# {'precision_at_k': 0.5, 'faithfulness': 0.85, ...}
```

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_EVALUATION` | false | Master switch for evaluation |
| `EVALUATION_DEFAULT_K` | 5 | K value for @K metrics |
| `ENABLE_RAGAS` | false | Enable RAGAS metrics (requires package) |
| `EVALUATION_LOG_RESULTS` | true | Log metrics to console |
| `EVALUATION_STORE_HISTORY` | false | Store history for analysis |

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | API key for OpenRouter |
| `OPENROUTER_BASE_URL` | No | https://openrouter.ai/api/v1 | API endpoint |
| `EMBEDDING_MODEL` | No | openai/text-embedding-3-small | Embedding model |
| `EMBEDDING_PROVIDER` | No | openrouter | Embedding provider name |
| `LLM_MODEL` | No | openai/gpt-3.5-turbo | LLM model for generation |
| `LLM_PROVIDER` | No | openrouter | LLM provider name |
| `VECTOR_STORE_PROVIDER` | No | qdrant | Vector database |
| `VECTOR_DIMENSION` | No | 1536 | Embedding dimension |
| `QDRANT_COLLECTION_NAME` | No | rag_documents | Collection name |
| `QDRANT_HOST` | No | - | For local Qdrant server |
| `QDRANT_PORT` | No | - | For local Qdrant server |
| `QDRANT_URL` | No | - | For Qdrant Cloud |
| `QDRANT_API_KEY` | No | - | For Qdrant Cloud |
| `CHUNKING_STRATEGY` | No | recursive | "recursive", "sentence", "semantic" |
| `MAX_CHUNK_SIZE` | No | 512 | Maximum chunk size (chars) |
| `MIN_CHUNK_SIZE` | No | 100 | Minimum chunk size (chars) |
| `CHUNK_OVERLAP` | No | 50 | Overlap between chunks |
| `ENABLE_RERANKING` | No | false | Enable/disable reranking |
| `RERANKER_TOP_K` | No | 5 | Documents to return after reranking |
| `ENABLE_EVALUATION` | No | false | Enable RAG quality metrics |
| `EVALUATION_DEFAULT_K` | No | 5 | K value for @K metrics |
| `ENABLE_RAGAS` | No | false | Enable RAGAS metrics |
| `EVALUATION_LOG_RESULTS` | No | true | Log evaluation results |
| `EVALUATION_STORE_HISTORY` | No | false | Store evaluation history |

### Configuration Examples

**Minimal (Development):**
```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

**With Reranking:**
```bash
OPENROUTER_API_KEY=sk-or-v1-...
ENABLE_RERANKING=true
RERANKER_TOP_K=3
```

**With Semantic Chunking:**
```bash
OPENROUTER_API_KEY=sk-or-v1-...
CHUNKING_STRATEGY=semantic
MAX_CHUNK_SIZE=1000
```

**Production with Qdrant Cloud:**
```bash
OPENROUTER_API_KEY=sk-or-v1-...
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-key
LLM_MODEL=openai/gpt-4
```

**With Evaluation Enabled:**
```bash
OPENROUTER_API_KEY=sk-or-v1-...
ENABLE_EVALUATION=true
EVALUATION_DEFAULT_K=5
EVALUATION_LOG_RESULTS=true
# For advanced metrics (requires: pip install ragas)
ENABLE_RAGAS=false
```

---

## Folder Structure

```
rag/
├── src/                          # Main source code
│   ├── main.py                   # Application entry point
│   │
│   ├── api/                      # REST API layer
│   │   ├── app.py                # FastAPI app factory
│   │   ├── dependencies.py       # Dependency injection
│   │   ├── routes/               # API endpoints
│   │   │   ├── query.py          # POST /query (IMPLEMENTED)
│   │   │   ├── ingest.py         # POST /ingest (IMPLEMENTED)
│   │   │   ├── health.py         # GET /health (IMPLEMENTED)
│   │   │   └── collections.py    # Collection management
│   │   └── schemas/              # Request/response models
│   │       ├── requests.py       # QueryRequest, IngestRequest
│   │       └── responses.py      # QueryResponse, IngestResponse
│   │
│   ├── core/                     # Core utilities
│   │   ├── config.py             # Settings from env vars
│   │   └── providers.py          # Shared provider instances
│   │
│   ├── rag/                      # RAG pipeline
│   │   └── pipeline.py           # Main orchestrator
│   │
│   ├── embeddings/               # Embedding layer (IMPLEMENTED)
│   │   ├── base.py               # EmbeddingProvider interface
│   │   ├── factory.py            # Provider factory
│   │   └── providers/
│   │       └── openrouter.py     # OpenRouter implementation
│   │
│   ├── vectorstore/              # Vector store layer (IMPLEMENTED)
│   │   ├── base.py               # VectorStoreProvider interface
│   │   ├── factory.py            # Provider factory
│   │   └── providers/
│   │       └── qdrant.py         # Qdrant implementation
│   │
│   ├── llm/                      # LLM layer (IMPLEMENTED)
│   │   ├── base.py               # LLMProvider interface
│   │   ├── factory.py            # Provider factory
│   │   └── providers/
│   │       └── openrouter.py     # OpenRouter implementation
│   │
│   ├── reranker/                 # Reranking layer (IMPLEMENTED)
│   │   ├── base.py               # RerankerProvider interface
│   │   ├── factory.py            # Provider factory
│   │   └── providers/
│   │       └── simple.py         # LLM-based reranker
│   │
│   ├── ingestion/                # Document ingestion (IMPLEMENTED)
│   │   ├── service.py            # Main IngestionService
│   │   ├── document_loader/      # File format handlers
│   │   │   ├── base.py           # DocumentLoader interface
│   │   │   ├── pdf_loader.py     # PDF support
│   │   │   ├── text_loader.py    # TXT support
│   │   │   ├── html_loader.py    # HTML support
│   │   │   └── docx_loader.py    # DOCX support
│   │   ├── chunking/             # Text splitting
│   │   │   ├── base.py           # Chunker interface
│   │   │   ├── recursive_splitter.py
│   │   │   ├── sentence_splitter.py
│   │   │   └── semantic_splitter.py
│   │   └── metadata/             # Metadata handling
│   │       ├── extractor.py
│   │       └── enricher.py
│   │
│   ├── evaluation/               # RAG evaluation (IMPLEMENTED)
│   │   ├── __init__.py           # Public API exports
│   │   ├── evaluator.py          # Main RAGEvaluator class
│   │   └── metrics/
│   │       ├── __init__.py       # Metrics exports
│   │       ├── retrieval_metrics.py  # Precision@K, Recall@K, MRR
│   │       ├── generation_metrics.py # Faithfulness, Coverage
│   │       └── ragas_adapter.py  # Optional RAGAS integration
│   │
│   ├── observability/            # Monitoring (STRUCTURE)
│   └── security/                 # Security (STRUCTURE)
│
├── tests/                        # Test suite
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # This documentation
├── summary.md                    # Brief project summary
└── README.md                     # User-facing README
```

---

## Design Patterns

### 1. Factory Pattern
**Used in:** `embeddings/factory.py`, `vectorstore/factory.py`, `llm/factory.py`, `reranker/factory.py`

```python
def create_embedding_provider(provider_name: str) -> EmbeddingProvider:
    if provider_name == "openrouter":
        return OpenRouterEmbeddingProvider()
    raise ValueError(f"Unknown provider: {provider_name}")
```

**Why:** Decouples provider creation from usage. Easy to add new providers.

### 2. Strategy Pattern
**Used in:** All provider layers

Different providers implement the same interface, allowing runtime swapping.

### 3. Abstract Base Class
**Used in:** `base.py` files in each layer

Defines contracts that all providers must follow.

### 4. Singleton (via Module-Level Instances)
**Used in:** `src/core/providers.py`

Ensures ingestion and queries use the same provider instances.

### 5. Facade Pattern
**Used in:** `RAGPipeline.run()`

Hides complexity behind a simple `run(question)` interface.

---

## Data Flow Diagrams

### Query Data Flow

```
User Question: "What is RAG?"
         │
         ▼
┌─────────────────────────────────────────┐
│         RAGPipeline.run()               │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│    EmbeddingProvider.embed_text()       │
│    "What is RAG?" → [0.12, 0.45, ...]   │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│    VectorStore.search()                 │
│    Returns top 5 similar documents      │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│    Reranker.rerank() (if enabled)       │
│    Re-scores documents by relevance     │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│    LLMProvider.generate_with_context()  │
│    Context + Question → Answer          │
└─────────────────────────────────────────┘
         │
         ▼
Response: {"answer": "RAG is...", "sources": [...]}
```

### Ingestion Data Flow

```
File: report.pdf
         │
         ▼
┌─────────────────────────────────────────┐
│    PDFLoader.load()                     │
│    Extracts text from PDF               │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│    Chunker.split()                      │
│    Splits into chunks (512 chars each)  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│    MetadataEnricher.enrich()            │
│    Adds source, page, timestamp         │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│    EmbeddingProvider.embed_texts()      │
│    Batch embed all chunks               │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│    VectorStore.upsert()                 │
│    Store vectors + text + metadata      │
└─────────────────────────────────────────┘
         │
         ▼
Result: {success: true, chunks: 45}
```

---

## Important Notes

1. **Shared Providers are Critical**: The ingestion service and RAG pipeline MUST use the same vector store instance. This is handled by `src/core/providers.py`.

2. **In-Memory Mode**: By default, Qdrant runs in-memory. Data is lost on restart. For persistence, configure `QDRANT_HOST`/`QDRANT_PORT` or use Qdrant Cloud.

3. **API Key Required**: Without `OPENROUTER_API_KEY`, the system cannot generate embeddings or LLM responses.

4. **Embedding Model Consistency**: The same embedding model must be used for both ingestion and queries. Different models produce incompatible vectors.

5. **Reranking Trade-offs**: Reranking improves quality but adds latency and cost (additional LLM calls).

---

## Resources

- **OpenRouter**: https://openrouter.ai - Unified LLM API
- **Qdrant**: https://qdrant.tech - Vector database
- **FastAPI**: https://fastapi.tiangolo.com - Web framework
- **RAG Overview**: https://www.pinecone.io/learn/retrieval-augmented-generation/
