# RAG Engine - Project Documentation

## Project Overview

This is a **production-grade RAG (Retrieval-Augmented Generation) Engine** built as an external API service. The system is designed to be used by multiple client applications, exposed via REST API, and is model-agnostic.

### What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances Large Language Models by:
1. Converting user questions into embeddings (numerical vectors)
2. Searching a vector database for similar document embeddings
3. Using retrieved documents as context for answer generation
4. Returning answers with source citations

---

## Current Implementation Status

### Completed (Steps 1-3)

| Step | Component | Status | Description |
|------|-----------|--------|-------------|
| 1 | API Layer | DONE | FastAPI app, routes, schemas, middleware structure |
| 2 | Embeddings | DONE | OpenRouter embedding provider with abstraction layer |
| 3 | Vector Store | DONE | Qdrant provider with in-memory mode |

### Not Yet Implemented (Steps 4+)

| Step | Component | Status | Description |
|------|-----------|--------|-------------|
| 4 | LLM Integration | EMPTY | Real answer generation with OpenAI/Anthropic |
| 5 | Ingestion Pipeline | EMPTY | Document loading, chunking, metadata |
| 6 | Reranking | EMPTY | Cohere/CrossEncoder reranking |
| 7 | Evaluation | EMPTY | RAGAS metrics, retrieval quality |
| 8 | Observability | EMPTY | Logging, tracing, metrics |
| 9 | Security | EMPTY | API keys, RBAC, multi-tenancy |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
set OPENROUTER_API_KEY=your-api-key-here   # Windows CMD
# OR
export OPENROUTER_API_KEY=your-api-key-here # Linux/Mac

# 3. Start the server
uvicorn src.main:app --reload

# 4. Test the API
# Health check: GET http://localhost:8000/health
# Query: POST http://localhost:8000/api/v1/query
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT APPLICATIONS                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API LAYER (FastAPI)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  /query  │  │ /health  │  │ /ingest  │  │  /collections    │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Embedding  │──│  Retrieval  │──│  Generation (MOCKED)    │  │
│  │  (DONE)     │  │  (DONE)     │  │  (NOT IMPLEMENTED)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌───────────────────┐      ┌───────────────────┐
        │  EMBEDDING LAYER  │      │  VECTOR STORE     │
        │  (OpenRouter)     │      │  (Qdrant)         │
        │  IMPLEMENTED      │      │  IMPLEMENTED      │
        └───────────────────┘      └───────────────────┘
```

---

## Folder Structure

```
rag/
├── src/                          # Main source code
│   ├── main.py                   # Application entry point
│   ├── __init__.py
│   │
│   ├── api/                      # REST API layer
│   │   ├── app.py                # FastAPI app factory (DONE)
│   │   ├── dependencies.py       # Dependency injection (DONE)
│   │   ├── routes/               # API endpoints
│   │   │   ├── query.py          # POST /query endpoint (DONE)
│   │   │   ├── health.py         # GET /health endpoint (DONE)
│   │   │   ├── ingest.py         # Document ingestion (EMPTY)
│   │   │   └── collections.py    # Collection management (EMPTY)
│   │   ├── schemas/              # Pydantic models
│   │   │   ├── requests.py       # Request models (DONE)
│   │   │   └── responses.py      # Response models (DONE)
│   │   └── middleware/           # Middleware (EMPTY - structure only)
│   │       ├── authentication.py
│   │       ├── rate_limiter.py
│   │       ├── error_handler.py
│   │       └── request_logging.py
│   │
│   ├── core/                     # Core utilities
│   │   ├── config.py             # Configuration management (DONE)
│   │   ├── settings.py           # (EMPTY)
│   │   ├── constants.py          # (EMPTY)
│   │   ├── exceptions.py         # (EMPTY)
│   │   └── secrets.py            # (EMPTY)
│   │
│   ├── rag/                      # RAG pipeline
│   │   ├── pipeline.py           # Main orchestrator (DONE)
│   │   ├── query_understanding/  # Query analysis (EMPTY)
│   │   ├── retrieval/            # Document retrieval (EMPTY)
│   │   └── generation/           # Answer generation (EMPTY)
│   │
│   ├── embeddings/               # Embedding layer (DONE)
│   │   ├── base.py               # Abstract base class (DONE)
│   │   ├── factory.py            # Provider factory (DONE)
│   │   └── providers/
│   │       ├── openrouter.py     # OpenRouter provider (DONE)
│   │       ├── openai.py         # (EMPTY)
│   │       ├── cohere.py         # (EMPTY)
│   │       ├── huggingface.py    # (EMPTY)
│   │       └── sentence_transformers.py  # (EMPTY)
│   │
│   ├── vectorstore/              # Vector database layer (DONE)
│   │   ├── base.py               # Abstract base class (DONE)
│   │   ├── factory.py            # Provider factory (DONE)
│   │   └── providers/
│   │       ├── qdrant.py         # Qdrant provider (DONE)
│   │       ├── pinecone.py       # (EMPTY)
│   │       ├── weaviate.py       # (EMPTY)
│   │       └── pgvector.py       # (EMPTY)
│   │
│   ├── llm/                      # LLM layer (EMPTY - structure only)
│   │   ├── base.py
│   │   ├── factory.py
│   │   └── providers/
│   │       ├── openai.py
│   │       ├── anthropic.py
│   │       ├── ollama.py
│   │       └── vllm.py
│   │
│   ├── reranker/                 # Reranking layer (EMPTY - structure only)
│   │   ├── base.py
│   │   ├── factory.py
│   │   └── providers/
│   │       ├── cohere.py
│   │       ├── cross_encoder.py
│   │       └── bge_reranker.py
│   │
│   ├── ingestion/                # Document ingestion (EMPTY - structure only)
│   │   ├── service.py
│   │   ├── document_loader/
│   │   │   ├── base.py
│   │   │   ├── pdf_loader.py
│   │   │   ├── text_loader.py
│   │   │   ├── html_loader.py
│   │   │   └── docx_loader.py
│   │   ├── chunking/
│   │   │   ├── base.py
│   │   │   ├── recursive_splitter.py
│   │   │   ├── semantic_splitter.py
│   │   │   └── sentence_splitter.py
│   │   └── metadata/
│   │       ├── enricher.py
│   │       └── extractor.py
│   │
│   ├── evaluation/               # RAG evaluation (EMPTY - structure only)
│   │   ├── evaluator.py
│   │   └── metrics/
│   │       ├── retrieval_metrics.py
│   │       ├── generation_metrics.py
│   │       └── ragas_adapter.py
│   │
│   ├── observability/            # Monitoring (EMPTY - structure only)
│   │   ├── logging.py
│   │   ├── tracing.py
│   │   ├── metrics.py
│   │   └── spans.py
│   │
│   └── security/                 # Security (EMPTY - structure only)
│       ├── api_key.py
│       ├── rbac.py
│       └── tenant.py
│
├── tests/                        # Test suite (EMPTY - structure only)
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── deploy/                       # Deployment configs (EMPTY)
├── scripts/                      # Utility scripts (EMPTY)
│
├── requirements.txt              # Python dependencies
├── Dockerfile                    # (EMPTY)
├── docker-compose.yml            # (EMPTY)
├── Makefile                      # (EMPTY)
├── pyproject.toml                # Project metadata
└── .env                          # Environment variables (DO NOT COMMIT)
```

---

## File Details

### Implemented Files (DONE)

#### `src/main.py`
- **Role**: Application entry point
- **What it does**: Imports FastAPI app and runs with uvicorn
- **Usage**: `uvicorn src.main:app --reload`

#### `src/core/config.py`
- **Role**: Centralized configuration management
- **What it does**: Reads environment variables, provides Settings class
- **Key settings**:
  - `OPENROUTER_API_KEY` - Required for embeddings
  - `OPENROUTER_BASE_URL` - API endpoint
  - `EMBEDDING_MODEL` - Model name (default: openai/text-embedding-3-small)
  - `EMBEDDING_PROVIDER` - Provider name (default: openrouter)
  - `VECTOR_STORE_PROVIDER` - Vector DB (default: qdrant)
  - `VECTOR_DIMENSION` - Embedding size (default: 1536)
  - `QDRANT_COLLECTION_NAME` - Collection name (default: rag_documents)
  - `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_URL`, `QDRANT_API_KEY` - Qdrant connection

#### `src/api/app.py`
- **Role**: FastAPI application factory
- **What it does**: Creates FastAPI app, includes routers, sets up CORS

#### `src/api/routes/query.py`
- **Role**: Main query endpoint
- **What it does**: Handles POST /api/v1/query requests
- **Endpoint**: `POST /api/v1/query`
- **Request body**: `{"question": "your question here"}`
- **Response**: `{"answer": "...", "sources": [...]}`

#### `src/api/routes/health.py`
- **Role**: Health check endpoint
- **What it does**: Returns service status
- **Endpoint**: `GET /health`

#### `src/api/schemas/requests.py`
- **Role**: Pydantic request models
- **Models**: `QueryRequest` with `question` field

#### `src/api/schemas/responses.py`
- **Role**: Pydantic response models
- **Models**: `QueryResponse` with `answer` and `sources` fields

#### `src/rag/pipeline.py`
- **Role**: Main RAG orchestrator
- **What it does**:
  1. Initializes embedding provider
  2. Initializes vector store (Qdrant)
  3. Seeds example documents at startup
  4. Handles query flow: embed → search → generate (mocked)
- **Key methods**:
  - `run(question)` - Main entry point
  - `_embed_question(question)` - Convert question to vector
  - `_retrieve_documents(embedding)` - Search vector store
  - `_seed_example_documents()` - Load example docs at startup
  - `_mock_generate_answer()` - Placeholder for LLM

#### `src/embeddings/base.py`
- **Role**: Abstract base class for embedding providers
- **Interface**:
  - `embed_text(text) -> List[float]`
  - `embed_texts(texts) -> List[List[float]]`
  - `get_dimension() -> int`
  - `get_model_name() -> str`

#### `src/embeddings/factory.py`
- **Role**: Creates embedding providers
- **What it does**: Reads config and returns appropriate provider
- **Supported**: `openrouter`

#### `src/embeddings/providers/openrouter.py`
- **Role**: OpenRouter embedding implementation
- **What it does**: Calls OpenRouter API to generate embeddings
- **Supports**: Any model available via OpenRouter (OpenAI, etc.)

#### `src/vectorstore/base.py`
- **Role**: Abstract base class for vector stores
- **Interface**:
  - `upsert(ids, embeddings, texts, metadata) -> bool`
  - `search(query_embedding, top_k, filter) -> List[Dict]`
  - `delete(ids) -> bool`
  - `count() -> int`
  - `get_info() -> Dict`

#### `src/vectorstore/factory.py`
- **Role**: Creates vector store providers
- **What it does**: Reads config and returns appropriate provider
- **Supported**: `qdrant`

#### `src/vectorstore/providers/qdrant.py`
- **Role**: Qdrant vector store implementation
- **What it does**:
  - Connects to Qdrant (in-memory, local, or cloud)
  - Stores document embeddings with metadata
  - Performs similarity search
- **Modes**:
  - In-memory (default): No persistence, perfect for learning
  - Local server: `QDRANT_HOST=localhost QDRANT_PORT=6333`
  - Cloud: `QDRANT_URL=... QDRANT_API_KEY=...`

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | API key for OpenRouter |
| `OPENROUTER_BASE_URL` | No | https://openrouter.ai/api/v1 | OpenRouter API URL |
| `EMBEDDING_MODEL` | No | openai/text-embedding-3-small | Embedding model |
| `EMBEDDING_PROVIDER` | No | openrouter | Embedding provider |
| `VECTOR_STORE_PROVIDER` | No | qdrant | Vector database |
| `VECTOR_DIMENSION` | No | 1536 | Embedding dimension |
| `QDRANT_COLLECTION_NAME` | No | rag_documents | Qdrant collection |
| `QDRANT_HOST` | No | - | Qdrant server host |
| `QDRANT_PORT` | No | - | Qdrant server port |
| `QDRANT_URL` | No | - | Qdrant Cloud URL |
| `QDRANT_API_KEY` | No | - | Qdrant Cloud API key |

---

## Data Flow

### Query Flow (Current Implementation)

```
1. User sends POST /api/v1/query with {"question": "What is RAG?"}
                    │
                    ▼
2. API route calls RAGPipeline.run(question)
                    │
                    ▼
3. Pipeline embeds question using OpenRouter
   question → [0.1, 0.2, 0.3, ...] (1536 floats)
                    │
                    ▼
4. Pipeline searches Qdrant for similar documents
   Returns top 3 documents with similarity scores
                    │
                    ▼
5. Pipeline generates answer (MOCKED - returns template)
                    │
                    ▼
6. API returns {"answer": "...", "sources": ["doc1.pdf", ...]}
```

### Document Seeding (At Startup)

```
1. RAGPipeline initializes
                    │
                    ▼
2. _seed_example_documents() called
                    │
                    ▼
3. 6 hardcoded documents about RAG topics
                    │
                    ▼
4. Each document embedded via OpenRouter
                    │
                    ▼
5. Embeddings stored in Qdrant with metadata
   - doc_id: "doc_001", "doc_002", etc.
   - text: document content
   - metadata: source, page, topic
```

---

## Design Patterns Used

1. **Factory Pattern**: `create_embedding_provider()`, `create_vector_store_provider()`
2. **Strategy Pattern**: Different providers implement same interface
3. **Abstract Base Class**: `EmbeddingProvider`, `VectorStoreProvider`
4. **Dependency Injection**: Pipeline accepts optional providers
5. **Facade Pattern**: `RAGPipeline.run()` hides complexity

---

## Next Steps to Implement

### Step 4: LLM Integration
1. Implement `src/llm/base.py` - Abstract LLM interface
2. Implement `src/llm/providers/openai.py` - OpenAI provider
3. Implement `src/llm/factory.py` - LLM factory
4. Update `src/rag/pipeline.py` - Replace mock generation
5. Add `LLM_PROVIDER`, `LLM_MODEL` to config

### Step 5: Ingestion Pipeline
1. Implement document loaders (PDF, TXT, HTML, DOCX)
2. Implement chunking strategies
3. Implement metadata extraction
4. Create `/ingest` API endpoint
5. Process real documents instead of hardcoded examples

### Step 6: Reranking
1. Implement `src/reranker/base.py`
2. Implement Cohere reranker
3. Add reranking step between retrieval and generation

### Step 7: Evaluation
1. Implement retrieval metrics (MRR, Recall@K)
2. Implement generation metrics (BLEU, ROUGE)
3. Add RAGAS integration

### Step 8: Observability
1. Add structured logging
2. Add OpenTelemetry tracing
3. Add Prometheus metrics

### Step 9: Security
1. Implement API key authentication
2. Add rate limiting
3. Add multi-tenancy support

---

## Testing

Currently, the test structure exists but tests are not implemented.

```bash
# Run tests (when implemented)
pytest tests/

# Run specific test file
pytest tests/unit/test_embeddings.py
```

---

## API Endpoints

| Method | Endpoint | Status | Description |
|--------|----------|--------|-------------|
| GET | `/health` | DONE | Health check |
| POST | `/api/v1/query` | DONE | Query the RAG system |
| POST | `/api/v1/ingest` | EMPTY | Ingest documents |
| GET | `/api/v1/collections` | EMPTY | List collections |

---

## Dependencies

```
# Core
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0

# Vector Store (Step 3)
qdrant-client>=1.7.0

# Future (commented in requirements.txt)
# openai>=1.10.0
# anthropic>=0.18.0
# cohere>=4.40.0
# pypdf>=3.17.0
# sentence-transformers>=2.2.0
```

---

## Important Notes

1. **In-Memory Mode**: By default, Qdrant runs in-memory. Data is lost when app restarts. For persistence, set `QDRANT_HOST` and `QDRANT_PORT`.

2. **API Key Required**: The system won't generate embeddings without `OPENROUTER_API_KEY`. It will fall back to mocked retrieval.

3. **Mocked Generation**: Answer generation is currently mocked. It returns a template response, not real LLM output.

4. **Example Documents**: 6 hardcoded documents are loaded at startup. In production, use the ingestion pipeline.

5. **UUID for Qdrant**: Qdrant requires UUID or integer point IDs. String IDs like "doc_001" are stored in metadata.

---

## Code Style

- All files have detailed comments explaining concepts
- Designed for beginners to understand
- Each module is self-contained with clear responsibilities
- Type hints used throughout
- Docstrings explain what, why, and how

---

## Contact / Resources

- OpenRouter: https://openrouter.ai
- Qdrant: https://qdrant.tech
- FastAPI: https://fastapi.tiangolo.com
