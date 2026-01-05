<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/main/icons/robot.svg" width="120" alt="RAG Engine Logo"/>
</p>

<h1 align="center">RAG Engine API</h1>

<p align="center">
  <strong>Production-Grade Retrieval-Augmented Generation as a Service</strong>
</p>

<p align="center">
  <em>Transform your applications with intelligent document retrieval and AI-powered answers</em>
</p>

<p align="center">
  <a href="#-quick-start">
    <img src="https://img.shields.io/badge/Quick%20Start-5%20min-brightgreen?style=for-the-badge&logo=rocket" alt="Quick Start"/>
  </a>
  <a href="#-api-reference">
    <img src="https://img.shields.io/badge/API-Docs-blue?style=for-the-badge&logo=swagger" alt="API Docs"/>
  </a>
  <a href="#-features">
    <img src="https://img.shields.io/badge/Features-Explore-purple?style=for-the-badge&logo=sparkles" alt="Features"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-FF6B6B?style=flat-square&logo=qdrant&logoColor=white" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/OpenRouter-AI%20Gateway-7C3AED?style=flat-square" alt="OpenRouter"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"/>
</p>

<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700" alt="Divider"/>
</p>

---

## What is RAG Engine?

**RAG Engine** is a powerful REST API that enables your applications to answer questions using your own documents. Simply upload your files, and the API transforms them into a searchable knowledge base powered by AI.

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/213910845-af37a709-8995-40d6-be59-724526e3c3d7.gif" width="500" alt="AI Animation"/>
</p>

### How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Your Docs     │────▶│   RAG Engine    │────▶│  Smart Answers  │
│  PDF, TXT, HTML │     │   Vector Search │     │  with Sources   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**1. Ingest** → Upload documents (PDF, TXT, HTML, DOCX)
**2. Query** → Ask questions in natural language
**3. Retrieve** → AI finds relevant document sections
**4. Generate** → LLM creates accurate answers with citations

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700" alt="Divider"/>
</p>

## Why Choose RAG Engine?

<table>
<tr>
<td width="50%">

### For Developers

- **Simple REST API** - Integrate in minutes
- **No ML Expertise Required** - We handle the complexity
- **Multiple File Formats** - PDF, TXT, HTML, DOCX
- **Instant Deployment** - One command to start

</td>
<td width="50%">

### For Businesses

- **Reduce Support Costs** - Automate Q&A
- **Accurate Answers** - Grounded in your documents
- **Source Citations** - Verify every answer
- **Scale Effortlessly** - Handle any document volume

</td>
</tr>
</table>

---

## Features

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/229223263-cf2e4b07-2615-4f87-9c38-e37600f8381a.gif" width="400" alt="Features Animation"/>
</p>

| Feature | Description |
|---------|-------------|
| **Document Ingestion** | Upload and process PDF, TXT, HTML, DOCX files automatically |
| **Semantic Search** | Find relevant content based on meaning, not just keywords |
| **AI-Powered Answers** | Generate human-like responses using leading LLMs |
| **Source Citations** | Every answer includes document references |
| **Model Agnostic** | Works with OpenAI, Anthropic, and more via OpenRouter |
| **Vector Storage** | Powered by Qdrant for lightning-fast retrieval |
| **Zero Infrastructure** | In-memory mode for instant setup |
| **Production Ready** | Scale to cloud with Qdrant Cloud |

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700" alt="Divider"/>
</p>

## Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenRouter API Key ([Get one free](https://openrouter.ai))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-engine.git
cd rag-engine

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Set your API key
# Windows (CMD)
set OPENROUTER_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:OPENROUTER_API_KEY="your-api-key-here"

# Linux/Mac
export OPENROUTER_API_KEY=your-api-key-here
```

### Launch

```bash
# Start the server
uvicorn src.main:app --reload

# Server running at: http://localhost:8000
# API Docs at: http://localhost:8000/docs
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284087-bbe7e430-757e-4901-90bf-4cd2ce3e1852.gif" width="50" alt="Rocket"/>
  <strong>That's it! Your RAG API is ready!</strong>
</p>

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700" alt="Divider"/>
</p>

## API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check API health status |
| `POST` | `/query` | Ask questions to your knowledge base |
| `POST` | `/api/v1/ingest` | Upload and process documents |
| `POST` | `/api/v1/ingest/text` | Ingest raw text directly |
| `GET` | `/api/v1/ingest/formats` | List supported file formats |

---

### Health Check

Check if the API is running.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "RAG Engine"
}
```

---

### Query Your Knowledge Base

Ask questions and get AI-powered answers with source citations.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval augmented generation?"}'
```

**Request Body:**
```json
{
  "question": "Your question here"
}
```

**Response:**
```json
{
  "answer": "Retrieval Augmented Generation (RAG) is a technique that enhances Large Language Models by retrieving relevant documents from a knowledge base before generating responses. This ensures answers are grounded in your actual data rather than the model's training data.",
  "sources": [
    "rag_introduction.pdf",
    "ai_fundamentals.txt"
  ]
}
```

---

### Upload Documents

Ingest documents into your knowledge base.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@your-document.pdf"
```

**Supported Formats:**
- PDF (`.pdf`)
- Text (`.txt`)
- HTML (`.html`)
- Word (`.docx`)

**Response:**
```json
{
  "success": true,
  "document_id": "doc_abc123",
  "chunks_created": 15,
  "message": "Document processed successfully"
}
```

---

### Ingest Raw Text

Add text content directly without file upload.

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/text" \
  -d "text=Your knowledge content here..." \
  -d "source_name=my_notes"
```

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700" alt="Divider"/>
</p>

## Integration Examples

### Python

```python
import requests

# Query the RAG API
response = requests.post(
    "http://localhost:8000/query",
    json={"question": "How do I reset my password?"}
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Sources: {data['sources']}")
```

### JavaScript / Node.js

```javascript
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: 'What are the pricing plans?' })
});

const data = await response.json();
console.log('Answer:', data.answer);
console.log('Sources:', data.sources);
```

### cURL

```bash
# Upload a document
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@company_handbook.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the vacation policy?"}'
```

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700" alt="Divider"/>
</p>

## Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | **Yes** | - | Your OpenRouter API key |
| `EMBEDDING_MODEL` | No | `openai/text-embedding-3-small` | Embedding model to use |
| `LLM_MODEL` | No | `openai/gpt-3.5-turbo` | LLM for answer generation |
| `VECTOR_DIMENSION` | No | `1536` | Embedding vector dimension |
| `QDRANT_COLLECTION_NAME` | No | `rag_documents` | Vector collection name |
| `SEED_DEMO_DOCUMENTS` | No | `false` | Load demo documents on startup |

### Vector Store Modes

<table>
<tr>
<td width="33%">

#### In-Memory (Default)
Perfect for development and testing.

```bash
# No additional config needed
uvicorn src.main:app --reload
```

</td>
<td width="33%">

#### Local Server
For persistent storage during development.

```bash
# Run Qdrant locally first
docker run -p 6333:6333 qdrant/qdrant

# Configure
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
```

</td>
<td width="33%">

#### Cloud (Production)
Scalable, managed vector storage.

```bash
export QDRANT_URL=https://xxx.qdrant.io
export QDRANT_API_KEY=your-key
```

</td>
</tr>
</table>

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700" alt="Divider"/>
</p>

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT APPLICATIONS                          │
│              (Web Apps, Mobile Apps, Chatbots, etc.)                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        REST API LAYER (FastAPI)                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐   │
│  │  /health   │  │   /query   │  │  /ingest   │  │   /formats   │   │
│  └────────────┘  └────────────┘  └────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          RAG PIPELINE                                │
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐    │
│  │  Embedding  │──▶│  Retrieval  │──▶│  Answer Generation      │    │
│  │  (OpenRouter)│   │  (Qdrant)   │   │  (LLM via OpenRouter)   │    │
│  └─────────────┘   └─────────────┘   └─────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌─────────────────┐             ┌─────────────────┐
          │  EMBEDDING API  │             │  VECTOR STORE   │
          │  (OpenRouter)   │             │  (Qdrant)       │
          │                 │             │                 │
          │  Converts text  │             │  Stores vectors │
          │  to vectors     │             │  for fast search│
          └─────────────────┘             └─────────────────┘
```

---

## Use Cases

<table>
<tr>
<td width="50%">

### Customer Support
- Automate FAQ responses
- Reduce support ticket volume
- 24/7 availability

### Documentation Search
- Instant answers from docs
- Code examples retrieval
- API reference lookup

</td>
<td width="50%">

### Knowledge Management
- Internal wiki search
- Policy Q&A
- Employee onboarding

### Research & Analysis
- Literature review
- Data extraction
- Report summarization

</td>
</tr>
</table>

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700" alt="Divider"/>
</p>

## Project Structure

```
rag/
├── src/
│   ├── main.py                 # Application entry point
│   ├── api/                    # REST API layer
│   │   ├── app.py              # FastAPI application factory
│   │   ├── routes/             # API endpoints
│   │   │   ├── query.py        # Query endpoint
│   │   │   ├── ingest.py       # Document ingestion
│   │   │   └── health.py       # Health check
│   │   └── schemas/            # Request/Response models
│   ├── core/                   # Configuration & utilities
│   │   ├── config.py           # Settings management
│   │   └── providers.py        # Shared service instances
│   ├── rag/                    # RAG pipeline orchestration
│   │   └── pipeline.py         # Main RAG logic
│   ├── embeddings/             # Embedding providers
│   │   ├── base.py             # Abstract interface
│   │   └── providers/          # Implementations
│   ├── vectorstore/            # Vector database layer
│   │   ├── base.py             # Abstract interface
│   │   └── providers/          # Implementations
│   ├── llm/                    # LLM integration
│   └── ingestion/              # Document processing
│       ├── document_loader/    # File format handlers
│       └── chunking/           # Text splitting strategies
├── tests/                      # Test suite
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Roadmap

- [x] Core API Layer (FastAPI)
- [x] Embedding Integration (OpenRouter)
- [x] Vector Store (Qdrant)
- [x] LLM Integration
- [x] Document Ingestion Pipeline
- [ ] Reranking Module
- [ ] Evaluation Metrics (RAGAS)
- [ ] Observability & Monitoring
- [ ] Authentication & Security
- [ ] Multi-tenancy Support

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700" alt="Divider"/>
</p>

## Support

<p align="center">
  <a href="https://github.com/yourusername/rag-engine/issues">
    <img src="https://img.shields.io/badge/Report-Issue-red?style=for-the-badge&logo=github" alt="Report Issue"/>
  </a>
  <a href="https://github.com/yourusername/rag-engine/discussions">
    <img src="https://img.shields.io/badge/Ask-Question-blue?style=for-the-badge&logo=github" alt="Ask Question"/>
  </a>
</p>

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400" alt="Footer Animation"/>
</p>

<p align="center">
  <strong>Built with AI</strong>
</p>

<p align="center">
  <sub>Transform your documents into intelligent APIs</sub>
</p>
