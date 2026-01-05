"""
RAG Service Source Package
===========================

This is the root package for the RAG Service application.

Package Structure:
- api/: FastAPI application and HTTP handlers
- rag/: Core RAG pipeline logic
- vectorstore/: Vector database abstractions
- embeddings/: Embedding model abstractions
- reranker/: Reranking model abstractions
- llm/: LLM provider abstractions
- ingestion/: Document ingestion pipeline
- evaluation/: RAG evaluation metrics
- observability/: Logging, tracing, metrics
- core/: Configuration and shared utilities
- security/: Authentication and authorization
"""

__version__ = "0.1.0"
