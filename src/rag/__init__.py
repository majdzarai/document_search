"""
RAG Package
============

This package contains the core RAG (Retrieval-Augmented Generation) pipeline.

The RAG pipeline is the heart of this service, responsible for:
1. Understanding user queries
2. Retrieving relevant documents
3. Generating accurate answers
4. Providing source citations

Modules:
- pipeline.py: Main RAG pipeline orchestrator
- query_understanding/: Query preprocessing and enhancement
- retrieval/: Document retrieval from vector stores
- generation/: Answer generation using LLMs
"""

# Import the main pipeline class for convenient access
from src.rag.pipeline import RAGPipeline

# Export list for explicit imports
__all__ = ["RAGPipeline"]
