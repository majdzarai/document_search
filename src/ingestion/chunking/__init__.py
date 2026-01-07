"""
Chunking Module
================

This module provides text chunking strategies for the RAG ingestion pipeline.
Chunking splits large documents into smaller pieces for embedding and retrieval.

Available Chunkers:
- RecursiveCharacterSplitter: Smart recursive splitting (default, fastest)
- SentenceSplitter: Splits on sentence boundaries
- SemanticSplitter: Splits by semantic meaning using embeddings (best quality)

Configuration (via environment variables):
- CHUNKING_STRATEGY: "recursive" (default), "sentence", or "semantic"
- MAX_CHUNK_SIZE: Maximum characters per chunk (default: 512)
- MIN_CHUNK_SIZE: Minimum characters per chunk (default: 100)
- CHUNK_OVERLAP: Overlap between chunks (default: 50)
- SEMANTIC_SIMILARITY_THRESHOLD: For semantic chunking (default: 0.75)

Usage:
    # Using RecursiveCharacterSplitter (default)
    from src.ingestion.chunking import RecursiveCharacterSplitter

    splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split(document_text)

    # Using SemanticSplitter (requires embedding provider)
    from src.ingestion.chunking import SemanticSplitter
    from src.core.providers import get_embedding_provider

    embedder = get_embedding_provider()
    splitter = SemanticSplitter(embedding_provider=embedder)
    chunks = splitter.split(document_text)

Recommended Settings:
- chunk_size: 300-1000 characters (512 is a good default)
- chunk_overlap: 10-20% of chunk_size (50 is good for 500)
- semantic_threshold: 0.5-0.8 (lower = more chunks)
"""

from src.ingestion.chunking.base import Chunker, Chunk
from src.ingestion.chunking.recursive_splitter import RecursiveCharacterSplitter
from src.ingestion.chunking.sentence_splitter import SentenceSplitter
from src.ingestion.chunking.semantic_splitter import SemanticSplitter

__all__ = [
    "Chunker",
    "Chunk",
    "RecursiveCharacterSplitter",
    "SentenceSplitter",
    "SemanticSplitter",
]
