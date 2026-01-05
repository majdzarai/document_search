"""
Chunking Module
================

This module provides text chunking strategies for the RAG ingestion pipeline.
Chunking splits large documents into smaller pieces for embedding and retrieval.

Available Chunkers:
- RecursiveCharacterSplitter: Smart recursive splitting (recommended)
- SentenceSplitter: Splits on sentence boundaries
- SemanticSplitter: Splits by semantic meaning (STUB - not yet implemented)

Usage:
    from src.ingestion.chunking import RecursiveCharacterSplitter

    splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split(document_text)

Recommended Settings:
- chunk_size: 300-1000 characters (500 is a good default)
- chunk_overlap: 10-20% of chunk_size (50 is good for 500)
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
