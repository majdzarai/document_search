"""
Document Ingestion Module (Enhanced)
=====================================

This module provides document ingestion capabilities for the RAG system.
It handles loading, chunking, embedding, and storing documents.

Main Components:
- IngestionService: Main orchestrator for document ingestion
- Document Loaders: Read various file formats (PDF, TXT, HTML, DOCX)
- Chunkers: Split text into smaller pieces (recursive, sentence, semantic)
- Metadata: Extract and enrich document metadata
- Text Utils: Text normalization and sentence splitting utilities

Features (STEP 5 ENHANCEMENTS):
- PDF OCR fallback for scanned documents (optional)
- Text normalization (whitespace, control chars)
- Semantic chunking using embeddings
- Config-driven chunking strategy selection

Configuration (via environment variables):
- CHUNKING_STRATEGY: "recursive" (default), "sentence", or "semantic"
- MAX_CHUNK_SIZE: Maximum chunk size (default: 512)
- MIN_CHUNK_SIZE: Minimum chunk size (default: 100)
- CHUNK_OVERLAP: Overlap between chunks (default: 50)
- SEMANTIC_SIMILARITY_THRESHOLD: For semantic chunking (default: 0.75)
- ENABLE_PDF_OCR: Enable OCR for scanned PDFs (default: false)

Usage:
    from src.ingestion import IngestionService

    # Create service (uses config defaults)
    service = IngestionService()

    # Ingest a file
    result = service.ingest_file("/path/to/document.pdf")

    if result.success:
        print(f"Created {result.chunk_count} chunks")
    else:
        print(f"Error: {result.error}")

    # Force semantic chunking
    service = IngestionService(chunking_strategy="semantic")

Submodules:
- document_loader: File format handlers (PDF, TXT, HTML, DOCX)
- chunking: Text splitting strategies (recursive, sentence, semantic)
- metadata: Metadata extraction and enrichment
- text_utils: Text normalization utilities
"""

from src.ingestion.service import IngestionService, IngestionResult

__all__ = [
    "IngestionService",
    "IngestionResult",
]
