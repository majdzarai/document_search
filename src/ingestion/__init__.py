"""
Document Ingestion Module
==========================

This module provides document ingestion capabilities for the RAG system.
It handles loading, chunking, embedding, and storing documents.

Main Components:
- IngestionService: Main orchestrator for document ingestion
- Document Loaders: Read various file formats (PDF, TXT, HTML, DOCX)
- Chunkers: Split text into smaller pieces
- Metadata: Extract and enrich document metadata

Usage:
    from src.ingestion import IngestionService

    # Create service
    service = IngestionService()

    # Ingest a file
    result = service.ingest_file("/path/to/document.pdf")

    if result.success:
        print(f"Created {result.chunk_count} chunks")
    else:
        print(f"Error: {result.error}")

Submodules:
- document_loader: File format handlers
- chunking: Text splitting strategies
- metadata: Metadata extraction and enrichment
"""

from src.ingestion.service import IngestionService, IngestionResult

__all__ = [
    "IngestionService",
    "IngestionResult",
]
