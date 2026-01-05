"""
Metadata Module
================

This module handles metadata extraction and enrichment for the RAG ingestion pipeline.

Components:
- MetadataExtractor: Extracts metadata from files and chunks
- MetadataEnricher: Adds system metadata and derived information

Usage:
    from src.ingestion.metadata import MetadataExtractor, MetadataEnricher

    extractor = MetadataExtractor()
    enricher = MetadataEnricher()

    # Extract file metadata
    file_meta = extractor.extract_file_metadata("document.pdf")

    # Enrich with system metadata
    enriched = enricher.enrich_chunk_metadata(
        metadata=file_meta,
        chunk_text="This is a chunk...",
        chunk_index=0
    )
"""

from src.ingestion.metadata.extractor import MetadataExtractor
from src.ingestion.metadata.enricher import MetadataEnricher

__all__ = [
    "MetadataExtractor",
    "MetadataEnricher",
]
