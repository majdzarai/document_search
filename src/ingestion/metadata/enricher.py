"""
Metadata Enricher
==================

WHAT IS METADATA ENRICHMENT?
----------------------------
Metadata enrichment adds additional information to metadata
that wasn't present in the original document.

This includes:
1. SYSTEM METADATA: Information about when/how the document was processed
   - Ingestion timestamp
   - Processing version
   - System identifiers

2. DERIVED METADATA: Information computed from the content
   - Character count
   - Word count
   - Language detection (future)

3. IDENTIFIERS: Unique IDs for tracking
   - Chunk ID
   - Document ID
   - Batch ID

WHY ENRICH METADATA?
--------------------
1. TRACEABILITY: Know when and how documents were ingested
2. DEBUGGING: Track issues to specific ingestion runs
3. VERSIONING: Manage document updates over time
4. FILTERING: Query by ingestion date, batch, etc.

HOW IT FITS IN THE PIPELINE:
----------------------------
1. Document Loader → extracts content + basic metadata
2. Metadata Extractor → extracts file/chunk metadata
3. Metadata Enricher → adds system/derived metadata
4. Result → complete metadata ready for storage
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


class MetadataEnricher:
    """
    Enriches metadata with system and derived information.

    This class adds additional metadata that helps with:
    - Tracking ingestion history
    - Debugging and auditing
    - Filtering and querying

    Example:
        enricher = MetadataEnricher()

        # Enrich chunk metadata
        enriched = enricher.enrich_chunk_metadata(
            metadata={"source": "doc.pdf"},
            chunk_text="This is the chunk content...",
            chunk_index=0
        )
        print(enriched["ingested_at"])  # "2024-01-15T10:30:00Z"
        print(enriched["word_count"])   # 5
    """

    def __init__(self, system_name: str = "rag-engine"):
        """
        Initialize the metadata enricher.

        Args:
            system_name: Identifier for this RAG system.
                        Useful if you have multiple systems.
        """
        self.system_name = system_name

    def enrich_chunk_metadata(
        self,
        metadata: Dict[str, Any],
        chunk_text: str,
        chunk_index: int,
        document_id: Optional[str] = None,
        batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrich metadata for a single chunk.

        This adds system and derived metadata to an existing
        metadata dictionary.

        Args:
            metadata: Existing metadata dictionary to enrich.
            chunk_text: The text content of this chunk.
            chunk_index: Position of this chunk in the document.
            document_id: Optional ID for the parent document.
            batch_id: Optional ID for the ingestion batch.

        Returns:
            Enriched metadata dictionary.

        Example:
            enriched = enricher.enrich_chunk_metadata(
                metadata={"source": "report.pdf", "file_type": "pdf"},
                chunk_text="Machine learning is a subset of AI.",
                chunk_index=0
            )
            # Adds:
            # - chunk_id: unique identifier
            # - ingested_at: timestamp
            # - char_count: 37
            # - word_count: 7
            # - system: "rag-engine"
        """
        # Start with a copy of existing metadata
        enriched = dict(metadata)

        # =====================================================================
        # Add unique identifiers
        # =====================================================================
        # Generate a unique ID for this chunk
        # This helps with deduplication and updates
        chunk_id = self._generate_chunk_id(chunk_text, chunk_index, metadata)
        enriched["chunk_id"] = chunk_id

        # Add document ID if provided
        if document_id:
            enriched["document_id"] = document_id

        # Add batch ID if provided (useful for bulk ingestion)
        if batch_id:
            enriched["batch_id"] = batch_id

        # =====================================================================
        # Add system metadata
        # =====================================================================
        # Timestamp when this chunk was ingested
        # Using UTC for consistency across timezones
        enriched["ingested_at"] = datetime.now(timezone.utc).isoformat()

        # System identifier
        enriched["system"] = self.system_name

        # =====================================================================
        # Add derived metadata
        # =====================================================================
        # Calculate text statistics
        enriched["char_count"] = len(chunk_text)
        enriched["word_count"] = len(chunk_text.split())

        # Chunk index (ensure it's present)
        enriched["chunk_index"] = chunk_index

        return enriched

    def enrich_document_metadata(
        self,
        metadata: Dict[str, Any],
        full_text: str,
        chunk_count: int
    ) -> Dict[str, Any]:
        """
        Enrich metadata for a complete document.

        This is called once per document, before chunking,
        to add document-level metadata.

        Args:
            metadata: Existing document metadata.
            full_text: The complete document text.
            chunk_count: Number of chunks created from this document.

        Returns:
            Enriched document metadata.

        Example:
            doc_meta = enricher.enrich_document_metadata(
                metadata={"source": "report.pdf"},
                full_text="Long document content...",
                chunk_count=15
            )
            # Adds document-level stats and identifiers
        """
        enriched = dict(metadata)

        # Generate a document ID
        enriched["document_id"] = self._generate_document_id(metadata)

        # Add ingestion timestamp
        enriched["ingested_at"] = datetime.now(timezone.utc).isoformat()

        # Add text statistics
        enriched["total_char_count"] = len(full_text)
        enriched["total_word_count"] = len(full_text.split())
        enriched["chunk_count"] = chunk_count

        # Add system info
        enriched["system"] = self.system_name

        return enriched

    def create_batch_id(self) -> str:
        """
        Create a unique batch ID for an ingestion run.

        A batch ID groups all documents ingested together.
        This is useful for:
        - Tracking which documents were ingested together
        - Rolling back a batch if needed
        - Debugging issues in a specific ingestion run

        Returns:
            A unique batch ID string.

        Example:
            batch_id = enricher.create_batch_id()
            # Returns: "batch_2024-01-15T10-30-00_a1b2c3d4"
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        unique_suffix = str(uuid.uuid4())[:8]
        return f"batch_{timestamp}_{unique_suffix}"

    def _generate_chunk_id(
        self,
        chunk_text: str,
        chunk_index: int,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Generate a unique ID for a chunk.

        The ID is based on:
        - Source document
        - Chunk index
        - A hash of the content

        This ensures:
        - Same content always gets same ID (deterministic)
        - Different chunks get different IDs

        Args:
            chunk_text: The chunk content.
            chunk_index: Position of this chunk.
            metadata: Existing metadata (for source info).

        Returns:
            Unique chunk ID string.
        """
        import hashlib

        # Build components for the ID
        source = metadata.get("source", "unknown")
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]

        return f"{source}_chunk_{chunk_index}_{content_hash}"

    def _generate_document_id(self, metadata: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a document.

        Args:
            metadata: Document metadata.

        Returns:
            Unique document ID string.
        """
        import hashlib

        source = metadata.get("source", "unknown")
        file_path = metadata.get("file_path", "")

        # Create hash from source + path
        content = f"{source}:{file_path}"
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

        return f"doc_{source}_{content_hash}"

    def prepare_for_storage(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for storage in the vector database.

        This ensures all values are serializable and clean.

        Args:
            metadata: Metadata dictionary to prepare.

        Returns:
            Cleaned metadata ready for storage.
        """
        prepared = {}

        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue

            # Convert datetime objects to strings
            if isinstance(value, datetime):
                prepared[key] = value.isoformat()
            # Convert other types to strings if needed
            elif isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif isinstance(value, (list, dict)):
                # Keep lists and dicts as-is (most vector DBs support them)
                prepared[key] = value
            else:
                # Convert other types to string
                prepared[key] = str(value)

        return prepared
