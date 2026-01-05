"""
Metadata Extractor
===================

WHAT IS METADATA?
-----------------
Metadata is "data about data" - information that describes a document.
For RAG systems, metadata helps:

1. IDENTIFY SOURCES: Know which document a chunk came from
2. FILTER RESULTS: Search only in specific types of documents
3. PROVIDE CONTEXT: Show users where information came from
4. DEBUG ISSUES: Track which documents have problems

TYPES OF METADATA:
------------------
1. FILE METADATA: Information from the file system
   - Filename
   - File type (extension)
   - File size
   - Creation/modification dates

2. CONTENT METADATA: Information extracted from the content
   - Title (from PDF/DOCX properties or HTML title)
   - Author
   - Page count
   - Word count

3. CHUNK METADATA: Information about individual chunks
   - Chunk index (position in document)
   - Start/end positions
   - Parent document ID

WHAT DOES THIS MODULE DO?
-------------------------
This module extracts metadata from documents and their chunks.
It works alongside the document loaders to capture useful information.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional


class MetadataExtractor:
    """
    Extracts metadata from documents and files.

    This class provides methods to extract various types of metadata
    that are useful for RAG systems.

    Example:
        extractor = MetadataExtractor()

        # Extract file metadata
        file_meta = extractor.extract_file_metadata("/path/to/doc.pdf")
        print(file_meta["source"])  # "doc.pdf"
        print(file_meta["file_type"])  # "pdf"

        # Combine with document metadata
        full_meta = extractor.combine_metadata(file_meta, doc_meta)
    """

    def extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a file's properties.

        This extracts information available from the file system:
        - Filename
        - Extension/type
        - Size
        - Modification time

        Args:
            file_path: Path to the file.

        Returns:
            Dictionary containing file metadata.

        Example:
            meta = extractor.extract_file_metadata("report.pdf")
            # Returns: {
            #     "source": "report.pdf",
            #     "file_type": "pdf",
            #     "file_path": "/full/path/to/report.pdf",
            #     "file_size_bytes": 1234567,
            #     "modified_at": "2024-01-15T10:30:00"
            # }
        """
        # Get absolute path for consistency
        abs_path = os.path.abspath(file_path)

        # Extract filename
        file_name = os.path.basename(file_path)

        # Extract extension (without the dot, lowercase)
        _, ext = os.path.splitext(file_path)
        file_type = ext.lstrip(".").lower()

        # Get file stats if file exists
        metadata = {
            "source": file_name,
            "file_type": file_type,
            "file_path": abs_path,
        }

        if os.path.exists(file_path):
            stats = os.stat(file_path)
            metadata["file_size_bytes"] = stats.st_size

            # Get modification time
            mod_time = datetime.fromtimestamp(stats.st_mtime)
            metadata["modified_at"] = mod_time.isoformat()

            # Get creation time (platform-dependent)
            try:
                # On Windows, st_ctime is creation time
                # On Unix, st_ctime is last metadata change
                create_time = datetime.fromtimestamp(stats.st_ctime)
                metadata["created_at"] = create_time.isoformat()
            except Exception:
                pass

        return metadata

    def extract_chunk_metadata(
        self,
        chunk_index: int,
        total_chunks: int,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        parent_doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata for a specific chunk.

        This creates metadata that identifies where a chunk
        fits within its parent document.

        Args:
            chunk_index: Position of this chunk (0-indexed).
            total_chunks: Total number of chunks from the document.
            start_char: Character offset where chunk starts (optional).
            end_char: Character offset where chunk ends (optional).
            parent_doc_id: ID of the parent document (optional).

        Returns:
            Dictionary containing chunk-specific metadata.

        Example:
            meta = extractor.extract_chunk_metadata(
                chunk_index=2,
                total_chunks=10,
                start_char=1500,
                end_char=2000
            )
            # Returns: {
            #     "chunk_index": 2,
            #     "total_chunks": 10,
            #     "chunk_position": "3 of 10",
            #     "start_char": 1500,
            #     "end_char": 2000
            # }
        """
        metadata = {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_position": f"{chunk_index + 1} of {total_chunks}"
        }

        if start_char is not None:
            metadata["start_char"] = start_char
        if end_char is not None:
            metadata["end_char"] = end_char
        if parent_doc_id is not None:
            metadata["parent_doc_id"] = parent_doc_id

        return metadata

    def combine_metadata(
        self,
        *metadata_dicts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine multiple metadata dictionaries into one.

        Later dictionaries override earlier ones if keys conflict.

        Args:
            *metadata_dicts: Variable number of metadata dictionaries.

        Returns:
            Combined metadata dictionary.

        Example:
            file_meta = {"source": "doc.pdf", "file_type": "pdf"}
            doc_meta = {"title": "My Document", "author": "John"}
            chunk_meta = {"chunk_index": 0}

            combined = extractor.combine_metadata(file_meta, doc_meta, chunk_meta)
            # Returns: {
            #     "source": "doc.pdf",
            #     "file_type": "pdf",
            #     "title": "My Document",
            #     "author": "John",
            #     "chunk_index": 0
            # }
        """
        result = {}
        for meta in metadata_dicts:
            if meta:
                result.update(meta)
        return result

    def clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata by removing None values and empty strings.

        This ensures the metadata stored in the vector database
        is clean and consistent.

        Args:
            metadata: Dictionary to clean.

        Returns:
            Cleaned dictionary with no None or empty values.

        Example:
            meta = {"title": "Doc", "author": None, "subject": ""}
            clean = extractor.clean_metadata(meta)
            # Returns: {"title": "Doc"}
        """
        return {
            key: value
            for key, value in metadata.items()
            if value is not None and value != ""
        }
