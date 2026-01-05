"""
Document Ingestion API Endpoint
================================

WHAT IS THIS ENDPOINT?
----------------------
This endpoint allows users to upload documents for ingestion into the RAG system.
Once ingested, documents can be queried through the /query endpoint.

ENDPOINT: POST /api/v1/ingest
-----------------------------
Accepts file uploads (multipart/form-data) and ingests them into the RAG system.

THE INGESTION PROCESS:
----------------------
1. User uploads a file (PDF, TXT, HTML, or DOCX)
2. File is temporarily saved
3. IngestionService processes the file:
   - Load → Chunk → Embed → Store
4. Temporary file is deleted
5. Result returned to user

SUPPORTED FILE TYPES:
---------------------
- PDF (.pdf)
- Text (.txt, .md)
- HTML (.html, .htm)
- Word Documents (.docx)

EXAMPLE USAGE:
--------------
Using curl:
    curl -X POST http://localhost:8000/api/v1/ingest \
         -F "file=@/path/to/document.pdf"

Using Python requests:
    import requests
    files = {"file": open("document.pdf", "rb")}
    response = requests.post("http://localhost:8000/api/v1/ingest", files=files)
    print(response.json())

RESPONSE FORMAT:
----------------
Success:
{
    "success": true,
    "message": "Document ingested successfully",
    "document_name": "document.pdf",
    "document_id": "doc_document.pdf_a1b2c3d4",
    "chunk_count": 15,
    "metadata": {
        "file_type": "pdf",
        "char_count": 12500
    }
}

Failure:
{
    "success": false,
    "message": "Failed to ingest document",
    "error": "Unsupported file type: .xyz"
}
"""

import os
import tempfile
import shutil
from typing import Optional, Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel

# Import the ingestion service
from src.ingestion import IngestionService


# =============================================================================
# Response Models
# =============================================================================

class IngestResponse(BaseModel):
    """
    Response model for document ingestion.

    Attributes:
        success: Whether ingestion was successful.
        message: Human-readable status message.
        document_name: Name of the ingested document.
        document_id: Unique ID assigned to the document.
        chunk_count: Number of chunks created.
        metadata: Additional information about the ingestion.
        error: Error message if ingestion failed.
    """
    success: bool
    message: str
    document_name: str = ""
    document_id: str = ""
    chunk_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SupportedFormatsResponse(BaseModel):
    """
    Response model for listing supported file formats.
    """
    formats: list
    description: str


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(
    prefix="/api/v1",
    tags=["ingestion"]
)

# =============================================================================
# SHARED INGESTION SERVICE INSTANCE
# =============================================================================
# We use lazy initialization to ensure the IngestionService uses the
# SHARED providers from src.core.providers.
#
# WHY IS SHARED CRITICAL?
# -----------------------
# The IngestionService must use the SAME vector store as the RAGPipeline.
# Otherwise:
#   - Documents uploaded here would go to one vector store
#   - Queries would search a different vector store
#   - User's documents would never be found!
#
# The IngestionService now uses get_vector_store() from src.core.providers,
# which returns the SAME vector store instance used by RAGPipeline.

_ingestion_service: Optional[IngestionService] = None


def get_ingestion_service() -> IngestionService:
    """
    Get the shared ingestion service instance (lazy initialization).

    This function ensures we have a single IngestionService that uses:
    1. The SHARED vector store (same as RAGPipeline queries)
    2. The SHARED embedding provider

    WHY SHARED VECTOR STORE MATTERS:
    ---------------------------------
    When users upload documents via /api/v1/ingest:
    1. Documents are chunked and embedded
    2. Chunks are stored in the SHARED vector store
    3. Later, when users query via /query, the RAGPipeline
       searches the SAME SHARED vector store
    4. User's uploaded documents are correctly found!

    Returns:
        The shared IngestionService instance.
    """
    global _ingestion_service

    if _ingestion_service is None:
        print("[IngestRoute] Creating shared ingestion service...")
        _ingestion_service = IngestionService()
        print("[IngestRoute] Ingestion service ready (using SHARED vector store)")

    return _ingestion_service


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest")
) -> IngestResponse:
    """
    Ingest a document into the RAG system.

    This endpoint accepts a file upload, processes it through the
    ingestion pipeline, and stores the resulting chunks in the
    vector database for later retrieval.

    Args:
        file: The document file to ingest.
              Supported formats: PDF, TXT, HTML, DOCX

    Returns:
        IngestResponse with success status and ingestion details.

    Raises:
        HTTPException: If there's a server error during ingestion.

    Example:
        curl -X POST http://localhost:8000/api/v1/ingest \
             -F "file=@document.pdf"
    """
    print(f"\n[IngestAPI] Received file: {file.filename}")
    print(f"[IngestAPI] Content type: {file.content_type}")

    # =========================================================================
    # Validate file
    # =========================================================================
    if not file.filename:
        return IngestResponse(
            success=False,
            message="No file provided",
            error="File name is empty"
        )

    # Get the ingestion service
    try:
        service = get_ingestion_service()
    except Exception as e:
        return IngestResponse(
            success=False,
            message="Ingestion service unavailable",
            document_name=file.filename or "",
            error=f"Could not initialize ingestion service: {str(e)}"
        )

    # Check if file type is supported
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in service.get_supported_extensions():
        return IngestResponse(
            success=False,
            message="Unsupported file type",
            document_name=file.filename,
            error=f"File type '{ext}' is not supported. "
                  f"Supported: {', '.join(service.get_supported_extensions())}"
        )

    # =========================================================================
    # Save file temporarily
    # =========================================================================
    # We need to save the uploaded file to disk because:
    # 1. Some loaders (like pypdf) require file paths
    # 2. It's more memory-efficient for large files

    temp_dir = None
    temp_path = None

    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix="rag_ingest_")
        temp_path = os.path.join(temp_dir, file.filename)

        print(f"[IngestAPI] Saving to temp file: {temp_path}")

        # Save uploaded file content
        with open(temp_path, "wb") as f:
            # Read in chunks to handle large files
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                f.write(chunk)

        print(f"[IngestAPI] File saved, size: {os.path.getsize(temp_path)} bytes")

        # =====================================================================
        # Ingest the file
        # =====================================================================
        result = service.ingest_file(temp_path)

        # Build response
        if result.success:
            return IngestResponse(
                success=True,
                message="Document ingested successfully",
                document_name=result.document_name,
                document_id=result.document_id,
                chunk_count=result.chunk_count,
                metadata=result.metadata
            )
        else:
            return IngestResponse(
                success=False,
                message="Failed to ingest document",
                document_name=result.document_name,
                document_id=result.document_id,
                error=result.error
            )

    except Exception as e:
        print(f"[IngestAPI] ERROR: {str(e)}")
        return IngestResponse(
            success=False,
            message="Server error during ingestion",
            document_name=file.filename or "",
            error=str(e)
        )

    finally:
        # =====================================================================
        # Cleanup: Delete temporary files
        # =====================================================================
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"[IngestAPI] Cleaned up temp directory")
            except Exception as e:
                print(f"[IngestAPI] Warning: Could not clean up temp dir: {e}")


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    text: str = Form(..., description="Text content to ingest"),
    source_name: str = Form("user_input", description="Name for this content")
) -> IngestResponse:
    """
    Ingest raw text directly into the RAG system.

    This endpoint accepts text content (without a file) and
    ingests it into the vector database.

    Args:
        text: The text content to ingest.
        source_name: A name to identify this content.

    Returns:
        IngestResponse with success status and details.

    Example:
        curl -X POST http://localhost:8000/api/v1/ingest/text \
             -d "text=This is some content to ingest" \
             -d "source_name=my_notes"
    """
    print(f"\n[IngestAPI] Received text ingestion: {source_name}")
    print(f"[IngestAPI] Text length: {len(text)} characters")

    if not text or not text.strip():
        return IngestResponse(
            success=False,
            message="Empty text provided",
            error="Text content cannot be empty"
        )

    try:
        service = get_ingestion_service()
        result = service.ingest_text(text, source_name)

        if result.success:
            return IngestResponse(
                success=True,
                message="Text ingested successfully",
                document_name=result.document_name,
                document_id=result.document_id,
                chunk_count=result.chunk_count,
                metadata=result.metadata
            )
        else:
            return IngestResponse(
                success=False,
                message="Failed to ingest text",
                document_name=result.document_name,
                error=result.error
            )

    except Exception as e:
        print(f"[IngestAPI] ERROR: {str(e)}")
        return IngestResponse(
            success=False,
            message="Server error during ingestion",
            document_name=source_name,
            error=str(e)
        )


@router.get("/ingest/formats", response_model=SupportedFormatsResponse)
async def get_supported_formats() -> SupportedFormatsResponse:
    """
    Get list of supported file formats for ingestion.

    Returns:
        List of supported file extensions.

    Example:
        curl http://localhost:8000/api/v1/ingest/formats
    """
    try:
        service = get_ingestion_service()
        formats = service.get_supported_extensions()
    except Exception:
        # Return default formats if service fails to initialize
        formats = [".txt", ".pdf", ".html", ".htm", ".docx", ".md"]

    return SupportedFormatsResponse(
        formats=formats,
        description="Supported file formats for document ingestion"
    )
