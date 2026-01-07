"""
Document Ingestion Service
===========================

WHAT IS THE INGESTION SERVICE?
------------------------------
The ingestion service is the main orchestrator for adding documents
to the RAG system. It coordinates all the steps needed to:

1. LOAD: Read the document file and extract text
2. CHUNK: Split the text into smaller pieces
3. EMBED: Convert each chunk to a vector
4. STORE: Save chunks and vectors to the database

THE INGESTION PIPELINE:
-----------------------
    User uploads file
           ↓
    Select appropriate loader (PDF, TXT, HTML, DOCX)
           ↓
    Load document → get text + metadata
           ↓
    Split text into chunks
           ↓
    Extract & enrich metadata for each chunk
           ↓
    Embed each chunk → get vectors
           ↓
    Store in vector database (Qdrant)
           ↓
    Return summary to user

WHY A SEPARATE SERVICE?
-----------------------
1. SEPARATION OF CONCERNS: Ingestion logic is separate from API handling
2. REUSABILITY: Can be called from API, CLI, or background jobs
3. TESTABILITY: Can test ingestion without HTTP
4. MAINTAINABILITY: Easy to modify ingestion logic

IMPORTANT DESIGN DECISIONS:
---------------------------
1. Uses EXISTING embedding and vector store providers
   - Does NOT duplicate logic from RAGPipeline
   - Reuses the same configuration

2. Supports multiple file formats via loaders
   - Automatically selects the right loader based on extension

3. Configurable chunking
   - Default: RecursiveCharacterSplitter with overlap

4. Complete metadata tracking
   - Source, chunk position, ingestion time, etc.
"""

import os
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Import document loaders
from src.ingestion.document_loader import (
    DocumentLoader,
    LoadedDocument,
    TextLoader,
    PDFLoader,
    HTMLLoader,
    DOCXLoader
)

# Import chunking
from src.ingestion.chunking import (
    Chunker,
    RecursiveCharacterSplitter,
    SentenceSplitter,
    SemanticSplitter
)

# Import metadata handling
from src.ingestion.metadata import MetadataExtractor, MetadataEnricher

# Import base classes for type hints
from src.embeddings.base import EmbeddingProvider
from src.vectorstore.base import VectorStoreProvider

# =============================================================================
# SHARED PROVIDERS - CRITICAL FOR CORRECT RAG OPERATION
# =============================================================================
# We use SHARED provider instances from src.core.providers to ensure that:
# 1. Documents ingested here are stored in the SAME vector store that
#    the query pipeline searches
# 2. We don't waste memory creating multiple provider instances
#
# WITHOUT shared providers:
#   - IngestionService creates vector store A, stores documents there
#   - RAGPipeline creates vector store B, searches there (finds nothing!)
#   - User's uploaded documents are never found by queries
#
# WITH shared providers (this implementation):
#   - Both use the SAME vector store
#   - Ingested documents are correctly found by queries!
#
# IMPORTANT: Do NOT use create_embedding_provider() or create_vector_store_provider()
# directly in services - always go through src.core.providers to get the shared instance.

from src.core.providers import get_embedding_provider, get_vector_store


@dataclass
class IngestionResult:
    """
    Result of a document ingestion operation.

    Attributes:
        success: Whether ingestion was successful.
        document_name: Name of the ingested document.
        chunk_count: Number of chunks created.
        document_id: Unique ID assigned to the document.
        error: Error message if ingestion failed.
        metadata: Additional information about the ingestion.
    """
    success: bool
    document_name: str
    chunk_count: int
    document_id: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IngestionService:
    """
    Orchestrates document ingestion into the RAG system.

    This service handles the complete process of:
    1. Loading documents from files
    2. Splitting them into chunks
    3. Embedding the chunks
    4. Storing them in the vector database

    Example:
        # Create service with default settings
        service = IngestionService()

        # Ingest a document
        result = service.ingest_file("/path/to/document.pdf")

        print(f"Ingested {result.chunk_count} chunks")
        print(f"Document ID: {result.document_id}")

    Attributes:
        embedding_provider: Provider for creating embeddings.
        vector_store: Provider for storing vectors.
        chunker: Strategy for splitting text into chunks.
        metadata_extractor: For extracting file/chunk metadata.
        metadata_enricher: For adding system metadata.
    """

    # Mapping of file extensions to loaders
    LOADER_MAP: Dict[str, DocumentLoader] = {}

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStoreProvider] = None,
        chunker: Optional[Chunker] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunking_strategy: Optional[str] = None
    ):
        """
        Initialize the ingestion service.

        Args:
            embedding_provider: Optional. Provider for embeddings.
                               If None, creates from settings.

            vector_store: Optional. Provider for vector storage.
                         If None, creates from settings.

            chunker: Optional. Strategy for chunking text.
                    If None, creates based on CHUNKING_STRATEGY config.

            chunk_size: Size of chunks in characters.
                       If None, reads from MAX_CHUNK_SIZE config.
                       Only used if chunker is None.

            chunk_overlap: Overlap between chunks.
                          If None, reads from CHUNK_OVERLAP config.
                          Only used if chunker is None.

            chunking_strategy: Which chunking strategy to use.
                              Options: "recursive", "sentence", "semantic"
                              If None, reads from CHUNKING_STRATEGY config.
                              Only used if chunker is None.

        Example:
            # Default settings (uses config)
            service = IngestionService()

            # Custom chunking
            service = IngestionService(chunk_size=1000, chunk_overlap=100)

            # Force semantic chunking
            service = IngestionService(chunking_strategy="semantic")

            # Custom chunker
            from src.ingestion.chunking import SentenceSplitter
            service = IngestionService(chunker=SentenceSplitter())
        """
        # Import settings for config-driven defaults
        from src.core.config import settings

        print("\n" + "=" * 60)
        print("INITIALIZING INGESTION SERVICE")
        print("=" * 60)

        # =====================================================================
        # Initialize embedding provider (USING SHARED INSTANCE)
        # =====================================================================
        # We use the SHARED embedding provider from src.core.providers
        # This ensures consistency between ingestion and retrieval
        #
        # IMPORTANT: Using the shared provider is CRITICAL because:
        # - The same embedding model must be used for ingestion and queries
        # - If different models are used, vectors won't be comparable
        # - Similarity search would produce incorrect results

        if embedding_provider is not None:
            # Allow override for testing
            self.embedding_provider = embedding_provider
            print("[IngestionService] Using provided embedding provider (override)")
        else:
            # USE SHARED PROVIDER - critical for correct operation
            self.embedding_provider = get_embedding_provider()
            if self.embedding_provider:
                print("[IngestionService] Using SHARED embedding provider")
            else:
                print("[IngestionService] WARNING: Embedding provider not available")

        # =====================================================================
        # Initialize vector store (USING SHARED INSTANCE)
        # =====================================================================
        # We use the SHARED vector store from src.core.providers
        # This is the most important part for correct RAG operation!
        #
        # WHY SHARED IS CRITICAL:
        # -----------------------
        # Without sharing:
        #   1. User uploads document here → stored in IngestionService's store
        #   2. User queries via RAGPipeline → searches Pipeline's store (different!)
        #   3. Result: User's documents are NEVER found!
        #
        # With sharing (this implementation):
        #   1. User uploads document → stored in SHARED store
        #   2. User queries → searches SAME SHARED store
        #   3. Result: User's documents are found correctly!

        if vector_store is not None:
            # Allow override for testing
            self.vector_store = vector_store
            print("[IngestionService] Using provided vector store (override)")
        else:
            # USE SHARED PROVIDER - critical for correct operation
            self.vector_store = get_vector_store()
            if self.vector_store:
                print("[IngestionService] Using SHARED vector store")
            else:
                print("[IngestionService] WARNING: Vector store not available")

        # =====================================================================
        # Initialize chunker (CONFIG-DRIVEN)
        # =====================================================================
        # The chunking strategy can be configured via environment variables:
        # - CHUNKING_STRATEGY: "recursive" (default), "sentence", or "semantic"
        # - MAX_CHUNK_SIZE: Maximum chunk size in characters
        # - MIN_CHUNK_SIZE: Minimum chunk size in characters
        # - CHUNK_OVERLAP: Overlap between chunks
        # - SEMANTIC_SIMILARITY_THRESHOLD: Threshold for semantic chunking

        if chunker is not None:
            self.chunker = chunker
            print("[IngestionService] Using provided chunker")
        else:
            # Get config values (with argument overrides)
            effective_chunk_size = chunk_size if chunk_size is not None else settings.max_chunk_size
            effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
            effective_strategy = chunking_strategy if chunking_strategy is not None else settings.chunking_strategy

            # Create chunker based on strategy
            self.chunker = self._create_chunker(
                strategy=effective_strategy,
                chunk_size=effective_chunk_size,
                chunk_overlap=effective_chunk_overlap,
                min_chunk_size=settings.min_chunk_size,
                similarity_threshold=settings.semantic_similarity_threshold
            )

        # =====================================================================
        # Initialize metadata handlers
        # =====================================================================
        self.metadata_extractor = MetadataExtractor()
        self.metadata_enricher = MetadataEnricher()

        # =====================================================================
        # Initialize document loaders
        # =====================================================================
        self._init_loaders()

        print("=" * 60)
        print("INGESTION SERVICE READY")
        print("=" * 60 + "\n")

    def _init_loaders(self) -> None:
        """
        Initialize document loaders for each supported file type.

        This creates a mapping of file extensions to loader instances.
        """
        # Create loader instances
        text_loader = TextLoader()
        pdf_loader = PDFLoader()
        html_loader = HTMLLoader()
        docx_loader = DOCXLoader()

        # Build extension → loader mapping
        self.loaders: Dict[str, DocumentLoader] = {}

        for loader in [text_loader, pdf_loader, html_loader, docx_loader]:
            for ext in loader.supported_extensions:
                self.loaders[ext.lower()] = loader

        print(f"[IngestionService] Registered loaders for: {list(self.loaders.keys())}")

    def get_loader(self, file_path: str) -> Optional[DocumentLoader]:
        """
        Get the appropriate loader for a file.

        Args:
            file_path: Path to the file.

        Returns:
            DocumentLoader instance or None if unsupported.
        """
        _, ext = os.path.splitext(file_path)
        return self.loaders.get(ext.lower())

    def ingest_file(
        self,
        file_path: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """
        Ingest a single file into the RAG system.

        This is the main entry point for document ingestion.
        It handles the complete pipeline:
        1. Load the document
        2. Split into chunks
        3. Embed each chunk
        4. Store in vector database

        Args:
            file_path: Path to the file to ingest.
            custom_metadata: Optional. Additional metadata to attach.

        Returns:
            IngestionResult with success status and details.

        Example:
            result = service.ingest_file("/path/to/report.pdf")
            if result.success:
                print(f"Created {result.chunk_count} chunks")
            else:
                print(f"Error: {result.error}")
        """
        file_name = os.path.basename(file_path)
        print(f"\n[Ingestion] Starting ingestion of: {file_name}")

        # =====================================================================
        # Validate prerequisites
        # =====================================================================
        if self.embedding_provider is None:
            return IngestionResult(
                success=False,
                document_name=file_name,
                chunk_count=0,
                document_id="",
                error="Embedding provider not configured"
            )

        if self.vector_store is None:
            return IngestionResult(
                success=False,
                document_name=file_name,
                chunk_count=0,
                document_id="",
                error="Vector store not configured"
            )

        # =====================================================================
        # Step 1: Select loader and load document
        # =====================================================================
        print(f"[Ingestion] Step 1: Loading document...")

        loader = self.get_loader(file_path)
        if loader is None:
            _, ext = os.path.splitext(file_path)
            return IngestionResult(
                success=False,
                document_name=file_name,
                chunk_count=0,
                document_id="",
                error=f"Unsupported file type: {ext}"
            )

        try:
            loaded_doc = loader.load(file_path)
            print(f"[Ingestion] Loaded {len(loaded_doc.text)} characters")
        except Exception as e:
            return IngestionResult(
                success=False,
                document_name=file_name,
                chunk_count=0,
                document_id="",
                error=f"Failed to load document: {str(e)}"
            )

        # =====================================================================
        # Step 2: Split into chunks
        # =====================================================================
        print(f"[Ingestion] Step 2: Chunking document...")

        try:
            chunks = self.chunker.split(loaded_doc.text)
            print(f"[Ingestion] Created {len(chunks)} chunks")
        except Exception as e:
            return IngestionResult(
                success=False,
                document_name=file_name,
                chunk_count=0,
                document_id="",
                error=f"Failed to chunk document: {str(e)}"
            )

        if not chunks:
            return IngestionResult(
                success=False,
                document_name=file_name,
                chunk_count=0,
                document_id="",
                error="Document produced no chunks"
            )

        # =====================================================================
        # Step 3: Prepare metadata for each chunk
        # =====================================================================
        print(f"[Ingestion] Step 3: Preparing metadata...")

        # Extract file metadata
        file_metadata = self.metadata_extractor.extract_file_metadata(file_path)

        # Combine with document metadata from loader
        base_metadata = self.metadata_extractor.combine_metadata(
            file_metadata,
            loaded_doc.metadata,
            custom_metadata or {}
        )

        # Generate document ID
        document_id = self.metadata_enricher._generate_document_id(base_metadata)

        # =====================================================================
        # Step 4: Embed each chunk
        # =====================================================================
        print(f"[Ingestion] Step 4: Embedding {len(chunks)} chunks...")

        try:
            embeddings = self.embedding_provider.embed_texts(chunks)
            print(f"[Ingestion] Generated {len(embeddings)} embeddings")
        except Exception as e:
            return IngestionResult(
                success=False,
                document_name=file_name,
                chunk_count=0,
                document_id=document_id,
                error=f"Failed to embed chunks: {str(e)}"
            )

        # =====================================================================
        # Step 5: Prepare data for storage
        # =====================================================================
        print(f"[Ingestion] Step 5: Preparing for storage...")

        chunk_ids = []
        chunk_texts = []
        chunk_metadata_list = []

        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            # Enrich metadata for this chunk
            chunk_metadata = self.metadata_enricher.enrich_chunk_metadata(
                metadata=dict(base_metadata),
                chunk_text=chunk_text,
                chunk_index=i,
                document_id=document_id
            )

            # Add total chunks info
            chunk_metadata["total_chunks"] = len(chunks)

            # Prepare for storage
            prepared_metadata = self.metadata_enricher.prepare_for_storage(chunk_metadata)

            chunk_ids.append(chunk_metadata["chunk_id"])
            chunk_texts.append(chunk_text)
            chunk_metadata_list.append(prepared_metadata)

        # =====================================================================
        # Step 6: Store in vector database
        # =====================================================================
        print(f"[Ingestion] Step 6: Storing in vector database...")

        try:
            success = self.vector_store.upsert(
                ids=chunk_ids,
                embeddings=embeddings,
                texts=chunk_texts,
                metadata=chunk_metadata_list
            )

            if not success:
                return IngestionResult(
                    success=False,
                    document_name=file_name,
                    chunk_count=0,
                    document_id=document_id,
                    error="Failed to store chunks in vector database"
                )

        except Exception as e:
            return IngestionResult(
                success=False,
                document_name=file_name,
                chunk_count=0,
                document_id=document_id,
                error=f"Failed to store chunks: {str(e)}"
            )

        # =====================================================================
        # Step 7: Return success result
        # =====================================================================
        print(f"[Ingestion] SUCCESS: Ingested {len(chunks)} chunks from {file_name}")

        return IngestionResult(
            success=True,
            document_name=file_name,
            chunk_count=len(chunks),
            document_id=document_id,
            metadata={
                "file_type": base_metadata.get("file_type", "unknown"),
                "char_count": len(loaded_doc.text),
                "chunk_ids": chunk_ids
            }
        )

    def ingest_text(
        self,
        text: str,
        source_name: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """
        Ingest raw text directly (without a file).

        This is useful when you have text from other sources
        like databases, APIs, or user input.

        Args:
            text: The text content to ingest.
            source_name: Name to identify this content.
            custom_metadata: Optional. Additional metadata.

        Returns:
            IngestionResult with success status and details.

        Example:
            result = service.ingest_text(
                text="This is some important content...",
                source_name="user_input",
                custom_metadata={"category": "notes"}
            )
        """
        print(f"\n[Ingestion] Starting text ingestion: {source_name}")

        # Validate
        if not text or not text.strip():
            return IngestionResult(
                success=False,
                document_name=source_name,
                chunk_count=0,
                document_id="",
                error="Empty text provided"
            )

        if self.embedding_provider is None or self.vector_store is None:
            return IngestionResult(
                success=False,
                document_name=source_name,
                chunk_count=0,
                document_id="",
                error="Embedding provider or vector store not configured"
            )

        # Create metadata
        base_metadata = {
            "source": source_name,
            "file_type": "text",
            **(custom_metadata or {})
        }

        # Chunk the text
        chunks = self.chunker.split(text)
        if not chunks:
            return IngestionResult(
                success=False,
                document_name=source_name,
                chunk_count=0,
                document_id="",
                error="Text produced no chunks"
            )

        # Generate document ID
        document_id = self.metadata_enricher._generate_document_id(base_metadata)

        # Embed chunks
        try:
            embeddings = self.embedding_provider.embed_texts(chunks)
        except Exception as e:
            return IngestionResult(
                success=False,
                document_name=source_name,
                chunk_count=0,
                document_id=document_id,
                error=f"Failed to embed: {str(e)}"
            )

        # Prepare for storage
        chunk_ids = []
        chunk_texts = []
        chunk_metadata_list = []

        for i, (chunk_text, _) in enumerate(zip(chunks, embeddings)):
            chunk_metadata = self.metadata_enricher.enrich_chunk_metadata(
                metadata=dict(base_metadata),
                chunk_text=chunk_text,
                chunk_index=i,
                document_id=document_id
            )
            chunk_metadata["total_chunks"] = len(chunks)

            chunk_ids.append(chunk_metadata["chunk_id"])
            chunk_texts.append(chunk_text)
            chunk_metadata_list.append(
                self.metadata_enricher.prepare_for_storage(chunk_metadata)
            )

        # Store
        try:
            success = self.vector_store.upsert(
                ids=chunk_ids,
                embeddings=embeddings,
                texts=chunk_texts,
                metadata=chunk_metadata_list
            )
            if not success:
                raise Exception("Upsert returned False")
        except Exception as e:
            return IngestionResult(
                success=False,
                document_name=source_name,
                chunk_count=0,
                document_id=document_id,
                error=f"Failed to store: {str(e)}"
            )

        return IngestionResult(
            success=True,
            document_name=source_name,
            chunk_count=len(chunks),
            document_id=document_id
        )

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of file extensions (e.g., [".pdf", ".txt", ".html"])
        """
        return list(self.loaders.keys())

    def _create_chunker(
        self,
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_size: int,
        similarity_threshold: float
    ) -> Chunker:
        """
        Create a chunker based on the specified strategy.

        This method creates the appropriate chunker instance based on
        the CHUNKING_STRATEGY configuration.

        Args:
            strategy: Chunking strategy ("recursive", "sentence", "semantic").
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between chunks.
            min_chunk_size: Minimum chunk size in characters.
            similarity_threshold: Threshold for semantic chunking.

        Returns:
            A Chunker instance configured according to the parameters.

        Supported strategies:
        - "recursive" (default): RecursiveCharacterSplitter
          Best for: General documents, preserves structure
          Pros: Fast, no API calls needed
          Cons: May split mid-topic

        - "sentence": SentenceSplitter
          Best for: Articles, news, conversational text
          Pros: Respects sentence boundaries
          Cons: May create uneven chunks

        - "semantic": SemanticSplitter
          Best for: Technical docs, topic-heavy content
          Pros: Chunks represent coherent ideas
          Cons: Slower, uses embedding API
        """
        strategy_lower = strategy.lower().strip()

        if strategy_lower == "semantic":
            # Semantic chunking - uses embeddings for topic detection
            print(f"[IngestionService] Creating SemanticSplitter")
            print(f"[IngestionService] Threshold={similarity_threshold}, Size={min_chunk_size}-{chunk_size}")

            # SemanticSplitter will get embedding provider from shared providers
            chunker = SemanticSplitter(
                embedding_provider=self.embedding_provider,
                similarity_threshold=similarity_threshold,
                min_chunk_size=min_chunk_size,
                max_chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return chunker

        elif strategy_lower == "sentence":
            # Sentence-based chunking
            print(f"[IngestionService] Creating SentenceSplitter (size={chunk_size})")

            chunker = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size
            )
            return chunker

        else:
            # Default: Recursive character splitting
            if strategy_lower != "recursive":
                print(f"[IngestionService] Unknown strategy '{strategy}', using 'recursive'")

            print(f"[IngestionService] Creating RecursiveCharacterSplitter (size={chunk_size})")

            chunker = RecursiveCharacterSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return chunker
