"""
Shared Provider Registry (Singleton Pattern)
=============================================

WHAT IS THIS MODULE?
--------------------
This module provides a SINGLE, SHARED instance of core providers (embedding,
vector store, LLM) that the entire application uses. This is critical for
correct RAG operation.

WHY IS THIS NECESSARY?
----------------------
In a RAG system, the ingestion and query pipelines MUST use the same vector
store instance. Otherwise:

    PROBLEM (without shared providers):
    -----------------------------------
    1. User uploads document via /api/v1/ingest
    2. IngestionService creates its OWN vector store instance
    3. Document chunks are embedded and stored in that instance
    4. User queries via /query
    5. RAGPipeline creates a DIFFERENT vector store instance
    6. Query searches the EMPTY pipeline vector store (or one with only demo docs)
    7. User's uploaded documents are NEVER found!

    SOLUTION (with shared providers):
    ---------------------------------
    1. Application startup creates ONE shared vector store
    2. User uploads document via /api/v1/ingest
    3. IngestionService uses the SHARED vector store
    4. Document chunks are stored in the shared instance
    5. User queries via /query
    6. RAGPipeline uses the SAME SHARED vector store
    7. Query correctly finds the user's uploaded documents!

THE SINGLETON PATTERN:
----------------------
We use the "singleton" pattern to ensure only ONE instance exists:
- `get_embedding_provider()` always returns the same instance
- `get_vector_store()` always returns the same instance
- `get_llm_provider()` always returns the same instance

WHEN TO USE THIS MODULE:
------------------------
ALWAYS use this module to get providers in:
- RAGPipeline
- IngestionService
- API routes

NEVER call `create_embedding_provider()` or `create_vector_store_provider()`
directly from services - always go through this module.

PERSISTENCE MODES:
------------------
The vector store can run in different modes:

1. IN-MEMORY (default for development):
   - Data is stored in RAM
   - Data is LOST when the application restarts
   - Perfect for development and testing
   - No configuration needed

2. LOCAL SERVER:
   - Run Qdrant: docker run -p 6333:6333 qdrant/qdrant
   - Set QDRANT_HOST=localhost and QDRANT_PORT=6333
   - Data PERSISTS across application restarts
   - Good for local development with persistence

3. QDRANT CLOUD:
   - Set QDRANT_URL and QDRANT_API_KEY
   - Data PERSISTS in the cloud
   - Production-ready with backups and scaling

IMPORTANT FOR PRODUCTION:
-------------------------
In production, you should:
1. Use persistent storage (local server or cloud)
2. Set SEED_DEMO_DOCUMENTS=false (or don't set it)
3. Never rely on in-memory mode for real data
"""

from typing import Optional
import threading

# Import base classes for type hints
from src.embeddings.base import EmbeddingProvider
from src.vectorstore.base import VectorStoreProvider
from src.llm.base import LLMProvider
from src.reranker.base import RerankerProvider

# Import factories for creating providers
from src.embeddings.factory import create_embedding_provider
from src.vectorstore.factory import create_vector_store_provider
from src.llm.factory import create_llm_provider
from src.reranker.factory import create_reranker_provider


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================
# These module-level variables hold the shared instances.
# They start as None and are lazily initialized on first access.

_embedding_provider: Optional[EmbeddingProvider] = None
_vector_store: Optional[VectorStoreProvider] = None
_llm_provider: Optional[LLMProvider] = None
_reranker: Optional[RerankerProvider] = None

# Thread lock for thread-safe initialization
# This prevents race conditions if multiple threads try to initialize at once
_lock = threading.Lock()

# Initialization status flags
_embedding_initialized = False
_vector_store_initialized = False
_llm_initialized = False
_reranker_initialized = False


# =============================================================================
# GETTER FUNCTIONS
# =============================================================================

def get_embedding_provider() -> Optional[EmbeddingProvider]:
    """
    Get the shared embedding provider instance.

    This function returns the SAME embedding provider instance every time
    it's called. If the provider hasn't been initialized yet, it creates one.

    WHY A SHARED INSTANCE?
    ----------------------
    - Embedding providers may cache connections or models
    - Creating multiple instances wastes memory
    - Ensures consistency across the application

    Returns:
        The shared EmbeddingProvider instance, or None if initialization fails.

    Example:
        from src.core.providers import get_embedding_provider

        embedding_provider = get_embedding_provider()
        if embedding_provider:
            vector = embedding_provider.embed_text("Hello world")
    """
    global _embedding_provider, _embedding_initialized

    # Double-checked locking pattern for thread safety
    if not _embedding_initialized:
        with _lock:
            if not _embedding_initialized:
                try:
                    _embedding_provider = create_embedding_provider()
                    print("[Providers] Shared embedding provider initialized")
                except Exception as e:
                    print(f"[Providers] WARNING: Could not initialize embedding provider: {e}")
                    _embedding_provider = None
                _embedding_initialized = True

    return _embedding_provider


def get_vector_store() -> Optional[VectorStoreProvider]:
    """
    Get the shared vector store instance.

    This function returns the SAME vector store instance every time
    it's called. This is CRITICAL for correct RAG operation.

    WHY IS THIS THE MOST IMPORTANT FUNCTION?
    ----------------------------------------
    The vector store is where documents are stored and searched.
    If ingestion and query use different instances:
    - Ingested documents go to instance A
    - Queries search instance B
    - User's documents are never found!

    By using a shared instance:
    - Ingested documents go to the SHARED instance
    - Queries search the SAME SHARED instance
    - User's documents are correctly retrieved!

    PERSISTENCE WARNING:
    --------------------
    By default, Qdrant runs in IN-MEMORY mode:
    - Data is LOST when the application restarts
    - This is fine for development and testing

    For production with persistence, set:
    - QDRANT_HOST and QDRANT_PORT (local server)
    - OR QDRANT_URL and QDRANT_API_KEY (cloud)

    Returns:
        The shared VectorStoreProvider instance, or None if initialization fails.

    Example:
        from src.core.providers import get_vector_store

        vector_store = get_vector_store()
        if vector_store:
            results = vector_store.search(query_embedding, top_k=5)
    """
    global _vector_store, _vector_store_initialized

    # Double-checked locking pattern for thread safety
    if not _vector_store_initialized:
        with _lock:
            if not _vector_store_initialized:
                try:
                    _vector_store = create_vector_store_provider()
                    print("[Providers] Shared vector store initialized")

                    # Log the mode for clarity
                    info = _vector_store.get_info()
                    mode = info.get("mode", "unknown")
                    print(f"[Providers] Vector store mode: {mode}")

                    if mode == "in-memory":
                        print("[Providers] WARNING: Using in-memory mode. Data will be lost on restart.")
                        print("[Providers] For persistence, configure QDRANT_HOST/PORT or QDRANT_URL.")
                except Exception as e:
                    print(f"[Providers] WARNING: Could not initialize vector store: {e}")
                    _vector_store = None
                _vector_store_initialized = True

    return _vector_store


def get_llm_provider() -> Optional[LLMProvider]:
    """
    Get the shared LLM provider instance.

    This function returns the SAME LLM provider instance every time
    it's called.

    WHY A SHARED INSTANCE?
    ----------------------
    - LLM providers may maintain connection pools
    - Creating multiple instances wastes resources
    - Ensures consistent model configuration

    Returns:
        The shared LLMProvider instance, or None if initialization fails.

    Example:
        from src.core.providers import get_llm_provider

        llm = get_llm_provider()
        if llm:
            answer = llm.generate_with_context(question, documents)
    """
    global _llm_provider, _llm_initialized

    # Double-checked locking pattern for thread safety
    if not _llm_initialized:
        with _lock:
            if not _llm_initialized:
                try:
                    _llm_provider = create_llm_provider()
                    print("[Providers] Shared LLM provider initialized")
                except Exception as e:
                    print(f"[Providers] WARNING: Could not initialize LLM provider: {e}")
                    _llm_provider = None
                _llm_initialized = True

    return _llm_provider


def get_reranker() -> Optional[RerankerProvider]:
    """
    Get the shared reranker provider instance.

    This function returns the SAME reranker provider instance every time
    it's called. The reranker is only initialized if ENABLE_RERANKING=true.

    WHAT IS RERANKING? (STEP 6)
    ---------------------------
    Reranking is a second-stage retrieval process that improves result quality:
    1. Vector search quickly retrieves candidate documents
    2. Reranker carefully scores each candidate's relevance
    3. Only the highest-scoring documents reach the LLM

    WHY A SHARED INSTANCE?
    ----------------------
    - Rerankers may cache model weights or connections
    - Creating multiple instances wastes resources
    - Ensures consistent reranking across the application

    WHEN IS RERANKING USED?
    -----------------------
    Reranking only happens when ALL of these are true:
    1. ENABLE_RERANKING=true in environment
    2. This function returns a valid reranker (not None)
    3. The pipeline has documents to rerank

    If ENABLE_RERANKING=false (default), this function returns None
    and no reranking overhead is incurred.

    Returns:
        The shared RerankerProvider instance, or None if:
        - ENABLE_RERANKING is false (default behavior)
        - Initialization fails

    Example:
        from src.core.providers import get_reranker

        reranker = get_reranker()
        if reranker:
            reranked_docs = reranker.rerank(query, documents, top_k=5)
    """
    global _reranker, _reranker_initialized

    # Import settings here to check if reranking is enabled
    from src.core.config import settings

    # Double-checked locking pattern for thread safety
    if not _reranker_initialized:
        with _lock:
            if not _reranker_initialized:
                # Only initialize if reranking is enabled
                if not settings.enable_reranking:
                    print("[Providers] Reranking is DISABLED (ENABLE_RERANKING=false)")
                    print("[Providers] To enable: set ENABLE_RERANKING=true")
                    _reranker = None
                else:
                    try:
                        _reranker = create_reranker_provider()
                        print(f"[Providers] Shared reranker initialized: {_reranker.get_provider_name()}")
                    except Exception as e:
                        print(f"[Providers] WARNING: Could not initialize reranker: {e}")
                        _reranker = None
                _reranker_initialized = True

    return _reranker


# =============================================================================
# INITIALIZATION FUNCTION
# =============================================================================

def initialize_providers() -> dict:
    """
    Initialize all shared providers at application startup.

    Call this function once when the application starts to ensure
    all providers are ready before handling requests.

    WHY INITIALIZE AT STARTUP?
    --------------------------
    1. Fail fast: If a provider can't be initialized, we know immediately
    2. Warm up: First request doesn't have initialization delay
    3. Clarity: All initialization happens in one place

    Returns:
        A dictionary with initialization status for each provider:
        {
            "embedding_provider": True/False,
            "vector_store": True/False,
            "llm_provider": True/False,
            "reranker": True/False/None  # None if reranking disabled
        }

    Example:
        from src.core.providers import initialize_providers

        # At application startup
        status = initialize_providers()
        print(f"Vector store ready: {status['vector_store']}")
    """
    # Import settings to check reranking config
    from src.core.config import settings

    print("\n" + "=" * 60)
    print("INITIALIZING SHARED PROVIDERS")
    print("=" * 60)

    # Initialize each provider
    embedding = get_embedding_provider()
    vector_store = get_vector_store()
    llm = get_llm_provider()
    reranker = get_reranker()  # Only initializes if ENABLE_RERANKING=true

    status = {
        "embedding_provider": embedding is not None,
        "vector_store": vector_store is not None,
        "llm_provider": llm is not None,
        # Reranker: True if initialized, False if failed, None if disabled
        "reranker": reranker is not None if settings.enable_reranking else None
    }

    print("\n" + "-" * 40)
    print("PROVIDER STATUS:")
    print(f"  Embedding Provider: {'OK' if status['embedding_provider'] else 'FAILED'}")
    print(f"  Vector Store:       {'OK' if status['vector_store'] else 'FAILED'}")
    print(f"  LLM Provider:       {'OK' if status['llm_provider'] else 'FAILED'}")
    # Reranker status depends on whether it's enabled
    if status['reranker'] is None:
        print(f"  Reranker:           DISABLED (set ENABLE_RERANKING=true to enable)")
    else:
        print(f"  Reranker:           {'OK' if status['reranker'] else 'FAILED'}")
    print("-" * 40)
    print("=" * 60 + "\n")

    return status


# =============================================================================
# RESET FUNCTION (FOR TESTING)
# =============================================================================

def reset_providers() -> None:
    """
    Reset all providers to uninitialized state.

    WARNING: This function is intended for testing only.
    It should NOT be called in production code.

    After calling this function, the next call to any get_* function
    will create a new instance.

    WHY IS THIS USEFUL?
    -------------------
    In tests, you may want to:
    - Test initialization logic
    - Reset state between test cases
    - Inject mock providers

    Example (in tests only):
        from src.core.providers import reset_providers, get_vector_store

        # Reset for clean test
        reset_providers()

        # Now get_vector_store() will create a fresh instance
        store = get_vector_store()
    """
    global _embedding_provider, _vector_store, _llm_provider, _reranker
    global _embedding_initialized, _vector_store_initialized, _llm_initialized, _reranker_initialized

    with _lock:
        _embedding_provider = None
        _vector_store = None
        _llm_provider = None
        _reranker = None
        _embedding_initialized = False
        _vector_store_initialized = False
        _llm_initialized = False
        _reranker_initialized = False

    print("[Providers] All providers reset to uninitialized state")
