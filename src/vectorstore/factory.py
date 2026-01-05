"""
Vector Store Factory
=====================

WHAT IS A FACTORY?
------------------
A factory is a design pattern that creates objects without specifying the exact
class of object that will be created. Think of it like a restaurant kitchen:
- You order "pizza" (not "MargheritaPizza" or "PepperoniPizza")
- The kitchen decides which specific pizza to make based on your order
- You get a pizza, regardless of the specific type

In our case:
- You ask for a "vector store"
- The factory decides which provider to use (Qdrant, Pinecone, etc.)
- You get a vector store, regardless of the specific implementation

WHY USE A FACTORY?
------------------
1. DECOUPLING: The rest of our code doesn't need to know about specific providers
2. CONFIGURATION: The factory reads settings and creates the right provider
3. FLEXIBILITY: Easy to add new providers or switch between them
4. TESTING: Can easily create mock providers for testing

HOW DOES IT WORK?
-----------------
1. Read the configuration (environment variables)
2. Based on the config, import and create the right provider
3. Return the provider (which implements our standard interface)

Example:
    # In your code, you just call:
    store = create_vector_store_provider()

    # The factory figures out that you want Qdrant (from config)
    # and returns a QdrantVectorStore instance

    # You can use it without knowing which provider it is:
    store.upsert(ids, embeddings, texts, metadata)
    results = store.search(query_embedding)

CONFIGURATION:
--------------
The factory reads these settings (from environment variables):
- VECTOR_STORE_PROVIDER: Which provider to use (e.g., "qdrant")
- QDRANT_COLLECTION_NAME: Name of the Qdrant collection
- VECTOR_DIMENSION: Dimension of embedding vectors

(See config.py for all vector store settings)
"""

from typing import Optional

# Import the base class (for type hints)
from src.vectorstore.base import VectorStoreProvider

# Import settings to read configuration
from src.core.config import settings


def create_vector_store_provider(
    provider: Optional[str] = None,
    collection_name: Optional[str] = None,
    vector_dimension: Optional[int] = None,
) -> VectorStoreProvider:
    """
    Create and return a vector store provider based on configuration.

    WHAT THIS FUNCTION DOES:
    ------------------------
    1. Determines which provider to use (from argument or config)
    2. Imports the specific provider class (lazy loading)
    3. Creates and returns an instance of that provider

    WHY LAZY LOADING?
    -----------------
    We only import the provider we need, when we need it. This means:
    - Faster startup (don't load unused providers)
    - Fewer dependencies required (only install what you use)
    - Better error messages (tells you exactly what's missing)

    Args:
        provider: Which provider to use.
                  Options: "qdrant" (more coming in future steps)
                  If None, uses the VECTOR_STORE_PROVIDER env variable.

        collection_name: Name for the vector collection.
                        If None, uses the QDRANT_COLLECTION_NAME env variable.

        vector_dimension: Dimension of embedding vectors.
                         If None, uses the VECTOR_DIMENSION env variable.
                         MUST match your embedding model!

    Returns:
        A VectorStoreProvider instance ready to use.

    Raises:
        ValueError: If the provider name is not recognized.
        ImportError: If the provider's library is not installed.

    Example Usage:
    --------------
        # Simple: use all defaults from environment
        store = create_vector_store_provider()

        # Custom: override specific settings
        store = create_vector_store_provider(
            provider="qdrant",
            collection_name="my_documents",
            vector_dimension=1536
        )

    SUPPORTED PROVIDERS:
    --------------------
    - "qdrant": Qdrant vector database (implemented in Step 3)

    Future providers (not yet implemented):
    - "pinecone": Pinecone cloud vector database
    - "weaviate": Weaviate vector database
    - "pgvector": PostgreSQL with vector extension
    """
    # =========================================================================
    # STEP 1: Determine which provider to use
    # =========================================================================
    # Priority: function argument > environment variable > default

    selected_provider = provider or settings.vector_store_provider

    print(f"[VectorStoreFactory] Creating vector store provider: {selected_provider}")

    # =========================================================================
    # STEP 2: Get configuration values
    # =========================================================================
    # Use provided values or fall back to settings

    actual_collection_name = collection_name or settings.qdrant_collection_name
    actual_dimension = vector_dimension or settings.vector_dimension

    print(f"[VectorStoreFactory] Collection name: {actual_collection_name}")
    print(f"[VectorStoreFactory] Vector dimension: {actual_dimension}")

    # =========================================================================
    # STEP 3: Create the appropriate provider
    # =========================================================================
    # We use if/elif to select the right provider based on the name.
    # Each provider is imported only when needed (lazy loading).

    if selected_provider == "qdrant":
        # ---------------------------------------------------------------------
        # QDRANT PROVIDER
        # ---------------------------------------------------------------------
        # Qdrant is our primary provider for Step 3.
        # It supports in-memory mode (no installation needed) and server mode.

        # Lazy import: only load qdrant code when we actually need it
        from src.vectorstore.providers.qdrant import QdrantVectorStore

        # Create the Qdrant vector store
        # We check if server settings are provided, otherwise use in-memory mode
        return QdrantVectorStore(
            collection_name=actual_collection_name,
            vector_dimension=actual_dimension,
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

    # =========================================================================
    # FUTURE PROVIDERS (placeholders for future steps)
    # =========================================================================
    # These are commented out because they're not implemented yet.
    # They show how easy it is to add new providers to the factory.

    # elif selected_provider == "pinecone":
    #     from src.vectorstore.providers.pinecone import PineconeVectorStore
    #     return PineconeVectorStore(...)

    # elif selected_provider == "weaviate":
    #     from src.vectorstore.providers.weaviate import WeaviateVectorStore
    #     return WeaviateVectorStore(...)

    # elif selected_provider == "pgvector":
    #     from src.vectorstore.providers.pgvector import PgVectorStore
    #     return PgVectorStore(...)

    else:
        # Unknown provider - raise a helpful error
        raise ValueError(
            f"Unknown vector store provider: '{selected_provider}'\n"
            f"Supported providers: 'qdrant'\n"
            f"(More providers will be added in future steps)\n"
            f"\n"
            f"To set the provider, use the VECTOR_STORE_PROVIDER environment variable:\n"
            f"  export VECTOR_STORE_PROVIDER=qdrant  (Linux/Mac)\n"
            f"  set VECTOR_STORE_PROVIDER=qdrant     (Windows CMD)\n"
            f"  $env:VECTOR_STORE_PROVIDER='qdrant'  (Windows PowerShell)"
        )
