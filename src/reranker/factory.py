"""
Reranker Factory
================

WHAT IS A FACTORY?
------------------
A factory is a design pattern that creates objects without exposing the
creation logic to the client. The client just says "give me a reranker"
and the factory handles all the details of which specific reranker to create.

WHY USE A FACTORY FOR RERANKERS?
--------------------------------
We want to support multiple reranking providers:
- Simple (LLM-based) reranker - uses the configured LLM to score relevance
- Cohere Rerank API - fast, hosted service (future)
- Cross-encoder models - local ML models (future)
- BGE rerankers - open-source models (future)

The factory pattern lets us:
1. Add new providers without changing other code
2. Configure the provider through environment variables
3. Keep the RAGPipeline simple - it just asks for "a reranker"

HOW THIS FACTORY WORKS:
-----------------------
1. Read the RERANKER_PROVIDER setting from config
2. Based on the setting, create the appropriate provider
3. Return the provider instance

EXAMPLE USAGE:
--------------
    from src.reranker.factory import create_reranker_provider

    # Creates the provider specified in RERANKER_PROVIDER env var
    reranker = create_reranker_provider()

    # Or explicitly request a specific provider
    reranker = create_reranker_provider(provider="simple")

    # Use the reranker
    reranked_docs = reranker.rerank(
        query="What is RAG?",
        documents=retrieved_docs,
        top_k=5
    )
"""

from typing import Optional

from src.reranker.base import RerankerProvider
from src.core.config import settings


def create_reranker_provider(
    provider: Optional[str] = None
) -> RerankerProvider:
    """
    Factory function to create a reranker provider.

    This function reads the configuration and creates the appropriate
    reranker provider based on the settings.

    HOW PROVIDER SELECTION WORKS:
    -----------------------------
    1. If `provider` argument is given, use that
    2. Otherwise, read RERANKER_PROVIDER from environment
    3. Create and return the corresponding provider

    Args:
        provider: Optional. The name of the provider to create.
                 If None, uses the RERANKER_PROVIDER environment variable.
                 Supported values:
                 - "simple": LLM-based reranker (default)
                 - "cohere": Cohere Rerank API (future)
                 - "cross_encoder": Cross-encoder model (future)
                 - "bge_reranker": BGE reranker model (future)

    Returns:
        A RerankerProvider instance ready to use.

    Raises:
        ValueError: If the specified provider is not supported.

    Example:
        # Use default from config
        reranker = create_reranker_provider()

        # Or specify explicitly
        reranker = create_reranker_provider(provider="simple")

    ADDING NEW PROVIDERS:
    ---------------------
    To add a new reranker provider:
    1. Create a new class in src/reranker/providers/
    2. Implement the RerankerProvider interface
    3. Add it to the if/elif chain below
    4. Update the docstrings to list the new provider
    """
    # Determine which provider to use
    provider_name = provider if provider is not None else settings.reranker_provider
    provider_name = provider_name.lower().strip()

    print(f"[RerankerFactory] Creating reranker provider: {provider_name}")

    # =========================================================================
    # SIMPLE (LLM-Based) Reranker
    # =========================================================================
    # This reranker uses the configured LLM (via OpenRouter) to score
    # document relevance. It's a good starting point because:
    # - No additional dependencies required
    # - Uses the same API key as the main LLM
    # - Good quality reranking for most use cases
    #
    # Tradeoff: Slower than specialized rerankers (requires LLM API calls)

    if provider_name == "simple":
        from src.reranker.providers.simple import SimpleLLMReranker
        return SimpleLLMReranker()

    # =========================================================================
    # FUTURE: Cohere Reranker
    # =========================================================================
    # Cohere provides a dedicated Rerank API that is:
    # - Fast and optimized for reranking
    # - High quality (trained specifically for this task)
    # - Simple API (just send query + documents)
    #
    # To use: pip install cohere, set COHERE_API_KEY
    # Not yet implemented

    elif provider_name == "cohere":
        raise ValueError(
            f"Cohere reranker is not yet implemented. "
            f"Please use 'simple' reranker or implement the Cohere provider. "
            f"See src/reranker/providers/cohere.py for the stub."
        )

    # =========================================================================
    # FUTURE: Cross-Encoder Reranker
    # =========================================================================
    # Cross-encoders are transformer models that directly compare
    # query and document to produce a relevance score.
    #
    # Popular models:
    # - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast)
    # - cross-encoder/ms-marco-MiniLM-L-12-v2 (balanced)
    # - BAAI/bge-reranker-large (high quality)
    #
    # To use: pip install sentence-transformers
    # Not yet implemented

    elif provider_name == "cross_encoder":
        raise ValueError(
            f"Cross-encoder reranker is not yet implemented. "
            f"Please use 'simple' reranker or implement the CrossEncoder provider. "
            f"See src/reranker/providers/cross_encoder.py for the stub."
        )

    # =========================================================================
    # FUTURE: BGE Reranker
    # =========================================================================
    # BGE (BAAI General Embedding) rerankers are open-source models
    # specifically designed for reranking.
    #
    # Popular models:
    # - BAAI/bge-reranker-base
    # - BAAI/bge-reranker-large
    #
    # To use: pip install FlagEmbedding
    # Not yet implemented

    elif provider_name == "bge_reranker":
        raise ValueError(
            f"BGE reranker is not yet implemented. "
            f"Please use 'simple' reranker or implement the BGE provider. "
            f"See src/reranker/providers/bge_reranker.py for the stub."
        )

    # =========================================================================
    # Unknown Provider
    # =========================================================================

    else:
        supported_providers = ["simple", "cohere (future)", "cross_encoder (future)", "bge_reranker (future)"]
        raise ValueError(
            f"Unknown reranker provider: '{provider_name}'. "
            f"Supported providers: {supported_providers}. "
            f"Set the RERANKER_PROVIDER environment variable to one of these values."
        )
