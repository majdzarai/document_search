"""
Embedding Provider Factory
===========================

WHAT IS A FACTORY?
------------------
A factory is a design pattern that creates objects for you. Instead of
directly creating objects (like `provider = OpenRouterEmbeddingProvider()`),
you ask the factory to create them for you (like `provider = create_embedding_provider()`).

WHY USE A FACTORY PATTERN?
--------------------------
1. CENTRALIZED CREATION: All provider creation logic is in one place
2. CONFIGURATION-DRIVEN: The factory reads settings to decide which provider to create
3. EASY TO EXTEND: Adding a new provider only requires modifying this file
4. LOOSE COUPLING: The rest of the code doesn't need to know about specific providers

EXAMPLE WITHOUT FACTORY (bad):
------------------------------
    # Every part of the code needs to know about all providers
    if config.provider == "openrouter":
        from src.embeddings.providers.openrouter import OpenRouterEmbeddingProvider
        provider = OpenRouterEmbeddingProvider()
    elif config.provider == "cohere":
        from src.embeddings.providers.cohere import CohereEmbeddingProvider
        provider = CohereEmbeddingProvider()
    # ... repeated everywhere we need a provider

EXAMPLE WITH FACTORY (good):
----------------------------
    from src.embeddings.factory import create_embedding_provider
    provider = create_embedding_provider()  # Factory handles the decision

HOW THIS FACTORY WORKS:
-----------------------
1. It reads the EMBEDDING_PROVIDER setting from environment variables
2. Based on that setting, it creates the appropriate provider class
3. It returns an EmbeddingProvider instance that you can use

SUPPORTED PROVIDERS:
--------------------
- "openrouter": Uses OpenRouter's OpenAI-compatible API (default)
- More providers can be added as needed (OpenAI, Cohere, HuggingFace, etc.)
"""

from typing import Optional

from src.embeddings.base import EmbeddingProvider
from src.core.config import settings


def create_embedding_provider(
    provider_name: Optional[str] = None,
    **kwargs
) -> EmbeddingProvider:
    """
    Create and return an embedding provider based on configuration.

    This is the main entry point for getting an embedding provider.
    It handles all the details of creating the right provider based on settings.

    HOW IT DECIDES WHICH PROVIDER TO CREATE:
    ----------------------------------------
    1. If you pass a provider_name, it uses that
    2. Otherwise, it reads from the EMBEDDING_PROVIDER environment variable
    3. If that's not set, it defaults to "openrouter"

    EXTENDING WITH NEW PROVIDERS:
    -----------------------------
    To add a new provider:
    1. Create a new class in src/embeddings/providers/ (e.g., cohere.py)
    2. Import it at the top of this file
    3. Add a new case to the if/elif chain below
    4. Set EMBEDDING_PROVIDER=your_provider to use it

    Args:
        provider_name: Optional. The name of the provider to create.
                       Supported values: "openrouter"
                       If None, reads from EMBEDDING_PROVIDER environment variable.

        **kwargs: Additional arguments to pass to the provider constructor.
                  These override environment variable settings.
                  Examples:
                  - api_key="sk-..." to override OPENROUTER_API_KEY
                  - model="openai/text-embedding-3-large" to override EMBEDDING_MODEL

    Returns:
        An instance of EmbeddingProvider that you can use to embed text.
        The specific class depends on the provider_name setting.

    Raises:
        ValueError: If the requested provider is not supported.

    Examples:
        # Basic usage (uses environment variables)
        provider = create_embedding_provider()
        embedding = provider.embed_text("Hello, world!")

        # Specify provider explicitly
        provider = create_embedding_provider("openrouter")

        # Override settings
        provider = create_embedding_provider(
            provider_name="openrouter",
            model="openai/text-embedding-3-large"
        )
    """
    # =========================================================================
    # Determine which provider to use
    # =========================================================================
    # Priority:
    # 1. Explicitly passed provider_name argument
    # 2. EMBEDDING_PROVIDER environment variable
    # 3. Default to "openrouter"

    effective_provider = provider_name if provider_name is not None else (
        settings.embedding_provider
    )

    # Normalize to lowercase for case-insensitive matching
    # This lets users write "OpenRouter" or "OPENROUTER" or "openrouter"
    effective_provider = effective_provider.lower().strip()

    # =========================================================================
    # Create the appropriate provider
    # =========================================================================
    # Each provider has its own class with specific initialization logic

    if effective_provider == "openrouter":
        # ---------------------------------------------------------------------
        # OpenRouter Provider
        # ---------------------------------------------------------------------
        # Uses OpenRouter's OpenAI-compatible embeddings API
        # This is the default and recommended provider

        # Import here to avoid circular imports and for lazy loading
        # Lazy loading means we only import what we actually need
        from src.embeddings.providers.openrouter import OpenRouterEmbeddingProvider

        # Log which provider we're creating (helpful for debugging)
        print(f"[EmbeddingFactory] Creating OpenRouter embedding provider")

        # Create and return the provider
        # The **kwargs allows passing custom settings like api_key or model
        return OpenRouterEmbeddingProvider(**kwargs)

    # =========================================================================
    # Future providers (uncomment and implement as needed)
    # =========================================================================

    # elif effective_provider == "openai":
    #     # Direct OpenAI API (without OpenRouter)
    #     from src.embeddings.providers.openai import OpenAIEmbeddingProvider
    #     print(f"[EmbeddingFactory] Creating OpenAI embedding provider")
    #     return OpenAIEmbeddingProvider(**kwargs)

    # elif effective_provider == "cohere":
    #     # Cohere's embedding API
    #     from src.embeddings.providers.cohere import CohereEmbeddingProvider
    #     print(f"[EmbeddingFactory] Creating Cohere embedding provider")
    #     return CohereEmbeddingProvider(**kwargs)

    # elif effective_provider == "huggingface":
    #     # HuggingFace Inference API
    #     from src.embeddings.providers.huggingface import HuggingFaceEmbeddingProvider
    #     print(f"[EmbeddingFactory] Creating HuggingFace embedding provider")
    #     return HuggingFaceEmbeddingProvider(**kwargs)

    # elif effective_provider == "sentence_transformers":
    #     # Local sentence-transformers models
    #     from src.embeddings.providers.sentence_transformers import (
    #         SentenceTransformersEmbeddingProvider
    #     )
    #     print(f"[EmbeddingFactory] Creating SentenceTransformers embedding provider")
    #     return SentenceTransformersEmbeddingProvider(**kwargs)

    # =========================================================================
    # Unknown provider - raise an error
    # =========================================================================
    else:
        # List of supported providers for the error message
        supported_providers = [
            "openrouter",
            # "openai",
            # "cohere",
            # "huggingface",
            # "sentence_transformers"
        ]

        raise ValueError(
            f"Unknown embedding provider: '{effective_provider}'. "
            f"Supported providers are: {', '.join(supported_providers)}. "
            f"Set the EMBEDDING_PROVIDER environment variable to one of these values."
        )


def get_default_provider() -> EmbeddingProvider:
    """
    Get the default embedding provider based on current settings.

    This is a convenience function that calls create_embedding_provider()
    with no arguments, using all default/environment variable settings.

    This is useful when you just want "the embedding provider" without
    thinking about configuration.

    Returns:
        The default EmbeddingProvider instance.

    Example:
        from src.embeddings.factory import get_default_provider

        provider = get_default_provider()
        embedding = provider.embed_text("Hello!")
    """
    return create_embedding_provider()


# =============================================================================
# Module-level Provider Instance (Optional Singleton)
# =============================================================================
# Some applications want a single, shared embedding provider instance.
# This avoids creating multiple instances and ensures consistency.
#
# IMPORTANT: This instance is created when the module is first imported.
# If you don't want this behavior, don't import 'default_embedding_provider'.
#
# WHY A SINGLETON?
# ----------------
# 1. EFFICIENCY: Creating providers can be expensive (API validation, etc.)
# 2. CONSISTENCY: All code uses the same provider instance
# 3. CONVENIENCE: Just import and use, no initialization needed
#
# CAUTION:
# --------
# The singleton is created at import time. If environment variables aren't
# set yet, it will fail. For applications with complex startup sequences,
# use create_embedding_provider() instead.

# Uncomment the following line to enable a module-level singleton:
# default_embedding_provider = create_embedding_provider()
