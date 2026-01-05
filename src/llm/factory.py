"""
LLM Provider Factory
=====================

WHAT IS A FACTORY?
------------------
A factory is a design pattern that creates objects for you. Instead of
directly creating objects (like `provider = OpenRouterLLMProvider()`),
you ask the factory to create them (like `provider = create_llm_provider()`).

WHY USE A FACTORY PATTERN?
--------------------------
1. CENTRALIZED CREATION: All provider creation logic is in one place
2. CONFIGURATION-DRIVEN: The factory reads settings to decide which provider
3. EASY TO EXTEND: Adding a new provider only requires modifying this file
4. LOOSE COUPLING: The rest of the code doesn't need to know about specific providers

HOW THIS FACTORY WORKS:
-----------------------
1. It reads the LLM_PROVIDER setting from environment variables
2. Based on that setting, it creates the appropriate provider class
3. It returns an LLMProvider instance that you can use

SUPPORTED PROVIDERS:
--------------------
- "openrouter": Uses OpenRouter's OpenAI-compatible API (default)
- More providers can be added as needed (OpenAI, Anthropic, Ollama, etc.)
"""

from typing import Optional

from src.llm.base import LLMProvider
from src.core.config import settings


def create_llm_provider(
    provider_name: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Create and return an LLM provider based on configuration.

    This is the main entry point for getting an LLM provider.
    It handles all the details of creating the right provider based on settings.

    HOW IT DECIDES WHICH PROVIDER TO CREATE:
    ----------------------------------------
    1. If you pass a provider_name, it uses that
    2. Otherwise, it reads from the LLM_PROVIDER environment variable
    3. If that's not set, it defaults to "openrouter"

    EXTENDING WITH NEW PROVIDERS:
    -----------------------------
    To add a new provider:
    1. Create a new class in src/llm/providers/ (e.g., openai.py)
    2. Make sure it inherits from LLMProvider
    3. Add a new case to the if/elif chain below
    4. Set LLM_PROVIDER=your_provider to use it

    Args:
        provider_name: Optional. The name of the provider to create.
                      Supported values: "openrouter"
                      If None, reads from LLM_PROVIDER environment variable.

        **kwargs: Additional arguments to pass to the provider constructor.
                 These override environment variable settings.
                 Examples:
                 - api_key="sk-..." to override OPENROUTER_API_KEY
                 - model="anthropic/claude-3-opus" to override LLM_MODEL

    Returns:
        An instance of LLMProvider that you can use to generate text.
        The specific class depends on the provider_name setting.

    Raises:
        ValueError: If the requested provider is not supported.

    Examples:
        # Basic usage (uses environment variables)
        provider = create_llm_provider()
        response = provider.generate("Explain RAG in simple terms")

        # Specify provider explicitly
        provider = create_llm_provider("openrouter")

        # Override settings
        provider = create_llm_provider(
            provider_name="openrouter",
            model="anthropic/claude-3-sonnet"
        )
    """
    # =========================================================================
    # Determine which provider to use
    # =========================================================================
    # Priority:
    # 1. Explicitly passed provider_name argument
    # 2. LLM_PROVIDER environment variable
    # 3. Default to "openrouter"

    effective_provider = provider_name if provider_name is not None else (
        settings.llm_provider
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
        # Uses OpenRouter's OpenAI-compatible API for text generation.
        # This gives access to many models (GPT-4, Claude, Llama, etc.)

        # Import here to avoid circular imports and for lazy loading
        from src.llm.providers.openrouter import OpenRouterLLMProvider

        # Log which provider we're creating (helpful for debugging)
        print(f"[LLMFactory] Creating OpenRouter LLM provider")

        # Create and return the provider
        return OpenRouterLLMProvider(**kwargs)

    # =========================================================================
    # Future providers (uncomment and implement as needed)
    # =========================================================================

    # elif effective_provider == "openai":
    #     # Direct OpenAI API (without OpenRouter)
    #     from src.llm.providers.openai import OpenAILLMProvider
    #     print(f"[LLMFactory] Creating OpenAI LLM provider")
    #     return OpenAILLMProvider(**kwargs)

    # elif effective_provider == "anthropic":
    #     # Direct Anthropic API for Claude models
    #     from src.llm.providers.anthropic import AnthropicLLMProvider
    #     print(f"[LLMFactory] Creating Anthropic LLM provider")
    #     return AnthropicLLMProvider(**kwargs)

    # elif effective_provider == "ollama":
    #     # Local models via Ollama
    #     from src.llm.providers.ollama import OllamaLLMProvider
    #     print(f"[LLMFactory] Creating Ollama LLM provider")
    #     return OllamaLLMProvider(**kwargs)

    # =========================================================================
    # Unknown provider - raise an error
    # =========================================================================
    else:
        # List of supported providers for the error message
        supported_providers = [
            "openrouter",
            # "openai",
            # "anthropic",
            # "ollama",
        ]

        raise ValueError(
            f"Unknown LLM provider: '{effective_provider}'. "
            f"Supported providers are: {', '.join(supported_providers)}. "
            f"Set the LLM_PROVIDER environment variable to one of these values."
        )


def get_default_provider() -> LLMProvider:
    """
    Get the default LLM provider based on current settings.

    This is a convenience function that calls create_llm_provider()
    with no arguments, using all default/environment variable settings.

    Returns:
        The default LLMProvider instance.

    Example:
        from src.llm.factory import get_default_provider

        provider = get_default_provider()
        response = provider.generate("Hello!")
    """
    return create_llm_provider()
