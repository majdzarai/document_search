"""
Reranker Providers
==================

This package contains implementations of different reranker providers.

AVAILABLE PROVIDERS:
--------------------
- SimpleLLMReranker: Uses an LLM (via OpenRouter) to score document relevance.
                     Good for getting started, no extra dependencies.

FUTURE PROVIDERS (not yet implemented):
---------------------------------------
- CohereReranker: Uses Cohere's Rerank API
- CrossEncoderReranker: Uses local cross-encoder models
- BGEReranker: Uses BGE reranker models

HOW TO ADD A NEW PROVIDER:
--------------------------
1. Create a new file in this directory (e.g., my_reranker.py)
2. Create a class that inherits from RerankerProvider
3. Implement the rerank() and get_provider_name() methods
4. Register it in the factory (src/reranker/factory.py)
"""

from src.reranker.providers.simple import SimpleLLMReranker

__all__ = [
    "SimpleLLMReranker",
]
