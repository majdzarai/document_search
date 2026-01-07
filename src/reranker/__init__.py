"""
Reranker Module
================

This module provides document reranking functionality for the RAG pipeline.

WHAT IS RERANKING?
------------------
Reranking is a second-stage retrieval process that improves the quality of
search results. It takes documents retrieved by vector search and re-scores
them to find the most relevant ones.

COMPONENTS:
-----------
- RerankerProvider: Abstract base class for all rerankers
- create_reranker_provider: Factory function to create rerankers
- SimpleLLMReranker: LLM-based reranker using OpenRouter

USAGE:
------
    from src.reranker import create_reranker_provider

    reranker = create_reranker_provider()
    reranked_docs = reranker.rerank(
        query="What is RAG?",
        documents=retrieved_docs,
        top_k=5
    )

Or through the shared providers system:
    from src.core.providers import get_reranker

    reranker = get_reranker()
    if reranker:
        docs = reranker.rerank(query, documents)
"""

from src.reranker.base import RerankerProvider
from src.reranker.factory import create_reranker_provider

__all__ = [
    "RerankerProvider",
    "create_reranker_provider",
]
