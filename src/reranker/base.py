"""
Reranker Base Class
====================

WHAT IS A RERANKER?
-------------------
A reranker is a second-stage retrieval component that re-scores documents
after the initial vector search. While vector search is fast, it doesn't
always rank documents by true relevance to the query.

EXAMPLE OF WHY RERANKING HELPS:
-------------------------------
Imagine a user asks: "What are the benefits of using Python for AI?"

Vector search might return:
1. "Python is a programming language" (score: 0.92) - too generic!
2. "Python has many AI libraries like PyTorch" (score: 0.89) - very relevant!
3. "AI is used in many industries" (score: 0.87) - not about Python!

After reranking:
1. "Python has many AI libraries like PyTorch" (score: 0.95) - promoted!
2. "Python is a programming language" (score: 0.65) - demoted!
3. "AI is used in many industries" (score: 0.45) - demoted!

The reranker understands the RELATIONSHIP between the query and documents,
not just their individual meanings.

WHY IS RERANKING MORE ACCURATE?
-------------------------------
1. CROSS-ATTENTION: Rerankers compare query and document directly
   - Vector search: Embed query and document SEPARATELY, then compare
   - Reranker: Look at query AND document TOGETHER

2. DEEPER UNDERSTANDING: Rerankers can understand context
   - "Python" (the snake) vs "Python" (the language)
   - Rerankers see the full context to disambiguate

3. SEMANTIC MATCHING: Rerankers catch nuances
   - Query: "How to fix bugs"
   - Relevant: "Debugging techniques" (no word overlap, but relevant!)

THE TRADEOFF:
-------------
- Vector search: FAST (milliseconds) but less precise
- Reranking: SLOWER (seconds) but more precise

That's why we use BOTH:
1. Vector search quickly gets top 20-50 candidates
2. Reranker carefully scores those candidates
3. Top 3-5 documents go to the LLM

WHAT IS THIS MODULE?
--------------------
This module defines an ABSTRACT BASE CLASS for reranker providers.
We want to support multiple reranking approaches:

- LLM-based rerankers (use an LLM to score relevance)
- Cross-encoder rerankers (specialized ML models)
- Cohere Rerank API (hosted service)
- BGE rerankers (open-source models)

By defining an abstract interface, we can swap providers without
changing the rest of our code.

DESIGN PATTERN: Strategy Pattern
---------------------------------
This follows the Strategy Pattern:
- The interface (RerankerProvider) stays the same
- The implementation can be swapped (LLM, CrossEncoder, Cohere, etc.)
- The client code (RAGPipeline) works with any implementation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class RerankerProvider(ABC):
    """
    Abstract base class defining the interface for reranker providers.

    This is an abstract class - you cannot create an instance of it directly.
    Instead, you create subclasses that implement the abstract methods.

    WHY USE AN ABSTRACT CLASS?
    --------------------------
    1. STANDARDIZATION: All reranker providers have the same interface
    2. SUBSTITUTION: We can swap providers without changing other code
    3. DOCUMENTATION: The methods here document what every provider must do
    4. SAFETY: Python raises an error if you forget to implement a method

    HOW TO CREATE A NEW PROVIDER:
    -----------------------------
    1. Create a new class that inherits from RerankerProvider
    2. Implement all methods marked with @abstractmethod
    3. Register it in the factory (see factory.py)

    Example:
        class MyRerankerProvider(RerankerProvider):
            def rerank(
                self,
                query: str,
                documents: List[Dict[str, Any]],
                top_k: int = 5
            ) -> List[Dict[str, Any]]:
                # Your implementation here
                return reranked_documents

            def get_provider_name(self) -> str:
                return "my-reranker"

    DOCUMENT FORMAT:
    ----------------
    Documents passed to rerank() are dictionaries with:
    - "id": Unique identifier (string)
    - "content": The document text (string)
    - "metadata": Additional info like source, page, etc. (dict)
    - "score": The original vector similarity score (float)

    After reranking, documents have:
    - All original fields preserved
    - "rerank_score": The new relevance score from the reranker (float)
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.

        THIS IS THE CORE METHOD!
        ------------------------
        This method takes documents retrieved by vector search and
        re-scores them to find the most relevant ones.

        THE PROCESS:
        ------------
        1. For each document, calculate a relevance score (0.0 to 1.0)
        2. Sort documents by their new scores (highest first)
        3. Filter out documents below min_score threshold
        4. Return the top_k documents

        Args:
            query: The user's original question.
                  Example: "What are the benefits of RAG?"

            documents: List of documents to rerank. Each document has:
                      - "id": Document identifier
                      - "content": The document text
                      - "metadata": Additional info (source, page, etc.)
                      - "score": Original vector similarity score

                      Example:
                      [
                          {
                              "id": "doc_001",
                              "content": "RAG improves accuracy by...",
                              "metadata": {"source": "rag_guide.pdf"},
                              "score": 0.89
                          },
                          ...
                      ]

            top_k: Maximum number of documents to return.
                  The reranker will return at most this many documents.
                  Default: 5

            min_score: Minimum relevance score to keep a document.
                      Documents with rerank_score below this are discarded.
                      Range: 0.0 to 1.0
                      Default: 0.0 (no filtering)

        Returns:
            A list of reranked documents, sorted by relevance (highest first).
            Each document has all original fields plus:
            - "rerank_score": The new relevance score (0.0 to 1.0)

            The list may have fewer than top_k documents if:
            - Input had fewer documents
            - Some documents were filtered by min_score

            Example return:
            [
                {
                    "id": "doc_003",
                    "content": "RAG combines retrieval...",
                    "metadata": {"source": "intro.pdf"},
                    "score": 0.85,  # Original vector score
                    "rerank_score": 0.95  # New relevance score
                },
                ...
            ]

        HOW TO IMPLEMENT:
        -----------------
        Different providers calculate rerank_score differently:

        - LLM-based: Ask the LLM "On a scale of 0-10, how relevant is
          this document to the query?" then normalize to 0-1

        - Cross-encoder: Pass query + document through a transformer
          that outputs a relevance score directly

        - Cohere: Call the Cohere Rerank API which handles scoring
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of the reranker provider.

        This is useful for:
        1. LOGGING: Know which reranker processed which queries
        2. DEBUGGING: Verify the correct provider is being used
        3. MONITORING: Track performance by provider

        Returns:
            A string with the provider name.
            Example: "simple" or "cohere" or "cross_encoder"

        Example:
            reranker = SimpleRerankerProvider()
            print(f"Using reranker: {reranker.get_provider_name()}")
        """
        pass

    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple queries' documents in batch.

        This is a convenience method for processing multiple queries.
        By default, it just calls rerank() for each query.

        Providers can override this for more efficient batch processing.
        For example, LLM providers might batch API calls.

        Args:
            queries: List of user queries
            documents_list: List of document lists (one per query)
            top_k: Maximum documents per query
            min_score: Minimum relevance score threshold

        Returns:
            List of reranked document lists (one per query)

        Example:
            queries = ["What is RAG?", "How do embeddings work?"]
            docs_list = [rag_docs, embedding_docs]
            results = reranker.batch_rerank(queries, docs_list, top_k=3)
        """
        # Default implementation: process each query separately
        results = []
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(
                query=query,
                documents=documents,
                top_k=top_k,
                min_score=min_score
            )
            results.append(reranked)
        return results
