"""
Simple LLM-Based Reranker
=========================

WHAT IS THIS RERANKER?
----------------------
This reranker uses a Large Language Model (LLM) to score document relevance.
It's called "simple" because:
1. No additional dependencies required
2. Uses the same OpenRouter API as the main LLM
3. Easy to understand how it works

HOW IT WORKS:
-------------
For each document, we ask the LLM a simple question:
"How relevant is this document to the user's query? Score 0-10."

The LLM reads both the query and document, then gives a score.
We normalize the score to 0.0-1.0 and use it to rank documents.

EXAMPLE:
--------
Query: "What are the benefits of RAG?"

Document 1: "RAG improves accuracy by grounding LLM responses in real data."
LLM Score: 9/10 -> rerank_score: 0.9 (Very relevant!)

Document 2: "Python is a popular programming language."
LLM Score: 2/10 -> rerank_score: 0.2 (Not relevant)

Document 3: "Machine learning models learn from data."
LLM Score: 4/10 -> rerank_score: 0.4 (Somewhat related)

After sorting: Document 1, Document 3, Document 2

WHY USE AN LLM FOR RERANKING?
-----------------------------
1. FLEXIBILITY: LLMs understand nuance and context
2. NO EXTRA SETUP: Uses existing OpenRouter API key
3. GOOD QUALITY: Modern LLMs are good at relevance judgment
4. EASY TO CUSTOMIZE: Can adjust the prompt for specific domains

TRADEOFFS:
----------
- SLOWER: Requires API calls for each document
- COSTLIER: Uses LLM tokens for scoring
- LESS SCALABLE: Not ideal for reranking 100+ documents

For production with high volume, consider:
- Cohere Rerank API (fast, optimized)
- Cross-encoder models (local, fast)

IMPLEMENTATION STRATEGY:
------------------------
We use a BATCH approach to minimize API calls:
1. Combine all documents into one prompt
2. Ask the LLM to score ALL documents at once
3. Parse the scores from the response

This is much faster than scoring each document separately!
"""

import json
import re
from typing import List, Dict, Any, Optional

from src.reranker.base import RerankerProvider
from src.core.config import settings


class SimpleLLMReranker(RerankerProvider):
    """
    A reranker that uses the configured LLM to score document relevance.

    This provider works by asking the LLM to rate how relevant each
    document is to the user's query on a scale of 0-10.

    USAGE:
    ------
        from src.reranker.providers.simple import SimpleLLMReranker

        reranker = SimpleLLMReranker()
        reranked = reranker.rerank(
            query="What is RAG?",
            documents=[...],
            top_k=5
        )

    CONFIGURATION:
    --------------
    This reranker uses the same configuration as the main LLM:
    - OPENROUTER_API_KEY: Required for API access
    - OPENROUTER_BASE_URL: API endpoint (default: openrouter.ai)
    - LLM_MODEL: Model to use for scoring (default: gpt-3.5-turbo)

    You can optionally specify a different model for reranking
    by passing it to the constructor.

    ATTRIBUTES:
    -----------
    model: The LLM model used for scoring (can be different from main LLM)
    api_key: OpenRouter API key for authentication
    base_url: OpenRouter API base URL
    """

    # The system prompt for relevance scoring
    SCORING_SYSTEM_PROMPT: str = (
        "You are a relevance scoring assistant. Your task is to score how relevant "
        "each document is to the user's query. Score each document from 0 to 10, where:\n"
        "- 0-2: Not relevant at all\n"
        "- 3-4: Slightly related but not helpful\n"
        "- 5-6: Somewhat relevant, might be useful\n"
        "- 7-8: Relevant and helpful\n"
        "- 9-10: Highly relevant, directly answers the query\n\n"
        "Be objective and consistent. Focus on whether the document actually helps "
        "answer the query, not just whether it contains similar words."
    )

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None:
        """
        Initialize the Simple LLM Reranker.

        Args:
            model: The LLM model to use for scoring.
                  If None, uses the LLM_MODEL from settings.
                  Consider using a fast, cheap model for reranking.

            api_key: OpenRouter API key.
                    If None, uses OPENROUTER_API_KEY from settings.

            base_url: OpenRouter API base URL.
                     If None, uses OPENROUTER_BASE_URL from settings.

        Example:
            # Use default settings
            reranker = SimpleLLMReranker()

            # Use a specific fast model for reranking
            reranker = SimpleLLMReranker(model="openai/gpt-3.5-turbo")
        """
        # Load configuration
        self.model: str = model if model is not None else settings.llm_model
        self.api_key: str = api_key if api_key is not None else (
            settings.openrouter_api_key or ""
        )
        self.base_url: str = base_url if base_url is not None else (
            settings.openrouter_base_url
        )

        # Validate API key
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required for Simple LLM Reranker. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )

        print(f"[SimpleLLMReranker] Initialized with model: {self.model}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using LLM-based relevance scoring.

        This method:
        1. Sends all documents to the LLM in a single prompt
        2. Asks the LLM to score each document 0-10
        3. Parses the scores and adds rerank_score to each document
        4. Filters by min_score and returns top_k documents

        Args:
            query: The user's question
            documents: List of documents to rerank
            top_k: Maximum number of documents to return
            min_score: Minimum rerank_score to include (0.0-1.0)

        Returns:
            List of reranked documents with rerank_score field

        HANDLING EDGE CASES:
        --------------------
        - Empty documents: Returns empty list
        - Single document: Returns it with a score
        - LLM fails: Returns original documents with neutral scores
        - Invalid scores: Assigns 0.5 as fallback
        """
        # Handle edge cases
        if not documents:
            print("[SimpleLLMReranker] No documents to rerank")
            return []

        if not query or not query.strip():
            print("[SimpleLLMReranker] Empty query, returning original order")
            return documents[:top_k]

        print(f"\n[SimpleLLMReranker] Reranking {len(documents)} documents...")
        print(f"[SimpleLLMReranker] Query: {query[:50]}...")

        try:
            # Get scores from LLM
            scores = self._score_documents(query, documents)

            # Add scores to documents
            scored_documents = []
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()  # Don't modify original
                doc_copy["rerank_score"] = scores.get(i, 0.5)  # Default: neutral score
                scored_documents.append(doc_copy)

            # Sort by rerank_score (highest first)
            scored_documents.sort(key=lambda d: d["rerank_score"], reverse=True)

            # Filter by min_score
            if min_score > 0:
                filtered = [d for d in scored_documents if d["rerank_score"] >= min_score]
                print(f"[SimpleLLMReranker] Filtered from {len(scored_documents)} to {len(filtered)} docs (min_score={min_score})")
                scored_documents = filtered

            # Return top_k
            result = scored_documents[:top_k]

            # Log results
            print(f"[SimpleLLMReranker] Returning {len(result)} reranked documents:")
            for i, doc in enumerate(result, 1):
                print(f"  {i}. {doc.get('id', 'unknown')}: rerank_score={doc['rerank_score']:.3f}")

            return result

        except Exception as e:
            # If scoring fails, return original documents with neutral scores
            print(f"[SimpleLLMReranker] ERROR: Scoring failed: {e}")
            print("[SimpleLLMReranker] Returning original documents with neutral scores")

            fallback = []
            for doc in documents[:top_k]:
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = 0.5  # Neutral score
                fallback.append(doc_copy)
            return fallback

    def _score_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[int, float]:
        """
        Get relevance scores for all documents from the LLM.

        This method batches all documents into a single prompt to minimize
        API calls and cost.

        Args:
            query: The user's question
            documents: Documents to score

        Returns:
            Dictionary mapping document index to score (0.0-1.0)
            Example: {0: 0.9, 1: 0.3, 2: 0.7}

        PROMPT STRUCTURE:
        -----------------
        We format the prompt as:
        - System: Instructions for scoring
        - User: Query + numbered list of documents
        - Expected response: JSON with document numbers and scores
        """
        import urllib.request
        import urllib.error

        # Build the document list for the prompt
        doc_list = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")[:500]  # Truncate long docs
            doc_list.append(f"[Document {i}]\n{content}")

        documents_text = "\n\n".join(doc_list)

        # Build the scoring prompt
        user_prompt = (
            f"Query: {query}\n\n"
            f"Documents to score:\n{documents_text}\n\n"
            f"Please score each document's relevance to the query from 0 to 10.\n"
            f"Respond ONLY with a JSON object mapping document numbers to scores.\n"
            f"Example: {{\"1\": 8, \"2\": 3, \"3\": 7}}\n"
            f"Your response:"
        )

        # Build API request
        url = f"{self.base_url}/chat/completions"
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0  # Deterministic scoring
        }

        json_data = json.dumps(request_body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rag-engine",
            "X-Title": "RAG Reranker"
        }

        request = urllib.request.Request(
            url=url,
            data=json_data,
            headers=headers,
            method="POST"
        )

        # Make API call
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response_body = response.read()
                response_data = json.loads(response_body.decode("utf-8"))

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"OpenRouter API error {e.code}: {error_body}")

        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error: {e.reason}")

        # Parse response
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Extract scores from JSON response
        scores = self._parse_scores(content, len(documents))

        return scores

    def _parse_scores(
        self,
        response: str,
        num_documents: int
    ) -> Dict[int, float]:
        """
        Parse LLM response to extract document scores.

        The LLM should respond with JSON like: {"1": 8, "2": 3, "3": 7}
        We convert this to 0-indexed scores normalized to 0.0-1.0.

        Args:
            response: The LLM's response text
            num_documents: Number of documents we asked about

        Returns:
            Dictionary mapping document index (0-based) to score (0.0-1.0)

        PARSING STRATEGY:
        -----------------
        1. Try to parse as JSON first
        2. If that fails, use regex to find number patterns
        3. Normalize scores from 0-10 to 0.0-1.0
        4. Fill missing scores with 0.5 (neutral)
        """
        scores: Dict[int, float] = {}

        # Strategy 1: Parse as JSON
        try:
            # Find JSON in the response (might be wrapped in text)
            json_match = re.search(r'\{[^{}]+\}', response)
            if json_match:
                raw_scores = json.loads(json_match.group())

                for key, value in raw_scores.items():
                    try:
                        # Convert "1" -> 0 (0-indexed)
                        doc_index = int(key) - 1
                        # Normalize 0-10 to 0.0-1.0
                        score = float(value) / 10.0
                        # Clamp to valid range
                        score = max(0.0, min(1.0, score))

                        if 0 <= doc_index < num_documents:
                            scores[doc_index] = score
                    except (ValueError, TypeError):
                        continue

        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 2: Regex fallback for patterns like "Document 1: 8"
        if not scores:
            patterns = [
                r'Document\s*(\d+)[:\s]+(\d+(?:\.\d+)?)',
                r'\[Document\s*(\d+)\][:\s]+(\d+(?:\.\d+)?)',
                r'"(\d+)":\s*(\d+(?:\.\d+)?)',
                r'(\d+)\s*[=:]\s*(\d+(?:\.\d+)?)'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    for match in matches:
                        try:
                            doc_index = int(match[0]) - 1
                            score = float(match[1]) / 10.0
                            score = max(0.0, min(1.0, score))

                            if 0 <= doc_index < num_documents:
                                scores[doc_index] = score
                        except (ValueError, TypeError):
                            continue
                    break

        # Fill missing scores with neutral value
        for i in range(num_documents):
            if i not in scores:
                scores[i] = 0.5
                print(f"[SimpleLLMReranker] WARNING: No score for doc {i+1}, using 0.5")

        print(f"[SimpleLLMReranker] Parsed {len(scores)} scores from LLM response")
        return scores

    def get_provider_name(self) -> str:
        """
        Get the name of this reranker provider.

        Returns:
            "simple" - indicating this is the simple LLM-based reranker
        """
        return "simple"
