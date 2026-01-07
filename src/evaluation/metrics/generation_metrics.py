"""
Generation Metrics Module
=========================

WHAT IS THIS MODULE?
--------------------
This module provides metrics for evaluating the quality of generated answers
in a RAG (Retrieval-Augmented Generation) system.

WHY GENERATION METRICS?
-----------------------
Generation is the final step in RAG. Even with good retrieval, the LLM might:
- Generate answers not grounded in the context (hallucination)
- Ignore relevant information from retrieved documents
- Produce overly long or short answers
- Fail to properly synthesize information from multiple sources

These metrics help measure and improve generation quality.

METRICS PROVIDED:
-----------------
1. Faithfulness Score: How well is the answer grounded in the context?
2. Context Coverage: How much of the retrieved context was used?
3. Answer Length Analysis: Basic statistics about answer length
4. Hallucination Risk: Estimate of potential hallucination

IMPORTANT - NO LLM CALLS:
-------------------------
These metrics are designed to be FAST and DETERMINISTIC.
They use text analysis heuristics, not LLM calls.
For LLM-based evaluation, see ragas_adapter.py.

ALL METRICS ARE:
----------------
- Deterministic: Same input always produces same output
- Fast: No external API calls required
- Production-safe: Can run on every query without cost concerns
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re


@dataclass
class GenerationMetricsResult:
    """
    Container for generation evaluation results.

    This dataclass holds all computed generation metrics in a structured format.

    Attributes:
        faithfulness_score: Estimate of how grounded the answer is (0.0-1.0)
        context_coverage: Fraction of context terms used in answer (0.0-1.0)
        answer_length: Number of characters in the answer
        answer_word_count: Number of words in the answer
        context_length: Total characters in provided context
        context_utilization_ratio: answer_length / context_length
        hallucination_risk: Estimate of hallucination risk (0.0-1.0, lower is better)
        key_terms_in_answer: Number of key context terms found in answer
        total_key_terms: Total key terms extracted from context
    """
    faithfulness_score: float
    context_coverage: float
    answer_length: int
    answer_word_count: int
    context_length: int
    context_utilization_ratio: float
    hallucination_risk: float
    key_terms_in_answer: int
    total_key_terms: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "faithfulness_score": round(self.faithfulness_score, 4),
            "context_coverage": round(self.context_coverage, 4),
            "answer_length": self.answer_length,
            "answer_word_count": self.answer_word_count,
            "context_length": self.context_length,
            "context_utilization_ratio": round(self.context_utilization_ratio, 4),
            "hallucination_risk": round(self.hallucination_risk, 4),
            "key_terms_in_answer": self.key_terms_in_answer,
            "total_key_terms": self.total_key_terms
        }


def _extract_key_terms(text: str, min_length: int = 4) -> Set[str]:
    """
    Extract key terms from text for comparison.

    This function extracts meaningful words/terms from text, filtering out
    common stop words and short words.

    Args:
        text: The text to extract terms from
        min_length: Minimum word length to consider (default: 4)

    Returns:
        Set of lowercase key terms
    """
    # Common English stop words to filter out
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "it", "its", "they", "them",
        "their", "what", "which", "who", "whom", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "also", "into", "over", "after",
        "before", "between", "through", "during", "about", "being", "here",
        "there", "then", "now", "any", "because", "while", "although"
    }

    # Extract words using regex (alphanumeric only)
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

    # Filter: remove stop words and short words
    key_terms = {
        word for word in words
        if word not in stop_words and len(word) >= min_length
    }

    return key_terms


def _extract_ngrams(text: str, n: int = 2) -> Set[str]:
    """
    Extract n-grams from text for more precise matching.

    Args:
        text: The text to extract n-grams from
        n: Size of n-grams (default: 2 for bigrams)

    Returns:
        Set of lowercase n-gram strings
    """
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    if len(words) < n:
        return set()

    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i + n])
        ngrams.add(ngram)

    return ngrams


def compute_faithfulness_score(
    answer: str,
    context_documents: List[Dict[str, Any]],
    use_ngrams: bool = True
) -> float:
    """
    Compute faithfulness score: How well is the answer grounded in context?

    WHAT IS FAITHFULNESS?
    ---------------------
    Faithfulness measures whether the answer contains information that
    can be traced back to the provided context. A faithful answer doesn't
    introduce new claims that aren't supported by the context.

    HOW IT WORKS:
    -------------
    1. Extract key terms and n-grams from the answer
    2. Extract key terms and n-grams from all context documents
    3. Calculate overlap: answer terms found in context
    4. Higher overlap = higher faithfulness

    LIMITATIONS:
    ------------
    This is a heuristic approximation. It cannot detect:
    - Semantic faithfulness (paraphrasing)
    - Logical inference
    - Factual correctness

    For more accurate faithfulness, use RAGAS or LLM-based evaluation.

    Args:
        answer: The generated answer text
        context_documents: List of context documents, each with "content" key
        use_ngrams: Whether to also check bigram overlap (default: True)

    Returns:
        Faithfulness score between 0.0 and 1.0
        - 1.0: All answer terms found in context (highly faithful)
        - 0.0: No answer terms found in context (likely hallucination)

    Example:
        >>> context = [{"content": "RAG improves accuracy by using retrieval."}]
        >>> answer = "RAG improves accuracy through retrieval mechanisms."
        >>> score = compute_faithfulness_score(answer, context)
        >>> print(f"Faithfulness: {score:.2f}")  # High score expected
    """
    if not answer or not answer.strip():
        return 0.0

    if not context_documents:
        return 0.0

    # Combine all context into one text
    context_text = " ".join(
        doc.get("content", "") or doc.get("text", "")
        for doc in context_documents
    )

    if not context_text.strip():
        return 0.0

    # Extract key terms
    answer_terms = _extract_key_terms(answer)
    context_terms = _extract_key_terms(context_text)

    if not answer_terms:
        return 1.0  # Empty answer with no key terms is technically faithful

    # Calculate term overlap
    term_overlap = len(answer_terms & context_terms)
    term_ratio = term_overlap / len(answer_terms)

    if use_ngrams:
        # Also check bigram overlap for phrase-level matching
        answer_ngrams = _extract_ngrams(answer, n=2)
        context_ngrams = _extract_ngrams(context_text, n=2)

        if answer_ngrams:
            ngram_overlap = len(answer_ngrams & context_ngrams)
            ngram_ratio = ngram_overlap / len(answer_ngrams)
            # Weighted average: terms (60%) + ngrams (40%)
            return 0.6 * term_ratio + 0.4 * ngram_ratio

    return term_ratio


def compute_context_coverage(
    answer: str,
    context_documents: List[Dict[str, Any]]
) -> float:
    """
    Compute context coverage: How much context information was used?

    WHAT IS CONTEXT COVERAGE?
    -------------------------
    Context coverage measures what fraction of the retrieved context
    appears in the generated answer. Low coverage might mean:
    - Retrieved documents weren't relevant
    - LLM ignored some relevant information
    - Answer is too short

    Args:
        answer: The generated answer text
        context_documents: List of context documents

    Returns:
        Coverage score between 0.0 and 1.0
        - 1.0: All context key terms used in answer
        - 0.0: No context terms used

    Example:
        >>> context = [{"content": "Python is great for AI and ML."}]
        >>> answer = "Python excels in AI applications."
        >>> coverage = compute_context_coverage(answer, context)
    """
    if not answer or not answer.strip():
        return 0.0

    if not context_documents:
        return 0.0

    # Combine all context
    context_text = " ".join(
        doc.get("content", "") or doc.get("text", "")
        for doc in context_documents
    )

    if not context_text.strip():
        return 0.0

    # Extract key terms
    answer_terms = _extract_key_terms(answer)
    context_terms = _extract_key_terms(context_text)

    if not context_terms:
        return 1.0  # No context terms to cover

    # Calculate what fraction of context terms appear in answer
    terms_used = len(context_terms & answer_terms)
    return terms_used / len(context_terms)


def compute_hallucination_risk(
    answer: str,
    context_documents: List[Dict[str, Any]],
    threshold: float = 0.3
) -> float:
    """
    Estimate hallucination risk based on answer-context divergence.

    WHAT IS HALLUCINATION RISK?
    ---------------------------
    Hallucination risk estimates how likely the answer contains
    information not supported by the context. This is the inverse
    of faithfulness, with additional heuristics.

    HOW IT WORKS:
    -------------
    1. Calculate faithfulness score
    2. Identify answer terms NOT in context
    3. Weight by term significance (longer terms = higher risk if missing)
    4. Return risk score (1 - adjusted_faithfulness)

    INTERPRETATION:
    ---------------
    - 0.0-0.2: Low risk (answer well-grounded)
    - 0.2-0.5: Moderate risk (some unsupported content)
    - 0.5-1.0: High risk (significant hallucination likely)

    Args:
        answer: The generated answer text
        context_documents: List of context documents
        threshold: Minimum faithfulness for low risk (default: 0.3)

    Returns:
        Hallucination risk score between 0.0 and 1.0
        - 0.0: Very low risk (highly faithful)
        - 1.0: Very high risk (likely hallucinating)
    """
    # Calculate base faithfulness
    faithfulness = compute_faithfulness_score(answer, context_documents)

    # Basic risk is inverse of faithfulness
    base_risk = 1.0 - faithfulness

    # Additional check: count "novel" terms in answer
    if context_documents:
        context_text = " ".join(
            doc.get("content", "") or doc.get("text", "")
            for doc in context_documents
        )

        answer_terms = _extract_key_terms(answer)
        context_terms = _extract_key_terms(context_text)

        # Novel terms: in answer but not in context
        novel_terms = answer_terms - context_terms

        if answer_terms:
            # Higher ratio of novel terms = higher risk
            novel_ratio = len(novel_terms) / len(answer_terms)
            # Combine base risk with novel term ratio
            combined_risk = 0.5 * base_risk + 0.5 * novel_ratio
            return min(1.0, combined_risk)

    return base_risk


def compute_answer_length_metrics(
    answer: str
) -> Dict[str, int]:
    """
    Compute basic length metrics for the answer.

    Args:
        answer: The generated answer text

    Returns:
        Dictionary with:
        - character_count: Number of characters
        - word_count: Number of words
        - sentence_count: Estimated number of sentences
    """
    if not answer:
        return {
            "character_count": 0,
            "word_count": 0,
            "sentence_count": 0
        }

    # Character count (excluding leading/trailing whitespace)
    char_count = len(answer.strip())

    # Word count
    words = answer.split()
    word_count = len(words)

    # Sentence count (rough estimate based on sentence-ending punctuation)
    sentences = re.split(r'[.!?]+', answer)
    sentence_count = len([s for s in sentences if s.strip()])

    return {
        "character_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count
    }


def compute_generation_metrics(
    answer: str,
    context_documents: List[Dict[str, Any]],
    question: Optional[str] = None
) -> GenerationMetricsResult:
    """
    Compute all generation metrics at once.

    This is the main entry point for generation evaluation.
    It computes all metrics in a single pass for efficiency.

    Args:
        answer: The generated answer text
        context_documents: List of context documents used for generation
        question: Optional original question (for future metrics)

    Returns:
        GenerationMetricsResult containing all metrics.

    Example:
        >>> answer = "RAG combines retrieval with generation for better answers."
        >>> context = [{"content": "RAG uses retrieval to augment generation."}]
        >>> results = compute_generation_metrics(answer, context)
        >>> print(f"Faithfulness: {results.faithfulness_score:.2f}")
        >>> print(f"Hallucination Risk: {results.hallucination_risk:.2f}")
    """
    # Combine context for length calculation
    context_text = " ".join(
        doc.get("content", "") or doc.get("text", "")
        for doc in context_documents
    ) if context_documents else ""

    # Compute individual metrics
    faithfulness = compute_faithfulness_score(answer, context_documents)
    coverage = compute_context_coverage(answer, context_documents)
    hallucination_risk = compute_hallucination_risk(answer, context_documents)
    length_metrics = compute_answer_length_metrics(answer)

    # Extract term counts for reporting
    answer_terms = _extract_key_terms(answer)
    context_terms = _extract_key_terms(context_text)
    terms_in_both = len(answer_terms & context_terms)

    # Context utilization ratio
    answer_len = len(answer.strip()) if answer else 0
    context_len = len(context_text.strip()) if context_text else 0
    utilization_ratio = answer_len / context_len if context_len > 0 else 0.0

    return GenerationMetricsResult(
        faithfulness_score=faithfulness,
        context_coverage=coverage,
        answer_length=length_metrics["character_count"],
        answer_word_count=length_metrics["word_count"],
        context_length=context_len,
        context_utilization_ratio=utilization_ratio,
        hallucination_risk=hallucination_risk,
        key_terms_in_answer=terms_in_both,
        total_key_terms=len(context_terms)
    )


def compute_average_generation_metrics(
    results_list: List[GenerationMetricsResult]
) -> Dict[str, float]:
    """
    Compute average metrics across multiple evaluations.

    For batch evaluation, this function averages metrics across
    multiple query results to get aggregate performance.

    Args:
        results_list: List of GenerationMetricsResult from multiple queries.

    Returns:
        Dictionary with average values for each metric.

    Example:
        >>> results = [
        ...     compute_generation_metrics(answer1, context1),
        ...     compute_generation_metrics(answer2, context2),
        ... ]
        >>> avg = compute_average_generation_metrics(results)
        >>> print(f"Average Faithfulness: {avg['faithfulness_score']:.2f}")
    """
    if not results_list:
        return {
            "faithfulness_score": 0.0,
            "context_coverage": 0.0,
            "hallucination_risk": 0.0,
            "context_utilization_ratio": 0.0,
            "avg_answer_length": 0.0,
            "avg_answer_word_count": 0.0,
            "num_evaluations": 0
        }

    n = len(results_list)

    return {
        "faithfulness_score": sum(r.faithfulness_score for r in results_list) / n,
        "context_coverage": sum(r.context_coverage for r in results_list) / n,
        "hallucination_risk": sum(r.hallucination_risk for r in results_list) / n,
        "context_utilization_ratio": sum(r.context_utilization_ratio for r in results_list) / n,
        "avg_answer_length": sum(r.answer_length for r in results_list) / n,
        "avg_answer_word_count": sum(r.answer_word_count for r in results_list) / n,
        "num_evaluations": n
    }
