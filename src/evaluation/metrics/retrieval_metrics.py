"""
Retrieval Metrics Module
========================

WHAT IS THIS MODULE?
--------------------
This module provides deterministic metrics for evaluating retrieval quality
in a RAG (Retrieval-Augmented Generation) system.

WHY RETRIEVAL METRICS?
----------------------
Good retrieval is the foundation of good RAG:
- If we don't retrieve relevant documents, the LLM can't generate good answers
- Retrieval metrics help us measure and improve this critical step
- They allow comparison between different retrieval configurations

METRICS PROVIDED:
-----------------
1. Recall@K: What fraction of relevant documents did we retrieve?
2. Precision@K: What fraction of retrieved documents are relevant?
3. Hit Rate: Did we retrieve at least one relevant document?
4. MRR (Mean Reciprocal Rank): How high are relevant documents ranked?

IMPORTANT - GROUND TRUTH:
-------------------------
These metrics require ground-truth annotations:
- A list of document IDs that are known to be relevant for each query
- Without ground truth, we can only compute basic statistics

When ground truth is NOT available:
- Use the basic_retrieval_stats() function for distribution metrics
- These include score statistics and coverage analysis

ALL METRICS ARE:
----------------
- Deterministic: Same input always produces same output
- Pure: No side effects, no external dependencies
- Normalized: Return values between 0.0 and 1.0
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass


@dataclass
class RetrievalMetricsResult:
    """
    Container for retrieval evaluation results.

    This dataclass holds all computed retrieval metrics in a structured format.
    Using a dataclass provides:
    - Clear documentation of what metrics are available
    - Type safety and IDE autocompletion
    - Easy conversion to dictionary for JSON serialization

    Attributes:
        recall_at_k: Fraction of relevant docs retrieved (0.0-1.0)
        precision_at_k: Fraction of retrieved docs that are relevant (0.0-1.0)
        hit_rate: Whether at least one relevant doc was retrieved (0.0 or 1.0)
        mrr: Mean Reciprocal Rank - how high relevant docs are ranked (0.0-1.0)
        k: The K value used for @K metrics
        num_retrieved: Total number of documents retrieved
        num_relevant: Total number of relevant documents (ground truth)
        num_relevant_retrieved: Number of relevant documents in retrieved set
    """
    recall_at_k: float
    precision_at_k: float
    hit_rate: float
    mrr: float
    k: int
    num_retrieved: int
    num_relevant: int
    num_relevant_retrieved: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "k": self.k,
            "num_retrieved": self.num_retrieved,
            "num_relevant": self.num_relevant,
            "num_relevant_retrieved": self.num_relevant_retrieved
        }


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: Optional[int] = None
) -> float:
    """
    Compute Recall@K: What fraction of relevant documents did we retrieve?

    WHAT IS RECALL@K?
    -----------------
    Recall measures completeness. It answers:
    "Of all the documents that SHOULD have been retrieved, how many did we get?"

    Formula: Recall@K = |relevant ∩ retrieved@K| / |relevant|

    EXAMPLE:
    --------
    - Relevant documents: [A, B, C, D] (4 total)
    - Retrieved documents: [A, X, B, Y, Z] (top 5)
    - Relevant in retrieved: [A, B] (2 of 4)
    - Recall@5 = 2/4 = 0.5

    WHY RECALL MATTERS:
    -------------------
    High recall means we're not missing relevant information.
    Low recall means we might be missing documents that could improve answers.

    Args:
        retrieved_ids: List of document IDs retrieved, in ranked order.
        relevant_ids: List of document IDs that are known to be relevant.
        k: Number of top results to consider. If None, uses all retrieved.

    Returns:
        Recall score between 0.0 and 1.0.
        Returns 0.0 if there are no relevant documents.

    Example:
        >>> compute_recall_at_k(['d1', 'd2', 'd3'], ['d1', 'd4'], k=3)
        0.5  # Found 1 of 2 relevant docs
    """
    # Handle edge cases
    if not relevant_ids:
        return 0.0  # No relevant docs means recall is undefined, return 0

    if not retrieved_ids:
        return 0.0  # Retrieved nothing

    # Apply K limit if specified
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    # Convert to sets for efficient intersection
    retrieved_set: Set[str] = set(retrieved_ids)
    relevant_set: Set[str] = set(relevant_ids)

    # Count relevant documents in retrieved set
    relevant_retrieved = len(retrieved_set & relevant_set)

    # Recall = relevant retrieved / total relevant
    return relevant_retrieved / len(relevant_set)


def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: Optional[int] = None
) -> float:
    """
    Compute Precision@K: What fraction of retrieved documents are relevant?

    WHAT IS PRECISION@K?
    --------------------
    Precision measures accuracy. It answers:
    "Of all the documents we retrieved, how many are actually relevant?"

    Formula: Precision@K = |relevant ∩ retrieved@K| / K

    EXAMPLE:
    --------
    - Relevant documents: [A, B, C, D]
    - Retrieved documents: [A, X, B, Y, Z] (top 5)
    - Relevant in retrieved: [A, B] (2 of 5 retrieved)
    - Precision@5 = 2/5 = 0.4

    WHY PRECISION MATTERS:
    ----------------------
    High precision means we're not wasting the LLM's context with irrelevant docs.
    Low precision means we're polluting context with noise.

    PRECISION VS RECALL:
    --------------------
    - High recall, low precision: "Get everything, even if some is junk"
    - High precision, low recall: "Get only good stuff, but might miss some"
    - Ideal: High both (but there's usually a tradeoff)

    Args:
        retrieved_ids: List of document IDs retrieved, in ranked order.
        relevant_ids: List of document IDs that are known to be relevant.
        k: Number of top results to consider. If None, uses all retrieved.

    Returns:
        Precision score between 0.0 and 1.0.
        Returns 0.0 if no documents were retrieved.

    Example:
        >>> compute_precision_at_k(['d1', 'd2', 'd3'], ['d1', 'd4'], k=3)
        0.333...  # 1 of 3 retrieved is relevant
    """
    # Handle edge cases
    if not retrieved_ids:
        return 0.0  # Retrieved nothing

    # Apply K limit if specified
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    if not retrieved_ids:  # After K limit
        return 0.0

    # Convert to sets for efficient intersection
    retrieved_set: Set[str] = set(retrieved_ids)
    relevant_set: Set[str] = set(relevant_ids)

    # Count relevant documents in retrieved set
    relevant_retrieved = len(retrieved_set & relevant_set)

    # Precision = relevant retrieved / total retrieved (up to K)
    return relevant_retrieved / len(retrieved_ids)


def compute_hit_rate(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: Optional[int] = None
) -> float:
    """
    Compute Hit Rate: Did we retrieve at least one relevant document?

    WHAT IS HIT RATE?
    -----------------
    Hit rate (also called "success rate") is a binary metric:
    - 1.0 if we found at least one relevant document
    - 0.0 if we found no relevant documents

    WHY HIT RATE?
    -------------
    Sometimes you just need to know: "Did retrieval find ANYTHING useful?"
    - Recall and precision give nuanced scores
    - Hit rate is a simple success/failure indicator
    - Useful for understanding baseline retrieval quality

    EXAMPLE:
    --------
    - Query: "What is machine learning?"
    - Relevant docs: [doc_ml_intro, doc_ml_basics]
    - Retrieved: [doc_random, doc_ml_intro, doc_other]
    - Hit Rate = 1.0 (we found doc_ml_intro)

    Args:
        retrieved_ids: List of document IDs retrieved, in ranked order.
        relevant_ids: List of document IDs that are known to be relevant.
        k: Number of top results to consider. If None, uses all retrieved.

    Returns:
        1.0 if at least one relevant document was retrieved, 0.0 otherwise.

    Example:
        >>> compute_hit_rate(['d1', 'd2'], ['d1', 'd3'])
        1.0  # d1 is relevant and was retrieved
        >>> compute_hit_rate(['d2', 'd4'], ['d1', 'd3'])
        0.0  # No relevant docs in retrieved
    """
    # Handle edge cases
    if not relevant_ids or not retrieved_ids:
        return 0.0

    # Apply K limit if specified
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    # Check if any relevant document is in retrieved set
    retrieved_set: Set[str] = set(retrieved_ids)
    relevant_set: Set[str] = set(relevant_ids)

    # Hit if intersection is non-empty
    return 1.0 if (retrieved_set & relevant_set) else 0.0


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: Optional[int] = None
) -> float:
    """
    Compute MRR (Mean Reciprocal Rank): How high are relevant documents ranked?

    WHAT IS MRR?
    ------------
    MRR measures the position of the FIRST relevant document:
    - If first relevant doc is at position 1: MRR = 1.0
    - If first relevant doc is at position 2: MRR = 0.5
    - If first relevant doc is at position 3: MRR = 0.333...
    - If no relevant doc found: MRR = 0.0

    Formula: MRR = 1 / rank_of_first_relevant

    WHY MRR?
    --------
    MRR captures ranking quality:
    - High MRR means relevant docs appear early
    - Low MRR means relevant docs are buried
    - Important because users (and LLMs) see top results first

    EXAMPLE:
    --------
    - Relevant documents: [A, B]
    - Retrieved documents: [X, A, Y, B, Z]
    - First relevant (A) is at position 2
    - MRR = 1/2 = 0.5

    NOTE ON "MEAN":
    ---------------
    This function computes RR (Reciprocal Rank) for a single query.
    "Mean" Reciprocal Rank averages RR across multiple queries.
    Use compute_average_mrr() for multiple queries.

    Args:
        retrieved_ids: List of document IDs retrieved, in ranked order.
        relevant_ids: List of document IDs that are known to be relevant.
        k: Number of top results to consider. If None, uses all retrieved.

    Returns:
        Reciprocal rank between 0.0 and 1.0.
        Returns 0.0 if no relevant documents are found.

    Example:
        >>> compute_mrr(['d2', 'd1', 'd3'], ['d1'])
        0.5  # d1 is at position 2, so RR = 1/2
    """
    # Handle edge cases
    if not relevant_ids or not retrieved_ids:
        return 0.0

    # Apply K limit if specified
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    # Convert relevant IDs to set for O(1) lookup
    relevant_set: Set[str] = set(relevant_ids)

    # Find position of first relevant document (1-indexed)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    # No relevant document found
    return 0.0


def compute_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: Optional[int] = None
) -> RetrievalMetricsResult:
    """
    Compute all retrieval metrics at once.

    This is the main entry point for retrieval evaluation.
    It computes all metrics in a single pass for efficiency.

    Args:
        retrieved_ids: List of document IDs retrieved, in ranked order.
        relevant_ids: List of document IDs that are known to be relevant.
        k: Number of top results to consider. If None, uses all retrieved.

    Returns:
        RetrievalMetricsResult containing all metrics.

    Example:
        >>> results = compute_retrieval_metrics(
        ...     retrieved_ids=['d1', 'd2', 'd3', 'd4', 'd5'],
        ...     relevant_ids=['d1', 'd3', 'd6'],
        ...     k=5
        ... )
        >>> print(f"Recall@5: {results.recall_at_k:.2f}")
        >>> print(f"Precision@5: {results.precision_at_k:.2f}")
    """
    # Determine effective K
    effective_k = k if k is not None else len(retrieved_ids)

    # Apply K limit
    retrieved_at_k = retrieved_ids[:effective_k] if k else retrieved_ids

    # Convert to sets once for all computations
    retrieved_set: Set[str] = set(retrieved_at_k)
    relevant_set: Set[str] = set(relevant_ids)

    # Count intersections
    relevant_retrieved = len(retrieved_set & relevant_set)

    # Compute metrics
    recall = relevant_retrieved / len(relevant_set) if relevant_set else 0.0
    precision = relevant_retrieved / len(retrieved_at_k) if retrieved_at_k else 0.0
    hit = 1.0 if relevant_retrieved > 0 else 0.0

    # Compute MRR (requires ordered traversal)
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_at_k, start=1):
        if doc_id in relevant_set:
            mrr = 1.0 / rank
            break

    return RetrievalMetricsResult(
        recall_at_k=recall,
        precision_at_k=precision,
        hit_rate=hit,
        mrr=mrr,
        k=effective_k,
        num_retrieved=len(retrieved_at_k),
        num_relevant=len(relevant_set),
        num_relevant_retrieved=relevant_retrieved
    )


def compute_average_metrics(
    results_list: List[RetrievalMetricsResult]
) -> Dict[str, float]:
    """
    Compute average metrics across multiple queries.

    For batch evaluation, this function averages metrics across
    multiple query results to get aggregate performance.

    Args:
        results_list: List of RetrievalMetricsResult from multiple queries.

    Returns:
        Dictionary with average values for each metric.

    Example:
        >>> results = [
        ...     compute_retrieval_metrics(['d1'], ['d1', 'd2'], k=5),
        ...     compute_retrieval_metrics(['d3'], ['d3'], k=5),
        ... ]
        >>> avg = compute_average_metrics(results)
        >>> print(f"Average Recall@5: {avg['recall_at_k']:.2f}")
    """
    if not results_list:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "num_queries": 0
        }

    n = len(results_list)

    return {
        "recall_at_k": sum(r.recall_at_k for r in results_list) / n,
        "precision_at_k": sum(r.precision_at_k for r in results_list) / n,
        "hit_rate": sum(r.hit_rate for r in results_list) / n,
        "mrr": sum(r.mrr for r in results_list) / n,
        "num_queries": n
    }


@dataclass
class BasicRetrievalStats:
    """
    Basic retrieval statistics when ground truth is not available.

    These metrics don't require knowing which documents are relevant.
    They provide insights into retrieval behavior and score distributions.
    """
    num_retrieved: int
    avg_score: float
    min_score: float
    max_score: float
    score_std: float
    has_scores: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_retrieved": self.num_retrieved,
            "avg_score": self.avg_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "score_std": self.score_std,
            "has_scores": self.has_scores
        }


def compute_basic_retrieval_stats(
    retrieved_documents: List[Dict[str, Any]],
    score_key: str = "score"
) -> BasicRetrievalStats:
    """
    Compute basic retrieval statistics when ground truth is not available.

    WHEN TO USE:
    ------------
    - No ground truth labels exist
    - Quick health check of retrieval
    - Understanding score distributions

    WHAT IT MEASURES:
    -----------------
    - Number of documents retrieved
    - Score statistics (mean, min, max, std)
    - Whether scores are present

    Args:
        retrieved_documents: List of retrieved document dictionaries.
                            Each should have an ID and optionally a score.
        score_key: Key to use for extracting scores (default: "score").
                  Also checks "rerank_score" if primary key not found.

    Returns:
        BasicRetrievalStats with score distribution info.

    Example:
        >>> docs = [
        ...     {"id": "d1", "score": 0.9},
        ...     {"id": "d2", "score": 0.7},
        ... ]
        >>> stats = compute_basic_retrieval_stats(docs)
        >>> print(f"Average score: {stats.avg_score:.2f}")
    """
    if not retrieved_documents:
        return BasicRetrievalStats(
            num_retrieved=0,
            avg_score=0.0,
            min_score=0.0,
            max_score=0.0,
            score_std=0.0,
            has_scores=False
        )

    # Extract scores - try primary key first, then rerank_score
    scores: List[float] = []
    for doc in retrieved_documents:
        score = doc.get(score_key)
        if score is None:
            score = doc.get("rerank_score")
        if score is not None:
            scores.append(float(score))

    has_scores = len(scores) > 0

    if not has_scores:
        return BasicRetrievalStats(
            num_retrieved=len(retrieved_documents),
            avg_score=0.0,
            min_score=0.0,
            max_score=0.0,
            score_std=0.0,
            has_scores=False
        )

    # Compute statistics
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)

    # Compute standard deviation
    if len(scores) > 1:
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        score_std = variance ** 0.5
    else:
        score_std = 0.0

    return BasicRetrievalStats(
        num_retrieved=len(retrieved_documents),
        avg_score=avg_score,
        min_score=min_score,
        max_score=max_score,
        score_std=score_std,
        has_scores=True
    )


def compute_reranking_impact(
    pre_rerank_ids: List[str],
    post_rerank_ids: List[str],
    relevant_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Measure the impact of reranking on retrieval quality.

    WHAT THIS MEASURES:
    -------------------
    - How much did reranking change the document order?
    - Did reranking improve metrics? (if ground truth available)
    - Overlap between pre and post reranking sets

    WHY THIS MATTERS:
    -----------------
    Reranking adds latency and cost. This function helps determine:
    - Is reranking actually improving results?
    - How much is the ranking changing?
    - Should we enable/disable reranking?

    Args:
        pre_rerank_ids: Document IDs before reranking (in order).
        post_rerank_ids: Document IDs after reranking (in order).
        relevant_ids: Optional ground truth for computing metric changes.

    Returns:
        Dictionary with impact metrics:
        - order_changed: bool - Did the order change at all?
        - overlap_ratio: float - Fraction of docs in both sets
        - position_changes: int - Sum of absolute position changes
        - metrics_before: dict - Metrics before reranking (if ground truth)
        - metrics_after: dict - Metrics after reranking (if ground truth)
        - metric_improvements: dict - Change in each metric (if ground truth)

    Example:
        >>> impact = compute_reranking_impact(
        ...     pre_rerank_ids=['d1', 'd2', 'd3'],
        ...     post_rerank_ids=['d2', 'd1', 'd3'],
        ...     relevant_ids=['d2']
        ... )
        >>> print(f"Order changed: {impact['order_changed']}")
    """
    result: Dict[str, Any] = {}

    # Check if order changed
    result["order_changed"] = pre_rerank_ids != post_rerank_ids

    # Compute overlap
    pre_set = set(pre_rerank_ids)
    post_set = set(post_rerank_ids)
    overlap = len(pre_set & post_set)
    total_unique = len(pre_set | post_set)
    result["overlap_ratio"] = overlap / total_unique if total_unique > 0 else 1.0

    # Compute position changes
    pre_positions = {doc_id: i for i, doc_id in enumerate(pre_rerank_ids)}
    post_positions = {doc_id: i for i, doc_id in enumerate(post_rerank_ids)}

    position_changes = 0
    for doc_id in pre_set & post_set:
        position_changes += abs(pre_positions[doc_id] - post_positions[doc_id])
    result["position_changes"] = position_changes

    # Compute metric changes if ground truth available
    if relevant_ids:
        k = max(len(pre_rerank_ids), len(post_rerank_ids))

        metrics_before = compute_retrieval_metrics(pre_rerank_ids, relevant_ids, k)
        metrics_after = compute_retrieval_metrics(post_rerank_ids, relevant_ids, k)

        result["metrics_before"] = metrics_before.to_dict()
        result["metrics_after"] = metrics_after.to_dict()
        result["metric_improvements"] = {
            "recall_at_k": metrics_after.recall_at_k - metrics_before.recall_at_k,
            "precision_at_k": metrics_after.precision_at_k - metrics_before.precision_at_k,
            "hit_rate": metrics_after.hit_rate - metrics_before.hit_rate,
            "mrr": metrics_after.mrr - metrics_before.mrr
        }

    return result
