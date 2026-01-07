"""
RAG Evaluator Module
====================

WHAT IS THIS MODULE?
--------------------
This module provides the main RAGEvaluator class that orchestrates all
evaluation metrics for the RAG system. It is the single entry point for
evaluating RAG pipeline quality.

WHY A UNIFIED EVALUATOR?
------------------------
1. SIMPLICITY: One class to call for all evaluation needs
2. CONFIGURATION: Reads from settings, respects enable/disable flags
3. ORCHESTRATION: Combines retrieval and generation metrics
4. LOGGING: Consistent logging of evaluation results
5. HISTORY: Optional storage of evaluation history for analysis

DESIGN PRINCIPLES:
------------------
1. NON-BLOCKING: Evaluation never affects the RAG response
2. CONFIG-DRIVEN: All features controlled by environment variables
3. OPTIONAL: Disabled by default, zero overhead when off
4. PRODUCTION-SAFE: No external calls in default mode
5. BACKWARD COMPATIBLE: Does NOT change existing pipeline behavior

USAGE:
------
The evaluator is called AFTER the RAG pipeline generates a response.
It measures quality but does NOT modify the response.

    from src.evaluation.evaluator import RAGEvaluator

    evaluator = RAGEvaluator()
    result = evaluator.evaluate(
        question="What is RAG?",
        answer="RAG combines retrieval with generation...",
        retrieved_documents=[...],
        ground_truth_ids=["doc1", "doc2"]  # Optional
    )

    print(result["retrieval_metrics"])
    print(result["generation_metrics"])
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Import metrics modules
from src.evaluation.metrics.retrieval_metrics import (
    compute_retrieval_metrics,
    compute_basic_retrieval_stats,
    RetrievalMetricsResult,
    BasicRetrievalStats
)
from src.evaluation.metrics.generation_metrics import (
    compute_generation_metrics,
    GenerationMetricsResult
)
from src.evaluation.metrics.ragas_adapter import (
    compute_ragas_metrics,
    is_ragas_available,
    RAGASMetricsResult
)

# Import settings
from src.core.config import settings

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Container for complete evaluation results.

    This dataclass holds all evaluation metrics from a single RAG query,
    including retrieval metrics, generation metrics, and optional RAGAS metrics.

    Attributes:
        timestamp: When the evaluation was performed
        question: The original question
        retrieval_metrics: Results from retrieval evaluation
        retrieval_stats: Basic statistics when no ground truth available
        generation_metrics: Results from generation evaluation
        ragas_metrics: Optional RAGAS metrics (if enabled)
        evaluation_config: Configuration used for this evaluation
    """
    timestamp: str
    question: str
    retrieval_metrics: Optional[Dict[str, Any]] = None
    retrieval_stats: Optional[Dict[str, Any]] = None
    generation_metrics: Optional[Dict[str, Any]] = None
    ragas_metrics: Optional[Dict[str, Any]] = None
    evaluation_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp,
            "question": self.question,
            "evaluation_config": self.evaluation_config
        }

        if self.retrieval_metrics:
            result["retrieval_metrics"] = self.retrieval_metrics
        if self.retrieval_stats:
            result["retrieval_stats"] = self.retrieval_stats
        if self.generation_metrics:
            result["generation_metrics"] = self.generation_metrics
        if self.ragas_metrics:
            result["ragas_metrics"] = self.ragas_metrics

        return result

    def get_summary(self) -> Dict[str, float]:
        """
        Get a summary of key metrics for quick overview.

        Returns:
            Dictionary with key metric values
        """
        summary = {}

        if self.retrieval_metrics:
            summary["precision_at_k"] = self.retrieval_metrics.get("precision_at_k", 0.0)
            summary["recall_at_k"] = self.retrieval_metrics.get("recall_at_k", 0.0)
            summary["mrr"] = self.retrieval_metrics.get("mrr", 0.0)

        if self.retrieval_stats:
            summary["avg_retrieval_score"] = self.retrieval_stats.get("avg_score", 0.0)

        if self.generation_metrics:
            summary["faithfulness"] = self.generation_metrics.get("faithfulness_score", 0.0)
            summary["hallucination_risk"] = self.generation_metrics.get("hallucination_risk", 0.0)

        if self.ragas_metrics and self.ragas_metrics.get("evaluation_successful"):
            if self.ragas_metrics.get("overall_score") is not None:
                summary["ragas_overall"] = self.ragas_metrics["overall_score"]

        return summary


class RAGEvaluator:
    """
    Main evaluator class for RAG system quality assessment.

    This class orchestrates all evaluation metrics and provides a unified
    interface for measuring RAG quality.

    CONFIGURATION:
    --------------
    The evaluator respects these settings from config.py:
    - ENABLE_EVALUATION: Master switch (default: false)
    - EVALUATION_DEFAULT_K: K value for @K metrics (default: 5)
    - ENABLE_RAGAS: Whether to run RAGAS metrics (default: false)
    - EVALUATION_LOG_RESULTS: Whether to log results (default: true)
    - EVALUATION_STORE_HISTORY: Whether to store history (default: false)

    THREAD SAFETY:
    --------------
    This class is thread-safe. Multiple threads can call evaluate()
    concurrently without issues.

    Example:
        >>> evaluator = RAGEvaluator()
        >>> result = evaluator.evaluate(
        ...     question="What is RAG?",
        ...     answer="RAG is...",
        ...     retrieved_documents=[{"id": "doc1", "content": "..."}]
        ... )
        >>> print(result.get_summary())
    """

    def __init__(
        self,
        enable_evaluation: Optional[bool] = None,
        enable_ragas: Optional[bool] = None,
        default_k: Optional[int] = None,
        log_results: Optional[bool] = None,
        store_history: Optional[bool] = None
    ):
        """
        Initialize the RAG Evaluator.

        Args:
            enable_evaluation: Override for ENABLE_EVALUATION setting
            enable_ragas: Override for ENABLE_RAGAS setting
            default_k: Override for EVALUATION_DEFAULT_K setting
            log_results: Override for EVALUATION_LOG_RESULTS setting
            store_history: Override for EVALUATION_STORE_HISTORY setting

        All parameters default to None, which means use settings from config.
        """
        # Use provided values or fall back to settings
        self._enabled = enable_evaluation if enable_evaluation is not None else settings.enable_evaluation
        self._enable_ragas = enable_ragas if enable_ragas is not None else settings.enable_ragas
        self._default_k = default_k if default_k is not None else settings.evaluation_default_k
        self._log_results = log_results if log_results is not None else settings.evaluation_log_results
        self._store_history = store_history if store_history is not None else settings.evaluation_store_history

        # History storage
        self._history: List[EvaluationResult] = []

        # Log initialization
        if self._enabled:
            logger.info("RAGEvaluator initialized (ENABLED)")
            logger.info(f"  - Default K: {self._default_k}")
            logger.info(f"  - RAGAS: {'enabled' if self._enable_ragas else 'disabled'}")
            logger.info(f"  - Log results: {self._log_results}")
            logger.info(f"  - Store history: {self._store_history}")
        else:
            logger.debug("RAGEvaluator initialized (DISABLED)")

    @property
    def is_enabled(self) -> bool:
        """Check if evaluation is enabled."""
        return self._enabled

    @property
    def is_ragas_enabled(self) -> bool:
        """Check if RAGAS evaluation is enabled."""
        return self._enable_ragas and is_ragas_available()

    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_documents: List[Dict[str, Any]],
        ground_truth_ids: Optional[List[str]] = None,
        ground_truth_answer: Optional[str] = None,
        k: Optional[int] = None
    ) -> EvaluationResult:
        """
        Evaluate a RAG query response.

        This is the main entry point for evaluation. It computes all
        configured metrics and returns a comprehensive result.

        THE EVALUATION DOES NOT:
        ------------------------
        - Modify the answer
        - Affect the RAG response
        - Block the query (runs after response is ready)

        Args:
            question: The user's original question
            answer: The generated answer from the RAG system
            retrieved_documents: List of retrieved documents. Each should have:
                - "id": Document identifier (required for retrieval metrics)
                - "content" or "text": Document text (required for generation metrics)
                - "score": Optional similarity score
                - "metadata": Optional metadata
            ground_truth_ids: Optional list of document IDs that are relevant
                             to this question. Required for precision/recall metrics.
            ground_truth_answer: Optional ground truth answer for RAGAS context_recall
            k: K value for @K metrics. If None, uses default_k from config.

        Returns:
            EvaluationResult containing all computed metrics.

        Example:
            >>> result = evaluator.evaluate(
            ...     question="What is RAG?",
            ...     answer="RAG combines retrieval with generation...",
            ...     retrieved_documents=[
            ...         {"id": "doc1", "content": "RAG is...", "score": 0.95}
            ...     ],
            ...     ground_truth_ids=["doc1", "doc3"]
            ... )
        """
        # Early return if evaluation is disabled
        if not self._enabled:
            return EvaluationResult(
                timestamp=datetime.utcnow().isoformat(),
                question=question,
                evaluation_config={"enabled": False}
            )

        # Use default K if not specified
        effective_k = k if k is not None else self._default_k

        # Track evaluation config
        eval_config = {
            "enabled": True,
            "k": effective_k,
            "ragas_enabled": self._enable_ragas,
            "has_ground_truth": ground_truth_ids is not None
        }

        # Initialize result containers
        retrieval_metrics_result = None
        retrieval_stats_result = None
        generation_metrics_result = None
        ragas_metrics_result = None

        # =================================================================
        # RETRIEVAL METRICS
        # =================================================================
        if retrieved_documents:
            # Extract document IDs
            retrieved_ids = [
                doc.get("id", str(i))
                for i, doc in enumerate(retrieved_documents)
            ]

            # If ground truth is provided, compute precision/recall/MRR
            if ground_truth_ids:
                metrics = compute_retrieval_metrics(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=ground_truth_ids,
                    k=effective_k
                )
                retrieval_metrics_result = metrics.to_dict()

            # Always compute basic stats (doesn't need ground truth)
            stats = compute_basic_retrieval_stats(
                retrieved_documents=retrieved_documents,
                score_key="score"
            )
            retrieval_stats_result = stats.to_dict()

        # =================================================================
        # GENERATION METRICS
        # =================================================================
        if answer and retrieved_documents:
            gen_metrics = compute_generation_metrics(
                answer=answer,
                context_documents=retrieved_documents,
                question=question
            )
            generation_metrics_result = gen_metrics.to_dict()

        # =================================================================
        # RAGAS METRICS (OPTIONAL)
        # =================================================================
        if self._enable_ragas and is_ragas_available():
            ragas_result = compute_ragas_metrics(
                question=question,
                answer=answer,
                context_documents=retrieved_documents,
                ground_truth=ground_truth_answer
            )
            ragas_metrics_result = ragas_result.to_dict()

        # =================================================================
        # BUILD RESULT
        # =================================================================
        result = EvaluationResult(
            timestamp=datetime.utcnow().isoformat(),
            question=question,
            retrieval_metrics=retrieval_metrics_result,
            retrieval_stats=retrieval_stats_result,
            generation_metrics=generation_metrics_result,
            ragas_metrics=ragas_metrics_result,
            evaluation_config=eval_config
        )

        # Log results if enabled
        if self._log_results:
            self._log_evaluation(result)

        # Store history if enabled
        if self._store_history:
            self._history.append(result)

        return result

    def _log_evaluation(self, result: EvaluationResult) -> None:
        """
        Log evaluation results.

        Args:
            result: The evaluation result to log
        """
        summary = result.get_summary()

        logger.info("=" * 50)
        logger.info("RAG EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Question: {result.question[:50]}...")

        if result.retrieval_metrics:
            logger.info("Retrieval Metrics:")
            logger.info(f"  Precision@K: {result.retrieval_metrics.get('precision_at_k', 0):.3f}")
            logger.info(f"  Recall@K: {result.retrieval_metrics.get('recall_at_k', 0):.3f}")
            logger.info(f"  MRR: {result.retrieval_metrics.get('mrr', 0):.3f}")
            logger.info(f"  Hit Rate: {result.retrieval_metrics.get('hit_rate', 0):.3f}")

        if result.retrieval_stats:
            logger.info("Retrieval Stats:")
            logger.info(f"  Avg Score: {result.retrieval_stats.get('avg_score', 0):.3f}")
            logger.info(f"  Docs Retrieved: {result.retrieval_stats.get('num_retrieved', 0)}")

        if result.generation_metrics:
            logger.info("Generation Metrics:")
            logger.info(f"  Faithfulness: {result.generation_metrics.get('faithfulness_score', 0):.3f}")
            logger.info(f"  Context Coverage: {result.generation_metrics.get('context_coverage', 0):.3f}")
            logger.info(f"  Hallucination Risk: {result.generation_metrics.get('hallucination_risk', 0):.3f}")

        if result.ragas_metrics and result.ragas_metrics.get("evaluation_successful"):
            logger.info("RAGAS Metrics:")
            if result.ragas_metrics.get("faithfulness") is not None:
                logger.info(f"  Faithfulness: {result.ragas_metrics['faithfulness']:.3f}")
            if result.ragas_metrics.get("answer_relevancy") is not None:
                logger.info(f"  Answer Relevancy: {result.ragas_metrics['answer_relevancy']:.3f}")
            if result.ragas_metrics.get("overall_score") is not None:
                logger.info(f"  Overall: {result.ragas_metrics['overall_score']:.3f}")

        logger.info("=" * 50)

    def get_history(self) -> List[EvaluationResult]:
        """
        Get evaluation history.

        Returns:
            List of past EvaluationResult objects.
            Empty if EVALUATION_STORE_HISTORY is false.
        """
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self._history.clear()
        logger.info("Evaluation history cleared")

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across evaluation history.

        Returns:
            Dictionary with average metrics across all stored evaluations.
        """
        if not self._history:
            return {"num_evaluations": 0}

        num_evals = len(self._history)

        # Aggregate retrieval metrics
        retrieval_precision = []
        retrieval_recall = []
        retrieval_mrr = []
        retrieval_hit_rate = []
        avg_scores = []

        # Aggregate generation metrics
        faithfulness_scores = []
        context_coverage_scores = []
        hallucination_risks = []

        for result in self._history:
            if result.retrieval_metrics:
                retrieval_precision.append(result.retrieval_metrics.get("precision_at_k", 0))
                retrieval_recall.append(result.retrieval_metrics.get("recall_at_k", 0))
                retrieval_mrr.append(result.retrieval_metrics.get("mrr", 0))
                retrieval_hit_rate.append(result.retrieval_metrics.get("hit_rate", 0))

            if result.retrieval_stats:
                avg_scores.append(result.retrieval_stats.get("avg_score", 0))

            if result.generation_metrics:
                faithfulness_scores.append(result.generation_metrics.get("faithfulness_score", 0))
                context_coverage_scores.append(result.generation_metrics.get("context_coverage", 0))
                hallucination_risks.append(result.generation_metrics.get("hallucination_risk", 0))

        def safe_avg(lst: List[float]) -> Optional[float]:
            return sum(lst) / len(lst) if lst else None

        return {
            "num_evaluations": num_evals,
            "retrieval": {
                "avg_precision_at_k": safe_avg(retrieval_precision),
                "avg_recall_at_k": safe_avg(retrieval_recall),
                "avg_mrr": safe_avg(retrieval_mrr),
                "avg_hit_rate": safe_avg(retrieval_hit_rate),
                "avg_similarity_score": safe_avg(avg_scores)
            },
            "generation": {
                "avg_faithfulness": safe_avg(faithfulness_scores),
                "avg_context_coverage": safe_avg(context_coverage_scores),
                "avg_hallucination_risk": safe_avg(hallucination_risks)
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get current evaluator status and configuration.

        Returns:
            Dictionary with evaluator status information.
        """
        return {
            "enabled": self._enabled,
            "ragas_enabled": self._enable_ragas,
            "ragas_available": is_ragas_available(),
            "default_k": self._default_k,
            "log_results": self._log_results,
            "store_history": self._store_history,
            "history_size": len(self._history)
        }


# =============================================================================
# SINGLETON INSTANCE (OPTIONAL)
# =============================================================================
# For convenience, we provide a module-level evaluator instance that can be
# imported and used directly. This is optional - you can create your own
# RAGEvaluator instances if you need different configurations.

_default_evaluator: Optional[RAGEvaluator] = None


def get_evaluator() -> RAGEvaluator:
    """
    Get the shared evaluator instance.

    This function returns a singleton RAGEvaluator configured from settings.
    Use this for consistent evaluation across the application.

    Returns:
        The shared RAGEvaluator instance.

    Example:
        >>> from src.evaluation.evaluator import get_evaluator
        >>> evaluator = get_evaluator()
        >>> result = evaluator.evaluate(question, answer, docs)
    """
    global _default_evaluator

    if _default_evaluator is None:
        _default_evaluator = RAGEvaluator()

    return _default_evaluator


def reset_evaluator() -> None:
    """
    Reset the shared evaluator instance.

    This is primarily for testing purposes. After calling this,
    get_evaluator() will create a fresh instance.
    """
    global _default_evaluator
    _default_evaluator = None
    logger.info("Shared evaluator reset")
