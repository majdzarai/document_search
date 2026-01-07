"""
RAGAS Adapter Module
====================

WHAT IS RAGAS?
--------------
RAGAS (Retrieval Augmented Generation Assessment) is a framework for
evaluating RAG systems. It provides LLM-based metrics that can assess
quality dimensions that heuristic approaches cannot capture.

RAGAS METRICS:
--------------
1. Faithfulness: Is the answer factually grounded in the context?
2. Answer Relevancy: Does the answer address the question?
3. Context Precision: Are retrieved documents relevant to the question?
4. Context Recall: Did we retrieve all relevant information?

WHY A SEPARATE ADAPTER?
-----------------------
1. OPTIONAL DEPENDENCY: RAGAS requires additional packages (ragas, langchain)
2. COST: RAGAS metrics require LLM calls ($$)
3. LATENCY: RAGAS is slow compared to heuristic metrics
4. FLEXIBILITY: Adapter pattern allows easy swapping

USAGE:
------
RAGAS is OPTIONAL and must be explicitly enabled:
1. Install ragas: pip install ragas
2. Set ENABLE_RAGAS=true in environment
3. Ensure LLM API key is configured

When RAGAS is not installed or disabled, this module provides graceful
fallbacks that return None for all metrics.

IMPORTANT - PRODUCTION CONSIDERATIONS:
--------------------------------------
RAGAS metrics are NOT recommended for production query evaluation because:
- They add significant latency (multiple LLM calls)
- They incur cost for every query
- They may fail if LLM service is unavailable

Use RAGAS for:
- Offline batch evaluation
- A/B testing
- Quality audits
- Development and tuning

Use heuristic metrics (generation_metrics.py) for:
- Production monitoring
- Real-time quality signals
- Every-query evaluation
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Track RAGAS availability
_ragas_available: Optional[bool] = None


def _check_ragas_available() -> bool:
    """
    Check if RAGAS library is installed and available.

    Returns:
        True if RAGAS can be imported, False otherwise.
    """
    global _ragas_available

    if _ragas_available is not None:
        return _ragas_available

    try:
        import ragas  # noqa: F401
        _ragas_available = True
        logger.info("RAGAS library is available")
    except ImportError:
        _ragas_available = False
        logger.info("RAGAS library not installed. Install with: pip install ragas")

    return _ragas_available


def is_ragas_available() -> bool:
    """
    Public function to check RAGAS availability.

    Returns:
        True if RAGAS is installed, False otherwise.

    Example:
        >>> if is_ragas_available():
        ...     metrics = compute_ragas_metrics(...)
        ... else:
        ...     print("RAGAS not available, using heuristics")
    """
    return _check_ragas_available()


@dataclass
class RAGASMetricsResult:
    """
    Container for RAGAS evaluation results.

    All scores are between 0.0 and 1.0, where higher is better.

    Attributes:
        faithfulness: How factually consistent is the answer with context? (0-1)
        answer_relevancy: How well does the answer address the question? (0-1)
        context_precision: How relevant are the retrieved documents? (0-1)
        context_recall: Did we retrieve all relevant information? (0-1)
        overall_score: Weighted average of all metrics (0-1)
        evaluation_successful: Whether RAGAS evaluation completed successfully
        error_message: Error message if evaluation failed
    """
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    overall_score: Optional[float] = None
    evaluation_successful: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "faithfulness": round(self.faithfulness, 4) if self.faithfulness is not None else None,
            "answer_relevancy": round(self.answer_relevancy, 4) if self.answer_relevancy is not None else None,
            "context_precision": round(self.context_precision, 4) if self.context_precision is not None else None,
            "context_recall": round(self.context_recall, 4) if self.context_recall is not None else None,
            "overall_score": round(self.overall_score, 4) if self.overall_score is not None else None,
            "evaluation_successful": self.evaluation_successful,
        }
        if self.error_message:
            result["error_message"] = self.error_message
        return result


def _create_ragas_dataset(
    question: str,
    answer: str,
    context_documents: List[Dict[str, Any]],
    ground_truth: Optional[str] = None
) -> Optional[Any]:
    """
    Create a RAGAS-compatible dataset from RAG pipeline output.

    Args:
        question: The user's question
        answer: The generated answer
        context_documents: List of retrieved documents
        ground_truth: Optional ground truth answer for recall metrics

    Returns:
        RAGAS Dataset object, or None if creation fails
    """
    try:
        from datasets import Dataset

        # Extract context texts
        contexts = [
            doc.get("content", "") or doc.get("text", "")
            for doc in context_documents
        ]

        # Build dataset
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        # Add ground truth if provided (required for context_recall)
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        return Dataset.from_dict(data)

    except Exception as e:
        logger.error(f"Failed to create RAGAS dataset: {e}")
        return None


def compute_ragas_metrics(
    question: str,
    answer: str,
    context_documents: List[Dict[str, Any]],
    ground_truth: Optional[str] = None,
    metrics: Optional[List[str]] = None
) -> RAGASMetricsResult:
    """
    Compute RAGAS metrics for a single RAG query.

    This function evaluates the quality of a RAG response using the RAGAS
    library. It requires RAGAS to be installed and an LLM API configured.

    AVAILABLE METRICS:
    ------------------
    - "faithfulness": Is the answer grounded in context?
    - "answer_relevancy": Does the answer address the question?
    - "context_precision": Are retrieved docs relevant?
    - "context_recall": Did we get all relevant info? (requires ground_truth)

    Args:
        question: The user's original question
        answer: The generated answer from the RAG system
        context_documents: List of retrieved documents with "content" key
        ground_truth: Optional ground truth answer (needed for context_recall)
        metrics: List of metrics to compute. If None, computes all available.

    Returns:
        RAGASMetricsResult with computed metrics.
        If RAGAS is not available or evaluation fails, returns result with
        evaluation_successful=False and appropriate error_message.

    Example:
        >>> result = compute_ragas_metrics(
        ...     question="What is RAG?",
        ...     answer="RAG combines retrieval with generation.",
        ...     context_documents=[{"content": "RAG is retrieval-augmented generation."}]
        ... )
        >>> if result.evaluation_successful:
        ...     print(f"Faithfulness: {result.faithfulness:.2f}")
    """
    # Check RAGAS availability
    if not _check_ragas_available():
        return RAGASMetricsResult(
            evaluation_successful=False,
            error_message="RAGAS library not installed. Install with: pip install ragas"
        )

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )

        # Create dataset
        dataset = _create_ragas_dataset(
            question=question,
            answer=answer,
            context_documents=context_documents,
            ground_truth=ground_truth
        )

        if dataset is None:
            return RAGASMetricsResult(
                evaluation_successful=False,
                error_message="Failed to create RAGAS dataset"
            )

        # Select metrics to evaluate
        available_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
        }

        # context_recall requires ground_truth
        if ground_truth:
            available_metrics["context_recall"] = context_recall

        if metrics:
            selected_metrics = [
                available_metrics[m] for m in metrics
                if m in available_metrics
            ]
        else:
            selected_metrics = list(available_metrics.values())

        if not selected_metrics:
            return RAGASMetricsResult(
                evaluation_successful=False,
                error_message="No valid metrics selected for evaluation"
            )

        # Run RAGAS evaluation
        logger.info(f"Running RAGAS evaluation with {len(selected_metrics)} metrics")
        result = evaluate(dataset, metrics=selected_metrics)

        # Extract scores
        faithfulness_score = result.get("faithfulness")
        relevancy_score = result.get("answer_relevancy")
        precision_score = result.get("context_precision")
        recall_score = result.get("context_recall") if ground_truth else None

        # Calculate overall score (weighted average of available metrics)
        scores = [s for s in [
            faithfulness_score, relevancy_score, precision_score, recall_score
        ] if s is not None]

        overall = sum(scores) / len(scores) if scores else None

        return RAGASMetricsResult(
            faithfulness=faithfulness_score,
            answer_relevancy=relevancy_score,
            context_precision=precision_score,
            context_recall=recall_score,
            overall_score=overall,
            evaluation_successful=True
        )

    except ImportError as e:
        error_msg = f"RAGAS import error: {e}. Some dependencies may be missing."
        logger.error(error_msg)
        return RAGASMetricsResult(
            evaluation_successful=False,
            error_message=error_msg
        )

    except Exception as e:
        error_msg = f"RAGAS evaluation failed: {str(e)}"
        logger.error(error_msg)
        return RAGASMetricsResult(
            evaluation_successful=False,
            error_message=error_msg
        )


def compute_ragas_batch(
    evaluations: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None
) -> List[RAGASMetricsResult]:
    """
    Compute RAGAS metrics for multiple RAG queries in batch.

    This is more efficient than calling compute_ragas_metrics multiple times
    because it batches the LLM calls.

    Args:
        evaluations: List of evaluation inputs, each containing:
            - "question": The user's question
            - "answer": The generated answer
            - "context_documents": List of retrieved documents
            - "ground_truth": Optional ground truth answer
        metrics: List of metrics to compute. If None, computes all available.

    Returns:
        List of RAGASMetricsResult, one per input evaluation.

    Example:
        >>> evaluations = [
        ...     {"question": "What is RAG?", "answer": "...", "context_documents": [...]},
        ...     {"question": "How do embeddings work?", "answer": "...", "context_documents": [...]}
        ... ]
        >>> results = compute_ragas_batch(evaluations)
        >>> for r in results:
        ...     print(f"Faithfulness: {r.faithfulness}")
    """
    if not _check_ragas_available():
        return [
            RAGASMetricsResult(
                evaluation_successful=False,
                error_message="RAGAS library not installed"
            )
            for _ in evaluations
        ]

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
        from datasets import Dataset

        # Build combined dataset
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []
        has_ground_truth = any(e.get("ground_truth") for e in evaluations)

        for eval_input in evaluations:
            questions.append(eval_input["question"])
            answers.append(eval_input["answer"])

            # Extract contexts
            docs = eval_input.get("context_documents", [])
            contexts = [
                doc.get("content", "") or doc.get("text", "")
                for doc in docs
            ]
            contexts_list.append(contexts)

            if has_ground_truth:
                ground_truths.append(eval_input.get("ground_truth", ""))

        # Build dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
        if has_ground_truth:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        # Select metrics
        available_metrics = [faithfulness, answer_relevancy, context_precision]
        if has_ground_truth:
            available_metrics.append(context_recall)

        # Run batch evaluation
        logger.info(f"Running RAGAS batch evaluation on {len(evaluations)} samples")
        result = evaluate(dataset, metrics=available_metrics)

        # Convert to per-sample results
        results = []
        for i in range(len(evaluations)):
            faith = result["faithfulness"][i] if "faithfulness" in result else None
            rel = result["answer_relevancy"][i] if "answer_relevancy" in result else None
            prec = result["context_precision"][i] if "context_precision" in result else None
            rec = result["context_recall"][i] if "context_recall" in result and has_ground_truth else None

            scores = [s for s in [faith, rel, prec, rec] if s is not None]
            overall = sum(scores) / len(scores) if scores else None

            results.append(RAGASMetricsResult(
                faithfulness=faith,
                answer_relevancy=rel,
                context_precision=prec,
                context_recall=rec,
                overall_score=overall,
                evaluation_successful=True
            ))

        return results

    except Exception as e:
        error_msg = f"RAGAS batch evaluation failed: {str(e)}"
        logger.error(error_msg)
        return [
            RAGASMetricsResult(
                evaluation_successful=False,
                error_message=error_msg
            )
            for _ in evaluations
        ]


def get_ragas_status() -> Dict[str, Any]:
    """
    Get status information about RAGAS integration.

    Returns:
        Dictionary with:
        - available: Whether RAGAS is installed
        - version: RAGAS version if installed
        - supported_metrics: List of supported metric names
    """
    if not _check_ragas_available():
        return {
            "available": False,
            "version": None,
            "supported_metrics": [],
            "message": "RAGAS not installed. Install with: pip install ragas"
        }

    try:
        import ragas
        version = getattr(ragas, "__version__", "unknown")

        return {
            "available": True,
            "version": version,
            "supported_metrics": [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall"
            ],
            "message": "RAGAS is ready for evaluation"
        }

    except Exception as e:
        return {
            "available": False,
            "version": None,
            "supported_metrics": [],
            "message": f"Error checking RAGAS status: {str(e)}"
        }
