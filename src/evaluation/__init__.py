"""
RAG Evaluation Module
=====================

This module provides comprehensive evaluation capabilities for RAG systems.

COMPONENTS:
-----------
1. RAGEvaluator: Main orchestrator for all evaluation metrics
2. Retrieval Metrics: Precision@K, Recall@K, MRR, Hit Rate
3. Generation Metrics: Faithfulness, Context Coverage, Hallucination Risk
4. RAGAS Adapter: Optional integration with RAGAS library

USAGE:
------
The simplest way to use evaluation is through the RAGEvaluator:

    from src.evaluation import RAGEvaluator, get_evaluator

    # Use the shared evaluator (recommended)
    evaluator = get_evaluator()

    # Or create a custom instance
    evaluator = RAGEvaluator(enable_evaluation=True)

    # Evaluate a RAG response
    result = evaluator.evaluate(
        question="What is RAG?",
        answer="RAG combines retrieval with generation...",
        retrieved_documents=[{"id": "doc1", "content": "..."}]
    )

    # Access results
    print(result.get_summary())
    print(result.to_dict())

CONFIGURATION:
--------------
All settings are controlled via environment variables:

- ENABLE_EVALUATION=false     Master switch (default: disabled)
- EVALUATION_DEFAULT_K=5      K value for @K metrics
- ENABLE_RAGAS=false          Enable RAGAS metrics (requires ragas package)
- EVALUATION_LOG_RESULTS=true Whether to log results
- EVALUATION_STORE_HISTORY=false Whether to store history

For direct metric computation, use the metrics modules:

    from src.evaluation.metrics.retrieval_metrics import compute_retrieval_metrics
    from src.evaluation.metrics.generation_metrics import compute_generation_metrics
"""

# Main evaluator class and factory
from src.evaluation.evaluator import (
    RAGEvaluator,
    EvaluationResult,
    get_evaluator,
    reset_evaluator
)

# Retrieval metrics
from src.evaluation.metrics.retrieval_metrics import (
    compute_retrieval_metrics,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_hit_rate,
    compute_basic_retrieval_stats,
    compute_reranking_impact,
    RetrievalMetricsResult,
    BasicRetrievalStats
)

# Generation metrics
from src.evaluation.metrics.generation_metrics import (
    compute_generation_metrics,
    compute_faithfulness_score,
    compute_context_coverage,
    compute_hallucination_risk,
    GenerationMetricsResult
)

# RAGAS adapter
from src.evaluation.metrics.ragas_adapter import (
    compute_ragas_metrics,
    is_ragas_available,
    get_ragas_status,
    RAGASMetricsResult
)

# Public API
__all__ = [
    # Main evaluator
    "RAGEvaluator",
    "EvaluationResult",
    "get_evaluator",
    "reset_evaluator",

    # Retrieval metrics
    "compute_retrieval_metrics",
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_mrr",
    "compute_hit_rate",
    "compute_basic_retrieval_stats",
    "compute_reranking_impact",
    "RetrievalMetricsResult",
    "BasicRetrievalStats",

    # Generation metrics
    "compute_generation_metrics",
    "compute_faithfulness_score",
    "compute_context_coverage",
    "compute_hallucination_risk",
    "GenerationMetricsResult",

    # RAGAS adapter
    "compute_ragas_metrics",
    "is_ragas_available",
    "get_ragas_status",
    "RAGASMetricsResult"
]
