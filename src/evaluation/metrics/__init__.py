"""
Evaluation Metrics Package
==========================

This package contains all evaluation metrics for the RAG system.

Modules:
--------
- retrieval_metrics: Precision@K, Recall@K, MRR, Hit Rate
- generation_metrics: Faithfulness, Context Coverage, Hallucination Risk
- ragas_adapter: Optional RAGAS integration for LLM-based metrics
"""

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

from src.evaluation.metrics.generation_metrics import (
    compute_generation_metrics,
    compute_faithfulness_score,
    compute_context_coverage,
    compute_hallucination_risk,
    GenerationMetricsResult
)

from src.evaluation.metrics.ragas_adapter import (
    compute_ragas_metrics,
    is_ragas_available,
    get_ragas_status,
    RAGASMetricsResult
)

__all__ = [
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
