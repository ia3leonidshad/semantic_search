"""Evaluation module for retrieval systems."""

from .metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    evaluate_dataset
)

__all__ = [
    "ndcg_at_k",
    "precision_at_k", 
    "recall_at_k",
    "mean_reciprocal_rank",
    "evaluate_dataset"
]
