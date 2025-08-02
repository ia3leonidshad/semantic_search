"""Retrieval evaluation metrics implementation.

This module provides functions to evaluate retrieval systems using standard metrics:
- nDCG@K (Normalized Discounted Cumulative Gain)
- Precision@K
- Recall@K  
- MRR (Mean Reciprocal Rank)

Ground truth format:
- Single query: Dict[item_id, relevance_score] where relevance_score is 0, 1, or 2
- Dataset: Dict[query_id, Dict[item_id, relevance_score]]

Relevance scores:
- 2 = Highly relevant (exact match)
- 1 = Partially relevant (shares key attributes)
- 0 = Irrelevant
"""

import math
from typing import List, Dict, Any
import logging

from ..retrievers.base import RetrievalResult

logger = logging.getLogger(__name__)


def ndcg_at_k(results: List[RetrievalResult], ground_truth: Dict[str, int], k: int) -> float:
    """Calculate nDCG@K for a single query.
    
    nDCG@K measures the quality of ranking by comparing against an ideal ranking.
    It weights higher positions more heavily and handles graded relevance labels.
    
    Args:
        results: List of RetrievalResult objects from retriever
        ground_truth: Dict mapping item_id to relevance score (0, 1, 2)
        k: Number of top results to consider
        
    Returns:
        nDCG@K score between 0 and 1 (1 = perfect ranking)
    """
    if not results or k <= 0:
        return 0.0
    
    # Limit to top k results
    top_k_results = results[:k]
    
    # Calculate DCG@K
    dcg = 0.0
    for i, result in enumerate(top_k_results):
        relevance = ground_truth.get(result.item_id, 0)
        if relevance > 0:
            # DCG formula: rel_i / log2(i + 2) where i is 0-indexed
            dcg += relevance / math.log2(i + 2)
    
    # Calculate IDCG@K (ideal DCG)
    # Sort relevance scores in descending order
    all_relevances = list(ground_truth.values())
    ideal_relevances = sorted([r for r in all_relevances if r > 0], reverse=True)
    
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances[:k]):
        idcg += relevance / math.log2(i + 2)
    
    # Return nDCG@K
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def precision_at_k(results: List[RetrievalResult], ground_truth: Dict[str, int], k: int, threshold: int = 1) -> float:
    """Calculate Precision@K for a single query.
    
    Precision@K measures what fraction of the top K results are relevant.
    
    Args:
        results: List of RetrievalResult objects from retriever
        ground_truth: Dict mapping item_id to relevance score (0, 1, 2)
        k: Number of top results to consider
        
    Returns:
        Precision@K score between 0 and 1
    """
    if not results or k <= 0:
        return 0.0
    
    # Limit to top k results
    top_k_results = results[:k]
    
    # Count relevant items (relevance >= 1)
    relevant_count = 0
    for result in top_k_results:
        relevance = ground_truth.get(result.item_id, 0)
        if relevance >= threshold:
            relevant_count += 1
    
    return relevant_count / len(top_k_results)


def recall_at_k(results: List[RetrievalResult], ground_truth: Dict[str, int], k: int, threshold: int = 1) -> float:
    """Calculate Recall@K for a single query.
    
    Recall@K measures what fraction of all relevant items are found in the top K results.
    
    Args:
        results: List of RetrievalResult objects from retriever
        ground_truth: Dict mapping item_id to relevance score (0, 1, 2)
        k: Number of top results to consider
        
    Returns:
        Recall@K score between 0 and 1
    """
    if not results or k <= 0:
        return 0.0
    
    # Count total relevant items in ground truth
    total_relevant = sum(1 for relevance in ground_truth.values() if relevance >= threshold)
    
    if total_relevant == 0:
        return 0.0
    
    # Limit to top k results
    top_k_results = results[:k]
    
    # Count relevant items found in top k
    found_relevant = 0
    for result in top_k_results:
        relevance = ground_truth.get(result.item_id, 0)
        if relevance >= threshold:
            found_relevant += 1
    
    return found_relevant / total_relevant


def mean_reciprocal_rank(results: List[RetrievalResult], ground_truth: Dict[str, int], threshold: int = 1) -> float:
    """Calculate MRR (Mean Reciprocal Rank) for a single query.
    
    MRR measures how quickly the first relevant item appears in the ranking.
    For a single query, this is just the reciprocal rank of the first relevant item.
    
    Args:
        results: List of RetrievalResult objects from retriever
        ground_truth: Dict mapping item_id to relevance score (0, 1, 2)
        
    Returns:
        Reciprocal rank (1/rank) of first relevant item, or 0 if no relevant items found
    """
    if not results:
        return 0.0
    
    # Find rank of first relevant item (1-indexed)
    for i, result in enumerate(results):
        relevance = ground_truth.get(result.item_id, 0)
        if relevance >= threshold:
            return 1.0 / (i + 1)
    
    # No relevant items found
    return 0.0


def evaluate_dataset(
    query_results: Dict[str, List[RetrievalResult]], 
    ground_truth: Dict[str, Dict[str, int]], 
    k_values: List[int] = [5, 10, 20],
    threshold: int = 1,
) -> Dict[str, float]:
    """Evaluate retrieval performance across an entire dataset.
    
    Aggregates metrics across all queries by taking the mean of individual query scores.
    
    Args:
        query_results: Dict mapping query_id to list of RetrievalResult objects
        ground_truth: Dict mapping query_id to dict of item_id -> relevance_score
        k_values: List of K values to evaluate for nDCG, Precision, and Recall
        
    Returns:
        Dict containing aggregated metrics:
        {
            'ndcg@5': float, 'ndcg@10': float, 'ndcg@20': float,
            'precision@5': float, 'precision@10': float, 'precision@20': float,
            'recall@5': float, 'recall@10': float, 'recall@20': float,
            'mrr': float,
            'num_queries': int
        }
    """
    if not query_results or not ground_truth:
        logger.warning("Empty query results or ground truth provided")
        return {}
    
    # Find common queries between results and ground truth
    common_queries = set(query_results.keys()) & set(ground_truth.keys())
    
    if not common_queries:
        logger.warning("No common queries found between results and ground truth")
        return {}
    
    logger.info(f"Evaluating {len(common_queries)} queries")
    
    # Initialize metric accumulators
    metrics = {}
    for k in k_values:
        metrics[f'ndcg@{k}'] = []
        metrics[f'precision@{k}'] = []
        metrics[f'recall@{k}'] = []
    metrics['mrr'] = []
    
    # Calculate metrics for each query
    for query_id in common_queries:
        results = query_results[query_id]
        gt = ground_truth[query_id]
        
        # Skip queries with no ground truth relevance judgments
        if not gt or all(rel == 0 for rel in gt.values()):
            logger.debug(f"Skipping query {query_id}: no relevant items in ground truth")
            continue
        
        # Calculate metrics for each k value
        for k in k_values:
            metrics[f'ndcg@{k}'].append(ndcg_at_k(results, gt, k))
            metrics[f'precision@{k}'].append(precision_at_k(results, gt, k, threshold))
            metrics[f'recall@{k}'].append(recall_at_k(results, gt, k, threshold))
        
        # Calculate MRR (no k parameter)
        metrics['mrr'].append(mean_reciprocal_rank(results, gt, threshold))
    
    # Aggregate by taking mean across queries
    aggregated_metrics = {}
    for metric_name, values in metrics.items():
        if values:  # Only aggregate if we have values
            aggregated_metrics[metric_name] = sum(values) / len(values)
        else:
            aggregated_metrics[metric_name] = 0.0
    
    # Add metadata
    aggregated_metrics['num_queries'] = len([q for q in common_queries 
                                           if ground_truth[q] and any(rel > 0 for rel in ground_truth[q].values())])
    
    logger.info(f"Evaluation completed for {aggregated_metrics['num_queries']} queries")
    
    return aggregated_metrics


def print_evaluation_results(metrics: Dict[str, float], title: str = "Evaluation Results") -> None:
    """Pretty print evaluation results.
    
    Args:
        metrics: Dict of metric names to values from evaluate_dataset()
        title: Title for the results display
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    if 'num_queries' in metrics:
        print(f"Queries evaluated: {metrics['num_queries']}")
        print()
    
    # Group metrics by type
    ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith('ndcg@')}
    precision_metrics = {k: v for k, v in metrics.items() if k.startswith('precision@')}
    recall_metrics = {k: v for k, v in metrics.items() if k.startswith('recall@')}
    
    if ndcg_metrics:
        print("nDCG (Normalized Discounted Cumulative Gain):")
        for metric, value in sorted(ndcg_metrics.items()):
            print(f"  {metric}: {value:.4f}")
        print()
    
    if precision_metrics:
        print("Precision:")
        for metric, value in sorted(precision_metrics.items()):
            print(f"  {metric}: {value:.4f}")
        print()
    
    if recall_metrics:
        print("Recall:")
        for metric, value in sorted(recall_metrics.items()):
            print(f"  {metric}: {value:.4f}")
        print()
    
    if 'mrr' in metrics:
        print(f"MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
