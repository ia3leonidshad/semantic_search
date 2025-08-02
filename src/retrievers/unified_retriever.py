"""Unified retriever that combines multiple retrievers with configurable weights."""

from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
from pathlib import Path
import pickle
import logging
from collections import defaultdict

from src.retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


# Shared fusion methods for both UnifiedRetriever and ExtensionRetriever
def _normalize_result_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Normalize scores to [0, 1] range using min-max normalization.
    
    Args:
        results: List of results to normalize
        
    Returns:
        List of results with normalized scores
    """
    if not results:
        return results
    
    scores = [r.score for r in results]
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        # All scores are the same
        for result in results:
            result.score = 1.0
    else:
        # Normalize to [0, 1]
        for result in results:
            result.score = (result.score - min_score) / (max_score - min_score)
    
    return results


def _weighted_sum_fusion_shared(
    all_results: List[List[RetrievalResult]],
    weights: List[float],
    k: int,
    normalize_scores: bool = True,
    fusion_metadata: Optional[Dict[str, Any]] = None
) -> List[RetrievalResult]:
    """Shared weighted sum fusion implementation.
    
    Args:
        all_results: List of result lists from each source
        weights: List of weights corresponding to each source
        k: Number of final results to return
        normalize_scores: Whether to normalize scores before fusion
        fusion_metadata: Additional metadata to include in results
        
    Returns:
        List of fused results
    """
    # Collect all scores by document/item ID
    item_scores = defaultdict(lambda: [0.0] * len(weights))
    
    # Normalize scores if requested
    if normalize_scores:
        all_results = [_normalize_result_scores(results) for results in all_results]
    
    # Collect scores from each source
    for source_idx, results in enumerate(all_results):
        for result in results:
            doc_id = result.item_id
            if doc_id is not None:
                item_scores[doc_id][source_idx] = result.score
    
    # Calculate fused scores
    fused_results = []
    for doc_id, scores in item_scores.items():
        fused_score = sum(weight * score for weight, score in zip(weights, scores))
        
        # Create metadata with individual scores
        metadata = {
            "fusion_method": "weighted_sum",
            "weights": weights,
            "individual_scores": scores,
        }
        if fusion_metadata:
            metadata.update(fusion_metadata)
        
        result = RetrievalResult(
            item_id=doc_id,
            score=fused_score,
            metadata=metadata
        )
        fused_results.append(result)
    
    # Sort by fused score and return top k
    fused_results.sort(key=lambda x: x.score, reverse=True)
    return fused_results[:k]


def _rank_fusion_shared(
    all_results: List[List[RetrievalResult]],
    weights: List[float],
    k: int,
    fusion_metadata: Optional[Dict[str, Any]] = None
) -> List[RetrievalResult]:
    """Shared rank-based fusion implementation.
    
    Args:
        all_results: List of result lists from each source
        weights: List of weights corresponding to each source
        k: Number of final results to return
        fusion_metadata: Additional metadata to include in results
        
    Returns:
        List of fused results
    """
    # Collect ranks by document/item ID
    item_ranks = defaultdict(lambda: [float('inf')] * len(weights))
    
    # Collect ranks from each source
    for source_idx, results in enumerate(all_results):
        for rank, result in enumerate(results):
            doc_id = result.item_id
            if doc_id is not None:
                item_ranks[doc_id][source_idx] = rank + 1  # 1-indexed ranks

    # Calculate fused scores
    fused_results = []
    for doc_id, ranks in item_ranks.items():
        # Convert ranks to scores and apply weights
        rank_scores = [1.0 / rank if rank != float('inf') else 0.0 for rank in ranks]
        fused_score = sum(weight * score for weight, score in zip(weights, rank_scores))

        # Create metadata
        metadata = {
            "fusion_method": "rank_fusion",
            "weights": weights,
            "individual_ranks": ranks,
            "individual_rank_scores": rank_scores,
        }
        if fusion_metadata:
            metadata.update(fusion_metadata)

        result = RetrievalResult(
            item_id=doc_id,
            score=fused_score,
            metadata=metadata
        )
        fused_results.append(result)

    # Sort by fused score and return top k
    fused_results.sort(key=lambda x: x.score, reverse=True)
    return fused_results[:k]


def _reciprocal_rank_fusion_shared(
    all_results: List[List[RetrievalResult]],
    weights: List[float],
    k: int,
    rrf_constant: int = 60,
    fusion_metadata: Optional[Dict[str, Any]] = None
) -> List[RetrievalResult]:
    """Shared Reciprocal Rank Fusion (RRF) implementation.
    
    Args:
        all_results: List of result lists from each source
        weights: List of weights corresponding to each source
        k: Number of final results to return
        rrf_constant: RRF constant (default 60)
        fusion_metadata: Additional metadata to include in results
        
    Returns:
        List of fused results
    """
    # Collect RRF scores by document/item ID
    item_scores = defaultdict(float)

    # Calculate RRF scores for each source
    for source_idx, results in enumerate(all_results):
        weight = weights[source_idx]
        for rank, result in enumerate(results):
            doc_id = result.item_id
            if doc_id is not None:
                rrf_score = weight / (rrf_constant + rank + 1)
                item_scores[doc_id] += rrf_score

    # Create fused results
    fused_results = []
    for doc_id, score in item_scores.items():
        # Create metadata
        metadata = {
            "fusion_method": "rrf",
            "rrf_constant": rrf_constant,
            "weights": weights,
        }
        if fusion_metadata:
            metadata.update(fusion_metadata)

        result = RetrievalResult(
            item_id=doc_id,
            score=score,
            metadata=metadata
        )
        fused_results.append(result)

    # Sort by fused score and return top k
    fused_results.sort(key=lambda x: x.score, reverse=True)
    return fused_results[:k]


def _reranking_merge_fusion_shared(
    all_results: List[List[RetrievalResult]],
    weights: List[float],
    k: int,
    fusion_metadata: Optional[Dict[str, Any]] = None
) -> List[RetrievalResult]:
    """Shared reranking merge fusion implementation.
    
    This method applies weights directly to the original scores and reranks.
    Unlike weighted_sum, it doesn't normalize scores first.
    
    Args:
        all_results: List of result lists from each source
        weights: List of weights corresponding to each source
        k: Number of final results to return
        fusion_metadata: Additional metadata to include in results
        
    Returns:
        List of fused results
    """
    # Collect all scores by document/item ID
    item_scores = defaultdict(lambda: [0.0] * len(weights))
    
    # Collect scores from each source (no normalization)
    for source_idx, results in enumerate(all_results):
        for result in results:
            doc_id = result.item_id
            if doc_id is not None:
                item_scores[doc_id][source_idx] = result.score
    
    # Calculate weighted scores
    fused_results = []
    for doc_id, scores in item_scores.items():
        # Apply weights directly to original scores
        weighted_score = sum(weight * score for weight, score in zip(weights, scores))
        
        # Create metadata with individual scores
        metadata = {
            "fusion_method": "reranking_merge",
            "weights": weights,
            "individual_scores": scores,
        }
        if fusion_metadata:
            metadata.update(fusion_metadata)
        
        result = RetrievalResult(
            item_id=doc_id,
            score=weighted_score,
            metadata=metadata
        )
        fused_results.append(result)
    
    # Sort by weighted score and return top k
    fused_results.sort(key=lambda x: x.score, reverse=True)
    return fused_results[:k]


class UnifiedRetriever(BaseRetriever):
    """Unified retriever that combines multiple pre-constructed retrievers with configurable weights.
    
    This class replaces both HybridRetriever and MultimodalRetriever by accepting a list of 
    pre-constructed retrievers and their corresponding weights. It supports various fusion 
    methods and automatically validates and normalizes weights.
    """
    
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: List[float],
        fusion_method: str = "weighted_sum",
        normalize_scores: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the unified retriever.
        
        Args:
            retrievers: List of pre-constructed retriever instances
            weights: List of weights corresponding to each retriever
            fusion_method: Method for fusing results ("weighted_sum", "rank_fusion", "rrf", "reranking_merge")
            normalize_scores: Whether to normalize scores before fusion
            config: Optional configuration dictionary
            
        Raises:
            ValueError: If retrievers and weights lists have different lengths,
                       if weights are negative, or if all weights are zero
        """
        super().__init__(config)
        
        # Validate inputs
        if len(retrievers) != len(weights):
            raise ValueError(
                f"Number of retrievers ({len(retrievers)}) must match "
                f"number of weights ({len(weights)})"
            )
        
        if len(retrievers) == 0:
            raise ValueError("At least one retriever must be provided")
        
        # Validate and normalize weights
        self.weights = self._validate_and_normalize_weights(weights)
        self.retrievers = retrievers
        
        # Fusion configuration
        self.fusion_method = fusion_method
        self.normalize_scores = normalize_scores
        
        # Validate fusion method
        valid_methods = {"weighted_sum", "rank_fusion", "rrf", "reranking_merge"}
        if fusion_method not in valid_methods:
            raise ValueError(f"fusion_method must be one of {valid_methods}")
        
        # Check if all retrievers are indexed
        self.is_indexed = all(retriever.is_indexed for retriever in self.retrievers)
        
        self.logger.info(
            f"Initialized UnifiedRetriever with {len(retrievers)} retrievers, "
            f"weights={[f'{w:.3f}' for w in self.weights]}, "
            f"fusion_method={fusion_method}"
        )
    
    def _validate_and_normalize_weights(self, weights: List[float]) -> List[float]:
        """Validate that weights are non-negative and normalize them to sum to 1.
        
        Args:
            weights: List of weight values
            
        Returns:
            Normalized weights that sum to 1.0
            
        Raises:
            ValueError: If any weight is negative or all weights are zero
        """
        # Check for negative weights
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")
        
        # Check that at least one weight is positive
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("At least one weight must be positive")
        
        # Normalize weights to sum to 1
        normalized_weights = [w / total_weight for w in weights]
        
        self.logger.info(
            f"Normalized weights from {[f'{w:.3f}' for w in weights]} "
            f"to {[f'{w:.3f}' for w in normalized_weights]}"
        )
        
        return normalized_weights
    
    def create_index(self, *args, **kwargs) -> None:
        """Create indices for all retrievers.
        
        Note: This method is not supported for UnifiedRetriever since it expects
        pre-constructed and pre-indexed retrievers. Use the individual retrievers'
        create_index methods before passing them to UnifiedRetriever.
        
        Raises:
            NotImplementedError: Always, as this retriever doesn't handle indexing
        """
        raise NotImplementedError(
            "UnifiedRetriever does not support create_index. "
            "Please index your retrievers individually before passing them to UnifiedRetriever."
        )
    
    def get_corpus_size(self):
        return self._get_min_corpus_size()

    def search(
        self,
        query: Union[str, Any],
        k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search using all retrievers and fuse results.
        
        Args:
            query: Search query (string for text, other types for multimodal)
            k: Number of results to return
            **kwargs: Additional search parameters passed to all retrievers
            
        Returns:
            List of RetrievalResult objects sorted by fused relevance scores
            
        Raises:
            RuntimeError: If not all retrievers are indexed
        """
        self._check_all_indexed()
        k = self._validate_k(k)
        
        # Get results from all retrievers
        all_results = []
        retrieval_k = min(k * 2, self._get_min_corpus_size())  # Get more for better fusion
        
        for i, retriever in enumerate(self.retrievers):
            try:
                results = retriever.search(query, k=retrieval_k, **kwargs)
                all_results.append(results)
                self.logger.debug(
                    f"Retriever {i} ({retriever.__class__.__name__}) "
                    f"returned {len(results)} results"
                )
            except Exception as e:
                self.logger.warning(
                    f"Retriever {i} ({retriever.__class__.__name__}) "
                    f"search failed: {e}"
                )
                all_results.append([])  # Empty results for failed retriever
        
        # Fuse results
        fused_results = self._fuse_results(all_results, k)
        
        return fused_results
    
    def _fuse_results(
        self,
        all_results: List[List[RetrievalResult]],
        k: int
    ) -> List[RetrievalResult]:
        """Fuse results from multiple retrievers using the configured fusion method.
        
        Args:
            all_results: List of result lists from each retriever
            k: Number of final results to return
            
        Returns:
            List of fused RetrievalResult objects
        """
        # Create metadata for retriever types
        fusion_metadata = {
            "retriever_types": [r.__class__.__name__ for r in self.retrievers]
        }
        
        if self.fusion_method == "weighted_sum":
            return _weighted_sum_fusion_shared(
                all_results, self.weights, k, self.normalize_scores, fusion_metadata
            )
        elif self.fusion_method == "rank_fusion":
            return _rank_fusion_shared(all_results, self.weights, k, fusion_metadata)
        elif self.fusion_method == "rrf":
            rrf_constant = self.config.get("rrf_constant", 60)
            return _reciprocal_rank_fusion_shared(
                all_results, self.weights, k, rrf_constant, fusion_metadata
            )
        elif self.fusion_method == "reranking_merge":
            return _reranking_merge_fusion_shared(
                all_results, self.weights, k, fusion_metadata
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    
    def _check_all_indexed(self) -> None:
        """Check if all retrievers have been indexed.
        
        Raises:
            RuntimeError: If any retriever hasn't been indexed
        """
        unindexed = [
            i for i, retriever in enumerate(self.retrievers) 
            if not retriever.is_indexed
        ]
        
        if unindexed:
            retriever_names = [
                self.retrievers[i].__class__.__name__ for i in unindexed
            ]
            raise RuntimeError(
                f"The following retrievers are not indexed: {retriever_names} "
                f"(indices: {unindexed}). Please index all retrievers before searching."
            )
    
    def _get_min_corpus_size(self) -> int:
        """Get the minimum corpus size across all retrievers.
        
        Returns:
            Minimum corpus size
        """
        if not self.retrievers:
            return 0
        
        return min(retriever.get_corpus_size() for retriever in self.retrievers)
    
    def get_corpus_size(self) -> int:
        """Get the corpus size (minimum across all retrievers).
        
        Returns:
            Corpus size
        """
        return self._get_min_corpus_size()
    
    def add_documents(self, *args, **kwargs) -> None:
        """Add documents to all retrievers.
        
        Note: This method is not supported for UnifiedRetriever since it manages
        pre-constructed retrievers. Use the individual retrievers' add_documents
        methods instead.
        
        Raises:
            NotImplementedError: Always, as this retriever doesn't handle document management
        """
        raise NotImplementedError(
            "UnifiedRetriever does not support add_documents. "
            "Please add documents to your individual retrievers directly."
        )
    
    def save_index(self, path: str) -> None:
        """Save all retriever indices and unified configuration.
        
        Args:
            path: Directory path to save the indices
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save each retriever's index
        for i, retriever in enumerate(self.retrievers):
            retriever_path = path / f"retriever_{i}_{retriever.__class__.__name__}"
            try:
                retriever.save_index(str(retriever_path))
                self.logger.info(f"Saved retriever {i} to {retriever_path}")
            except NotImplementedError:
                self.logger.warning(
                    f"Retriever {i} ({retriever.__class__.__name__}) "
                    f"does not support save_index"
                )
        
        # Save unified configuration
        unified_config = {
            "weights": self.weights,
            "fusion_method": self.fusion_method,
            "normalize_scores": self.normalize_scores,
            "config": self.config,
            "retriever_types": [r.__class__.__name__ for r in self.retrievers],
            "num_retrievers": len(self.retrievers)
        }
        
        with open(path / "unified_config.pkl", "wb") as f:
            pickle.dump(unified_config, f)
        
        self.logger.info(f"Saved unified retriever configuration to {path}")
    
    def load_index(self, path: str) -> None:
        """Load all retriever indices and unified configuration.
        
        Args:
            path: Directory path to load the indices from
            
        Note:
            This method can only load the configuration. The individual retrievers
            must be reconstructed and their indices loaded separately.
        """
        path = Path(path)
        
        if not (path / "unified_config.pkl").exists():
            raise FileNotFoundError(f"Unified config not found at {path}")
        
        # Load unified configuration
        with open(path / "unified_config.pkl", "rb") as f:
            unified_config = pickle.load(f)
        
        self.weights = unified_config["weights"]
        self.fusion_method = unified_config["fusion_method"]
        self.normalize_scores = unified_config["normalize_scores"]
        
        if "config" in unified_config:
            self.config.update(unified_config["config"])
        
        self.logger.info(f"Loaded unified retriever configuration from {path}")
        self.logger.warning(
            "Individual retrievers must be reconstructed and loaded separately. "
            "UnifiedRetriever.load_index only loads the fusion configuration."
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the unified retriever.
        
        Returns:
            Dictionary containing retriever statistics
        """
        stats = super().get_stats()
        stats.update({
            "num_retrievers": len(self.retrievers),
            "weights": self.weights,
            "fusion_method": self.fusion_method,
            "normalize_scores": self.normalize_scores,
            "retriever_stats": [
                {
                    "type": retriever.__class__.__name__,
                    "weight": weight,
                    "is_indexed": retriever.is_indexed,
                    "corpus_size": retriever.get_corpus_size() if retriever.is_indexed else 0
                }
                for retriever, weight in zip(self.retrievers, self.weights)
            ]
        })
        
        return stats


class ExtensionRetriever(BaseRetriever):
    """Extension retriever that uses query extension with a single retriever.
    
    This retriever takes a single retriever and a query extension function. It extends
    the input query into multiple queries, searches with each extended query using the
    same retriever, and then fuses the results using configurable fusion methods.
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        query_extension: Callable[[str], List[str]],
        weights: Optional[List[float]] = None,
        fusion_method: str = "weighted_sum",
        normalize_scores: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the extension retriever.
        
        Args:
            retriever: Pre-constructed retriever instance
            query_extension: Callable that takes a query and returns list of extended queries
            weights: List of weights corresponding to each extended query
            fusion_method: Method for fusing results ("weighted_sum", "rank_fusion", "rrf", "reranking_merge")
            normalize_scores: Whether to normalize scores before fusion
            config: Optional configuration dictionary
            
        Raises:
            ValueError: If weights are negative, all weights are zero, or fusion method is invalid
        """
        super().__init__(config)
        
        if not callable(query_extension):
            raise ValueError("query_extension must be callable")
        
        # Validate and normalize weights
        if weights:
            self.weights = self._validate_and_normalize_weights(weights)
        else:
            self.weights = None
        self.retriever = retriever
        self.query_extension = query_extension
        
        # Fusion configuration
        self.fusion_method = fusion_method
        self.normalize_scores = normalize_scores
        
        # Validate fusion method
        valid_methods = {"weighted_sum", "rank_fusion", "rrf", "reranking_merge"}
        if fusion_method not in valid_methods:
            raise ValueError(f"fusion_method must be one of {valid_methods}")
        
        # Check if retriever is indexed
        self.is_indexed = retriever.is_indexed
        
        self.logger.info(
            f"Initialized ExtensionRetriever with {len(weights) if weights else 'unknown'} query extensions, "
            f"weights={[f'{w:.3f}' for w in self.weights or []]}, "
            f"fusion_method={fusion_method}, "
            f"retriever_type={retriever.__class__.__name__}"
        )
    
    def _validate_and_normalize_weights(self, weights: List[float]) -> List[float]:
        """Validate that weights are non-negative and normalize them to sum to 1.
        
        Args:
            weights: List of weight values
            
        Returns:
            Normalized weights that sum to 1.0
            
        Raises:
            ValueError: If any weight is negative or all weights are zero
        """
        if len(weights) == 0:
            raise ValueError("At least one weight must be provided")
        
        # Check for negative weights
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")
        
        # Check that at least one weight is positive
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("At least one weight must be positive")
        
        # Normalize weights to sum to 1
        normalized_weights = [w / total_weight for w in weights]
        
        self.logger.info(
            f"Normalized weights from {[f'{w:.3f}' for w in weights]} "
            f"to {[f'{w:.3f}' for w in normalized_weights]}"
        )
        
        return normalized_weights
    
    def create_index(self, documents: List[str], **kwargs) -> None:
        """Create index using the underlying retriever.
        
        Args:
            documents: List of document strings to index
            **kwargs: Additional arguments for indexing
        """
        self.retriever.create_index(documents, **kwargs)
        self.is_indexed = self.retriever.is_indexed
        self.documents = self.retriever.documents
        
        self.logger.info(f"Created index with {len(documents)} documents")
    
    def search(
        self,
        query: str,
        k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search using query extension and result fusion.
        
        Args:
            query: Search query string
            k: Number of results to return
            **kwargs: Additional search parameters passed to the retriever
            
        Returns:
            List of RetrievalResult objects sorted by fused relevance scores
            
        Raises:
            RuntimeError: If retriever is not indexed
            ValueError: If query is invalid or number of extended queries doesn't match weights
        """
        self._check_indexed()
        self._validate_query(query)
        k = self._validate_k(k)
        
        # Extend the query
        try:
            extended_queries = self.query_extension(query)
        except Exception as e:
            self.logger.error(f"Query extension failed: {e}")
            raise RuntimeError(f"Query extension failed: {e}")
        
        if not extended_queries:
            raise ValueError("Query extension returned empty list")
        
        # Validate that number of extended queries matches weights
        if self.weights and len(extended_queries) != len(self.weights):
            raise ValueError(
                f"Number of extended queries ({len(extended_queries)}) must match "
                f"number of weights ({len(self.weights)})"
            )
        
        # Get results from each extended query
        all_results = []
        retrieval_k = min(k * 2, self.get_corpus_size())  # Get more for better fusion
        
        for i, ext_query in enumerate(extended_queries):
            # try:
                results = self.retriever.search(ext_query, k=retrieval_k, **kwargs)
                all_results.append(results)
                self.logger.debug(
                    f"Extended query {i} ('{ext_query[:50]}...') "
                    f"returned {len(results)} results"
                )
            # except Exception as e:
            #     self.logger.warning(
            #         f"Extended query {i} ('{ext_query[:50]}...') search failed: {e}"
            #     )
            #     all_results.append([])  # Empty results for failed query
        
        # Fuse results
        fused_results = self._fuse_results(all_results, extended_queries, k)
        
        return fused_results
    
    def _fuse_results(
        self,
        all_results: List[List[RetrievalResult]],
        extended_queries: List[str],
        k: int
    ) -> List[RetrievalResult]:
        """Fuse results from multiple extended queries using the configured fusion method.
        
        Args:
            all_results: List of result lists from each extended query
            extended_queries: List of extended queries used
            k: Number of final results to return
            
        Returns:
            List of fused RetrievalResult objects
        """
        # Create metadata for extended queries
        fusion_metadata = {
            "retriever_type": self.retriever.__class__.__name__,
            "extended_queries": extended_queries,
            "num_extended_queries": len(extended_queries)
        }
        if self.weights:
            weights = self.weights
        else:
            weights = [1 / len(all_results) for _ in all_results]

        if self.fusion_method == "weighted_sum":
            return _weighted_sum_fusion_shared(
                all_results, weights, k, self.normalize_scores, fusion_metadata
            )
        elif self.fusion_method == "rank_fusion":
            return _rank_fusion_shared(all_results, weights, k, fusion_metadata)
        elif self.fusion_method == "rrf":
            rrf_constant = self.config.get("rrf_constant", 60)
            return _reciprocal_rank_fusion_shared(
                all_results, weights, k, rrf_constant, fusion_metadata
            )
        elif self.fusion_method == "reranking_merge":
            return _reranking_merge_fusion_shared(
                all_results, weights, k, fusion_metadata
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _check_indexed(self) -> None:
        """Check if the retriever has been indexed.
        
        Raises:
            RuntimeError: If the retriever hasn't been indexed
        """
        if not self.is_indexed:
            raise RuntimeError(
                f"Retriever ({self.retriever.__class__.__name__}) must be indexed before searching."
            )
    
    def get_corpus_size(self) -> int:
        """Get the corpus size from the underlying retriever.
        
        Returns:
            Corpus size
        """
        return self.retriever.get_corpus_size()
    
    def add_documents(self, documents: List[str], **kwargs) -> None:
        """Add documents to the underlying retriever.
        
        Args:
            documents: List of document strings to add
            **kwargs: Additional arguments for indexing
        """
        self.retriever.add_documents(documents, **kwargs)
        self.is_indexed = self.retriever.is_indexed
        self.documents = self.retriever.documents
        
        self.logger.info(f"Added {len(documents)} documents")
    
    def save_index(self, path: str) -> None:
        """Save the retriever index and extension configuration.
        
        Args:
            path: Directory path to save the index
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save the underlying retriever's index
        retriever_path = path / f"retriever_{self.retriever.__class__.__name__}"
        try:
            self.retriever.save_index(str(retriever_path))
            self.logger.info(f"Saved retriever to {retriever_path}")
        except NotImplementedError:
            self.logger.warning(
                f"Retriever ({self.retriever.__class__.__name__}) "
                f"does not support save_index"
            )
        
        # Save extension configuration (note: query_extension function cannot be pickled)
        extension_config = {
            "weights": self.weights,
            "fusion_method": self.fusion_method,
            "normalize_scores": self.normalize_scores,
            "config": self.config,
            "retriever_type": self.retriever.__class__.__name__,
            "num_weights": len(self.weights)
        }
        
        with open(path / "extension_config.pkl", "wb") as f:
            pickle.dump(extension_config, f)
        
        self.logger.info(f"Saved extension retriever configuration to {path}")
        self.logger.warning(
            "Query extension function cannot be saved and must be provided "
            "when reconstructing the ExtensionRetriever."
        )
    
    def load_index(self, path: str) -> None:
        """Load the retriever index and extension configuration.
        
        Args:
            path: Directory path to load the index from
            
        Note:
            This method can only load the configuration. The underlying retriever
            and query extension function must be reconstructed separately.
        """
        path = Path(path)
        
        if not (path / "extension_config.pkl").exists():
            raise FileNotFoundError(f"Extension config not found at {path}")
        
        # Load extension configuration
        with open(path / "extension_config.pkl", "rb") as f:
            extension_config = pickle.load(f)
        
        self.weights = extension_config["weights"]
        self.fusion_method = extension_config["fusion_method"]
        self.normalize_scores = extension_config["normalize_scores"]
        
        if "config" in extension_config:
            self.config.update(extension_config["config"])
        
        self.logger.info(f"Loaded extension retriever configuration from {path}")
        self.logger.warning(
            "Underlying retriever and query extension function must be "
            "reconstructed and loaded separately. ExtensionRetriever.load_index "
            "only loads the fusion configuration."
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the extension retriever.
        
        Returns:
            Dictionary containing retriever statistics
        """
        stats = super().get_stats()
        stats.update({
            "num_weights": len(self.weights),
            "weights": self.weights,
            "fusion_method": self.fusion_method,
            "normalize_scores": self.normalize_scores,
            "underlying_retriever": {
                "type": self.retriever.__class__.__name__,
                "is_indexed": self.retriever.is_indexed,
                "corpus_size": self.retriever.get_corpus_size() if self.retriever.is_indexed else 0
            }
        })
        
        return stats
