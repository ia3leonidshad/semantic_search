"""BM25-based lexical retriever implementation."""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from src.retrievers.base import BaseRetriever, RetrievalResult


class BM25Retriever(BaseRetriever):
    """Lexical retriever using BM25 algorithm."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the BM25 retriever.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.bm25 = None
        self.tokenized_docs = None
        self.item_ids = []  # Track item IDs parallel to documents
        
        # BM25 parameters
        self.k1 = self.config.get("k1", 1.2)
        self.b = self.config.get("b", 0.75)
        self.epsilon = self.config.get("epsilon", 0.25)
        
        # Preprocessing parameters
        self.lowercase = self.config.get("lowercase", True)
        self.remove_punctuation = self.config.get("remove_punctuation", True)
        self.min_token_length = self.config.get("min_token_length", 2)
        
        self.logger.info(f"Initialized BM25Retriever with k1={self.k1}, b={self.b}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing.
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed tokens
        """
        import re
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Filter by minimum length
        if self.min_token_length > 0:
            tokens = [token for token in tokens if len(token) >= self.min_token_length]
        
        return tokens
    
    def create_index(self, documents: List[str], item_ids: List[str] = None, **kwargs) -> None:
        """Index documents using BM25.
        
        Args:
            documents: List of document strings to index
            item_ids: Optional list of item IDs corresponding to documents
            **kwargs: Additional arguments
        """
        if not documents:
            raise ValueError("Cannot index empty document list")
        
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for BM25Retriever. "
                "Install it with: pip install rank-bm25"
            )
        
        self.logger.info(f"Indexing {len(documents)} documents with BM25")
        
        # Store documents and item IDs
        self.documents = documents
        self.item_ids = item_ids or [str(i) for i in range(len(documents))]
        
        if len(self.item_ids) != len(documents):
            raise ValueError("Number of item_ids must match number of documents")
        
        # Preprocess documents
        self.logger.info("Preprocessing documents...")
        self.tokenized_docs = [self._preprocess_text(doc) for doc in documents]
        
        # Create BM25 index
        self.logger.info("Creating BM25 index...")
        self.bm25 = BM25Okapi(
            self.tokenized_docs,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )
        
        self.is_indexed = True
        self.logger.info(f"Successfully indexed {len(documents)} documents")
    
    def search(
        self,
        query: str,
        k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search for relevant documents using BM25.
        
        Args:
            query: Search query string
            k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        self._check_indexed()
        self._validate_query(query)
        k = self._validate_k(k)
        
        # Preprocess query
        query_tokens = self._preprocess_text(query)
        
        if not query_tokens:
            self.logger.warning("Query resulted in no tokens after preprocessing")
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = scores.argsort()[-k:][::-1]  # Sort descending
        
        # Convert to RetrievalResult objects
        results = []
        for i, doc_id in enumerate(top_indices):
            score = float(scores[doc_id])
            
            # Skip documents with zero score
            if score <= 0:
                continue
            
            # Get item ID
            item_id = self.item_ids[doc_id]
            
            result = RetrievalResult(
                item_id=item_id,
                score=score,
                metadata={
                    "bm25_score": score,
                    "query_tokens": query_tokens,
                    "rank": i + 1,
                    "document_text": self.documents[doc_id]  # Store original document text
                }
            )
            results.append(result)
        
        return results
    
    def add_documents(self, documents: List[str], **kwargs) -> None:
        """Add new documents to the existing index.
        
        Args:
            documents: List of document strings to add
            **kwargs: Additional arguments for indexing
        """
        if not self.is_indexed:
            self.index(documents, **kwargs)
            return
        
        # BM25Okapi doesn't support incremental indexing
        # So we need to re-index everything
        self.logger.info(f"Adding {len(documents)} documents (re-indexing required)")
        all_documents = self.documents + documents
        self.index(all_documents, **kwargs)
    
    def save_index(self, path: str) -> None:
        """Save the BM25 index and metadata to disk.
        
        Args:
            path: Directory path to save the index
        """
        if not self.is_indexed:
            raise RuntimeError("Cannot save unindexed retriever")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save all data
        data = {
            "documents": self.documents,
            "tokenized_docs": self.tokenized_docs,
            "bm25": self.bm25,
            "config": self.config,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon
        }
        
        with open(path / "bm25_index.pkl", "wb") as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved BM25 index to {path}")
    
    def load_index(self, path: str) -> None:
        """Load a BM25 index and metadata from disk.
        
        Args:
            path: Directory path to load the index from
        """
        path = Path(path)
        
        if not (path / "bm25_index.pkl").exists():
            raise FileNotFoundError(f"BM25 index not found at {path}")
        
        # Load data
        with open(path / "bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.tokenized_docs = data["tokenized_docs"]
        self.bm25 = data["bm25"]
        self.k1 = data["k1"]
        self.b = data["b"]
        self.epsilon = data["epsilon"]
        
        # Update config
        if "config" in data:
            self.config.update(data["config"])
        
        self.is_indexed = True
        self.logger.info(f"Loaded BM25 index from {path}")
    
    def get_document_frequency(self, term: str) -> int:
        """Get document frequency for a term.
        
        Args:
            term: Term to get frequency for
            
        Returns:
            Number of documents containing the term
        """
        if not self.is_indexed:
            raise RuntimeError("Retriever must be indexed first")
        
        term_tokens = self._preprocess_text(term)
        if not term_tokens:
            return 0
        
        # For multi-token terms, return frequency of the first token
        token = term_tokens[0]
        return self.bm25.doc_freqs.get(token, 0)
    
    def get_term_frequencies(self, document_id: int) -> Dict[str, int]:
        """Get term frequencies for a document.
        
        Args:
            document_id: Document index
            
        Returns:
            Dictionary mapping terms to their frequencies
        """
        if not self.is_indexed:
            raise RuntimeError("Retriever must be indexed first")
        
        if not (0 <= document_id < len(self.tokenized_docs)):
            raise IndexError(f"Document ID {document_id} out of range")
        
        doc_tokens = self.tokenized_docs[document_id]
        term_freqs = {}
        
        for token in doc_tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1
        
        return term_freqs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 retriever.
        
        Returns:
            Dictionary containing retriever statistics
        """
        stats = super().get_stats()
        stats.update({
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "preprocessing": {
                "lowercase": self.lowercase,
                "remove_punctuation": self.remove_punctuation,
                "min_token_length": self.min_token_length
            }
        })
        
        if self.bm25 is not None:
            stats.update({
                "vocabulary_size": len(self.bm25.doc_freqs),
                "average_document_length": self.bm25.avgdl,
                "total_tokens": sum(len(doc) for doc in self.tokenized_docs)
            })
        
        return stats
