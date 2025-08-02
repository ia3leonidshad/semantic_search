"""Base retriever class for all retrieval implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RetrievalResult:
    """Container for retrieval results."""
    
    def __init__(
        self,
        item_id: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize retrieval result.
        
        Args:
            item_id: Item identifier
            score: Relevance score
            metadata: Optional metadata about the result
        """
        self.item_id = item_id
        self.score = score
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"RetrievalResult(item_id={self.item_id}, score={self.score:.4f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "item_id": self.item_id,
            "score": self.score,
            "metadata": self.metadata
        }


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the retriever.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.documents = []
        self.is_indexed = False
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def create_index(self, documents: List[str], **kwargs) -> None:
        """Index a collection of documents.
        
        Args:
            documents: List of document strings to index
            **kwargs: Additional arguments for indexing
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search for relevant documents.
        
        Args:
            query: Search query string
            k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        pass
    
    def add_documents(self, documents: List[str], **kwargs) -> None:
        """Add new documents to the existing index.
        
        Args:
            documents: List of document strings to add
            **kwargs: Additional arguments for indexing
        """
        if not self.is_indexed:
            self.index(documents, **kwargs)
        else:
            # Default implementation: re-index everything
            all_documents = self.documents + documents
            self.index(all_documents, **kwargs)
    
    def get_document(self, document_id: int) -> str:
        """Get a document by its ID.
        
        Args:
            document_id: Document index
            
        Returns:
            Document text
            
        Raises:
            IndexError: If document_id is out of range
        """
        if not (0 <= document_id < len(self.documents)):
            raise IndexError(f"Document ID {document_id} out of range")
        return self.documents[document_id]
    
    def get_corpus_size(self) -> int:
        """Get the number of indexed documents.
        
        Returns:
            Number of documents in the corpus
        """
        return len(self.documents)
    
    def clear_index(self) -> None:
        """Clear the current index."""
        self.documents = []
        self.is_indexed = False
        self.logger.info("Index cleared")
    
    def save_index(self, path: str) -> None:
        """Save the index to disk.
        
        Args:
            path: Path to save the index
            
        Note:
            Default implementation raises NotImplementedError.
            Subclasses should override this method.
        """
        raise NotImplementedError("save_index not implemented for this retriever")
    
    def load_index(self, path: str) -> None:
        """Load an index from disk.
        
        Args:
            path: Path to load the index from
            
        Note:
            Default implementation raises NotImplementedError.
            Subclasses should override this method.
        """
        raise NotImplementedError("load_index not implemented for this retriever")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever.
        
        Returns:
            Dictionary containing retriever statistics
        """
        return {
            "retriever_type": self.__class__.__name__,
            "corpus_size": self.get_corpus_size(),
            "is_indexed": self.is_indexed,
            "config": self.config
        }
    
    def _validate_query(self, query: str) -> None:
        """Validate the search query.
        
        Args:
            query: Query string to validate
            
        Raises:
            ValueError: If query is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
    
    def _validate_k(self, k: int) -> int:
        """Validate and adjust the k parameter.
        
        Args:
            k: Number of results requested
            
        Returns:
            Validated k value (capped by corpus size)
        """
        if k <= 0:
            raise ValueError("k must be positive")
        
        # Cap k by corpus size
        return min(k, self.get_corpus_size())
    
    def _check_indexed(self) -> None:
        """Check if the retriever has been indexed.
        
        Raises:
            RuntimeError: If the retriever hasn't been indexed
        """
        if not self.is_indexed:
            raise RuntimeError("Retriever must be indexed before searching")
