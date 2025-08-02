"""Faiss-based vector retriever implementation."""

import numpy as np
from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path

from src.retrievers.base import BaseRetriever, RetrievalResult
from src.models.embedding_models import BaseEmbeddingModel


class FaissRetriever(BaseRetriever):
    """Vector retriever using Faiss for similarity search."""
    
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        items_db: Optional[Dict[str, Dict]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Faiss retriever.
        
        Args:
            embedding_model: Embedding model for encoding documents and queries
            items_db: Optional items database for metadata lookup
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.embedding_model = embedding_model
        self.items_db = items_db or {}
        self.index = None
        self.embeddings = None
        self.item_ids = []  # Track item IDs parallel to embeddings
        
        # Configuration
        self.similarity_metric = self.config.get("similarity_metric", "cosine")
        self.index_type = self.config.get("index_type", "flat")
        self.nprobe = self.config.get("nprobe", 10)  # For IVF indices
        
        self.logger.info(f"Initialized FaissRetriever with {self.similarity_metric} similarity")
    
    def create_index(self, documents: List[str], item_ids: List[str] = None, **kwargs) -> None:
        """Index documents using Faiss.
        
        Args:
            documents: List of document strings to index
            item_ids: Optional list of item IDs corresponding to documents
            **kwargs: Additional arguments (show_progress, etc.)
        """
        if not documents:
            raise ValueError("Cannot index empty document list")
        
        self.logger.info(f"Indexing {len(documents)} documents with Faiss")
        
        # Store documents and item IDs
        self.documents = documents
        self.item_ids = item_ids or [str(i) for i in range(len(documents))]
        
        if len(self.item_ids) != len(documents):
            raise ValueError("Number of item_ids must match number of documents")
        
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        self.embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=kwargs.get("show_progress", True)
        )
        
        # Create Faiss index
        self._create_faiss_index()
        self.is_indexed = True
        self.logger.info(f"Successfully indexed {len(documents)} documents")
    
    def create_text_index(self, text_documents: List[str], item_ids: List[str], **kwargs) -> None:
        """Create index from text documents with item IDs.
        
        Args:
            text_documents: List of text documents
            item_ids: List of item IDs corresponding to documents
            **kwargs: Additional arguments
        """
        self.create_index(text_documents, item_ids, **kwargs)
    
    def create_image_index(self, image_paths: List[str], item_ids: List[str], **kwargs) -> None:
        """Create index from image paths with item IDs.
        
        Args:
            image_paths: List of image file paths
            item_ids: List of item IDs corresponding to images
            **kwargs: Additional arguments
        """
        if not image_paths:
            raise ValueError("Cannot index empty image list")
        
        # Check if model supports images
        if not hasattr(self.embedding_model, 'encode_images'):
            raise ValueError(f"Model {self.embedding_model.model_name} does not support image encoding")
        
        self.logger.info(f"Indexing {len(image_paths)} images with Faiss")
        
        # Store image paths and item IDs
        self.documents = image_paths  # Store paths as "documents"
        self.item_ids = item_ids
        
        if len(self.item_ids) != len(image_paths):
            raise ValueError("Number of item_ids must match number of images")
        
        # Generate embeddings
        self.logger.info("Generating image embeddings...")
        self.embeddings = self.embedding_model.encode_images(
            image_paths,
            show_progress_bar=kwargs.get("show_progress", True)
        )
        
        # Create Faiss index
        self._create_faiss_index()
        self.is_indexed = True
        self.logger.info(f"Successfully indexed {len(image_paths)} images")
    
    def _create_faiss_index(self) -> None:
        """Create and populate the Faiss index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required for FaissRetriever. "
                "Install it with: pip install faiss-cpu or pip install faiss-gpu"
            )
        
        dimension = self.embeddings.shape[1]
        
        # Choose index type based on configuration
        if self.index_type == "flat":
            if self.similarity_metric == "cosine":
                # For cosine similarity, use inner product with normalized vectors
                self.index = faiss.IndexFlatIP(dimension)
                # Normalize embeddings for cosine similarity
                # faiss.normalize_L2(self.embeddings)
                self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            else:  # L2 distance
                self.index = faiss.IndexFlatL2(dimension)
        
        elif self.index_type == "ivf":
            # IVF (Inverted File) index for faster search on large datasets
            nlist = min(100, len(self.documents) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            
            if self.similarity_metric == "cosine":
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                # faiss.normalize_L2(self.embeddings)
                self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            else:
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            
            # Train the index
            self.index.train(self.embeddings)
            self.index.nprobe = self.nprobe
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # Add embeddings to index
        self.index.add(self.embeddings)
        self.logger.info(f"Created {self.index_type} index with {self.similarity_metric} similarity")
    
    def search(
        self,
        query: str,
        k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search for similar documents using Faiss.
        
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
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            # prompt_name='retrieval.query'
        )

        # Normalize for cosine similarity
        if self.similarity_metric == "cosine":
            query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Convert to RetrievalResult objects
        results = []
        for i, (score, doc_id) in enumerate(zip(scores[0], indices[0])):
            if doc_id == -1:  # Faiss returns -1 for empty slots
                break
            
            # Convert score based on similarity metric
            if self.similarity_metric == "cosine":
                # For cosine similarity, higher scores are better
                final_score = float(score)
            else:
                # For L2 distance, lower scores are better, so invert
                final_score = 1.0 / (1.0 + float(score))
            
            # Get item ID
            item_id = self.item_ids[doc_id]
            
            result = RetrievalResult(
                item_id=item_id,
                score=final_score,
                metadata={
                    "raw_score": float(score),
                    "similarity_metric": self.similarity_metric,
                    "rank": i + 1,
                    "document_path": self.documents[doc_id]  # Store original document/image path
                }
            )
            results.append(result)
        
        return results
    
    def search_with_embedding(
        self,
        embedding: np.ndarray,
        k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """Search using a pre-computed embedding.
        
        Args:
            embedding: Pre-computed embedding vector
            k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        self._check_indexed()
        k = self._validate_k(k)
        
        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        if self.similarity_metric == "cosine":
            embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(embedding, k)
        
        # Convert to RetrievalResult objects
        results = []
        for i, (score, doc_id) in enumerate(zip(scores[0], indices[0])):
            if doc_id == -1:  # Faiss returns -1 for empty slots
                break
            
            # Convert score based on similarity metric
            if self.similarity_metric == "cosine":
                final_score = float(score)
            else:
                final_score = 1.0 / (1.0 + float(score))
            
            # Get item ID
            item_id = self.item_ids[doc_id]
            
            result = RetrievalResult(
                item_id=item_id,
                score=final_score,
                metadata={
                    "raw_score": float(score),
                    "similarity_metric": self.similarity_metric,
                    "rank": i + 1,
                    "document_path": self.documents[doc_id]
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
        
        self.logger.info(f"Adding {len(documents)} documents to existing index")
        
        # Generate embeddings for new documents
        new_embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=kwargs.get("show_progress", True)
        )
        
        # Normalize if using cosine similarity
        if self.similarity_metric == "cosine":
            # import faiss
            # faiss.normalize_L2(new_embeddings)
            self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Add to index
        self.index.add(new_embeddings)
        
        # Update stored data
        self.documents.extend(documents)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.logger.info(f"Successfully added {len(documents)} documents")
    
    def save_index(self, path: str) -> None:
        """Save the Faiss index and metadata to disk.
        
        Args:
            path: Directory path to save the index
        """
        if not self.is_indexed:
            raise RuntimeError("Cannot save unindexed retriever")
        
        import faiss
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save Faiss index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save metadata
        metadata = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "config": self.config,
            "similarity_metric": self.similarity_metric,
            "index_type": self.index_type,
            "embedding_model_config": self.embedding_model.config,
            "items_db": self.items_db,
            "item_ids": self.item_ids,
        }
        
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        self.logger.info(f"Saved index to {path}")
    
    def load_index(self, path: str) -> None:
        """Load a Faiss index and metadata from disk.
        
        Args:
            path: Directory path to load the index from
        """
        import faiss
        
        path = Path(path)
        
        if not (path / "faiss.index").exists():
            raise FileNotFoundError(f"Faiss index not found at {path}")
        
        # Load Faiss index
        self.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load metadata
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        self.documents = metadata["documents"]
        self.embeddings = metadata["embeddings"]
        self.similarity_metric = metadata["similarity_metric"]
        self.index_type = metadata["index_type"]
        self.item_ids = metadata['item_ids']
        self.items_db = metadata['items_db']
        
        # Update config
        if "config" in metadata:
            self.config.update(metadata["config"])
        
        self.is_indexed = True
        self.logger.info(f"Loaded index from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Faiss retriever.
        
        Returns:
            Dictionary containing retriever statistics
        """
        stats = super().get_stats()
        stats.update({
            "similarity_metric": self.similarity_metric,
            "index_type": self.index_type,
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else None,
            "embedding_model": self.embedding_model.model_name,
        })
        
        if self.index is not None:
            stats["index_size"] = self.index.ntotal
        
        return stats
