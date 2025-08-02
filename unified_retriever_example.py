"""Example demonstrating the new UnifiedRetriever class."""

import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import the new unified retriever and base classes
from src.retrievers import UnifiedRetriever, BM25Retriever, FaissRetriever
from src.models.embedding_models import BaseEmbeddingModel

# Mock embedding model for demonstration
class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock embedding model for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "mock-model"
        self.embedding_dim = 384
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Return mock embeddings."""
        import random
        return [[random.random() for _ in range(self.embedding_dim)] for _ in texts]
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim


def demonstrate_unified_retriever():
    """Demonstrate the UnifiedRetriever functionality."""
    
    print("=== UnifiedRetriever Demonstration ===\n")
    
    # Sample documents
    documents = [
        "Python is a high-level programming language",
        "Machine learning algorithms can process large datasets",
        "Natural language processing enables computers to understand text",
        "Deep learning uses neural networks with multiple layers",
        "Information retrieval systems help find relevant documents"
    ]
    
    print(f"Sample documents ({len(documents)} total):")
    for i, doc in enumerate(documents):
        print(f"  {i}: {doc}")
    print()
    
    # Create individual retrievers
    print("1. Creating individual retrievers...")
    
    # BM25 retriever
    bm25_retriever = BM25Retriever()
    bm25_retriever.create_index(documents)
    print("   ✓ BM25 retriever indexed")
    
    # FAISS retriever with mock embedding model
    embedding_model = MockEmbeddingModel()
    faiss_retriever = FaissRetriever(embedding_model)
    faiss_retriever.create_index(documents)
    print("   ✓ FAISS retriever indexed")
    
    # 2. Create UnifiedRetriever with different weight configurations
    print("\n2. Creating UnifiedRetriever instances with different configurations...")
    
    # Configuration 1: Equal weights
    print("\n   Configuration 1: Equal weights (0.5, 0.5)")
    unified_equal = UnifiedRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
        fusion_method="weighted_sum"
    )
    
    # Configuration 2: BM25 dominant
    print("   Configuration 2: BM25 dominant (0.8, 0.2)")
    unified_bm25_heavy = UnifiedRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.8, 0.2],
        fusion_method="weighted_sum"
    )
    
    # Configuration 3: FAISS dominant with rank fusion
    print("   Configuration 3: FAISS dominant with rank fusion (0.2, 0.8)")
    unified_faiss_heavy = UnifiedRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.2, 0.8],
        fusion_method="rank_fusion"
    )
    
    # Configuration 4: RRF fusion
    print("   Configuration 4: RRF fusion (0.6, 0.4)")
    unified_rrf = UnifiedRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.6, 0.4],
        fusion_method="rrf"
    )
    
    # 3. Test searches
    print("\n3. Testing searches with different queries...")
    
    queries = [
        "machine learning algorithms",
        "neural networks",
        "programming language"
    ]
    
    retrievers_to_test = [
        ("Equal weights", unified_equal),
        ("BM25 heavy", unified_bm25_heavy),
        ("FAISS heavy", unified_faiss_heavy),
        ("RRF fusion", unified_rrf)
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        print("   " + "="*50)
        
        for name, retriever in retrievers_to_test:
            try:
                results = retriever.search(query, k=3)
                print(f"\n   {name}:")
                for i, result in enumerate(results):
                    doc_text = result.document if result.document else documents[result.document_id]
                    print(f"     {i+1}. Score: {result.score:.4f} - {doc_text[:50]}...")
                    if result.metadata:
                        fusion_method = result.metadata.get('fusion_method', 'unknown')
                        print(f"        Fusion: {fusion_method}")
            except Exception as e:
                print(f"   {name}: Error - {e}")
    
    # 4. Demonstrate weight validation
    print("\n4. Demonstrating weight validation...")
    
    try:
        # This should work - weights will be normalized
        print("   Testing non-normalized weights [2.0, 3.0]...")
        test_retriever = UnifiedRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[2.0, 3.0]
        )
        print(f"   ✓ Normalized to: {test_retriever.weights}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        # This should fail - negative weights
        print("   Testing negative weights [-0.5, 1.0]...")
        UnifiedRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[-0.5, 1.0]
        )
        print("   ✗ Should have failed!")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")
    
    try:
        # This should fail - all zero weights
        print("   Testing zero weights [0.0, 0.0]...")
        UnifiedRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.0, 0.0]
        )
        print("   ✗ Should have failed!")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")
    
    # 5. Show statistics
    print("\n5. Retriever statistics...")
    stats = unified_equal.get_stats()
    print(f"   Retriever type: {stats['retriever_type']}")
    print(f"   Number of retrievers: {stats['num_retrievers']}")
    print(f"   Weights: {stats['weights']}")
    print(f"   Fusion method: {stats['fusion_method']}")
    print(f"   Corpus size: {stats['corpus_size']}")
    print("   Individual retriever stats:")
    for i, retriever_stat in enumerate(stats['retriever_stats']):
        print(f"     {i}: {retriever_stat['type']} (weight: {retriever_stat['weight']:.3f}, "
              f"corpus: {retriever_stat['corpus_size']})")


def show_migration_example():
    """Show how to migrate from old HybridRetriever to UnifiedRetriever."""
    
    print("\n\n=== Migration Example ===\n")
    
    print("OLD WAY (HybridRetriever):")
    print("```python")
    print("from src.retrievers import HybridRetriever")
    print("from src.models.embedding_models import SomeEmbeddingModel")
    print("")
    print("# Old approach - retriever constructs sub-retrievers internally")
    print("embedding_model = SomeEmbeddingModel()")
    print("config = {")
    print("    'vector_weight': 0.7,")
    print("    'bm25_weight': 0.3,")
    print("    'fusion_method': 'weighted_sum'")
    print("}")
    print("hybrid = HybridRetriever(embedding_model, config)")
    print("hybrid.create_index(documents)")
    print("```")
    
    print("\nNEW WAY (UnifiedRetriever):")
    print("```python")
    print("from src.retrievers import UnifiedRetriever, FaissRetriever, BM25Retriever")
    print("from src.models.embedding_models import SomeEmbeddingModel")
    print("")
    print("# New approach - construct retrievers separately, then combine")
    print("embedding_model = SomeEmbeddingModel()")
    print("")
    print("# Create and index individual retrievers")
    print("faiss_retriever = FaissRetriever(embedding_model)")
    print("faiss_retriever.create_index(documents)")
    print("")
    print("bm25_retriever = BM25Retriever()")
    print("bm25_retriever.create_index(documents)")
    print("")
    print("# Combine with UnifiedRetriever")
    print("unified = UnifiedRetriever(")
    print("    retrievers=[faiss_retriever, bm25_retriever],")
    print("    weights=[0.7, 0.3],")
    print("    fusion_method='weighted_sum'")
    print(")")
    print("```")
    
    print("\nBENEFITS of the new approach:")
    print("✓ More flexible - can combine any number and type of retrievers")
    print("✓ Simpler - no complex internal construction logic")
    print("✓ Better separation of concerns - each retriever handles its own indexing")
    print("✓ Automatic weight validation and normalization")
    print("✓ Easier to test and debug individual components")
    print("✓ Supports both text and multimodal retrievers seamlessly")


if __name__ == "__main__":
    demonstrate_unified_retriever()
    show_migration_example()
