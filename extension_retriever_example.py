"""Example demonstrating the ExtensionRetriever with query extension."""

import logging
from typing import List

from src.retrievers.unified_retriever import ExtensionRetriever
from src.retrievers.bm25_retriever import BM25Retriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_query_extension(query: str) -> List[str]:
    """Simple query extension function that creates variations of the input query.
    
    Args:
        query: Original query string
        
    Returns:
        List of extended queries
    """
    # Create variations of the query
    extended_queries = [
        query,  # Original query
        f"What is {query}?",  # Question form
        f"{query} definition",  # Definition seeking
        f"{query} explanation"  # Explanation seeking
    ]
    
    return extended_queries


def synonym_query_extension(query: str) -> List[str]:
    """Query extension with simple synonym replacement.
    
    Args:
        query: Original query string
        
    Returns:
        List of extended queries with synonyms (always returns 2 queries)
    """
    # Simple synonym mapping
    synonyms = {
        "car": "vehicle",
        "house": "home",
        "big": "large",
        "small": "tiny",
        "fast": "quick",
        "good": "excellent"
    }
    
    # Start with original query
    extended_queries = [query]
    
    # Create one synonym variation by replacing the first found synonym
    words = query.lower().split()
    synonym_query = query.lower()
    
    for word in words:
        if word in synonyms:
            synonym_query = synonym_query.replace(word, synonyms[word])
            break  # Only replace the first synonym found
    
    # Always return exactly 2 queries: original and synonym version
    extended_queries.append(synonym_query)
    
    return extended_queries


def main():
    """Demonstrate ExtensionRetriever functionality."""
    
    # Sample documents
    documents = [
        "The car is a fast vehicle used for transportation.",
        "A house is a large building where people live.",
        "Python is a programming language used for software development.",
        "Machine learning is a subset of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence helps automate complex tasks.",
        "A vehicle can be a car, truck, or motorcycle.",
        "Programming languages include Python, Java, and C++.",
        "Large buildings require strong foundations.",
        "Transportation systems include cars, trains, and buses."
    ]
    
    # Create base retriever
    base_retriever = BM25Retriever()
    base_retriever.create_index(documents)
    
    print("=== ExtensionRetriever Example ===\n")
    
    # Example 1: Simple query extension with equal weights
    print("1. Simple Query Extension with Equal Weights")
    print("-" * 50)
    
    extension_retriever = ExtensionRetriever(
        retriever=base_retriever,
        query_extension=simple_query_extension,
        weights=[0.4, 0.2, 0.2, 0.2],  # Higher weight for original query
        fusion_method="weighted_sum",
        normalize_scores=True
    )
    
    query = "car"
    results = extension_retriever.search(query, k=3)
    
    print(f"Query: '{query}'")
    print(f"Extended queries: {simple_query_extension(query)}")
    print("\nTop 3 results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f} - {documents[int(result.item_id)]}")
        print(f"   Extended queries: {result.metadata.get('extended_queries', [])}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 2: Synonym-based extension with reranking merge
    print("2. Synonym-based Extension with Reranking Merge")
    print("-" * 50)
    
    synonym_retriever = ExtensionRetriever(
        retriever=base_retriever,
        query_extension=synonym_query_extension,
        weights=[0.7, 0.3],  # Higher weight for original, lower for synonym
        fusion_method="reranking_merge",  # New fusion method
        normalize_scores=False  # Don't normalize for reranking merge
    )
    
    query = "fast car"
    results = synonym_retriever.search(query, k=3)
    
    print(f"Query: '{query}'")
    print(f"Extended queries: {synonym_query_extension(query)}")
    print("\nTop 3 results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f} - {documents[int(result.item_id)]}")
        print(f"   Individual scores: {result.metadata.get('individual_scores', [])}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 3: Rank fusion method
    print("3. Query Extension with Rank Fusion")
    print("-" * 50)
    
    rank_retriever = ExtensionRetriever(
        retriever=base_retriever,
        query_extension=simple_query_extension,
        weights=[0.5, 0.2, 0.15, 0.15],
        fusion_method="rank_fusion"
    )
    
    query = "programming language"
    results = rank_retriever.search(query, k=3)
    
    print(f"Query: '{query}'")
    print(f"Extended queries: {simple_query_extension(query)}")
    print("\nTop 3 results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f} - {documents[int(result.item_id)]}")
        print(f"   Individual ranks: {result.metadata.get('individual_ranks', [])}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 4: Compare with base retriever
    print("4. Comparison with Base Retriever")
    print("-" * 50)
    
    query = "artificial intelligence"
    
    # Base retriever results
    base_results = base_retriever.search(query, k=3)
    print(f"Base retriever results for '{query}':")
    for i, result in enumerate(base_results, 1):
        print(f"{i}. Score: {result.score:.4f} - {documents[int(result.item_id)]}")
    
    print()
    
    # Extension retriever results
    ext_results = extension_retriever.search(query, k=3)
    print(f"Extension retriever results for '{query}':")
    for i, result in enumerate(ext_results, 1):
        print(f"{i}. Score: {result.score:.4f} - {documents[int(result.item_id)]}")
    
    print("\n" + "="*70 + "\n")
    
    # Show statistics
    print("5. Retriever Statistics")
    print("-" * 50)
    
    stats = extension_retriever.get_stats()
    print("ExtensionRetriever stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
