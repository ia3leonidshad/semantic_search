#!/usr/bin/env python3
"""
Simple script to create/load retrievers, run predictions, and save results as JSON.

Usage:
    python scripts/create_retrievers.py <queries_csv> <data_csv> <output_json>

Example:
    python scripts/create_retrievers.py ./data/processed/queries_extended_english.csv ./data/raw/5k_items_curated.csv ./results.json
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.faiss_retriever import FaissRetriever
from src.retrievers.unified_retriever import ExtensionRetriever
from src.models.model_factory import ModelFactory
from src.data.ecommerce_loader import EcommerceDataLoader


def create_or_load_bm25_retriever(text_documents: List[str], text_item_ids: List[str]) -> BM25Retriever:
    """Create BM25 retriever.
    
    Args:
        text_documents: List of text documents for indexing
        text_item_ids: List of item IDs corresponding to documents
        
    Returns:
        BM25Retriever instance
    """
    retriever = BM25Retriever(config={
        "k1": 1.2,
        "b": 0.75,
        "lowercase": True,
        "remove_punctuation": True
    })
    
    print(f"Creating new BM25 retriever")
    retriever.create_index(text_documents, text_item_ids)

    return retriever


def create_or_load_image_retriever(save_path: str, items_db: Dict, image_paths: List[str], image_item_ids: List[str]) -> FaissRetriever:
    """Create or load CLIP image retriever.
    
    Args:
        save_path: Path to save/load the retriever
        items_db: Items database
        image_paths: List of image file paths
        image_item_ids: List of item IDs corresponding to images
        
    Returns:
        FaissRetriever instance for images
    """
    save_path = Path(save_path)
    image_model = ModelFactory.create_model("clip", "openai-clip-vit-large-patch14")
    retriever = FaissRetriever(image_model, items_db, config={
        "similarity_metric": "cosine",
        "index_type": "flat"
    })

    if save_path.exists():
        print(f"Loading image retriever from {save_path}")    
        retriever.load_index(str(save_path))
    else:
        print(f"Creating new image retriever and saving to {save_path}")
        
        retriever.create_image_index(image_paths, image_item_ids, show_progress=True)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        retriever.save_index(str(save_path))
    return retriever


def create_or_load_text_embedding_retriever(save_path: str, items_db: Dict, text_documents: List[str], text_item_ids: List[str]) -> FaissRetriever:
    """Create or load OpenAI text embedding retriever.
    
    Args:
        save_path: Path to save/load the retriever
        items_db: Items database
        text_documents: List of text documents for indexing
        text_item_ids: List of item IDs corresponding to documents
        
    Returns:
        FaissRetriever instance for text embeddings
    """
    save_path = Path(save_path)
    
    embedding_model = ModelFactory.create_model("openai", "text-embedding-3-large")
    retriever = FaissRetriever(embedding_model, items_db, config={
        "similarity_metric": "cosine",
        "index_type": "flat"
    })
    if save_path.exists():
        print(f"Loading text embedding retriever from {save_path}")
        
        retriever.load_index(str(save_path))
    else:
        print(f"Creating new text embedding retriever and saving to {save_path}")
        retriever.create_text_index(text_documents, text_item_ids, show_progress=True)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        retriever.save_index(str(save_path))
    return retriever


def run_predictions(queries: List[str], retriever, k: int = 30) -> Dict[str, List[Dict]]:
    """Run predictions using a retriever.
    
    Args:
        queries: List of query strings
        retriever: Retriever instance
        k: Number of results to return per query
        
    Returns:
        Dictionary mapping queries to their results
    """
    results = {}
    
    for query in queries:
        search_results = retriever.search(query, k=k)
        
        # Convert RetrievalResult objects to dictionaries
        results[query] = [result.to_dict() for result in search_results]
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Create retrievers and run predictions")
    parser.add_argument("queries_csv", help="Path to CSV file containing queries")
    parser.add_argument("data_csv", help="Path to CSV file containing item data")
    parser.add_argument("output_json", help="Path to output JSON file for results")
    parser.add_argument("--k", type=int, default=30, help="Number of results per query (default: 30)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.queries_csv).exists():
        raise FileNotFoundError(f"Queries CSV not found: {args.queries_csv}")
    if not Path(args.data_csv).exists():
        raise FileNotFoundError(f"Data CSV not found: {args.data_csv}")
    
    # Load queries
    print(f"Loading queries from {args.queries_csv}")
    queries_df = pd.read_csv(args.queries_csv)
    
    # Assume queries are in 'search_term_pt' column, adjust if needed
    if 'search_term_pt' in queries_df.columns:
        queries = queries_df['search_term_pt'].tolist()
    else:
        raise ValueError(f'No column search_term_pt in {queries_df.columns}')
    
    mapping_extension = {}

    for i, row in queries_df.iterrows():
        mapping_extension[row.search_term_pt] = {
            'english': row.english,
            'extensions': [row.extend_1, row.extend_2, row.extend_3,],
            'extensions_english': [row.extend_eng_1, row.extend_eng_2, row.extend_eng_3,],
            'category': row.category,
        }

    def simple_query_extension(query):
        return [query]

    def category_query_extension(query):
        return [f"Category: {mapping_extension[query]['category']}\nName: {query}"]

    def category_query_extension_multiple(query):
        return [
            f"Category: {mapping_extension[query]['category']}\nName: {q}"
            for q in [query] + mapping_extension[query]['extensions']
        ]

    def query_extension_multiple(query):
        return [query] + mapping_extension[query]['extensions']

    def query_extension_english(query):
        return [mapping_extension[query]['english']] + mapping_extension[query]['extensions_english']

    mapping = dict(zip(queries_df.search_term_pt, queries_df.english))

    def static_query_translator(query):
        return mapping[query]

    print(f"Loaded {len(queries)} queries")
    
    # Load data
    print(f"Loading data from {args.data_csv}")
    items_db, text_documents, text_item_ids, image_paths, image_item_ids = EcommerceDataLoader.load_from_csv(
        args.data_csv, images_dir="./data/raw/images"
    )
    
    # Define save paths for retrievers
    image_path = "./data/indices/image_clip_patch14"
    text_embedding_path = "./data/indices/text-embedding-3-large-full"
    
    # Create or load retrievers
    print("\n=== Creating/Loading BM25 Retriever ===")
    bm25_retriever = create_or_load_bm25_retriever(text_documents, text_item_ids)

    bm25_retriever_extension = ExtensionRetriever(
        retriever=bm25_retriever,
        query_extension=query_extension_multiple,
        weights=[1, 1, 1, 1],  # Higher weight for original query
        fusion_method="weighted_sum",
        normalize_scores=False
    )
    
    # print("\n=== Creating/Loading Image Retriever ===")
    # image_retriever = create_or_load_image_retriever(image_path, items_db, image_paths, image_item_ids)
    
    # image_retriever_extension = ExtensionRetriever(
    #     retriever=image_retriever,
    #     query_extension=query_extension_english,
    #     weights=[1, 1, 1, 1],  # Higher weight for original query
    #     fusion_method="weighted_sum",
    #     normalize_scores=False
    # )

    # print("\n=== Creating/Loading Text Embedding Retriever ===")
    # text_embedding_retriever = create_or_load_text_embedding_retriever(text_embedding_path, items_db, text_documents, text_item_ids)
    
    # Run predictions for all retrievers
    print("\n=== Running Predictions ===")
    
    print("Running BM25 predictions...")
    bm25_results = run_predictions(queries, bm25_retriever_extension, k=args.k)
    
    # print("Running image retriever predictions...")
    # image_results = run_predictions(queries, image_retriever_extension, k=args.k)
    
    # print("Running text embedding predictions...")
    # text_embedding_results = run_predictions(queries, text_embedding_retriever, k=args.k)
    
    # Combine all results
    all_results = {
        "bm25": bm25_results,
        # "image_clip": image_results,
        # "text_embedding": text_embedding_results,
        "metadata": {
            "num_queries": len(queries),
            "k": args.k,
            "queries_file": args.queries_csv,
            "data_file": args.data_csv,
            "num_items": len(items_db),
            "num_text_documents": len(text_documents),
            "num_images": len(image_paths)
        }
    }
    
    # Save results
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Complete! Results saved to {output_path}")
    print(f"📊 Processed {len(queries)} queries across 3 retrievers")


if __name__ == "__main__":
    main()
