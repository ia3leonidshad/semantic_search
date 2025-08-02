#!/usr/bin/env python3
"""
Example demonstrating how to use the retrieval evaluation module.

This script shows how to:
1. Create mock retrieval results
2. Define ground truth data
3. Calculate individual metrics
4. Evaluate across a dataset
"""

import pandas as pd
import json
from pathlib import Path

import logging
from src.retrievers.base import RetrievalResult
from src.models.model_factory import ModelFactory
from src.data.ecommerce_loader import EcommerceDataLoader
from src.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    evaluate_dataset,
    print_evaluation_results
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_data():
    """Create mock retrieval results and ground truth for demonstration."""
    
    # Mock retrieval results for different queries
    query_results = {
        "pasta_query": [
            RetrievalResult("item_001", 0.95),  # Highly relevant pasta
            RetrievalResult("item_002", 0.87),  # Partially relevant pasta
            RetrievalResult("item_003", 0.82),  # Irrelevant item
            RetrievalResult("item_004", 0.78),  # Another pasta dish
            RetrievalResult("item_005", 0.65),  # Irrelevant item
        ],
        "chicken_query": [
            RetrievalResult("item_101", 0.92),  # Chicken dish
            RetrievalResult("item_102", 0.88),  # Irrelevant item
            RetrievalResult("item_103", 0.85),  # Another chicken dish
            RetrievalResult("item_104", 0.79),  # Partially relevant
            RetrievalResult("item_105", 0.72),  # Irrelevant item
        ],
        "salad_query": [
            RetrievalResult("item_201", 0.89),  # Salad dish
            RetrievalResult("item_202", 0.84),  # Salad dish
            RetrievalResult("item_203", 0.81),  # Partially relevant
            RetrievalResult("item_204", 0.77),  # Irrelevant item
            RetrievalResult("item_205", 0.73),  # Irrelevant item
        ]
    }
    
    # Ground truth relevance judgments
    ground_truth = {
        "pasta_query": {
            "item_001": 2,  # Highly relevant
            "item_002": 1,  # Partially relevant
            "item_003": 0,  # Irrelevant
            "item_004": 2,  # Highly relevant
            "item_005": 0,  # Irrelevant
            "item_006": 1,  # Partially relevant (not in results)
        },
        "chicken_query": {
            "item_101": 2,  # Highly relevant
            "item_102": 0,  # Irrelevant
            "item_103": 2,  # Highly relevant
            "item_104": 1,  # Partially relevant
            "item_105": 0,  # Irrelevant
            "item_106": 1,  # Partially relevant (not in results)
        },
        "salad_query": {
            "item_201": 2,  # Highly relevant
            "item_202": 2,  # Highly relevant
            "item_203": 1,  # Partially relevant
            "item_204": 0,  # Irrelevant
            "item_205": 0,  # Irrelevant
            "item_206": 1,  # Partially relevant (not in results)
        }
    }
    
    return query_results, ground_truth


def main():
    queries = pd.read_csv("/Users/lekimov/Documents/search_ai/data/processed/queries_english.csv")

    csv_path = "/Users/lekimov/Documents/search_ai/data/raw/5k_items_curated.csv"
    if not Path(csv_path).exists():
        raise ValueError("Sample CSV not found.")

    items_db, text_documents, text_item_ids, image_paths, image_item_ids = EcommerceDataLoader.load_from_csv(
        csv_path, images_dir="/Users/lekimov/Documents/search_ai/data/raw/images"
    )

    open_ai_embedding_model = ModelFactory.create_model("openai", "text-embedding-3-small")


if __name__ == "__main__":
    main()
