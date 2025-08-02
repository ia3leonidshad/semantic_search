#!/usr/bin/env python3
"""
Feature extraction CLI script for search AI evaluation.

Extracts features from retriever results, item data, and ground truth for ML model training.

Usage:
    python scripts/extract_features_cli.py \
        --retriever-results data/processed/results_3_retrievers.json \
        --item-data data/raw/5k_items_curated.csv \
        --ground-truth data/processed/ground_truth_final.json \
        --output data/processed/features.csv
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
import sys
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.ecommerce_loader import EcommerceDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts features from retriever results for ML training."""
    
    def __init__(self, items_db: Dict[str, Dict]):
        """Initialize feature extractor.
        
        Args:
            items_db: Items database from EcommerceDataLoader
        """
        self.items_db = items_db
        self.stats = {
            'total_queries': 0,
            'total_items': 0,
            'total_features': 0,
            'coverage': 0
        }
    
    def extract_features(self, retriever_results: Dict[str, Any], ground_truth: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """Extract features from retriever results and ground truth.
        
        Args:
            retriever_results: Results from create_retrievers.py
            ground_truth: Ground truth relevance labels
            
        Returns:
            DataFrame with extracted features
        """
        logger.info("Starting feature extraction...")
        
        rows = []
        coverage = 0
        
        # Get all retriever names (excluding metadata)
        retriever_names = [name for name in retriever_results.keys() if name != 'metadata']
        logger.info(f"Found retrievers: {retriever_names}")
        
        # Process each query
        for query in ground_truth.keys():
            logger.debug(f"Processing query: {query}")
            
            # Get ground truth for this query
            gt_dict = ground_truth[query]
            
            # Collect all unique items from all retrievers for this query
            all_items = set()
            for retriever_name in retriever_names:
                if query in retriever_results[retriever_name]:
                    for result in retriever_results[retriever_name][query]:
                        all_items.add(result['item_id'])
            
            # Extract features for each item
            features = defaultdict(dict)
            
            # Calculate category weights for each retriever
            for retriever_name in retriever_names:
                if query not in retriever_results[retriever_name]:
                    continue
                    
                retriever_data = retriever_results[retriever_name][query]
                categories_from_pred = Counter()
                
                # Calculate category distribution weighted by scores
                for result in retriever_data:
                    item_id = result['item_id']
                    if item_id in self.items_db:
                        taxonomy = self.items_db[item_id].get('taxonomy', {})
                        category = ' '.join(taxonomy.values()) if taxonomy else 'unknown'
                        categories_from_pred[category] += result['score']
                
                # Normalize category weights
                sum_score = sum(categories_from_pred.values())
                if sum_score > 0:
                    for k, v in categories_from_pred.items():
                        categories_from_pred[k] /= sum_score
                
                # Extract features for each item in this retriever's results
                for i, result in enumerate(retriever_data):
                    item_id = result['item_id']
                    
                    # Get item category
                    if item_id in self.items_db:
                        taxonomy = self.items_db[item_id].get('taxonomy', {})
                        category = ' '.join(taxonomy.values()) if taxonomy else 'unknown'
                    else:
                        category = 'unknown'
                    
                    # Store features
                    features[item_id][f'{retriever_name}_category_weight'] = categories_from_pred.get(category, 0.0)
                    features[item_id][f'{retriever_name}_score'] = result['score']
                    features[item_id][f'{retriever_name}_rank'] = i + 1
                    features[item_id]['label'] = gt_dict.get(item_id, 0)
                    features[item_id]['query'] = query
                    features[item_id]['item_id'] = item_id
                    
                    # Track coverage
                    if item_id in gt_dict:
                        coverage += 1
            
            # Add features for all items (including those not retrieved by some retrievers)
            for item_id in all_items:
                if item_id not in features:
                    features[item_id] = {
                        'label': gt_dict.get(item_id, 0),
                        'query': query,
                        'item_id': item_id
                    }
            
            # Convert to list of dictionaries
            rows.extend(features.values())
        
        # Create DataFrame
        logger.info("Creating feature DataFrame...")
        df_features = pd.DataFrame(rows)
        
        if df_features.empty:
            logger.warning("No features extracted!")
            return df_features
        
        # Add hit features (binary indicators)
        for retriever_name in retriever_names:
            score_col = f'{retriever_name}_score'
            hit_col = f'{retriever_name}_hit'
            df_features[hit_col] = ~df_features[score_col].isna()
        
        # Fill missing values
        for retriever_name in retriever_names:
            score_col = f'{retriever_name}_score'
            rank_col = f'{retriever_name}_rank'
            category_col = f'{retriever_name}_category_weight'
            
            df_features[score_col].fillna(0.0, inplace=True)
            df_features[rank_col].fillna(40.0, inplace=True)  # Max rank + 1
            df_features[category_col].fillna(0.0, inplace=True)
        
        # Update stats
        self.stats['total_queries'] = len(ground_truth)
        self.stats['total_items'] = len(df_features['item_id'].unique())
        self.stats['total_features'] = len(df_features)
        self.stats['coverage'] = coverage
        
        logger.info(f"Feature extraction complete:")
        logger.info(f"  - Queries processed: {self.stats['total_queries']}")
        logger.info(f"  - Unique items: {self.stats['total_items']}")
        logger.info(f"  - Total feature rows: {self.stats['total_features']}")
        logger.info(f"  - Coverage: {self.stats['coverage']}")
        
        return df_features


def load_retriever_results(file_path: str) -> Dict[str, Any]:
    """Load retriever results from JSON file.
    
    Args:
        file_path: Path to retriever results JSON file
        
    Returns:
        Retriever results dictionary
    """
    logger.info(f"Loading retriever results from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Log metadata if available
    if 'metadata' in results:
        metadata = results['metadata']
        logger.info(f"Retriever results metadata:")
        logger.info(f"  - Queries: {metadata.get('num_queries', 'unknown')}")
        logger.info(f"  - Items: {metadata.get('num_items', 'unknown')}")
        logger.info(f"  - K: {metadata.get('k', 'unknown')}")
    
    return results


def load_ground_truth(file_path: str) -> Dict[str, Dict[str, int]]:
    """Load ground truth from JSON file.
    
    Args:
        file_path: Path to ground truth JSON file
        
    Returns:
        Ground truth dictionary
    """
    logger.info(f"Loading ground truth from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    logger.info(f"Loaded ground truth for {len(ground_truth)} queries")
    return ground_truth


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract features from retriever results for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/extract_features_cli.py \\
        --retriever-results data/processed/results_3_retrievers.json \\
        --item-data data/raw/5k_items_curated.csv \\
        --ground-truth data/processed/ground_truth_final.json \\
        --output data/processed/features.csv
        """
    )
    
    parser.add_argument(
        "--retriever-results",
        required=True,
        help="Path to JSON file with retriever results (from create_retrievers.py)"
    )
    parser.add_argument(
        "--item-data", 
        required=True,
        help="Path to CSV file with item data"
    )
    parser.add_argument(
        "--ground-truth",
        required=True, 
        help="Path to JSON file with ground truth labels"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file for extracted features"
    )
    parser.add_argument(
        "--images-dir",
        default="data/raw/images",
        help="Directory containing item images (default: data/raw/images)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path, name in [
        (args.retriever_results, "Retriever results"),
        (args.item_data, "Item data"),
        (args.ground_truth, "Ground truth")
    ]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{name} file not found: {file_path}")
    
    try:
        # Load data
        retriever_results = load_retriever_results(args.retriever_results)
        ground_truth = load_ground_truth(args.ground_truth)
        
        # Load item data
        logger.info(f"Loading item data from {args.item_data}")
        items_db, text_documents, text_item_ids, image_paths, image_item_ids = EcommerceDataLoader.load_from_csv(
            args.item_data, images_dir=args.images_dir
        )
        
        # Extract features
        extractor = FeatureExtractor(items_db)
        features_df = extractor.extract_features(retriever_results, ground_truth)
        
        if features_df.empty:
            logger.error("No features were extracted. Check your input data.")
            return 1
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving features to {output_path}")
        features_df.to_csv(output_path, index=False)
        
        logger.info("✅ Feature extraction completed successfully!")
        logger.info(f"📊 Features saved to: {output_path}")
        logger.info(f"📈 Shape: {features_df.shape}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during feature extraction: {e}")
        raise


if __name__ == "__main__":
    exit(main())
