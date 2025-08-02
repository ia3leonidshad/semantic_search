#!/usr/bin/env python3
"""
Enhanced Query Expansion Script

This script expands queries using various LLM-based methods with configurable options.
Supports rewriting queries, extending with categories, and generating English variants.
"""

import argparse
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_factory import ModelFactory
from src.data.ecommerce_loader import EcommerceDataLoader


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_files(input_file: Path, items_file: Optional[Path] = None) -> None:
    """Validate that required files exist."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input queries file not found: {input_file}")
    
    if items_file and not items_file.exists():
        raise FileNotFoundError(f"Items CSV file not found: {items_file}")


def load_queries(input_file: Path, query_column: str) -> pd.DataFrame:
    """Load queries from CSV file."""
    try:
        queries_df = pd.read_csv(input_file)
        if query_column not in queries_df.columns:
            raise ValueError(f"Query column '{query_column}' not found in {input_file}")
        
        logging.info(f"Loaded {len(queries_df)} queries from {input_file}")
        return queries_df
    except Exception as e:
        raise RuntimeError(f"Failed to load queries: {e}")


def load_items_data(items_file: Path, images_dir: Optional[Path] = None) -> tuple:
    """Load items data if needed for context."""
    try:
        items_db, text_documents, text_item_ids, image_paths, image_item_ids = \
            EcommerceDataLoader.load_from_csv(str(items_file), images_dir=str(images_dir) if images_dir else None)
        
        logging.info(f"Loaded {len(text_documents)} items from {items_file}")
        return items_db, text_documents, text_item_ids, image_paths, image_item_ids
    except Exception as e:
        raise RuntimeError(f"Failed to load items data: {e}")


def process_queries(
    queries_df: pd.DataFrame,
    processor: Any,
    query_column: str,
    operations: List[str],
    batch_size: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """Process queries with specified operations."""
    
    results = {
        'rewrite': [],
        'extend': [],
        'extend_english': []
    }
    
    queries = queries_df[query_column].tolist()
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(queries), batch_size), desc="Processing queries"):
        batch_queries = queries[i:i + batch_size]
        
        for query in batch_queries:
            try:
                # Rewrite query (Portuguese to English)
                if 'rewrite' in operations:
                    response = processor.rewrite_query(query=query)
                    results['rewrite'].append({
                        'english': response.get('query', ''),
                        'thoughts': response.get('thoughts', ''),
                    })
                else:
                    results['rewrite'].append({})
                
                # Extend query with English names
                if 'extend_english' in operations:
                    response = processor.extend_query_english(query=query)
                    extend_eng_dict = {f'extend_eng_{i+1}': v for i, v in enumerate(response.get('names', []))}
                    results['extend_english'].append(extend_eng_dict)
                else:
                    results['extend_english'].append({})
                
                # Extend query with categories and names
                if 'extend' in operations:
                    response = processor.extend_query(query=query)
                    extend_dict = {f'extend_{i+1}': v for i, v in enumerate(response.get('names', []))}
                    extend_dict['category'] = response.get('category', '')
                    results['extend'].append(extend_dict)
                else:
                    results['extend'].append({})
                    
            except Exception as e:
                logging.error(f"Error processing query '{query}': {e}")
                # Add empty results for failed queries
                results['rewrite'].append({})
                results['extend_english'].append({})
                results['extend'].append({})
    
    return results


def save_results(
    original_df: pd.DataFrame,
    results: Dict[str, List[Dict[str, Any]]],
    output_file: Path,
    operations: List[str]
) -> None:
    """Save processed results to CSV."""
    
    dataframes_to_concat = [original_df]
    
    # Add results based on operations performed
    if 'extend' in operations and results['extend']:
        extend_df = pd.DataFrame(results['extend'])
        dataframes_to_concat.append(extend_df)
    
    if 'extend_english' in operations and results['extend_english']:
        extend_eng_df = pd.DataFrame(results['extend_english'])
        dataframes_to_concat.append(extend_eng_df)
    
    if 'rewrite' in operations and results['rewrite']:
        rewrite_df = pd.DataFrame(results['rewrite'])
        dataframes_to_concat.append(rewrite_df)
    
    # Concatenate all dataframes
    final_df = pd.concat(dataframes_to_concat, axis=1)
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")
    logging.info(f"Final dataset shape: {final_df.shape}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Expand queries using LLM-based methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/expand_queries_cli.py --input data/raw/queries.csv --output data/processed/expanded.csv
  
  # With specific operations
  python scripts/expand_queries_cli.py --input queries.csv --output expanded.csv --operations rewrite,extend
  
  # With custom model
  python scripts/expand_queries_cli.py --input queries.csv --output expanded.csv --model-name gpt-4
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input CSV file containing queries'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output CSV file for expanded queries'
    )
    
    # Optional arguments
    parser.add_argument(
        '--query-column',
        type=str,
        default='search_term_pt',
        help='Name of the column containing queries (default: search_term_pt)'
    )
    
    parser.add_argument(
        '--operations',
        type=str,
        default='rewrite,extend,extend_english',
        help='Comma-separated list of operations: rewrite,extend,extend_english (default: all)'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='openai',
        help='Model type to use (default: openai)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='gpt4.1-mini',
        help='Model name to use (default: gpt4.1-mini)'
    )
    
    parser.add_argument(
        '--items-csv',
        type=Path,
        help='Optional items CSV file for context'
    )
    
    parser.add_argument(
        '--images-dir',
        type=Path,
        help='Optional images directory path'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for processing (default: 10)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Validate inputs
        validate_files(args.input, args.items_csv)
        
        # Parse operations
        operations = [op.strip() for op in args.operations.split(',')]
        valid_operations = {'rewrite', 'extend', 'extend_english'}
        invalid_ops = set(operations) - valid_operations
        if invalid_ops:
            raise ValueError(f"Invalid operations: {invalid_ops}. Valid: {valid_operations}")
        
        logging.info(f"Starting query expansion with operations: {operations}")
        
        # Initialize model
        logging.info(f"Initializing {args.model_type} model: {args.model_name}")
        processor = ModelFactory.create_prompt_processor(
            model_type=args.model_type,
            model_name=args.model_name
        )
        
        # Load queries
        queries_df = load_queries(args.input, args.query_column)
        
        # Load items data if provided
        if args.items_csv:
            load_items_data(args.items_csv, args.images_dir)
        
        # Process queries
        results = process_queries(
            queries_df=queries_df,
            processor=processor,
            query_column=args.query_column,
            operations=operations,
            batch_size=args.batch_size
        )
        
        # Save results
        save_results(queries_df, results, args.output, operations)
        
        logging.info("Query expansion completed successfully!")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
