#!/usr/bin/env python3
"""
Query Generation Script

This script generates new queries using LLM-based methods.
Supports generating queries for specific items or from example patterns.
"""

import argparse
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import sys
from tqdm import tqdm
import random

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


def validate_files(*files: Path) -> None:
    """Validate that required files exist."""
    for file_path in files:
        if file_path and not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")


def load_examples(examples_file: Path, query_column: str = 'search_term_pt') -> List[str]:
    """Load example queries from CSV file."""
    try:
        df = pd.read_csv(examples_file)
        if query_column not in df.columns:
            raise ValueError(f"Query column '{query_column}' not found in {examples_file}")
        
        queries = df[query_column].dropna().tolist()
        logging.info(f"Loaded {len(queries)} example queries from {examples_file}")
        return queries
    except Exception as e:
        raise RuntimeError(f"Failed to load examples: {e}")


def load_items_data(items_file: Path, images_dir: Optional[Path] = None) -> tuple:
    """Load items data using EcommerceDataLoader."""
    try:
        items_db, text_documents, text_item_ids, image_paths, image_item_ids = \
            EcommerceDataLoader.load_from_csv(str(items_file), images_dir=str(images_dir) if images_dir else None)
        
        logging.info(f"Loaded {len(text_documents)} items from {items_file}")
        return items_db, text_documents, text_item_ids, image_paths, image_item_ids
    except Exception as e:
        raise RuntimeError(f"Failed to load items data: {e}")


def generate_queries_for_items(
    processor: Any,
    text_documents: List[str],
    examples: List[str],
    count: int,
    batch_size: int = 5
) -> List[Dict[str, Any]]:
    """Generate queries for specific items."""
    
    results = []
    
    # Sample items if we have more than needed
    if len(text_documents) > count:
        selected_items = random.sample(text_documents, count)
    else:
        selected_items = text_documents
        if len(text_documents) < count:
            logging.warning(f"Only {len(text_documents)} items available, but {count} queries requested")
    
    # Process items in batches
    for i in tqdm(range(0, len(selected_items), batch_size), desc="Generating queries for items"):
        batch_items = selected_items[i:i + batch_size]
        
        for item in batch_items:
            try:
                response = processor.generate_query_for_item(queries=examples, item=item)
                
                result = {
                    'generated_query': response.get('query', ''),
                    'source_item': item,
                    'generation_method': 'for_item'
                }
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error generating query for item '{item[:50]}...': {e}")
                results.append({
                    'generated_query': '',
                    'source_item': item,
                    'generation_method': 'for_item',
                    'error': str(e)
                })
    
    return results


def generate_queries_from_examples(
    processor: Any,
    examples: List[str],
    count: int,
    batch_size: int = 10,
) -> List[Dict[str, Any]]:
    """Generate queries based on example patterns."""
    
    results = []
    # Process in batches
    for i in tqdm(range(0, count, batch_size), desc="Generating queries from examples"):
        batch_size_actual = min(batch_size, count - i)
        
        for _ in range(batch_size_actual):
            try:
                response = processor.generate_query_from_examples(queries=random.choices(examples, k=20))
                
                result = {
                    'generated_query': response.get('query', ''),
                    'source_item': '',
                    'generation_method': 'from_examples'
                }
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error generating query from examples: {e}")
                results.append({
                    'generated_query': '',
                    'source_item': '',
                    'generation_method': 'from_examples',
                    'error': str(e)
                })
    
    return results


def filter_and_deduplicate(
    results: List[Dict[str, Any]],
    min_length: int = 2,
    max_length: int = 50,
    remove_duplicates: bool = True
) -> List[Dict[str, Any]]:
    """Filter and deduplicate generated queries."""
    
    filtered_results = []
    seen_queries = set()
    
    for result in results:
        query = result.get('generated_query', '').strip()
        
        # Skip empty or error results
        if not query or 'error' in result:
            continue
        
        # Length filtering
        word_count = len(query.split())
        if word_count < min_length or word_count > max_length:
            continue
        
        # Deduplication
        if remove_duplicates:
            query_lower = query.lower()
            if query_lower in seen_queries:
                continue
            seen_queries.add(query_lower)
        
        filtered_results.append(result)
    
    logging.info(f"Filtered {len(results)} -> {len(filtered_results)} queries")
    return filtered_results


def save_results(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Save generated queries to CSV file."""
    
    if not results:
        logging.warning("No results to save")
        return
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    logging.info(f"Saved {len(results)} generated queries to {output_file}")
    
    # Print some statistics
    if 'generated_query' in df.columns:
        successful_queries = df[df['generated_query'].str.len() > 0]
        logging.info(f"Successful generations: {len(successful_queries)}/{len(df)}")
        
        if len(successful_queries) > 0:
            avg_length = successful_queries['generated_query'].str.split().str.len().mean()
            logging.info(f"Average query length: {avg_length:.1f} words")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate queries using LLM-based methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate queries for specific items
  python scripts/generate_queries_cli.py --mode for_items \\
    --items-csv data/raw/5k_items_curated.csv \\
    --examples-csv data/raw/queries.csv \\
    --output data/processed/generated_queries.csv \\
    --count 100
  
  # Generate queries from examples only
  python scripts/generate_queries_cli.py --mode from_examples \\
    --examples-csv data/raw/queries.csv \\
    --output data/processed/generated_queries.csv \\
    --count 50
  
  # With custom filtering
  python scripts/generate_queries_cli.py --mode from_examples \\
    --examples-csv queries.csv --output generated.csv \\
    --count 100 --min-length 3 --max-length 8
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['for_items', 'from_examples'],
        help='Generation mode: for_items or from_examples'
    )
    
    parser.add_argument(
        '--examples-csv',
        type=Path,
        required=True,
        help='CSV file containing example queries'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output CSV file for generated queries'
    )
    
    parser.add_argument(
        '--count', '-c',
        type=int,
        required=True,
        help='Number of queries to generate'
    )
    
    # Mode-specific arguments
    parser.add_argument(
        '--items-csv',
        type=Path,
        help='CSV file containing items (required for for_items mode)'
    )
    
    parser.add_argument(
        '--images-dir',
        type=Path,
        help='Optional images directory path'
    )
    
    # Optional arguments
    parser.add_argument(
        '--query-column',
        type=str,
        default='search_term_pt',
        help='Column name containing example queries (default: search_term_pt)'
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
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for processing (default: 10)'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=2,
        help='Minimum query length in words (default: 2)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=50,
        help='Maximum query length in words (default: 50)'
    )
    
    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Disable deduplication of generated queries'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        logging.info(f"Random seed set to {args.seed}")
    
    try:
        # Validate mode-specific requirements
        if args.mode == 'for_items' and not args.items_csv:
            raise ValueError("--items-csv is required for 'for_items' mode")
        
        # Validate files
        files_to_check = [args.examples_csv]
        if args.items_csv:
            files_to_check.append(args.items_csv)
        validate_files(*files_to_check)
        
        logging.info(f"Starting query generation in '{args.mode}' mode")
        logging.info(f"Target count: {args.count} queries")
        
        # Initialize model
        logging.info(f"Initializing {args.model_type} model: {args.model_name}")
        processor = ModelFactory.create_prompt_processor(
            model_type=args.model_type,
            model_name=args.model_name
        )
        
        # Load example queries
        examples = load_examples(args.examples_csv, args.query_column)
        
        # Generate queries based on mode
        if args.mode == 'for_items':
            # Load items data using EcommerceDataLoader
            items_db, text_documents, text_item_ids, image_paths, image_item_ids = \
                load_items_data(args.items_csv, args.images_dir)
            
            results = generate_queries_for_items(
                processor=processor,
                text_documents=text_documents,
                examples=examples,
                count=args.count,
                batch_size=args.batch_size
            )
        else:  # from_examples
            results = generate_queries_from_examples(
                processor=processor,
                examples=examples,
                count=args.count,
                batch_size=args.batch_size
            )
        
        # Filter and deduplicate results
        filtered_results = filter_and_deduplicate(
            results=results,
            min_length=args.min_length,
            max_length=args.max_length,
            remove_duplicates=not args.no_dedup
        )
        
        # Save results
        save_results(filtered_results, args.output)
        
        logging.info("Query generation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
