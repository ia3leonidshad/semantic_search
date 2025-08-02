#!/usr/bin/env python3
"""
Script to create ground truth file from retriever results using LLM processor judge.

Takes results from create_retrievers.py and generates ground truth in format:
{query: {item_id: score, ...}, ...}

Usage:
    python scripts/create_ground_truth.py <retriever_results.json> <data_csv> <output_ground_truth.json>

Example:
    python scripts/create_ground_truth.py ./results.json ./data/raw/5k_items_curated.csv ./ground_truth.json
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from tqdm import tqdm
import traceback

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_factory import ModelFactory
from src.data.ecommerce_loader import EcommerceDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ground_truth_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Thread-safe locks
results_lock = Lock()
progress_lock = Lock()


class GroundTruthGenerator:
    """Generator for creating ground truth using LLM judge."""
    
    def __init__(self, model_type: str = "openai", model_name: str = "gpt-4.1-mini", 
                 threads: int = 8, timeout: int = 30, retry_attempts: int = 3):
        """Initialize the ground truth generator.
        
        Args:
            model_type: Type of LLM model to use
            model_name: Specific model name
            threads: Number of concurrent threads
            timeout: Timeout per LLM call in seconds
            retry_attempts: Number of retry attempts for failed calls
        """
        self.model_type = model_type
        self.model_name = model_name
        self.threads = threads
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        
        # Statistics tracking
        self.stats = {
            'total_pairs': 0,
            'successful_judgments': 0,
            'failed_judgments': 0,
            'score_distribution': {0: 0, 1: 0, 2: 0},
            'processing_time': 0
        }
        
        logger.info(f"Initialized GroundTruthGenerator with {threads} threads")
    
    def create_prompt_processor(self):
        """Create a new prompt processor instance for a thread."""
        try:
            return ModelFactory.create_prompt_processor(
                model_type=self.model_type,
                model_name=self.model_name
            )
        except Exception as e:
            logger.error(f"Failed to create prompt processor: {e}")
            raise
    
    def extract_query_item_pairs(self, retriever_results: Dict[str, Any]) -> Set[Tuple[str, str]]:
        """Extract unique query-item pairs from retriever results.
        
        Args:
            retriever_results: Results from create_retrievers.py
            
        Returns:
            Set of (query, item_id) tuples
        """
        pairs = set()
        
        # Iterate through all retriever types
        for retriever_type, retriever_data in retriever_results.items():
            if retriever_type == 'metadata':
                continue
                
            logger.info(f"Processing {retriever_type} results")
            
            for query, results_list in retriever_data.items():
                for result in results_list:
                    if isinstance(result, dict) and 'item_id' in result:
                        pairs.add((query, result['item_id']))
        
        logger.info(f"Extracted {len(pairs)} unique query-item pairs")
        return pairs
    
    def judge_query_item_pair(self, query: str, item_id: str, item_info: Dict[str, Any], 
                             processor) -> Tuple[str, str, int, str, bool]:
        """Judge a single query-item pair using LLM.
        
        Args:
            query: User search query
            item_id: Item identifier
            item_info: Item metadata
            processor: OpenAIPromptProcessor instance
            
        Returns:
            Tuple of (query, item_id, score, reasoning, success)
        """
        try:
            # Format item information for the judge
            item_text = EcommerceDataLoader.create_text_content(
                item_info, add_category=True, add_tags=True
            )
            
            # Call LLM judge with retry logic
            for attempt in range(self.retry_attempts):
                try:
                    result = processor.judge_relevance(query, item_text)
                    
                    if isinstance(result, dict) and 'score' in result:
                        score = result.get('score', 0)
                        reasoning = result.get('reason', '')
                        
                        # Validate score
                        if score in [0, 1, 2]:
                            return query, item_id, score, reasoning, True
                        else:
                            logger.warning(f"Invalid score {score} for query '{query}', item '{item_id}'")
                    
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for query '{query}', item '{item_id}': {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    continue
            
            logger.error(f"All attempts failed for query '{query}', item '{item_id}'")
            return query, item_id, 0, "Failed to get judgment", False
            
        except Exception as e:
            logger.error(f"Error judging query '{query}', item '{item_id}': {e}")
            return query, item_id, 0, f"Error: {str(e)}", False
    
    def process_pairs_batch(self, pairs_batch: List[Tuple[str, str]], items_db: Dict[str, Dict],
                           processor, pbar: tqdm) -> List[Tuple[str, str, int, str, bool]]:
        """Process a batch of query-item pairs in a single thread.
        
        Args:
            pairs_batch: List of (query, item_id) tuples
            items_db: Items database
            processor: OpenAIPromptProcessor instance
            pbar: Progress bar for updates
            
        Returns:
            List of judgment results
        """
        results = []
        
        for query, item_id in pairs_batch:
            if item_id not in items_db:
                logger.warning(f"Item {item_id} not found in database")
                continue
            
            result = self.judge_query_item_pair(query, item_id, items_db[item_id], processor)
            results.append(result)
            
            # Update progress bar (thread-safe)
            with progress_lock:
                pbar.update(1)
        
        return results
    
    def generate_ground_truth(self, retriever_results: Dict[str, Any], 
                            items_db: Dict[str, Dict]) -> Dict[str, Dict[str, int]]:
        """Generate ground truth using multi-threaded LLM judging.
        
        Args:
            retriever_results: Results from create_retrievers.py
            items_db: Items database
            
        Returns:
            Ground truth in format {query: {item_id: score, ...}}
        """
        start_time = time.time()
        
        # Extract unique query-item pairs
        pairs = list(self.extract_query_item_pairs(retriever_results))
        self.stats['total_pairs'] = len(pairs)
        
        if not pairs:
            logger.warning("No query-item pairs found")
            return {}
        
        # Split pairs into batches for threads
        batch_size = max(1, len(pairs) // self.threads)
        batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
        
        logger.info(f"Processing {len(pairs)} pairs in {len(batches)} batches using {self.threads} threads")
        
        # Initialize progress bar
        pbar = tqdm(total=len(pairs), desc="Judging query-item pairs")
        
        all_results = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                # Create processor instances for each thread
                processors = [self.create_prompt_processor() for _ in range(self.threads)]
                
                # Submit batches to threads
                futures = []
                for i, batch in enumerate(batches):
                    processor = processors[i % len(processors)]
                    future = executor.submit(self.process_pairs_batch, batch, items_db, processor, pbar)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        batch_results = future.result(timeout=self.timeout * len(batches[0]) + 60)
                        all_results.extend(batch_results)
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        traceback.print_exc()
        
        finally:
            pbar.close()
        
        # Process results into ground truth format
        ground_truth = {}
        
        for query, item_id, score, reasoning, success in all_results:
            if success:
                self.stats['successful_judgments'] += 1
                self.stats['score_distribution'][score] += 1
            else:
                self.stats['failed_judgments'] += 1
            
            # Add to ground truth
            if query not in ground_truth:
                ground_truth[query] = {}
            ground_truth[query][item_id] = score
        
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info(f"Ground truth generation completed in {self.stats['processing_time']:.2f} seconds")
        self.log_statistics()
        
        return ground_truth
    
    def log_statistics(self):
        """Log processing statistics."""
        logger.info("=== Ground Truth Generation Statistics ===")
        logger.info(f"Total pairs processed: {self.stats['total_pairs']}")
        logger.info(f"Successful judgments: {self.stats['successful_judgments']}")
        logger.info(f"Failed judgments: {self.stats['failed_judgments']}")
        logger.info(f"Success rate: {self.stats['successful_judgments'] / max(1, self.stats['total_pairs']) * 100:.1f}%")
        logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        logger.info("Score distribution:")
        for score, count in self.stats['score_distribution'].items():
            logger.info(f"  Score {score}: {count} items")


def main():
    parser = argparse.ArgumentParser(description="Create ground truth from retriever results using LLM judge")
    parser.add_argument("retriever_results", help="Path to JSON file from create_retrievers.py")
    parser.add_argument("data_csv", help="Path to CSV file containing item data")
    parser.add_argument("output_json", help="Path to output ground truth JSON file")
    parser.add_argument("--model-type", default="openai", help="LLM model type (default: openai)")
    parser.add_argument("--model-name", default="gpt-4.1-mini", help="LLM model name (default: gpt-4.1-mini)")
    parser.add_argument("--threads", type=int, default=8, help="Number of concurrent threads (default: 8)")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per LLM call in seconds (default: 30)")
    parser.add_argument("--retry-attempts", type=int, default=3, help="Number of retry attempts (default: 3)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.retriever_results).exists():
        raise FileNotFoundError(f"Retriever results file not found: {args.retriever_results}")
    if not Path(args.data_csv).exists():
        raise FileNotFoundError(f"Data CSV not found: {args.data_csv}")
    
    # Load retriever results
    logger.info(f"Loading retriever results from {args.retriever_results}")
    with open(args.retriever_results, 'r', encoding='utf-8') as f:
        retriever_results = json.load(f)
    
    # Load items database
    logger.info(f"Loading items database from {args.data_csv}")
    items_db, _, _, _, _ = EcommerceDataLoader.load_from_csv(
        args.data_csv, images_dir="./data/raw/images"
    )
    
    # Initialize generator
    generator = GroundTruthGenerator(
        model_type=args.model_type,
        model_name=args.model_name,
        threads=args.threads,
        timeout=args.timeout,
        retry_attempts=args.retry_attempts
    )
    
    # Generate ground truth
    logger.info("Starting ground truth generation...")
    ground_truth = generator.generate_ground_truth(retriever_results, items_db)
    
    # Save results
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving ground truth to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Ground truth generation complete!")
    logger.info(f"📊 Generated ground truth for {len(ground_truth)} queries")
    logger.info(f"💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()
