#!/usr/bin/env python3
"""
Image Download Script

Downloads images from URLs stored in CSV metadata to a specified destination folder.
"""

import argparse
import pandas as pd
import json
import logging
import os
import requests
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_files(csv_file: Path) -> None:
    """Validate that required files exist."""
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")


def download_single_image(url: str, dest_folder: str) -> tuple[bool, str]:
    """
    Download a single image from URL.
    
    Returns:
        tuple: (success: bool, error_message: str)
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Extract filename from URL
        path = urlparse(url).path
        filename = '_'.join(path.split('/')[-2:])
        file_path = os.path.join(dest_folder, filename)
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        
        return True, ""
    
    except requests.RequestException as e:
        return False, str(e)


def download_images(urls: list, dest_folder: str, num_threads: int = 8) -> None:
    """
    Download images from a list of URLs into dest_folder using multithreading.
    
    Args:
        urls: List of image URLs to download
        dest_folder: Destination folder for downloaded images
        num_threads: Number of threads to use for downloading
    """
    os.makedirs(dest_folder, exist_ok=True)
    
    failed_downloads = 0
    successful_downloads = 0
    lock = Lock()
    
    def update_counters(success: bool):
        nonlocal successful_downloads, failed_downloads
        with lock:
            if success:
                successful_downloads += 1
            else:
                failed_downloads += 1
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_single_image, url, dest_folder): url 
            for url in urls
        }
        
        # Process completed downloads with progress bar
        with tqdm(total=len(urls), desc="Downloading images") as pbar:
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    success, error_msg = future.result()
                    update_counters(success)
                    
                    if not success:
                        logging.error(f"Failed to download {url}: {error_msg}")
                
                except Exception as e:
                    update_counters(False)
                    logging.error(f"Unexpected error downloading {url}: {e}")
                
                pbar.update(1)
    
    logging.info(f"Download completed: {successful_downloads} successful, {failed_downloads} failed")


def process_csv_and_download(csv_file: Path, dest_folder: str, num_threads: int = 8) -> None:
    """Process CSV file and download images."""
    try:
        # Read CSV file
        data = pd.read_csv(csv_file)
        logging.info(f"Loaded {len(data)} rows from {csv_file}")
        
        # Check if itemMetadata column exists
        if 'itemMetadata' not in data.columns:
            raise ValueError("CSV file must contain 'itemMetadata' column")
        
        # Extract URLs from metadata
        url_list = []
        for i, row in data.iterrows():
            try:
                meta_data = json.loads(row.itemMetadata)
                if 'images' in meta_data:
                    urls = [f'https://static.ifood-static.com.br/image/upload/t_low/pratos/{url}' 
                           for url in meta_data["images"]]
                    url_list.extend(urls)
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Failed to parse metadata for row {i}: {e}")
        
        logging.info(f"Found {len(url_list)} image URLs to download")
        logging.info(f"Using {num_threads} threads for downloading")
        
        if not url_list:
            logging.warning("No image URLs found in the CSV file")
            return
        
        # Download images
        download_images(url_list, dest_folder, num_threads)
        
    except Exception as e:
        raise RuntimeError(f"Failed to process CSV and download images: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download images from CSV metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses 8 threads by default)
  python scripts/download_images.py -i data/raw/5k_items_curated.csv -o data/raw/images/
  
  # With custom number of threads
  python scripts/download_images.py -i data/raw/5k_items_curated.csv -o data/raw/images/ --threads 16
  
  # With verbose logging and custom threads
  python scripts/download_images.py -i data/raw/5k_items_curated.csv -o data/raw/images/ --threads 4 --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--csv-input', '-i',
        type=Path,
        required=True,
        help='Input CSV file containing itemMetadata with image URLs'
    )
    
    parser.add_argument(
        '--dest-folder', '-o',
        type=str,
        required=True,
        help='Destination folder for downloaded images'
    )
    
    # Optional arguments
    parser.add_argument(
        '--threads', '-t',
        type=int,
        default=8,
        help='Number of threads to use for downloading (default: 8)'
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
        validate_files(args.csv_input)
        
        logging.info(f"Starting image download from {args.csv_input} to {args.dest_folder}")
        
        # Process CSV and download images
        process_csv_and_download(args.csv_input, args.dest_folder, args.threads)
        
        logging.info("Image download completed successfully!")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
