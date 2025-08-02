#!/usr/bin/env python3
"""
Test script to verify data loading for the Streamlit application.
This script tests the data loading functions without running the full Streamlit app.
"""

import pandas as pd
from pathlib import Path
import sys
import logging

# Add src to path for imports
sys.path.append('.')

from src.data.ecommerce_loader import EcommerceDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_features_data():
    """Test loading the features validation dataset."""
    print("🔍 Testing features data loading...")
    
    features_path = "data/processed/features_val.csv"
    if not Path(features_path).exists():
        print(f"❌ Features file not found: {features_path}")
        return False
    
    try:
        df = pd.read_csv(features_path)
        print(f"✅ Loaded features data with {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['query', 'item_id', 'predictions', 'bm25_score', 'bm25_rank', 
                        'image_clip_score', 'image_clip_rank', 'text_embedding_score', 
                        'text_embedding_rank', 'bm25_hit', 'image_clip_hit', 'text_embedding_hit']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
        else:
            print("✅ All required columns present")
        
        # Show sample data
        print(f"   Unique queries: {df['query'].nunique()}")
        print(f"   Sample queries: {list(df['query'].unique()[:3])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading features data: {e}")
        return False

def test_items_database():
    """Test loading the items database."""
    print("\n📦 Testing items database loading...")
    
    csv_path = "data/raw/5k_items_curated.csv"
    images_dir = "./data/raw/images"
    
    if not Path(csv_path).exists():
        print(f"❌ Items CSV not found: {csv_path}")
        return False
    
    if not Path(images_dir).exists():
        print(f"⚠️  Images directory not found: {images_dir}")
    
    try:
        items_db, text_documents, text_item_ids, image_paths, image_item_ids = EcommerceDataLoader.load_from_csv(
            csv_path, images_dir=images_dir
        )
        
        print(f"✅ Loaded {len(items_db)} items from database")
        print(f"   Text documents: {len(text_documents)}")
        print(f"   Image paths: {len(image_paths)}")
        
        # Show sample item
        if items_db:
            sample_item_id = list(items_db.keys())[0]
            sample_item = items_db[sample_item_id]
            print(f"   Sample item: {sample_item.get('name', 'Unknown')}")
            print(f"   Sample category: {sample_item.get('category_name', 'Unknown')}")
            print(f"   Sample images: {len(sample_item.get('images', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading items database: {e}")
        return False

def test_data_integration():
    """Test integration between features data and items database."""
    print("\n🔗 Testing data integration...")
    
    try:
        # Load both datasets
        features_df = pd.read_csv("data/processed/features_val.csv")
        items_db, _, _, _, _ = EcommerceDataLoader.load_from_csv(
            "data/raw/5k_items_curated.csv", images_dir="./data/raw/images"
        )
        
        # Check item_id overlap
        features_item_ids = set(features_df['item_id'].unique())
        db_item_ids = set(items_db.keys())
        
        overlap = features_item_ids.intersection(db_item_ids)
        missing_in_db = features_item_ids - db_item_ids
        
        print(f"   Features dataset items: {len(features_item_ids)}")
        print(f"   Items database items: {len(db_item_ids)}")
        print(f"   Overlapping items: {len(overlap)}")
        
        if missing_in_db:
            print(f"⚠️  Items in features but not in database: {len(missing_in_db)}")
            print(f"   Sample missing: {list(missing_in_db)[:3]}")
        else:
            print("✅ All feature items found in database")
        
        # Test a sample query
        sample_query = features_df['query'].iloc[0]
        query_results = features_df[features_df['query'] == sample_query]
        print(f"   Sample query: '{sample_query}'")
        print(f"   Results for sample query: {len(query_results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing data integration: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Streamlit Data Loading")
    print("=" * 50)
    
    tests = [
        test_features_data,
        test_items_database,
        test_data_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n📊 Test Summary")
    print("-" * 20)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! The Streamlit app should work correctly.")
        print("\nTo run the Streamlit app:")
        print("1. Install dependencies: pip install streamlit pillow")
        print("2. Run: streamlit run streamlit_app.py")
    else:
        print("❌ Some tests failed. Please check the data files and paths.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
