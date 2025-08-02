#!/usr/bin/env python3
"""
E-commerce Multi-Modal Search Example

This script demonstrates how to use the enhanced retrieval framework for e-commerce search
with support for text and image embeddings.

Features demonstrated:
1. Load e-commerce data from CSV
2. Create text and image indices
3. Perform multi-modal hybrid search
4. Compare different search approaches
"""

import logging
from pathlib import Path

from src.data.ecommerce_loader import EcommerceDataLoader
from src.models.model_factory import ModelFactory
from src.retrievers.faiss_retriever import FaissRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.multimodal_hybrid_retriever import MultiModalHybridRetriever
from config.settings import ensure_directories

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    
    # Ensure data directories exist
    ensure_directories()
    
    print("🛒 E-commerce Multi-Modal Search Example")
    print("=" * 50)
    
    # Step 1: Load e-commerce data
    print("\n📄 Loading e-commerce data...")
    
    # For this example, we'll create some sample data if CSV doesn't exist
    csv_path = "data/raw/5k_items_curated.csv"
    if not Path(csv_path).exists():
        raise ValueError("Sample CSV not found. Creating sample data...")
    
    try:
        items_db, text_documents, text_item_ids, image_paths, image_item_ids = EcommerceDataLoader.load_from_csv(
            csv_path, images_dir="data/raw/images"
        )
        
        print(f"✅ Loaded {len(items_db)} items")
        print(f"✅ Loaded unique {len(set(text_item_ids))}, {len(set(image_item_ids))} items")
        print(f"   - Text documents: {len(text_documents)}")
        print(f"   - Images found: {len(image_paths)}")
        
        print(text_documents[0])

        # Show sample item
        if items_db:
            sample_item_id = list(items_db.keys())[0]
            sample_item = items_db[sample_item_id]
            print(f"\n📦 Sample item: {sample_item.get('name', 'Unknown')}")
            print(f"   Category: {sample_item.get('category_name', 'Unknown')}")
            print(f"   Taxonomy: {sample_item.get('taxonomy', 'Unknown')}")
            print(f"   Price: ${sample_item.get('price', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Please ensure you have a CSV file with the correct format.")
        return
    
    # Step 2: Create embedding models
    print("\n🤖 Creating embedding models...")
    
    open_ai_embedding_model = ModelFactory.create_model("openai", "text-embedding-3-small")

    print(open_ai_embedding_model.encode('Chicken salad'))

    try:
        # Text embedding model
        text_model = ModelFactory.create_model("sentence_transformers", "paraphrase-multilingual-mpnet-base-v2")
        print(f"✅ Text model: {text_model.model_name} (dimension: {text_model.dimension})")
        
        # Image embedding model (CLIP for cross-modal search)
        image_model = ModelFactory.create_model("clip", "openai-clip-vit-base-patch32")
        # image_model = ModelFactory.create_model(
        #     model_type="sentence_transformers",
        #     model_name="sentence-transformers-clip-ViT-B-32-multilingual-v1"
        # )
        print(f"✅ Image model: {image_model.image_model_name} (dimension: {image_model.dimension})")
        
    except Exception as e:
        print(f"❌ Error creating models: {e}")
        print("Make sure required packages are installed:")
        print("  pip install sentence-transformers transformers torch")
        return
    
    # Step 3: Demonstrate different search approaches
    
    # 3a. Text-only search with FAISS
    print("\n\n🔤 Text-Only Vector Search (FAISS)")
    print("-" * 40)
    
    text_retriever = FaissRetriever(text_model, items_db, config={
        "similarity_metric": "cosine",
        "index_type": "flat"
    })
    
    # Get text data for indexing
    text_docs, text_item_ids = EcommerceDataLoader.get_text_index_data(items_db, True, True, True)
    
    print("Indexing text documents...")
    text_retriever.create_text_index(text_docs, text_item_ids, show_progress=False)
    
    # Search
    query = "pasta macarrão ovos"
    print(f"\nSearching for: '{query}'")
    text_results = text_retriever.search(query, k=3)
    
    print(f"Found {len(text_results)} results:")
    for i, result in enumerate(text_results, 1):
        item_data = items_db[result.item_id]
        print(f"\n{i}. {item_data.get('name', 'Unknown')} (Score: {result.score:.4f})")
        print(f"   Category: {item_data.get('category_name', 'Unknown')}")
        print(f"   Taxonomy: {item_data.get('taxonomy', 'Unknown')}")
        print(f"   Price: ${item_data.get('price', 0):.2f}")
    
    # query = "chicken salad"
    # 3b. Image-only search (if images available)
    if image_paths:
        print("\n\n🖼️  Image-Only Vector Search (CLIP)")
        print("-" * 40)
        
        image_retriever = FaissRetriever(image_model, items_db, config={
            "similarity_metric": "cosine",
            "index_type": "flat"
        })
        
        # Get image data for indexing
        img_paths, img_item_ids = EcommerceDataLoader.get_image_index_data(items_db)
        
        if img_paths:
            print(f"Indexing {len(img_paths)} images...")
            image_retriever.create_image_index(img_paths, img_item_ids, show_progress=False)
            
            # Search with text query (cross-modal)
            print(f"Cross-modal search for: '{query}'")
            image_results = image_retriever.search(query, k=3)
            
            print(f"Found {len(image_results)} results:")
            for i, result in enumerate(image_results, 1):
                item_data = items_db[result.item_id]
                print(f"\n{i}. {item_data.get('name', 'Unknown')} (Score: {result.score:.4f})")
                print(f"   Category: {item_data.get('category_name', 'Unknown')}")
                print(f"   Images: {len(item_data.get('images', []))}")
        else:
            print("No images found for indexing")
    
    # 3c. BM25 lexical search
    print("\n\n📝 Lexical Search (BM25)")
    print("-" * 30)
    
    bm25_retriever = BM25Retriever(config={
        "k1": 1.2,
        "b": 0.75,
        "lowercase": True,
        "remove_punctuation": True
    })
    text_docs, text_item_ids = EcommerceDataLoader.get_text_index_data(items_db, True, False, False)

    print("Indexing documents with BM25...")
    bm25_retriever.create_index(text_docs, text_item_ids)
    
    print(f"Searching for: '{query}'")
    bm25_results = bm25_retriever.search(query, k=3)
    
    print(f"Found {len(bm25_results)} results:")
    for i, result in enumerate(bm25_results, 1):
        item_data = items_db[result.item_id]
        print(f"\n{i}. {item_data.get('name', 'Unknown')} (Score: {result.score:.4f})")
        print(f"   Category: {item_data.get('category_name', 'Unknown')}")
        print(f"   Taxonomy: {item_data.get('taxonomy', 'Unknown')}")
        print(f"   Matched tokens: {result.metadata.get('query_tokens', [])}")
    return

    # 3d. Multi-modal hybrid search
    print("\n\n🔀 Multi-Modal Hybrid Search")
    print("-" * 35)
    
    hybrid_retriever = MultiModalHybridRetriever(
        text_embedding_model=text_model,
        image_embedding_model=image_model,
        items_db=items_db,
        config={
            "text_weight": 0.4,
            "image_weight": 0.4,
            "bm25_weight": 0.2,
            "fusion_method": "weighted_sum",
            "normalize_scores": True
        }
    )
    
    print("Creating multi-modal hybrid index...")
    
    # Prepare data for hybrid indexing
    # For images, we need to align with text documents
    aligned_image_paths = []
    for item_id in text_item_ids:
        item_images = EcommerceDataLoader.get_item_image_paths(
            items_db[item_id], Path("data/raw/images")
        )
        # Take first image or empty string if no images
        aligned_image_paths.append(item_images[0] if item_images else "")
    
    hybrid_retriever.create_index(
        text_documents=text_docs,
        image_paths=aligned_image_paths,
        item_ids=text_item_ids,
        show_progress=False
    )
    
    print(f"Searching for: '{query}'")
    hybrid_results = hybrid_retriever.search(query, k=5)
    
    print(f"Found {len(hybrid_results)} results:")
    for i, result in enumerate(hybrid_results, 1):
        item_data = items_db[result.item_id]
        print(f"\n{i}. {item_data.get('name', 'Unknown')} (Score: {result.score:.4f})")
        print(f"   Category: {item_data.get('category_name', 'Unknown')}")
        print(f"   Text Score: {result.metadata.get('text_score', 0):.4f}")
        print(f"   Image Score: {result.metadata.get('image_score', 0):.4f}")
        print(f"   BM25 Score: {result.metadata.get('bm25_score', 0):.4f}")
    
    # Step 4: Save indices for future use
    print("\n\n💾 Saving Indices")
    print("-" * 20)
    
    try:
        text_retriever.save_index("data/indices/ecommerce_text")
        bm25_retriever.save_index("data/indices/ecommerce_bm25")
        hybrid_retriever.save_index("data/indices/ecommerce_hybrid")
        print("✅ All indices saved successfully!")
    except Exception as e:
        print(f"❌ Error saving indices: {e}")
    
    # Step 5: Show statistics
    print("\n\n📊 Search Statistics")
    print("-" * 25)
    
    print("\nItems Database Stats:")
    stats = EcommerceDataLoader.get_item_stats(items_db)
    for key, value in stats.items():
        if key not in ["categories", "taxonomy_levels"]:
            print(f"  {key}: {value}")
    
    print("\nHybrid Retriever Stats:")
    hybrid_stats = hybrid_retriever.get_stats()
    for key, value in hybrid_stats.items():
        if key not in ["config", "text_faiss_stats", "image_faiss_stats", "bm25_stats"]:
            print(f"  {key}: {value}")
    
    print("\n✨ Example completed successfully!")
    print("\nNext steps:")
    print("- Try different queries to test search quality")
    print("- Experiment with different fusion weights")
    print("- Add your own e-commerce CSV data")
    print("- Test with actual product images")


if __name__ == "__main__":
    main()
