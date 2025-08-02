#!/usr/bin/env python3
"""
Test script to demonstrate the new sentence-transformers CLIP integration.
This script shows how to use CLIP models for both text and image encoding.
"""

import numpy as np
from sentence_transformers import util
from src.models.model_factory import ModelFactory

def test_clip_model():
    """Test the sentence-transformers CLIP model integration."""
    
    print("Testing sentence-transformers CLIP integration...")
    print("=" * 50)
    
    # Create a CLIP model using the model factory
    try:
        model = ModelFactory.create_model(
            model_type="sentence_transformers",
            model_name="sentence-transformers-clip-ViT-B-32-multilingual-v1"
        )
        print(f"✓ Successfully loaded model: {model.model_name}, {model.image_model_name}")
        print(f"  - Dimension: {model.dimension}")
        print(f"  - Supports text: {model.supports_text}")
        print(f"  - Supports images: {model.supports_images}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Test text encoding
    print("Testing text encoding...")
    try:
        texts = [
            "Two dogs in the snow",
            "A cat on a table", 
            "A picture of London at night",
            "A beautiful sunset over the ocean"
        ]
        
        text_embeddings = model.encode(texts, input_type="text")
        print(f"✓ Encoded {len(texts)} texts")
        print(f"  - Shape: {text_embeddings.shape}")
        print(f"  - Sample text: '{texts[0]}'")
        print(f"  - Embedding norm: {np.linalg.norm(text_embeddings[0]):.4f}")
        print()
        
    except Exception as e:
        print(f"✗ Text encoding failed: {e}")
        return
    
    # Test auto-detection
    print("Testing auto-detection...")
    try:
        # This should be detected as text
        auto_text_emb = model.encode("A beautiful landscape")
        print(f"✓ Auto-detected text input")
        print(f"  - Shape: {auto_text_emb.shape}")
        print()
        
    except Exception as e:
        print(f"✗ Auto-detection failed: {e}")
    
    # Test similarity computation
    print("Testing similarity computation...")
    try:
        # Compute similarities between texts
        similarities = util.cos_sim(text_embeddings, text_embeddings)
        print(f"✓ Computed similarity matrix")
        print(f"  - Shape: {similarities.shape}")
        print(f"  - Self-similarity (should be ~1.0): {similarities[0][0]:.4f}")
        print(f"  - Cross-similarity example: {similarities[0][1]:.4f}")
        print()
        
    except Exception as e:
        print(f"✗ Similarity computation failed: {e}")
    
    # Show available CLIP models
    print("Available CLIP models in configuration:")
    try:
        all_models = ModelFactory.list_configured_models()
        clip_models = {k: v for k, v in all_models.items() 
                      if 'clip' in k.lower() and v['type'] == 'sentence_transformers'}
        
        for model_key, info in clip_models.items():
            print(f"  - {model_key}")
            print(f"    Description: {info['description']}")
            print(f"    Dimension: {info['dimension']}")
            print()
            
    except Exception as e:
        print(f"✗ Failed to list models: {e}")
    
    print("=" * 50)
    print("CLIP integration test completed!")
    print("\nUsage example:")
    print("```python")
    print("from src.models.model_factory import ModelFactory")
    print("from sentence_transformers import util")
    print("from PIL import Image")
    print("")
    print("# Load CLIP model")
    print('model = ModelFactory.create_model("sentence_transformers", "clip-ViT-B-32")')
    print("")
    print("# Encode text")
    print('text_emb = model.encode([\"Two dogs in snow\"], input_type=\"text\")')
    print("")
    print("# Encode images (when you have image files)")
    print('# img_emb = model.encode([\"path/to/image.jpg\"], input_type=\"image\")')
    print("")
    print("# Compute similarity")
    print("# cos_scores = util.cos_sim(img_emb, text_emb)")
    print("```")

if __name__ == "__main__":
    test_clip_model()
