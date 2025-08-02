#!/usr/bin/env python3
"""
Example demonstrating CLIP usage similar to the original code snippet.
This shows how to use the sentence-transformers CLIP integration for 
text-image similarity computation.
"""

from sentence_transformers import util
from PIL import Image
from src.models.model_factory import ModelFactory
import numpy as np

def main():
    """Demonstrate CLIP usage for text-image similarity."""
    
    print("CLIP Text-Image Similarity Example")
    print("=" * 40)
    
    # Load CLIP model using the model factory
    model = ModelFactory.create_model(
        model_type="sentence_transformers",
        model_name="sentence-transformers-clip-ViT-B-32"
    )
    
    print(f"Loaded model: {model.model_name}")
    print(f"Dimension: {model.dimension}")
    print()
    
    # Example 1: Text-only similarity (like your original snippet)
    print("Example 1: Text encoding and similarity")
    print("-" * 40)
    
    # Encode text descriptions
    texts = [
        'Two dogs in the snow', 
        'A cat on a table', 
        'A picture of London at night'
    ]
    
    text_embeddings = model.encode(texts, input_type="text")
    print(f"Encoded {len(texts)} text descriptions")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Compute similarities between texts
    text_similarities = util.cos_sim(text_embeddings, text_embeddings)
    print(f"Text similarity matrix shape: {text_similarities.shape}")
    print()
    
    # Show similarity scores
    for i, text1 in enumerate(texts):
        for j, text2 in enumerate(texts):
            if i <= j:  # Only show upper triangle
                score = text_similarities[i][j].item()
                print(f"'{text1}' <-> '{text2}': {score:.4f}")
    print()
    
    # Example 2: Image encoding (when you have actual image files)
    print("Example 2: Image encoding capability")
    print("-" * 40)
    print("Note: This would work with actual image files:")
    print("# img_emb = model.encode(['path/to/image.jpg'], input_type='image')")
    print("# cos_scores = util.cos_sim(img_emb, text_embeddings)")
    print()
    
    # Example 3: Auto-detection
    print("Example 3: Auto-detection of input type")
    print("-" * 40)
    
    # Auto-detect text
    auto_text = model.encode("A beautiful sunset")
    print(f"Auto-detected text input shape: {auto_text.shape}")
    
    # This would auto-detect as image if file exists:
    # auto_image = model.encode("image.jpg")  # if file exists
    print()
    
    # Example 4: Batch processing
    print("Example 4: Batch processing")
    print("-" * 40)
    
    large_text_batch = [
        "A dog running in a park",
        "A cat sleeping on a sofa", 
        "Mountains covered in snow",
        "A busy city street at night",
        "Children playing in a garden"
    ]
    
    batch_embeddings = model.encode(large_text_batch, input_type="text")
    print(f"Batch encoded {len(large_text_batch)} texts")
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    
    # Find most similar pair
    batch_similarities = util.cos_sim(batch_embeddings, batch_embeddings)
    max_sim = 0
    best_pair = (0, 0)
    
    for i in range(len(large_text_batch)):
        for j in range(i+1, len(large_text_batch)):
            sim = batch_similarities[i][j].item()
            if sim > max_sim:
                max_sim = sim
                best_pair = (i, j)
    
    print(f"Most similar texts:")
    print(f"  '{large_text_batch[best_pair[0]]}'")
    print(f"  '{large_text_batch[best_pair[1]]}'")
    print(f"  Similarity: {max_sim:.4f}")
    print()
    
    print("Integration complete! The sentence-transformers CLIP model")
    print("is now available through your existing model factory system.")

if __name__ == "__main__":
    main()
