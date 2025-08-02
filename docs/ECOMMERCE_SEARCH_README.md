# E-commerce Multi-Modal Search System

This document describes the enhanced search system that supports both text and image embeddings for e-commerce applications.

## 🚀 Features

### Multi-Modal Search Capabilities
- **Text-to-Text Search**: Traditional semantic search using text embeddings
- **Text-to-Image Search**: Find products by searching text queries against product images (using CLIP)
- **Image-to-Image Search**: Find similar products using image embeddings
- **Hybrid Search**: Combine text FAISS, image FAISS, and BM25 for optimal results

### Supported Models
- **Text Embeddings**: SentenceTransformers, HuggingFace, OpenAI
- **Image Embeddings**: CLIP (text+image)
- **Cross-Modal**: CLIP enables text queries to search image indices

### Search Fusion Methods
- **Weighted Sum**: Combine scores with configurable weights
- **Rank Fusion**: Merge results based on ranking positions
- **Reciprocal Rank Fusion (RRF)**: Advanced ranking-based fusion

## 📁 Architecture

```
src/
├── data/
│   └── ecommerce_loader.py          # CSV data loading with image support
├── models/
│   ├── embedding_models.py          # Text embedding models
│   ├── image_models.py              # Image embedding models (NEW)
│   └── model_factory.py             # Enhanced model factory
├── retrievers/
│   ├── base.py                      # Updated with item_id support
│   ├── faiss_retriever.py           # Enhanced with image support
│   ├── bm25_retriever.py            # Updated with item_id support
│   └── multimodal_hybrid_retriever.py  # NEW: Multi-modal fusion
└── config/
    └── models/embedding_models.yaml  # Updated with image models
```

## 🛠️ Installation

### Required Dependencies
```bash
# Core dependencies
pip install numpy pandas faiss-cpu sentence-transformers

# For image support
pip install torch torchvision transformers Pillow

# For BM25
pip install rank-bm25
```

### Optional GPU Support
```bash
pip install faiss-gpu  # Instead of faiss-cpu
```

## 📊 Data Format

### CSV Structure
Your CSV should have these columns:
- `_id`: Document ID
- `itemId`: Unique item identifier
- `itemMetadata`: JSON string containing item data

### Item Metadata Structure
```json
{
  "category_name": "Alimentos Básicos",
  "description": "Pacote 500g", 
  "name": "Macarrão Pena com Ovos Adria 500g",
  "price": 3.56,
  "images": [
    "820af392-002c-47b1-bfae-d7ef31743c7f/202210182253_3h93mu9eg9y.jpg"
  ],
  "taxonomy": {
    "l0": "MERCEARIA",
    "l1": "MASSAS_SECAS", 
    "l2": "MASSA_MACARRAO"
  },
  "tags": [{"key": "PORTION_SIZE", "value": ["NOT_APPLICABLE"]}]
}
```

### Image Storage
- Images should be stored in `data/raw/images/`
- Image paths in metadata use `/` separator
- Actual filenames use `_` separator (automatic conversion)
- Example: `"path/to/image.jpg"` → `"path_to_image.jpg"`

## 🔍 Usage Examples

### 1. Basic Text Search
```python
from src.data.ecommerce_loader import EcommerceDataLoader
from src.models.model_factory import ModelFactory
from src.retrievers.faiss_retriever import FaissRetriever

# Load data
items_db, text_docs, image_paths, item_ids = EcommerceDataLoader.load_from_csv("data.csv")

# Create model and retriever
model = ModelFactory.create_model("sentence_transformers", "all-MiniLM-L6-v2")
retriever = FaissRetriever(model, items_db)

# Index and search
text_docs, text_item_ids = EcommerceDataLoader.get_text_index_data(items_db)
retriever.create_text_index(text_docs, text_item_ids)

results = retriever.search("pasta with eggs", k=5)
for result in results:
    item = items_db[result.item_id]
    print(f"{item['name']} - Score: {result.score:.4f}")
```

### 2. Cross-Modal Search (Text → Images)
```python
# Create CLIP model for cross-modal search
clip_model = ModelFactory.create_model("clip", "openai-clip-vit-base-patch32")
image_retriever = FaissRetriever(clip_model, items_db)

# Index images
img_paths, img_item_ids = EcommerceDataLoader.get_image_index_data(items_db)
image_retriever.create_image_index(img_paths, img_item_ids)

# Search images with text query
results = image_retriever.search("red pasta sauce", k=5)
```

### 3. Multi-Modal Hybrid Search
```python
from src.retrievers.multimodal_hybrid_retriever import MultiModalHybridRetriever

# Create hybrid retriever
hybrid = MultiModalHybridRetriever(
    text_embedding_model=text_model,
    image_embedding_model=clip_model,
    items_db=items_db,
    config={
        "text_weight": 0.4,
        "image_weight": 0.4, 
        "bm25_weight": 0.2,
        "fusion_method": "weighted_sum"
    }
)

# Index all modalities
hybrid.create_index(text_docs, image_paths, item_ids)

# Search combines all approaches
results = hybrid.search("organic pasta", k=10)
```

### 4. Search with Pre-computed Embeddings
```python
# For offline embedding computation
query_embedding = model.encode(["search query"])
results = retriever.search_with_embedding(query_embedding, k=5)
```

## ⚙️ Configuration

### Model Configuration
Add new models to `config/models/embedding_models.yaml`:

```yaml
clip:
  openai-clip-vit-base-patch32:
    type: "clip"
    model_name: "openai/clip-vit-base-patch32"
    dimension: 512
    supports_text: true
    supports_images: true

resnet:
  resnet50:
    type: "resnet" 
    model_name: "resnet50"
    dimension: 2048
    supports_images: true
```

### Fusion Configuration
```python
config = {
    "text_weight": 0.4,      # Weight for text FAISS
    "image_weight": 0.4,     # Weight for image FAISS  
    "bm25_weight": 0.2,      # Weight for BM25
    "fusion_method": "weighted_sum",  # or "rank_fusion", "rrf"
    "normalize_scores": True,
    "rrf_constant": 60       # For RRF fusion
}
```

## 🎯 Search Strategies

### When to Use Each Approach

1. **Text-Only FAISS**: 
   - Best for semantic similarity
   - Good for synonyms and related concepts
   - Fast and efficient

2. **Image-Only FAISS**:
   - Visual similarity search
   - Cross-modal text→image search with CLIP
   - Good for visual product discovery

3. **BM25**:
   - Exact keyword matching
   - Good for specific product names/brands
   - Handles rare terms well

4. **Hybrid Multi-Modal**:
   - Best overall performance
   - Combines strengths of all approaches
   - Configurable for different use cases

### Fusion Method Selection

- **Weighted Sum**: Best when you have good score calibration
- **Rank Fusion**: More robust to score differences
- **RRF**: Good general-purpose fusion method

## 📈 Performance Tips

### Indexing
- Use batch processing for large datasets
- Consider IVF indices for >10k items
- Pre-compute embeddings offline when possible

### Search
- Cache frequently used embeddings
- Use appropriate k values (don't over-retrieve)
- Consider score thresholds for quality

### Memory Optimization
- Use float16 for embeddings if precision allows
- Implement embedding compression for large indices
- Consider approximate search for very large datasets

## 🔧 Extending the System

### Adding New Models
1. Create model class inheriting from `BaseImageEmbeddingModel`
2. Register in `ModelFactory._model_registry`
3. Add configuration to YAML file

### Custom Fusion Methods
1. Extend `MultiModalHybridRetriever._fuse_results()`
2. Add new fusion method to configuration options

### New Data Sources
1. Extend `EcommerceDataLoader` for different formats
2. Implement custom preprocessing pipelines

## 🚀 Quick Start

Run the example script:
```bash
python examples/ecommerce_search_example.py
```

This will:
1. Create sample data if needed
2. Demonstrate all search approaches
3. Show performance comparisons
4. Save indices for future use

## 📝 API Reference

### Key Classes

- `EcommerceDataLoader`: Load and preprocess e-commerce data
- `FaissRetriever`: Vector search with text/image support
- `BM25Retriever`: Lexical search with item ID support
- `MultiModalHybridRetriever`: Multi-modal fusion search
- `ModelFactory`: Create embedding models dynamically

### Key Methods

- `create_text_index()`: Index text documents
- `create_image_index()`: Index images
- `search()`: Text query search
- `search_with_embedding()`: Pre-computed embedding search
- `save_index()` / `load_index()`: Persistence

## 🎉 Results

The system returns `RetrievalResult` objects with:
- `item_id`: Unique item identifier
- `score`: Relevance score
- `metadata`: Additional information (scores, ranks, etc.)

Access full item data via: `items_db[result.item_id]`
