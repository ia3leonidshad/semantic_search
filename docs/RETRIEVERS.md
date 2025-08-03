# Retrieval Experiments

A configurable Python framework for experimenting with different retrieval methods including vector search (Faiss), lexical search (BM25), and hybrid approaches.

## Features

- **Configurable Models**: Easy switching between different embedding models
- **Multiple Retrieval Methods**: Faiss vector search, BM25 lexical search, and hybrid fusion
- **Extensible Architecture**: Plugin-style system for adding new retrievers and models
- **Configuration-Driven**: YAML-based configuration for reproducible experiments

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models.model_factory import ModelFactory
from src.retrievers.faiss_retriever import FaissRetriever

# Load a model
model = ModelFactory.create_model("sentence_transformers", "all-MiniLM-L6-v2")

# Create a retriever
retriever = FaissRetriever(model)

# Index documents
items_db, text_documents, text_item_ids, image_paths, image_item_ids = EcommerceDataLoader.load_from_csv(
        args.data_csv, images_dir="./data/raw/images"
)
retriever.create_text_index(text_documents, text_item_ids)

# Search
results = retriever.search("query text", k=5)
```

## Configuration

Models and retrievers are configured via YAML files in the `config/` directory.

## Extending

- **Add new models**: Implement the model interface and register in `config/models/embedding_models.yaml`
- **Add new retrievers**: Inherit from `BaseRetriever` and implement required methods

## License

MIT License
