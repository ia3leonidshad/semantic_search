"""Global settings for retrieval experiments."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDICES_DIR = DATA_DIR / "indices"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Configuration directories
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_CONFIG_DIR = CONFIG_DIR / "models"

# Default model settings
DEFAULT_EMBEDDING_MODEL = "sentence_transformers"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Retrieval settings
DEFAULT_TOP_K = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.0

# Hybrid retrieval settings
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_BM25_WEIGHT = 0.3

# Cache settings
ENABLE_MODEL_CACHE = True
ENABLE_EMBEDDING_CACHE = True
CACHE_DIR = PROJECT_ROOT / ".cache"

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        INDICES_DIR,
        EMBEDDINGS_DIR,
        CACHE_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
