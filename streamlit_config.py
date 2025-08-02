"""
Configuration settings for the Streamlit E-commerce Search Evaluation UI.
"""

# Data file paths
DATA_PATHS = {
    "features_csv": "data/processed/features_val_2_wp.csv",
    "queries_extended": "data/processed/queries_extended_english.csv",
    "items_csv": "data/raw/5k_items_curated.csv",
    "images_dir": "./data/raw/images"
}

# UI Configuration
UI_CONFIG = {
    "items_per_page": 30,
    "image_size": (150, 150),
    "default_score_threshold": 0.0,
    "show_debug_info": False
}

# Score badge colors
SCORE_COLORS = {
    "prediction": "#28a745",  # Green
    "bm25": "#007bff",        # Blue
    "image_clip": "#6f42c1",  # Purple
    "text_embedding": "#fd7e14"  # Orange
}

# Column mappings for different datasets
COLUMN_MAPPINGS = {
    "default": {
        "query": "query",
        "item_id": "item_id",
        "predictions": "predictions",
        "bm25_score": "bm25_score",
        "bm25_rank": "bm25_rank",
        "bm25_hit": "bm25_hit",
        "image_clip_score": "image_clip_score",
        "image_clip_rank": "image_clip_rank",
        "image_clip_hit": "image_clip_hit",
        "text_embedding_score": "text_embedding_score",
        "text_embedding_rank": "text_embedding_rank",
        "text_embedding_hit": "text_embedding_hit"
    }
}

# App metadata
APP_CONFIG = {
    "title": "🛒 E-commerce Search Evaluation Dashboard",
    "description": """
    This dashboard allows you to explore search evaluation results. Select a query to see 
    the ranked items with their retrieval scores from different methods.
    """,
    "page_icon": "🛒",
    "layout": "wide"
}
