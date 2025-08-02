#!/usr/bin/env python3
"""
Streamlit UI for E-commerce Search Evaluation

This application provides a user interface to explore search results from the evaluation dataset,
allowing users to select queries and view ranked results with item details and retrieval scores.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional

# Import project modules
from src.data.ecommerce_loader import EcommerceDataLoader
from streamlit_config import DATA_PATHS, UI_CONFIG, SCORE_COLORS, APP_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_features_data() -> pd.DataFrame:
    """Load the features validation dataset."""
    features_path = DATA_PATHS["features_csv"]
    if not Path(features_path).exists():
        st.error(f"Features file not found: {features_path}")
        st.stop()
    
    try:
        df = pd.read_csv(features_path)
        logger.info(f"Loaded features data with {len(df)} rows")
        return df
    except Exception as e:
        st.error(f"Error loading features data: {e}")
        st.stop()

@st.cache_data
def load_items_database() -> Tuple[Dict, List[str], List[str], List[str], List[str]]:
    """Load the items database using EcommerceDataLoader."""
    csv_path = DATA_PATHS["items_csv"]
    images_dir = DATA_PATHS["images_dir"]
    
    if not Path(csv_path).exists():
        st.error(f"Items CSV not found: {csv_path}")
        st.stop()
    
    try:
        items_db, text_documents, text_item_ids, image_paths, image_item_ids = EcommerceDataLoader.load_from_csv(
            csv_path, images_dir=images_dir
        )
        logger.info(f"Loaded {len(items_db)} items from database")
        return items_db, text_documents, text_item_ids, image_paths, image_item_ids
    except Exception as e:
        st.error(f"Error loading items database: {e}")
        st.stop()

def get_item_image_path(item_data: Dict, images_dir: str = None) -> Optional[str]:
    """Get the first available image path for an item."""
    if images_dir is None:
        images_dir = DATA_PATHS["images_dir"]
    
    images = item_data.get('images', [])
    if not images:
        return None
    
    images_dir = Path(images_dir)
    for image_name in images:
        if not image_name:
            continue
        # Convert path to filename: replace '/' with '_'
        filename = image_name.replace('/', '_')
        image_path = images_dir / filename
        if image_path.exists():
            return str(image_path)
    
    return None

def display_score_badge(score: float, rank: int, label: str, color: str = "blue"):
    """Display a score badge with rank information."""
    st.markdown(f"""
    <div style="
        background-color: {color}; 
        color: white; 
        padding: 4px 8px; 
        border-radius: 4px; 
        margin: 2px; 
        text-align: center;
        font-size: 12px;
    ">
        <strong>{label}</strong><br>
        Score: {score:.3f}<br>
        Rank: #{rank}
    </div>
    """, unsafe_allow_html=True)

def display_item_card(item_data: Dict, scores_data: Dict, images_dir: str = None):
    """Display an item card with image, details, and scores."""
    
    if images_dir is None:
        images_dir = DATA_PATHS["images_dir"]
    
    # Get item image
    image_path = get_item_image_path(item_data, images_dir)
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display image
        if image_path and Path(image_path).exists():
            try:
                image = Image.open(image_path)
                # Resize image to consistent size
                image_size = UI_CONFIG["image_size"]
                image = image.resize(image_size, Image.Resampling.LANCZOS)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.write("🖼️ Image not available")
                logger.warning(f"Error loading image {image_path}: {e}")
        else:
            st.write("🖼️ No image available")
    
    with col2:
        # Item details
        st.subheader(item_data.get('name', 'Unknown Item'))
        st.write(f"**Category:** {item_data.get('category_name', 'Unknown')}")
        
        # Price
        price = item_data.get('price', 0)
        if price > 0:
            st.write(f"**Price:** ${price:.2f}")
        
        # Description
        description = item_data.get('description', '')
        if description:
            st.write(f"**Description:** {description}")
        
        # Taxonomy
        taxonomy = item_data.get('taxonomy', {})
        if taxonomy:
            taxonomy_str = " > ".join([
                taxonomy.get('l0', ''),
                taxonomy.get('l1', ''),
                taxonomy.get('l2', '')
            ])
            taxonomy_str = taxonomy_str.strip(' > ')
            if taxonomy_str:
                st.write(f"**Taxonomy:** {taxonomy_str}")
    
    # Scores section
    st.markdown("**Retrieval Scores:**")
    score_cols = st.columns(4)
    
    with score_cols[0]:
        display_score_badge(
            scores_data.get('predictions', 0),
            1,  # Prediction doesn't have rank
            "Prediction",
            SCORE_COLORS["prediction"]
        )
    
    with score_cols[1]:
        display_score_badge(
            scores_data.get('bm25_score', 0),
            scores_data.get('bm25_rank', 0),
            "BM25",
            SCORE_COLORS["bm25"]
        )
    
    with score_cols[2]:
        display_score_badge(
            scores_data.get('image_clip_score', 0),
            scores_data.get('image_clip_rank', 0),
            "Image CLIP",
            SCORE_COLORS["image_clip"]
        )
    
    with score_cols[3]:
        display_score_badge(
            scores_data.get('text_embedding_score', 0),
            scores_data.get('text_embedding_rank', 0),
            "Text Embed",
            SCORE_COLORS["text_embedding"]
        )
    
    # Hit indicators
    hits = []
    if scores_data.get('bm25_hit', False):
        hits.append("BM25")
    if scores_data.get('image_clip_hit', False):
        hits.append("Image CLIP")
    if scores_data.get('text_embedding_hit', False):
        hits.append("Text Embedding")
    
    if hits:
        st.write(f"**Hits:** {', '.join(hits)}")
    
    st.write(f"Label {scores_data.get('label', 0)}")
    
    st.markdown("---")

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title(APP_CONFIG["title"])
    st.markdown(APP_CONFIG["description"])
    
    # Load data
    with st.spinner("Loading data..."):
        features_df = load_features_data()
        items_db, text_documents, text_item_ids, image_paths, image_item_ids = load_items_database()
    
    # Sidebar for query selection
    st.sidebar.header("Query Selection")
    
    # Get unique queries
    unique_queries = sorted(features_df['query'].unique())
    
    # Query selector
    selected_query = st.sidebar.selectbox(
        "Choose a query:",
        unique_queries,
        help="Select a query to see its search results"
    )
    
    # Filter data for selected query
    query_results = features_df[features_df['query'] == selected_query].copy()
    
    # Sort by predictions (descending)
    query_results = query_results.sort_values('predictions', ascending=False)
    
    # Display query information
    st.header(f"Results for: '{selected_query}'")
    st.write(f"Found **{len(query_results)}** items for this query")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Score threshold filter
    min_prediction_score = st.sidebar.slider(
        "Minimum Prediction Score",
        min_value=float(query_results['predictions'].min()),
        max_value=float(query_results['predictions'].max()),
        value=float(query_results['predictions'].min()),
        step=0.01,
        help="Filter items by minimum prediction score"
    )
    
    # Apply filters
    filtered_results = query_results[query_results['predictions'] >= min_prediction_score]
    
    # Show hit statistics
    st.sidebar.header("Hit Statistics")
    total_items = len(filtered_results)
    bm25_hits = filtered_results['bm25_hit'].sum()
    image_hits = filtered_results['image_clip_hit'].sum()
    text_hits = filtered_results['text_embedding_hit'].sum()
    
    st.sidebar.metric("Total Items", total_items)
    st.sidebar.metric("BM25 Hits", f"{bm25_hits} ({bm25_hits/total_items*100:.1f}%)")
    st.sidebar.metric("Image CLIP Hits", f"{image_hits} ({image_hits/total_items*100:.1f}%)")
    st.sidebar.metric("Text Embedding Hits", f"{text_hits} ({text_hits/total_items*100:.1f}%)")
    
    # Display results
    if len(filtered_results) == 0:
        st.warning("No items match the current filters.")
        return
    
    st.write(f"Showing **{len(filtered_results)}** items (filtered)")
    
    # Pagination
    items_per_page = UI_CONFIG["items_per_page"]
    total_pages = (len(filtered_results) - 1) // items_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox(
            "Page:",
            range(1, total_pages + 1),
            help=f"Navigate through {total_pages} pages of results"
        )
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_results))
        page_results = filtered_results.iloc[start_idx:end_idx]
    else:
        page_results = filtered_results
    
    # Display items
    for idx, (_, row) in enumerate(page_results.iterrows(), 1):
        item_id = row['item_id']
        
        # Get item data
        if item_id not in items_db:
            st.warning(f"Item {item_id} not found in database")
            continue
        
        item_data = items_db[item_id]
        
        # Prepare scores data
        scores_data = {
            'predictions': row['predictions'],
            'bm25_score': row['bm25_score'],
            'bm25_rank': row['bm25_rank'],
            'image_clip_score': row['image_clip_score'],
            'image_clip_rank': row['image_clip_rank'],
            'text_embedding_score': row['text_embedding_score'],
            'text_embedding_rank': row['text_embedding_rank'],
            'bm25_hit': row['bm25_hit'],
            'image_clip_hit': row['image_clip_hit'],
            'text_embedding_hit': row['text_embedding_hit'],
            'label': row['label'],
        }
        
        # Display item card
        with st.container():
            st.markdown(f"### #{idx + (page - 1) * items_per_page if total_pages > 1 else idx}")
            display_item_card(item_data, scores_data)

if __name__ == "__main__":
    main()
