# E-commerce Search Evaluation Streamlit UI

This Streamlit application provides an interactive dashboard to explore search evaluation results from your e-commerce search system.

## Features

- **Query Selection**: Choose from available queries in the evaluation dataset
- **Item Cards**: View item details including images, names, categories, and prices
- **Retrieval Scores**: Compare BM25, Image CLIP, and Text Embedding scores
- **Hit Analysis**: See which retrieval methods found each item
- **Filtering**: Filter results by prediction score threshold
- **Pagination**: Navigate through large result sets
- **Statistics**: View hit statistics for different retrieval methods

## Installation

1. Install the required dependencies:
```bash
pip install streamlit pillow
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have the required data files:
   - `data/processed/features_val.csv` - Evaluation features dataset
   - `data/raw/5k_items_curated.csv` - Items database
   - `data/raw/images/` - Product images directory

2. Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

3. Open your browser to the displayed URL (typically http://localhost:8501)

## Application Structure

### Data Loading
- **Features Data**: Loads evaluation results from `features_val.csv`
- **Items Database**: Uses `EcommerceDataLoader` to load item metadata and images
- **Caching**: Uses Streamlit's caching to improve performance

### UI Components

#### Sidebar
- Query dropdown selector
- Prediction score filter slider
- Hit statistics display

#### Main Area
- Query results header
- Paginated item cards
- Item details with images and scores

#### Item Cards
Each item card displays:
- Product image (if available)
- Item name and category
- Price and description
- Taxonomy hierarchy
- Retrieval scores with color-coded badges:
  - **Prediction Score** (Green) - Final model prediction
  - **BM25 Score** (Blue) - Lexical matching score
  - **Image CLIP Score** (Purple) - Visual similarity score
  - **Text Embedding Score** (Orange) - Semantic similarity score
- Hit indicators showing which methods found the item

## Data Requirements

### features_val.csv columns:
- `query` - Search query text
- `item_id` - Unique item identifier
- `predictions` - Final prediction score
- `bm25_score`, `bm25_rank`, `bm25_hit` - BM25 retrieval results
- `image_clip_score`, `image_clip_rank`, `image_clip_hit` - Image CLIP results
- `text_embedding_score`, `text_embedding_rank`, `text_embedding_hit` - Text embedding results

### Items database structure:
Items are loaded using `EcommerceDataLoader.load_from_csv()` and should contain:
- `name` - Product name
- `category_name` - Product category
- `price` - Product price
- `description` - Product description
- `images` - List of image filenames
- `taxonomy` - Hierarchical category structure (l0, l1, l2)

## Troubleshooting

### Common Issues

1. **"Features file not found"**
   - Ensure `data/processed/features_val.csv` exists
   - Check file path and permissions

2. **"Items CSV not found"**
   - Ensure `data/raw/5k_items_curated.csv` exists
   - Verify the CSV format matches expected structure

3. **Images not displaying**
   - Check that `data/raw/images/` directory exists
   - Verify image files are present and accessible
   - Images should be named according to the pattern in the items database

4. **Performance issues**
   - Large datasets may take time to load initially
   - Streamlit caching helps with subsequent loads
   - Consider reducing the dataset size for testing

### Performance Tips

- Use the prediction score filter to reduce the number of displayed items
- The application caches data loading for better performance
- Images are loaded lazily to improve initial load time

## Customization

You can customize the application by modifying:

- **Items per page**: Change `items_per_page` variable in `main()`
- **Score badge colors**: Modify colors in `display_score_badge()`
- **Image size**: Adjust image resize dimensions in `display_item_card()`
- **Filters**: Add additional filters in the sidebar section

## Dependencies

- `streamlit>=1.28.0` - Web application framework
- `pandas>=1.3.0` - Data manipulation
- `pillow>=9.0.0` - Image processing
- `numpy>=1.21.0` - Numerical operations
- Project-specific modules from `src/` directory
