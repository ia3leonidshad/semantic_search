# Query Processing Scripts

This directory contains enhanced CLI scripts for query processing using LLM-based methods.

## Scripts Overview

### 1. `expand_queries_cli.py` - Enhanced Query Expansion

An improved version of the original `expand_queries.py` with proper CLI interface and configurable options.

**Features:**
- Command-line arguments for input/output files
- Configurable model selection
- Optional processing steps (choose which expansions to run)
- Progress tracking and logging
- Error handling and validation
- Batch processing support

**Operations supported:**
- `rewrite`: Translate Portuguese queries to English
- `extend`: Extend queries with categories and product names
- `extend_english`: Generate English product names from Portuguese queries

### 2. `generate_queries_cli.py` - Query Generation

A new script that generates queries using LLM methods from `llm_models.py`.

**Features:**
- Two generation modes: `for_items` and `from_examples`
- Configurable filtering and deduplication
- Batch processing with progress tracking
- Quality control options
- Random seed support for reproducible results

**Generation modes:**
- `for_items`: Generate queries for specific items using example patterns
- `from_examples`: Generate new queries based on example query patterns

### 3. `extract_features_cli.py` - Feature Extraction for ML Training

A CLI script that extracts features from retriever results, item data, and ground truth for machine learning model training.

**Features:**
- Extracts multiple feature types from retriever results
- Handles multiple retrievers automatically
- Calculates category-based weights and rankings
- Adds ground truth labels for supervised learning
- Comprehensive logging and statistics
- Robust error handling and validation

**Feature types extracted:**
- `{retriever}_score`: Raw retriever scores
- `{retriever}_rank`: Item ranking within retriever results
- `{retriever}_category_weight`: Category-based weight distribution
- `{retriever}_hit`: Binary indicator if item was retrieved
- `label`: Ground truth relevance score
- `query`: Search query
- `item_id`: Item identifier

## Usage Examples

### Query Expansion

```bash
# Basic usage - expand all operations
python scripts/expand_queries_cli.py \
  --input data/raw/queries.csv \
  --output data/processed/expanded_queries.csv

# Specific operations only
python scripts/expand_queries_cli.py \
  --input data/raw/queries.csv \
  --output data/processed/expanded_queries.csv \
  --operations rewrite,extend

# With custom model and batch size
python scripts/expand_queries_cli.py \
  --input data/raw/queries.csv \
  --output data/processed/expanded_queries.csv \
  --model-name gpt-4 \
  --batch-size 5 \
  --verbose
```

### Query Generation

```bash
# Generate queries for specific items
python scripts/generate_queries_cli.py \
  --mode for_items \
  --items-csv data/raw/5k_items_curated.csv \
  --examples-csv data/raw/queries.csv \
  --output data/processed/generated_queries.csv \
  --count 100

# Generate queries from examples only
python scripts/generate_queries_cli.py \
  --mode from_examples \
  --examples-csv data/raw/queries.csv \
  --output data/processed/generated_queries.csv \
  --count 50

# With custom filtering and reproducible results
python scripts/generate_queries_cli.py \
  --mode from_examples \
  --examples-csv data/raw/queries.csv \
  --output data/processed/generated_queries.csv \
  --count 100 \
  --min-length 3 \
  --max-length 8 \
  --seed 42 \
  --verbose
```

### Creating retrievers and candidate list

```bash
python scripts/create_retrievers.py  \
  data/processed/queries_extended_english.csv  \
  data/raw/5k_items_curated.csv  \
  data/processed/results_3_retrievers.json
```

### Generating ground truth labels

```bash
python scripts/create_ground_truth.py  \
  data/processed/results_3_retrievers.json \
  data/raw/5k_items_curated.csv  \
  data/processed/ground_truth_final.json  
```

### Feature Extraction

```bash
# Basic feature extraction
python scripts/extract_features_cli.py \
  --retriever-results data/processed/results_3_retrievers.json \
  --item-data data/raw/5k_items_curated.csv \
  --ground-truth data/processed/ground_truth_final.json \
  --output data/processed/features.csv

# With custom images directory
python scripts/extract_features_cli.py \
  --retriever-results data/processed/results_3_retrievers.json \
  --item-data data/raw/5k_items_curated.csv \
  --ground-truth data/processed/ground_truth_final.json \
  --output data/processed/features.csv \
  --images-dir data/raw/custom_images
```

### Train re-ranker model

```bash
# Basic feature extraction
python scripts/train_xgboost.py \
  --train-features data/processed/features_train.csv \
  --val-features data/processed/features_val.csv \
  --output-results data/processed/results_xgb_val.json
```

### Evaluate results

```bash
# Path are hardcoded, check the script
python scripts/evaluate_retrievers.py
```

## Common Parameters

### Model Configuration
- `--model-type`: Model type (default: `openai`)
- `--model-name`: Model name (default: `gpt4.1-mini`)

### Input/Output
- `--input` / `-i`: Input CSV file
- `--output` / `-o`: Output CSV file
- `--query-column`: Column name containing queries (default: `search_term_pt`)

### Processing Options
- `--batch-size`: Batch size for processing (default: 10)
- `--verbose` / `-v`: Enable verbose logging

## File Requirements

### For Query Expansion
- **Input CSV**: Must contain a column with queries (default: `search_term_pt`)
- **Items CSV** (optional): For additional context

### For Query Generation
- **Examples CSV**: Must contain example queries in specified column
- **Items CSV**: Required for `for_items` mode, contains item descriptions

### For Feature Extraction
- **Retriever Results JSON**: Output from `create_retrievers.py` containing search results
- **Item Data CSV**: CSV file with item metadata (same format as used by EcommerceDataLoader)
- **Ground Truth JSON**: JSON file with relevance labels (output from `create_ground_truth.py`)
- **Images Directory**: Directory containing item images (optional, default: `data/raw/images`)

## Output Formats

### Query Expansion Output
The output CSV contains the original data plus additional columns based on operations:
- `english`: English translation of query
- `thoughts`: Translation reasoning
- `extend_1`, `extend_2`, etc.: Extended product names
- `category`: Predicted category
- `extend_eng_1`, `extend_eng_2`, etc.: English extended names

### Query Generation Output
The output CSV contains:
- `generated_query`: The generated query
- `source_item`: Source item (for `for_items` mode)
- `generation_method`: Method used (`for_item` or `from_examples`)

### Feature Extraction Output
The output CSV contains one row per query-item pair with the following columns:
- `query`: The search query
- `item_id`: Item identifier
- `label`: Ground truth relevance score (0 for not relevant, 1+ for relevant)
- `{retriever}_score`: Raw score from each retriever (e.g., `bm25_score`, `image_clip_score`)
- `{retriever}_rank`: Ranking position from each retriever (1-based, 40.0 for not retrieved)
- `{retriever}_category_weight`: Category-based weight distribution from each retriever
- `{retriever}_hit`: Binary indicator (True/False) if item was retrieved by each retriever

Example columns for 3 retrievers (bm25, image_clip, text_embedding):
- `bm25_score`, `bm25_rank`, `bm25_category_weight`, `bm25_hit`
- `image_clip_score`, `image_clip_rank`, `image_clip_category_weight`, `image_clip_hit`
- `text_embedding_score`, `text_embedding_rank`, `text_embedding_category_weight`, `text_embedding_hit`

## Error Handling

Both scripts include comprehensive error handling:
- File validation before processing
- Individual query error handling (continues processing on failures)
- Detailed logging with timestamps
- Progress tracking with `tqdm`

## Dependencies

Required packages:
- `pandas`
- `tqdm`
- `pathlib`
- Custom modules from `src/` directory

## Migration from Original Scripts

### From `expand_queries.py`
Replace hardcoded usage:
```python
# Old way
processor = ModelFactory.create_prompt_processor(model_type='openai', model_name='gpt4.1-mini')
queries = pd.read_csv("./data/raw/queries.csv")
# ... processing logic
```

With CLI usage:
```bash
# New way
python scripts/expand_queries_cli.py \
  --input data/raw/queries.csv \
  --output data/processed/expanded_queries.csv
```

### From `feature_extraction.py`
Replace hardcoded file paths and incomplete script:
```python
# Old way (incomplete and hardcoded)
with open('/Users/lekimov/Documents/search_ai/data/processed/ground_truth_final.json') as f:
    ground_truth = json.load(f)
queries = pd.read_csv("/Users/lekimov/Documents/search_ai/data/processed/queries_extended_english.csv")
# ... incomplete feature extraction logic
```

With CLI usage:
```bash
# New way (complete and configurable)
python scripts/extract_features_cli.py \
  --retriever-results data/processed/results_3_retrievers.json \
  --item-data data/raw/5k_items_curated.csv \
  --ground-truth data/processed/ground_truth_final.json \
  --output data/processed/features.csv
```

### Benefits of New Scripts
1. **Flexibility**: Configurable parameters without code changes
2. **Robustness**: Better error handling and validation
3. **Usability**: Progress tracking and clear logging
4. **Maintainability**: Clean separation of concerns
5. **Reproducibility**: Seed support and consistent output formats
