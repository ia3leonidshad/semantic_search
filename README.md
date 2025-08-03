# Retrieval Experiments

A configurable Python framework for experimenting with different retrieval methods including vector search (Faiss), lexical search (BM25), and hybrid approaches.

## Solution

- Multiple base retrievers are used to build the final pipeline: clip, bm25 and openai text embeddings large.
- Queries are extended to diversify recall sets of retrievers.
- Base retrievers pull candidates, which are labelled by LLM-as-a-judge to build ground truth dataset for evaluation.
- Another 100 queries are generated, passes through the same pipeline and used as a training set for the final ranking model.
- Final ranking model is xboost build on features provided from base retrievers.
- Streamlit demo visualizes features and final results.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

## Running evaluation and model creation

Check out scripts folder [README](./scripts/README.md).

Demo app [README](./docs/README_streamlit.md).

More info in [docs folder](./docs/).

### Scripts to reproduce results

Put `data/raw/queries.csv`, `data/raw/5k_items_curated.csv`.

```bash
mkdir -p data/raw/images
mkdir -p data/processed
mkdir -p data/indicies

# Download images
python scripts/download_images.py -i data/raw/5k_items_curated.csv -o data/raw/images/ --verbose

# Generate queries for train set
python scripts/generate_queries_cli.py \
  --mode from_examples \
  --examples-csv data/raw/queries.csv \
  --output data/processed/generated_queries.csv \
  --count 100

# Query expansion
python scripts/expand_queries_cli.py \
  --input data/raw/queries.csv \
  --output data/processed/expanded_queries.csv

# Query expansion for train
python scripts/expand_queries_cli.py \
  --input data/raw/generated_queries.csv \
  --output data/processed/expanded_queries_generated.csv

# Create base retrievers and candidate sets
python scripts/create_retrievers.py  \
  data/processed/expanded_queries.csv  \
  data/raw/5k_items_curated.csv  \
  data/processed/results_3_retrievers.json

# For train set
python scripts/create_retrievers.py  \
  data/processed/expanded_queries_generated.csv  \
  data/raw/5k_items_curated.csv  \
  data/processed/results_3_retrievers_generated.json

# Label the results
python scripts/create_ground_truth.py  \
  data/processed/results_3_retrievers.json \
  data/raw/5k_items_curated.csv  \
  data/processed/ground_truth_final.json  

# For train set
python scripts/create_ground_truth.py  \
  data/processed/results_3_retrievers_generated.json \
  data/raw/5k_items_curated.csv  \
  data/processed/ground_truth_final_generated.json  

# Generate features
python scripts/extract_features_cli.py \
  --retriever-results data/processed/results_3_retrievers.json \
  --item-data data/raw/5k_items_curated.csv \
  --ground-truth data/processed/ground_truth_final.json \
  --output data/processed/features_val.csv

# For train set
python scripts/extract_features_cli.py \
  --retriever-results data/processed/results_3_retrievers_generated.json \
  --item-data data/raw/5k_items_curated.csv \
  --ground-truth data/processed/ground_truth_final_generated.json \
  --output data/processed/features_train.csv

# Train re-ranker and get the results
python scripts/train_xgboost.py \
  --train-features data/processed/features_train.csv \
  --val-features data/processed/features_val.csv \
  --output-results data/processed/results_xgb_val.json

# Evaluate final results. Path are hardcoded, check the script
python scripts/evaluate_retrievers.py
```

## Project Structure

```
src/
├── models/          # Embedding model implementations
├── retrievers/      # Retrieval method implementations
├── data/            # Data loading and preprocessing
└── utils/           # Utilities and configuration
└── scripts/         # Script to build models and evaluation
```

## License

MIT License
