# XGBoost Training Script

This script trains an XGBoost ranking model using extracted features and saves the results.

## Prerequisites

Install the required dependencies:
```bash
pip install pandas xgboost scikit-learn
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python scripts/train_xgboost.py
```

This will use the default paths:
- Training features: `data/processed/features_train.csv`
- Validation features: `data/processed/features_val.csv`
- Output model: `data/processed/xgboost_model.json`
- Output results: `data/processed/results_xgb_val.json`

### Custom Paths
```bash
python scripts/train_xgboost.py \
    --train-features data/processed/features_train.csv \
    --val-features data/processed/features_val.csv \
    --output-model data/processed/xgboost_model.json \
    --output-results data/processed/results_xgb_val.json
```

### Custom Training Parameters
```bash
python scripts/train_xgboost.py \
    --eta 0.05 \
    --max-depth 5 \
    --num-boost-round 200 \
    --early-stopping-rounds 30
```

## Parameters

- `--train-features`: Path to training features CSV file
- `--val-features`: Path to validation features CSV file  
- `--output-model`: Path to save trained model
- `--output-results`: Path to save validation results
- `--eta`: Learning rate (default: 0.1)
- `--max-depth`: Maximum tree depth (default: 3)
- `--num-boost-round`: Number of boosting rounds (default: 100)
- `--early-stopping-rounds`: Early stopping rounds (default: 20)

## Input Data Format

The script expects CSV files with the following columns:
- `query`: Query identifier
- `item_id`: Item identifier  
- `label`: Relevance label (will be converted to binary: label == 2)
- Feature columns: All other columns are treated as features

## Output

The script produces:
1. **Trained model**: Saved as XGBoost JSON format
2. **Validation results**: JSON file with predictions formatted as:
   ```json
   {
     "xgb": {
       "query1": [
         {"item_id": "item1", "score": 0.85},
         {"item_id": "item2", "score": 0.72}
       ]
     }
   }
   ```
3. **NDCG@5 evaluation**: Printed to console

## Features

- **Ranking objective**: Uses XGBoost's `rank:ndcg` objective
- **Group-aware training**: Properly handles query groups for ranking
- **Early stopping**: Prevents overfitting with validation monitoring
- **NDCG@5 evaluation**: Calculates ranking quality metric
- **Comprehensive logging**: Detailed progress and results logging
- **Flexible parameters**: Configurable training hyperparameters

## Example Output

```
2025-01-08 17:00:00,000 - INFO - Loading features from data/processed/features_train.csv
2025-01-08 17:00:01,000 - INFO - Loaded features with shape: (50000, 15)
2025-01-08 17:00:01,000 - INFO - Using 12 features: ['bm25_score', 'faiss_score', ...]
2025-01-08 17:00:02,000 - INFO - Starting training with parameters: {'objective': 'rank:ndcg', ...}
[0]	train-ndcg@5:0.85432	validation-ndcg@5:0.82156
[10]	train-ndcg@5:0.89234	validation-ndcg@5:0.84567
...
2025-01-08 17:00:30,000 - INFO - Training completed!
2025-01-08 17:00:31,000 - INFO - Average NDCG@5 on validation set: 0.8456
2025-01-08 17:00:32,000 - INFO - ✅ XGBoost training completed successfully!
2025-01-08 17:00:32,000 - INFO - 🎯 Final NDCG@5: 0.8456
