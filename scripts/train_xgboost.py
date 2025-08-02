#!/usr/bin/env python3
"""
XGBoost training script for search AI ranking.

Trains an XGBoost ranking model using extracted features and saves results.

Usage:
    python scripts/train_xgboost.py \
        --train-features data/processed/features_train.csv \
        --val-features data/processed/features_val.csv \
        --output-model data/processed/xgboost_model.json \
        --output-results data/processed/results_xgb_val.json
"""

import argparse
import json
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import ndcg_score
import sys
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XGBoostRanker:
    """XGBoost ranking model trainer and evaluator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize XGBoost ranker.
        
        Args:
            params: XGBoost training parameters
        """
        self.params = params or {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg@5',
            'eta': 0.1,
            'max_depth': 3,
            'verbosity': 1
        }
        self.model = None
        
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for XGBoost training.
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            Tuple of (X, y, groups)
        """
        logger.info(f"Preparing data with shape: {df.shape}")
        
        # Convert labels to binary (label == 2)
        df = df.copy()
        df['label'] = df['label'] == 2
        
        # Extract feature columns
        feature_cols = [c for c in df.columns if c not in ['query', 'item_id', 'label']]
        logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
        
        X = df[feature_cols]
        y = df['label']
        
        # Create groups for ranking (number of items per query)
        groups = df.groupby('query').size().tolist()
        logger.info(f"Created {len(groups)} query groups")
        
        return X, y, groups
    
    def train(self, X_train, y_train, groups_train, X_val=None, y_val=None, groups_val=None,
              num_boost_round: int = 100, early_stopping_rounds: int = 20) -> xgb.Booster:
        """Train XGBoost ranking model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            groups_train: Training query groups
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            groups_val: Validation query groups (optional)
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Trained XGBoost booster
        """
        logger.info("Creating DMatrix for training...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(groups_train)
        
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None and groups_val is not None:
            logger.info("Creating DMatrix for validation...")
            dval = xgb.DMatrix(X_val, label=y_val)
            dval.set_group(groups_val)
            evals.append((dval, 'validation'))
        
        logger.info(f"Starting training with parameters: {self.params}")
        logger.info(f"Boost rounds: {num_boost_round}, Early stopping: {early_stopping_rounds}")
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10
        )
        
        logger.info("Training completed!")
        return self.model
    
    def evaluate_ndcg(self, X_val, y_val, queries_val, k: int = 5) -> float:
        """Evaluate NDCG@k on validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            queries_val: Validation query identifiers
            k: NDCG@k parameter
            
        Returns:
            Average NDCG@k score
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        logger.info(f"Evaluating NDCG@{k}...")
        
        # Create DMatrix for prediction
        dval = xgb.DMatrix(X_val)
        
        # Get predictions
        preds = self.model.predict(dval)
        
        # Calculate NDCG@k for each query
        ndcg_scores = []
        for q in set(queries_val):
            mask = queries_val == q
            if mask.sum() > 0:  # Ensure we have items for this query
                ndcg = ndcg_score(
                    y_val[mask].values.reshape(1, -1),
                    preds[mask].reshape(1, -1),
                    k=k
                )
                ndcg_scores.append(ndcg)
        
        avg_ndcg = pd.Series(ndcg_scores).mean()
        logger.info(f'Average NDCG@{k} on validation set: {avg_ndcg:.4f}')
        
        return avg_ndcg
    
    def predict_and_format_results(self, df_val: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions and format results for output.
        
        Args:
            df_val: Validation DataFrame
            
        Returns:
            Formatted results dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        logger.info("Making predictions and formatting results...")
        
        # Prepare validation data
        X_val, y_val, _ = self.prepare_data(df_val)
        
        # Create DMatrix and predict
        dval = xgb.DMatrix(X_val)
        preds = self.model.predict(dval)
        
        # Add predictions to dataframe
        df_val = df_val.copy()
        df_val['preds'] = preds
        
        # Format results by query
        results_xgb = {}
        for query, group in df_val.groupby('query'):
            # Sort by predictions (descending)
            g = group.sort_values('preds', ascending=False)
            results_xgb[str(query)] = [
                {'item_id': row.item_id, 'score': row.preds} 
                for _, row in g.iterrows()
            ]
        
        logger.info(f"Formatted results for {len(results_xgb)} queries")
        
        return {'xgb': results_xgb}
    
    def save_model(self, output_path: str):
        """Save trained model to file.
        
        Args:
            output_path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        logger.info(f"Saving model to {output_path}")
        self.model.save_model(output_path)


def load_features(file_path: str) -> pd.DataFrame:
    """Load features from CSV file.
    
    Args:
        file_path: Path to features CSV file
        
    Returns:
        Features DataFrame
    """
    logger.info(f"Loading features from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded features with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost ranking model for search AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/train_xgboost.py \\
        --train-features data/processed/features_train.csv \\
        --val-features data/processed/features_val.csv \\
        --output-model data/processed/xgboost_model.json \\
        --output-results data/processed/results_xgb_val.json
        """
    )
    
    parser.add_argument(
        "--train-features",
        default="data/processed/features_train.csv",
        help="Path to training features CSV file"
    )
    parser.add_argument(
        "--val-features", 
        default="data/processed/features_val.csv",
        help="Path to validation features CSV file"
    )
    parser.add_argument(
        "--output-model",
        default="data/processed/xgboost_model.json",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--output-results",
        default="data/processed/results_xgb_val.json",
        help="Path to save validation results"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum tree depth (default: 3)"
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=100,
        help="Number of boosting rounds (default: 100)"
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=20,
        help="Early stopping rounds (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path, name in [
        (args.train_features, "Training features"),
        (args.val_features, "Validation features")
    ]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{name} file not found: {file_path}")
    
    try:
        # Load data
        df_train = load_features(args.train_features)
        df_val = load_features(args.val_features)
        
        # Set up training parameters
        params = {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg@5',
            'eta': args.eta,
            'max_depth': args.max_depth,
            'verbosity': 1
        }
        
        # Initialize ranker
        ranker = XGBoostRanker(params)
        
        # Prepare training data
        X_train, y_train, groups_train = ranker.prepare_data(df_train)
        X_val, y_val, groups_val = ranker.prepare_data(df_val)
        
        # Train model
        ranker.train(
            X_train, y_train, groups_train,
            X_val, y_val, groups_val,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds
        )
        
        # Evaluate NDCG@5
        queries_val = df_val['query'].values
        avg_ndcg = ranker.evaluate_ndcg(X_val, y_val, queries_val, k=5)
        
        # Create output directories
        Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_results).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        ranker.save_model(args.output_model)
        
        # Generate and save results
        results = ranker.predict_and_format_results(df_val)
        
        X_val, y_val, _ = ranker.prepare_data(df_val)
        
        # Create DMatrix and predict
        dval = xgb.DMatrix(X_val)
        preds = ranker.model.predict(dval)

        df_val_wp = df_val.copy()
        df_val_wp['predictions'] = preds

        val_save = args.val_features[:-4] + '_wp.csv'
        logger.info(f"Saving val with predictions to {val_save}")
        df_val_wp.to_csv(val_save, index=False)

        logger.info(f"Saving results to {args.output_results}")
        with open(args.output_results, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("✅ XGBoost training completed successfully!")
        logger.info(f"📊 Model saved to: {args.output_model}")
        logger.info(f"📈 Results saved to: {args.output_results}")
        logger.info(f"🎯 Final NDCG@5: {avg_ndcg:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during XGBoost training: {e}")
        raise


if __name__ == "__main__":
    exit(main())
