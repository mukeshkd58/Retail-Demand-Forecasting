#!/usr/bin/env python3
"""
scripts/train_pipeline.py
--------------------------
End-to-end training pipeline:
  1. Load raw data
  2. Clean
  3. Feature engineering
  4. Train-test split
  5. Train Linear Regression, Random Forest, XGBoost
  6. Evaluate all models
  7. Save best model + all models
  8. Print summary table

Usage:
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --test-weeks 26 --output-dir models/
"""

import argparse
import json
import sys
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, ensure_dir
from src.data_preprocessing import load_raw_data, clean_data, split_train_test, save_processed
from src.feature_engineering import build_features
from src.models import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    save_model,
    predict,
)
from src.evaluation import evaluate_all_models

import logging
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retail Demand Forecasting — Training Pipeline")
    p.add_argument("--data-dir",    default="data/raw",    help="Raw data directory")
    p.add_argument("--output-dir",  default="models",       help="Where to save trained models")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--test-weeks",  type=int, default=26,   help="Weeks to hold out for testing")
    p.add_argument("--log-level",   default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    ensure_dir(args.output_dir, args.processed_dir)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading raw data")
    df_raw = load_raw_data(args.data_dir)

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    logger.info("STEP 2 — Cleaning data")
    df_clean = clean_data(df_raw)

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    logger.info("STEP 3 — Feature engineering")
    df_feat = build_features(df_clean, drop_na=True)
    save_processed(df_feat, args.processed_dir, "features.csv")
    logger.info("Feature matrix: %d rows × %d cols", *df_feat.shape)

    # ── Step 4: Train-Test Split ──────────────────────────────────────────────
    logger.info("STEP 4 — Train/Test split (%d weeks held out)", args.test_weeks)
    train_df, test_df = split_train_test(df_feat, args.test_weeks)

    # ── Step 5: Train Models ──────────────────────────────────────────────────
    logger.info("STEP 5 — Training models")

    logger.info("  → Linear Regression")
    lr_model = train_linear_regression(train_df)
    save_model(lr_model, "linear_regression", Path(args.output_dir))

    logger.info("  → Random Forest (this may take ~30s)")
    rf_model = train_random_forest(train_df, n_estimators=200, max_depth=12)
    save_model(rf_model, "random_forest", Path(args.output_dir))

    # Use last 20% of training data as validation for XGBoost early stopping
    val_cutoff = int(len(train_df) * 0.8)
    xgb_train  = train_df.iloc[:val_cutoff]
    xgb_val    = train_df.iloc[val_cutoff:]

    logger.info("  → XGBoost")
    xgb_model = train_xgboost(xgb_train, val_df=xgb_val)
    save_model(xgb_model, "xgboost", Path(args.output_dir))

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    logger.info("STEP 6 — Evaluating all models on test set")
    models = {
        "LinearRegression": lr_model,
        "RandomForest":     rf_model,
        "XGBoost":          xgb_model,
    }
    results = evaluate_all_models(models, test_df)

    logger.info("\n%s", results.to_string())
    results.to_csv(Path(args.processed_dir) / "model_comparison.csv")

    # ── Step 7: Save best model separately ───────────────────────────────────
    best_model_name = results["RMSE"].idxmin()
    best_model      = models[best_model_name]
    save_model(best_model, "best_model", Path(args.output_dir))
    logger.info("✅  Best model: %s (RMSE=%.2f)", best_model_name,
                results.loc[best_model_name, "RMSE"])

    # Save metadata
    meta = {
        "best_model":  best_model_name,
        "test_weeks":  args.test_weeks,
        "train_rows":  len(train_df),
        "test_rows":   len(test_df),
        "metrics":     results.loc[best_model_name].to_dict(),
    }
    with open(Path(args.output_dir) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training complete. Models saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
