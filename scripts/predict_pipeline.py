#!/usr/bin/env python3
"""
scripts/predict_pipeline.py
-----------------------------
Loads the best trained model and generates next-week demand forecasts
+ inventory decisions for all Store-Dept combinations.

Usage:
    python scripts/predict_pipeline.py
    python scripts/predict_pipeline.py --weeks 4 --store 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, ensure_dir
from src.data_preprocessing import load_raw_data, clean_data
from src.feature_engineering import build_features, FEATURE_COLS
from src.models import load_model, predict
from src.business_logic import generate_inventory_decisions, summary_report

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    default="data/raw")
    p.add_argument("--model-dir",   default="models")
    p.add_argument("--output-dir",  default="data/processed")
    p.add_argument("--weeks",       type=int, default=1, help="Weeks ahead to forecast")
    p.add_argument("--store",       type=int, default=None, help="Filter to single store")
    p.add_argument("--log-level",   default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    ensure_dir(args.output_dir)

    # ── Load and prepare data ─────────────────────────────────────────────────
    logger.info("Loading data …")
    df = clean_data(load_raw_data(args.data_dir))
    df_feat = build_features(df, drop_na=True)

    if args.store:
        df_feat = df_feat[df_feat["Store"] == args.store]
        df      = df[df["Store"] == args.store]

    # Latest week per store-dept as the "current state"
    latest = df_feat.sort_values("Date").groupby(["Store", "Dept"]).last().reset_index()

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading best model …")
    model = load_model("best_model", Path(args.model_dir))

    # ── Predict ───────────────────────────────────────────────────────────────
    logger.info("Generating forecasts for %d Store-Dept combos …", len(latest))
    latest["Predicted_Sales"] = predict(model, latest)

    # ── Business logic ────────────────────────────────────────────────────────
    inventory_path = Path(args.data_dir) / "inventory_snapshot.csv"
    if inventory_path.exists():
        inventory_df = pd.read_csv(inventory_path)
    else:
        # Fallback: synthetic inventory
        inventory_df = latest[["Store", "Dept"]].copy()
        inventory_df["Current_Stock"] = (latest["Predicted_Sales"] * 1.3).astype(int).values
        inventory_df["Lead_Time_Days"] = 7

    decisions_df = generate_inventory_decisions(
        forecast_df  = latest[["Store", "Dept", "Predicted_Sales"]],
        inventory_df = inventory_df,
        history_df   = df,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_forecast  = Path(args.output_dir) / "forecasts.csv"
    out_decisions = Path(args.output_dir) / "inventory_decisions.csv"

    latest[["Store", "Dept", "Date", "Predicted_Sales"]].to_csv(out_forecast, index=False)
    decisions_df.to_csv(out_decisions, index=False)

    summary = summary_report(decisions_df)

    logger.info("\n── Inventory Summary ──────────────────────────────────────")
    for k, v in summary.items():
        logger.info("  %-25s : %s", k, v)
    logger.info("────────────────────────────────────────────────────────────")
    logger.info("Forecasts   → %s", out_forecast)
    logger.info("Decisions   → %s", out_decisions)


if __name__ == "__main__":
    main()
