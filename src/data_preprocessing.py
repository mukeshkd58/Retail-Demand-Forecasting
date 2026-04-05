"""
src/data_preprocessing.py
--------------------------
Handles all data cleaning, validation, and basic transformation steps
before feature engineering.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_raw_data(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load and merge raw Walmart sales + store metadata."""
    data_dir = Path(data_dir)
    sales_path = data_dir / "walmart_sales.csv"
    stores_path = data_dir / "stores.csv"

    if not sales_path.exists():
        raise FileNotFoundError(
            f"Sales data not found at {sales_path}. "
            "Run `python data/generate_sample_data.py` first."
        )

    df = pd.read_csv(sales_path, parse_dates=["Date"])

    if stores_path.exists() and "Type" not in df.columns:
        stores = pd.read_csv(stores_path)
        df = df.merge(stores, on="Store", how="left")

    logger.info("Loaded raw data: %s rows, %s cols", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
      - Remove duplicate rows
      - Fix negative sales (set to 0)
      - Impute missing markdown values with 0 (no promotion)
      - Fill remaining numeric NaN with column median
      - Validate date column
    """
    df = df.copy()

    # ── Duplicates ─────────────────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset=["Store", "Dept", "Date"], inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d duplicate rows", dropped)

    # ── Negative sales ─────────────────────────────────────────────────────────
    neg_mask = df["Weekly_Sales"] < 0
    if neg_mask.any():
        logger.warning("Replacing %d negative sales values with 0", neg_mask.sum())
        df.loc[neg_mask, "Weekly_Sales"] = 0.0

    # ── MarkDown columns ───────────────────────────────────────────────────────
    markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
    df[markdown_cols] = df[markdown_cols].fillna(0.0)

    # ── Remaining numeric NaN → median ─────────────────────────────────────────
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info("Imputed '%s' NaN with median %.4f", col, median_val)

    # ── Sort ───────────────────────────────────────────────────────────────────
    df.sort_values(["Store", "Dept", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Cleaned data shape: %s", df.shape)
    return df


def split_train_test(
    df: pd.DataFrame,
    test_weeks: int = 26,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/test split — the last `test_weeks` weeks are held out.
    Never shuffle time-series data randomly.
    """
    cutoff = df["Date"].max() - pd.Timedelta(weeks=test_weeks)
    train = df[df["Date"] <= cutoff].copy()
    test  = df[df["Date"] >  cutoff].copy()
    logger.info(
        "Train: %d rows (%s → %s)  |  Test: %d rows (%s → %s)",
        len(train), train["Date"].min().date(), train["Date"].max().date(),
        len(test),  test["Date"].min().date(),  test["Date"].max().date(),
    )
    return train, test


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict — used by the Streamlit dashboard."""
    return {
        "n_rows":    len(df),
        "n_stores":  df["Store"].nunique(),
        "n_depts":   df["Dept"].nunique(),
        "date_min":  str(df["Date"].min().date()),
        "date_max":  str(df["Date"].max().date()),
        "total_sales": df["Weekly_Sales"].sum(),
        "avg_weekly":  df["Weekly_Sales"].mean(),
        "missing_pct": df.isnull().mean().mean() * 100,
    }


def save_processed(df: pd.DataFrame, output_dir: str | Path = "data/processed",
                   filename: str = "processed_sales.csv") -> Path:
    """Persist the cleaned/processed DataFrame."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    df.to_csv(path, index=False)
    logger.info("Saved processed data → %s", path)
    return path
