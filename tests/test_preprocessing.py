"""
tests/test_preprocessing.py
----------------------------
Unit tests for data_preprocessing module.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import clean_data, split_train_test, get_data_summary


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame that mimics the real dataset."""
    dates = pd.date_range("2020-01-01", periods=52, freq="W-FRI")
    records = []
    for store in [1, 2]:
        for dept in [1, 2]:
            for date in dates:
                records.append({
                    "Store":        store,
                    "Dept":         dept,
                    "Date":         date,
                    "Weekly_Sales": np.random.uniform(1000, 20000),
                    "IsHoliday":    False,
                    "Temperature":  70.0,
                    "Fuel_Price":   3.0,
                    "MarkDown1":    np.nan,
                    "MarkDown2":    500.0,
                    "MarkDown3":    np.nan,
                    "MarkDown4":    np.nan,
                    "MarkDown5":    300.0,
                    "CPI":          200.0,
                    "Unemployment": 6.5,
                    "Type":         "A",
                    "Size":         150_000,
                    "Current_Stock": 5000,
                    "Reorder_Point": 2000,
                    "Lead_Time_Days": 7,
                })
    return pd.DataFrame(records)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCleanData:
    def test_removes_duplicates(self, sample_df):
        duped = pd.concat([sample_df, sample_df.iloc[:10]], ignore_index=True)
        cleaned = clean_data(duped)
        assert len(cleaned) == len(sample_df)

    def test_negative_sales_zeroed(self, sample_df):
        sample_df.loc[0, "Weekly_Sales"] = -999.99
        cleaned = clean_data(sample_df)
        assert cleaned["Weekly_Sales"].min() >= 0.0

    def test_markdown_nan_filled(self, sample_df):
        cleaned = clean_data(sample_df)
        md_cols = [c for c in cleaned.columns if c.startswith("MarkDown")]
        for col in md_cols:
            assert cleaned[col].isna().sum() == 0

    def test_sorted_by_store_dept_date(self, sample_df):
        shuffled = sample_df.sample(frac=1, random_state=42)
        cleaned  = clean_data(shuffled)
        # After cleaning, rows should be in ascending order
        diffs = cleaned.groupby(["Store", "Dept"])["Date"].diff()
        assert (diffs.dropna() > pd.Timedelta(0)).all()

    def test_returns_dataframe(self, sample_df):
        result = clean_data(sample_df)
        assert isinstance(result, pd.DataFrame)


class TestSplitTrainTest:
    def test_no_overlap(self, sample_df):
        cleaned = clean_data(sample_df)
        train, test = split_train_test(cleaned, test_weeks=8)
        assert train["Date"].max() < test["Date"].min()

    def test_correct_split_size(self, sample_df):
        cleaned = clean_data(sample_df)
        train, test = split_train_test(cleaned, test_weeks=8)
        assert len(train) + len(test) == len(cleaned)

    def test_test_not_empty(self, sample_df):
        cleaned = clean_data(sample_df)
        _, test = split_train_test(cleaned, test_weeks=4)
        assert len(test) > 0


class TestGetDataSummary:
    def test_keys_present(self, sample_df):
        summary = get_data_summary(sample_df)
        expected_keys = {"n_rows", "n_stores", "n_depts",
                         "date_min", "date_max", "total_sales"}
        assert expected_keys.issubset(summary.keys())

    def test_n_stores_correct(self, sample_df):
        summary = get_data_summary(sample_df)
        assert summary["n_stores"] == 2

    def test_n_rows_correct(self, sample_df):
        summary = get_data_summary(sample_df)
        assert summary["n_rows"] == len(sample_df)
