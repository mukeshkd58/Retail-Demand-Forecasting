"""
tests/test_models.py
---------------------
Unit tests for model training, prediction, and persistence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import clean_data
from src.feature_engineering import build_features


@pytest.fixture(scope="module")
def train_df():
    """Build a small feature DataFrame for testing model code."""
    dates = pd.date_range("2018-01-01", periods=104, freq="W-FRI")
    records = []
    for store in [1, 2]:
        for dept in [1, 2]:
            for date in dates:
                records.append({
                    "Store": store, "Dept": dept, "Date": date,
                    "Weekly_Sales": np.random.uniform(2000, 20000),
                    "IsHoliday": False,
                    "Temperature": 65.0, "Fuel_Price": 3.2,
                    "MarkDown1": 0.0, "MarkDown2": 0.0,
                    "MarkDown3": 0.0, "MarkDown4": 0.0, "MarkDown5": 0.0,
                    "CPI": 200.0, "Unemployment": 7.0,
                    "Type": "A", "Size": 150_000,
                    "Current_Stock": 5000,
                    "Reorder_Point": 2000,
                    "Lead_Time_Days": 7,
                })
    df = pd.DataFrame(records)
    df = clean_data(df)
    return build_features(df, drop_na=True)


class TestLinearRegression:
    def test_trains_and_predicts(self, train_df):
        from src.models import train_linear_regression, predict
        model = train_linear_regression(train_df)
        preds = predict(model, train_df)
        assert len(preds) == len(train_df)
        assert np.all(preds >= 0)

    def test_predictions_are_float(self, train_df):
        from src.models import train_linear_regression, predict
        model  = train_linear_regression(train_df)
        preds  = predict(model, train_df)
        assert preds.dtype in (np.float32, np.float64)


class TestRandomForest:
    def test_trains_and_predicts(self, train_df):
        from src.models import train_random_forest, predict
        model = train_random_forest(train_df, n_estimators=10, max_depth=4)
        preds = predict(model, train_df)
        assert len(preds) == len(train_df)
        assert np.all(preds >= 0)

    def test_feature_importance_available(self, train_df):
        from src.models import train_random_forest, get_feature_importance
        model = train_random_forest(train_df, n_estimators=10, max_depth=4)
        fi = get_feature_importance(model)
        assert not fi.empty
        assert "Feature" in fi.columns
        assert "Importance" in fi.columns


class TestXGBoost:
    def test_trains_and_predicts(self, train_df):
        from src.models import train_xgboost, predict
        model = train_xgboost(train_df, params={"n_estimators": 20})
        preds = predict(model, train_df)
        assert len(preds) == len(train_df)
        assert np.all(preds >= 0)


class TestSaveLoad:
    def test_save_and_load(self, train_df):
        from src.models import train_linear_regression, save_model, load_model, predict
        model = train_linear_regression(train_df)

        with tempfile.TemporaryDirectory() as tmp:
            save_model(model, "test_lr", Path(tmp))
            loaded = load_model("test_lr", Path(tmp))
            preds_original = predict(model, train_df)
            preds_loaded   = predict(loaded, train_df)
            np.testing.assert_array_almost_equal(preds_original, preds_loaded)

    def test_load_missing_raises(self):
        from src.models import load_model
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model_xyz", Path("/tmp"))
