"""
src/models.py
-------------
Defines, trains, and saves all forecasting models:
  1. Linear Regression       (baseline)
  2. Random Forest Regressor
  3. XGBoost Regressor       (best performer)
  4. Prophet                 (time-series, per-store/dept)
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

from src.feature_engineering import FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


# ── Helper ─────────────────────────────────────────────────────────────────────

def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df[TARGET_COL]
    return X, y


# ── 1. Linear Regression (Baseline) ───────────────────────────────────────────

def train_linear_regression(train_df: pd.DataFrame) -> Pipeline:
    """Scaled linear regression baseline."""
    X, y = _xy(train_df)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LinearRegression()),
    ])
    model.fit(X, y)
    logger.info("Linear Regression trained on %d samples", len(X))
    return model


# ── 2. Random Forest ───────────────────────────────────────────────────────────

def train_random_forest(
    train_df: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int = 12,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Random Forest with sensible defaults."""
    X, y = _xy(train_df)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=n_jobs,
    )
    model.fit(X, y)
    logger.info("Random Forest trained  — OOB score: N/A (oob_score=False)")
    return model


# ── 3. XGBoost (Best) ──────────────────────────────────────────────────────────

def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    params: dict | None = None,
) -> xgb.XGBRegressor:
    """
    XGBoost regressor with early stopping if val_df is provided.
    Default hyper-parameters are tuned for weekly retail sales.
    """
    X_train, y_train = _xy(train_df)
    eval_set = None

    if val_df is not None:
        X_val, y_val = _xy(val_df)
        eval_set = [(X_val, y_val)]

    default_params = {
        "n_estimators":    800,
        "learning_rate":   0.05,
        "max_depth":       7,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":       0.1,
        "reg_lambda":      1.0,
        "random_state":    42,
        "n_jobs":         -1,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)

    fit_kwargs: dict[str, Any] = {}
    if eval_set:
        fit_kwargs = {
            "eval_set":    eval_set,
            "verbose":     100,
        }

    model.fit(X_train, y_train, **fit_kwargs)
    logger.info("XGBoost trained — %d trees", model.n_estimators)
    return model


# ── 4. Prophet (per Store-Dept) ────────────────────────────────────────────────

def train_prophet_single(series_df: pd.DataFrame) -> Any:
    """
    Train a Prophet model for a single (Store, Dept) time series.
    series_df must have columns: Date, Weekly_Sales, IsHoliday.
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Install prophet: `pip install prophet`")

    # Prophet needs ds / y columns
    prophet_df = series_df[["Date", "Weekly_Sales"]].rename(
        columns={"Date": "ds", "Weekly_Sales": "y"}
    )

    # Mark holidays as special regressors
    holidays_df = series_df[series_df["IsHoliday"] == 1][["Date"]].rename(
        columns={"Date": "ds"}
    )
    holidays_df["holiday"] = "retail_holiday"
    holidays_df["lower_window"] = 0
    holidays_df["upper_window"] = 1

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df if len(holidays_df) else None,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)
    return model


def predict_prophet_single(model: Any, periods: int = 26) -> pd.DataFrame:
    """Generate future forecast for a single (Store, Dept) Prophet model."""
    future = model.make_future_dataframe(periods=periods, freq="W")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)


# ── Save / Load helpers ────────────────────────────────────────────────────────

def save_model(model: Any, name: str, output_dir: Path = MODELS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.pkl"
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)
    return path


def load_model(name: str, model_dir: Path = MODELS_DIR) -> Any:
    path = model_dir / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found at {path}. Run the training pipeline first."
        )
    model = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return model


# ── Predict wrappers ───────────────────────────────────────────────────────────

def predict(model: Any, df: pd.DataFrame) -> np.ndarray:
    """Generic predict — works for LR, RF, XGBoost pipelines."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    preds = model.predict(X)
    return np.maximum(preds, 0)   # sales cannot be negative


def get_feature_importance(model: Any, top_n: int = 20) -> pd.DataFrame:
    """Extract feature importances from RF or XGBoost."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):   # Pipeline
        inner = list(model.named_steps.values())[-1]
        importances = getattr(inner, "feature_importances_", None)
        if importances is None:
            importances = getattr(inner, "coef_", np.zeros(1))
    else:
        return pd.DataFrame()

    # Build feature name list
    from src.feature_engineering import FEATURE_COLS
    feat_names = FEATURE_COLS[:len(importances)]

    fi_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return fi_df
