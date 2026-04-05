"""
src/evaluation.py
-----------------
Model evaluation metrics and comparison utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


# ── Core Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)

    # WMAE — Walmart's official competition metric
    # holiday weeks are weighted 5x
    wmae  = mae   # fallback if no holiday column

    # MAPE (avoid division by zero)
    nonzero = y_true != 0
    mape  = mean_absolute_percentage_error(y_true[nonzero], y_pred[nonzero]) * 100

    return {
        "RMSE":  round(rmse, 4),
        "MAE":   round(mae,  4),
        "MAPE":  round(mape, 2),
        "R2":    round(r2,   4),
    }


def wmae(y_true: np.ndarray, y_pred: np.ndarray,
         weights: np.ndarray | None = None) -> float:
    """
    Weighted MAE — Walmart competition metric.
    Holiday weeks get weight=5; normal weeks get weight=1.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if weights is None:
        weights = np.ones(len(y_true))
    weights = np.asarray(weights, dtype=float)

    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def evaluate_all_models(
    models: dict,
    test_df: pd.DataFrame,
    holiday_col: str = "IsHoliday",
) -> pd.DataFrame:
    """
    Evaluate multiple models on the test set.

    Parameters
    ----------
    models   : dict of {name: fitted_model}
    test_df  : test DataFrame with features + target
    """
    from src.models import predict

    y_true = test_df["Weekly_Sales"].values
    weights = np.where(test_df[holiday_col].values == 1, 5.0, 1.0) \
              if holiday_col in test_df.columns else None

    rows = []
    for name, model in models.items():
        try:
            y_pred = predict(model, test_df)
            metrics = compute_metrics(y_true, y_pred)
            metrics["WMAE"]  = round(wmae(y_true, y_pred, weights), 4)
            metrics["Model"] = name
            rows.append(metrics)
        except Exception as exc:
            print(f"  ⚠  Could not evaluate {name}: {exc}")

    result = pd.DataFrame(rows).set_index("Model")
    result = result[["RMSE", "MAE", "MAPE", "R2", "WMAE"]]
    return result


def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return a DataFrame of residuals for analysis / plotting."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return pd.DataFrame({
        "y_true":    y_true,
        "y_pred":    y_pred,
        "residual":  residuals,
        "abs_error": np.abs(residuals),
        "pct_error": np.abs(residuals) / np.where(y_true == 0, 1, y_true) * 100,
    })
