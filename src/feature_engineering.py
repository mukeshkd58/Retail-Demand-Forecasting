"""
src/feature_engineering.py
---------------------------
Creates all features used by the ML models:
  - Calendar features (year, month, week, quarter, day-of-week, season)
  - Lag features (previous N weeks' sales)
  - Rolling statistics (mean, std, min, max)
  - Markdown aggregation
  - Categorical encoding
"""

import numpy as np
import pandas as pd


# ── Season mapping ─────────────────────────────────────────────────────────────
_MONTH_TO_SEASON = {
    1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall",   10: "Fall",  11: "Fall",
    12: "Winter",
}

_SEASON_ENCODE = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}


# ── Public API ─────────────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from the Date column."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]     = df["Date"].dt.quarter
    df["DayOfWeek"]   = df["Date"].dt.dayofweek          # 0=Mon
    df["IsWeekend"]   = (df["DayOfWeek"] >= 5).astype(int)
    df["Season"]      = df["Month"].map(_MONTH_TO_SEASON)
    df["Season_Code"] = df["Season"].map(_SEASON_ENCODE)

    # Cyclical encoding for month and week (avoids discontinuity at Dec→Jan)
    df["Month_Sin"]   = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"]   = np.cos(2 * np.pi * df["Month"] / 12)
    df["Week_Sin"]    = np.sin(2 * np.pi * df["Week"]  / 52)
    df["Week_Cos"]    = np.cos(2 * np.pi * df["Week"]  / 52)

    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """
    Add lagged sales features within each (Store, Dept) group.
    Default lags: [1, 2, 3, 4, 8, 13, 26, 52] weeks.
    """
    if lags is None:
        lags = [1, 2, 3, 4, 8, 13, 26, 52]

    df = df.copy().sort_values(["Store", "Dept", "Date"])
    grp = df.groupby(["Store", "Dept"])["Weekly_Sales"]

    for lag in lags:
        df[f"Lag_{lag}w"] = grp.shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """
    Add rolling-window statistics (mean, std, min, max) for sales
    within each (Store, Dept) group.
    Default windows: [4, 8, 13] weeks.
    """
    if windows is None:
        windows = [4, 8, 13]

    df = df.copy().sort_values(["Store", "Dept", "Date"])

    for w in windows:
        rolled = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"Roll_{w}w_Mean"] = rolled

        rolled_std = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())
        )
        df[f"Roll_{w}w_Std"] = rolled_std.fillna(0)

    return df


def add_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the 5 MarkDown columns into summary features."""
    df = df.copy()
    md_cols = [c for c in df.columns if c.startswith("MarkDown")]
    if not md_cols:
        return df

    df["MarkDown_Total"]  = df[md_cols].sum(axis=1)
    df["MarkDown_Count"]  = (df[md_cols] > 0).sum(axis=1)
    df["MarkDown_Max"]    = df[md_cols].max(axis=1)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode Store Type; convert IsHoliday to int if needed."""
    df = df.copy()

    if "Type" in df.columns:
        type_map = {"A": 0, "B": 1, "C": 2}
        df["Type_Code"] = df["Type"].map(type_map).fillna(-1).astype(int)

    if df["IsHoliday"].dtype == bool:
        df["IsHoliday"] = df["IsHoliday"].astype(int)

    return df


def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Full feature-engineering pipeline — applies all transformations
    in the correct order.

    Parameters
    ----------
    df       : cleaned DataFrame from data_preprocessing.clean_data()
    drop_na  : whether to drop rows with NaN lag features (first N rows
               per group have no history)
    """
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_markdown_features(df)
    df = encode_categoricals(df)

    if drop_na:
        before = len(df)
        df.dropna(inplace=True)
        dropped = before - len(df)
        if dropped:
            print(f"  ↳ Dropped {dropped:,} rows with NaN lag/rolling features")

    df.reset_index(drop=True, inplace=True)
    return df


# ── Feature list used by models ────────────────────────────────────────────────

FEATURE_COLS = [
    # Store/Dept identifiers (numeric)
    "Store", "Dept",
    # Calendar
    "Year", "Month", "Week", "Quarter",
    "IsWeekend", "Season_Code",
    "Month_Sin", "Month_Cos", "Week_Sin", "Week_Cos",
    # Holiday
    "IsHoliday",
    # External
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    # Store meta
    "Size", "Type_Code",
    # MarkDown
    "MarkDown_Total", "MarkDown_Count", "MarkDown_Max",
    # Lags
    "Lag_1w", "Lag_2w", "Lag_3w", "Lag_4w",
    "Lag_8w", "Lag_13w", "Lag_26w", "Lag_52w",
    # Rolling
    "Roll_4w_Mean", "Roll_4w_Std",
    "Roll_8w_Mean", "Roll_8w_Std",
    "Roll_13w_Mean", "Roll_13w_Std",
]

TARGET_COL = "Weekly_Sales"
