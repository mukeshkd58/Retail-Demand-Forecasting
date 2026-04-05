"""
generate_sample_data.py
-----------------------
Generates a realistic Walmart-style retail sales dataset.
Run this once to create the raw CSV files used by the pipeline.

Columns mirror the Kaggle Walmart Recruiting - Store Sales Forecasting dataset:
  Store, Dept, Date, Weekly_Sales, IsHoliday,
  Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, Type, Size
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# ── Constants ──────────────────────────────────────────────────────────────────
N_STORES   = 10
N_DEPTS    = 20
START_DATE = "2010-02-05"
END_DATE   = "2013-10-25"
STORE_TYPES = ["A", "B", "C"]
STORE_SIZE_MAP = {"A": (150_000, 220_000), "B": (80_000, 150_000), "C": (30_000, 80_000)}

HOLIDAY_WEEKS = [
    "2010-02-12", "2010-09-10", "2010-11-26", "2010-12-31",
    "2011-02-11", "2011-09-09", "2011-11-25", "2011-12-30",
    "2012-02-10", "2012-09-07", "2012-11-23", "2012-12-28",
    "2013-02-08", "2013-09-06", "2013-11-29", "2013-12-27",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _store_meta(store_id: int) -> dict:
    store_type = RNG.choice(STORE_TYPES, p=[0.4, 0.4, 0.2])
    lo, hi = STORE_SIZE_MAP[store_type]
    size = int(RNG.integers(lo, hi))
    return {"Store": store_id, "Type": store_type, "Size": size}


def _weekly_sales(base: float, date: pd.Timestamp, is_holiday: bool,
                  store_type: str, dept: int) -> float:
    """Simulate weekly sales with seasonality, trend, and noise."""
    # Seasonal multiplier (Christmas/Thanksgiving spike)
    month = date.month
    seasonal = 1.0
    if month == 11:
        seasonal = 1.25
    elif month == 12:
        seasonal = 1.45
    elif month in (6, 7):
        seasonal = 1.10
    elif month in (1, 2):
        seasonal = 0.85

    # Holiday uplift
    holiday_boost = 1.18 if is_holiday else 1.0

    # Store-type multiplier
    type_mult = {"A": 1.2, "B": 1.0, "C": 0.75}[store_type]

    # Dept-level base (some depts sell more)
    dept_mult = 0.5 + (dept % 10) * 0.1

    # Slight upward trend over the years
    trend = 1 + (date.year - 2010) * 0.02

    noise = RNG.normal(1.0, 0.08)

    sales = base * seasonal * holiday_boost * type_mult * dept_mult * trend * noise
    return max(0.0, round(sales, 2))


# ── Build Dataset ──────────────────────────────────────────────────────────────

def generate_sales(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(START_DATE, END_DATE, freq="W-FRI")
    holiday_set = set(pd.to_datetime(HOLIDAY_WEEKS).strftime("%Y-%m-%d"))

    stores_meta = [_store_meta(s) for s in range(1, N_STORES + 1)]
    store_df = pd.DataFrame(stores_meta)

    records = []
    for smeta in stores_meta:
        sid   = smeta["Store"]
        stype = smeta["Type"]
        base_sales = RNG.uniform(10_000, 60_000)   # per-store base

        for dept in range(1, N_DEPTS + 1):
            for date in dates:
                date_str   = date.strftime("%Y-%m-%d")
                is_holiday = date_str in holiday_set

                sales = _weekly_sales(base_sales, date, is_holiday, stype, dept)

                records.append({
                    "Store":       sid,
                    "Dept":        dept,
                    "Date":        date_str,
                    "Weekly_Sales": sales,
                    "IsHoliday":   is_holiday,
                    "Temperature": round(RNG.uniform(10, 100), 2),
                    "Fuel_Price":  round(RNG.uniform(2.5, 4.5), 3),
                    "MarkDown1":   round(RNG.choice([np.nan, RNG.uniform(0, 10_000)],
                                                    p=[0.5, 0.5]), 2),
                    "MarkDown2":   round(RNG.choice([np.nan, RNG.uniform(0, 5_000)],
                                                    p=[0.6, 0.4]), 2),
                    "MarkDown3":   round(RNG.choice([np.nan, RNG.uniform(0, 2_000)],
                                                    p=[0.65, 0.35]), 2),
                    "MarkDown4":   round(RNG.choice([np.nan, RNG.uniform(0, 4_000)],
                                                    p=[0.55, 0.45]), 2),
                    "MarkDown5":   round(RNG.choice([np.nan, RNG.uniform(0, 8_000)],
                                                    p=[0.5, 0.5]), 2),
                    "CPI":         round(RNG.uniform(126, 228), 3),
                    "Unemployment": round(RNG.uniform(3.5, 14.0), 3),
                })

    df = pd.DataFrame(records)

    # Merge store metadata
    df = df.merge(store_df, on="Store")

    # Add current inventory (simulated for business-logic demo)
    df["Current_Stock"] = (df["Weekly_Sales"] * RNG.uniform(0.5, 2.0,
                                                              size=len(df))).round().astype(int)
    df["Reorder_Point"] = (df["Weekly_Sales"] * 0.6).round().astype(int)
    df["Lead_Time_Days"] = RNG.integers(3, 15, size=len(df))

    df.to_csv(output_dir / "walmart_sales.csv", index=False)
    store_df.to_csv(output_dir / "stores.csv", index=False)

    print(f"✅  walmart_sales.csv  → {len(df):,} rows")
    print(f"✅  stores.csv         → {len(store_df)} stores")
    print(f"   Date range: {df['Date'].min()} → {df['Date'].max()}")


# ── Inventory snapshot (current state for dashboard) ──────────────────────────

def generate_inventory_snapshot(output_dir: Path) -> None:
    """Latest-week inventory snapshot used by the dashboard."""
    sales_path = output_dir / "walmart_sales.csv"
    if not sales_path.exists():
        raise FileNotFoundError("Run generate_sales() first.")

    df = pd.read_csv(sales_path)
    latest = df[df["Date"] == df["Date"].max()].copy()
    latest = latest[["Store", "Dept", "Date", "Weekly_Sales",
                      "Current_Stock", "Reorder_Point", "Lead_Time_Days"]].copy()
    latest.to_csv(output_dir / "inventory_snapshot.csv", index=False)
    print(f"✅  inventory_snapshot.csv → {len(latest):,} rows")


if __name__ == "__main__":
    raw_dir = Path(__file__).parent / "raw"
    generate_sales(raw_dir)
    generate_inventory_snapshot(raw_dir)
