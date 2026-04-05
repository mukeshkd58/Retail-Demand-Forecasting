"""
src/business_logic.py
----------------------
Inventory optimization layer — the part that converts raw demand forecasts
into actionable procurement decisions.

Logic:
  - Economic Order Quantity (EOQ)
  - Reorder Point (ROP) = avg_demand * lead_time + safety_stock
  - Safety Stock = z_score * std_demand * sqrt(lead_time)
  - Classify each SKU: OK | LOW_STOCK | OVERSTOCK | CRITICAL
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

# Service-level z-scores
_Z_SCORES = {
    0.90: 1.28,
    0.95: 1.645,
    0.99: 2.33,
}

# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class InventoryDecision:
    Store:             int
    Dept:              int
    Predicted_Demand:  float   # units next week
    Current_Stock:     int
    Safety_Stock:      int
    Reorder_Point:     int
    EOQ:               int
    Status:            str     # OK | LOW_STOCK | OVERSTOCK | CRITICAL
    Suggested_Order:   int
    Days_Of_Cover:     float
    Urgency:           str     # HIGH | MEDIUM | LOW | NONE


# ── Core Functions ─────────────────────────────────────────────────────────────

def compute_safety_stock(
    std_demand: float,
    lead_time_days: float,
    service_level: float = 0.95,
) -> int:
    """Safety stock = z * std_demand * sqrt(lead_time / 7)."""
    z = _Z_SCORES.get(service_level, 1.645)
    lead_time_weeks = lead_time_days / 7.0
    ss = z * std_demand * np.sqrt(lead_time_weeks)
    return max(0, int(np.ceil(ss)))


def compute_reorder_point(
    avg_demand: float,
    lead_time_days: float,
    safety_stock: int,
) -> int:
    """ROP = avg_demand_per_day * lead_time + safety_stock."""
    avg_daily = avg_demand / 7.0
    rop = avg_daily * lead_time_days + safety_stock
    return max(0, int(np.ceil(rop)))


def compute_eoq(
    annual_demand: float,
    order_cost: float = 50.0,
    holding_cost_rate: float = 0.25,
    unit_cost: float = 10.0,
) -> int:
    """
    Wilson EOQ formula: sqrt(2 * D * S / (H))
    D = annual demand, S = order cost, H = holding cost per unit
    """
    H = holding_cost_rate * unit_cost
    if H <= 0 or annual_demand <= 0:
        return 0
    eoq = np.sqrt((2 * annual_demand * order_cost) / H)
    return max(1, int(np.ceil(eoq)))


def classify_inventory_status(
    current_stock: float,
    reorder_point: float,
    predicted_demand: float,
    overstock_multiplier: float = 3.0,
) -> tuple[str, str]:
    """
    Returns (Status, Urgency).
    Status  : CRITICAL | LOW_STOCK | OVERSTOCK | OK
    Urgency : HIGH | MEDIUM | LOW | NONE
    """
    if current_stock <= 0 or (
        predicted_demand > 0 and current_stock < predicted_demand * 0.25
    ):
        return "CRITICAL", "HIGH"

    if current_stock <= reorder_point:
        urgency = "HIGH" if current_stock < reorder_point * 0.5 else "MEDIUM"
        return "LOW_STOCK", urgency

    if current_stock > predicted_demand * overstock_multiplier:
        return "OVERSTOCK", "LOW"

    return "OK", "NONE"


def generate_inventory_decisions(
    forecast_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    history_df: pd.DataFrame,
    service_level: float = 0.95,
) -> pd.DataFrame:
    """
    Master function: merges forecasts with current inventory and outputs
    a decision table for every Store-Dept combination.

    Parameters
    ----------
    forecast_df  : DataFrame with columns [Store, Dept, Predicted_Sales]
    inventory_df : DataFrame with columns [Store, Dept, Current_Stock,
                                           Lead_Time_Days]
    history_df   : Historical sales for std/avg demand computation

    Returns
    -------
    DataFrame with one row per (Store, Dept) with full decision columns.
    """
    # Compute historical demand statistics per (Store, Dept)
    stats = (
        history_df.groupby(["Store", "Dept"])["Weekly_Sales"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "Avg_Weekly", "std": "Std_Weekly"})
        .reset_index()
    )
    stats["Std_Weekly"] = stats["Std_Weekly"].fillna(0)
    stats["Annual_Demand"] = stats["Avg_Weekly"] * 52

    # Merge all datasets
    merged = (
        forecast_df
        .merge(inventory_df, on=["Store", "Dept"], how="left")
        .merge(stats, on=["Store", "Dept"], how="left")
    )

    # Fill missing inventory data with reasonable defaults
    merged["Current_Stock"]  = merged.get("Current_Stock", merged["Predicted_Sales"] * 1.5).fillna(0)
    merged["Lead_Time_Days"] = merged.get("Lead_Time_Days", pd.Series(7, index=merged.index)).fillna(7)

    decisions = []
    for _, row in merged.iterrows():
        ss  = compute_safety_stock(row["Std_Weekly"],
                                   row["Lead_Time_Days"], service_level)
        rop = compute_reorder_point(row["Avg_Weekly"],
                                    row["Lead_Time_Days"], ss)
        eoq = compute_eoq(row["Annual_Demand"])

        status, urgency = classify_inventory_status(
            current_stock    = row["Current_Stock"],
            reorder_point    = rop,
            predicted_demand = row["Predicted_Sales"],
        )

        # Suggested order quantity
        if status in ("LOW_STOCK", "CRITICAL"):
            # Order enough to cover lead time + EOQ
            suggested = max(eoq, int(row["Predicted_Sales"] * 2))
        elif status == "OVERSTOCK":
            suggested = 0
        else:
            suggested = eoq if row["Current_Stock"] <= rop else 0

        # Days of cover
        daily_demand = row["Predicted_Sales"] / 7.0
        days_cover   = (row["Current_Stock"] / daily_demand) if daily_demand > 0 else 999

        decisions.append(InventoryDecision(
            Store            = int(row["Store"]),
            Dept             = int(row["Dept"]),
            Predicted_Demand = round(float(row["Predicted_Sales"]), 2),
            Current_Stock    = int(row["Current_Stock"]),
            Safety_Stock     = ss,
            Reorder_Point    = rop,
            EOQ              = eoq,
            Status           = status,
            Suggested_Order  = suggested,
            Days_Of_Cover    = round(days_cover, 1),
            Urgency          = urgency,
        ))

    return pd.DataFrame([asdict(d) for d in decisions])


def summary_report(decisions_df: pd.DataFrame) -> dict:
    """Return high-level KPIs from the decision table."""
    total = len(decisions_df)
    status_counts = decisions_df["Status"].value_counts().to_dict()

    return {
        "total_skus":        total,
        "critical":          status_counts.get("CRITICAL", 0),
        "low_stock":         status_counts.get("LOW_STOCK", 0),
        "overstock":         status_counts.get("OVERSTOCK", 0),
        "ok":                status_counts.get("OK", 0),
        "total_order_value": int(decisions_df["Suggested_Order"].sum()),
        "high_urgency_skus": int((decisions_df["Urgency"] == "HIGH").sum()),
    }
