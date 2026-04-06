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
    Predicted_Demand:  float
    Current_Stock:     int
    Safety_Stock:      int
    Reorder_Point:     int
    EOQ:               int
    Status:            str     # OK | LOW_STOCK | OVERSTOCK | CRITICAL
    Suggested_Order:   int
    Days_Of_Cover:     float
    Urgency:           str     # HIGH | MEDIUM | LOW | NONE


# ── Helper: safe float ─────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    """Return default if val is None, NaN, or infinite."""
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


# ── Core Functions ─────────────────────────────────────────────────────────────

def compute_safety_stock(
    std_demand: float,
    lead_time_days: float,
    service_level: float = 0.95,
) -> int:
    """Safety stock = z * std_demand * sqrt(lead_time_weeks)."""
    std_demand     = _safe_float(std_demand,     0.0)
    lead_time_days = _safe_float(lead_time_days, 7.0)

    z = _Z_SCORES.get(service_level, 1.645)
    lead_time_weeks = lead_time_days / 7.0
    ss = z * std_demand * np.sqrt(lead_time_weeks)
    return max(0, int(np.ceil(_safe_float(ss, 0.0))))


def compute_reorder_point(
    avg_demand: float,
    lead_time_days: float,
    safety_stock: int,
) -> int:
    """ROP = avg_demand_per_day * lead_time + safety_stock."""
    avg_demand     = _safe_float(avg_demand,     0.0)
    lead_time_days = _safe_float(lead_time_days, 7.0)
    safety_stock   = int(_safe_float(safety_stock, 0))

    avg_daily = avg_demand / 7.0
    rop = avg_daily * lead_time_days + safety_stock
    return max(0, int(np.ceil(_safe_float(rop, 0.0))))


def compute_eoq(
    annual_demand: float,
    order_cost: float = 50.0,
    holding_cost_rate: float = 0.25,
    unit_cost: float = 10.0,
) -> int:
    """
    Wilson EOQ formula: sqrt(2 * D * S / H)
    D = annual demand, S = order cost, H = holding cost per unit
    """
    annual_demand = _safe_float(annual_demand, 0.0)
    H = holding_cost_rate * unit_cost
    if H <= 0 or annual_demand <= 0:
        return 0
    eoq = np.sqrt((2 * annual_demand * order_cost) / H)
    return max(1, int(np.ceil(_safe_float(eoq, 1.0))))


def classify_inventory_status(
    current_stock: float,
    reorder_point: float,
    predicted_demand: float,
    overstock_multiplier: float = 3.0,
) -> tuple:
    """
    Returns (Status, Urgency).
    Status  : CRITICAL | LOW_STOCK | OVERSTOCK | OK
    Urgency : HIGH | MEDIUM | LOW | NONE
    """
    current_stock    = _safe_float(current_stock,    0.0)
    reorder_point    = _safe_float(reorder_point,    0.0)
    predicted_demand = _safe_float(predicted_demand, 0.0)

    if current_stock <= 0 or (
        predicted_demand > 0 and current_stock < predicted_demand * 0.25
    ):
        return "CRITICAL", "HIGH"

    if current_stock <= reorder_point:
        urgency = "HIGH" if current_stock < reorder_point * 0.5 else "MEDIUM"
        return "LOW_STOCK", urgency

    if predicted_demand > 0 and current_stock > predicted_demand * overstock_multiplier:
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
    # ── Step 1: Historical demand stats per (Store, Dept) ─────────────────────
    if history_df.empty or "Weekly_Sales" not in history_df.columns:
        stats = pd.DataFrame(columns=["Store", "Dept",
                                       "Avg_Weekly", "Std_Weekly", "Annual_Demand"])
    else:
        stats = (
            history_df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "Avg_Weekly", "std": "Std_Weekly"})
            .reset_index()
        )
        stats["Std_Weekly"]    = stats["Std_Weekly"].fillna(0.0)
        stats["Annual_Demand"] = stats["Avg_Weekly"] * 52

    # ── Step 2: Merge everything ───────────────────────────────────────────────
    merged = (
        forecast_df
        .merge(inventory_df, on=["Store", "Dept"], how="left")
        .merge(stats,        on=["Store", "Dept"], how="left")
    )

    # ── Step 3: Fill ALL NaN with safe defaults BEFORE the loop ───────────────
    # If a Store-Dept combo has no history (e.g. heavy filter applied),
    # fall back to the predicted sales value so the logic never sees NaN.
    merged["Avg_Weekly"] = merged["Avg_Weekly"].fillna(merged["Predicted_Sales"])
    merged["Std_Weekly"] = merged["Std_Weekly"].fillna(0.0)
    merged["Annual_Demand"] = merged["Annual_Demand"].fillna(
        merged["Predicted_Sales"] * 52
    )

    # Inventory defaults
    if "Current_Stock" not in merged.columns:
        merged["Current_Stock"] = (merged["Predicted_Sales"] * 1.5).astype(int)
    else:
        merged["Current_Stock"] = merged["Current_Stock"].fillna(
            (merged["Predicted_Sales"] * 1.5)
        ).astype(float).astype(int)

    if "Lead_Time_Days" not in merged.columns:
        merged["Lead_Time_Days"] = 7
    else:
        merged["Lead_Time_Days"] = merged["Lead_Time_Days"].fillna(7)

    # Final safety net — replace any remaining NaN in numeric cols with 0
    for col in ["Avg_Weekly", "Std_Weekly", "Annual_Demand",
                "Current_Stock", "Lead_Time_Days", "Predicted_Sales"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    # ── Step 4: Build decisions row by row ────────────────────────────────────
    decisions = []
    for _, row in merged.iterrows():
        predicted = _safe_float(row["Predicted_Sales"], 0.0)
        avg_w     = _safe_float(row["Avg_Weekly"],      predicted)
        std_w     = _safe_float(row["Std_Weekly"],      0.0)
        annual_d  = _safe_float(row["Annual_Demand"],   avg_w * 52)
        lead_days = _safe_float(row["Lead_Time_Days"],  7.0)
        cur_stock = _safe_float(row["Current_Stock"],   predicted * 1.5)

        ss  = compute_safety_stock(std_w, lead_days, service_level)
        rop = compute_reorder_point(avg_w, lead_days, ss)
        eoq = compute_eoq(annual_d)

        status, urgency = classify_inventory_status(
            current_stock    = cur_stock,
            reorder_point    = rop,
            predicted_demand = predicted,
        )

        # Suggested order quantity
        if status in ("LOW_STOCK", "CRITICAL"):
            suggested = max(eoq, int(predicted * 2))
        elif status == "OVERSTOCK":
            suggested = 0
        else:
            suggested = eoq if cur_stock <= rop else 0

        # Days of cover
        daily_demand = predicted / 7.0
        days_cover   = (cur_stock / daily_demand) if daily_demand > 0 else 999.0

        decisions.append(InventoryDecision(
            Store            = int(_safe_float(row["Store"], 0)),
            Dept             = int(_safe_float(row["Dept"],  0)),
            Predicted_Demand = round(predicted, 2),
            Current_Stock    = int(cur_stock),
            Safety_Stock     = ss,
            Reorder_Point    = rop,
            EOQ              = eoq,
            Status           = status,
            Suggested_Order  = int(suggested),
            Days_Of_Cover    = round(_safe_float(days_cover, 999.0), 1),
            Urgency          = urgency,
        ))

    return pd.DataFrame([asdict(d) for d in decisions])


def summary_report(decisions_df: pd.DataFrame) -> dict:
    """Return high-level KPIs from the decision table."""
    if decisions_df.empty:
        return {
            "total_skus": 0, "critical": 0, "low_stock": 0,
            "overstock": 0, "ok": 0, "total_order_value": 0,
            "high_urgency_skus": 0,
        }

    total         = len(decisions_df)
    status_counts = decisions_df["Status"].value_counts().to_dict()

    return {
        "total_skus":        total,
        "critical":          status_counts.get("CRITICAL",  0),
        "low_stock":         status_counts.get("LOW_STOCK", 0),
        "overstock":         status_counts.get("OVERSTOCK", 0),
        "ok":                status_counts.get("OK",        0),
        "total_order_value": int(decisions_df["Suggested_Order"].sum()),
        "high_urgency_skus": int((decisions_df["Urgency"] == "HIGH").sum()),
    }