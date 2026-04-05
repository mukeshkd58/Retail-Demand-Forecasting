"""
src/eda.py
----------
Exploratory Data Analysis functions.
Each function returns a matplotlib Figure — callable from notebooks or
the Streamlit app (via st.pyplot).
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = "viridis"
sns.set_theme(style="darkgrid", palette=PALETTE)


def _fmt_millions(x, _): return f"${x/1e6:.1f}M"


# ── 1. Total weekly sales trend ────────────────────────────────────────────────

def plot_sales_trend(df: pd.DataFrame, store: int | None = None) -> plt.Figure:
    """Aggregate weekly sales over time (optionally filtered by store)."""
    data = df[df["Store"] == store] if store else df
    weekly = data.groupby("Date")["Weekly_Sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(weekly["Date"], weekly["Weekly_Sales"], linewidth=1.5, color="#4C72B0")
    ax.fill_between(weekly["Date"], weekly["Weekly_Sales"], alpha=0.15, color="#4C72B0")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax.set_title(f"Weekly Sales Trend {'— Store ' + str(store) if store else '(All Stores)'}")
    ax.set_xlabel("")
    ax.set_ylabel("Sales")
    fig.tight_layout()
    return fig


# ── 2. Seasonality / Monthly patterns ─────────────────────────────────────────

def plot_monthly_seasonality(df: pd.DataFrame) -> plt.Figure:
    """Average sales by month — shows seasonality."""
    monthly = (
        df.groupby(df["Date"].dt.month)["Weekly_Sales"]
        .mean()
        .reset_index()
        .rename(columns={"Date": "Month"})
    )
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["Month_Name"] = monthly["Month"].apply(lambda m: month_names[m-1])

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(monthly["Month_Name"], monthly["Weekly_Sales"],
                  color=sns.color_palette("viridis", 12))
    ax.set_title("Average Weekly Sales by Month")
    ax.set_ylabel("Avg Weekly Sales ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


# ── 3. Holiday vs Non-Holiday ──────────────────────────────────────────────────

def plot_holiday_effect(df: pd.DataFrame) -> plt.Figure:
    """Box plot comparing holiday vs. regular week sales."""
    fig, ax = plt.subplots(figsize=(7, 4))
    df_plot = df.copy()
    df_plot["Holiday"] = df_plot["IsHoliday"].map({0: "Regular", 1: "Holiday",
                                                    False: "Regular", True: "Holiday"})
    sns.boxplot(data=df_plot, x="Holiday", y="Weekly_Sales",
                palette={"Regular": "#4C72B0", "Holiday": "#DD8452"}, ax=ax)
    ax.set_title("Sales Distribution: Holiday vs. Regular Weeks")
    ax.set_ylabel("Weekly Sales ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


# ── 4. Sales by store type ─────────────────────────────────────────────────────

def plot_sales_by_store_type(df: pd.DataFrame) -> plt.Figure:
    """Average weekly sales for each store type (A / B / C)."""
    if "Type" not in df.columns:
        return plt.figure()

    type_sales = (
        df.groupby("Type")["Weekly_Sales"].mean().reset_index()
        .sort_values("Weekly_Sales", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"A": "#2ecc71", "B": "#3498db", "C": "#e74c3c"}
    bars = ax.bar(type_sales["Type"],
                  type_sales["Weekly_Sales"],
                  color=[colors.get(t, "#999") for t in type_sales["Type"]],
                  width=0.5)
    ax.set_title("Avg Weekly Sales by Store Type")
    ax.set_ylabel("Avg Weekly Sales ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return fig


# ── 5. Top departments ─────────────────────────────────────────────────────────

def plot_top_departments(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Horizontal bar chart of top N departments by total sales."""
    dept_sales = (
        df.groupby("Dept")["Weekly_Sales"].sum()
        .nlargest(top_n)
        .reset_index()
        .sort_values("Weekly_Sales")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(dept_sales["Dept"].astype(str), dept_sales["Weekly_Sales"],
            color=sns.color_palette("viridis", top_n))
    ax.set_title(f"Top {top_n} Departments by Total Sales")
    ax.set_xlabel("Total Sales ($)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    fig.tight_layout()
    return fig


# ── 6. Correlation heatmap ─────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of feature correlations with Weekly_Sales."""
    numeric_cols = [
        "Weekly_Sales", "Temperature", "Fuel_Price", "CPI",
        "Unemployment", "MarkDown1", "MarkDown2", "MarkDown3",
        "MarkDown4", "MarkDown5",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    return fig


# ── 7. Predicted vs Actual ─────────────────────────────────────────────────────

def plot_predicted_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
) -> plt.Figure:
    """Scatter plot of predicted vs. actual sales."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, color="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Sales ($)")
    ax.set_ylabel("Predicted Sales ($)")
    ax.set_title(f"{model_name}: Predicted vs. Actual")
    ax.legend()
    fig.tight_layout()
    return fig


# ── 8. Feature importances ─────────────────────────────────────────────────────

def plot_feature_importance(fi_df: pd.DataFrame, model_name: str = "Model") -> plt.Figure:
    """Horizontal bar chart of feature importances."""
    fi_df = fi_df.sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(fi_df) * 0.3)))
    ax.barh(fi_df["Feature"], fi_df["Importance"],
            color=sns.color_palette("viridis", len(fi_df)))
    ax.set_title(f"{model_name} — Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig
