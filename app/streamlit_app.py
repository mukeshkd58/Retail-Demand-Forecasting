"""
app/streamlit_app.py
---------------------
Retail Demand Forecasting & Inventory Optimization Dashboard

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import sys
from pathlib import Path

# def auto_setup():
#     """Generate data and train model if not present (for cloud deploy)."""
#     # Absolute paths — app/ folder se root folder
#     root = Path(__file__).parent.parent
#     model_path = root / "models" / "best_model.pkl"
#     data_path  = root / "data" / "raw" / "walmart_sales.csv"
#     python     = sys.executable  # correct python path on any system

#     if not data_path.exists():
#         import streamlit as st
#         st.info("⏳ Generating data... (first run only, ~10 seconds)")
#         subprocess.run(
#             [python, str(root / "data" / "generate_sample_data.py")],
#             cwd=str(root),   # working directory = project root
#             check=True
#         )

#     if not model_path.exists():
#         import streamlit as st
#         st.info("⏳ Training models... (first run only, ~2 minutes)")
#         subprocess.run(
#             [python, str(root / "scripts" / "train_pipeline.py")],
#             cwd=str(root),   # working directory = project root
#             check=True
#         )
#         st.rerun()

# auto_setup()
def auto_setup():
    """Generate data and train model if not present (for cloud deploy)."""
    root = Path(__file__).parent.parent
    model_path = root / "models" / "best_model.pkl"
    data_path  = root / "data" / "raw" / "walmart_sales.csv"
    python     = sys.executable

    import streamlit as st

    # Session flag to avoid rerun loop
    if "setup_done" in st.session_state:
        return

    if not data_path.exists():
        st.info("⏳ Generating data... (first run only)")
        subprocess.run(
            [python, str(root / "data" / "generate_sample_data.py")],
            cwd=str(root),
            check=True
        )

    if not model_path.exists():
        st.info("⏳ Training models... (first run only, ~2 minutes)")
        subprocess.run(
            [python, str(root / "scripts" / "train_pipeline.py")],
            cwd=str(root),
            check=True
        )
        st.success("✅ Training completed!")

    # Mark setup done
    st.session_state["setup_done"] = True

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_preprocessing import load_raw_data, clean_data, get_data_summary
from src.feature_engineering import build_features
from src.business_logic import generate_inventory_decisions, summary_report

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Overrides ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-top: 4px;
    }
    .status-critical { background: #c0392b; color: white; border-radius: 6px; padding: 2px 8px; }
    .status-low      { background: #e67e22; color: white; border-radius: 6px; padding: 2px 8px; }
    .status-over     { background: #8e44ad; color: white; border-radius: 6px; padding: 2px 8px; }
    .status-ok       { background: #27ae60; color: white; border-radius: 6px; padding: 2px 8px; }
    .section-header  { font-size: 1.25rem; font-weight: 600; border-left: 4px solid #2d6a9f;
                       padding-left: 10px; margin: 20px 0 12px 0; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading (cached) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data …")
def load_data():
    df_raw   = load_raw_data("data/raw")
    df_clean = clean_data(df_raw)
    return df_clean


@st.cache_data(show_spinner="Building features …")
def load_features(df):
    return build_features(df, drop_na=True)


@st.cache_resource(show_spinner="Loading model …")
def load_best_model():
    try:
        from src.models import load_model
        return load_model("best_model", Path("models"))
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Generating forecasts …")
def get_forecasts(_model, df_feat):
    if _model is None:
        # Fallback: use rolling mean as pseudo-forecast
        df = df_feat.copy()
        df["Predicted_Sales"] = df["Roll_4w_Mean"].fillna(df["Weekly_Sales"])
        return df
    from src.models import predict
    df = df_feat.copy()
    df["Predicted_Sales"] = predict(_model, df_feat)
    return df


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.image("https://img.icons8.com/color/96/box.png", width=60)
    st.sidebar.title("📦 RetailForecast")
    st.sidebar.markdown("---")

    stores  = sorted(df["Store"].unique())
    depts   = sorted(df["Dept"].unique())

    selected_store = st.sidebar.selectbox("🏪 Select Store", ["All"] + stores)
    selected_dept  = st.sidebar.selectbox("📂 Select Department", ["All"] + depts)

    date_min = df["Date"].min()
    date_max = df["Date"].max()
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(date_min.date(), date_max.date()),
        min_value=date_min.date(),
        max_value=date_max.date(),
    )

    service_level = st.sidebar.select_slider(
        "🎯 Service Level",
        options=[0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x*100)}%"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: Walmart-style synthetic dataset  \n"
                       "Model: XGBoost / Random Forest  \n"
                       "Built by Mukesh Kumar")

    return selected_store, selected_dept, date_range, service_level


# ── KPI Row ────────────────────────────────────────────────────────────────────
def render_kpis(df_filtered, decisions_df):
    summary = summary_report(decisions_df)
    total_sales  = df_filtered["Weekly_Sales"].sum()
    avg_weekly   = df_filtered.groupby("Date")["Weekly_Sales"].sum().mean()

    cols = st.columns(6)
    kpis = [
        ("💰 Total Sales",    f"${total_sales/1e6:.1f}M"),
        ("📊 Avg Weekly",     f"${avg_weekly/1e3:.1f}K"),
        ("🔴 Critical SKUs",  str(summary["critical"])),
        ("🟠 Low Stock",      str(summary["low_stock"])),
        ("🟣 Overstock",      str(summary["overstock"])),
        ("📦 Total Orders",   f"{summary['total_order_value']/1e6:.1f}M"),
    ]
    for col, (label, value) in zip(cols, kpis):
        col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
        """, unsafe_allow_html=True)


# ── Tab: Overview ──────────────────────────────────────────────────────────────
def tab_overview(df_filtered):
    st.markdown('<div class="section-header">📈 Weekly Sales Trend</div>',
                unsafe_allow_html=True)

    # Weekly aggregate
    weekly = df_filtered.groupby("Date")["Weekly_Sales"].sum().reset_index()
    weekly["MA_4w"] = weekly["Weekly_Sales"].rolling(4).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly["Date"], y=weekly["Weekly_Sales"],
        name="Weekly Sales", line=dict(color="#2196F3", width=1.5),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.1)"
    ))
    fig.add_trace(go.Scatter(
        x=weekly["Date"], y=weekly["MA_4w"],
        name="4-Week MA", line=dict(color="#FF5722", width=2, dash="dot")
    ))
    # Mark holiday weeks
    holidays = df_filtered[df_filtered["IsHoliday"].isin([1, True])]["Date"].unique()
    for hd in holidays[:20]:  # show first 20 holidays
        fig.add_vline(x=str(hd)[:10], line_dash="dash",
                      line_color="rgba(255,200,0,0.4)", line_width=1)
    fig.update_layout(
        height=350, hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        yaxis_title="Weekly Sales ($)", xaxis_title="",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2-col row
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">📆 Sales by Month</div>',
                    unsafe_allow_html=True)
        month_df = df_filtered.copy()
        month_df["Month"] = pd.to_datetime(month_df["Date"]).dt.month
        monthly = month_df.groupby("Month")["Weekly_Sales"].mean().reset_index()
        monthly["Month_Name"] = monthly["Month"].map({
            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
            7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
        })
        fig2 = px.bar(monthly, x="Month_Name", y="Weekly_Sales",
                      color="Weekly_Sales", color_continuous_scale="Viridis",
                      labels={"Weekly_Sales": "Avg Sales ($)", "Month_Name": ""})
        fig2.update_layout(height=300, showlegend=False,
                           coloraxis_showscale=False,
                           margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">🏪 Sales by Store Type</div>',
                    unsafe_allow_html=True)
        if "Type" in df_filtered.columns:
            type_df = df_filtered.groupby("Type")["Weekly_Sales"].mean().reset_index()
            fig3 = px.pie(type_df, values="Weekly_Sales", names="Type",
                          color_discrete_sequence=["#2196F3", "#4CAF50", "#FF9800"])
            fig3.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">🏆 Top 10 Departments</div>',
                unsafe_allow_html=True)
    top_dept = (
        df_filtered.groupby("Dept")["Weekly_Sales"].sum()
        .nlargest(10).reset_index()
        .sort_values("Weekly_Sales")
    )
    fig4 = px.bar(top_dept, x="Weekly_Sales", y="Dept", orientation="h",
                  color="Weekly_Sales", color_continuous_scale="Blues",
                  labels={"Weekly_Sales": "Total Sales ($)", "Dept": "Dept ID"})
    fig4.update_layout(height=300, showlegend=False,
                       coloraxis_showscale=False,
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig4, use_container_width=True)


# ── Tab: Demand Forecast ───────────────────────────────────────────────────────
def tab_forecast(df_filtered, df_forecast):
    st.markdown('<div class="section-header">🔮 Demand Forecast vs. Actuals</div>',
                unsafe_allow_html=True)

    # Aggregate actual + predicted by date
    actual_weekly = df_filtered.groupby("Date")["Weekly_Sales"].sum().reset_index()
    actual_weekly.columns = ["Date", "Actual"]

    if "Predicted_Sales" in df_forecast.columns:
        pred_df = df_forecast[df_forecast["Store"].isin(df_filtered["Store"].unique())]
        pred_weekly = pred_df.groupby("Date")["Predicted_Sales"].sum().reset_index()
        pred_weekly.columns = ["Date", "Predicted"]

        merged = actual_weekly.merge(pred_weekly, on="Date", how="inner")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged["Date"], y=merged["Actual"],
            name="Actual Sales", line=dict(color="#2196F3", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=merged["Date"], y=merged["Predicted"],
            name="Predicted Sales",
            line=dict(color="#FF5722", width=2, dash="dot")
        ))
        fig.update_layout(
            height=380, hovermode="x unified",
            yaxis_title="Weekly Sales ($)", xaxis_title="",
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter: predicted vs actual
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">🎯 Predicted vs. Actual (Scatter)</div>',
                        unsafe_allow_html=True)
            sample = pred_df.sample(min(2000, len(pred_df)), random_state=42)
            fig2 = px.scatter(sample, x="Weekly_Sales", y="Predicted_Sales",
                              opacity=0.4, color_discrete_sequence=["#2196F3"],
                              labels={"Weekly_Sales": "Actual ($)",
                                      "Predicted_Sales": "Predicted ($)"})
            max_val = max(sample["Weekly_Sales"].max(), sample["Predicted_Sales"].max())
            fig2.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                      mode="lines", name="Perfect",
                                      line=dict(color="red", dash="dash")))
            fig2.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">📊 Residual Distribution</div>',
                        unsafe_allow_html=True)
            residuals = sample["Weekly_Sales"] - sample["Predicted_Sales"]
            fig3 = px.histogram(residuals, nbins=60, color_discrete_sequence=["#4CAF50"],
                                labels={"value": "Residual ($)", "count": "Count"})
            fig3.update_layout(height=350, showlegend=False,
                               margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Run the training pipeline first: `python scripts/train_pipeline.py`")


# ── Tab: Inventory ─────────────────────────────────────────────────────────────
def tab_inventory(decisions_df):
    st.markdown('<div class="section-header">📦 Inventory Status Overview</div>',
                unsafe_allow_html=True)

    # Status donut
    c1, c2 = st.columns([1, 2])
    with c1:
        status_counts = decisions_df["Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        color_map = {
            "OK":        "#27ae60",
            "LOW_STOCK": "#e67e22",
            "OVERSTOCK": "#8e44ad",
            "CRITICAL":  "#c0392b",
        }
        fig_donut = px.pie(
            status_counts, values="Count", names="Status",
            hole=0.5, title="Inventory Status Distribution",
            color="Status", color_discrete_map=color_map,
        )
        fig_donut.update_layout(height=320, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        # Days of cover distribution
        fig_cover = px.histogram(
            decisions_df[decisions_df["Days_Of_Cover"] < 100],
            x="Days_Of_Cover", color="Status",
            nbins=50, barmode="overlay",
            color_discrete_map=color_map,
            labels={"Days_Of_Cover": "Days of Cover", "count": "SKU Count"},
            title="Days-of-Cover Distribution by Status",
        )
        fig_cover.add_vline(x=14, line_dash="dash", line_color="red",
                            annotation_text="2-week threshold")
        fig_cover.update_layout(height=320, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_cover, use_container_width=True)

    # Heatmap: status by Store x Dept
    st.markdown('<div class="section-header">🗺️ Inventory Heatmap (Store × Dept)</div>',
                unsafe_allow_html=True)
    status_numeric = {"OK": 3, "OVERSTOCK": 2, "LOW_STOCK": 1, "CRITICAL": 0}
    pivot = decisions_df.pivot_table(
        index="Store", columns="Dept",
        values="Status", aggfunc=lambda x: status_numeric.get(x.iloc[0], 2)
    )
    fig_heat = px.imshow(
        pivot, color_continuous_scale=["#c0392b", "#e67e22", "#8e44ad", "#27ae60"],
        aspect="auto", title="Inventory Health (Red=Critical, Green=OK)",
        labels={"color": "Status Score"},
    )
    fig_heat.update_layout(height=400, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Decision table
    st.markdown('<div class="section-header">📋 Reorder Decision Table</div>',
                unsafe_allow_html=True)

    urgent = decisions_df[decisions_df["Urgency"].isin(["HIGH", "MEDIUM"])].copy()
    urgent = urgent.sort_values(["Urgency", "Days_Of_Cover"])

    def color_status(val):
        colors = {"CRITICAL": "background-color:#fadbd8",
                  "LOW_STOCK": "background-color:#fdebd0",
                  "OVERSTOCK": "background-color:#e8daef",
                  "OK":        "background-color:#d5f5e3"}
        return colors.get(val, "")

    display_cols = ["Store", "Dept", "Predicted_Demand", "Current_Stock",
                    "Reorder_Point", "Suggested_Order", "Days_Of_Cover",
                    "Status", "Urgency"]
    styled = urgent[display_cols].style.applymap(color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, height=400)


# ── Tab: Model Performance ─────────────────────────────────────────────────────
def tab_model_performance():
    st.markdown('<div class="section-header">🧠 Model Comparison</div>',
                unsafe_allow_html=True)

    results_path = Path("data/processed/model_comparison.csv")
    if results_path.exists():
        results = pd.read_csv(results_path, index_col=0)
        st.dataframe(
            results.style.highlight_min(axis=0, subset=["RMSE","MAE","MAPE","WMAE"],
                                         color="#d5f5e3")
                         .highlight_max(axis=0, subset=["R2"], color="#d5f5e3")
                         .format("{:.2f}"),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                results.reset_index(), x="Model", y="RMSE",
                color="RMSE", color_continuous_scale="RdYlGn_r",
                title="RMSE by Model (lower is better)",
            )
            fig.update_layout(height=300, showlegend=False,
                              margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.bar(
                results.reset_index(), x="Model", y="R2",
                color="R2", color_continuous_scale="RdYlGn",
                title="R² Score by Model (higher is better)",
            )
            fig2.update_layout(height=300, showlegend=False,
                               margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        # Training metadata
        meta_path = Path("models/training_meta.json")
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
            st.info(
                f"**Best Model:** {meta['best_model']}  |  "
                f"Train rows: {meta['train_rows']:,}  |  "
                f"Test rows: {meta['test_rows']:,}  |  "
                f"RMSE: {meta['metrics']['RMSE']:.2f}  |  "
                f"R²: {meta['metrics']['R2']:.4f}"
            )
    else:
        st.warning(
            "No model comparison found. Run the training pipeline first:\n\n"
            "```bash\npython scripts/train_pipeline.py\n```"
        )

    # Feature importance (if available)
    fi_path = Path("data/processed/feature_importance.csv")
    if fi_path.exists():
        st.markdown('<div class="section-header">📊 Feature Importances (XGBoost)</div>',
                    unsafe_allow_html=True)
        fi_df = pd.read_csv(fi_path).head(20).sort_values("Importance")
        fig3 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Blues")
        fig3.update_layout(height=500, showlegend=False,
                           margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig3, use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Data not found. Run: `python data/generate_sample_data.py`")
        st.stop()

    # Sidebar filters
    selected_store, selected_dept, date_range, service_level = render_sidebar(df)

    # Apply filters
    df_filtered = df.copy()
    df_filtered["Date"] = pd.to_datetime(df_filtered["Date"])

    if selected_store != "All":
        df_filtered = df_filtered[df_filtered["Store"] == selected_store]
    if selected_dept != "All":
        df_filtered = df_filtered[df_filtered["Dept"] == selected_dept]
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered["Date"].dt.date >= date_range[0]) &
            (df_filtered["Date"].dt.date <= date_range[1])
        ]

    # Load model + generate forecasts (lazy)
    model = load_best_model()
    try:
        df_feat = load_features(df)
        if selected_store != "All":
            df_feat_filtered = df_feat[df_feat["Store"] == selected_store]
        else:
            df_feat_filtered = df_feat
        df_forecast = get_forecasts(model, df_feat_filtered)
    except Exception:
        df_forecast = df_filtered.copy()
        df_forecast["Predicted_Sales"] = df_forecast["Weekly_Sales"]

    # Inventory decisions
    inventory_path = Path("data/raw/inventory_snapshot.csv")
    if inventory_path.exists():
        inventory_df = pd.read_csv(inventory_path)
        if selected_store != "All":
            inventory_df = inventory_df[inventory_df["Store"] == selected_store]
    else:
        inventory_df = pd.DataFrame()

    latest_forecast = (
        df_forecast.sort_values("Date")
        .groupby(["Store", "Dept"])
        .last()
        .reset_index()[["Store", "Dept", "Predicted_Sales"]]
        if "Predicted_Sales" in df_forecast.columns
        else df_filtered.groupby(["Store", "Dept"])["Weekly_Sales"].mean()
                        .reset_index().rename(columns={"Weekly_Sales": "Predicted_Sales"})
    )

    if inventory_df.empty:
        inventory_df = latest_forecast[["Store", "Dept"]].copy()
        inventory_df["Current_Stock"] = (latest_forecast["Predicted_Sales"] * 1.3).astype(int).values
        inventory_df["Lead_Time_Days"] = 7

    decisions_df = generate_inventory_decisions(
        forecast_df  = latest_forecast,
        inventory_df = inventory_df,
        history_df   = df_filtered,
        service_level= service_level,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("📦 Retail Demand Forecasting & Inventory Optimization")
    st.caption(f"Data: {df['Date'].min().date()} → {df['Date'].max().date()}  |  "
               f"Stores: {df['Store'].nunique()}  |  Depts: {df['Dept'].nunique()}")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    render_kpis(df_filtered, decisions_df)
    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "📈 Overview",
        "🔮 Demand Forecast",
        "📦 Inventory Management",
        "🧠 Model Performance",
    ])

    with t1:
        tab_overview(df_filtered)
    with t2:
        tab_forecast(df_filtered, df_forecast)
    with t3:
        tab_inventory(decisions_df)
    with t4:
        tab_model_performance()


if __name__ == "__main__":
    main()
