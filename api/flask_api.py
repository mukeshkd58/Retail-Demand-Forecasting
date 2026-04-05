"""
api/flask_api.py
-----------------
RESTful API for the Retail Demand Forecasting system.

Endpoints:
  GET  /health                        → service health check
  GET  /api/v1/stores                 → list all stores
  POST /api/v1/predict                → predict demand for a store/dept
  POST /api/v1/inventory/decisions    → get inventory recommendations
  GET  /api/v1/summary                → global KPI summary

Run locally:
    python api/flask_api.py
    # or with gunicorn:
    gunicorn -w 4 -b 0.0.0.0:5000 "api.flask_api:create_app()"
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request, abort
import pandas as pd
import numpy as np

from src.data_preprocessing import load_raw_data, clean_data
from src.feature_engineering import build_features, FEATURE_COLS
from src.business_logic import generate_inventory_decisions, summary_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")

# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # ── Load resources once at startup ──────────────────────────────────────
    try:
        _df_raw   = load_raw_data("data/raw")
        _df_clean = clean_data(_df_raw)
        _df_feat  = build_features(_df_clean, drop_na=True)
        app.config["DF_CLEAN"] = _df_clean
        app.config["DF_FEAT"]  = _df_feat
        logger.info("Data loaded — %d rows", len(_df_clean))
    except Exception as exc:
        logger.warning("Could not load data at startup: %s", exc)
        app.config["DF_CLEAN"] = pd.DataFrame()
        app.config["DF_FEAT"]  = pd.DataFrame()

    try:
        from src.models import load_model
        _model = load_model("best_model", Path("models"))
        app.config["MODEL"] = _model
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.warning("Model not loaded: %s", exc)
        app.config["MODEL"] = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_df() -> pd.DataFrame:
        return app.config.get("DF_CLEAN", pd.DataFrame())

    def _get_feat() -> pd.DataFrame:
        return app.config.get("DF_FEAT", pd.DataFrame())

    def _get_model():
        return app.config.get("MODEL")

    def _predict_sales(store: int, dept: int, weeks_ahead: int = 1) -> list[dict]:
        """Generate demand forecast for a specific store-dept pair."""
        df_feat = _get_feat()
        model   = _get_model()

        subset = df_feat[
            (df_feat["Store"] == store) & (df_feat["Dept"] == dept)
        ].sort_values("Date")

        if subset.empty:
            abort(404, description=f"Store {store} / Dept {dept} not found in dataset.")

        latest = subset.iloc[[-1]].copy()   # most recent row

        if model is not None:
            from src.models import predict
            pred_value = float(predict(model, latest)[0])
        else:
            # Fallback: rolling mean
            pred_value = float(latest["Roll_4w_Mean"].iloc[0])

        results = []
        base_date = pd.to_datetime(latest["Date"].iloc[0])
        for w in range(1, weeks_ahead + 1):
            forecast_date = base_date + pd.Timedelta(weeks=w)
            # Apply simple decay for multi-week ahead
            noise_factor  = 1.0 + np.random.normal(0, 0.03)
            results.append({
                "forecast_date": str(forecast_date.date()),
                "week_ahead":    w,
                "predicted_sales": round(pred_value * noise_factor, 2),
            })
        return results

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.route("/health", methods=["GET"])
    def health():
        df   = _get_df()
        model = _get_model()
        return jsonify({
            "status":       "ok",
            "timestamp":    datetime.utcnow().isoformat() + "Z",
            "data_loaded":  not df.empty,
            "model_loaded": model is not None,
            "n_rows":       len(df),
        })

    @app.route("/api/v1/stores", methods=["GET"])
    def list_stores():
        """Return all store IDs and metadata."""
        df = _get_df()
        if df.empty:
            return jsonify({"stores": []})

        stores = (
            df.groupby("Store")
            .agg(
                n_depts    = ("Dept",        "nunique"),
                type       = ("Type",        "first")  if "Type" in df.columns else ("Store", "count"),
                size       = ("Size",        "first")  if "Size" in df.columns else ("Store", "count"),
                total_sales= ("Weekly_Sales","sum"),
            )
            .reset_index()
            .rename(columns={"Store": "store_id"})
        )
        return jsonify({
            "count":  len(stores),
            "stores": stores.to_dict(orient="records"),
        })

    @app.route("/api/v1/departments", methods=["GET"])
    def list_departments():
        """Return department IDs (optionally filtered by store)."""
        df    = _get_df()
        store = request.args.get("store", type=int)
        if store:
            df = df[df["Store"] == store]

        depts = sorted(df["Dept"].unique().tolist())
        return jsonify({"store": store, "departments": depts})

    @app.route("/api/v1/predict", methods=["POST"])
    def predict_demand():
        """
        Predict future demand.

        Request body (JSON):
          {
            "store":       1,
            "dept":        5,
            "weeks_ahead": 4    // optional, default 1
          }

        Response:
          {
            "store": 1, "dept": 5,
            "forecasts": [
              {"forecast_date": "2024-01-05", "week_ahead": 1, "predicted_sales": 12345.67},
              ...
            ]
          }
        """
        body = request.get_json(force=True, silent=True)
        if not body:
            abort(400, description="Request body must be valid JSON.")

        store       = body.get("store")
        dept        = body.get("dept")
        weeks_ahead = int(body.get("weeks_ahead", 1))

        if store is None or dept is None:
            abort(400, description="Fields 'store' and 'dept' are required.")
        if not (1 <= weeks_ahead <= 52):
            abort(400, description="'weeks_ahead' must be between 1 and 52.")

        forecasts = _predict_sales(int(store), int(dept), weeks_ahead)

        return jsonify({
            "store":     store,
            "dept":      dept,
            "forecasts": forecasts,
        })

    @app.route("/api/v1/inventory/decisions", methods=["POST"])
    def inventory_decisions():
        """
        Return inventory reorder decisions for given store(s).

        Request body (JSON):
          {
            "stores":          [1, 2],          // optional; omit for all stores
            "service_level":   0.95             // optional, default 0.95
          }
        """
        body          = request.get_json(force=True, silent=True) or {}
        stores_filter = body.get("stores")
        service_level = float(body.get("service_level", 0.95))

        df_feat = _get_feat()
        df_clean= _get_df()
        model   = _get_model()

        if df_feat.empty:
            abort(503, description="Data not available. Check server logs.")

        if stores_filter:
            df_feat  = df_feat[df_feat["Store"].isin(stores_filter)]
            df_clean = df_clean[df_clean["Store"].isin(stores_filter)]

        # Latest row per store-dept
        latest = (
            df_feat.sort_values("Date")
            .groupby(["Store", "Dept"])
            .last()
            .reset_index()
        )

        if model:
            from src.models import predict
            latest["Predicted_Sales"] = predict(model, latest)
        else:
            latest["Predicted_Sales"] = latest.get("Roll_4w_Mean",
                                                    latest["Weekly_Sales"])

        # Synthetic inventory for demo
        inventory_df = latest[["Store", "Dept"]].copy()
        inventory_df["Current_Stock"]  = (latest["Predicted_Sales"] * 1.3).astype(int).values
        inventory_df["Lead_Time_Days"] = 7

        decisions = generate_inventory_decisions(
            forecast_df  = latest[["Store", "Dept", "Predicted_Sales"]],
            inventory_df = inventory_df,
            history_df   = df_clean,
            service_level= service_level,
        )

        summary = summary_report(decisions)

        return jsonify({
            "service_level": service_level,
            "summary":       summary,
            "decisions":     decisions.to_dict(orient="records"),
        })

    @app.route("/api/v1/summary", methods=["GET"])
    def global_summary():
        """Return high-level KPIs."""
        df = _get_df()
        if df.empty:
            abort(503, description="Data not loaded.")

        return jsonify({
            "n_stores":     int(df["Store"].nunique()),
            "n_departments":int(df["Dept"].nunique()),
            "date_range": {
                "from": str(pd.to_datetime(df["Date"]).min().date()),
                "to":   str(pd.to_datetime(df["Date"]).max().date()),
            },
            "total_sales":   round(float(df["Weekly_Sales"].sum()), 2),
            "avg_weekly_sales": round(float(df["Weekly_Sales"].mean()), 2),
        })

    # ── Error handlers ────────────────────────────────────────────────────────

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad Request", "message": str(e.description)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not Found", "message": str(e.description)}), 404

    @app.errorhandler(503)
    def service_unavailable(e):
        return jsonify({"error": "Service Unavailable",
                        "message": str(e.description)}), 503

    return app


# ── Dev runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
