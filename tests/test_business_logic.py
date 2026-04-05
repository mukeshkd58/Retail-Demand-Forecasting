"""
tests/test_business_logic.py
-----------------------------
Unit tests for the inventory optimization / business logic module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest

from src.business_logic import (
    compute_safety_stock,
    compute_reorder_point,
    compute_eoq,
    classify_inventory_status,
    generate_inventory_decisions,
    summary_report,
)


class TestSafetyStock:
    def test_positive_output(self):
        ss = compute_safety_stock(std_demand=500, lead_time_days=7)
        assert ss >= 0

    def test_higher_service_level_gives_more_stock(self):
        ss_95 = compute_safety_stock(500, 7, service_level=0.95)
        ss_99 = compute_safety_stock(500, 7, service_level=0.99)
        assert ss_99 >= ss_95

    def test_zero_std_gives_zero(self):
        ss = compute_safety_stock(std_demand=0, lead_time_days=7)
        assert ss == 0


class TestReorderPoint:
    def test_greater_than_safety_stock(self):
        ss  = compute_safety_stock(300, 7)
        rop = compute_reorder_point(avg_demand=2000, lead_time_days=7, safety_stock=ss)
        assert rop >= ss

    def test_non_negative(self):
        rop = compute_reorder_point(avg_demand=1000, lead_time_days=5, safety_stock=100)
        assert rop >= 0


class TestEOQ:
    def test_positive_result(self):
        eoq = compute_eoq(annual_demand=52_000)
        assert eoq > 0

    def test_zero_demand_returns_zero(self):
        eoq = compute_eoq(annual_demand=0)
        assert eoq == 0

    def test_classic_formula(self):
        # D=1000, S=50, h=0.25*10=2.5  → EOQ=sqrt(2*1000*50/2.5)=200
        eoq = compute_eoq(annual_demand=1000, order_cost=50,
                          holding_cost_rate=0.25, unit_cost=10)
        assert abs(eoq - 200) <= 1   # within rounding


class TestClassifyInventoryStatus:
    def test_critical_when_stock_zero(self):
        status, urgency = classify_inventory_status(
            current_stock=0, reorder_point=500, predicted_demand=1000
        )
        assert status == "CRITICAL"
        assert urgency == "HIGH"

    def test_low_stock_below_rop(self):
        status, _ = classify_inventory_status(
            current_stock=300, reorder_point=500, predicted_demand=800
        )
        assert status == "LOW_STOCK"

    def test_overstock_when_way_above_demand(self):
        status, _ = classify_inventory_status(
            current_stock=50_000, reorder_point=500, predicted_demand=1000
        )
        assert status == "OVERSTOCK"

    def test_ok_status(self):
        status, urgency = classify_inventory_status(
            current_stock=2000, reorder_point=800, predicted_demand=1000
        )
        assert status == "OK"
        assert urgency == "NONE"


class TestGenerateDecisions:
    @pytest.fixture
    def mini_data(self):
        forecast = pd.DataFrame({
            "Store": [1, 1, 2],
            "Dept":  [1, 2, 1],
            "Predicted_Sales": [5000.0, 3000.0, 7000.0],
        })
        inventory = pd.DataFrame({
            "Store": [1, 1, 2],
            "Dept":  [1, 2, 1],
            "Current_Stock":  [4000, 100, 15000],
            "Lead_Time_Days": [7,    7,   7],
        })
        history_records = []
        for store, dept in [(1,1),(1,2),(2,1)]:
            for w in range(52):
                history_records.append({
                    "Store": store, "Dept": dept,
                    "Weekly_Sales": np.random.uniform(2000, 8000)
                })
        history = pd.DataFrame(history_records)
        return forecast, inventory, history

    def test_returns_dataframe(self, mini_data):
        forecast, inventory, history = mini_data
        result = generate_inventory_decisions(forecast, inventory, history)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_skus(self, mini_data):
        forecast, inventory, history = mini_data
        result = generate_inventory_decisions(forecast, inventory, history)
        assert len(result) == 3

    def test_required_columns_present(self, mini_data):
        forecast, inventory, history = mini_data
        result = generate_inventory_decisions(forecast, inventory, history)
        for col in ["Status", "Suggested_Order", "Days_Of_Cover", "Urgency"]:
            assert col in result.columns

    def test_low_stock_triggers_order(self, mini_data):
        forecast, inventory, history = mini_data
        result = generate_inventory_decisions(forecast, inventory, history)
        # Dept 2 has Current_Stock=100 which is very low → should suggest order
        dept2 = result[(result["Store"] == 1) & (result["Dept"] == 2)]
        assert dept2["Suggested_Order"].iloc[0] > 0


class TestSummaryReport:
    def test_all_keys_present(self):
        df = pd.DataFrame({
            "Status":         ["OK", "LOW_STOCK", "CRITICAL", "OVERSTOCK"],
            "Urgency":        ["NONE", "HIGH", "HIGH", "LOW"],
            "Suggested_Order":[0,     500,   1000,  0],
        })
        summary = summary_report(df)
        for key in ["total_skus", "critical", "low_stock", "overstock", "ok",
                    "total_order_value", "high_urgency_skus"]:
            assert key in summary

    def test_counts_correct(self):
        df = pd.DataFrame({
            "Status":         ["OK", "LOW_STOCK", "CRITICAL"],
            "Urgency":        ["NONE", "MEDIUM", "HIGH"],
            "Suggested_Order":[0, 200, 400],
        })
        summary = summary_report(df)
        assert summary["total_skus"]  == 3
        assert summary["critical"]    == 1
        assert summary["low_stock"]   == 1
        assert summary["ok"]          == 1
