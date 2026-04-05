#!/usr/bin/env python3
"""
docs/generate_report.py
------------------------
Generates a professional project report PDF using ReportLab.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas
from datetime import datetime

# ── Colours ────────────────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#1a2e4a")
BLUE    = colors.HexColor("#2d6a9f")
TEAL    = colors.HexColor("#17a589")
LIGHT   = colors.HexColor("#eaf4fb")
GRAY    = colors.HexColor("#555555")
WHITE   = colors.white

PAGE_W, PAGE_H = A4
OUT_PATH = Path(__file__).parent / "project_report.pdf"


# ── Custom canvas (header + footer on every page) ──────────────────────────────
class ReportCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        page_num = self._pageNumber

        # Header bar (skip title page)
        if page_num > 1:
            self.setFillColor(NAVY)
            self.rect(0, PAGE_H - 1.2*cm, PAGE_W, 1.2*cm, fill=1, stroke=0)
            self.setFillColor(WHITE)
            self.setFont("Helvetica-Bold", 8)
            self.drawString(1.5*cm, PAGE_H - 0.85*cm,
                            "Retail Demand Forecasting & Inventory Optimization")
            self.setFont("Helvetica", 8)
            self.drawRightString(PAGE_W - 1.5*cm, PAGE_H - 0.85*cm,
                                 "Mukesh Kumar | Politecnico di Torino")

        # Footer
        self.setFillColor(NAVY)
        self.rect(0, 0, PAGE_W, 0.9*cm, fill=1, stroke=0)
        self.setFillColor(WHITE)
        self.setFont("Helvetica", 7)
        self.drawString(1.5*cm, 0.3*cm, f"Page {page_num} of {page_count}")
        self.drawCentredString(PAGE_W/2, 0.3*cm,
                               "Confidential – Portfolio Project")
        self.drawRightString(PAGE_W - 1.5*cm, 0.3*cm,
                             datetime.now().strftime("%B %Y"))


# ── Style helpers ──────────────────────────────────────────────────────────────
def get_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "rpt_title", parent=base["Title"],
            fontSize=28, textColor=WHITE, alignment=TA_CENTER,
            spaceAfter=6, fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "rpt_subtitle", parent=base["Normal"],
            fontSize=14, textColor=LIGHT, alignment=TA_CENTER,
            spaceAfter=4, fontName="Helvetica",
        ),
        "h1": ParagraphStyle(
            "rpt_h1", parent=base["Heading1"],
            fontSize=16, textColor=NAVY, spaceBefore=18, spaceAfter=8,
            fontName="Helvetica-Bold", borderPad=4,
        ),
        "h2": ParagraphStyle(
            "rpt_h2", parent=base["Heading2"],
            fontSize=13, textColor=BLUE, spaceBefore=12, spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "rpt_body", parent=base["Normal"],
            fontSize=10, textColor=GRAY, leading=16, alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "rpt_bullet", parent=base["Normal"],
            fontSize=10, textColor=GRAY, leading=15,
            leftIndent=16, spaceAfter=4,
            bulletIndent=6,
        ),
        "code": ParagraphStyle(
            "rpt_code", parent=base["Code"],
            fontSize=8.5, textColor=colors.HexColor("#1a1a1a"),
            backColor=colors.HexColor("#f4f4f4"),
            fontName="Courier", leading=13,
            leftIndent=12, rightIndent=12,
            spaceBefore=4, spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "rpt_caption", parent=base["Normal"],
            fontSize=8, textColor=GRAY, alignment=TA_CENTER,
            spaceAfter=8, fontName="Helvetica-Oblique",
        ),
        "kpi_label": ParagraphStyle(
            "rpt_kpi_label", parent=base["Normal"],
            fontSize=9, textColor=GRAY, alignment=TA_CENTER, fontName="Helvetica",
        ),
        "kpi_value": ParagraphStyle(
            "rpt_kpi_value", parent=base["Normal"],
            fontSize=20, textColor=BLUE, alignment=TA_CENTER, fontName="Helvetica-Bold",
        ),
    }
    return styles


def section_rule():
    return HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=8)


def kpi_table(kpis: list[tuple]) -> Table:
    """kpis = [(label, value), ...]"""
    st = get_styles()
    data = [[Paragraph(v, st["kpi_value"]) for _, v in kpis],
            [Paragraph(l, st["kpi_label"]) for l, _ in kpis]]
    tbl = Table(data, colWidths=[4*cm]*len(kpis))
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), LIGHT),
        ("BACKGROUND", (0,1), (-1,1), WHITE),
        ("BOX",        (0,0), (-1,-1), 0.5, BLUE),
        ("INNERGRID",  (0,0), (-1,-1), 0.3, colors.HexColor("#c0d8ee")),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    return tbl


# ── Build document ─────────────────────────────────────────────────────────────
def build_pdf():
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=2.5*cm,  bottomMargin=1.8*cm,
    )

    st    = get_styles()
    story = []

    # ── TITLE PAGE ─────────────────────────────────────────────────────────────
    # Navy background block (simulate with table)
    title_data = [[
        Paragraph("📦 Retail Demand Forecasting", st["title"]),
    ],[
        Paragraph("&amp; Inventory Optimization System", st["title"]),
    ],[
        Spacer(1, 0.4*cm),
    ],[
        Paragraph("End-to-End Machine Learning Portfolio Project", st["subtitle"]),
    ],[
        Paragraph("XGBoost · Random Forest · Prophet · Streamlit · Flask · Docker",
                  st["subtitle"]),
    ]]
    title_tbl = Table(title_data, colWidths=[PAGE_W - 3.6*cm])
    title_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 20),
        ("RIGHTPADDING",  (0,0), (-1,-1), 20),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(Spacer(1, 1.5*cm))
    story.append(title_tbl)
    story.append(Spacer(1, 1.2*cm))

    # Author block
    author_data = [[
        Paragraph("<b>Author</b>", st["body"]),
        Paragraph("Mukesh Kumar", st["body"]),
    ],[
        Paragraph("<b>Degree</b>", st["body"]),
        Paragraph("M.E. Georesource &amp; Geoenergy Engineering", st["body"]),
    ],[
        Paragraph("<b>University</b>", st["body"]),
        Paragraph("Politecnico di Torino, Italy", st["body"]),
    ],[
        Paragraph("<b>Date</b>", st["body"]),
        Paragraph(datetime.now().strftime("%B %Y"), st["body"]),
    ],[
        Paragraph("<b>GitHub</b>", st["body"]),
        Paragraph("github.com/mukeshkd58/retail-demand-forecasting", st["body"]),
    ],[
        Paragraph("<b>Portfolio</b>", st["body"]),
        Paragraph("mukeshkd58.github.io", st["body"]),
    ]]
    author_tbl = Table(author_data, colWidths=[4*cm, 12*cm])
    author_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), LIGHT),
        ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("BOX",        (0,0), (-1,-1), 0.5, BLUE),
        ("INNERGRID",  (0,0), (-1,-1), 0.3, colors.HexColor("#c0d8ee")),
    ]))
    story.append(author_tbl)
    story.append(PageBreak())

    # ── 1. EXECUTIVE SUMMARY ───────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", st["h1"]))
    story.append(section_rule())
    story.append(Paragraph(
        "This project delivers a production-ready Retail Demand Forecasting and Inventory "
        "Optimization system. Retail companies lose an estimated 1.75 trillion USD annually "
        "from inventory imbalances — overstocking ties up capital and leads to spoilage, "
        "while stockouts result in lost sales and customer dissatisfaction.",
        st["body"]
    ))
    story.append(Paragraph(
        "The system uses machine learning models trained on 39,000 rows of Walmart-style "
        "weekly sales data to predict future demand, then applies classical inventory "
        "optimization theory (EOQ, Safety Stock, Reorder Point) to generate actionable "
        "procurement decisions for every Store-Department (SKU) combination.",
        st["body"]
    ))
    story.append(Spacer(1, 0.4*cm))

    # KPI row
    story.append(kpi_table([
        ("Training Rows", "28,600"),
        ("Stores",        "10"),
        ("Departments",   "20"),
        ("Best R²",       "0.9648"),
        ("Best RMSE",     "2,901"),
        ("Tests Passing", "36/36"),
    ]))
    story.append(Spacer(1, 0.5*cm))

    # ── 2. PROBLEM STATEMENT ───────────────────────────────────────────────────
    story.append(Paragraph("2. Problem Statement", st["h1"]))
    story.append(section_rule())
    story.append(Paragraph(
        "Traditional retail inventory management relies on manual rules of thumb and "
        "historical averages that fail to capture complex temporal patterns including "
        "seasonality, holiday effects, promotional markdowns, and macroeconomic drivers.",
        st["body"]
    ))

    problems = [
        ("Overstock Risk",   "Excess inventory → capital tied up, storage costs, product spoilage / obsolescence."),
        ("Stockout Risk",    "Insufficient inventory → lost sales revenue, poor customer experience, brand damage."),
        ("Manual Processes", "Spreadsheet-based forecasting → slow, error-prone, non-scalable."),
        ("No Uncertainty",   "Point estimates without safety buffers → stockouts during demand spikes."),
    ]
    prob_data = [["Issue", "Business Impact"]] + [[p, d] for p, d in problems]
    prob_tbl  = Table(prob_data, colWidths=[5*cm, 11*cm])
    prob_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("BACKGROUND",    (0,1), (-1,-1), LIGHT),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("BOX",           (0,0), (-1,-1), 0.5, BLUE),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#c0d8ee")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LIGHT]),
    ]))
    story.append(Spacer(1, 0.3*cm))
    story.append(prob_tbl)

    # ── 3. DATASET ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("3. Dataset", st["h1"]))
    story.append(section_rule())
    story.append(Paragraph(
        "The project uses a Walmart-style retail dataset (based on the Kaggle Walmart "
        "Recruiting - Store Sales Forecasting competition). The synthetic generator "
        "replicates all statistical properties of the real dataset including seasonal "
        "patterns, holiday effects, store-type heterogeneity, and promotional markdowns.",
        st["body"]
    ))

    ds_data = [
        ["Attribute", "Value"],
        ["Total Rows",   "39,000"],
        ["Stores",       "10 (Types: A, B, C)"],
        ["Departments",  "20 per store"],
        ["Date Range",   "Feb 2010 — Oct 2013 (195 weeks)"],
        ["Target",       "Weekly_Sales (continuous, USD)"],
        ["Features",     "Date, Store, Dept, IsHoliday, Temperature, Fuel_Price,"],
        ["",             "MarkDown1-5, CPI, Unemployment, Type, Size"],
        ["Train/Test",   "Last 26 weeks held out (temporal split)"],
    ]
    ds_tbl = Table(ds_data, colWidths=[5*cm, 11*cm])
    ds_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("BOX",           (0,0), (-1,-1), 0.5, BLUE),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#c0d8ee")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LIGHT]),
        ("FONTNAME",      (0,1), (0,-1),  "Helvetica-Bold"),
    ]))
    story.append(Spacer(1, 0.3*cm))
    story.append(ds_tbl)
    story.append(PageBreak())

    # ── 4. METHODOLOGY ─────────────────────────────────────────────────────────
    story.append(Paragraph("4. Methodology", st["h1"]))
    story.append(section_rule())

    story.append(Paragraph("4.1 Data Preprocessing", st["h2"]))
    for step in [
        "Duplicate removal (Store, Dept, Date composite key)",
        "Negative sales clamped to zero (holiday return corrections)",
        "MarkDown NaN → 0 (no promotion active during those weeks)",
        "Remaining numeric NaN → column median imputation",
        "Temporal sort: ascending by Store → Dept → Date",
    ]:
        story.append(Paragraph(f"• {step}", st["bullet"]))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("4.2 Feature Engineering", st["h2"]))
    feat_data = [
        ["Category", "Features Created", "Rationale"],
        ["Calendar",   "Year, Month, Week, Quarter, Season_Code",  "Capture time-of-year demand"],
        ["Cyclical",   "Month_Sin/Cos, Week_Sin/Cos",               "Avoid Dec→Jan discontinuity"],
        ["Lag",        "Lag_1w to Lag_52w (8 lags)",                "Autoregressive signal"],
        ["Rolling",    "Roll_4/8/13w Mean & Std",                   "Trend smoothing"],
        ["Promotion",  "MarkDown_Total, Count, Max",                 "Aggregate 5 markdown cols"],
        ["Store Meta", "Type_Code (A=0,B=1,C=2), Size",             "Store-level fixed effects"],
        ["External",   "Temperature, Fuel_Price, CPI, Unemployment","Macroeconomic drivers"],
    ]
    feat_tbl = Table(feat_data, colWidths=[4*cm, 7*cm, 5*cm])
    feat_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("BOX",           (0,0), (-1,-1), 0.5, BLUE),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#c0d8ee")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LIGHT]),
        ("FONTNAME",      (0,1), (0,-1),  "Helvetica-Bold"),
    ]))
    story.append(Spacer(1, 0.2*cm))
    story.append(feat_tbl)

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("4.3 Machine Learning Models", st["h2"]))
    story.append(Paragraph(
        "Three models were trained and compared, progressing from baseline to state-of-the-art:",
        st["body"]
    ))

    models_info = [
        ("Linear Regression", "Baseline", "Scaled inputs via StandardScaler Pipeline. "
         "Establishes minimum acceptable performance."),
        ("Random Forest", "Ensemble",
         "200 trees, max_depth=12, min_samples_leaf=5. "
         "Captures non-linear interactions, robust to outliers."),
        ("XGBoost", "Gradient Boosting (Best)",
         "800 estimators, lr=0.05, max_depth=7, subsample=0.8. "
         "Regularized with L1/L2. Best generalization on test set."),
    ]
    for name, category, desc in models_info:
        story.append(Paragraph(f"<b>{name}</b> ({category})", st["h2"]))
        story.append(Paragraph(desc, st["body"]))

    story.append(PageBreak())

    # ── 5. RESULTS ─────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Model Results", st["h1"]))
    story.append(section_rule())

    results_data = [
        ["Model", "RMSE ↓", "MAE ↓", "MAPE ↓", "R² ↑", "WMAE ↓", "Winner"],
        ["Linear Regression", "3,042", "2,148", "7.90%", "0.9613", "2,229", ""],
        ["Random Forest",     "3,029", "2,089", "7.51%", "0.9616", "2,206", ""],
        ["XGBoost",           "2,901", "2,015", "7.23%", "0.9648", "2,087", "✅"],
    ]
    r_tbl = Table(results_data, colWidths=[4.5*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm, 1.5*cm])
    r_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("BACKGROUND",    (0,3), (-1,3),  colors.HexColor("#d5f5e3")),
        ("FONTNAME",      (0,3), (-1,3),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("BOX",           (0,0), (-1,-1), 0.5, BLUE),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#c0d8ee")),
        ("ROWBACKGROUNDS",(0,1), (-1,2), [WHITE, LIGHT]),
    ]))
    story.append(r_tbl)
    story.append(Paragraph(
        "Table 1: Model comparison on 26-week hold-out test set. "
        "WMAE = Weighted MAE (holiday weeks weighted 5x, Walmart competition metric).",
        st["caption"]
    ))

    story.append(Paragraph("Key Findings:", st["h2"]))
    for finding in [
        "XGBoost outperforms all baselines on every metric, achieving R² = 0.9648.",
        "Lag_52w (same week last year) is the single most predictive feature (51.2% importance).",
        "Rolling 4-week mean is the second most important (20.1%) — captures recent trend.",
        "IsHoliday adds 2.1% importance; holiday weeks require separate weighting (WMAE).",
        "XGBoost MAPE of 7.23% means predictions are within 7% of actual on average.",
    ]:
        story.append(Paragraph(f"• {finding}", st["bullet"]))

    # ── 6. BUSINESS LOGIC ─────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("6. Inventory Optimization Logic", st["h1"]))
    story.append(section_rule())
    story.append(Paragraph(
        "Raw demand forecasts are transformed into procurement decisions using "
        "classical inventory management theory:", st["body"]
    ))

    formulas = [
        ("Safety Stock", "SS = Z x sigma_demand x sqrt(lead_time_weeks)",
         "Buffer stock to absorb demand variability. Z=1.645 at 95% service level."),
        ("Reorder Point",
         "ROP = (avg_daily_demand x lead_time_days) + SS",
         "Trigger point for placing a new order."),
        ("Economic Order Quantity",
         "EOQ = sqrt(2 x D x S / H)",
         "Optimal order size minimising total ordering + holding costs."),
    ]
    for name, formula, desc in formulas:
        story.append(Paragraph(f"<b>{name}</b>", st["h2"]))
        story.append(Paragraph(formula, st["code"]))
        story.append(Paragraph(desc, st["body"]))

    story.append(Paragraph("SKU Status Classification:", st["h2"]))
    story.append(Paragraph(
        "Each Store-Department is classified into one of four states:", st["body"]
    ))
    status_data = [
        ["Status", "Condition", "Action", "Urgency"],
        ["CRITICAL",   "stock = 0 or stock < demand x 0.25", "Emergency order", "HIGH"],
        ["LOW_STOCK",  "stock <= Reorder Point",              "Place order (EOQ)", "MEDIUM/HIGH"],
        ["OVERSTOCK",  "stock > demand x 3",                 "Pause ordering",    "LOW"],
        ["OK",         "ROP < stock <= 3x demand",            "Monitor",           "NONE"],
    ]
    s_tbl = Table(status_data, colWidths=[3.2*cm, 6.5*cm, 3.5*cm, 2.8*cm])
    s_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("BACKGROUND",    (0,1), (-1,1),  colors.HexColor("#fadbd8")),  # CRITICAL
        ("BACKGROUND",    (0,2), (-1,2),  colors.HexColor("#fdebd0")),  # LOW
        ("BACKGROUND",    (0,3), (-1,3),  colors.HexColor("#e8daef")),  # OVER
        ("BACKGROUND",    (0,4), (-1,4),  colors.HexColor("#d5f5e3")),  # OK
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("BOX",           (0,0), (-1,-1), 0.5, BLUE),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#c0d8ee")),
    ]))
    story.append(Spacer(1, 0.2*cm))
    story.append(s_tbl)
    story.append(PageBreak())

    # ── 7. SYSTEM ARCHITECTURE ────────────────────────────────────────────────
    story.append(Paragraph("7. System Architecture", st["h1"]))
    story.append(section_rule())

    components = [
        ("Data Layer",     "data/",        "Raw CSV → clean → feature-engineered DataFrame"),
        ("Source Package", "src/",         "6 modules: preprocessing, features, models, evaluation, business_logic, eda"),
        ("Training",       "scripts/train_pipeline.py", "CLI script: load → clean → engineer → train 3 models → evaluate → save"),
        ("Prediction",     "scripts/predict_pipeline.py", "CLI script: load model → forecast → generate decisions"),
        ("Automation",     "scripts/cron_job.py", "Weekly retrain + daily predict via APScheduler or cron"),
        ("Dashboard",      "app/streamlit_app.py", "4-tab Streamlit app with Plotly charts and real-time filters"),
        ("REST API",       "api/flask_api.py",     "5 Flask endpoints: health, stores, predict, decisions, summary"),
        ("Tests",          "tests/",       "36 unit tests across preprocessing, models, business logic"),
        ("Docker",         "docker/",      "Dockerfile + docker-compose for dashboard + API + scheduler"),
    ]
    comp_data = [["Component", "Location", "Description"]] + components
    comp_tbl  = Table(comp_data, colWidths=[3.5*cm, 5.5*cm, 7*cm])
    comp_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("BOX",           (0,0), (-1,-1), 0.5, BLUE),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#c0d8ee")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LIGHT]),
        ("FONTNAME",      (0,1), (0,-1),  "Helvetica-Bold"),
    ]))
    story.append(comp_tbl)

    # ── 8. DEPLOYMENT ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("8. Deployment Guide", st["h1"]))
    story.append(section_rule())

    story.append(Paragraph("8.1 Local Setup", st["h2"]))
    for cmd in [
        "git clone https://github.com/mukeshkd58/retail-demand-forecasting.git",
        "pip install -r requirements.txt",
        "python data/generate_sample_data.py",
        "python scripts/train_pipeline.py",
        "streamlit run app/streamlit_app.py        # Dashboard on :8501",
        "python api/flask_api.py                   # API on :5000",
    ]:
        story.append(Paragraph(f"$ {cmd}", st["code"]))

    story.append(Paragraph("8.2 Docker", st["h2"]))
    for cmd in [
        "cd docker/ && docker-compose up --build",
        "# Dashboard → http://localhost:8501",
        "# API       → http://localhost:5000",
    ]:
        story.append(Paragraph(f"$ {cmd}", st["code"]))

    story.append(Paragraph("8.3 Render (Cloud — Free Tier)", st["h2"]))
    story.append(Paragraph(
        "Push repository to GitHub. Create new Web Service on render.com. "
        "Build Command: pip install -r requirements.txt &amp;&amp; "
        "python data/generate_sample_data.py &amp;&amp; python scripts/train_pipeline.py. "
        "Start Command: streamlit run app/streamlit_app.py --server.port $PORT "
        "--server.address 0.0.0.0. Deploy takes approximately 5 minutes.",
        st["body"]
    ))

    # ── 9. API REFERENCE ──────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("9. API Reference", st["h1"]))
    story.append(section_rule())

    api_data = [
        ["Method", "Endpoint", "Description", "Auth"],
        ["GET",  "/health",                       "Health check + model status", "None"],
        ["GET",  "/api/v1/stores",                "All stores + metadata",        "None"],
        ["GET",  "/api/v1/departments?store=N",   "Departments for a store",      "None"],
        ["POST", "/api/v1/predict",               "Demand forecast (JSON body)",  "None"],
        ["POST", "/api/v1/inventory/decisions",   "Reorder recommendations",      "None"],
        ["GET",  "/api/v1/summary",               "Global KPI summary",           "None"],
    ]
    api_tbl = Table(api_data, colWidths=[1.8*cm, 6.2*cm, 6.2*cm, 1.8*cm])
    api_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("BOX",           (0,0), (-1,-1), 0.5, BLUE),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#c0d8ee")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LIGHT]),
    ]))
    story.append(api_tbl)

    # ── 10. TECH STACK ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("10. Technology Stack", st["h1"]))
    story.append(section_rule())

    tech_data = [
        ["Layer",           "Technology",                    "Version"],
        ["Language",        "Python",                        "3.11+"],
        ["Data",            "Pandas, NumPy",                 "2.1, 1.26"],
        ["Visualisation",   "Plotly, Matplotlib, Seaborn",   "5.18, 3.8, 0.13"],
        ["ML — Classical",  "Scikit-learn",                  "1.4"],
        ["ML — Boosting",   "XGBoost",                       "2.0"],
        ["ML — Time Series","Prophet",                       "1.1"],
        ["Dashboard",       "Streamlit",                     "1.31"],
        ["REST API",        "Flask",                         "3.0"],
        ["WSGI Server",     "Gunicorn",                      "21.2"],
        ["Containerisation","Docker + docker-compose",       "latest"],
        ["Testing",         "Pytest",                        "8.0"],
        ["Automation",      "APScheduler / cron",            "3.10 / system"],
    ]
    tech_tbl = Table(tech_data, colWidths=[5*cm, 7*cm, 4*cm])
    tech_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("BOX",           (0,0), (-1,-1), 0.5, BLUE),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#c0d8ee")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LIGHT]),
        ("FONTNAME",      (0,1), (0,-1),  "Helvetica-Bold"),
    ]))
    story.append(tech_tbl)

    # ── 11. CONCLUSION ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("11. Conclusion", st["h1"]))
    story.append(section_rule())
    story.append(Paragraph(
        "This project demonstrates a complete data science lifecycle from raw data "
        "ingestion through to cloud-deployed, production-ready ML application. "
        "Key achievements:", st["body"]
    ))
    for achievement in [
        "XGBoost model achieves R² = 0.9648, explaining 96.5% of weekly sales variance.",
        "Feature engineering with lag/rolling features captures temporal autocorrelation effectively.",
        "Business logic translates raw predictions into EOQ-based purchase orders.",
        "Streamlit dashboard provides interactive, real-time inventory monitoring.",
        "Flask REST API enables programmatic integration with ERP/WMS systems.",
        "Docker deployment makes the system reproducible and cloud-agnostic.",
        "36 unit tests ensure reliability and regression safety.",
        "Automated scheduler supports continuous learning with weekly retraining.",
    ]:
        story.append(Paragraph(f"• {achievement}", st["bullet"]))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Future Enhancements:", st["h2"]))
    for future in [
        "PostgreSQL + Airflow for production-grade data pipelines",
        "LSTM / Transformer models for longer-horizon forecasting",
        "Multi-objective optimisation balancing service level vs. cost",
        "Real-time data ingestion via Kafka / REST webhooks",
        "A/B testing framework for model deployment decisions",
    ]:
        story.append(Paragraph(f"• {future}", st["bullet"]))

    story.append(PageBreak())
    # ── CONTACT PAGE ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 2*cm))
    contact_data = [[
        Paragraph("Contact & Links", st["title"]),
    ]]
    contact_tbl = Table(contact_data, colWidths=[PAGE_W - 3.6*cm])
    contact_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 20),
        ("BOTTOMPADDING", (0,0), (-1,-1), 20),
    ]))
    story.append(contact_tbl)
    story.append(Spacer(1, 0.8*cm))

    links = [
        ("Author",    "Mukesh Kumar"),
        ("Email",     "mukeshkumardharani58@gmail.com"),
        ("GitHub",    "github.com/mukeshkd58"),
        ("Portfolio", "mukeshkd58.github.io"),
        ("LinkedIn",  "linkedin.com/in/mukeshkd58"),
        ("Dataset",   "kaggle.com/c/walmart-recruiting-store-sales-forecasting"),
    ]
    link_data = [[Paragraph(f"<b>{l}</b>", st["body"]),
                  Paragraph(v, st["body"])] for l, v in links]
    link_tbl  = Table(link_data, colWidths=[4*cm, 12*cm])
    link_tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("BACKGROUND",    (0,0), (0,-1),  LIGHT),
        ("BOX",           (0,0), (-1,-1), 0.5, BLUE),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#c0d8ee")),
    ]))
    story.append(link_tbl)

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story, canvasmaker=ReportCanvas)
    print(f"✅  Report saved → {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    build_pdf()
