# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Mukesh Kumar <mukeshkumardharani58@gmail.com>"
LABEL description="Retail Demand Forecasting & Inventory Optimization"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────────────────────
COPY . .

# ── Generate data & train model if no model exists ────────────────────────────
RUN python data/generate_sample_data.py \
 && python scripts/train_pipeline.py

# ── Expose ports ───────────────────────────────────────────────────────────────
# 8501 = Streamlit   |   5000 = Flask API
EXPOSE 8501 5000

# ── Default: run Streamlit dashboard ──────────────────────────────────────────
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
