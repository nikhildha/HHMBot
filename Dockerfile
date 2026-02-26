# ── Stage 1: Python Bot ──────────────────────────────────────────────
FROM python:3.11-slim AS bot

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot source code
COPY main.py config.py data_pipeline.py execution_engine.py \
     feature_engine.py hmm_brain.py risk_manager.py \
     coin_scanner.py sideways_strategy.py tradebook.py \
     backtest.py backtester.py ./

# Data directory (mounted as volume)
RUN mkdir -p /app/data

CMD ["python", "-u", "main.py"]


# ── Stage 2: Node.js Dashboard ──────────────────────────────────────
FROM node:20-slim AS dashboard

WORKDIR /app/web-dashboard

COPY web-dashboard/package*.json ./
RUN npm install --production

COPY web-dashboard/ ./

# Data directory will be mounted
RUN mkdir -p /app/data

ENV DATA_DIR=/app/data
ENV PORT=3001

EXPOSE 3001

CMD ["node", "server.js"]
