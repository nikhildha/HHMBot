# ── Unified Image: Python Bot + Node.js Dashboard ─────────────────────
FROM python:3.11-slim

# Install Node.js 20
RUN apt-get update && apt-get install -y curl && \
     curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
     apt-get install -y nodejs && \
     apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Node.js dependencies ─────────────────────────────────────────────
COPY web-dashboard/package*.json ./web-dashboard/
RUN cd web-dashboard && npm install --production

# ── Copy all source code ─────────────────────────────────────────────
COPY . .

# ── Make startup script executable ───────────────────────────────────
RUN chmod +x /app/start.sh

# ── Data directory ───────────────────────────────────────────────────
RUN mkdir -p /app/data

# ── Environment ──────────────────────────────────────────────────────
ENV DATA_DIR=/app/data
ENV PORT=3001
ENV PYTHONUNBUFFERED=1

EXPOSE 3001

# ── Start both Python bot + Node dashboard ───────────────────────────
CMD ["/app/start.sh"]
