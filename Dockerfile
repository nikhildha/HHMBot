# ── Unified Image: Python Bot + Next.js SaaS Dashboard ────────────────
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
COPY sentinel-saas/nextjs_space/package*.json ./sentinel-saas/nextjs_space/
RUN cd sentinel-saas/nextjs_space && npm install

# ── Copy all source code (ARG busts Docker cache on each deploy) ─────
ARG CACHEBUST=1
COPY . .

# ── Build Next.js production bundle ──────────────────────────────────
RUN cd sentinel-saas/nextjs_space && npm run build

# ── Make startup script executable ───────────────────────────────────
RUN chmod +x /app/start.sh

# ── Data directory ───────────────────────────────────────────────────
RUN mkdir -p /app/data

# ── Environment ──────────────────────────────────────────────────────
ENV DATA_DIR=/app/data
ENV PORT=3000
ENV PYTHONUNBUFFERED=1

EXPOSE 3000

# ── Start both Python bot + Next.js dashboard ────────────────────────
CMD ["/app/start.sh"]
