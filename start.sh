#!/bin/bash
# ── SENTINEL Startup Script ──────────────────────────────────────────
# Starts the Next.js SaaS dashboard. The Python bot engine is started
# via the "Start Engine" button on the dashboard (spawns main.py).

set -e

echo "🚀 Starting SENTINEL Dashboard..."

# Ensure data directory exists
mkdir -p /app/data

# Start Next.js SaaS dashboard (the bot is started via the UI)
cd /app/sentinel-saas/nextjs_space
exec npx next start -p ${PORT:-3000}
