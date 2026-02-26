#!/bin/bash
# ── SENTINEL Startup Script ──────────────────────────────────────────
# Only starts the Node.js dashboard. The Python bot engine is started
# via the "Start Engine" button on the dashboard (spawns main.py).

set -e

echo "🚀 Starting SENTINEL Dashboard..."

# Ensure data directory exists
mkdir -p /app/data

# Start Node.js dashboard (the bot is started via the UI)
cd /app/web-dashboard
exec node server.js
