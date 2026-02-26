#!/bin/bash
# ── SENTINEL Startup Script ──────────────────────────────────────────
# Runs both the Python bot engine and Node.js dashboard in one container.

set -e

echo "🚀 Starting SENTINEL..."

# Ensure data directory exists
mkdir -p /app/data

# Start Python bot engine in background
echo "🐍 Starting bot engine (main.py)..."
cd /app
python -u main.py &
BOT_PID=$!
echo "   Bot PID: $BOT_PID"

# Start Node.js dashboard in foreground
echo "🌐 Starting dashboard (server.js)..."
cd /app/web-dashboard
exec node server.js
