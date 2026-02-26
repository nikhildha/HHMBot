#!/bin/bash
# ── SENTINEL Startup Script ──────────────────────────────────────────
# Starts BOTH the Python bot engine AND the Node.js dashboard.
# The dashboard is the foreground process (serves the healthcheck).
# The bot runs in the background with automatic restart on crash.

set -e

echo "🚀 Starting SENTINEL (Bot Engine + Dashboard)..."

# Ensure data directory exists
mkdir -p /app/data

# ── Start Python Bot Engine (background, auto-restart) ────────────────
(
    while true; do
        echo "🤖 Starting Python bot engine..."
        cd /app
        python3 -u main.py >> /app/data/bot.log 2>&1 || true
        echo "⚠️ Bot engine exited. Restarting in 10s..."
        sleep 10
    done
) &
BOT_PID=$!
echo "🤖 Bot engine started (PID: $BOT_PID)"

# ── Start Node.js Dashboard (foreground — serves healthcheck) ─────────
cd /app/web-dashboard
exec node server.js
