"""
Engine Worker — Per-Bot Trading Process

Launched by the BotOrchestrator as a subprocess for each active bot.
Reads bot config & API keys from PostgreSQL, runs the HMM analysis loop,
and writes trade events back to the database.

Environment:
  BOT_ID       — the Prisma Bot ID to run
  DATABASE_URL — PostgreSQL connection string

This is essentially a multi-tenant version of main.py's RegimeMasterBot,
but scoped to a single bot with config from the database.
"""
import os
import sys
import json
import time
import signal
import logging
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import config
from hmm_brain import HMMBrain
from data_pipeline import fetch_klines, get_multi_timeframe_data
from feature_engine import compute_all_features, compute_hmm_features, compute_trend, compute_support_resistance
from execution_engine import ExecutionEngine
from risk_manager import RiskManager
from coin_scanner import get_top_coins_by_volume
from db_adapter import DBAdapter

IST = timezone(timedelta(hours=5, minutes=30))

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

BOT_ID = os.environ.get("BOT_ID", "")
logger = logging.getLogger(f"Worker:{BOT_ID[:8]}")


class EngineWorker:
    """
    Single-bot engine worker. Reads config from DB, runs HMM cycles,
    persists trades to PostgreSQL.
    """

    def __init__(self, bot_id: str):
        self.bot_id = bot_id
        self.db = DBAdapter()
        self._running = True
        self._cycle_count = 0
        self._coin_brains = {}
        self._coin_states = {}

        # Load bot config from DB
        self.bot_config = self.db.get_bot_config(bot_id)
        if not self.bot_config:
            raise RuntimeError(f"No BotConfig found for bot {bot_id}")

        # Get user ID for exchange key lookup
        self.user_id = self.db.get_bot_user_id(bot_id)
        if not self.user_id:
            raise RuntimeError(f"No user found for bot {bot_id}")

        # Initialize engine components
        self.executor = ExecutionEngine()
        self.risk = RiskManager()

        # Extract config values
        self.mode = self.bot_config.get("mode", "paper")
        self.coin_list = self.bot_config.get("coinList", [])
        self.capital_per_trade = self.bot_config.get("capitalPerTrade", 100)
        self.max_open_trades = self.bot_config.get("maxOpenTrades", 5)

        logger.info("🤖 Engine Worker initialized for bot %s", bot_id)
        logger.info("   Mode: %s | Coins: %d | Capital: $%.0f",
                     self.mode, len(self.coin_list), self.capital_per_trade)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("⛔ Received signal %d — shutting down...", signum)
        self._running = False

    def run(self):
        """Main loop — runs HMM analysis cycles."""
        self.db.set_bot_status(self.bot_id, "running", True)
        self.db.update_bot_state(self.bot_id, engine_status="running")

        logger.info("🟢 Engine started for bot %s", self.bot_id)

        analysis_interval = config.ANALYSIS_INTERVAL_SECONDS  # 15 min
        heartbeat_interval = config.LOOP_INTERVAL_SECONDS      # 1 min
        last_analysis = 0.0

        try:
            while self._running:
                cycle_start = time.time()

                try:
                    # ── Heartbeat: update PNL, check SL/TP ──
                    self._heartbeat_cycle()

                    # ── Full analysis every 15 min ──
                    if (time.time() - last_analysis) >= analysis_interval:
                        self._analysis_cycle()
                        last_analysis = time.time()
                        self._cycle_count += 1

                    # ── Update bot state ──
                    cycle_ms = int((time.time() - cycle_start) * 1000)
                    self.db.update_bot_state(
                        self.bot_id,
                        engine_status="running",
                        last_cycle_at=datetime.now(timezone.utc),
                        cycle_count=self._cycle_count,
                        cycle_duration_ms=cycle_ms,
                        coin_states=self._coin_states,
                    )

                except Exception as e:
                    logger.error("❌ Cycle error: %s", e, exc_info=True)
                    self.db.update_bot_state(
                        self.bot_id,
                        engine_status="error",
                        error_message=str(e),
                        error_at=datetime.now(timezone.utc),
                    )

                # Sleep until next heartbeat
                elapsed = time.time() - cycle_start
                sleep_time = max(0, heartbeat_interval - elapsed)
                if sleep_time > 0 and self._running:
                    time.sleep(sleep_time)

        finally:
            self.db.set_bot_status(self.bot_id, "stopped", False)
            self.db.update_bot_state(self.bot_id, engine_status="idle")
            self.db.close()
            logger.info("🔴 Engine stopped for bot %s", self.bot_id)

    def _heartbeat_cycle(self):
        """Quick cycle: update PNL, check SL/TP hits."""
        active_trades = self.db.get_active_trades(self.bot_id)
        for trade in active_trades:
            try:
                symbol = trade["coin"]
                # Fetch latest price
                df = fetch_klines(symbol, interval="1m", limit=1)
                if df is None or df.empty:
                    continue
                current_price = float(df["close"].iloc[-1])

                # Calculate PNL
                entry = trade["entryPrice"]
                side = trade["position"]
                lev = trade["leverage"]

                if side == "long":
                    pnl_pct = ((current_price - entry) / entry) * 100 * lev
                else:
                    pnl_pct = ((entry - current_price) / entry) * 100 * lev

                capital = trade["capital"]
                pnl = capital * (pnl_pct / 100)

                self.db.update_trade(
                    trade["id"],
                    current_price=current_price,
                    active_pnl=round(pnl, 2),
                    active_pnl_pct=round(pnl_pct, 2),
                )

                # Check SL hit
                sl = trade["stopLoss"]
                if side == "long" and current_price <= sl:
                    self.db.close_trade(trade["id"], current_price, "FIXED_SL", pnl, pnl_pct)
                elif side == "short" and current_price >= sl:
                    self.db.close_trade(trade["id"], current_price, "FIXED_SL", pnl, pnl_pct)

                # Check multi-target hits
                self._check_targets(trade, current_price, pnl_pct, capital, lev)

                # Check MAX_LOSS guard
                max_loss = self.bot_config.get("maxLossPct", -15)
                if pnl_pct <= max_loss:
                    self.db.close_trade(trade["id"], current_price, "MAX_LOSS", pnl, pnl_pct)
                    logger.warning("🛑 MAX_LOSS hit on %s: %.2f%%", symbol, pnl_pct)

            except Exception as e:
                logger.error("Heartbeat error for %s: %s", trade.get("coin", "?"), e)

    def _check_targets(self, trade: dict, current_price: float,
                        pnl_pct: float, capital: float, lev: float):
        """Check T1/T2/T3 targets for partial booking."""
        if not self.bot_config.get("multiTargetEnabled", True):
            return

        side = trade["position"]
        trade_id = trade["id"]

        # T1
        t1 = trade.get("t1Price")
        if t1 and not trade.get("t1Hit", False):
            hit = (side == "long" and current_price >= t1) or \
                  (side == "short" and current_price <= t1)
            if hit:
                book_pct = self.bot_config.get("t1BookPct", 0.25)
                qty = (trade.get("quantity", 0) or 0) * book_pct
                partial_pnl = capital * book_pct * (pnl_pct / 100)

                self.db.update_trade(trade_id, t1_hit=True)
                self.db.create_partial_booking(
                    trade_id, "T1", book_pct, qty, current_price,
                    round(partial_pnl, 2), round(pnl_pct, 2)
                )
                # Move SL to entry (breakeven)
                self.db.update_trade(trade_id, stop_loss=trade["entryPrice"])
                logger.info("🎯 T1 hit on %s — booked %.0f%%", trade["coin"], book_pct * 100)

        # T2
        t2 = trade.get("t2Price")
        if t2 and trade.get("t1Hit", False) and not trade.get("t2Hit", False):
            hit = (side == "long" and current_price >= t2) or \
                  (side == "short" and current_price <= t2)
            if hit:
                book_pct = self.bot_config.get("t2BookPct", 0.50)
                remaining = 1.0 - self.bot_config.get("t1BookPct", 0.25)
                qty = (trade.get("quantity", 0) or 0) * remaining * (book_pct / remaining)
                partial_pnl = capital * book_pct * (pnl_pct / 100)

                self.db.update_trade(trade_id, t2_hit=True)
                self.db.create_partial_booking(
                    trade_id, "T2", book_pct, qty, current_price,
                    round(partial_pnl, 2), round(pnl_pct, 2)
                )
                # Move SL to T1
                self.db.update_trade(trade_id, stop_loss=trade.get("t1Price", trade["entryPrice"]))
                logger.info("🎯🎯 T2 hit on %s — booked %.0f%%", trade["coin"], book_pct * 100)

        # T3 = full close
        t3 = trade.get("t3Price")
        if t3 and trade.get("t2Hit", False):
            hit = (side == "long" and current_price >= t3) or \
                  (side == "short" and current_price <= t3)
            if hit:
                self.db.close_trade(trade_id, current_price, "T3", 
                                     round(capital * (pnl_pct / 100), 2), round(pnl_pct, 2))
                logger.info("🎯🎯🎯 T3 hit on %s — FULL CLOSE", trade["coin"])

    def _analysis_cycle(self):
        """Full HMM analysis cycle across all coins."""
        logger.info("━" * 50)
        logger.info("🔬 Analysis cycle #%d starting (%d coins)",
                     self._cycle_count + 1, len(self.coin_list))

        active_trades = self.db.get_active_trades(self.bot_id)
        active_symbols = {t["coin"] for t in active_trades}

        for symbol in self.coin_list:
            try:
                # Skip if already have a position
                if symbol in active_symbols:
                    continue

                # Check max concurrent positions
                if len(active_symbols) >= self.max_open_trades:
                    break

                # Fetch data
                df = fetch_klines(symbol, interval="15m", limit=200)
                if df is None or df.empty or len(df) < 50:
                    continue

                # Compute features
                features = compute_all_features(df)
                if features is None:
                    continue

                # Get/create HMM brain for this coin
                if symbol not in self._coin_brains:
                    self._coin_brains[symbol] = HMMBrain()

                brain = self._coin_brains[symbol]
                hmm_feats = compute_hmm_features(df)
                if hmm_feats is None or len(hmm_feats) < 30:
                    continue

                brain.fit(hmm_feats)
                regime = brain.predict_regime(hmm_feats)
                confidence = brain.confidence

                # Store state for dashboard
                self._coin_states[symbol] = {
                    "regime": int(regime),
                    "confidence": round(confidence, 1),
                    "last_price": float(df["close"].iloc[-1]),
                    "updated_at": datetime.now(IST).isoformat(),
                }

                # Check trade eligibility
                if confidence < config.MIN_CONFIDENCE:
                    continue

                trend = compute_trend(df)
                side = None
                if regime == 1 and trend == "bullish":
                    side = "long"
                elif regime == 2 and trend == "bearish":
                    side = "short"

                if not side:
                    continue

                # Calculate risk levels
                close = float(df["close"].iloc[-1])
                atr = float(features.get("atr", close * 0.02))
                sl_mult = self.bot_config.get("slMultiplier", 0.8)
                tp_mult = self.bot_config.get("tpMultiplier", 1.0)

                if side == "long":
                    sl = close - atr * sl_mult
                    tp = close + atr * tp_mult
                else:
                    sl = close + atr * sl_mult
                    tp = close - atr * tp_mult

                # Multi-targets
                t1_mult = self.bot_config.get("t1Multiplier", 0.5)
                t2_mult = self.bot_config.get("t2Multiplier", 1.0)
                t3_mult = self.bot_config.get("t3Multiplier", 1.5)

                if side == "long":
                    t1 = close + atr * t1_mult
                    t2 = close + atr * t2_mult
                    t3 = close + atr * t3_mult
                else:
                    t1 = close - atr * t1_mult
                    t2 = close - atr * t2_mult
                    t3 = close - atr * t3_mult

                # Leverage from config tiers
                leverage = self._get_leverage(confidence)
                quantity = (self.capital_per_trade * leverage) / close

                # Create trade in DB
                trade_id = self.db.create_trade(self.bot_id, {
                    "symbol": symbol,
                    "side": side,
                    "regime": "bullish" if regime == 1 else "bearish",
                    "confidence": confidence,
                    "mode": self.mode,
                    "leverage": leverage,
                    "capital": self.capital_per_trade,
                    "quantity": quantity,
                    "entry_price": close,
                    "stop_loss": round(sl, 6),
                    "take_profit": round(tp, 6),
                    "t1_price": round(t1, 6),
                    "t2_price": round(t2, 6),
                    "t3_price": round(t3, 6),
                })

                active_symbols.add(symbol)
                logger.info("🟢 NEW TRADE: %s %s @ %.4f (conf=%.1f%% lev=%dx)",
                             side.upper(), symbol, close, confidence, leverage)

            except Exception as e:
                logger.error("Analysis error for %s: %s", symbol, e, exc_info=True)
                self._coin_states[symbol] = {
                    "regime": -1,
                    "confidence": 0,
                    "error": str(e),
                    "updated_at": datetime.now(IST).isoformat(),
                }

    def _get_leverage(self, confidence: float) -> int:
        """Get leverage from config tiers based on confidence."""
        tiers = self.bot_config.get("leverageTiers")
        if tiers and isinstance(tiers, list):
            for tier in tiers:
                if tier.get("min", 0) <= confidence <= tier.get("max", 100):
                    return int(tier.get("lev", 5))

        # Default tiers
        if confidence >= 70:
            return 15
        elif confidence >= 55:
            return 10
        else:
            return 5


def main():
    if not BOT_ID:
        logger.error("BOT_ID env var required")
        sys.exit(1)

    logger.info("🚀 Starting engine worker for bot %s", BOT_ID)

    try:
        worker = EngineWorker(BOT_ID)
        worker.run()
    except Exception as e:
        logger.error("💥 Worker crashed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
