"""
Project Regime-Master — Risk Manager
Position sizing, dynamic leverage, kill switch, and ATR-based stops.
"""
import json
import logging
import numpy as np
from datetime import datetime

import config

logger = logging.getLogger("RiskManager")


class RiskManager:
    """
    Enforces the "Anti-Liquidation" rules:
      • 2% risk per trade
      • Dynamic leverage based on HMM confidence
      • Kill switch on 10% drawdown in 24h
      • ATR-based stop-loss placement
    """

    def __init__(self):
        self.equity_history = []   # List of (timestamp, balance) tuples
        self._killed = False

    # ─── Dynamic Leverage ────────────────────────────────────────────────────

    @staticmethod
    def get_dynamic_leverage(confidence, regime):
        """
        Map HMM confidence and regime → leverage multiplier.

        Rules (updated):
          • Crash regime → 0 (stay out)
          • Chop regime  → 15x (mean reversion)
          • Trend (Bull/Bear):
              confidence ≥ 95%  → 35x
              confidence 91–95% → 25x
              confidence 85–90% → 15x
              confidence < 85%  → 0 (DO NOT DEPLOY)

        Parameters
        ----------
        confidence : float (0..1)
        regime : int (config.REGIME_*)

        Returns
        -------
        int : leverage value (0 = skip trade)
        """
        # Crash regime → stay out completely
        if regime == config.REGIME_CRASH:
            return 0

        # Chop regime → low leverage for mean reversion (still requires 85%+ confidence)
        if regime == config.REGIME_CHOP:
            return config.LEVERAGE_LOW if confidence >= config.CONFIDENCE_LOW else 0

        # Trend regimes (Bull / Bear) — scale by confidence
        # > 95% → 35x
        if confidence >= config.CONFIDENCE_HIGH:
            return config.LEVERAGE_HIGH
        # 91–95% → 25x
        elif confidence >= config.CONFIDENCE_MEDIUM:
            return config.LEVERAGE_MODERATE
        # 85–90% → 15x
        elif confidence >= config.CONFIDENCE_LOW:
            return config.LEVERAGE_LOW
        else:
            return 0  # Below 85% — do not deploy

    # ─── Position Sizing (2% Rule) ───────────────────────────────────────────

    @staticmethod
    def calculate_position_size(balance, entry_price, atr, leverage=1, risk_pct=None):
        """
        Position size so that a 1-ATR adverse move ≤ risk_pct of balance.
        
        Formula:
          risk_amount = balance * risk_pct
          stop_distance = atr * ATR_SL_MULTIPLIER
          raw_qty = risk_amount / stop_distance
          leveraged_qty = raw_qty  (leverage amplifies PnL, not qty)
        
        Returns
        -------
        float : quantity in base asset
        """
        risk_pct = risk_pct or config.RISK_PER_TRADE
        risk_amount = balance * risk_pct
        stop_distance = atr * config.get_atr_multipliers(leverage)[0]

        if stop_distance <= 0 or entry_price <= 0:
            return config.DEFAULT_QUANTITY

        quantity = risk_amount / stop_distance
        # Ensure we don't exceed balance even with leverage
        max_qty = (balance * leverage) / entry_price
        quantity = min(quantity, max_qty)

        # Round to reasonable precision
        quantity = round(quantity, 6)
        return max(quantity, 0.0001)  # Binance minimum

    # ─── ATR Stop Loss / Take Profit ─────────────────────────────────────────

    @staticmethod
    def calculate_atr_stops(entry_price, atr, side, leverage=1):
        """
        Compute SL and TP based on ATR, adjusted for leverage.
        
        Parameters
        ----------
        entry_price : float
        atr : float
        side : str ('BUY' or 'SELL')
        leverage : int
        
        Returns
        -------
        (stop_loss: float, take_profit: float)
        """
        sl_mult, tp_mult = config.get_atr_multipliers(leverage)
        sl_dist = atr * sl_mult
        tp_dist = atr * tp_mult

        # Adaptive precision: more decimals for cheaper coins
        if entry_price >= 100:
            decimals = 2
        elif entry_price >= 1:
            decimals = 4
        else:
            decimals = 6

        if side == "BUY":
            stop_loss   = round(entry_price - sl_dist, decimals)
            take_profit = round(entry_price + tp_dist, decimals)
        else:
            stop_loss   = round(entry_price + sl_dist, decimals)
            take_profit = round(entry_price - tp_dist, decimals)

        return stop_loss, take_profit

    # ─── Kill Switch ─────────────────────────────────────────────────────────

    def record_equity(self, balance):
        """Record current equity for drawdown monitoring."""
        self.equity_history.append((datetime.utcnow(), balance))
        # Keep only last 24h
        cutoff = datetime.utcnow().timestamp() - 86400
        self.equity_history = [
            (t, b) for t, b in self.equity_history
            if t.timestamp() > cutoff
        ]

    def check_kill_switch(self):
        """
        If portfolio dropped ≥ KILL_SWITCH_DRAWDOWN (10%) in the last 24h → KILL.
        
        Returns
        -------
        bool : True if kill switch triggered
        """
        if self._killed:
            return True

        if len(self.equity_history) < 2:
            return False

        peak = max(b for _, b in self.equity_history)
        current = self.equity_history[-1][1]

        drawdown = (peak - current) / peak if peak > 0 else 0

        if drawdown >= config.KILL_SWITCH_DRAWDOWN:
            logger.critical(
                "🚨 KILL SWITCH TRIGGERED! Drawdown: %.2f%% (peak=%.2f, now=%.2f)",
                drawdown * 100, peak, current,
            )
            self._killed = True
            # Write kill command
            self._write_kill_command()
            return True

        return False

    def _write_kill_command(self):
        """Persist kill command so dashboard can detect it."""
        try:
            with open(config.COMMANDS_FILE, "w") as f:
                json.dump({"command": "KILL", "timestamp": datetime.utcnow().isoformat()}, f)
        except Exception as e:
            logger.error("Failed to write kill command: %s", e)

    def reset_kill_switch(self):
        """Manual reset (via dashboard)."""
        self._killed = False
        self.equity_history.clear()
        logger.info("Kill switch reset.")

    @property
    def is_killed(self):
        return self._killed
