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

    # ─── Conviction Scoring (8-factor, 0-100) ─────────────────────────────────

    @staticmethod
    def compute_conviction_score(
        confidence: float,
        regime: int,
        side: str,
        btc_regime=None,
        funding_rate=None,
        sr_position=None,
        vwap_position=None,
        oi_change=None,
        volatility=None,
        sentiment_score=None,
        orderflow_score=None,
    ) -> float:
        """
        Compute a 0–100 conviction score from 8 independent factors.

        Factors and max weights
        ───────────────────────
        1. HMM Confidence       (22 pts) — core signal quality
        2. BTC Macro Regime     (18 pts) — macro alignment
        3. Funding Rate         (12 pts) — perpetual swap carry signal
        4. S/R + VWAP Position  (10 pts) — price vs key structural levels
        5. Open Interest Change  (8 pts) — smart-money positioning
        6. Volatility Quality    (5 pts) — regime quality filter
        7. Sentiment Score      (15 pts) — social/news signal (alert = hard veto)
        8. Order Flow           (10 pts) — L2 depth + taker flow + cumDelta

        Total max = 100 pts.
        Conviction → leverage via get_conviction_leverage().
        """
        # ── Hard veto: sentiment ALERT (hack, exploit, rug-pull, etc.) ─────────
        if sentiment_score is not None and sentiment_score <= -1.0:
            logger.warning("🚨 Sentiment ALERT veto — conviction forced to 0")
            return 0.0

        score = 0.0

        # ── 1. HMM Confidence (22 pts) ────────────────────────────────────────
        w = config.CONVICTION_WEIGHT_HMM
        if confidence >= 0.97:
            score += w
        elif confidence >= 0.94:
            score += w * 0.85
        elif confidence >= 0.90:
            score += w * 0.65
        elif confidence >= 0.85:
            score += w * 0.40
        else:
            score += 0  # below 85% — no contribution

        # ── 2. BTC Macro Alignment (18 pts) ───────────────────────────────────
        w = config.CONVICTION_WEIGHT_BTC_MACRO
        if btc_regime is not None:
            if btc_regime == config.REGIME_CRASH:
                score -= 10                       # crash macro → heavy penalty
            elif (side == "BUY"  and btc_regime == config.REGIME_BULL) or \
                 (side == "SELL" and btc_regime == config.REGIME_BEAR):
                score += w                        # aligned with macro
            elif (side == "BUY"  and btc_regime == config.REGIME_BEAR) or \
                 (side == "SELL" and btc_regime == config.REGIME_BULL):
                score -= 8                        # fighting macro
            else:
                score += w * 0.35                 # chop / unknown — small boost
        else:
            score += w * 0.50                     # no BTC data — neutral half

        # ── 3. Funding Rate (12 pts) ───────────────────────────────────────────
        w = config.CONVICTION_WEIGHT_FUNDING
        if funding_rate is not None:
            if side == "BUY":
                if funding_rate < -0.0001:        # negative funding → longs paid
                    score += w
                elif funding_rate < 0.0003:
                    score += w * 0.55
                else:
                    score -= 4                    # high positive → crowded longs
            else:  # SELL
                if funding_rate > 0.0001:         # positive funding → shorts paid
                    score += w
                elif funding_rate > -0.0003:
                    score += w * 0.55
                else:
                    score -= 4
        else:
            score += w * 0.55                     # no data — mild positive

        # ── 4. S/R + VWAP Position (10 pts) ───────────────────────────────────
        w = config.CONVICTION_WEIGHT_SR_VWAP
        if sr_position is not None or vwap_position is not None:
            sr_pts  = 0.0
            vwap_pts = 0.0
            if sr_position is not None:
                # sr_position: 0=at support (BUY ideal), 1=at resistance (SELL ideal)
                if side == "BUY":
                    sr_pts = (1.0 - sr_position) * (w * 0.6)
                else:
                    sr_pts = sr_position * (w * 0.6)
            if vwap_position is not None:
                # vwap_position: >0 means price above VWAP (bullish), <0 below (bearish)
                if (side == "BUY"  and vwap_position > 0) or \
                   (side == "SELL" and vwap_position < 0):
                    vwap_pts = w * 0.4
                else:
                    vwap_pts = 0
            score += sr_pts + vwap_pts
        else:
            score += w * 0.45                     # no data — mild positive

        # ── 5. Open Interest Change (8 pts) ────────────────────────────────────
        w = config.CONVICTION_WEIGHT_OI
        if oi_change is not None:
            if side == "BUY":
                if oi_change > 0.03:              # OI growing > 3% → strong positioning
                    score += w
                elif oi_change > 0.01:
                    score += w * 0.60
                elif oi_change < -0.03:           # OI falling → short covering risk
                    score -= 3
                else:
                    score += w * 0.30
            else:  # SELL
                if oi_change < -0.03:             # OI falling → shorts winning
                    score += w
                elif oi_change < -0.01:
                    score += w * 0.60
                elif oi_change > 0.03:
                    score -= 3
                else:
                    score += w * 0.30
        else:
            score += w * 0.50

        # ── 6. Volatility Quality (5 pts) ─────────────────────────────────────
        w = config.CONVICTION_WEIGHT_VOL
        if volatility is not None:
            if config.VOL_MIN_ATR_PCT <= volatility <= config.VOL_MAX_ATR_PCT * 0.5:
                score += w                        # ideal vol range
            elif volatility <= config.VOL_MAX_ATR_PCT:
                score += w * 0.60
            else:
                score += w * 0.10                 # too volatile
        else:
            score += w * 0.60

        # ── 7. Sentiment Score (15 pts) ────────────────────────────────────────
        w = config.CONVICTION_WEIGHT_SENTIMENT
        if sentiment_score is not None:
            if sentiment_score < config.SENTIMENT_VETO_THRESHOLD:
                score -= 12                       # strong negative news
            elif sentiment_score < -0.20:
                score -= 4
            elif sentiment_score < 0.20:
                score += w * 0.30                 # neutral
            elif sentiment_score < config.SENTIMENT_STRONG_POS:
                score += w * 0.75                 # moderately positive
            else:
                score += w                        # strongly positive
        else:
            score += w * 0.30                     # no sentiment data — mild

        # ── 8. Order Flow (10 pts) ─────────────────────────────────────────────
        w = config.CONVICTION_WEIGHT_ORDERFLOW
        if orderflow_score is not None:
            # orderflow_score is -1..+1; map so that trade-aligned flow adds max pts
            if side == "BUY":
                aligned = orderflow_score          # positive = buy pressure = aligned
            else:
                aligned = 0.0 - orderflow_score   # negative = sell pressure = aligned

            if aligned > 0.5:
                score += w                        # strong flow confirmation
            elif aligned > 0.2:
                score += w * 0.70
            elif aligned > -0.2:
                score += w * 0.30                 # neutral flow
            elif aligned > -0.5:
                score -= 3                        # mild opposing flow
            else:
                score -= 7                        # strong opposing flow

        # ── Cap and floor ──────────────────────────────────────────────────────
        return float(max(0.0, min(100.0, score)))

    @staticmethod
    def get_conviction_leverage(conviction_score: float) -> int:
        """
        Map conviction score (0–100) to leverage.

        Bands
        ─────
        < 40      → 0  (no trade)
        40–54     → 10x
        55–69     → 15x
        70–84     → 25x
        85–100    → 35x
        """
        if conviction_score < 40:
            return 0
        elif conviction_score < 55:
            return 10
        elif conviction_score < 70:
            return 15
        elif conviction_score < 85:
            return 25
        else:
            return 35
