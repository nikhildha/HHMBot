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

    # ─── Conviction Scoring System ─────────────────────────────────────────

    @staticmethod
    def compute_conviction_score(confidence, regime, side,
                                  btc_regime=None,
                                  funding_rate=None,
                                  sr_position=None,
                                  vwap_position=None,
                                  oi_change=None,
                                  volatility=None,
                                  sentiment_score=None):
        """
        Compute a multi-factor conviction score (0-100) for a trade.

        Factors (positive max pts sum to 100):
          1. HMM Confidence       (25 pts) — core model signal
          2. BTC Macro Alignment   (20 pts) — does BTC agree with our trade?
          3. Funding Rate          (12 pts) — is the opposite side crowded?
          4. S/R + VWAP Position   (12 pts) — are we entering at a good level?
          5. OI Momentum            (8 pts) — is smart money building our way?
          6. Volatility Regime       (5 pts) — is vol tradeable?
          7. Sentiment Score        (18 pts) — social/news sentiment signal [NEW]

        Parameters
        ----------
        sentiment_score : float | None
            Composite sentiment from SentimentEngine, range -1 to +1.
            Pass -1.0 if a ALERT (hack/exploit/scam) was detected by the engine.

        Returns
        -------
        float : conviction score 0-100
        """
        score = 0.0

        # ─── 1. HMM Confidence (25 pts max) ──────────────────────────
        # 92% → 0pts, 96% → 12pts, 99% → 21pts, 100% → 25pts
        if confidence >= 0.92:
            conf_score = min(25, (confidence - 0.92) / 0.08 * 25)
            score += conf_score

        # ─── 2. BTC Macro Alignment (20 pts max) ─────────────────────
        if btc_regime is not None:
            if btc_regime == config.REGIME_CRASH:
                score -= 15  # Strong penalty — BTC crashing
            elif side == 'LONG' and btc_regime == config.REGIME_BULL:
                score += 20  # Perfect alignment
            elif side == 'SHORT' and btc_regime == config.REGIME_BEAR:
                score += 20  # Perfect alignment
            elif side == 'LONG' and btc_regime == config.REGIME_BEAR:
                score -= 10  # Counter-BTC — dangerous
            elif side == 'SHORT' and btc_regime == config.REGIME_BULL:
                score -= 10  # Counter-BTC — dangerous
            elif btc_regime == config.REGIME_CHOP:
                score += 5   # Neutral — slight bonus for coin-specific signal
        else:
            score += 8  # No BTC data → give partial neutral credit

        # ─── 3. Funding Rate (12 pts max) ────────────────────────────
        if funding_rate is not None:
            if side == 'LONG' and funding_rate < -0.0001:
                score += 12  # Shorts crowded → squeeze potential
            elif side == 'LONG' and funding_rate > 0.0005:
                score -= 5   # Longs crowded → risky
            elif side == 'SHORT' and funding_rate > 0.0003:
                score += 12  # Longs crowded → dump potential
            elif side == 'SHORT' and funding_rate < -0.0003:
                score -= 5   # Shorts crowded → risky
            else:
                score += 6   # Neutral funding
        else:
            score += 6  # No data → neutral

        # ─── 4. S/R + VWAP Position (12 pts max) ─────────────────────
        sr_score = 0
        if sr_position is not None:
            # sr_position: -1 = at support, +1 = at resistance
            if side == 'LONG':
                sr_score += max(0, (1 - sr_position) / 2 * 6)  # Closer to support = more pts
            else:
                sr_score += max(0, (1 + sr_position) / 2 * 6)  # Closer to resistance = more pts
        else:
            sr_score += 3

        if vwap_position is not None:
            if side == 'LONG' and vwap_position < 0:
                sr_score += 6  # Below VWAP — buying cheap
            elif side == 'SHORT' and vwap_position > 0:
                sr_score += 6  # Above VWAP — selling expensive
            else:
                sr_score += 2  # Neutral
        else:
            sr_score += 2

        score += min(12, sr_score)

        # ─── 5. OI Momentum (8 pts max) ──────────────────────────────
        if oi_change is not None:
            if side == 'LONG' and oi_change > 0.02:
                score += 8   # OI building → conviction rising
            elif side == 'SHORT' and oi_change < -0.02:
                score += 8   # OI dropping → panic selling
            elif abs(oi_change) < 0.01:
                score += 4   # OI stable → neutral
            else:
                score += 2   # Slight adverse OI
        else:
            score += 4  # No data → neutral

        # ─── 6. Volatility Regime (5 pts max) ────────────────────────
        if volatility is not None:
            if 0.005 < volatility < 0.03:
                score += 5   # Moderate vol → tradeable
            elif volatility >= 0.03:
                score += 1   # High vol → risky but opportunity
            else:
                score += 3   # Low vol → less opportunity
        else:
            score += 3  # No data → neutral

        # ─── 7. Sentiment Score (18 pts max) ─────────────────────────
        # sentiment_score in [-1, +1]; pass -1.0 when SentimentEngine fires an ALERT.
        if sentiment_score is not None:
            if sentiment_score <= -1.0:
                # ALERT detected (hack/exploit/scam) — hard veto, never trade
                return 0
            elif sentiment_score < config.SENTIMENT_VETO_THRESHOLD:
                # Strong negative sentiment (e.g. regulatory ban news)
                score -= 15
            elif sentiment_score < -0.2:
                # Moderate negative
                score -= 5
            elif sentiment_score < 0.2:
                # Neutral — partial credit
                score += 5
            elif sentiment_score < config.SENTIMENT_STRONG_POS:
                # Moderate positive
                score += 12
            else:
                # Strongly positive — full bonus
                score += 18
        else:
            score += 5  # No sentiment data → neutral partial credit

        return max(0, min(100, score))
    
    @staticmethod
    def get_conviction_leverage(score):
        """
        Map conviction score (0-100) to continuous leverage (10x-35x).
        
        Score 0-39   → 0  (skip trade — insufficient conviction)
        Score 40     → 10x  (minimum deployment)
        Score 55     → 16x
        Score 70     → 22x
        Score 85     → 29x
        Score 100    → 35x  (maximum conviction)
        """
        if score < 40:
            return 0
        
        # Continuous: 40→10x, 100→35x
        leverage = 10 + (score - 40) / 60 * 25
        return max(10, min(35, round(leverage)))

    # ─── Legacy Dynamic Leverage (kept for backward compatibility) ───────

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
        if confidence >= config.CONFIDENCE_HIGH:
            return config.LEVERAGE_HIGH
        elif confidence >= config.CONFIDENCE_MEDIUM:
            return config.LEVERAGE_MODERATE
        elif confidence >= config.CONFIDENCE_LOW:
            return config.LEVERAGE_LOW
        else:
            return 0  # Below threshold — do not deploy

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
