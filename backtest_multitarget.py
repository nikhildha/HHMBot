"""
SENTINEL — Multi-Target Partial Profit Booking Backtester
==========================================================
Tests a 3-target profit booking system with progressive SL tightening:
  T1 hit → close 25%, SL → entry (breakeven)
  T2 hit → close 50% of remaining (37.5% of original), SL → T1
  T3 hit → close remaining 37.5%

Phase 2: Tests MAX_LOSS variations (fixed + leverage-dependent)
across the top 4 spacing configs with 1:5 R:R (proven best from Phase 1).
"""

import sys
import os
import json
import logging
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_pipeline import fetch_klines
from feature_engine import compute_all_features, compute_hmm_features, compute_sr_position, compute_ema
from hmm_brain import HMMBrain
from coin_scanner import get_top_coins_by_volume
from risk_manager import RiskManager

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("MultiTargetBT")
logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
BT_INITIAL_BALANCE = 10000.0
BT_CAPITAL_PER_TRADE = 100.0
BT_TIMEFRAME = "1h"
BT_MACRO_TF = "4h"
BT_LOOKBACK = 500
BT_TRAIN_PERIOD = 200
BT_TEST_PERIOD_START = 200
BT_COIN_LIMIT = 15  # Use top 15 coins for speed

# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Target spacing configs: T1 and T2 as fraction of T3 distance
TARGET_SPACINGS = {
    "A-Even":  {"t1_frac": 0.333, "t2_frac": 0.666, "desc": "Even (33%/66%/100%)"},
    "B-Front": {"t1_frac": 0.250, "t2_frac": 0.500, "desc": "Front-loaded (25%/50%/100%)"},
    "C-Back":  {"t1_frac": 0.500, "t2_frac": 0.750, "desc": "Back-loaded (50%/75%/100%)"},
    "D-Fib":   {"t1_frac": 0.382, "t2_frac": 0.618, "desc": "Fibonacci (38.2%/61.8%/100%)"},
}

# R:R ratios (SL : T3) — Phase 2 uses only 1:5 (proven best)
RR_RATIOS = [5]

# Trend adaptive modes — Phase 2 uses only fixed (proven best)
TREND_MODES = {
    "fixed":    {"desc": "Fixed R:R for all trades"},
}

# Capital protection toggle — disabled (Phase 3 proved it hurts performance)
CAPPROTECT_MODES = {
    "NoCap":  False,
}
CAP_TRIGGER_PCT = 10.0
CAP_LOCK_PCT = 4.0

# Fixed leverage overrides for Phase 4 testing
LEVERAGE_MODES = {
    "Lev2x":  2,
    "Lev5x":  5,
    "Lev7x":  7,
}

# MAX_LOSS configurations — Phase 3: Only ML-15 (proven best)
MAX_LOSS_MODES = {
    "ML-30":     {"desc": "Fixed -30%",           "type": "fixed", "pct": -30},
    "ML-25":     {"desc": "Fixed -25%",           "type": "fixed", "pct": -25},
    "ML-20":     {"desc": "Fixed -20%",           "type": "fixed", "pct": -20},
    "ML-15":     {"desc": "Fixed -15%",           "type": "fixed", "pct": -15},
    "ML-LevDep": {"desc": "Leverage-dependent",   "type": "lev_dep",
                  "tiers": {10: -25, 15: -20, 25: -15, 35: -10}},
}

# SL distance in ATR multiples (from current config)
SL_ATR_MULT = 1.5


# ═══════════════════════════════════════════════════════════════════════════════
#  TREND STRENGTH CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
def compute_trend_strength(df, i, lookback=5):
    """
    Compute trend strength score (0–100) using indicators available in df.

    Components (equal weight):
    1. EMA20 slope (5-bar rate of change, normalized by ATR)
    2. EMA20 vs EMA50 spread (normalized by ATR)
    3. RSI extremity (distance from 50)

    Parameters
    ----------
    df : DataFrame with 'close', 'atr' columns
    i : current bar index
    lookback : bars for slope calculation

    Returns
    -------
    float : 0–100 trend strength score
    """
    if i < 50 or i >= len(df):
        return 50.0  # Neutral default

    close = df["close"].values
    atr_val = float(df["atr"].iloc[i]) if "atr" in df.columns else 1.0
    if atr_val <= 0:
        return 50.0

    # 1. EMA20 slope (rate of change over lookback bars)
    ema20 = df["close"].ewm(span=20, adjust=False).mean().values
    if i >= lookback:
        ema_slope = (ema20[i] - ema20[i - lookback]) / atr_val
        # Normalize: ±2 ATR slope → 0-100
        slope_score = min(abs(ema_slope) / 2.0, 1.0) * 100
    else:
        slope_score = 50.0

    # 2. EMA20 vs EMA50 spread
    ema50 = df["close"].ewm(span=50, adjust=False).mean().values
    spread = abs(ema20[i] - ema50[i]) / atr_val
    # Normalize: 0-3 ATR spread → 0-100
    spread_score = min(spread / 3.0, 1.0) * 100

    # 3. RSI extremity
    if "rsi" in df.columns:
        rsi = float(df["rsi"].iloc[i])
    elif "rsi_norm" in df.columns:
        rsi = float(df["rsi_norm"].iloc[i]) * 50 + 50
    else:
        rsi = 50.0
    rsi_extremity = abs(rsi - 50) / 50.0 * 100  # 0 at RSI=50, 100 at RSI=0 or 100

    # Weighted average
    trend_score = (slope_score * 0.40 + spread_score * 0.30 + rsi_extremity * 0.30)
    return min(max(trend_score, 0), 100)


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-TARGET TRADE CLASS
# ═══════════════════════════════════════════════════════════════════════════════
class MultiTargetTrade:
    """
    Trade with 3 take-profit targets and partial profit booking.

    T1 hit → close 25%, SL → entry (breakeven)
    T2 hit → close 50% of remaining (37.5% of original), SL → T1
    T3 hit → close remaining 37.5%
    """

    def __init__(self, symbol, side, entry_price, leverage, atr, confidence,
                 regime, entry_idx, rr_ratio, t1_frac, t2_frac, max_loss_pct=-30,
                 cap_protect=False):
        self.symbol = symbol
        self.side = side  # BUY or SELL
        self.entry_price = entry_price
        self.leverage = leverage
        self.atr = atr
        self.confidence = confidence
        self.regime = regime
        self.entry_idx = entry_idx
        self.rr_ratio = rr_ratio
        self.max_loss_pct = max_loss_pct


        # ── SL and Targets ──
        sl_dist = atr * SL_ATR_MULT
        t3_dist = sl_dist * rr_ratio  # R:R = 1:rr_ratio

        if side == "BUY":
            self.sl_price = entry_price - sl_dist
            self.t1 = entry_price + t3_dist * t1_frac
            self.t2 = entry_price + t3_dist * t2_frac
            self.t3 = entry_price + t3_dist
        else:
            self.sl_price = entry_price + sl_dist
            self.t1 = entry_price - t3_dist * t1_frac
            self.t2 = entry_price - t3_dist * t2_frac
            self.t3 = entry_price - t3_dist

        # ── State tracking ──
        self.current_sl = self.sl_price  # Active SL (moves up)
        self.qty_remaining = 1.0  # Fraction of original position
        self.t1_hit = False
        self.t2_hit = False
        self.closed = False

        # ── P&L tracking (per tranche) ──
        self.tranches = []  # list of {"exit_price", "qty_frac", "reason", "pnl_pct"}
        self.total_pnl = 0.0
        self.total_pnl_pct = 0.0
        self.exit_idx = None
        self.exit_reason = None
        self.commission = 0.0
        self.cap_protect = cap_protect
        self.cap_protect_active = False
        self.cap_protect_sl = None

    def check_exit(self, high, low, close, idx):
        """
        Check all exit conditions on this bar.
        Returns True if trade is fully closed.
        """
        if self.closed:
            return True

        is_long = self.side == "BUY"

        # ── HARD MAX LOSS GUARD ──
        if is_long:
            worst_pnl_pct = ((low - self.entry_price) / self.entry_price) * 100 * self.leverage
        else:
            worst_pnl_pct = ((self.entry_price - high) / self.entry_price) * 100 * self.leverage

        if worst_pnl_pct <= self.max_loss_pct:
            price_move_pct = self.max_loss_pct / (100 * self.leverage)
            if is_long:
                exit_px = self.entry_price * (1 + price_move_pct)
            else:
                exit_px = self.entry_price * (1 - price_move_pct)
            self._close_remaining(exit_px, idx, f"MAX_LOSS_{int(self.max_loss_pct)}%")
            return True

        # ── Check SL ──
        if is_long and low <= self.current_sl:
            reason = "SL_T1" if self.t1_hit else ("SL_T2" if self.t2_hit else "STOP_LOSS")
            if self.cap_protect_active:
                reason = "CAP_PROTECT_SL"
            self._close_remaining(self.current_sl, idx, reason)
            return True
        elif not is_long and high >= self.current_sl:
            reason = "SL_T1" if self.t1_hit else ("SL_T2" if self.t2_hit else "STOP_LOSS")
            if self.cap_protect_active:
                reason = "CAP_PROTECT_SL"
            self._close_remaining(self.current_sl, idx, reason)
            return True

        # ── Capital Protection: lock +4% when P&L >= 10% ──
        if self.cap_protect and not self.cap_protect_active:
            if is_long:
                current_pnl_pct = ((close - self.entry_price) / self.entry_price) * 100 * self.leverage
            else:
                current_pnl_pct = ((self.entry_price - close) / self.entry_price) * 100 * self.leverage
            if current_pnl_pct >= CAP_TRIGGER_PCT:
                lock_price_pct = CAP_LOCK_PCT / (100 * self.leverage)
                if is_long:
                    protect_sl = self.entry_price * (1 + lock_price_pct)
                else:
                    protect_sl = self.entry_price * (1 - lock_price_pct)
                # Only tighten, never loosen
                if (is_long and protect_sl > self.current_sl) or (not is_long and protect_sl < self.current_sl):
                    self.current_sl = protect_sl
                    self.cap_protect_active = True

        # ── Check targets (in order: T1 → T2 → T3) ──
        if not self.t1_hit:
            if (is_long and high >= self.t1) or (not is_long and low <= self.t1):
                # T1 hit: book 25% of original, SL → entry
                book_qty = 0.25
                self._book_partial(self.t1, book_qty, idx, "T1")
                self.current_sl = self.entry_price  # SL → breakeven
                self.t1_hit = True

        if not self.t2_hit and self.t1_hit:
            if (is_long and high >= self.t2) or (not is_long and low <= self.t2):
                # T2 hit: book 50% of remaining (37.5% of original)
                book_qty = self.qty_remaining * 0.50
                self._book_partial(self.t2, book_qty, idx, "T2")
                self.current_sl = self.t1  # SL → T1
                self.t2_hit = True

        if self.t2_hit:
            if (is_long and high >= self.t3) or (not is_long and low <= self.t3):
                # T3 hit: close everything remaining
                self._close_remaining(self.t3, idx, "T3")
                return True

        return self.closed

    def _book_partial(self, exit_price, qty_frac, idx, reason):
        """Book partial profit for a fraction of the position."""
        if is_long := (self.side == "BUY"):
            pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100 * self.leverage
        else:
            pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100 * self.leverage

        # Deduct commission for this tranche
        comm_pct = config.TAKER_FEE * 2 * 100  # Entry + exit legs
        pnl_pct -= comm_pct

        # Dollar P&L for this tranche
        tranche_pnl = (pnl_pct / 100) * BT_CAPITAL_PER_TRADE * qty_frac
        tranche_comm = BT_CAPITAL_PER_TRADE * qty_frac * config.TAKER_FEE * 2

        self.tranches.append({
            "exit_price": exit_price,
            "qty_frac": qty_frac,
            "reason": reason,
            "pnl_pct": round(pnl_pct, 4),
            "pnl": round(tranche_pnl, 4),
        })

        self.qty_remaining -= qty_frac
        self.total_pnl += tranche_pnl
        self.commission += tranche_comm

        if self.qty_remaining <= 0.001:
            self.closed = True
            self.exit_idx = idx
            self.exit_reason = reason
            self._finalize()

    def _close_remaining(self, exit_price, idx, reason):
        """Close all remaining quantity at given price."""
        if self.qty_remaining > 0.001:
            self._book_partial(exit_price, self.qty_remaining, idx, reason)
        self.closed = True
        self.exit_idx = idx
        self.exit_reason = reason
        self._finalize()

    def _finalize(self):
        """Compute total P&L percentage."""
        self.total_pnl_pct = (self.total_pnl / BT_CAPITAL_PER_TRADE) * 100

    def force_close(self, close_price, idx):
        """Force close at end of data."""
        self._close_remaining(close_price, idx, "END_OF_DATA")


# ═══════════════════════════════════════════════════════════════════════════════
#  COIN DATA FETCHER (shared across experiments)
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_coin_data(symbol):
    """
    Fetch and prepare all data for a coin (1h + 4h + features + HMM brain).
    Returns dict with everything needed for experiments, or None if insufficient.
    """
    df_1h = fetch_klines(symbol, BT_TIMEFRAME, limit=BT_LOOKBACK)
    if df_1h is None or len(df_1h) < BT_TRAIN_PERIOD + 50:
        return None

    df_4h = fetch_klines(symbol, BT_MACRO_TF, limit=BT_LOOKBACK)

    # Compute features
    df_feat = compute_all_features(df_1h)
    df_hmm = compute_hmm_features(df_1h)

    # Train 1h HMM
    brain = HMMBrain()
    brain.train(df_hmm.iloc[:BT_TRAIN_PERIOD])
    if not brain.is_trained:
        return None

    # Train 4h macro brain
    macro_brain = None
    df_4h_feat = None
    if df_4h is not None and len(df_4h) >= 100:
        df_4h_feat_full = compute_all_features(df_4h)
        df_4h_hmm = compute_hmm_features(df_4h)
        macro_brain = HMMBrain()
        macro_brain.train(df_4h_hmm.iloc[:min(BT_TRAIN_PERIOD, len(df_4h_hmm))])
        df_4h_feat = df_4h_feat_full

    return {
        "symbol": symbol,
        "df_feat": df_feat,
        "brain": brain,
        "macro_brain": macro_brain,
        "df_4h_feat": df_4h_feat,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLE COIN BACKTESTER (for a given experiment config)
# ═══════════════════════════════════════════════════════════════════════════════
def backtest_coin_multitarget(coin_data, rr_ratio, t1_frac, t2_frac,
                               trend_adaptive=False, btc_macro_regimes=None,
                               max_loss_mode=None, cap_protect=False,
                               lev_override=None):
    """
    Run multi-target backtest on a single coin with given config.
    Returns list of completed MultiTargetTrade objects.
    """
    symbol = coin_data["symbol"]
    df_feat = coin_data["df_feat"]
    brain = coin_data["brain"]
    macro_brain = coin_data["macro_brain"]
    df_4h_feat = coin_data["df_4h_feat"]

    trades = []
    active_trade = None

    for i in range(BT_TEST_PERIOD_START, len(df_feat)):
        row = df_feat.iloc[i]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # Check active trade exits
        if active_trade:
            if active_trade.check_exit(high, low, close, i):
                trades.append(active_trade)
                active_trade = None
            continue  # One trade at a time per coin

        # Predict regime
        window = df_feat.iloc[:i + 1]
        try:
            regime, conf = brain.predict(window)
            regime_name = brain.get_regime_name(regime)
        except Exception:
            continue

        # Skip CRASH
        if regime == config.REGIME_CRASH:
            continue

        # 4h macro confirmation
        if macro_brain and macro_brain.is_trained and df_4h_feat is not None:
            try:
                macro_idx = min(i // 4, len(df_4h_feat) - 1)
                macro_window = df_4h_feat.iloc[:macro_idx + 1]
                if len(macro_window) > 10:
                    macro_regime, _ = macro_brain.predict(macro_window)
                    macro_name = macro_brain.get_regime_name(macro_regime)
                    if regime_name == "BULLISH" and macro_name == "BEARISH":
                        continue
                    if regime_name == "BEARISH" and macro_name == "BULLISH":
                        continue
                    if macro_name == "CRASH":
                        continue
            except Exception:
                pass

        # ATR check
        atr = float(row["atr"]) if "atr" in df_feat.columns else 0
        if atr <= 0:
            continue

        # Volatility filter
        vol_ratio = atr / close
        if config.VOL_FILTER_ENABLED:
            if vol_ratio < config.VOL_MIN_ATR_PCT or vol_ratio > config.VOL_MAX_ATR_PCT:
                continue

        # Determine side
        if regime == config.REGIME_BULL:
            side = "BUY"
        elif regime == config.REGIME_BEAR:
            side = "SELL"
        else:
            continue  # Skip chop

        # Conviction scoring
        btc_regime = None
        if btc_macro_regimes is not None and i < len(btc_macro_regimes):
            btc_regime = int(btc_macro_regimes.iloc[i])

        sr_pos, vwap_pos = compute_sr_position(df_feat.iloc[:i + 1], lookback=50)

        conviction = RiskManager.compute_conviction_score(
            confidence=conf,
            regime=regime,
            side=side,
            btc_regime=btc_regime,
            funding_rate=None,
            sr_position=sr_pos,
            vwap_position=vwap_pos,
            oi_change=None,
            volatility=vol_ratio,
            sentiment_score=None,
            orderflow_score=None,
        )

        if lev_override is not None:
            leverage = lev_override
        else:
            leverage = RiskManager.get_conviction_leverage(conviction)
            if leverage <= 0:
                continue

        # ── Determine R:R ratio ──
        effective_rr = rr_ratio
        if trend_adaptive:
            trend_score = compute_trend_strength(df_feat, i)
            if trend_score >= 60:
                effective_rr = 5  # Strong trend → let profits run
            else:
                effective_rr = 3  # Normal → tighter targets

        # ── Determine max loss for this trade ──
        if max_loss_mode is None:
            ml_pct = config.MAX_LOSS_PER_TRADE_PCT
        elif max_loss_mode["type"] == "fixed":
            ml_pct = max_loss_mode["pct"]
        elif max_loss_mode["type"] == "lev_dep":
            tiers = max_loss_mode["tiers"]
            # Find closest tier <= leverage
            matching = [v for k, v in sorted(tiers.items()) if leverage >= k]
            ml_pct = matching[-1] if matching else -30
        else:
            ml_pct = -30

        # Open trade
        active_trade = MultiTargetTrade(
            symbol=symbol,
            side=side,
            entry_price=close,
            leverage=leverage,
            atr=atr,
            confidence=conf,
            regime=regime_name,
            entry_idx=i,
            rr_ratio=effective_rr,
            t1_frac=t1_frac,
            t2_frac=t2_frac,
            max_loss_pct=ml_pct,
            cap_protect=cap_protect,
        )

    # Force close remaining
    if active_trade:
        active_trade.force_close(float(df_feat.iloc[-1]["close"]), len(df_feat) - 1)
        trades.append(active_trade)

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def run_experiment(coin_data_list, spacing_name, spacing_cfg, rr_ratio,
                   trend_mode, btc_macro_regimes=None, max_loss_name="ML-30",
                   max_loss_mode=None, cap_name="NoCap", cap_protect=False,
                   lev_name="Dynamic", lev_override=None):
    """
    Run one experiment config across all coins.
    Returns summary dict.
    """
    trend_adaptive = (trend_mode == "adaptive")
    all_trades = []

    for cd in coin_data_list:
        trades = backtest_coin_multitarget(
            coin_data=cd,
            rr_ratio=rr_ratio,
            t1_frac=spacing_cfg["t1_frac"],
            t2_frac=spacing_cfg["t2_frac"],
            trend_adaptive=trend_adaptive,
            btc_macro_regimes=btc_macro_regimes,
            max_loss_mode=max_loss_mode,
            cap_protect=cap_protect,
            lev_override=lev_override,
        )
        all_trades.extend(trades)

    if not all_trades:
        return {
            "config": f"{spacing_name}_RR1:{rr_ratio}_{trend_mode}_{max_loss_name}_{cap_name}",
            "trades": 0, "wins": 0, "win_rate": 0,
            "total_pnl": 0, "profit_factor": 0, "max_dd_pct": 0,
            "avg_win": 0, "avg_loss": 0,
            "t1_hits": 0, "t2_hits": 0, "t3_hits": 0,
        }

    # ── Compute metrics ──
    total = len(all_trades)
    wins = [t for t in all_trades if t.total_pnl > 0]
    losses = [t for t in all_trades if t.total_pnl <= 0]

    total_pnl = sum(t.total_pnl for t in all_trades)
    win_pnl = sum(t.total_pnl for t in wins) if wins else 0
    loss_pnl = sum(t.total_pnl for t in losses) if losses else 0
    avg_win = win_pnl / len(wins) if wins else 0
    avg_loss = loss_pnl / len(losses) if losses else 0
    profit_factor = abs(win_pnl / loss_pnl) if loss_pnl != 0 else float("inf")
    win_rate = len(wins) / total * 100

    # Max drawdown
    equity = BT_INITIAL_BALANCE
    peak = equity
    max_dd_pct = 0
    for t in sorted(all_trades, key=lambda x: x.entry_idx):
        equity += t.total_pnl
        if equity > peak:
            peak = equity
        dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0
        max_dd_pct = max(max_dd_pct, dd_pct)

    # Target hit analysis
    t1_hits = sum(1 for t in all_trades if t.t1_hit)
    t2_hits = sum(1 for t in all_trades if t.t2_hit)
    t3_hits = sum(1 for t in all_trades if t.exit_reason == "T3")

    # Exit reason breakdown
    exit_reasons = {}
    for t in all_trades:
        r = t.exit_reason
        if r not in exit_reasons:
            exit_reasons[r] = {"count": 0, "pnl": 0}
        exit_reasons[r]["count"] += 1
        exit_reasons[r]["pnl"] += t.total_pnl
    for r in exit_reasons:
        exit_reasons[r]["pnl"] = round(exit_reasons[r]["pnl"], 2)

    return {
        "config": f"{spacing_name}_RR1:{rr_ratio}_{trend_mode}_{max_loss_name}_{cap_name}",
        "spacing": spacing_name,
        "rr_ratio": f"1:{rr_ratio}",
        "trend_mode": trend_mode,
        "max_loss": max_loss_name,
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / BT_INITIAL_BALANCE * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "max_dd_pct": round(max_dd_pct, 2),
        "final_equity": round(equity, 2),
        "t1_hit_pct": round(t1_hits / total * 100, 1) if total else 0,
        "t2_hit_pct": round(t2_hits / total * 100, 1) if total else 0,
        "t3_hit_pct": round(t3_hits / total * 100, 1) if total else 0,
        "exit_reasons": exit_reasons,
    }


def run_all_experiments():
    """Run all experiment configs and generate comparison."""
    W = 100
    DIV = "=" * W

    # Phase 4: Leverage A/B — ML-15 × 4 spacings × 3 fixed leverage levels
    ml_15_cfg = {"desc": "Fixed -15%", "type": "fixed", "pct": -15}
    total_exps = len(TARGET_SPACINGS) * len(LEVERAGE_MODES)

    logger.info(DIV)
    logger.info("  SENTINEL — MULTI-TARGET BACKTESTER (Phase 4: Leverage A/B)")
    logger.info("  %d configs × %d coins", total_exps, BT_COIN_LIMIT)
    logger.info(DIV)

    # ── Pre-compute BTC macro regime ──
    logger.info("\n📡 Pre-computing BTC macro regime...")
    btc_macro_regimes = None
    try:
        df_btc = fetch_klines("BTCUSDT", BT_TIMEFRAME, limit=BT_LOOKBACK)
        if df_btc is not None and len(df_btc) >= 200:
            import pandas as pd
            df_btc_hmm = compute_hmm_features(df_btc)
            btc_brain = HMMBrain()
            btc_brain.train(df_btc_hmm.iloc[:BT_TRAIN_PERIOD])
            if btc_brain.is_trained:
                states = btc_brain.predict_all(df_btc_hmm)
                btc_macro_regimes = pd.Series(states, index=df_btc.index[:len(states)])
                bull = int((btc_macro_regimes == config.REGIME_BULL).sum())
                bear = int((btc_macro_regimes == config.REGIME_BEAR).sum())
                chop = int((btc_macro_regimes == config.REGIME_CHOP).sum())
                logger.info("   ✅ BTC macro: %d Bull | %d Bear | %d Chop", bull, bear, chop)
    except Exception as e:
        logger.warning("   ⚠️ BTC macro failed: %s", e)

    # ── Fetch coin data (shared across all experiments) ──
    logger.info("\n🔍 Fetching top %d coins...", BT_COIN_LIMIT)
    coins = get_top_coins_by_volume(limit=BT_COIN_LIMIT)
    logger.info("   Found %d coins", len(coins))

    coin_data_list = []
    for idx, symbol in enumerate(coins):
        logger.info("   [%d/%d] Loading %s...", idx + 1, len(coins), symbol)
        cd = fetch_coin_data(symbol)
        if cd:
            coin_data_list.append(cd)
        else:
            logger.info("      ⚠️ Skipped (insufficient data)")
        time.sleep(0.3)

    logger.info("\n   ✅ %d coins loaded successfully\n", len(coin_data_list))

    if not coin_data_list:
        logger.error("No coins available for testing!")
        return

    # ── Run all experiments ──
    results = []
    exp_num = 0

    for spacing_name, spacing_cfg in TARGET_SPACINGS.items():
        for lev_name, lev_val in LEVERAGE_MODES.items():
            exp_num += 1
            cfg_label = f"{spacing_name}_1:5_ML-15_{lev_name}"
            logger.info("[%d/%d] Running %s...", exp_num, total_exps, cfg_label)

            result = run_experiment(
                coin_data_list=coin_data_list,
                spacing_name=spacing_name,
                spacing_cfg=spacing_cfg,
                rr_ratio=5,
                trend_mode="fixed",
                btc_macro_regimes=btc_macro_regimes,
                max_loss_name="ML-15",
                max_loss_mode=ml_15_cfg,
                cap_name="NoCap",
                cap_protect=False,
                lev_name=lev_name,
                lev_override=lev_val,
            )
            results.append(result)

            logger.info("   → %d trades | WR: %.1f%% | P&L: $%+.2f | PF: %s",
                        result["trades"], result["win_rate"],
                        result["total_pnl"], result["profit_factor"])

    # ── Sort by total P&L (best first) ──
    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    # ── Print comparison table ──
    print_comparison(results)

    # ── Save full report ──
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "phase": "Phase 2: MAX_LOSS tuning",
        "coins_tested": len(coin_data_list),
        "coin_symbols": [cd["symbol"] for cd in coin_data_list],
        "baseline": "Phase 1 best: A-Even 1:5 fixed ML-30 → +$73.57 (PF 1.03)",
        "results": results,
    }

    report_path = os.path.join(config.DATA_DIR, "backtest_multitarget_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("\n📁 Full report saved to: %s", report_path)

    return results


def print_comparison(results):
    """Print formatted comparison table of all experiment results."""
    print("\n" + "=" * 120)
    print("  📊 MULTI-TARGET BACKTEST — COMPARISON TABLE (sorted by P&L)")
    print("=" * 120)

    header = f"  {'Rank':<5} {'Config':<40} {'Trades':>6} {'WR%':>6} {'Total P&L':>11} " \
             f"{'Return%':>8} {'PF':>6} {'MaxDD%':>7} {'AvgWin':>8} {'AvgLoss':>8} " \
             f"{'T1%':>5} {'T2%':>5} {'T3%':>5}"
    print(header)
    print("  " + "─" * 126)

    for rank, r in enumerate(results, 1):
        pf_str = f"{r['profit_factor']}" if isinstance(r['profit_factor'], str) else f"{r['profit_factor']:.2f}"
        pnl_color = "+" if r["total_pnl"] > 0 else ""
        print(f"  {rank:<5} {r['config']:<40} {r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"${pnl_color}{r['total_pnl']:>9.2f} {r['return_pct']:>+7.2f}% {pf_str:>6} "
              f"{r['max_dd_pct']:>6.1f}% ${r['avg_win']:>+7.2f} ${r['avg_loss']:>7.2f} "
              f"{r.get('t1_hit_pct', 0):>4.0f}% {r.get('t2_hit_pct', 0):>4.0f}% {r.get('t3_hit_pct', 0):>4.0f}%")

    print("=" * 130)
    print(f"\n  📍 Phase 1 best: A-Even 1:5 fixed ML-30 → +$73.57 (PF 1.03)")
    print(f"  📍 Phase 2 best: {results[0]['config']} → ${results[0]['total_pnl']:+.2f}")
    if results[0]["total_pnl"] > 73.57:
        improvement = results[0]["total_pnl"] - 73.57
        print(f"  ✅ Improvement over Phase 1 best: ${improvement:+.2f}")
    print()

    # ── Print exit reason breakdown for top 3 ──
    print("  " + "─" * 80)
    print("  EXIT REASON BREAKDOWN (Top 3 configs)")
    print("  " + "─" * 80)
    for i, r in enumerate(results[:3]):
        print(f"\n  #{i+1} {r['config']}:")
        if "exit_reasons" in r:
            for reason, stats in sorted(r["exit_reasons"].items(), key=lambda x: -abs(x[1]["pnl"])):
                print(f"      {reason:<20} | {stats['count']:>4} trades | P&L: ${stats['pnl']:>+9.2f}")


if __name__ == "__main__":
    run_all_experiments()
