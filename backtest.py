"""
SENTINEL — Backtesting Engine
Tests the HMM regime strategy across 50 coins using historical data.
Simulates entries/exits with ATR-based SL/TP and dynamic leverage.
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
from feature_engine import compute_all_features, compute_hmm_features
from hmm_brain import HMMBrain
from coin_scanner import get_top_coins_by_volume
from risk_manager import RiskManager
from sideways_strategy import evaluate_mean_reversion

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("Backtest")
logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
BT_INITIAL_BALANCE = 10000.0  # Starting paper balance
BT_CAPITAL_PER_TRADE = 100.0  # Per-trade allocation
BT_TIMEFRAME = "1h"           # Primary analysis timeframe
BT_MACRO_TF = "4h"            # Macro confirmation timeframe
BT_LOOKBACK = 500             # Candles to fetch
BT_TRAIN_PERIOD = 200         # First N candles used for HMM training
BT_TEST_PERIOD_START = 200    # Start testing from this candle index


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATED TRADE
# ═══════════════════════════════════════════════════════════════════════════════
class SimTrade:
    def __init__(self, symbol, side, entry_price, leverage, atr, confidence, regime, entry_idx):
        self.symbol = symbol
        self.side = side  # BUY or SELL
        self.entry_price = entry_price
        self.leverage = leverage
        self.atr = atr
        self.confidence = confidence
        self.regime = regime
        self.entry_idx = entry_idx
        self.exit_price = None
        self.exit_idx = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_pct = 0.0

        # Fixed SL / TP (adjusted for leverage)
        sl_mult, tp_mult = config.get_atr_multipliers(leverage)
        if side == "BUY":
            self.sl = entry_price - atr * sl_mult
            self.tp = entry_price + atr * tp_mult
        else:
            self.sl = entry_price + atr * sl_mult
            self.tp = entry_price - atr * tp_mult

        # Trailing state
        self.trailing_sl = self.sl
        self.trailing_tp = self.tp
        self.peak_price = entry_price
        self.trailing_active = False
        self.tp_extensions = 0
        self.commission = 0.0

    def check_exit(self, high, low, close, idx):
        """Check if SL or TP hit, with trailing logic. Returns True if trade closed."""
        is_long = self.side == "BUY"
        atr = self.atr

        # ── Update peak price (high-water mark) ──
        if is_long:
            self.peak_price = max(self.peak_price, high)
        else:
            self.peak_price = min(self.peak_price, low)

        # ── Trailing Stop Loss ──
        if config.TRAILING_SL_ENABLED and atr > 0:
            activation_dist = atr * config.TRAILING_SL_ACTIVATION_ATR
            if is_long:
                favorable_move = high - self.entry_price
            else:
                favorable_move = self.entry_price - low

            if favorable_move >= activation_dist:
                self.trailing_active = True

            if self.trailing_active:
                trail_dist = atr * config.TRAILING_SL_DISTANCE_ATR
                if is_long:
                    new_sl = self.peak_price - trail_dist
                    if new_sl > self.trailing_sl:
                        self.trailing_sl = new_sl
                else:
                    new_sl = self.peak_price + trail_dist
                    if new_sl < self.trailing_sl:
                        self.trailing_sl = new_sl

        # ── Trailing Take Profit ──
        if config.TRAILING_TP_ENABLED and atr > 0:
            max_ext = config.TRAILING_TP_MAX_EXTENSIONS
            if self.tp_extensions < max_ext:
                if is_long:
                    tp_dist = self.trailing_tp - self.entry_price
                    progress = (high - self.entry_price) / tp_dist if tp_dist > 0 else 0
                else:
                    tp_dist = self.entry_price - self.trailing_tp
                    progress = (self.entry_price - low) / tp_dist if tp_dist > 0 else 0

                if progress >= config.TRAILING_TP_ACTIVATION_PCT:
                    ext_amount = atr * config.TRAILING_TP_EXTENSION_ATR
                    if is_long:
                        self.trailing_tp += ext_amount
                    else:
                        self.trailing_tp -= ext_amount
                    self.tp_extensions += 1

        # ── Check exits using effective (trailing) levels ──
        effective_sl = self.trailing_sl
        effective_tp = self.trailing_tp

        if is_long:
            if low <= effective_sl:
                reason = "TRAILING_SL" if self.trailing_active else "STOP_LOSS"
                self._close(effective_sl, idx, reason)
                return True
            if high >= effective_tp:
                reason = "TRAILING_TP" if self.tp_extensions > 0 else "TAKE_PROFIT"
                self._close(effective_tp, idx, reason)
                return True
        else:
            if high >= effective_sl:
                reason = "TRAILING_SL" if self.trailing_active else "STOP_LOSS"
                self._close(effective_sl, idx, reason)
                return True
            if low <= effective_tp:
                reason = "TRAILING_TP" if self.tp_extensions > 0 else "TAKE_PROFIT"
                self._close(effective_tp, idx, reason)
                return True
        return False

    def _close(self, exit_price, idx, reason):
        self.exit_price = exit_price
        self.exit_idx = idx
        self.exit_reason = reason
        if self.side == "BUY":
            self.pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100 * self.leverage
        else:
            self.pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100 * self.leverage

        # Deduct Binance Futures commission (taker fee on entry + exit)
        commission_pct = config.TAKER_FEE * 2 * 100  # Both legs, as percentage
        self.pnl_pct -= commission_pct
        self.pnl = (self.pnl_pct / 100) * BT_CAPITAL_PER_TRADE
        self.commission = round(BT_CAPITAL_PER_TRADE * config.TAKER_FEE * 2, 4)

    def force_close(self, close_price, idx):
        self._close(close_price, idx, "END_OF_DATA")


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def backtest_coin(symbol):
    """Run backtest on a single coin. Returns list of closed trades."""
    # Fetch data
    df_1h = fetch_klines(symbol, BT_TIMEFRAME, limit=BT_LOOKBACK)
    if df_1h is None or len(df_1h) < BT_TRAIN_PERIOD + 50:
        return [], symbol, "INSUFFICIENT_DATA"

    df_4h = fetch_klines(symbol, BT_MACRO_TF, limit=BT_LOOKBACK)

    # Compute features on full dataset
    df_feat = compute_all_features(df_1h)
    df_hmm = compute_hmm_features(df_1h)

    # Train HMM on first portion
    brain = HMMBrain()
    train_data = df_hmm.iloc[:BT_TRAIN_PERIOD]
    brain.train(train_data)

    if not brain.is_trained:
        return [], symbol, "TRAIN_FAILED"

    # Train 4h brain if data available
    macro_brain = None
    df_4h_feat = None
    if df_4h is not None and len(df_4h) >= 100:
        df_4h_feat_full = compute_all_features(df_4h)
        df_4h_hmm = compute_hmm_features(df_4h)
        macro_brain = HMMBrain()
        macro_brain.train(df_4h_hmm.iloc[:min(BT_TRAIN_PERIOD, len(df_4h_hmm))])
        df_4h_feat = df_4h_feat_full

    # Walk forward through test period
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
            continue  # Only one trade at a time per coin

        # Predict regime on data up to current candle
        window = df_feat.iloc[:i+1]
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
                # Map 1h index to approximate 4h index
                macro_idx = min(i // 4, len(df_4h_feat) - 1)
                macro_window = df_4h_feat.iloc[:macro_idx+1]
                if len(macro_window) > 10:
                    macro_regime, _ = macro_brain.predict(macro_window)
                    macro_regime_name = macro_brain.get_regime_name(macro_regime)

                    # Conflict filter
                    if regime_name == "BULLISH" and macro_regime_name == "BEARISH":
                        continue
                    if regime_name == "BEARISH" and macro_regime_name == "BULLISH":
                        continue
                    if macro_regime_name == "CRASH":
                        continue
            except Exception:
                pass

        # Get ATR
        atr = float(row["atr"]) if "atr" in df_feat.columns else 0
        if atr <= 0:
            continue

        # Volatility band filter: skip if coin is too dead or too wild
        if config.VOL_FILTER_ENABLED:
            vol_ratio = atr / close
            if vol_ratio < config.VOL_MIN_ATR_PCT:
                continue  # Too dead — no edge
            if vol_ratio > config.VOL_MAX_ATR_PCT:
                continue  # Too wild — high SL hit rate

        # Determine leverage
        leverage = RiskManager.get_dynamic_leverage(conf, regime)
        if leverage == 0:
            continue

        # Determine side
        if regime == config.REGIME_BULL:
            side = "BUY"
        elif regime == config.REGIME_BEAR:
            side = "SELL"
        elif regime == config.REGIME_CHOP:
            # Simple mean reversion: if RSI > 70 sell, RSI < 30 buy
            rsi = float(row.get("rsi", 50))
            if rsi > 70:
                side = "SELL"
            elif rsi < 30:
                side = "BUY"
            else:
                continue
        else:
            continue

        # Open trade
        active_trade = SimTrade(
            symbol=symbol,
            side=side,
            entry_price=close,
            leverage=leverage,
            atr=atr,
            confidence=conf,
            regime=regime_name,
            entry_idx=i,
        )

    # Force close any remaining trade
    if active_trade:
        active_trade.force_close(float(df_feat.iloc[-1]["close"]), len(df_feat) - 1)
        trades.append(active_trade)

    return trades, symbol, "OK"


def load_server_params(param_path):
    """Load params from server JSON file and override globals + config."""
    global BT_INITIAL_BALANCE, BT_CAPITAL_PER_TRADE, BT_TIMEFRAME, BT_MACRO_TF
    global BT_LOOKBACK, BT_TRAIN_PERIOD, BT_TEST_PERIOD_START

    with open(param_path, "r") as f:
        p = json.load(f)

    BT_INITIAL_BALANCE = p.get("initialBalance", BT_INITIAL_BALANCE)
    BT_CAPITAL_PER_TRADE = p.get("capitalPerTrade", BT_CAPITAL_PER_TRADE)
    BT_TIMEFRAME = p.get("tfPrimary", BT_TIMEFRAME)
    BT_MACRO_TF = p.get("tfMacro", BT_MACRO_TF)
    BT_LOOKBACK = p.get("hmmLookback", BT_LOOKBACK)
    BT_TRAIN_PERIOD = p.get("trainPeriod", BT_TRAIN_PERIOD)
    BT_TEST_PERIOD_START = BT_TRAIN_PERIOD

    # Override config values
    config.LEVERAGE_HIGH = p.get("levHigh", config.LEVERAGE_HIGH)
    config.LEVERAGE_MODERATE = p.get("levModerate", config.LEVERAGE_MODERATE)
    config.LEVERAGE_LOW = p.get("levLow", config.LEVERAGE_LOW)
    config.CONFIDENCE_HIGH = p.get("confHigh", config.CONFIDENCE_HIGH)
    config.CONFIDENCE_MEDIUM = p.get("confMedium", config.CONFIDENCE_MEDIUM)
    config.CONFIDENCE_LOW = p.get("confLow", config.CONFIDENCE_LOW)

    # SL/TP mode: 'fixed' uses manual values, 'dynamic' uses get_atr_multipliers()
    atr_mode = p.get("atrMode", "dynamic")
    if atr_mode == "fixed":
        config.ATR_SL_MULTIPLIER = p.get("atrSL", config.ATR_SL_MULTIPLIER)
        config.ATR_TP_MULTIPLIER = p.get("atrTP", config.ATR_TP_MULTIPLIER)
    # If dynamic, leave config.ATR_SL/TP_MULTIPLIER as-is; SimTrade uses get_atr_multipliers()

    config.RISK_PER_TRADE = p.get("riskPerTrade", config.RISK_PER_TRADE)
    config.MAX_LOSS_PER_TRADE_PCT = p.get("maxLoss", config.MAX_LOSS_PER_TRADE_PCT)
    config.MIN_HOLD_MINUTES = p.get("minHoldMinutes", getattr(config, "MIN_HOLD_MINUTES", 30))
    config.HMM_N_STATES = p.get("hmmStates", config.HMM_N_STATES)
    config.HMM_COVARIANCE = p.get("hmmCovariance", config.HMM_COVARIANCE)
    config.TIMEFRAME_MACRO = BT_MACRO_TF

    # Taker fee override
    taker_fee = p.get("takerFeePct", None)
    if taker_fee is not None:
        config.TAKER_FEE = taker_fee

    # Trailing SL/TP overrides
    config.TRAILING_SL_ACTIVATION_ATR = p.get("trailSlActivation", config.TRAILING_SL_ACTIVATION_ATR)
    config.TRAILING_SL_DISTANCE_ATR = p.get("trailSlDistance", config.TRAILING_SL_DISTANCE_ATR)
    config.TRAILING_TP_ACTIVATION_PCT = p.get("trailTpActivation", config.TRAILING_TP_ACTIVATION_PCT)
    config.TRAILING_TP_EXTENSION_ATR = p.get("trailTpExtension", config.TRAILING_TP_EXTENSION_ATR)
    config.TRAILING_TP_MAX_EXTENSIONS = p.get("trailTpMaxExt", config.TRAILING_TP_MAX_EXTENSIONS)

    # Volatility filter overrides
    config.VOL_FILTER_ENABLED = p.get("volFilterEnabled", config.VOL_FILTER_ENABLED)
    config.VOL_MIN_ATR_PCT = p.get("volMinPct", config.VOL_MIN_ATR_PCT)
    config.VOL_MAX_ATR_PCT = p.get("volMaxPct", config.VOL_MAX_ATR_PCT)

    return p.get("coinCount", 50)


def emit(msg):
    """Print JSON message to stdout for server streaming."""
    print(json.dumps(msg), flush=True)


def run_backtest(server_mode=False, coin_limit=50):
    """Run backtest across all coins and generate report."""
    if not server_mode:
        logger.info("=" * 70)
        logger.info("  SENTINEL — BACKTEST ENGINE")
        logger.info("  Strategy: HMM Regime + Multi-TF (1h + 4h) + ATR SL/TP")
        logger.info("=" * 70)

    # Get coin list
    if server_mode:
        emit({"type": "progress", "pct": 2, "label": "Fetching coin list...", "log": "🔍 Fetching top coins by volume..."})
    else:
        logger.info("\n🔍 Fetching top %d coins by volume...", coin_limit)

    coins = get_top_coins_by_volume(limit=coin_limit)

    if server_mode:
        emit({"type": "progress", "pct": 5, "label": f"Found {len(coins)} coins", "log": f"   Found {len(coins)} coins"})
    else:
        logger.info(f"   Found {len(coins)} coins")

    all_trades = []
    coin_results = {}
    failed = []

    for idx, symbol in enumerate(coins):
        pct = 5 + int((idx / len(coins)) * 90)
        if server_mode:
            emit({"type": "progress", "pct": pct, "label": f"Backtesting {symbol} ({idx+1}/{len(coins)})", "log": f"[{idx+1}/{len(coins)}] {symbol}..."})
        else:
            logger.info(f"\n[{idx+1}/{len(coins)}] Backtesting {symbol}...")

        try:
            trades, sym, status = backtest_coin(symbol)
            if status != "OK":
                if server_mode:
                    emit({"type": "progress", "pct": pct, "label": f"{symbol}: {status}", "log": f"   ⚠️ {symbol}: {status}"})
                else:
                    logger.info(f"   ⚠️ Skipped: {status}")
                failed.append((symbol, status))
                continue

            coin_results[symbol] = trades
            all_trades.extend(trades)

            wins = sum(1 for t in trades if t.pnl > 0)
            losses = sum(1 for t in trades if t.pnl <= 0)
            total_pnl = sum(t.pnl for t in trades)

            log_msg = f"   📊 {len(trades)} trades | W:{wins} L:{losses} | P&L: ${total_pnl:.2f}"
            if server_mode:
                emit({"type": "progress", "pct": pct, "label": f"{symbol}: {len(trades)} trades, ${total_pnl:.2f}", "log": log_msg})
            else:
                logger.info(log_msg)

            time.sleep(0.5)  # Rate limit

        except Exception as e:
            if server_mode:
                emit({"type": "progress", "pct": pct, "label": f"{symbol}: Error", "log": f"   ❌ {symbol}: {e}"})
            else:
                logger.warning(f"   ❌ Error: {e}")
            failed.append((symbol, str(e)))

    # Generate report
    if server_mode:
        emit({"type": "progress", "pct": 98, "label": "Generating report...", "log": "📊 Generating report..."})

    report = generate_report(all_trades, coin_results, coins, failed)

    report_path = os.path.join(config.DATA_DIR, "backtest_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if server_mode:
        emit({"type": "progress", "pct": 100, "label": "Complete!", "log": "✅ Backtest complete!"})
    else:
        print_report(report)

    return report



def generate_report(all_trades, coin_results, coins, failed):
    """Generate comprehensive backtest report."""
    if not all_trades:
        return {"error": "No trades generated"}

    total_trades = len(all_trades)
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]

    total_pnl = sum(t.pnl for t in all_trades)
    total_commission = sum(getattr(t, 'commission', 0) for t in all_trades)
    win_pnl = sum(t.pnl for t in wins)
    loss_pnl = sum(t.pnl for t in losses)

    avg_win = win_pnl / len(wins) if wins else 0
    avg_loss = loss_pnl / len(losses) if losses else 0
    win_rate = len(wins) / total_trades * 100

    # Profit factor
    profit_factor = abs(win_pnl / loss_pnl) if loss_pnl != 0 else float("inf")

    # Max drawdown (equity curve)
    equity = BT_INITIAL_BALANCE
    peak = equity
    max_dd = 0
    max_dd_pct = 0
    equity_curve = [equity]

    for t in sorted(all_trades, key=lambda x: x.entry_idx):
        equity += t.pnl
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = (dd / peak * 100) if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd = dd

    # By regime breakdown
    regime_stats = {}
    for t in all_trades:
        r = t.regime
        if r not in regime_stats:
            regime_stats[r] = {"trades": 0, "wins": 0, "pnl": 0}
        regime_stats[r]["trades"] += 1
        if t.pnl > 0:
            regime_stats[r]["wins"] += 1
        regime_stats[r]["pnl"] += t.pnl

    for r in regime_stats:
        s = regime_stats[r]
        s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0
        s["pnl"] = round(s["pnl"], 2)

    # By exit reason
    exit_reasons = {}
    for t in all_trades:
        reason = t.exit_reason
        if reason not in exit_reasons:
            exit_reasons[reason] = {"count": 0, "pnl": 0}
        exit_reasons[reason]["count"] += 1
        exit_reasons[reason]["pnl"] += t.pnl

    for r in exit_reasons:
        exit_reasons[r]["pnl"] = round(exit_reasons[r]["pnl"], 2)

    # By leverage
    lev_stats = {}
    for t in all_trades:
        l = t.leverage
        if l not in lev_stats:
            lev_stats[l] = {"trades": 0, "wins": 0, "pnl": 0}
        lev_stats[l]["trades"] += 1
        if t.pnl > 0:
            lev_stats[l]["wins"] += 1
        lev_stats[l]["pnl"] += t.pnl

    for l in lev_stats:
        s = lev_stats[l]
        s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0
        s["pnl"] = round(s["pnl"], 2)

    # Top / bottom coins
    coin_pnl = {}
    for sym, trades in coin_results.items():
        coin_pnl[sym] = {
            "trades": len(trades),
            "pnl": round(sum(t.pnl for t in trades), 2),
            "win_rate": round(sum(1 for t in trades if t.pnl > 0) / len(trades) * 100, 1) if trades else 0,
        }

    top_coins = sorted(coin_pnl.items(), key=lambda x: x[1]["pnl"], reverse=True)[:10]
    bottom_coins = sorted(coin_pnl.items(), key=lambda x: x[1]["pnl"])[:10]

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "initial_balance": BT_INITIAL_BALANCE,
            "capital_per_trade": BT_CAPITAL_PER_TRADE,
            "primary_tf": BT_TIMEFRAME,
            "macro_tf": BT_MACRO_TF,
            "lookback_candles": BT_LOOKBACK,
            "train_period": BT_TRAIN_PERIOD,
            "atr_sl_mult": config.ATR_SL_MULTIPLIER,
            "atr_tp_mult": config.ATR_TP_MULTIPLIER,
            "leverage_tiers": {
                "high": config.LEVERAGE_HIGH,
                "moderate": config.LEVERAGE_MODERATE,
                "low": config.LEVERAGE_LOW,
            },
        },
        "summary": {
            "coins_tested": len(coin_results),
            "coins_failed": len(failed),
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "total_commission": round(total_commission, 2),
            "commission_rate": f"{config.TAKER_FEE * 100:.2f}% taker (both legs)",
            "total_return_pct": round(total_pnl / BT_INITIAL_BALANCE * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "final_equity": round(equity, 2),
            "best_trade": round(max(t.pnl for t in all_trades), 2),
            "worst_trade": round(min(t.pnl for t in all_trades), 2),
        },
        "by_regime": regime_stats,
        "by_exit_reason": exit_reasons,
        "by_leverage": {str(k): v for k, v in lev_stats.items()},
        "top_10_coins": {k: v for k, v in top_coins},
        "bottom_10_coins": {k: v for k, v in bottom_coins},
        "failed_coins": failed,
    }


def print_report(report):
    """Print formatted backtest report."""
    s = report["summary"]
    c = report["config"]

    print("\n" + "=" * 70)
    print("  📊 SENTINEL BACKTEST REPORT")
    print("=" * 70)

    print(f"\n  Config:")
    print(f"    Initial Balance:    ${c['initial_balance']:,.0f}")
    print(f"    Capital/Trade:      ${c['capital_per_trade']:,.0f}")
    print(f"    Timeframes:         {c['primary_tf']} (primary) + {c['macro_tf']} (macro)")
    print(f"    SL/TP ATR:          {c['atr_sl_mult']}x / {c['atr_tp_mult']}x")
    print(f"    Leverage:           {c['leverage_tiers']}")

    print(f"\n  {'─' * 50}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"  {'─' * 50}")
    print(f"    Coins Tested:       {s['coins_tested']}")
    print(f"    Total Trades:       {s['total_trades']}")
    print(f"    Win Rate:           {s['win_rate_pct']}%")
    print(f"    Total P&L:          ${s['total_pnl']:+,.2f}")
    print(f"    Total Return:       {s['total_return_pct']:+.2f}%")
    print(f"    Final Equity:       ${s['final_equity']:,.2f}")
    print(f"    Avg Win:            ${s['avg_win']:+.2f}")
    print(f"    Avg Loss:           ${s['avg_loss']:.2f}")
    print(f"    Profit Factor:      {s['profit_factor']}")
    print(f"    Max Drawdown:       ${s['max_drawdown']:,.2f} ({s['max_drawdown_pct']:.1f}%)")
    print(f"    Best Trade:         ${s['best_trade']:+.2f}")
    print(f"    Worst Trade:        ${s['worst_trade']:.2f}")

    print(f"\n  {'─' * 50}")
    print(f"  BY REGIME")
    print(f"  {'─' * 50}")
    for regime, stats in report["by_regime"].items():
        print(f"    {regime:20s} | {stats['trades']:3d} trades | WR: {stats['win_rate']:5.1f}% | P&L: ${stats['pnl']:+8.2f}")

    print(f"\n  {'─' * 50}")
    print(f"  BY EXIT REASON")
    print(f"  {'─' * 50}")
    for reason, stats in report["by_exit_reason"].items():
        print(f"    {reason:20s} | {stats['count']:3d} trades | P&L: ${stats['pnl']:+8.2f}")

    print(f"\n  {'─' * 50}")
    print(f"  BY LEVERAGE")
    print(f"  {'─' * 50}")
    for lev, stats in report["by_leverage"].items():
        print(f"    {lev:>3s}x               | {stats['trades']:3d} trades | WR: {stats['win_rate']:5.1f}% | P&L: ${stats['pnl']:+8.2f}")

    print(f"\n  {'─' * 50}")
    print(f"  TOP 10 COINS (by P&L)")
    print(f"  {'─' * 50}")
    for sym, stats in report["top_10_coins"].items():
        print(f"    {sym:15s} | {stats['trades']:3d} trades | WR: {stats['win_rate']:5.1f}% | P&L: ${stats['pnl']:+8.2f}")

    print(f"\n  {'─' * 50}")
    print(f"  BOTTOM 10 COINS (by P&L)")
    print(f"  {'─' * 50}")
    for sym, stats in report["bottom_10_coins"].items():
        print(f"    {sym:15s} | {stats['trades']:3d} trades | WR: {stats['win_rate']:5.1f}% | P&L: ${stats['pnl']:+8.2f}")

    print("\n" + "=" * 70)
    print(f"  Report saved to: data/backtest_report.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    server_mode = "--server" in sys.argv
    coin_limit = 50

    if server_mode:
        # Find param file path (argument after --server)
        idx = sys.argv.index("--server")
        if idx + 1 < len(sys.argv):
            param_path = sys.argv[idx + 1]
            coin_limit = load_server_params(param_path)
        logging.basicConfig(level=logging.WARNING)

    run_backtest(server_mode=server_mode, coin_limit=coin_limit)

