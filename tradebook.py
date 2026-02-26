"""
Project Regime-Master — Tradebook
Comprehensive trade journal tracking every entry, exit, and P&L metric.
Persists to JSON for the dashboard. Supports live unrealized P&L updates.
"""
import json
import os
import logging
from datetime import datetime
from data_pipeline import get_current_price
import config
import telegram as tg

logger = logging.getLogger("Tradebook")

TRADEBOOK_FILE = os.path.join(config.DATA_DIR, "tradebook.json")


def _load_book():
    """Load tradebook from disk."""
    if not os.path.exists(TRADEBOOK_FILE):
        return {"trades": [], "summary": {}}
    try:
        with open(TRADEBOOK_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"trades": [], "summary": {}}


def _save_book(book):
    """Save tradebook to disk."""
    try:
        with open(TRADEBOOK_FILE, "w") as f:
            json.dump(book, f, indent=2)
    except Exception as e:
        logger.error("Failed to save tradebook: %s", e)


def _next_id(book):
    """Generate next trade ID."""
    if not book["trades"]:
        return "T-0001"
    last = book["trades"][-1]["trade_id"]
    num = int(last.split("-")[1]) + 1
    return f"T-{num:04d}"


def _compute_summary(book):
    """Compute aggregate portfolio stats."""
    trades = book["trades"]
    total = len(trades)
    active = [t for t in trades if t["status"] == "ACTIVE"]
    closed = [t for t in trades if t["status"] == "CLOSED"]
    wins = [t for t in closed if t.get("realized_pnl", 0) > 0]
    losses = [t for t in closed if t.get("realized_pnl", 0) < 0]

    total_realized = sum(t.get("realized_pnl", 0) for t in closed)
    total_unrealized = sum(t.get("unrealized_pnl", 0) for t in active)
    max_capital = config.PAPER_MAX_CAPITAL if hasattr(config, 'PAPER_MAX_CAPITAL') else 2500
    deployed_capital = len(active) * 100  # $100 per active trade

    book["summary"] = {
        "total_trades": total,
        "active_trades": len(active),
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_realized_pnl": round(total_realized, 4),
        "total_realized_pnl_pct": round(total_realized / max_capital * 100, 2) if max_capital else 0,
        "total_unrealized_pnl": round(total_unrealized, 4),
        "total_unrealized_pnl_pct": round(total_unrealized / deployed_capital * 100, 2) if deployed_capital else 0,
        "cumulative_pnl": round(total_realized + total_unrealized, 4),
        "cumulative_pnl_pct": round((total_realized + total_unrealized) / max_capital * 100, 2) if max_capital else 0,
        "best_trade": round(max((t.get("realized_pnl", 0) for t in closed), default=0), 4),
        "worst_trade": round(min((t.get("realized_pnl", 0) for t in closed), default=0), 4),
        "avg_leverage": round(sum(t.get("leverage", 1) for t in trades) / total, 1) if total else 0,
        "last_updated": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def open_trade(symbol, side, leverage, quantity, entry_price, atr,
               regime, confidence, reason="", capital=100.0, mode=None):
    """
    Record a new trade entry in the tradebook.

    Parameters
    ----------
    symbol      : str   — e.g. 'BTCUSDT'
    side        : str   — 'BUY' or 'SELL' (mapped to LONG/SHORT)
    leverage    : int
    quantity    : float
    entry_price : float
    atr         : float — ATR at entry (for SL/TP reference)
    regime      : str   — regime name
    confidence  : float — HMM confidence
    reason      : str
    capital     : float — capital allocated ($100 default)

    Returns
    -------
    str : trade_id
    """
    book = _load_book()

    # Guard: prevent duplicate ACTIVE trades for the same symbol
    existing = [t for t in book["trades"] if t["symbol"] == symbol and t["status"] == "ACTIVE"]
    if existing:
        logger.warning("⚠️ Skipping duplicate trade for %s — already have ACTIVE trade %s",
                       symbol, existing[0]["trade_id"])
        return existing[0]["trade_id"]

    trade_id = _next_id(book)
    position = "LONG" if side == "BUY" else "SHORT"

    # Compute SL/TP based on ATR (adjusted for leverage)
    sl_mult, tp_mult = config.get_atr_multipliers(leverage)
    if position == "LONG":
        stop_loss = round(entry_price - atr * sl_mult, 6)
        take_profit = round(entry_price + atr * tp_mult, 6)
    else:
        stop_loss = round(entry_price + atr * sl_mult, 6)
        take_profit = round(entry_price - atr * tp_mult, 6)

    trade = {
        "trade_id":         trade_id,
        "entry_timestamp":  datetime.utcnow().isoformat(),
        "exit_timestamp":   None,
        "symbol":           symbol,
        "position":         position,
        "side":             side,
        "regime":           regime,
        "confidence":       round(confidence, 4) if confidence else 0,
        "leverage":         leverage,
        "capital":          capital,
        "quantity":         round(quantity, 6),
        "entry_price":      round(entry_price, 6),
        "exit_price":       None,
        "current_price":    round(entry_price, 6),
        "stop_loss":        stop_loss,
        "take_profit":      take_profit,
        "atr_at_entry":     round(atr, 6),
        "trailing_sl":      stop_loss,
        "trailing_tp":      take_profit,
        "peak_price":       round(entry_price, 6),
        "trailing_active":  False,
        "tp_extensions":    0,
        "status":           "ACTIVE",
        "exit_reason":      None,
        "realized_pnl":     0,
        "realized_pnl_pct": 0,
        "unrealized_pnl":   0,
        "unrealized_pnl_pct": 0,
        "max_favorable":    0,
        "max_adverse":      0,
        "duration_minutes":  0,
        "mode":             mode if mode else ("PAPER" if config.PAPER_TRADE else "LIVE"),
        "commission":       0,
    }

    book["trades"].append(trade)
    _compute_summary(book)
    _save_book(book)

    logger.info("📗 Tradebook OPEN: %s %s %s @ %.6f | %dx | Capital: $%.0f",
                trade_id, position, symbol, entry_price, leverage, capital)

    return trade_id


def close_trade(trade_id=None, symbol=None, exit_price=None, reason="MANUAL"):
    """
    Close a trade by ID, or ALL active trades for a symbol.

    Parameters
    ----------
    trade_id   : str (optional) — close specific trade
    symbol     : str (optional) — close ALL active trades for this symbol
    exit_price : float (if None, fetches current price)
    reason     : str — why the trade was closed

    Returns
    -------
    dict or list : closed trade record(s)
    """
    book = _load_book()

    # Find target trade(s)
    targets = []
    for trade in book["trades"]:
        if trade["status"] != "ACTIVE":
            continue
        if trade_id and trade["trade_id"] == trade_id:
            targets = [trade]
            break
        if symbol and trade["symbol"] == symbol:
            targets.append(trade)

    if not targets:
        logger.warning("No active trade found for id=%s symbol=%s", trade_id, symbol)
        return None

    closed = []
    for target in targets:
        # Get exit price
        px = exit_price
        if px is None:
            px = get_current_price(target["symbol"]) or target["entry_price"]
        px = round(px, 6)

        # Calculate P&L
        entry = target["entry_price"]
        qty = target["quantity"]
        lev = target["leverage"]
        capital = target["capital"]

        if target["position"] == "LONG":
            raw_pnl = (px - entry) * qty
        else:
            raw_pnl = (entry - px) * qty

        # Commission: taker fee on both entry and exit notional
        entry_notional = entry * qty
        exit_notional = px * qty
        commission = round((entry_notional + exit_notional) * config.TAKER_FEE, 4)

        leveraged_pnl = round(raw_pnl * lev - commission, 4)
        pnl_pct = round(leveraged_pnl / capital * 100, 2) if capital else 0

        # Duration
        entry_time = datetime.fromisoformat(target["entry_timestamp"])
        duration = (datetime.utcnow() - entry_time).total_seconds() / 60

        target["exit_timestamp"] = datetime.utcnow().isoformat()
        target["exit_price"] = px
        target["current_price"] = px
        target["status"] = "CLOSED"
        target["exit_reason"] = reason
        target["commission"] = commission
        target["realized_pnl"] = leveraged_pnl
        target["realized_pnl_pct"] = pnl_pct
        target["unrealized_pnl"] = 0
        target["unrealized_pnl_pct"] = 0
        target["duration_minutes"] = round(duration, 1)

        logger.info("📕 Tradebook CLOSE: %s %s %s @ %.6f → %.6f | P&L: $%.4f (%.2f%%)",
                    target["trade_id"], target["position"], target["symbol"],
                    entry, px, leveraged_pnl, pnl_pct)
        closed.append(target)

    _compute_summary(book)
    _save_book(book)

    return closed[0] if len(closed) == 1 else closed


def _close_trade_inline(trade, exit_price, reason):
    """
    Close a trade INLINE (mutates the trade dict directly).
    Used by update_unrealized() to avoid the load/save race condition.
    """
    px = round(exit_price, 6)
    entry = trade["entry_price"]
    qty = trade["quantity"]
    lev = trade["leverage"]
    capital = trade["capital"]

    if trade["position"] == "LONG":
        raw_pnl = (px - entry) * qty
    else:
        raw_pnl = (entry - px) * qty

    entry_notional = entry * qty
    exit_notional = px * qty
    commission = round((entry_notional + exit_notional) * config.TAKER_FEE, 4)

    leveraged_pnl = round(raw_pnl * lev - commission, 4)
    pnl_pct = round(leveraged_pnl / capital * 100, 2) if capital else 0

    entry_time = datetime.fromisoformat(trade["entry_timestamp"])
    duration = (datetime.utcnow() - entry_time).total_seconds() / 60

    trade["exit_timestamp"] = datetime.utcnow().isoformat()
    trade["exit_price"] = px
    trade["current_price"] = px
    trade["status"] = "CLOSED"
    trade["exit_reason"] = reason
    trade["commission"] = commission
    trade["realized_pnl"] = leveraged_pnl
    trade["realized_pnl_pct"] = pnl_pct
    trade["unrealized_pnl"] = 0
    trade["unrealized_pnl_pct"] = 0
    trade["duration_minutes"] = round(duration, 1)

    logger.info("📕 Tradebook CLOSE: %s %s %s @ %.6f → %.6f | P&L: $%.4f (%.2f%%) [%s]",
                trade["trade_id"], trade["position"], trade["symbol"],
                entry, px, leveraged_pnl, pnl_pct, reason)

    # Telegram notification
    try:
        tg.notify_trade_close(trade)
        if reason == "MAX_LOSS":
            tg.notify_max_loss(trade["symbol"], pnl_pct, trade["trade_id"])
    except Exception:
        pass


def update_unrealized(prices=None):
    """
    Update unrealized P&L for all active trades using live prices.
    Auto-closes trades that hit MAX_LOSS, SL, or TP thresholds.

    IMPORTANT: All closes happen INLINE on the same book object to avoid
    the race condition where close_trade() would save independently and
    then this function would overwrite with a stale copy.

    Parameters
    ----------
    prices : dict (optional) — {symbol: price}. If None, fetches live.
    """
    book = _load_book()
    changed = False

    for trade in book["trades"]:
        if trade["status"] != "ACTIVE":
            continue

        symbol = trade["symbol"]
        if prices and symbol in prices:
            current = prices[symbol]
        else:
            current = get_current_price(symbol)
            if not current:
                continue

        current = round(current, 6)
        entry = trade["entry_price"]
        qty = trade["quantity"]
        lev = trade["leverage"]
        capital = trade["capital"]

        if trade["position"] == "LONG":
            raw_pnl = (current - entry) * qty
        else:
            raw_pnl = (entry - current) * qty

        # For LIVE trades: qty from CoinDCX IS the leveraged quantity,
        # so raw_pnl is already the full P&L — do NOT multiply by leverage.
        # Also skip commission estimation — CoinDCX handles actual fees.
        # For PAPER trades: qty is the base position, multiply by leverage.
        is_live = trade.get("mode") == "LIVE"
        if is_live:
            est_commission = 0
            leveraged_pnl = round(raw_pnl, 4)
        else:
            entry_notional = entry * qty
            exit_notional = current * qty
            est_commission = (entry_notional + exit_notional) * config.TAKER_FEE
            leveraged_pnl = round(raw_pnl * lev - est_commission, 4)
        pnl_pct = round(leveraged_pnl / capital * 100, 2) if capital else 0

        # Track max favorable / adverse excursion
        if leveraged_pnl > trade.get("max_favorable", 0):
            trade["max_favorable"] = leveraged_pnl
        if leveraged_pnl < trade.get("max_adverse", 0):
            trade["max_adverse"] = leveraged_pnl

        # Duration
        entry_time = datetime.fromisoformat(trade["entry_timestamp"])
        duration = (datetime.utcnow() - entry_time).total_seconds() / 60

        trade["current_price"] = current
        trade["unrealized_pnl"] = leveraged_pnl
        trade["unrealized_pnl_pct"] = pnl_pct
        trade["duration_minutes"] = round(duration, 1)

        # ── Trailing SL / TP Logic ────────────────────────────────
        atr = trade.get("atr_at_entry", 0)
        is_long = trade["position"] == "LONG"

        # Initialize trailing fields for legacy trades that lack them
        if "trailing_sl" not in trade:
            trade["trailing_sl"] = trade["stop_loss"]
        if "trailing_tp" not in trade:
            trade["trailing_tp"] = trade["take_profit"]
        if "peak_price" not in trade:
            trade["peak_price"] = entry
        if "trailing_active" not in trade:
            trade["trailing_active"] = False
        if "tp_extensions" not in trade:
            trade["tp_extensions"] = 0

        # --- Trailing Stop Loss ---
        if config.TRAILING_SL_ENABLED and atr > 0:
            # Update peak price (high-water mark for LONG, low-water mark for SHORT)
            if is_long:
                if current > trade["peak_price"]:
                    trade["peak_price"] = current
            else:
                if current < trade["peak_price"]:
                    trade["peak_price"] = current

            # Check activation: price moved enough in our favor
            activation_dist = atr * config.TRAILING_SL_ACTIVATION_ATR
            if is_long:
                favorable_move = current - entry
            else:
                favorable_move = entry - current

            if favorable_move >= activation_dist:
                trade["trailing_active"] = True

            # Trail the SL (only tightens, never loosens)
            if trade["trailing_active"]:
                trail_dist = atr * config.TRAILING_SL_DISTANCE_ATR
                if is_long:
                    new_sl = round(trade["peak_price"] - trail_dist, 6)
                    if new_sl > trade["trailing_sl"]:
                        trade["trailing_sl"] = new_sl
                else:
                    new_sl = round(trade["peak_price"] + trail_dist, 6)
                    if new_sl < trade["trailing_sl"]:
                        trade["trailing_sl"] = new_sl

        # --- Trailing Take Profit ---
        if config.TRAILING_TP_ENABLED and atr > 0:
            max_ext = config.TRAILING_TP_MAX_EXTENSIONS
            if trade["tp_extensions"] < max_ext:
                # Distance from entry to current TP
                if is_long:
                    tp_dist = trade["trailing_tp"] - entry
                    progress = (current - entry) / tp_dist if tp_dist > 0 else 0
                else:
                    tp_dist = entry - trade["trailing_tp"]
                    progress = (entry - current) / tp_dist if tp_dist > 0 else 0

                if progress >= config.TRAILING_TP_ACTIVATION_PCT:
                    ext_amount = atr * config.TRAILING_TP_EXTENSION_ATR
                    if is_long:
                        trade["trailing_tp"] = round(trade["trailing_tp"] + ext_amount, 6)
                    else:
                        trade["trailing_tp"] = round(trade["trailing_tp"] - ext_amount, 6)
                    trade["tp_extensions"] += 1
                    logger.info(
                        "📈 Trailing TP extended for %s: new TP=%.6f (ext #%d)",
                        trade["trade_id"], trade["trailing_tp"], trade["tp_extensions"],
                    )

        # ── EXIT CHECKS ──────────────────────────────────────────────
        # For LIVE trades, CoinDCX handles SL/TP/MAX_LOSS via exchange
        # orders. The heartbeat _sync_coindcx_positions() detects when
        # exchange closes a position. We ONLY auto-close in tradebook
        # for paper trades.
        is_live = trade.get("mode") == "LIVE"

        if not is_live:
            # HARD MAX LOSS GUARD — paper trades only
            if pnl_pct <= config.MAX_LOSS_PER_TRADE_PCT:
                logger.warning(
                    "🛑 MAX LOSS hit on %s (%.2f%% <= %.0f%%) — auto-closing trade %s",
                    symbol, pnl_pct, config.MAX_LOSS_PER_TRADE_PCT, trade["trade_id"],
                )
                _close_trade_inline(trade, current, "MAX_LOSS")
                changed = True
                continue

            # Use trailing values for SL/TP hit checks
            effective_sl = trade.get("trailing_sl", trade["stop_loss"])
            effective_tp = trade.get("trailing_tp", trade["take_profit"])

            sl_hit = False
            tp_hit = False
            if is_long:
                sl_hit = current <= effective_sl
                tp_hit = current >= effective_tp
            else:
                sl_hit = current >= effective_sl
                tp_hit = current <= effective_tp

            if sl_hit:
                reason = "TRAILING_SL" if trade["trailing_active"] else "STOP_LOSS"
                _close_trade_inline(trade, effective_sl, reason)
                changed = True
                continue
            if tp_hit:
                reason = "TRAILING_TP" if trade["tp_extensions"] > 0 else "TAKE_PROFIT"
                _close_trade_inline(trade, effective_tp, reason)
                changed = True
                continue

        changed = True

    if changed:
        _compute_summary(book)
        _save_book(book)


def get_tradebook():
    """Return the full tradebook dict."""
    return _load_book()


def get_active_trades():
    """Return only active trades."""
    book = _load_book()
    return [t for t in book["trades"] if t["status"] == "ACTIVE"]


def get_closed_trades():
    """Return only closed trades."""
    book = _load_book()
    return [t for t in book["trades"] if t["status"] == "CLOSED"]


def get_current_loss_streak():
    """Return (streak_count, last_loss_timestamp) for the current consecutive losing streak.
    Counts backwards from the most recent closed trade.
    """
    closed = get_closed_trades()
    if not closed:
        return 0, None

    # Sort by exit timestamp descending (most recent first)
    closed.sort(key=lambda t: t.get("exit_timestamp", ""), reverse=True)

    streak = 0
    last_loss_ts = None
    for t in closed:
        pnl = t.get("realized_pnl", 0)
        if pnl < 0:
            streak += 1
            if last_loss_ts is None:
                last_loss_ts = t.get("exit_timestamp")
        else:
            break  # Streak broken by a win
    return streak, last_loss_ts


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE TRAILING SL/TP SYNC
# ═══════════════════════════════════════════════════════════════════════════════

def _close_live_position(symbol):
    """Close a live CoinDCX position when SL/TP is hit."""
    try:
        import coindcx_client as cdx
        pair = cdx.to_coindcx_pair(symbol)
        positions = cdx.list_positions()
        for p in positions:
            if p.get("pair") == pair and float(p.get("active_pos", 0)) != 0:
                cdx.exit_position(p["id"])
                logger.info("📤 Closed CoinDCX position %s for %s", p["id"], symbol)
                return True
        logger.warning("No CoinDCX position found for %s to close", symbol)
    except Exception as e:
        logger.error("Failed to close CoinDCX position for %s: %s", symbol, e)
    return False


def _price_round(p):
    """Round price to CoinDCX-compatible tick size."""
    if p >= 1000:   return round(p, 1)
    elif p >= 10:   return round(p, 2)
    elif p >= 1:    return round(p, 3)
    elif p >= 0.01: return round(p, 4)
    else:           return round(p, 5)


def sync_live_tpsl():
    """
    Push updated trailing SL/TP to CoinDCX for live positions.

    Called from the heartbeat loop (main.py) AFTER update_unrealized().
    Only runs in LIVE mode. Compares current trailing_sl/trailing_tp
    with the last values pushed to CoinDCX and updates if changed.
    """
    if config.PAPER_TRADE:
        return

    try:
        import coindcx_client as cdx
    except ImportError:
        return

    book = _load_book()
    updated_count = 0

    for trade in book["trades"]:
        if trade["status"] != "ACTIVE":
            continue
        if trade.get("mode") != "LIVE":
            continue

        symbol = trade["symbol"]
        trailing_sl = trade.get("trailing_sl", trade["stop_loss"])
        trailing_tp = trade.get("trailing_tp", trade["take_profit"])

        # Compare with last-pushed values
        last_sl = trade.get("_cdx_last_sl", trade["stop_loss"])
        last_tp = trade.get("_cdx_last_tp", trade["take_profit"])

        sl_changed = abs(trailing_sl - last_sl) > 1e-8
        tp_changed = abs(trailing_tp - last_tp) > 1e-8

        if not sl_changed and not tp_changed:
            continue

        # Find CoinDCX position ID
        pair = cdx.to_coindcx_pair(symbol)
        try:
            positions = cdx.list_positions()
            pos_id = None
            for p in positions:
                if p.get("pair") == pair and float(p.get("active_pos", 0)) != 0:
                    pos_id = p["id"]
                    break

            if not pos_id:
                logger.debug("No CoinDCX position for %s — skip TPSL sync", symbol)
                continue

            # Round to CoinDCX tick sizes
            rounded_sl = _price_round(trailing_sl)
            rounded_tp = _price_round(trailing_tp)

            cdx.create_tpsl(
                position_id=pos_id,
                take_profit_price=rounded_tp,
                stop_loss_price=rounded_sl,
            )

            # Record pushed values
            trade["_cdx_last_sl"] = trailing_sl
            trade["_cdx_last_tp"] = trailing_tp
            updated_count += 1

            logger.info(
                "🔄 TPSL updated on CoinDCX for %s: SL=$%.6f → $%.6f | TP=$%.6f → $%.6f",
                symbol, last_sl, rounded_sl, last_tp, rounded_tp,
            )

        except Exception as e:
            logger.error("Failed to sync TPSL for %s: %s", symbol, e)

    if updated_count > 0:
        _save_book(book)
        logger.info("📊 Synced trailing SL/TP for %d live positions", updated_count)
