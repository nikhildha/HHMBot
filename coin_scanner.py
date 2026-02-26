"""
Project Regime-Master — Coin Scanner
Fetches top coins by 24h trading volume and runs HMM regime analysis.
  • Paper mode → Binance tickers
  • Live mode  → CoinDCX Futures instruments
"""
import json
import logging
import time
from datetime import datetime

import pandas as pd
import numpy as np

import config
from data_pipeline import fetch_klines, _get_binance_client
from feature_engine import compute_hmm_features, compute_all_features
from hmm_brain import HMMBrain

logger = logging.getLogger("CoinScanner")

# ─── Path for multi-coin state ──────────────────────────────────────────────────
SCANNER_STATE_FILE = __import__("os").path.join(config.DATA_DIR, "scanner_state.json")


def _get_top_coins_binance(limit=50, quote="USDT"):
    """Fetch top coins from Binance by 24h quote volume (paper mode)."""
    client = _get_binance_client()
    try:
        tickers = client.get_ticker()
    except Exception as e:
        logger.error("Failed to fetch Binance tickers: %s", e)
        return [config.PRIMARY_SYMBOL]

    exclude_keywords = ("UP", "DOWN", "BULL", "BEAR")
    usdt_tickers = [
        t for t in tickers
        if t["symbol"].endswith(quote)
        and not any(kw in t["symbol"].replace(quote, "") for kw in exclude_keywords)
    ]
    usdt_tickers.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
    top_symbols = [t["symbol"] for t in usdt_tickers[:limit]]
    logger.info("Binance: Top %d coins by volume (%d total USDT pairs).", len(top_symbols), len(usdt_tickers))
    return top_symbols


def _get_top_coins_coindcx(limit=50):
    """
    Fetch top coins from CoinDCX Futures by 24h volume (live mode).
    Returns Binance-style symbols (BTCUSDT) for compatibility.
    """
    import coindcx_client as cdx

    instruments = cdx.get_active_instruments()
    if not instruments:
        logger.warning("No CoinDCX instruments — falling back to primary symbol.")
        return [config.PRIMARY_SYMBOL]

    # Get current prices with volume data
    prices = cdx.get_current_prices()

    # Build (instrument, volume) list and sort by 24h volume
    scored = []
    for inst in instruments:
        info = prices.get(inst, {})
        volume = float(info.get("v", 0))
        scored.append((inst, volume))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Convert to Binance-style symbols and take top N
    top_pairs = scored[:limit]
    top_symbols = [cdx.from_coindcx_pair(pair) for pair, vol in top_pairs]

    logger.info("CoinDCX: Top %d coins by volume (%d total instruments).", len(top_symbols), len(instruments))
    return top_symbols


def get_top_coins_by_volume(limit=50, quote="USDT"):
    """
    Fetch top trading pairs ranked by 24h volume.

    Routes:
      Paper mode → Binance
      Live mode  → CoinDCX Futures

    Returns
    -------
    list[str] — Binance-style symbols, e.g. ['BTCUSDT', 'ETHUSDT', ...]
    """
    if config.PAPER_TRADE:
        return _get_top_coins_binance(limit=limit, quote=quote)
    else:
        return _get_top_coins_coindcx(limit=limit)


def scan_all_regimes(symbols=None, limit=50, timeframe="1h", kline_limit=500):
    """
    Run HMM regime classification on each symbol.

    Returns
    -------
    list[dict] — one entry per symbol:
        {symbol, regime, regime_name, confidence, price, volume_24h, timestamp}
    """
    if symbols is None:
        symbols = get_top_coins_by_volume(limit=limit)

    results = []
    brain = HMMBrain()

    for i, symbol in enumerate(symbols):
        try:
            df = fetch_klines(symbol, timeframe, limit=kline_limit)
            if df is None or len(df) < 60:
                logger.debug("Skipping %s — insufficient data.", symbol)
                continue

            # Compute features & train per-coin HMM
            df_feat = compute_all_features(df)
            df_hmm = compute_hmm_features(df)

            brain_copy = HMMBrain()
            brain_copy.train(df_hmm)

            if not brain_copy.is_trained:
                continue

            state, conf = brain_copy.predict(df_feat)
            regime_name = brain_copy.get_regime_name(state)

            results.append({
                "rank":       i + 1,
                "symbol":     symbol,
                "regime":     int(state),
                "regime_name": regime_name,
                "confidence": round(conf, 4),
                "price":      round(float(df["close"].iloc[-1]), 4),
                "volume_24h": round(float(df["volume"].sum()), 2),
                "timestamp":  datetime.utcnow().isoformat(),
            })

            # Rate-limit to avoid API throttling
            if (i + 1) % 10 == 0:
                logger.info("Scanned %d/%d coins...", i + 1, len(symbols))
                time.sleep(1)

        except Exception as e:
            logger.warning("Error scanning %s: %s", symbol, e)
            continue

    # Save results for the dashboard
    _save_scanner_state(results)
    logger.info("Scan complete: %d coins classified.", len(results))
    return results


def _save_scanner_state(results):
    """Persist scanner results for the dashboard."""
    try:
        with open(SCANNER_STATE_FILE, "w") as f:
            json.dump({
                "last_scan": datetime.utcnow().isoformat(),
                "count": len(results),
                "coins": results,
            }, f, indent=2)
    except Exception as e:
        logger.error("Failed to save scanner state: %s", e)


def load_scanner_state():
    """Load the latest scanner results (used by dashboard)."""
    import os
    if not os.path.exists(SCANNER_STATE_FILE):
        return None
    try:
        with open(SCANNER_STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def print_scanner_report(results):
    """Pretty-print scanner results to console."""
    print("\n" + "=" * 90)
    print("  🔍 REGIME-MASTER: TOP COINS SCANNER")
    print("=" * 90)
    print(f"  {'#':<4} {'Symbol':<12} {'Regime':<16} {'Confidence':<12} {'Price':<14} ")
    print("-" * 90)

    for r in results:
        emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "SIDEWAYS/CHOP": "🟡", "CRASH/PANIC": "💀"}.get(r["regime_name"], "❓")
        print(f"  {r['rank']:<4} {r['symbol']:<12} {emoji} {r['regime_name']:<14} {r['confidence']*100:>6.1f}%      ${r['price']:<12,.4f}")

    # Summary
    bull = sum(1 for r in results if r["regime"] == config.REGIME_BULL)
    bear = sum(1 for r in results if r["regime"] == config.REGIME_BEAR)
    chop = sum(1 for r in results if r["regime"] == config.REGIME_CHOP)
    crash = sum(1 for r in results if r["regime"] == config.REGIME_CRASH)
    print("-" * 90)
    print(f"  Summary: 🟢 {bull} Bull | 🔴 {bear} Bear | 🟡 {chop} Chop | 💀 {crash} Crash")
    print("=" * 90 + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    print("Scanning top 50 coins by volume...")
    results = scan_all_regimes(limit=50)
    print_scanner_report(results)
