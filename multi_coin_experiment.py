"""
Multi-Coin Walk-Forward Experiment
===================================
Runs the enhanced HMM strategy (8 features, S/R, VWAP, OI, funding)
across top 25 and top 50 coins by volume, excluding meme coins.

Tests whether the model generalizes beyond BTC.
"""
import sys
import os
import time
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_pipeline import fetch_futures_klines, _get_binance_client

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("MultiCoinExperiment")

# ═══════════════════════════════════════════════════════════════════════════════
#  MEME COIN EXCLUSION LIST
# ═══════════════════════════════════════════════════════════════════════════════

MEME_COINS = {
    # Major meme coins
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT",
    "WIFUSDT", "BOMEUSDT", "MEMEUSDT", "PEOPLEUSDT", "ELONUSDT",
    "BABYDOGEUSDT", "NEIROUSDT", "TURBO", "TURBOUSDT",
    "MOGUSDT", "COQUSDT", "MYRO", "MYROUSDT",
    "1000PEPEUSDT", "1000SHIBUSDT", "1000FLOKIUSDT", "1000BONKUSDT",
    "1000LUNCUSDT", "LUNCUSDT", "LUNAUSDT",
    # Leveraged / special tokens
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "DAIUSDT",
    "USTUSDT", "EURUSDT",
}

EXCLUDE_KEYWORDS = ("UP", "DOWN", "BULL", "BEAR", "1000")


def get_top_futures_coins(limit=50):
    """Fetch top futures coins by volume, excluding memes and leveraged tokens."""
    client = _get_binance_client()
    
    try:
        tickers = client.futures_ticker()
    except Exception as e:
        logger.error("Failed to fetch futures tickers: %s", e)
        return ["BTCUSDT"]
    
    # Filter: USDT pairs, exclude leveraged tokens and meme coins
    valid = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        base = sym.replace("USDT", "")
        if any(kw in base for kw in EXCLUDE_KEYWORDS):
            continue
        if sym in MEME_COINS:
            continue
        vol = float(t.get("quoteVolume", 0))
        valid.append((sym, vol))
    
    valid.sort(key=lambda x: x[1], reverse=True)
    top = [sym for sym, _ in valid[:limit]]
    return top


def run_coin_backtest(symbol, verbose=False):
    """Run walk-forward backtest on a single coin. Returns results dict or None."""
    from backtester_v2 import walk_forward_backtest
    
    try:
        df = fetch_futures_klines(symbol, "1h", limit=1000)
        if df is None or len(df) < 500:
            return None
        
        results = walk_forward_backtest(
            df, train_window=400, test_window=100, step=100,
            initial_capital=10000, verbose=verbose
        )
        
        if results is None:
            return None
        
        total_trades = results.get("total_trades", 0)
        if total_trades < 3:
            return None
        
        # Calculate buy & hold
        first_close = df["close"].iloc[0]
        last_close = df["close"].iloc[-1]
        buy_hold = (last_close - first_close) / first_close * 100
        
        total_return = results.get("total_return", 0)
        
        return {
            "symbol": symbol,
            "return_pct": total_return,
            "max_dd": results.get("max_drawdown", 0),
            "sharpe": results.get("sharpe_ratio", 0),
            "win_rate": results.get("win_rate", 0) / 100,  # Convert from % to ratio
            "total_trades": total_trades,
            "profit_factor": results.get("profit_factor", 0),
            "buy_hold": buy_hold,
            "alpha": total_return - buy_hold,
        }
    except Exception as e:
        if verbose:
            logger.warning("Error backtesting %s: %s", symbol, e)
        return None


def run_experiment(coin_limit, label):
    """Run backtest across top N coins and print results."""
    print(f"\n{'='*100}")
    print(f"  🔬 MULTI-COIN EXPERIMENT: Top {coin_limit} Coins (excl. memes)")
    print(f"  Model: 8-feature HMM + S/R + VWAP + OI + Funding + Breakeven + Counter-trend")
    print(f"{'='*100}")
    
    # Fetch top coins
    print(f"\n  📡 Fetching top {coin_limit} futures coins by volume...")
    coins = get_top_futures_coins(limit=coin_limit)
    print(f"  ✅ Got {len(coins)} coins: {', '.join(coins[:10])}{'...' if len(coins) > 10 else ''}")
    
    # Run backtests
    results = []
    for i, symbol in enumerate(coins):
        sys.stdout.write(f"\r  ⏳ Testing {symbol:12s} ({i+1}/{len(coins)})...")
        sys.stdout.flush()
        
        result = run_coin_backtest(symbol)
        if result:
            results.append(result)
        
        # Rate limit
        if (i + 1) % 5 == 0:
            time.sleep(1)
    
    print(f"\r  ✅ Completed {len(results)}/{len(coins)} coins successfully" + " " * 30)
    
    if not results:
        print("  ❌ No valid results!")
        return results
    
    # Sort by alpha (outperformance vs buy & hold)
    results.sort(key=lambda r: r["alpha"], reverse=True)
    
    # Print results table
    print(f"\n  {'Symbol':<12} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} {'WR':>6} {'Trades':>7} {'B&H':>8} {'Alpha':>8}  Rating")
    print(f"  {'─'*85}")
    
    for r in results:
        # Rating
        if r["return_pct"] > 0:
            rating = "🟢"
        elif r["alpha"] > 0:
            rating = "🟡"
        else:
            rating = "🔴"
        
        print(f"  {r['symbol']:<12} {r['return_pct']:>+7.1f}% {r['max_dd']:>-7.1f}% {r['sharpe']:>+7.3f} {r['win_rate']*100:>5.0f}% {r['total_trades']:>6d}  {r['buy_hold']:>+7.1f}% {r['alpha']:>+7.1f}%  {rating}")
    
    # Summary statistics
    print(f"\n  {'─'*85}")
    avg_return = np.mean([r["return_pct"] for r in results])
    avg_alpha = np.mean([r["alpha"] for r in results])
    avg_wr = np.mean([r["win_rate"] for r in results])
    avg_dd = np.mean([r["max_dd"] for r in results])
    profitable = sum(1 for r in results if r["return_pct"] > 0)
    beat_bh = sum(1 for r in results if r["alpha"] > 0)
    
    print(f"  📊 SUMMARY ({label}):")
    print(f"     Coins tested:     {len(results)}")
    print(f"     Avg return:       {avg_return:+.1f}%")
    print(f"     Avg alpha:        {avg_alpha:+.1f}% (vs buy & hold)")
    print(f"     Avg win rate:     {avg_wr*100:.0f}%")
    print(f"     Avg max DD:       {avg_dd:.1f}%")
    print(f"     Profitable:       {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)")
    print(f"     Beat buy & hold:  {beat_bh}/{len(results)} ({beat_bh/len(results)*100:.0f}%)")
    
    # Top 5 & Bottom 5
    top5 = ', '.join(f"{r['symbol']}({r['alpha']:+.1f}%)" for r in results[:5])
    bot5 = ', '.join(f"{r['symbol']}({r['alpha']:+.1f}%)" for r in results[-5:])
    print(f"\n     🏆 Top 5:  {top5}")
    print(f"     💀 Bot 5:  {bot5}")
    
    print(f"  {'='*85}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 100)
    print("  REGIME-MASTER: Multi-Coin Walk-Forward Experiment")
    print("  Testing enhanced model across top futures coins (memes excluded)")
    print("█" * 100)
    
    # Run Top 25
    results_25 = run_experiment(25, "Top 25")
    
    # Run Top 50 (includes the top 25 + next 25)
    results_50 = run_experiment(50, "Top 50")
    
    # Combined summary
    if results_25 and results_50:
        print(f"\n\n{'='*100}")
        print("  📊 COMBINED COMPARISON")
        print(f"{'='*100}")
        
        avg_alpha_25 = np.mean([r["alpha"] for r in results_25])
        avg_alpha_50 = np.mean([r["alpha"] for r in results_50])
        beat_25 = sum(1 for r in results_25 if r["alpha"] > 0)
        beat_50 = sum(1 for r in results_50 if r["alpha"] > 0)
        
        print(f"  {'Metric':<25} {'Top 25':>12} {'Top 50':>12}")
        print(f"  {'─'*50}")
        print(f"  {'Coins tested':<25} {len(results_25):>12} {len(results_50):>12}")
        print(f"  {'Avg alpha':<25} {avg_alpha_25:>+11.1f}% {avg_alpha_50:>+11.1f}%")
        print(f"  {'Beat buy & hold':<25} {beat_25:>12} {beat_50:>12}")
        print(f"  {'Avg return':<25} {np.mean([r['return_pct'] for r in results_25]):>+11.1f}% {np.mean([r['return_pct'] for r in results_50]):>+11.1f}%")
        print(f"  {'Avg win rate':<25} {np.mean([r['win_rate'] for r in results_25])*100:>11.0f}% {np.mean([r['win_rate'] for r in results_50])*100:>11.0f}%")
        print(f"  {'='*50}")
    
    print("\n✅ Experiment complete!")
