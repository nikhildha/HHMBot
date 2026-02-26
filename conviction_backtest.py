"""
Conviction Scoring Backtest
============================
Tests the new multi-factor conviction → continuous leverage (2x-35x) system
on the best-performing coins from the multi-coin experiment.

Compares: OLD system (fixed tiers) vs NEW system (conviction-based)
"""
import sys
import os
import time
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_pipeline import fetch_futures_klines, enrich_with_oi_funding
from feature_engine import compute_all_features, compute_hmm_features
from hmm_brain import HMMBrain, HMM_FEATURES
from risk_manager import RiskManager

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("ConvictionBacktest")

# Best coins from multi-coin experiment (high alpha, consistent)
STABLE_COINS = [
    "BNBUSDT",      # +103% alpha, 79% WR — large cap, stable
    "ETHUSDT",       # +26% alpha, 60% WR — must-have major
    "BTCUSDT",       # Benchmark — the hardest coin to trade
    "XRPUSDT",       # +16% alpha, 56% WR — major
    "BCHUSDT",       # +12% alpha, 52% WR — major
]

# ═══════════════════════════════════════════════════════════════════════════════
# BTC MACRO REGIME (global filter)
# ═══════════════════════════════════════════════════════════════════════════════

def get_btc_regime_series(df_btc):
    """Pre-compute BTC regime for each candle. Returns Series of regime ints."""
    brain = HMMBrain(n_states=4)
    df_feat = compute_hmm_features(df_btc)
    brain.train(df_feat)
    
    if not brain.is_trained:
        return pd.Series(config.REGIME_CHOP, index=df_btc.index)
    
    states = brain.predict_all(df_feat)
    return pd.Series(states, index=df_btc.index[:len(states)])


# ═══════════════════════════════════════════════════════════════════════════════
# CONVICTION-BASED BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_with_conviction(df_raw, btc_regimes=None, symbol="",
                              initial_capital=10000, verbose=False):
    """
    Walk-forward backtest using conviction scoring + continuous leverage.
    
    Parameters
    ----------
    df_raw : OHLCV DataFrame (with OI/funding if available)
    btc_regimes : pd.Series of BTC regime states (optional)
    
    Returns
    -------
    dict with results
    """
    df = compute_all_features(df_raw)
    n = len(df)
    
    train_window = 400
    test_window = 100
    step = 100
    
    capital = initial_capital
    equity = [capital]
    trades = []
    open_trade = None
    
    brain = HMMBrain(n_states=4)
    
    fold = 0
    start = 0
    
    while start + train_window + test_window <= n:
        fold += 1
        train_end = start + train_window
        test_end = min(train_end + test_window, n)
        
        df_train = df.iloc[start:train_end]
        df_test = df.iloc[train_end:test_end]
        
        # Train HMM
        df_hmm_train = compute_hmm_features(df_train)
        brain_copy = HMMBrain(n_states=4)
        brain_copy.train(df_hmm_train)
        
        if not brain_copy.is_trained:
            start += step
            continue
        
        # Test on OOS window
        for idx in range(len(df_test)):
            i = train_end + idx
            row = df.iloc[i]
            price = row["close"]
            atr = row.get("atr", price * 0.02)
            if atr is None or atr <= 0 or np.isnan(atr):
                atr = price * 0.02
            
            # Check open trade
            if open_trade is not None:
                pnl_pct = 0
                if open_trade["side"] == "LONG":
                    pnl_pct = (price - open_trade["entry"]) / open_trade["entry"]
                else:
                    pnl_pct = (open_trade["entry"] - price) / open_trade["entry"]
                
                pnl_pct *= open_trade["leverage"]
                
                # Check SL/TP
                hit_sl = price <= open_trade["sl"] if open_trade["side"] == "LONG" else price >= open_trade["sl"]
                hit_tp = price >= open_trade["tp"] if open_trade["side"] == "LONG" else price <= open_trade["tp"]
                
                # Breakeven stop: if PnL > +10%, move SL to entry
                if pnl_pct > 0.10 and not open_trade.get("breakeven_set"):
                    open_trade["sl"] = open_trade["entry"]
                    open_trade["breakeven_set"] = True
                
                if hit_sl or hit_tp:
                    trade_pnl = capital * 0.02 * (pnl_pct / abs(pnl_pct)) if hit_sl else capital * 0.02 * pnl_pct
                    # Simplified: use actual PnL based on position
                    risk_amount = capital * config.RISK_PER_TRADE
                    trade_result = risk_amount * pnl_pct
                    
                    # Fee deduction
                    fee = risk_amount * open_trade["leverage"] * config.TAKER_FEE * 2
                    trade_result -= fee
                    
                    capital += trade_result
                    
                    trades.append({
                        "symbol": symbol,
                        "side": open_trade["side"],
                        "leverage": open_trade["leverage"],
                        "conviction": open_trade["conviction"],
                        "entry": open_trade["entry"],
                        "exit": price,
                        "pnl": trade_result,
                        "pnl_pct": pnl_pct * 100,
                        "hit": "TP" if hit_tp else "SL",
                    })
                    
                    open_trade = None
                
                equity.append(capital)
                continue
            
            # No open trade — check for new entry
            # Get HMM prediction
            window = df.iloc[max(0, i - train_window):i + 1]
            df_hmm = compute_hmm_features(window)
            
            if len(df_hmm.dropna()) < 20:
                equity.append(capital)
                continue
            
            state, conf = brain_copy.predict(df_hmm)
            regime_name = brain_copy.get_regime_name(state)
            
            # Skip CHOP
            if state == config.REGIME_CHOP:
                equity.append(capital)
                continue
            
            # Determine side
            if regime_name == "BULLISH":
                side = "LONG"
            elif regime_name in ("BEARISH", "CRASH/PANIC"):
                side = "SHORT"
            else:
                equity.append(capital)
                continue
            
            # Get BTC regime at this timestamp
            btc_regime = None
            if btc_regimes is not None and i < len(btc_regimes):
                btc_regime = btc_regimes.iloc[i] if i < len(btc_regimes) else None
            
            # Get contextual features for conviction scoring
            funding = row.get("funding_rate", None)
            sr_pos = row.get("sr_position", None)
            vwap_pos = row.get("vwap_position", None)
            oi_chg = row.get("oi_change", None)
            vol = row.get("volatility", None)
            
            # Clean NaN values
            if funding is not None and np.isnan(funding): funding = None
            if sr_pos is not None and np.isnan(sr_pos): sr_pos = None
            if vwap_pos is not None and np.isnan(vwap_pos): vwap_pos = None
            if oi_chg is not None and np.isnan(oi_chg): oi_chg = None
            if vol is not None and np.isnan(vol): vol = None
            
            # ─── CONVICTION SCORE ────────────────────────────────────
            conviction = RiskManager.compute_conviction_score(
                confidence=conf,
                regime=state,
                side=side,
                btc_regime=btc_regime,
                funding_rate=funding,
                sr_position=sr_pos,
                vwap_position=vwap_pos,
                oi_change=oi_chg,
                volatility=vol,
            )
            
            leverage = RiskManager.get_conviction_leverage(conviction)
            
            if leverage <= 0:
                equity.append(capital)
                continue
            
            # Calculate SL/TP
            sl_mult, tp_mult = config.get_atr_multipliers(leverage)
            if side == "LONG":
                sl = price - atr * sl_mult
                tp = price + atr * tp_mult
            else:
                sl = price + atr * sl_mult
                tp = price - atr * tp_mult
            
            open_trade = {
                "side": side,
                "entry": price,
                "sl": sl,
                "tp": tp,
                "leverage": leverage,
                "conviction": conviction,
                "breakeven_set": False,
            }
            
            equity.append(capital)
        
        start += step
    
    # Compute results
    if not trades:
        return None
    
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() * 100
    
    total_return = (capital / initial_capital - 1) * 100
    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    avg_conviction = np.mean([t["conviction"] for t in trades])
    avg_leverage = np.mean([t["leverage"] for t in trades])
    
    equity_returns = pd.Series(equity).pct_change().dropna()
    sharpe = (equity_returns.mean() / equity_returns.std() * np.sqrt(365 * 24)) if equity_returns.std() > 0 else 0
    
    return {
        "symbol": symbol,
        "total_return": round(total_return, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "win_rate": round(win_rate, 1),
        "total_trades": len(trades),
        "avg_conviction": round(avg_conviction, 1),
        "avg_leverage": round(avg_leverage, 1),
        "max_leverage": max(t["leverage"] for t in trades),
        "min_leverage": min(t["leverage"] for t in trades),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 100)
    print("  CONVICTION SCORING BACKTEST")
    print("  Testing 2x-35x continuous leverage with BTC macro filter")
    print("█" * 100)
    
    # Step 1: Fetch BTC data and compute macro regimes
    print("\n  📡 Fetching BTC macro data for regime filter...")
    df_btc = fetch_futures_klines("BTCUSDT", "1h", limit=1000)
    
    if df_btc is not None:
        try:
            df_btc = enrich_with_oi_funding(df_btc, "BTCUSDT")
        except Exception:
            pass
        df_btc_feat = compute_all_features(df_btc)
        btc_regimes = get_btc_regime_series(df_btc_feat)
        
        # Regime summary
        bull = (btc_regimes == config.REGIME_BULL).sum()
        bear = (btc_regimes == config.REGIME_BEAR).sum()
        chop = (btc_regimes == config.REGIME_CHOP).sum()
        crash = (btc_regimes == config.REGIME_CRASH).sum()
        print(f"  ✅ BTC macro: {bull} Bull | {bear} Bear | {chop} Chop | {crash} Crash candles")
    else:
        btc_regimes = None
        print("  ⚠️ No BTC data — running without macro filter")
    
    # Step 2: Run conviction backtest on each coin
    print(f"\n  Testing {len(STABLE_COINS)} stable coins with conviction scoring...")
    
    results = []
    for symbol in STABLE_COINS:
        sys.stdout.write(f"\r  ⏳ Testing {symbol:12s}...")
        sys.stdout.flush()
        
        df = fetch_futures_klines(symbol, "1h", limit=1000)
        if df is None or len(df) < 500:
            print(f"\r  ⚠️ {symbol}: insufficient data")
            continue
        
        # Enrich with OI + Funding
        try:
            df = enrich_with_oi_funding(df, symbol)
        except Exception:
            pass
        
        result = backtest_with_conviction(df, btc_regimes=btc_regimes,
                                           symbol=symbol, verbose=False)
        if result:
            # Calculate buy & hold
            first_close = df["close"].iloc[0]
            last_close = df["close"].iloc[-1]
            result["buy_hold"] = round((last_close - first_close) / first_close * 100, 1)
            result["alpha"] = round(result["total_return"] - result["buy_hold"], 1)
            results.append(result)
            print(f"\r  ✅ {symbol:12s} Return: {result['total_return']:+.1f}% | "
                  f"Alpha: {result['alpha']:+.1f}% | "
                  f"Avg Lev: {result['avg_leverage']:.0f}x | "
                  f"Conv: {result['avg_conviction']:.0f}")
        else:
            print(f"\r  ❌ {symbol}: no valid trades")
    
    if not results:
        print("\n  ❌ No results!")
        sys.exit(1)
    
    # Step 3: Print results
    print(f"\n{'='*110}")
    print(f"  📊 CONVICTION SCORING RESULTS — Continuous Leverage (2x-35x)")
    print(f"{'='*110}")
    print(f"  {'Symbol':<12} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} {'WR':>6} {'Trades':>7} "
          f"{'AvgConv':>8} {'AvgLev':>7} {'LevRange':>10} {'B&H':>8} {'Alpha':>8}")
    print(f"  {'─'*105}")
    
    for r in sorted(results, key=lambda x: x["alpha"], reverse=True):
        if r["total_return"] > 0:
            rating = "🟢"
        elif r["alpha"] > 0:
            rating = "🟡"
        else:
            rating = "🔴"
        
        print(f"  {r['symbol']:<12} {r['total_return']:>+7.1f}% {r['max_drawdown']:>-7.1f}% "
              f"{r['sharpe_ratio']:>+7.3f} {r['win_rate']:>5.0f}% {r['total_trades']:>6d}  "
              f"{r['avg_conviction']:>6.0f}   {r['avg_leverage']:>5.0f}x  "
              f"{r['min_leverage']:>2d}-{r['max_leverage']:<2d}x    "
              f"{r['buy_hold']:>+7.1f}% {r['alpha']:>+7.1f}%  {rating}")
    
    # Summary
    print(f"\n  {'─'*105}")
    avg_return = np.mean([r["total_return"] for r in results])
    avg_alpha = np.mean([r["alpha"] for r in results])
    avg_wr = np.mean([r["win_rate"] for r in results])
    avg_dd = np.mean([r["max_drawdown"] for r in results])
    avg_conv = np.mean([r["avg_conviction"] for r in results])
    avg_lev = np.mean([r["avg_leverage"] for r in results])
    
    print(f"  PORTFOLIO SUMMARY:")
    print(f"     Avg return:       {avg_return:+.1f}%")
    print(f"     Avg alpha:        {avg_alpha:+.1f}%")
    print(f"     Avg win rate:     {avg_wr:.0f}%")
    print(f"     Avg max DD:       {avg_dd:.1f}%")
    print(f"     Avg conviction:   {avg_conv:.0f}")
    print(f"     Avg leverage:     {avg_lev:.0f}x")
    print(f"{'='*110}")
    print("\n✅ Conviction backtest complete!")
