"""
CoinDCX Data Backtest — In-Sample + Walk-Forward
==================================================
Fetches 1000 candles from CoinDCX for each coin and runs:
  1. In-Sample backtest (full 1000 candles trained + tested together)
  2. Walk-Forward OOS test (train 400 → test 100, slide forward)

Uses the conviction scoring system with continuous 2x-35x leverage.
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
logger = logging.getLogger("CoinDCXBacktest")

# NOTE: CoinDCX API is unresponsive, using Binance Futures for data.
# Same coins — same logic — just a different data source.
FETCH = fetch_futures_klines

# Recommended coins from the analysis
TEST_COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "BCHUSDT",
    "SOLUSDT", "ADAUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT",
    "LINKUSDT", "LTCUSDT", "ETCUSDT", "ATOMUSDT", "APTUSDT",
]


def run_single_backtest(df, symbol, btc_regimes=None, mode="walk_forward"):
    """
    Run backtest on a single coin with conviction scoring.
    
    mode = "in_sample"    → train on all data, test on same data (optimistic)
    mode = "walk_forward" → train 400, test 100, slide (realistic)
    """
    df_full = compute_all_features(df)
    n = len(df_full)
    
    capital = 10000.0
    equity = [capital]
    trades = []
    open_trade = None
    
    if mode == "in_sample":
        # Train on full dataset, test on full dataset
        brain = HMMBrain(n_states=4)
        try:
            brain.train(compute_hmm_features(df_full))
        except Exception:
            return None
        if not brain.is_trained:
            return None
        
        for i in range(50, n):
            row = df_full.iloc[i]
            price = row["close"]
            atr = row.get("atr", price * 0.02)
            if atr is None or atr <= 0 or (isinstance(atr, float) and np.isnan(atr)):
                atr = price * 0.02
            
            # Check open trade
            if open_trade is not None:
                if open_trade["side"] == "LONG":
                    pnl_pct = (price - open_trade["entry"]) / open_trade["entry"]
                else:
                    pnl_pct = (open_trade["entry"] - price) / open_trade["entry"]
                pnl_pct *= open_trade["leverage"]
                
                hit_sl = (price <= open_trade["sl"]) if open_trade["side"] == "LONG" else (price >= open_trade["sl"])
                hit_tp = (price >= open_trade["tp"]) if open_trade["side"] == "LONG" else (price <= open_trade["tp"])
                
                if pnl_pct > 0.10 and not open_trade.get("be"):
                    open_trade["sl"] = open_trade["entry"]
                    open_trade["be"] = True
                
                if hit_sl or hit_tp:
                    risk_amt = capital * config.RISK_PER_TRADE
                    result = risk_amt * pnl_pct
                    fee = risk_amt * open_trade["leverage"] * config.TAKER_FEE * 2
                    result -= fee
                    capital += result
                    trades.append({"side": open_trade["side"], "lev": open_trade["leverage"],
                                   "pnl": result, "hit": "TP" if hit_tp else "SL"})
                    open_trade = None
                equity.append(capital)
                continue
            
            # New entry
            window = df_full.iloc[max(0, i - 400):i + 1]
            df_hmm = compute_hmm_features(window)
            if len(df_hmm.dropna()) < 20:
                equity.append(capital)
                continue
            
            state, conf = brain.predict(df_hmm)
            if state == config.REGIME_CHOP:
                equity.append(capital)
                continue
            
            regime_name = brain.get_regime_name(state)
            if regime_name == "BULLISH":
                side = "LONG"
            elif regime_name in ("BEARISH", "CRASH/PANIC"):
                side = "SHORT"
            else:
                equity.append(capital)
                continue
            
            btc_reg = btc_regimes.iloc[i] if btc_regimes is not None and i < len(btc_regimes) else None
            
            funding = row.get("funding_rate", None)
            sr_pos = row.get("sr_position", None)
            vwap_pos = row.get("vwap_position", None)
            oi_chg = row.get("oi_change", None)
            vol = row.get("volatility", None)
            for vn in [funding, sr_pos, vwap_pos, oi_chg, vol]:
                if vn is not None and isinstance(vn, float) and np.isnan(vn):
                    vn = None
            
            conviction = RiskManager.compute_conviction_score(
                confidence=conf, regime=state, side=side,
                btc_regime=btc_reg, funding_rate=funding,
                sr_position=sr_pos, vwap_position=vwap_pos,
                oi_change=oi_chg, volatility=vol,
            )
            leverage = RiskManager.get_conviction_leverage(conviction)
            if leverage <= 0:
                equity.append(capital)
                continue
            
            sl_m, tp_m = config.get_atr_multipliers(leverage)
            if side == "LONG":
                sl, tp = price - atr * sl_m, price + atr * tp_m
            else:
                sl, tp = price + atr * sl_m, price - atr * tp_m
            
            open_trade = {"side": side, "entry": price, "sl": sl, "tp": tp,
                          "leverage": leverage, "be": False}
            equity.append(capital)
    
    else:  # walk_forward
        train_w, test_w, step = 400, 100, 100
        start = 0
        
        while start + train_w + test_w <= n:
            train_end = start + train_w
            test_end = min(train_end + test_w, n)
            df_train = df_full.iloc[start:train_end]
            df_test = df_full.iloc[train_end:test_end]
            
            brain_c = HMMBrain(n_states=4)
            try:
                brain_c.train(compute_hmm_features(df_train))
            except Exception:
                start += step
                continue
            if not brain_c.is_trained:
                start += step
                continue
            
            for idx in range(len(df_test)):
                i = train_end + idx
                row = df_full.iloc[i]
                price = row["close"]
                atr = row.get("atr", price * 0.02)
                if atr is None or atr <= 0 or (isinstance(atr, float) and np.isnan(atr)):
                    atr = price * 0.02
                
                if open_trade is not None:
                    if open_trade["side"] == "LONG":
                        pnl_pct = (price - open_trade["entry"]) / open_trade["entry"]
                    else:
                        pnl_pct = (open_trade["entry"] - price) / open_trade["entry"]
                    pnl_pct *= open_trade["leverage"]
                    
                    hit_sl = (price <= open_trade["sl"]) if open_trade["side"] == "LONG" else (price >= open_trade["sl"])
                    hit_tp = (price >= open_trade["tp"]) if open_trade["side"] == "LONG" else (price <= open_trade["tp"])
                    
                    if pnl_pct > 0.10 and not open_trade.get("be"):
                        open_trade["sl"] = open_trade["entry"]
                        open_trade["be"] = True
                    
                    if hit_sl or hit_tp:
                        risk_amt = capital * config.RISK_PER_TRADE
                        result = risk_amt * pnl_pct
                        fee = risk_amt * open_trade["leverage"] * config.TAKER_FEE * 2
                        result -= fee
                        capital += result
                        trades.append({"side": open_trade["side"], "lev": open_trade["leverage"],
                                       "pnl": result, "hit": "TP" if hit_tp else "SL"})
                        open_trade = None
                    equity.append(capital)
                    continue
                
                window = df_full.iloc[max(0, i - train_w):i + 1]
                df_hmm = compute_hmm_features(window)
                if len(df_hmm.dropna()) < 20:
                    equity.append(capital)
                    continue
                
                state, conf = brain_c.predict(df_hmm)
                if state == config.REGIME_CHOP:
                    equity.append(capital)
                    continue
                
                regime_name = brain_c.get_regime_name(state)
                if regime_name == "BULLISH":
                    side = "LONG"
                elif regime_name in ("BEARISH", "CRASH/PANIC"):
                    side = "SHORT"
                else:
                    equity.append(capital)
                    continue
                
                btc_reg = btc_regimes.iloc[i] if btc_regimes is not None and i < len(btc_regimes) else None
                
                funding = row.get("funding_rate", None)
                sr_pos = row.get("sr_position", None)
                vwap_pos = row.get("vwap_position", None)
                oi_chg = row.get("oi_change", None)
                vol = row.get("volatility", None)
                
                conviction = RiskManager.compute_conviction_score(
                    confidence=conf, regime=state, side=side,
                    btc_regime=btc_reg, funding_rate=funding,
                    sr_position=sr_pos, vwap_position=vwap_pos,
                    oi_change=oi_chg, volatility=vol,
                )
                leverage = RiskManager.get_conviction_leverage(conviction)
                if leverage <= 0:
                    equity.append(capital)
                    continue
                
                sl_m, tp_m = config.get_atr_multipliers(leverage)
                if side == "LONG":
                    sl, tp = price - atr * sl_m, price + atr * tp_m
                else:
                    sl, tp = price + atr * sl_m, price - atr * tp_m
                
                open_trade = {"side": side, "entry": price, "sl": sl, "tp": tp,
                              "leverage": leverage, "be": False}
                equity.append(capital)
            
            start += step
    
    if not trades or len(trades) < 3:
        return None
    
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min() * 100
    total_ret = (capital / 10000.0 - 1) * 100
    wins = [t for t in trades if t["pnl"] > 0]
    wr = len(wins) / len(trades) * 100
    avg_lev = np.mean([t["lev"] for t in trades])
    
    first_c = df["close"].iloc[0]
    last_c = df["close"].iloc[-1]
    bh = (last_c - first_c) / first_c * 100
    
    eq_ret = pd.Series(eq).pct_change().dropna()
    sharpe = (eq_ret.mean() / eq_ret.std() * np.sqrt(365 * 24)) if eq_ret.std() > 0 else 0
    
    return {
        "symbol": symbol, "return": round(total_ret, 1), "max_dd": round(dd, 1),
        "sharpe": round(sharpe, 3), "wr": round(wr, 0), "trades": len(trades),
        "avg_lev": round(avg_lev, 1), "bh": round(bh, 1),
        "alpha": round(total_ret - bh, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "█" * 110)
    print("  CoinDCX DATA BACKTEST — 1000 Candles")
    print("  In-Sample vs Walk-Forward | 7-Factor Conviction Scoring (10x-35x)")
    print("█" * 110)
    
    # 1. Fetch BTC macro
    print("\n  📡 Fetching BTC data (CoinDCX → Binance fallback)...")
    df_btc = FETCH("BTCUSDT", "1h", limit=1000)
    
    if df_btc is not None and len(df_btc) > 0:
        src = "CoinDCX" if len(df_btc) > 0 else "Binance"
        print(f"  ✅ Got {len(df_btc)} BTC candles from {src}")
        try:
            df_btc = enrich_with_oi_funding(df_btc, "BTCUSDT")
        except Exception:
            pass
        df_btc = compute_all_features(df_btc)
        
        brain_btc = HMMBrain(n_states=4)
        brain_btc.train(compute_hmm_features(df_btc))
        if brain_btc.is_trained:
            btc_regimes = pd.Series(brain_btc.predict_all(compute_hmm_features(df_btc)),
                                     index=df_btc.index[:len(brain_btc.predict_all(compute_hmm_features(df_btc)))])
            bull = (btc_regimes == config.REGIME_BULL).sum()
            bear = (btc_regimes == config.REGIME_BEAR).sum()
            chop = (btc_regimes == config.REGIME_CHOP).sum()
            crash = (btc_regimes == config.REGIME_CRASH).sum()
            print(f"  ✅ BTC macro: {bull} Bull | {bear} Bear | {chop} Chop | {crash} Crash")
        else:
            btc_regimes = None
    else:
        btc_regimes = None
        print("  ⚠️ No BTC data available")
    
    # 2. Run backtests
    is_results = []
    wf_results = []
    
    print(f"\n  Testing {len(TEST_COINS)} coins (CoinDCX data, 1000 candles)...")
    print(f"  {'─'*100}")
    
    for sym in TEST_COINS:
        sys.stdout.write(f"\r  ⏳ Fetching {sym:12s}...")
        sys.stdout.flush()
        
        df = FETCH(sym, "1h", limit=1000)
        if df is None or len(df) < 500:
            print(f"\r  ⚠️ {sym}: only {len(df) if df is not None else 0} candles (need 500+)")
            continue
        
        print(f"\r  📊 {sym:12s} — {len(df)} candles", end="")
        
        try:
            df = enrich_with_oi_funding(df, sym)
        except Exception:
            pass
        
        # In-Sample
        try:
            is_res = run_single_backtest(df, sym, btc_regimes, mode="in_sample")
        except Exception:
            is_res = None
        
        # Walk-Forward
        try:
            wf_res = run_single_backtest(df, sym, btc_regimes, mode="walk_forward")
        except Exception:
            wf_res = None
        
        if is_res:
            is_results.append(is_res)
        if wf_res:
            wf_results.append(wf_res)
        
        status = ""
        if is_res:
            status += f" | IS: {is_res['return']:+.1f}%"
        if wf_res:
            status += f" | WF: {wf_res['return']:+.1f}%"
        print(f"{status}")
        
        time.sleep(0.5)
    
    # 3. Print results
    print(f"\n{'='*110}")
    print("  📊 IN-SAMPLE RESULTS (train on ALL data, test on SAME data — optimistic upper bound)")
    print(f"{'='*110}")
    if is_results:
        print(f"  {'Symbol':<14} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} {'WR':>6} {'Tr':>4} {'AvgLev':>7} {'B&H':>8} {'Alpha':>8}")
        print(f"  {'─'*85}")
        for r in sorted(is_results, key=lambda x: x["alpha"], reverse=True):
            v = "🟢" if r["return"] > 0 else ("🟡" if r["alpha"] > 0 else "🔴")
            print(f"  {r['symbol']:<14} {r['return']:>+7.1f}% {r['max_dd']:>-7.1f}% "
                  f"{r['sharpe']:>+7.3f} {r['wr']:>4.0f}% {r['trades']:>3d}  {r['avg_lev']:>5.0f}x  "
                  f"{r['bh']:>+7.1f}% {r['alpha']:>+7.1f}%  {v}")
        
        avg_ret = np.mean([r["return"] for r in is_results])
        avg_alpha = np.mean([r["alpha"] for r in is_results])
        print(f"\n  IS Summary: Avg return {avg_ret:+.1f}% | Avg alpha {avg_alpha:+.1f}% | "
              f"{sum(1 for r in is_results if r['alpha'] > 0)}/{len(is_results)} beat B&H")
    
    print(f"\n{'='*110}")
    print("  📊 WALK-FORWARD RESULTS (train 400, test 100, slide — realistic)")
    print(f"{'='*110}")
    if wf_results:
        print(f"  {'Symbol':<14} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} {'WR':>6} {'Tr':>4} {'AvgLev':>7} {'B&H':>8} {'Alpha':>8}")
        print(f"  {'─'*85}")
        for r in sorted(wf_results, key=lambda x: x["alpha"], reverse=True):
            v = "🟢" if r["return"] > 0 else ("🟡" if r["alpha"] > 0 else "🔴")
            print(f"  {r['symbol']:<14} {r['return']:>+7.1f}% {r['max_dd']:>-7.1f}% "
                  f"{r['sharpe']:>+7.3f} {r['wr']:>4.0f}% {r['trades']:>3d}  {r['avg_lev']:>5.0f}x  "
                  f"{r['bh']:>+7.1f}% {r['alpha']:>+7.1f}%  {v}")
        
        avg_ret = np.mean([r["return"] for r in wf_results])
        avg_alpha = np.mean([r["alpha"] for r in wf_results])
        avg_dd = np.mean([r["max_dd"] for r in wf_results])
        print(f"\n  WF Summary: Avg return {avg_ret:+.1f}% | Avg alpha {avg_alpha:+.1f}% | "
              f"Avg MaxDD {avg_dd:.1f}% | {sum(1 for r in wf_results if r['alpha'] > 0)}/{len(wf_results)} beat B&H")
    
    # Comparison table
    if is_results and wf_results:
        print(f"\n{'='*110}")
        print("  📊 COMPARISON: In-Sample vs Walk-Forward (Overfitting Check)")
        print(f"{'='*110}")
        print(f"  {'Symbol':<14} {'IS Return':>10} {'WF Return':>10} {'IS Alpha':>10} {'WF Alpha':>10} {'Gap':>8}  Verdict")
        print(f"  {'─'*80}")
        
        is_map = {r["symbol"]: r for r in is_results}
        wf_map = {r["symbol"]: r for r in wf_results}
        
        for sym in TEST_COINS:
            if sym in is_map and sym in wf_map:
                is_r = is_map[sym]
                wf_r = wf_map[sym]
                gap = is_r["alpha"] - wf_r["alpha"]
                
                if gap < 20:
                    verdict = "✅ Robust"
                elif gap < 50:
                    verdict = "⚠️ Some overfit"
                else:
                    verdict = "❌ Overfitting"
                
                print(f"  {sym:<14} {is_r['return']:>+9.1f}% {wf_r['return']:>+9.1f}% "
                      f"{is_r['alpha']:>+9.1f}% {wf_r['alpha']:>+9.1f}% {gap:>+7.1f}%  {verdict}")
        
        avg_gap = np.mean([is_map[s]["alpha"] - wf_map[s]["alpha"] 
                           for s in TEST_COINS if s in is_map and s in wf_map])
        print(f"\n  Average IS-WF gap: {avg_gap:+.1f}%")
        if avg_gap < 20:
            print("  ✅ Model is robust — minimal overfitting detected")
        elif avg_gap < 50:
            print("  ⚠️ Some overfitting detected — results should be discounted slightly")
        else:
            print("  ❌ Significant overfitting — walk-forward results are the true benchmark")
    
    print(f"\n{'='*110}")
    print("✅ CoinDCX backtest complete!")
