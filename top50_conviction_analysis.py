"""
Top-50 Conviction Scoring Deep Analysis
=========================================
Runs conviction-based backtest on top 50 coins (excl. memes) and produces:
  1. Per-coin results table with alpha, leverage, conviction
  2. Feature correlation breakdown — which factors drive each coin
  3. Final conviction leaderboard
"""
import sys
import os
import time
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_pipeline import fetch_futures_klines, enrich_with_oi_funding, _get_binance_client
from feature_engine import compute_all_features, compute_hmm_features
from hmm_brain import HMMBrain, HMM_FEATURES
from risk_manager import RiskManager

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Top50Analysis")

# ── Meme & junk exclusion ────────────────────────────────────────────────────
MEME_COINS = {
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT",
    "WIFUSDT", "BOMEUSDT", "MEMEUSDT", "PEOPLEUSDT", "ELONUSDT",
    "BABYDOGEUSDT", "NEIROUSDT", "TURBOUSDT", "MOGUSDT",
    "MYROUSDT", "LUNCUSDT", "LUNAUSDT",
    "1000PEPEUSDT", "1000SHIBUSDT", "1000FLOKIUSDT", "1000BONKUSDT",
    "1000LUNCUSDT",
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "DAIUSDT",
}
EXCLUDE_KW = ("UP", "DOWN", "BULL", "BEAR", "1000")


def get_top_futures(limit=50):
    client = _get_binance_client()
    tickers = client.futures_ticker()
    valid = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        base = sym.replace("USDT", "")
        if any(kw in base for kw in EXCLUDE_KW):
            continue
        if sym in MEME_COINS:
            continue
        vol = float(t.get("quoteVolume", 0))
        valid.append((sym, vol))
    valid.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in valid[:limit]]


def get_btc_regimes(df_btc):
    brain = HMMBrain(n_states=4)
    df_feat = compute_hmm_features(df_btc)
    brain.train(df_feat)
    if not brain.is_trained:
        return pd.Series(config.REGIME_CHOP, index=df_btc.index)
    states = brain.predict_all(df_feat)
    return pd.Series(states, index=df_btc.index[:len(states)])


def backtest_coin_with_details(df_raw, btc_regimes, symbol):
    """
    Backtest a single coin and capture detailed per-trade feature scores.
    Returns (result_dict, feature_stats_dict) or (None, None).
    """
    df = compute_all_features(df_raw)
    n = len(df)
    train_w, test_w, step = 400, 100, 100

    capital = 10000.0
    equity = [capital]
    trades = []
    open_trade = None

    # Accumulate feature score contributions across all entries
    feature_accum = {
        "hmm_conf": [], "btc_macro": [], "funding": [],
        "sr_vwap": [], "oi": [], "vol": [],
    }

    brain = HMMBrain(n_states=4)
    start = 0

    while start + train_w + test_w <= n:
        train_end = start + train_w
        test_end = min(train_end + test_w, n)
        df_train = df.iloc[start:train_end]
        df_test = df.iloc[train_end:test_end]

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
            row = df.iloc[i]
            price = row["close"]
            atr = row.get("atr", price * 0.02)
            if atr is None or atr <= 0 or (isinstance(atr, float) and np.isnan(atr)):
                atr = price * 0.02

            # ── Check open trade ──
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
                    trades.append({
                        "side": open_trade["side"], "lev": open_trade["leverage"],
                        "conv": open_trade["conv"], "pnl": result,
                        "pnl_pct": pnl_pct * 100, "hit": "TP" if hit_tp else "SL",
                    })
                    open_trade = None
                equity.append(capital)
                continue

            # ── New entry logic ──
            window = df.iloc[max(0, i - train_w):i + 1]
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

            btc_reg = btc_regimes.iloc[i] if i < len(btc_regimes) else None

            # Raw feature values
            funding = row.get("funding_rate", None)
            sr_pos = row.get("sr_position", None)
            vwap_pos = row.get("vwap_position", None)
            oi_chg = row.get("oi_change", None)
            vol = row.get("volatility", None)
            for v_name in ['funding', 'sr_pos', 'vwap_pos', 'oi_chg', 'vol']:
                val = locals()[v_name]
                if val is not None and isinstance(val, float) and np.isnan(val):
                    locals()[v_name] = None

            # ── Compute individual factor scores for analysis ──
            # 1. HMM conf score (0-30)
            hmm_s = min(30, max(0, (conf - 0.92) / 0.08 * 30)) if conf >= 0.92 else 0

            # 2. BTC macro (−15 to +25)
            btc_s = 10  # default (no data)
            if btc_reg is not None:
                if btc_reg == config.REGIME_CRASH:
                    btc_s = -15
                elif (side == 'LONG' and btc_reg == config.REGIME_BULL) or \
                     (side == 'SHORT' and btc_reg == config.REGIME_BEAR):
                    btc_s = 25
                elif (side == 'LONG' and btc_reg == config.REGIME_BEAR) or \
                     (side == 'SHORT' and btc_reg == config.REGIME_BULL):
                    btc_s = -10
                elif btc_reg == config.REGIME_CHOP:
                    btc_s = 5

            # 3. Funding (−5 to +15)
            fund_s = 7
            if funding is not None and not np.isnan(funding):
                if side == 'LONG' and funding < -0.0001:
                    fund_s = 15
                elif side == 'LONG' and funding > 0.0005:
                    fund_s = -5
                elif side == 'SHORT' and funding > 0.0003:
                    fund_s = 15
                elif side == 'SHORT' and funding < -0.0003:
                    fund_s = -5

            # 4. S/R + VWAP (0-15)
            sr_s = 7  # default neutral
            if sr_pos is not None and not np.isnan(sr_pos):
                if side == 'LONG':
                    sr_s = max(0, (1 - sr_pos) / 2 * 8)
                else:
                    sr_s = max(0, (1 + sr_pos) / 2 * 8)
            if vwap_pos is not None and not np.isnan(vwap_pos):
                if (side == 'LONG' and vwap_pos < 0) or (side == 'SHORT' and vwap_pos > 0):
                    sr_s += 7
                else:
                    sr_s += 3
            else:
                sr_s += 3
            sr_s = min(15, sr_s)

            # 5. OI (0-10)
            oi_s = 5
            if oi_chg is not None and not np.isnan(oi_chg):
                if (side == 'LONG' and oi_chg > 0.02) or (side == 'SHORT' and oi_chg < -0.02):
                    oi_s = 10
                elif abs(oi_chg) < 0.01:
                    oi_s = 5
                else:
                    oi_s = 3

            # 6. Vol (0-5)
            vol_s = 3
            if vol is not None and not np.isnan(vol):
                if 0.005 < vol < 0.03:
                    vol_s = 5
                elif vol >= 0.03:
                    vol_s = 1

            # Accumulate
            feature_accum["hmm_conf"].append(hmm_s)
            feature_accum["btc_macro"].append(btc_s)
            feature_accum["funding"].append(fund_s)
            feature_accum["sr_vwap"].append(sr_s)
            feature_accum["oi"].append(oi_s)
            feature_accum["vol"].append(vol_s)

            conviction = max(0, min(100, hmm_s + btc_s + fund_s + sr_s + oi_s + vol_s))
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
                          "leverage": leverage, "conv": conviction, "be": False}
            equity.append(capital)

        start += step

    if not trades or len(trades) < 3:
        return None, None

    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min() * 100
    total_ret = (capital / 10000.0 - 1) * 100
    wins = [t for t in trades if t["pnl"] > 0]
    wr = len(wins) / len(trades) * 100

    first_c = df["close"].iloc[0]
    last_c = df["close"].iloc[-1]
    bh = (last_c - first_c) / first_c * 100

    eq_ret = pd.Series(eq).pct_change().dropna()
    sharpe = (eq_ret.mean() / eq_ret.std() * np.sqrt(365 * 24)) if eq_ret.std() > 0 else 0

    result = {
        "symbol": symbol, "return": round(total_ret, 1), "max_dd": round(dd, 1),
        "sharpe": round(sharpe, 3), "wr": round(wr, 0),
        "trades": len(trades), "bh": round(bh, 1), "alpha": round(total_ret - bh, 1),
        "avg_conv": round(np.mean([t["conv"] for t in trades]), 1),
        "avg_lev": round(np.mean([t["lev"] for t in trades]), 1),
        "max_lev": max(t["lev"] for t in trades),
    }

    # Feature stats — average contribution per factor
    feat_stats = {}
    for k, v in feature_accum.items():
        if v:
            feat_stats[k] = round(np.mean(v), 1)
        else:
            feat_stats[k] = 0.0

    return result, feat_stats


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "█" * 120)
    print("  TOP-50 CONVICTION SCORING DEEP ANALYSIS")
    print("  8-feature HMM + BTC Macro Filter + Continuous 2x-35x Leverage")
    print("█" * 120)

    # 1. BTC macro
    print("\n  📡 Fetching BTC macro regime...")
    df_btc = fetch_futures_klines("BTCUSDT", "1h", limit=1000)
    try:
        df_btc = enrich_with_oi_funding(df_btc, "BTCUSDT")
    except Exception:
        pass
    df_btc = compute_all_features(df_btc)
    btc_reg = get_btc_regimes(df_btc)
    bull = (btc_reg == config.REGIME_BULL).sum()
    bear = (btc_reg == config.REGIME_BEAR).sum()
    chop = (btc_reg == config.REGIME_CHOP).sum()
    crash = (btc_reg == config.REGIME_CRASH).sum()
    print(f"  ✅ BTC macro: {bull} Bull | {bear} Bear | {chop} Chop | {crash} Crash")

    # 2. Fetch top 50 coins
    print(f"\n  📡 Fetching top 50 futures coins...")
    coins = get_top_futures(50)
    print(f"  ✅ Got {len(coins)} coins")

    # 3. Run backtests
    results = []
    feature_table = []

    for idx, sym in enumerate(coins):
        sys.stdout.write(f"\r  ⏳ [{idx+1:2d}/{len(coins)}] {sym:14s}...")
        sys.stdout.flush()

        df = fetch_futures_klines(sym, "1h", limit=1000)
        if df is None or len(df) < 500:
            continue

        try:
            df = enrich_with_oi_funding(df, sym)
        except Exception:
            pass

        try:
            res, feats = backtest_coin_with_details(df, btc_reg, sym)
        except Exception:
            continue
        if res and feats:
            results.append(res)
            feature_table.append({"symbol": sym, **feats, **res})

        if (idx + 1) % 5 == 0:
            time.sleep(1)

    print(f"\r  ✅ Completed {len(results)}/{len(coins)} coins successfully" + " " * 30)

    if not results:
        print("  ❌ No results!")
        sys.exit(1)

    # Sort by alpha
    results.sort(key=lambda r: r["alpha"], reverse=True)
    feature_table.sort(key=lambda r: r["alpha"], reverse=True)

    # ── TABLE 1: Performance ──────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("  📊 TABLE 1: PERFORMANCE SUMMARY (sorted by Alpha)")
    print(f"{'='*120}")
    print(f"  {'#':>3} {'Symbol':<14} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} {'WR':>5} {'Tr':>4} "
          f"{'AvgConv':>8} {'AvgLev':>7} {'MaxLev':>7} {'B&H':>8} {'Alpha':>8}  Verdict")
    print(f"  {'─'*115}")

    for i, r in enumerate(results):
        if r["return"] > 0:
            v = "🟢 PROFIT"
        elif r["alpha"] > 0:
            v = "🟡 ALPHA"
        else:
            v = "🔴 LOSS"

        print(f"  {i+1:3d} {r['symbol']:<14} {r['return']:>+7.1f}% {r['max_dd']:>-7.1f}% "
              f"{r['sharpe']:>+7.3f} {r['wr']:>4.0f}% {r['trades']:>3d}  "
              f"{r['avg_conv']:>6.0f}   {r['avg_lev']:>5.0f}x  {r['max_lev']:>5d}x  "
              f"{r['bh']:>+7.1f}% {r['alpha']:>+7.1f}%  {v}")

    # ── TABLE 2: Feature Correlation ──────────────────────────────────────
    print(f"\n{'='*120}")
    print("  📊 TABLE 2: FEATURE CONTRIBUTION BREAKDOWN (avg score per factor)")
    print("  HMM Conf (max 30) | BTC Macro (max 25) | Funding (max 15) | S/R+VWAP (max 15) | OI (max 10) | Vol (max 5)")
    print(f"{'='*120}")
    print(f"  {'#':>3} {'Symbol':<14} {'HMM':>6} {'BTC':>6} {'Fund':>6} {'S/R':>6} {'OI':>6} {'Vol':>6}  "
          f"{'Total':>6} {'Alpha':>8}  {'Dominant Factor':<20}")
    print(f"  {'─'*110}")

    for i, ft in enumerate(feature_table):
        # Find dominant factor
        factors = {
            "HMM Confidence": ft.get("hmm_conf", 0),
            "BTC Macro": ft.get("btc_macro", 0),
            "Funding Rate": ft.get("funding", 0),
            "S/R + VWAP": ft.get("sr_vwap", 0),
            "OI Momentum": ft.get("oi", 0),
            "Volatility": ft.get("vol", 0),
        }
        dominant = max(factors, key=factors.get)
        total = sum(factors.values())

        print(f"  {i+1:3d} {ft['symbol']:<14} {ft.get('hmm_conf',0):>5.1f} {ft.get('btc_macro',0):>5.1f} "
              f"{ft.get('funding',0):>5.1f} {ft.get('sr_vwap',0):>5.1f} {ft.get('oi',0):>5.1f} "
              f"{ft.get('vol',0):>5.1f}  {total:>5.1f} {ft['alpha']:>+7.1f}%  {dominant}")

    # ── TABLE 3: Summary stats ────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("  📊 TABLE 3: PORTFOLIO SUMMARY")
    print(f"{'='*120}")

    profitable = sum(1 for r in results if r["return"] > 0)
    alpha_pos = sum(1 for r in results if r["alpha"] > 0)
    avg_ret = np.mean([r["return"] for r in results])
    avg_alpha = np.mean([r["alpha"] for r in results])
    avg_wr = np.mean([r["wr"] for r in results])
    avg_dd = np.mean([r["max_dd"] for r in results])
    avg_conv = np.mean([r["avg_conv"] for r in results])
    avg_lev = np.mean([r["avg_lev"] for r in results])
    med_alpha = np.median([r["alpha"] for r in results])

    print(f"  Coins tested:        {len(results)}")
    print(f"  Profitable:          {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)")
    print(f"  Beat buy & hold:     {alpha_pos}/{len(results)} ({alpha_pos/len(results)*100:.0f}%)")
    print(f"  Avg return:          {avg_ret:+.1f}%")
    print(f"  Avg alpha:           {avg_alpha:+.1f}%")
    print(f"  Median alpha:        {med_alpha:+.1f}%")
    print(f"  Avg win rate:        {avg_wr:.0f}%")
    print(f"  Avg max drawdown:    {avg_dd:.1f}%")
    print(f"  Avg conviction:      {avg_conv:.0f}")
    print(f"  Avg leverage:        {avg_lev:.0f}x")

    # Feature importance across all coins
    print(f"\n  🔑 FEATURE IMPORTANCE (avg contribution across all coins):")
    all_hmm = np.mean([ft.get("hmm_conf", 0) for ft in feature_table])
    all_btc = np.mean([ft.get("btc_macro", 0) for ft in feature_table])
    all_fund = np.mean([ft.get("funding", 0) for ft in feature_table])
    all_sr = np.mean([ft.get("sr_vwap", 0) for ft in feature_table])
    all_oi = np.mean([ft.get("oi", 0) for ft in feature_table])
    all_vol = np.mean([ft.get("vol", 0) for ft in feature_table])

    total_avg = all_hmm + all_btc + all_fund + all_sr + all_oi + all_vol
    print(f"     HMM Confidence:  {all_hmm:>5.1f}/30  ({all_hmm/30*100:.0f}% utilization)")
    print(f"     BTC Macro:       {all_btc:>5.1f}/25  ({max(0,all_btc)/25*100:.0f}% utilization)")
    print(f"     Funding Rate:    {all_fund:>5.1f}/15  ({all_fund/15*100:.0f}% utilization)")
    print(f"     S/R + VWAP:      {all_sr:>5.1f}/15  ({all_sr/15*100:.0f}% utilization)")
    print(f"     OI Momentum:     {all_oi:>5.1f}/10  ({all_oi/10*100:.0f}% utilization)")
    print(f"     Volatility:      {all_vol:>5.1f}/5   ({all_vol/5*100:.0f}% utilization)")
    print(f"     TOTAL:           {total_avg:>5.1f}/100")

    # Correlation: alpha vs each factor
    print(f"\n  📈 ALPHA CORRELATION (which features predict profitability):")
    alphas = np.array([ft["alpha"] for ft in feature_table])
    for feat_name, feat_key in [("HMM Confidence", "hmm_conf"), ("BTC Macro", "btc_macro"),
                                 ("Funding Rate", "funding"), ("S/R + VWAP", "sr_vwap"),
                                 ("OI Momentum", "oi"), ("Volatility", "vol")]:
        vals = np.array([ft.get(feat_key, 0) for ft in feature_table])
        if len(vals) > 2 and np.std(vals) > 0:
            corr = np.corrcoef(alphas, vals)[0, 1]
            bar = "█" * int(abs(corr) * 20)
            sign = "+" if corr > 0 else "−"
            print(f"     {feat_name:<18} r={corr:>+.3f}  {sign}{bar}")
        else:
            print(f"     {feat_name:<18} r= N/A")

    print(f"\n{'='*120}")

    # Top 10 recommendation
    top10 = [r for r in results if r["alpha"] > 0][:10]
    if top10:
        print(f"\n  🏆 TOP 10 RECOMMENDED COINS FOR LIVE TRADING:")
        for i, r in enumerate(top10):
            print(f"     {i+1:2d}. {r['symbol']:<14} Alpha: {r['alpha']:>+.1f}%  |  "
                  f"Return: {r['return']:>+.1f}%  |  MaxDD: {r['max_dd']:.1f}%  |  "
                  f"Conv: {r['avg_conv']:.0f}  |  Lev: {r['avg_lev']:.0f}x")

    print(f"\n{'='*120}")
    print("✅ Analysis complete!")
