#!/usr/bin/env python3
"""
tools/experiment_features_weights.py

Four experiments to empirically guide production improvements.
RESULTS ONLY — does NOT modify any production code.
Share output with user → get LGTM → then apply changes.

EXP 1 — HMM Feature Ablation:    14 new features tested one-by-one vs baseline 4
EXP 2 — Greedy Feature Selection: optimal feature combination via forward selection
EXP 3 — Conviction Factor IC:     predictive power of each conviction score factor
EXP 4 — Weight Optimization:      random search over conviction score weights

Run:  python tools/experiment_features_weights.py
Expected runtime: 8–15 min (HMM fitting × 15 coins × 14+ feature sets)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# Reuse existing feature_engine helpers where available
from feature_engine import compute_ema, compute_atr, compute_bollinger_bands

# ─── Configuration ─────────────────────────────────────────────────────────────

TEST_COINS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AAVEUSDT", "WLDUSDT", "ETCUSDT", "FETUSDT", "BCHUSDT",
    "ZECUSDT", "PEPEUSDT", "ENAUSDT", "SOLUSDT", "LINKUSDT",
]

BASELINE_FEATURES = ["log_return", "volatility", "volume_change", "rsi_norm"]

NEW_FEATURES_META = {
    "adx":           "ADX Trend Strength (14)",
    "macd_hist":     "MACD Histogram (12/26/9)",
    "ema200_dist":   "EMA-200 Distance",
    "hh_ll_seq":     "HH/LL Sequence (20-bar)",
    "atr_ratio":     "ATR Ratio (7÷28)",
    "bb_width":      "Bollinger Band Width",
    "obv_norm":      "OBV Change (norm)",
    "vol_slope":     "Volume Slope (5-bar)",
    "hurst":         "Hurst Exponent (40-bar)",
    "ret_autocorr":  "Return Autocorr (20-bar)",
    "btc_return":    "BTC Same-Period Return",
    "btc_corr":      "BTC Rolling Corr (20)",
    "funding_proxy": "Funding Rate Proxy",
    "oi_proxy":      "OI Change Proxy",
}

N_TRAIN      = 400    # HMM training bars
N_TEST       = 200    # evaluation bars
FORWARD_H    = 12     # 12 × 4h = 48h forward return (ground truth horizon)
WEIGHT_TRIALS = 300   # random search trials in EXP 4

# ─── Data Fetcher ──────────────────────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        from binance.client import Client
        _client = Client(tld="com")
    return _client


def fetch_ohlcv(symbol, interval="4h", start="2022-06-01", end="2026-03-01"):
    """Fetch OHLCV from Binance production API (no auth needed for historical data)."""
    try:
        client = _get_client()
        klines = client.get_historical_klines(symbol, interval, start, end)
        if not klines:
            return None
        df = pd.DataFrame(klines, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "ct", "qa", "t", "tb", "tq", "i",
        ])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["ts", "open", "high", "low", "close", "volume"]].set_index("ts")
    except Exception as e:
        return None


# ─── Feature Engine (all 4 baseline + 14 new) ─────────────────────────────────

def compute_features(df, btc_df=None):
    """
    Compute all 18 features from 4h OHLCV.
    Returns a DataFrame with one column per feature (all normalized to ~[-1, +1]).
    """
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]
    result = pd.DataFrame(index=df.index)

    # ── Baseline 4 ──────────────────────────────────────────────────────────────
    log_ret = np.log(c / c.shift(1))
    result["log_return"]    = log_ret
    result["volatility"]    = (h - l) / (c + 1e-10)         # candle range / close
    result["volume_change"] = np.log(v / v.shift(1)).clip(-3, 3)

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    result["rsi_norm"] = ((100 - 100 / (1 + gain / (loss + 1e-10))) - 50) / 50

    # ── F1: ADX (14) ────────────────────────────────────────────────────────────
    tr   = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    hdiff = h.diff(); ldiff = -l.diff()
    dmp  = pd.Series(np.where((hdiff > 0) & (hdiff > ldiff), hdiff.values, 0.0), index=df.index)
    dmn  = pd.Series(np.where((ldiff > 0) & (ldiff > hdiff), ldiff.values, 0.0), index=df.index)
    tr14 = tr.ewm(com=13, adjust=False).mean()
    di_p = dmp.ewm(com=13, adjust=False).mean() / (tr14 + 1e-10)
    di_n = dmn.ewm(com=13, adjust=False).mean() / (tr14 + 1e-10)
    dx   = (di_p - di_n).abs() / (di_p + di_n + 1e-10)
    result["adx"] = dx.ewm(com=13, adjust=False).mean()  # 0=no trend, 1=strong

    # ── F2: MACD Histogram ──────────────────────────────────────────────────────
    macd_line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    result["macd_hist"] = (macd_hist / (c.rolling(20).std() + 1e-10)).clip(-3, 3) / 3

    # ── F3: EMA-200 Distance ────────────────────────────────────────────────────
    ema200 = compute_ema(c, 200)
    result["ema200_dist"] = ((c - ema200) / (ema200 + 1e-10)).clip(-0.5, 0.5)

    # ── F4: HH / LL Sequence (20-bar) ───────────────────────────────────────────
    result["hh_ll_seq"] = (
        (c > c.rolling(20).max().shift(1)).astype(float)
        - (c < c.rolling(20).min().shift(1)).astype(float)
    )

    # ── F5: ATR Ratio (7 ÷ 28) ─────────────────────────────────────────────────
    atr7  = compute_atr(df, 7)
    atr28 = compute_atr(df, 28)
    result["atr_ratio"] = (atr7 / (atr28 + 1e-10)).clip(0.1, 5.0) / 5.0

    # ── F6: Bollinger Band Width ─────────────────────────────────────────────────
    bb_mid, bb_upper, bb_lower = compute_bollinger_bands(c, 20, 2.0)
    result["bb_width"] = ((bb_upper - bb_lower) / (bb_mid + 1e-10)).clip(0, 0.5) / 0.5

    # ── F7: OBV Change (normalized) ─────────────────────────────────────────────
    obv_chg = (v * np.sign(c.diff())).rolling(10).sum()
    obv_std = obv_chg.rolling(30).std()
    result["obv_norm"] = (obv_chg / (obv_std + 1e-10)).clip(-3, 3) / 3

    # ── F8: Volume Slope (5-bar log-diff) ───────────────────────────────────────
    result["vol_slope"] = np.log(v + 1).diff(5).clip(-0.5, 0.5) / 0.5

    # ── F9: Hurst Exponent (40-bar, computed every 5 bars then ffill) ────────────
    def sparse_hurst(series, window=40, step=5):
        arr = np.log(series.values + 1e-10)
        out = np.full(len(arr), np.nan)
        lags = [2, 4, 8, 16]
        log_lags = np.log(lags)
        for i in range(window, len(arr), step):
            ts = arr[i - window:i]
            if np.any(np.isnan(ts)):
                continue
            try:
                tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
                if all(t > 1e-10 for t in tau):
                    out[i] = float(np.clip(np.polyfit(log_lags, np.log(tau), 1)[0], 0, 1))
            except Exception:
                pass
        s = pd.Series(out, index=series.index).ffill()
        return s.clip(0, 1)

    result["hurst"] = sparse_hurst(c)

    # ── F10: Return Autocorrelation (lag-1, 20-bar rolling) ─────────────────────
    r_lag  = log_ret.shift(1)
    cov    = ((log_ret - log_ret.rolling(20).mean()) *
              (r_lag - r_lag.rolling(20).mean())).rolling(20).mean()
    result["ret_autocorr"] = (
        cov / (log_ret.rolling(20).std() * r_lag.rolling(20).std() + 1e-10)
    ).clip(-1, 1)

    # ── F11 + F12: BTC Cross-Asset Features ─────────────────────────────────────
    if btc_df is not None:
        btc_ret = np.log(btc_df["close"] / btc_df["close"].shift(1))
        btc_ret_aligned = btc_ret.reindex(df.index, method="ffill")
        result["btc_return"] = btc_ret_aligned.clip(-0.1, 0.1) / 0.1

        btc_cov = log_ret.rolling(20).cov(btc_ret_aligned)
        btc_corr = (btc_cov / (log_ret.rolling(20).std() * btc_ret_aligned.rolling(20).std() + 1e-10))
        result["btc_corr"] = btc_corr.clip(-1, 1)
    else:
        result["btc_return"] = 0.0
        result["btc_corr"]   = 0.0

    # ── F13: Funding Rate Proxy ──────────────────────────────────────────────────
    # 8-bar cumulative return: high = crowded longs, funding rate likely high
    result["funding_proxy"] = c.pct_change(8).clip(-0.3, 0.3) / 0.3

    # ── F14: OI Change Proxy ─────────────────────────────────────────────────────
    # Volume spike × price direction = new positions opening
    vol_norm = v / (v.rolling(20).mean() + 1e-10)
    result["oi_proxy"] = (vol_norm * np.sign(c.diff())).clip(-3, 3) / 3

    return result


# ─── HMM Helpers ───────────────────────────────────────────────────────────────

def state_map_3(model):
    """Map 3 raw HMM states → canonical: BULL=0, CHOP=2, BEAR=1 (sort by log-return mean)."""
    idx = np.argsort(model.means_[:, 0])[::-1]
    return {int(idx[0]): 0, int(idx[1]): 2, int(idx[2]): 1}


def margin_conf(probs):
    """Margin confidence: best_prob - 2nd_best_prob."""
    s = np.sort(probs, axis=1)[:, ::-1]
    return s[:, 0] - s[:, 1]


def ground_truth_labels(close_series, h=FORWARD_H):
    """Forward return-based labels: BULL=0, BEAR=1, CHOP=2."""
    fwd = np.log(close_series.shift(-h) / close_series)
    gt  = np.full(len(close_series), 2, dtype=int)  # CHOP default
    gt[fwd >  0.02] = 0   # BULL: >2% gain in 48h
    gt[fwd < -0.02] = 1   # BEAR: >2% loss in 48h
    return gt


def train_eval_hmm(feat_df, feature_cols, original_df,
                   n_train=N_TRAIN, n_test=N_TEST):
    """
    Train 3-state HMM on n_train bars, evaluate on next n_test bars.

    Returns dict: {acc, sharpe, margin_gap, hi_acc, lo_acc}
    or None if insufficient data / training failed.
    """
    try:
        sub = feat_df[feature_cols].copy()
        # Drop rows where any feature is NaN
        sub = sub.dropna()

        needed = n_train + n_test + FORWARD_H
        if len(sub) < needed:
            return None

        # Use the last `needed` bars (most recent data)
        sub = sub.iloc[-needed:]
        X_tr = sub.iloc[:n_train].values
        X_te = sub.iloc[n_train:n_train + n_test].values

        mu, std = X_tr.mean(0), X_tr.std(0)
        std[std < 1e-10] = 1e-10

        model = GaussianHMM(n_components=3, covariance_type="full",
                            n_iter=100, random_state=42)
        model.fit((X_tr - mu) / std)
        sm = state_map_3(model)

        X_te_s    = (X_te - mu) / std
        raw_preds = model.predict(X_te_s)
        preds     = np.array([sm.get(int(s), 2) for s in raw_preds])
        probs     = model.predict_proba(X_te_s)
        margin    = margin_conf(probs)

        # Ground truth for the test window
        test_close = original_df["close"].loc[sub.index[n_train:n_train + n_test + FORWARD_H]]
        gt = ground_truth_labels(test_close)[:n_test]

        # Accuracy
        acc = float(np.mean(preds == gt))

        # Margin calibration gap (top-20% vs bottom-20% confidence)
        hi_mask = margin > np.percentile(margin, 80)
        lo_mask = margin < np.percentile(margin, 20)
        hi_acc = float(np.mean(preds[hi_mask] == gt[hi_mask])) if hi_mask.sum() > 5 else np.nan
        lo_acc = float(np.mean(preds[lo_mask] == gt[lo_mask])) if lo_mask.sum() > 5 else np.nan
        mgap = (hi_acc - lo_acc) if not (np.isnan(hi_acc) or np.isnan(lo_acc)) else 0.0

        # Sharpe: margin-weighted directional signal vs forward return
        fwd_close = original_df["close"].loc[sub.index[n_train:n_train + n_test + FORWARD_H]]
        fwd_ret   = np.log(fwd_close.shift(-FORWARD_H) / fwd_close).values[:n_test]
        signal    = np.where(preds == 0, 1.0, np.where(preds == 1, -1.0, 0.0)) * margin
        port      = signal * fwd_ret
        sharpe    = float(port.mean() / (port.std() + 1e-10) * np.sqrt(252 / 4))

        return {"acc": acc, "sharpe": sharpe, "margin_gap": mgap,
                "hi_acc": hi_acc, "lo_acc": lo_acc}

    except Exception:
        return None


# ─── EXP 1: Feature Ablation ───────────────────────────────────────────────────

def exp1_feature_ablation(coin_dfs, all_feat_dfs):
    print("\n" + "═" * 82)
    print("  EXP 1 — HMM FEATURE ABLATION")
    print("  Baseline: log_return + volatility + volume_change + rsi_norm (4 features)")
    print("  Each new feature tested by adding it to the baseline.")
    print("═" * 82)

    # ── Baseline ──────────────────────────────────────────────────────────────
    bl_list = []
    for sym, df in coin_dfs.items():
        r = train_eval_hmm(all_feat_dfs[sym], BASELINE_FEATURES, df)
        if r:
            bl_list.append(r)

    if not bl_list:
        print("  ERROR: could not compute baseline results")
        return {}, {}

    baseline = {k: float(np.nanmean([r[k] for r in bl_list])) for k in bl_list[0]}
    print(f"\n  Baseline (4 features):  "
          f"Acc={baseline['acc']:.1%}  "
          f"Sharpe={baseline['sharpe']:.3f}  "
          f"MarginGap={baseline['margin_gap']:+.3f}")

    # ── Per-feature test ──────────────────────────────────────────────────────
    results = {}
    for feat, name in NEW_FEATURES_META.items():
        feat_list = []
        for sym, df in coin_dfs.items():
            fd = all_feat_dfs[sym]
            if feat not in fd.columns:
                continue
            r = train_eval_hmm(fd, BASELINE_FEATURES + [feat], df)
            if r:
                feat_list.append(r)

        if not feat_list:
            continue

        avg = {k: float(np.nanmean([r[k] for r in feat_list])) for k in feat_list[0]}
        avg["delta_sharpe"] = avg["sharpe"] - baseline["sharpe"]
        avg["delta_acc"]    = avg["acc"]    - baseline["acc"]
        avg["delta_mgap"]   = avg["margin_gap"] - baseline["margin_gap"]
        results[feat] = avg

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n  {'Feature':<30} {'Acc':>6} {'ΔAcc':>6} {'Sharpe':>8} {'ΔSharpe':>9} "
          f"{'ΔMarginGap':>12}  Verdict")
    print("  " + "─" * 82)

    for feat, r in sorted(results.items(), key=lambda x: x[1]["delta_sharpe"], reverse=True):
        name    = NEW_FEATURES_META[feat]
        verdict = ("✓ ADD"    if r["delta_sharpe"] >  0.05 else
                   "~ neutral" if r["delta_sharpe"] > -0.05 else
                   "✗ SKIP")
        print(f"  {name:<30} {r['acc']:>6.1%} {r['delta_acc']:>+6.1%} "
              f"{r['sharpe']:>8.3f} {r['delta_sharpe']:>+9.3f} "
              f"{r['delta_mgap']:>+12.3f}  {verdict}")

    return results, baseline


# ─── EXP 2: Greedy Feature Selection ───────────────────────────────────────────

def exp2_greedy_selection(coin_dfs, all_feat_dfs, exp1_results, baseline):
    print("\n" + "═" * 82)
    print("  EXP 2 — GREEDY FEATURE SELECTION")
    print("  Start from baseline, greedily add features that improve Sharpe by >0.02.")
    print("═" * 82)

    candidates = sorted(
        [(f, r["delta_sharpe"]) for f, r in exp1_results.items() if r["delta_sharpe"] > 0],
        key=lambda x: x[1], reverse=True,
    )

    if not candidates:
        print("  No features improved Sharpe. Baseline is already optimal.")
        return BASELINE_FEATURES.copy(), baseline["sharpe"]

    current_set    = BASELINE_FEATURES.copy()
    current_sharpe = baseline["sharpe"]
    added = []

    print(f"\n  Start: baseline 4 features  Sharpe={current_sharpe:.3f}")

    for feat, pre_delta in candidates[:10]:
        test_set = current_set + [feat]
        rlist = []
        for sym, df in coin_dfs.items():
            fd = all_feat_dfs[sym]
            if feat not in fd.columns:
                continue
            r = train_eval_hmm(fd, test_set, df)
            if r:
                rlist.append(r)

        if not rlist:
            continue

        new_sharpe = float(np.nanmean([r["sharpe"] for r in rlist]))
        delta      = new_sharpe - current_sharpe
        name       = NEW_FEATURES_META.get(feat, feat)

        if delta > 0.02:
            current_set    = test_set
            current_sharpe = new_sharpe
            added.append(feat)
            print(f"  + {name:<30}  Sharpe {new_sharpe:.3f}  ({delta:>+.3f})  ✓ ADDED")
        else:
            print(f"  ~ {name:<30}  Sharpe {new_sharpe:.3f}  ({delta:>+.3f})  ─ skipped")

    print(f"\n  ── Recommended Feature Set  ({len(current_set)} features, "
          f"Sharpe={current_sharpe:.3f}) ──")
    for f in current_set:
        tag = "  ← NEW" if f in added else ""
        print(f"    {f}{tag}")

    return current_set, current_sharpe


# ─── EXP 3: Conviction Factor IC Analysis ──────────────────────────────────────

def exp3_factor_ic(coin_dfs, all_feat_dfs, btc_df):
    print("\n" + "═" * 82)
    print("  EXP 3 — CONVICTION FACTOR IC ANALYSIS")
    print("  IC = Pearson correlation(factor signal, 48h forward return).")
    print("  t-stat > 1.5 = statistically meaningful signal.")
    print("═" * 82)

    FACTOR_LABELS = {
        "hmm":     "HMM Direction × Margin",
        "btc":     "BTC Macro (5-bar return)",
        "sr":      "SR/VWAP (RSI contrarian)",
        "vol":     "Volatility Quality (low=good)",
        "oi":      "OI Change Proxy",
        "funding": "Funding Rate Proxy (contrarian)",
        "adx":     "ADX × MACD direction",
        "macd":    "MACD Histogram",
        "hurst":   "Hurst × Recent Momentum",
        "obv":     "OBV Change",
    }

    # Current conviction weights (for reference)
    CURRENT_WEIGHTS = {
        "hmm": 22, "btc": 18, "sr": 10, "vol": 5,
        "oi": 8, "funding": 12,
        "adx": 0, "macd": 0, "hurst": 0, "obv": 0,
    }

    all_ics = {k: [] for k in FACTOR_LABELS}

    for sym, df in coin_dfs.items():
        fd  = all_feat_dfs[sym]
        fwd = np.log(df["close"].shift(-FORWARD_H) / df["close"])

        common = fd.index.intersection(fwd.dropna().index)
        if len(common) < 300:
            continue

        fwd_c = fwd.loc[common]
        fd_c  = fd.loc[common]

        # ── HMM signal ──────────────────────────────────────────────────────
        try:
            X = fd_c[BASELINE_FEATURES].dropna().values
            half = len(X) // 2
            if half < 200:
                raise ValueError
            mu, std = X[:half].mean(0), X[:half].std(0)
            std[std < 1e-10] = 1e-10
            X_s = (X - mu) / std
            m = GaussianHMM(n_components=3, covariance_type="full",
                            n_iter=100, random_state=42)
            m.fit(X_s[:half])
            sm     = state_map_3(m)
            probs  = m.predict_proba(X_s[half:])
            states = m.predict(X_s[half:])
            dir_   = np.array([{0: 1., 1: -1., 2: 0.}[sm.get(int(s), 2)] for s in states])
            hmm_sig = pd.Series(dir_ * margin_conf(probs),
                                index=fd_c.iloc[half:half + len(probs)].index)
            ic = float(hmm_sig.corr(fwd_c.loc[hmm_sig.index]))
            if not np.isnan(ic):
                all_ics["hmm"].append(ic)
        except Exception:
            pass

        # ── BTC macro ──────────────────────────────────────────────────────
        if btc_df is not None:
            btc_5 = np.log(btc_df["close"] / btc_df["close"].shift(5))
            btc_sig = np.sign(btc_5).reindex(common, method="ffill").fillna(0)
            ic = float(btc_sig.corr(fwd_c))
            if not np.isnan(ic):
                all_ics["btc"].append(ic)

        # ── SR / RSI contrarian ────────────────────────────────────────────
        if "rsi_norm" in fd_c:
            rsi  = fd_c["rsi_norm"]
            sr_s = pd.Series(
                np.where(rsi < -0.3, 1., np.where(rsi > 0.3, -1., 0.)),
                index=common)
            ic = float(sr_s.corr(fwd_c))
            if not np.isnan(ic):
                all_ics["sr"].append(ic)

        # ── Volatility quality (low vol = +1) ─────────────────────────────
        if "volatility" in fd_c:
            vol_rank = fd_c["volatility"].rank(pct=True)
            vol_s    = (1 - vol_rank * 2).fillna(0)
            ic = float(vol_s.corr(fwd_c))
            if not np.isnan(ic):
                all_ics["vol"].append(ic)

        # ── OI proxy ──────────────────────────────────────────────────────
        if "oi_proxy" in fd_c:
            ic = float(fd_c["oi_proxy"].corr(fwd_c))
            if not np.isnan(ic):
                all_ics["oi"].append(ic)

        # ── Funding proxy (contrarian) ────────────────────────────────────
        if "funding_proxy" in fd_c:
            ic = float((-fd_c["funding_proxy"]).corr(fwd_c))
            if not np.isnan(ic):
                all_ics["funding"].append(ic)

        # ── ADX × MACD direction ──────────────────────────────────────────
        if "adx" in fd_c and "macd_hist" in fd_c:
            adx_s = fd_c["adx"] * np.sign(fd_c["macd_hist"])
            ic = float(adx_s.corr(fwd_c))
            if not np.isnan(ic):
                all_ics["adx"].append(ic)

        # ── MACD histogram ────────────────────────────────────────────────
        if "macd_hist" in fd_c:
            ic = float(fd_c["macd_hist"].corr(fwd_c))
            if not np.isnan(ic):
                all_ics["macd"].append(ic)

        # ── Hurst × momentum ──────────────────────────────────────────────
        if "hurst" in fd_c and "log_return" in fd_c:
            mom   = fd_c["log_return"].rolling(5).sum().fillna(0)
            hurst_s = (fd_c["hurst"] - 0.5) * 2 * np.sign(mom)
            ic = float(hurst_s.corr(fwd_c))
            if not np.isnan(ic):
                all_ics["hurst"].append(ic)

        # ── OBV ───────────────────────────────────────────────────────────
        if "obv_norm" in fd_c:
            ic = float(fd_c["obv_norm"].corr(fwd_c))
            if not np.isnan(ic):
                all_ics["obv"].append(ic)

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n  {'Factor':<36} {'AvgIC':>8} {'StdIC':>8} {'N':>4} "
          f"{'t-stat':>8} {'Curr Wt':>9}  Signal?")
    print("  " + "─" * 82)

    ic_summary = {}
    for key, label in FACTOR_LABELS.items():
        ics = all_ics[key]
        if not ics:
            print(f"  {label:<36}  {'n/a':>8}")
            continue
        avg_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        n      = len(ics)
        t      = avg_ic / (std_ic / np.sqrt(n) + 1e-10)
        sig    = "✓ YES" if abs(t) > 1.5 else "~ weak"
        cw     = CURRENT_WEIGHTS.get(key, 0)
        print(f"  {label:<36} {avg_ic:>+8.4f} {std_ic:>8.4f} {n:>4} "
              f"{t:>+8.2f} {cw:>9}  {sig}")
        ic_summary[key] = (avg_ic, std_ic, n, t, cw)

    # ── IC-proportional weight recommendations ────────────────────────────────
    total_abs = sum(abs(v[0]) for v in ic_summary.values()) + 1e-10
    print(f"\n  ── IC-Proportional Weight Recommendations ──")
    print(f"  (Scaled to 75 pts; remaining ~25 pts for Sentiment + OrderFlow)")
    print(f"  {'Factor':<36} {'IC':>8} {'Rec Wt':>8} {'Curr Wt':>9}  Change")
    print("  " + "─" * 70)

    for key, label in FACTOR_LABELS.items():
        if key not in ic_summary:
            continue
        avg_ic, _, _, _, curr_w = ic_summary[key]
        rec_w  = round(abs(avg_ic) / total_abs * 75, 1)
        delta  = rec_w - curr_w
        arrow  = "↑ increase" if delta > 2 else ("↓ decrease" if delta < -2 else "→ keep")
        print(f"  {label:<36} {avg_ic:>+8.4f} {rec_w:>8.1f} {curr_w:>9}  {arrow}")

    return ic_summary


# ─── EXP 4: Conviction Weight Optimization ─────────────────────────────────────

def exp4_weight_tuning(coin_dfs, all_feat_dfs, btc_df, ic_summary):
    print("\n" + "═" * 82)
    print(f"  EXP 4 — CONVICTION WEIGHT OPTIMIZATION  ({WEIGHT_TRIALS} random trials)")
    print("  6 computable factors: HMM | BTC | SR | Vol | OI | Funding")
    print("  (Sentiment + OrderFlow excluded — need live exchange data)")
    print("═" * 82)

    FACTORS    = ["hmm", "btc", "sr", "vol", "oi", "funding"]
    CURR_W_PTS = np.array([22, 18, 10, 5, 8, 12], dtype=float)
    CURR_W_NRM = CURR_W_PTS / CURR_W_PTS.sum()

    # ── Precompute per-coin factor signals ────────────────────────────────────
    signals_by_coin = {}
    for sym, df in coin_dfs.items():
        fd  = all_feat_dfs[sym]
        fwd = np.log(df["close"].shift(-FORWARD_H) / df["close"])

        try:
            X = fd[BASELINE_FEATURES].dropna().values
            half = len(X) // 2
            if half < 200:
                continue
            mu, std = X[:half].mean(0), X[:half].std(0)
            std[std < 1e-10] = 1e-10
            X_s = (X - mu) / std
            m = GaussianHMM(n_components=3, covariance_type="full",
                            n_iter=100, random_state=42)
            m.fit(X_s[:half])
            sm     = state_map_3(m)
            idx    = fd.iloc[half:half + len(X_s[half:])].index
            probs  = m.predict_proba(X_s[half:])
            states = m.predict(X_s[half:])
            dir_   = np.array([{0: 1., 1: -1., 2: 0.}[sm.get(int(s), 2)] for s in states])
            hmm_s  = pd.Series(dir_ * margin_conf(probs), index=idx)

            btc_s  = pd.Series(0., index=idx)
            if btc_df is not None:
                btc_5 = np.log(btc_df["close"] / btc_df["close"].shift(5))
                btc_s = np.sign(btc_5).reindex(idx, method="ffill").fillna(0)

            rsi   = fd["rsi_norm"].reindex(idx)
            sr_s  = pd.Series(
                np.where(rsi < -0.3, 1., np.where(rsi > 0.3, -1., 0.)), index=idx
            ).fillna(0)

            vol_rank = fd["volatility"].reindex(idx).rank(pct=True)
            vol_s    = (1 - vol_rank * 2).fillna(0)

            oi_s = (fd["oi_proxy"].reindex(idx).fillna(0)
                    if "oi_proxy" in fd.columns else pd.Series(0., index=idx))

            fund_s = (-fd["funding_proxy"].reindex(idx).fillna(0)
                      if "funding_proxy" in fd.columns else pd.Series(0., index=idx))

            fwd_c = fwd.reindex(idx).fillna(0)

            signals_by_coin[sym] = pd.DataFrame({
                "hmm": hmm_s, "btc": btc_s, "sr": sr_s,
                "vol": vol_s, "oi": oi_s, "funding": fund_s,
                "fwd": fwd_c,
            })
        except Exception:
            continue

    if not signals_by_coin:
        print("  ERROR: no factor signals computed — check data availability")
        return None, FACTORS

    # ── Sharpe objective ──────────────────────────────────────────────────────
    def portfolio_sharpe(w):
        all_ret = []
        for sym, sdf in signals_by_coin.items():
            combo = sum(w[i] * sdf[f] for i, f in enumerate(FACTORS))
            combo = combo.clip(-1, 1)
            all_ret.extend((combo * sdf["fwd"]).dropna().tolist())
        if not all_ret:
            return -np.inf
        pr = np.array(all_ret)
        return float(pr.mean() / (pr.std() + 1e-10) * np.sqrt(252 / 4))

    baseline_sharpe = portfolio_sharpe(CURR_W_NRM)
    print(f"\n  Current weights (pts): {dict(zip(FACTORS, CURR_W_PTS.astype(int)))}  "
          f"Sharpe={baseline_sharpe:.4f}")

    # ── IC-guided alpha for Dirichlet sampling ────────────────────────────────
    ic_alpha = np.array([
        abs(ic_summary.get(f, (0.1,))[0]) + 0.05
        for f in FACTORS
    ])

    np.random.seed(42)
    best_sharpe = baseline_sharpe
    best_w      = CURR_W_NRM.copy()
    all_trials  = []

    for i in range(WEIGHT_TRIALS):
        if i < 100:
            w = np.random.dirichlet(np.ones(len(FACTORS)))          # pure random
        elif i < 200:
            w = np.random.dirichlet(ic_alpha * 5)                   # IC-guided
        else:
            noise = np.random.normal(0, 0.04, len(FACTORS))
            w     = np.abs(best_w + noise)
            w    /= w.sum()                                          # perturb best

        s = portfolio_sharpe(w)
        all_trials.append((w.copy(), s))
        if s > best_sharpe:
            best_sharpe = s
            best_w      = w.copy()

    all_trials.sort(key=lambda x: x[1], reverse=True)

    # ── Print top-5 ───────────────────────────────────────────────────────────
    print(f"\n  ── Top 5 Weight Combinations (scaled to 75 pts) ──")
    hdr = "  " + "".join(f"{f:>10}" for f in FACTORS) + "   Sharpe"
    print(hdr)
    print("  " + "─" * 75)
    for w, s in all_trials[:5]:
        w_pts = (w * 75).round(1)
        row   = "  " + "".join(f"{x:>10.1f}" for x in w_pts) + f"   {s:.4f}"
        print(row)

    # ── Current vs optimal ────────────────────────────────────────────────────
    print(f"\n  ── Current vs Optimal Weights (scaled to 75 pts) ──")
    print(f"  {'Factor':<22} {'Current':>9} {'Optimal':>9} {'Δ':>7}  Direction")
    print("  " + "─" * 60)

    for i, f in enumerate(FACTORS):
        curr_pts = round(CURR_W_NRM[i] * 75, 1)
        opt_pts  = round(best_w[i] * 75, 1)
        delta    = opt_pts - curr_pts
        arrow    = "↑ increase" if delta > 1.5 else ("↓ decrease" if delta < -1.5 else "→ keep")
        print(f"  {f:<22} {curr_pts:>9.1f} {opt_pts:>9.1f} {delta:>+7.1f}  {arrow}")

    improvement = best_sharpe - baseline_sharpe
    print(f"\n  Sharpe:  {baseline_sharpe:.4f}  →  {best_sharpe:.4f}  "
          f"({improvement:>+.4f}  {'IMPROVEMENT' if improvement > 0 else 'no improvement'})")

    # ── Final recommendation block ────────────────────────────────────────────
    print(f"\n  ── RECOMMENDED WEIGHTS (75 pts total; +Sentiment 15 + OrderFlow 10 = 100) ──")
    for i, f in enumerate(FACTORS):
        opt = int(round(best_w[i] * 75))
        print(f"    {f.upper():<12}: {opt} pts")
    print(f"    SENTIMENT   : 15 pts  (unchanged — not testable from OHLCV)")
    print(f"    ORDERFLOW   : 10 pts  (unchanged — not testable from OHLCV)")

    return best_w, FACTORS


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 82)
    print("  HMMBOT FEATURE + CONVICTION WEIGHT EXPERIMENT")
    print("  Pure analysis — zero production code changes")
    print("═" * 82)

    # ── Fetch data ────────────────────────────────────────────────────────────
    print("\n  Fetching 4h OHLCV data from Binance (2022-06-01 → 2026-03-01)...")
    btc_df = fetch_ohlcv("BTCUSDT")
    if btc_df is None:
        print("  ERROR: could not fetch BTC reference data")
        btc_df = None

    coin_dfs = {}
    for sym in TEST_COINS:
        df = fetch_ohlcv(sym)
        min_bars = N_TRAIN + N_TEST + FORWARD_H + 200
        if df is not None and len(df) >= min_bars:
            coin_dfs[sym] = df
            print(f"  ✓ {sym}: {len(df)} bars")
        else:
            bars = len(df) if df is not None else 0
            print(f"  ✗ {sym}: {bars} bars — skip")

    if not coin_dfs:
        print("\n  ERROR: no coins loaded. Check Binance API connection.")
        return

    print(f"\n  {len(coin_dfs)} coins loaded. Computing all features...")

    # ── Compute features for all coins once (reused across experiments) ───────
    all_feat_dfs = {}
    for sym, df in coin_dfs.items():
        fd = compute_features(df, btc_df if sym != "BTCUSDT" else None)
        all_feat_dfs[sym] = fd

    # ── Run experiments ───────────────────────────────────────────────────────
    exp1_results, baseline = exp1_feature_ablation(coin_dfs, all_feat_dfs)
    exp2_greedy_selection(coin_dfs, all_feat_dfs, exp1_results, baseline)
    ic_summary = exp3_factor_ic(coin_dfs, all_feat_dfs, btc_df)
    exp4_weight_tuning(coin_dfs, all_feat_dfs, btc_df, ic_summary)

    print("\n" + "═" * 82)
    print("  EXPERIMENT COMPLETE")
    print("  Review all 4 tables above → give LGTM → then production changes applied.")
    print("═" * 82)


if __name__ == "__main__":
    main()
