"""
tools/backtest_fronttest_100coins.py

Comprehensive evaluation of the AFTER (4-feature fixed) HMM across 100 coins.
Measures performance on 3 distinct periods to diagnose overfitting/underfitting.

  TRAIN    : Jan 2024 – Jun 2024  (HMM trained here — never evaluated)
  BACKTEST : Jul 2024 – Dec 2024  (in-distribution OOS, 6 months)
  FORWARD  : Jan 2025 – Mar 2026  (true forward test, ~15 months OOS)

Overfitting signals:
  • Backtest accuracy >> Forward accuracy  (>10pp gap)
  • Backtest Sharpe   >> Forward Sharpe    (>0.5 gap)
  • High confidence but low accuracy       (overconfident)
  • Large cross-coin variance in forward   (coin-specific patterns)

Underfitting signals:
  • All periods accuracy ≈ 25%            (random baseline for 4 classes)
  • Sharpe ≈ 0 everywhere
  • Model assigns same regime to everything

Usage: python tools/backtest_fronttest_100coins.py
       python tools/backtest_fronttest_100coins.py --top 50
"""
import sys
import os
import time
import argparse
import csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import config
from data_pipeline import _parse_klines_df
from feature_engine import compute_hmm_features

# ── Settings ──────────────────────────────────────────────────────────────────
DEFAULT_TOP_N    = 100
INTERVAL         = "4h"          # 6 bars/day → 1000 candles ≈ 166 days
TRAIN_START      = "1 Jan, 2024"
TRAIN_CUTOFF     = "2024-07-01"  # end of training period
BACKTEST_CUTOFF  = "2025-01-01"  # end of backtest period (forward test starts here)
HMM_FEATURES     = ["log_return", "volatility", "volume_change", "rsi_norm"]
BARS_PER_YEAR    = 2190          # 4h bars in a year (for Sharpe annualisation)
RANDOM_BASELINE  = 1 / config.HMM_N_STATES   # 25% for 4-state model
MIN_TRAIN_BARS   = 200           # minimum required to fit HMM
MIN_TEST_BARS    = 30            # minimum required for evaluation
SLEEP_COINS      = 0.8           # seconds between coins to respect Binance rate limit
FWD_BARS         = 12            # candles ahead for ground truth (12×4h = 48h horizon)
OUTPUT_CSV       = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eval_100coins.csv"
)

# Coins to always exclude (stablecoins, wrapped, leveraged)
COIN_EXCLUDE = {
    "EURUSDT", "WBTCUSDT", "USDCUSDT", "TUSDUSDT", "BUSDUSDT",
    "USTUSDT", "DAIUSDT", "FDUSDUSDT", "CVCUSDT", "USD1USDT",
}
LEV_KEYWORDS = ("UP", "DOWN", "BULL", "BEAR")


# ── Step 1: Get top-N coins by volume ─────────────────────────────────────────

def get_top_coins(n):
    from binance.client import Client
    client = Client(tld="com")
    print(f"  Fetching top-{n} USDT coins by 24h volume from Binance...")
    try:
        tickers = client.get_ticker()
    except Exception as e:
        print(f"  ERROR fetching tickers: {e}")
        return []

    usdt = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and t["symbol"] not in COIN_EXCLUDE
        and not any(kw in t["symbol"].replace("USDT", "") for kw in LEV_KEYWORDS)
    ]
    usdt.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
    symbols = [t["symbol"] for t in usdt[:n]]
    print(f"  Got {len(symbols)} symbols  (top: {symbols[:5]}...)")
    return symbols


# ── Step 2: Fetch historical klines ───────────────────────────────────────────

def fetch_klines(symbol, interval, start_date):
    from binance.client import Client
    interval_map = {
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
    }
    client = Client(tld="com")
    try:
        klines = client.get_historical_klines(
            symbol, interval_map[interval], start_date
        )
        if not klines:
            return None
        return _parse_klines_df(klines)
    except Exception as e:
        print(f"    FETCH ERROR {symbol}: {e}")
        return None


# ── Step 3: Ground truth labels ───────────────────────────────────────────────

def ground_truth_labels(df, fwd=FWD_BARS):
    """
    Post-hoc regime labels based on forward realised returns (for evaluation only,
    never used in HMM training).

    Thresholds (4h bars, fwd=12 → 48h horizon):
      CRASH : fwd_ret < -6%  OR  (fwd_ret < -2% AND fwd_vol > 4%)
      BEAR  : fwd_ret < -2%
      BULL  : fwd_ret > +3%  AND fwd_vol < 2.5%
      CHOP  : else
      -1    : unknown (end of series)
    """
    lr = np.log(df["close"] / df["close"].shift(1))
    fwd_ret = lr.rolling(fwd).sum().shift(-fwd)
    fwd_vol = lr.rolling(fwd).std().shift(-fwd)
    n = len(df)
    labels = np.full(n, config.REGIME_CHOP, dtype=int)
    for i in range(n):
        r = fwd_ret.iloc[i];  v = fwd_vol.iloc[i]
        if pd.isna(r) or pd.isna(v):
            labels[i] = -1
        elif r < -0.06 or (r < -0.02 and v > 0.04):
            labels[i] = config.REGIME_CRASH
        elif r < -0.02:
            labels[i] = config.REGIME_BEAR
        elif r > 0.03 and v < 0.025:
            labels[i] = config.REGIME_BULL
    return labels


# ── Step 4: HMM training ──────────────────────────────────────────────────────

def train_hmm(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = GaussianHMM(
        n_components=config.HMM_N_STATES,
        covariance_type="full",
        n_iter=config.HMM_ITERATIONS,
        random_state=42,
        tol=1e-4,
    )
    model.fit(Xs)
    return model, scaler


def build_state_map(model):
    """Sort states by means[:,0] (log_return) descending → BULL/BEAR/CHOP/CRASH."""
    order = np.argsort(model.means_[:, 0])[::-1]
    canonical = [config.REGIME_BULL, config.REGIME_BEAR,
                 config.REGIME_CHOP, config.REGIME_CRASH]
    return {int(raw): canonical[i] for i, raw in enumerate(order)}


def predict(model, scaler, state_map, X):
    Xs = scaler.transform(X)
    raw = model.predict(Xs)
    prob = model.predict_proba(Xs)
    states = np.array([state_map[s] for s in raw], dtype=int)
    confs  = np.array([prob[i, s] for i, s in enumerate(raw)])
    return states, confs


# ── Step 5: Metrics for one period ────────────────────────────────────────────

REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "CHOP", 3: "CRASH"}

def period_metrics(preds, confs, true, log_returns_next):
    """
    preds, confs, true : aligned arrays of same length
    log_returns_next   : log return of the next bar (trading P&L)
    """
    if len(preds) == 0:
        return None

    # Exclude unknown ground truth
    mask = true != -1
    if mask.sum() < MIN_TEST_BARS:
        return None

    p = preds[mask]; c = confs[mask]; t = true[mask]
    lr = log_returns_next[mask]

    # Regime label accuracy
    acc = np.mean(p == t)

    # Per-regime accuracy
    per_acc = {}
    for r, name in REGIME_NAMES.items():
        rm = t == r
        per_acc[name] = float(np.mean(p[rm] == r)) if rm.sum() > 0 else float("nan")

    # Strategy P&L (BULL=long, BEAR/CRASH=short, CHOP=flat)
    strat = np.where(p == config.REGIME_BULL, lr,
            np.where((p == config.REGIME_BEAR) | (p == config.REGIME_CRASH), -lr,
            0.0))
    total_pnl  = float(np.sum(strat))
    win_rate   = float(np.mean(strat > 0)) if len(strat) > 0 else float("nan")
    sharpe     = (float(strat.mean() / strat.std()) * np.sqrt(BARS_PER_YEAR)
                  if strat.std() > 1e-10 else 0.0)

    # Mean confidence
    mean_conf = float(c.mean())

    # Confidence calibration buckets
    calib = {}
    for lo, hi in [(0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 0.99), (0.99, 1.01)]:
        bm = (c >= lo) & (c < hi)
        if bm.sum() >= 5:
            calib[f"{lo:.0%}-{hi:.0%}"] = {
                "n": int(bm.sum()),
                "accuracy": float(np.mean(p[bm] == t[bm])),
            }

    # Regime distribution in predictions
    dist = {name: float(np.mean(p == r)) for r, name in REGIME_NAMES.items()}

    return {
        "n": int(mask.sum()),
        "accuracy": float(acc),
        "per_acc": per_acc,
        "sharpe": sharpe,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "mean_conf": mean_conf,
        "dist": dist,
        "calib": calib,
    }


# ── Step 6: Evaluate one coin ─────────────────────────────────────────────────

def evaluate_coin(symbol, df_raw):
    """
    Returns dict with 'train', 'backtest', 'forward' metrics,
    or None if data is insufficient.
    """
    df = compute_hmm_features(df_raw)
    gt  = ground_truth_labels(df)
    lr  = np.log(df["close"] / df["close"].shift(1)).fillna(0).values

    # Date-based splits
    ts = df["timestamp"]
    train_mask    = ts <  TRAIN_CUTOFF
    backtest_mask = (ts >= TRAIN_CUTOFF) & (ts < BACKTEST_CUTOFF)
    forward_mask  = ts >= BACKTEST_CUTOFF

    # Training data
    train_df = df[HMM_FEATURES][train_mask].dropna()
    if len(train_df) < MIN_TRAIN_BARS:
        return None   # not enough train data (coin listed after Jan 2024)

    try:
        model, scaler = train_hmm(train_df.values)
    except Exception:
        return None

    state_map = build_state_map(model)

    result = {}
    for period_name, pmask in [("train",    train_mask),
                                ("backtest", backtest_mask),
                                ("forward",  forward_mask)]:
        period_df  = df[HMM_FEATURES][pmask].dropna()
        if len(period_df) < MIN_TEST_BARS:
            result[period_name] = None
            continue

        idx     = period_df.index
        preds, confs = predict(model, scaler, state_map, period_df.values)
        true_lbl     = gt[idx]
        # Next-bar log return (for P&L)
        next_lr = np.array([lr[i + 1] if i + 1 < len(lr) else 0.0 for i in idx])

        result[period_name] = period_metrics(preds, confs, true_lbl, next_lr)

    # Model diagnostics: means per state
    result["model_means"] = {
        REGIME_NAMES[state_map[r]]: {
            "log_return": float(model.means_[r, 0]),
            "volatility": float(model.means_[r, 1]),
        }
        for r in range(config.HMM_N_STATES)
    }
    return result


# ── Step 7: Aggregate and diagnose ────────────────────────────────────────────

def agg(results, period, key, sub=None):
    """Collect metric values across all coins for a given period."""
    vals = []
    for r in results.values():
        if r is None or r.get(period) is None:
            continue
        v = r[period].get(key) if sub is None else r[period][key].get(sub)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return np.array(vals)


def confidence_calibration_agg(results, period):
    """Aggregate calibration data across all coins for a period."""
    merged = {}
    for r in results.values():
        if r is None or r.get(period) is None:
            continue
        for bucket, data in r[period]["calib"].items():
            if bucket not in merged:
                merged[bucket] = {"n": 0, "correct": 0}
            merged[bucket]["n"]       += data["n"]
            merged[bucket]["correct"] += int(data["n"] * data["accuracy"])
    return {b: {"n": d["n"], "accuracy": d["correct"] / d["n"]}
            for b, d in merged.items() if d["n"] > 0}


def diagnose_overfitting(bt_acc, fw_acc, bt_sh, fw_sh,
                         bt_conf, fw_conf, fw_acc_arr, random_baseline=0.25):
    """Return a verdict string and list of evidence points."""
    evidence = []
    score    = 0   # positive → overfitting, negative → underfitting

    acc_gap = bt_acc - fw_acc
    sh_gap  = bt_sh  - fw_sh

    if acc_gap > 12:
        evidence.append(f"  [OVERFIT]  Accuracy drops {acc_gap:.1f}pp from backtest to forward (>12pp threshold)")
        score += 2
    elif acc_gap > 6:
        evidence.append(f"  [MILD-OF]  Accuracy drops {acc_gap:.1f}pp backtest→forward (6-12pp range)")
        score += 1
    else:
        evidence.append(f"  [OK]       Accuracy gap backtest→forward = {acc_gap:.1f}pp (<6pp is good)")

    if sh_gap > 0.8:
        evidence.append(f"  [OVERFIT]  Sharpe drops {sh_gap:.2f} backtest→forward (>0.8 threshold)")
        score += 2
    elif sh_gap > 0.4:
        evidence.append(f"  [MILD-OF]  Sharpe drops {sh_gap:.2f} backtest→forward (0.4-0.8 range)")
        score += 1
    else:
        evidence.append(f"  [OK]       Sharpe gap = {sh_gap:.2f} (<0.4 is good)")

    if fw_acc < random_baseline + 0.02:
        evidence.append(f"  [UNDERFIT] Forward accuracy {fw_acc:.1%} ≈ random baseline {random_baseline:.0%}")
        score -= 2
    elif fw_acc < random_baseline + 0.05:
        evidence.append(f"  [MILD-UF]  Forward accuracy {fw_acc:.1%} only marginally above random {random_baseline:.0%}")
        score -= 1
    else:
        evidence.append(f"  [OK]       Forward accuracy {fw_acc:.1%} meaningfully above random {random_baseline:.0%}")

    conf_gap = bt_conf - fw_conf
    if conf_gap > 0.03:
        evidence.append(f"  [OVERFIT]  Confidence drops {conf_gap:.1%} in forward test → distribution shift")
        score += 1

    pct_pos_sharpe = np.mean(fw_acc_arr > random_baseline) if len(fw_acc_arr) > 0 else 0
    evidence.append(f"  [INFO]     {pct_pos_sharpe:.0%} of coins beat random baseline in forward test")

    std_fw = np.std(fw_acc_arr) if len(fw_acc_arr) > 0 else 0
    if std_fw > 0.12:
        evidence.append(f"  [OVERFIT]  High cross-coin accuracy variance ({std_fw:.1%}) → coin-specific patterns")
        score += 1
    else:
        evidence.append(f"  [OK]       Cross-coin accuracy variance = {std_fw:.1%} (consistent generalisation)")

    if score >= 4:
        verdict = "STRONG OVERFITTING"
    elif score >= 2:
        verdict = "MILD OVERFITTING"
    elif score <= -2:
        verdict = "UNDERFITTING"
    elif score <= -1:
        verdict = "MILD UNDERFITTING"
    else:
        verdict = "GOOD GENERALISATION"

    return verdict, evidence


# ── Step 8: Report ────────────────────────────────────────────────────────────

def print_report(all_results, n_coins_attempted):
    valid = {s: r for s, r in all_results.items() if r is not None}
    n_valid = len(valid)
    W = 76

    print(f"\n{'═'*W}")
    print("  REGIME-MASTER HMM — 100-COIN EVALUATION REPORT")
    print(f"  Train: Jan–Jun 2024 | Backtest: Jul–Dec 2024 | Forward: Jan 2025–Mar 2026")
    print(f"  Model: 4-feature Gaussian HMM  |  Evaluated: {n_valid}/{n_coins_attempted} coins")
    print(f"{'═'*W}")

    # ── Aggregate table ───────────────────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  AGGREGATE RESULTS  (medians across {n_valid} coins)")
    print(f"  {'─'*W}")
    header = f"  {'Metric':<28} {'IN-SAMPLE':>12} {'BACKTEST':>12} {'FORWARD':>12} {'BT→FW Δ':>10}"
    print(header)
    print(f"  {'─'*W}")

    def med(period, key, sub=None):
        a = agg(valid, period, key, sub)
        return float(np.median(a)) if len(a) > 0 else float("nan")

    rows = [
        ("Label Accuracy",      "accuracy",  None,  "%",  ".1%"),
        ("Strategy P&L",        "total_pnl", None,  "%",  ".1%"),
        ("Sharpe Ratio",        "sharpe",    None,  "",   ".2f"),
        ("Win Rate",            "win_rate",  None,  "%",  ".1%"),
        ("Mean Confidence",     "mean_conf", None,  "%",  ".1%"),
    ]
    for label, key, sub, sfx, fmt in rows:
        IS = med("train",    key, sub)
        BT = med("backtest", key, sub)
        FW = med("forward",  key, sub)
        delta = FW - BT if not (np.isnan(FW) or np.isnan(BT)) else float("nan")
        is_s  = f"{IS:{fmt}}{sfx}" if not np.isnan(IS) else "N/A"
        bt_s  = f"{BT:{fmt}}{sfx}" if not np.isnan(BT) else "N/A"
        fw_s  = f"{FW:{fmt}}{sfx}" if not np.isnan(FW) else "N/A"
        dl_s  = f"{delta:+{fmt}}{sfx}" if not np.isnan(delta) else "N/A"
        print(f"  {label:<28} {is_s:>12} {bt_s:>12} {fw_s:>12} {dl_s:>10}")

    # Per-regime accuracy
    print(f"\n  Per-regime accuracy (median):")
    for r, name in REGIME_NAMES.items():
        BT = med("backtest", "per_acc", name)
        FW = med("forward",  "per_acc", name)
        delta = FW - BT if not (np.isnan(FW) or np.isnan(BT)) else float("nan")
        bt_s = f"{BT:.1%}" if not np.isnan(BT) else "N/A"
        fw_s = f"{FW:.1%}" if not np.isnan(FW) else "N/A"
        dl_s = f"{delta:+.1%}" if not np.isnan(delta) else "N/A"
        print(f"    {name:<10} BT={bt_s:>8}  FW={fw_s:>8}  Δ={dl_s:>8}")

    # ── Ground truth distribution ─────────────────────────────────────────────
    print(f"\n  Ground truth regime distribution (forward period, median % across coins):")
    # We infer this from per_acc denominators (approximation via dist in backtest/forward)
    for period in ("backtest", "forward"):
        dists = []
        for r in valid.values():
            if r and r.get(period):
                dists.append(r[period]["dist"])
        if dists:
            avg_d = {name: np.median([d.get(name, 0) for d in dists])
                     for name in REGIME_NAMES.values()}
            pname = "Backtest" if period == "backtest" else "Forward "
            print(f"    {pname}: " +
                  "  ".join(f"{n}={v:.0%}" for n, v in avg_d.items()))

    # ── Confidence calibration ────────────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print("  CONFIDENCE CALIBRATION  (forward test, all coins pooled)")
    print(f"  {'─'*W}")
    print(f"  {'Confidence bucket':<22} {'N samples':>10} {'Actual acc':>12} {'Expected':>12} {'Overconfident?':>14}")
    calib = confidence_calibration_agg(valid, "forward")
    for bucket in sorted(calib.keys()):
        d    = calib[bucket]
        lo   = float(bucket.split("-")[0].strip("%")) / 100
        hi   = float(bucket.split("-")[1].strip("%")) / 100
        exp  = (lo + hi) / 2
        diff = d["accuracy"] - exp
        flag = "YES ⚠" if d["accuracy"] < exp - 0.05 else ("ok" if abs(diff) < 0.05 else "underconf")
        print(f"  {bucket:<22} {d['n']:>10,} {d['accuracy']:>11.1%} {exp:>11.1%} {flag:>14}")

    # ── Overfitting diagnosis ─────────────────────────────────────────────────
    bt_acc  = med("backtest", "accuracy")
    fw_acc  = med("forward",  "accuracy")
    bt_sh   = med("backtest", "sharpe")
    fw_sh   = med("forward",  "sharpe")
    bt_conf = med("backtest", "mean_conf")
    fw_conf = med("forward",  "mean_conf")
    fw_acc_arr = agg(valid, "forward", "accuracy")

    verdict, evidence = diagnose_overfitting(
        bt_acc, fw_acc, bt_sh, fw_sh, bt_conf, fw_conf, fw_acc_arr
    )

    print(f"\n  {'─'*W}")
    print("  OVERFITTING / UNDERFITTING DIAGNOSIS")
    print(f"  {'─'*W}")
    for line in evidence:
        print(f"  {line}")
    print(f"\n  ┌{'─'*(W-2)}┐")
    print(f"  │  VERDICT: {verdict:<{W-14}}│")
    print(f"  └{'─'*(W-2)}┘")

    # ── Top / bottom coins ────────────────────────────────────────────────────
    coin_fwd = []
    for sym, r in valid.items():
        if r and r.get("forward"):
            coin_fwd.append((sym, r["forward"]["sharpe"], r["forward"]["accuracy"],
                             r["forward"]["total_pnl"]))
    coin_fwd.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'─'*W}")
    print(f"  TOP 10 coins in FORWARD test (by Sharpe)")
    print(f"  {'─'*W}")
    print(f"  {'Symbol':<14} {'Sharpe':>8} {'Accuracy':>10} {'P&L':>10}")
    for sym, sh, acc, pnl in coin_fwd[:10]:
        print(f"  {sym:<14} {sh:>8.2f} {acc:>9.1%} {pnl:>+9.1%}")

    print(f"\n  BOTTOM 10 coins in FORWARD test")
    print(f"  {'─'*W}")
    print(f"  {'Symbol':<14} {'Sharpe':>8} {'Accuracy':>10} {'P&L':>10}")
    for sym, sh, acc, pnl in coin_fwd[-10:]:
        print(f"  {sym:<14} {sh:>8.2f} {acc:>9.1%} {pnl:>+9.1%}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    try:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "period", "n", "accuracy", "sharpe",
                             "total_pnl", "win_rate", "mean_conf",
                             "bull_acc", "bear_acc", "chop_acc", "crash_acc"])
            for sym, r in valid.items():
                for period in ("train", "backtest", "forward"):
                    m = r.get(period)
                    if m is None:
                        continue
                    writer.writerow([
                        sym, period, m["n"],
                        round(m["accuracy"], 4),
                        round(m["sharpe"], 4),
                        round(m["total_pnl"], 4),
                        round(m["win_rate"], 4),
                        round(m["mean_conf"], 4),
                        round(m["per_acc"].get("BULL",  float("nan")), 4),
                        round(m["per_acc"].get("BEAR",  float("nan")), 4),
                        round(m["per_acc"].get("CHOP",  float("nan")), 4),
                        round(m["per_acc"].get("CRASH", float("nan")), 4),
                    ])
        print(f"\n  Full results saved → {OUTPUT_CSV}")
    except Exception as e:
        print(f"\n  CSV save failed: {e}")

    print(f"\n{'═'*W}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(top_n=DEFAULT_TOP_N):
    print(f"\n{'═'*76}")
    print(f"  BACKTEST + FORWARD TEST: TOP {top_n} COINS")
    print(f"  Train: {TRAIN_START} → {TRAIN_CUTOFF}")
    print(f"  Backtest: {TRAIN_CUTOFF} → {BACKTEST_CUTOFF}")
    print(f"  Forward:  {BACKTEST_CUTOFF} → now")
    print(f"{'═'*76}\n")

    symbols = get_top_coins(top_n)
    if not symbols:
        print("Failed to fetch coin list.")
        return

    all_results = {}
    n_ok = 0; n_skip = 0

    for i, sym in enumerate(symbols):
        print(f"  [{i+1:>3}/{len(symbols)}] {sym:<14}", end="", flush=True)

        df_raw = fetch_klines(sym, INTERVAL, TRAIN_START)
        if df_raw is None or len(df_raw) < MIN_TRAIN_BARS + MIN_TEST_BARS:
            print("  SKIP (no data)")
            all_results[sym] = None
            n_skip += 1
            time.sleep(SLEEP_COINS)
            continue

        result = evaluate_coin(sym, df_raw)
        all_results[sym] = result

        if result is None:
            print("  SKIP (insufficient train bars)")
            n_skip += 1
        else:
            fw = result.get("forward")
            bt = result.get("backtest")
            if fw and bt:
                print(f"  BT acc={bt['accuracy']:.1%} sh={bt['sharpe']:+.2f}  "
                      f"FW acc={fw['accuracy']:.1%} sh={fw['sharpe']:+.2f}  "
                      f"n_fw={fw['n']}")
                n_ok += 1
            else:
                print("  partial data")

        time.sleep(SLEEP_COINS)

    print(f"\n  Done: {n_ok} coins evaluated, {n_skip} skipped.\n")
    print_report(all_results, len(symbols))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                        help="Number of top coins to evaluate")
    args = parser.parse_args()
    # Suppress hmmlearn convergence warnings
    import logging
    logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
    run(top_n=args.top)
