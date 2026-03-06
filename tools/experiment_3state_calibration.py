"""
tools/experiment_3state_calibration.py

Three experiments running in one script:

  EXP 1 — Coin categorisation
    Reads eval_100coins.csv (already computed) and classifies each coin into:
      TIER A  "Trade it"   — forward Sharpe > 1.0
      TIER B  "Monitor"    — forward Sharpe 0.0 – 1.0
      TIER C  "Avoid"      — forward Sharpe < 0.0
    Also clusters by behavioural pattern (trending / mean-reverting / random)

  EXP 2 — 3-state HMM (BULL / CHOP / BEAR, CRASH merged into BEAR)
    Compares 4-state vs 3-state across the same coins.
    Motivation: CRASH accuracy was 10.9% (worse than random), suggesting the
    model cannot distinguish CRASH from BEAR — merging them should help.

  EXP 3 — Confidence calibration fix
    Tests two alternative confidence metrics that are more informative than
    the raw HMM posterior (which showed 99%+ confidence but only 29% accuracy):
      • Margin    = prob[best_state] – prob[2nd_state]
      • Certainty = 1 – normalised_entropy  (1 – H / log(n_states))
    A well-calibrated metric should show monotonically increasing accuracy
    as the metric rises from 0 → 1.

Usage: python tools/experiment_3state_calibration.py
       (reads data/eval_100coins.csv; no API calls for EXP 1)
       (fetches live data for EXP 2 + 3 — top 30 coins, ~10 min)
"""
import sys, os, csv, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import logging
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)

import config
from data_pipeline import _parse_klines_df
from feature_engine import compute_hmm_features

# ── Settings ──────────────────────────────────────────────────────────────────
CSV_PATH       = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "data", "eval_100coins.csv")
HMM_FEATURES   = ["log_return", "volatility", "volume_change", "rsi_norm"]
TRAIN_START    = "1 Jan, 2024"
TRAIN_CUTOFF   = "2024-07-01"
BACKTEST_CUTOFF= "2025-01-01"
INTERVAL       = "4h"
BARS_PER_YEAR  = 2190
FWD_BARS       = 12
MIN_TRAIN_BARS = 200
MIN_TEST_BARS  = 30
SLEEP_COINS    = 0.8
EXP23_TOP_N    = 30   # coins to (re-)fetch for EXP 2 + 3

# Regime constants
BULL  = config.REGIME_BULL    # 0
BEAR  = config.REGIME_BEAR    # 1
CHOP  = config.REGIME_CHOP    # 2
CRASH = config.REGIME_CRASH   # 3
RNAMES = {BULL: "BULL", BEAR: "BEAR", CHOP: "CHOP", CRASH: "CRASH"}

W = 78   # report width


# ════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1 — Coin Categorisation from CSV
# ════════════════════════════════════════════════════════════════════════════

def load_csv(path):
    """Load eval_100coins.csv and pivot to {symbol: {period: metrics}}."""
    if not os.path.exists(path):
        return None
    coins = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            sym = row["symbol"]
            per = row["period"]
            if sym not in coins:
                coins[sym] = {}
            coins[sym][per] = {
                "n":         int(row["n"]),
                "accuracy":  float(row["accuracy"]),
                "sharpe":    float(row["sharpe"]),
                "total_pnl": float(row["total_pnl"]),
                "win_rate":  float(row["win_rate"]),
                "mean_conf": float(row["mean_conf"]),
                "bull_acc":  float(row["bull_acc"])  if row["bull_acc"]  != "nan" else float("nan"),
                "bear_acc":  float(row["bear_acc"])  if row["bear_acc"]  != "nan" else float("nan"),
                "chop_acc":  float(row["chop_acc"])  if row["chop_acc"]  != "nan" else float("nan"),
                "crash_acc": float(row["crash_acc"]) if row["crash_acc"] != "nan" else float("nan"),
            }
    return coins


def categorise_coins(coins):
    """
    Assign each coin to a tier based on forward-test Sharpe ratio.

    Also classify by behavioural pattern:
      TRENDING     — model finds clear direction (BULL/BEAR accuracy both > 20%)
      MEAN-REV     — model finds CHOP well (CHOP accuracy > 40%) but direction weak
      RANDOM       — nothing works (all per-regime accuracy < 20%)
    """
    tiers = {
        "A": [],   # forward Sharpe > 1.0  → Trade it
        "B": [],   # forward Sharpe 0.0–1.0 → Monitor
        "C": [],   # forward Sharpe < 0.0  → Avoid
    }
    for sym, data in coins.items():
        fw = data.get("forward")
        bt = data.get("backtest")
        if fw is None:
            continue
        sh  = fw["sharpe"]
        acc = fw["accuracy"]
        pnl = fw["total_pnl"]

        # Behavioural pattern
        ba  = fw.get("bull_acc",  float("nan"))
        bra = fw.get("bear_acc",  float("nan"))
        ca  = fw.get("chop_acc",  float("nan"))

        if not np.isnan(ba) and not np.isnan(bra) and (ba + bra) / 2 > 0.20:
            pattern = "TRENDING"
        elif not np.isnan(ca) and ca > 0.40:
            pattern = "MEAN-REV"
        else:
            pattern = "RANDOM"

        # Stability: Sharpe direction matches across BT → FW
        bt_sh = bt["sharpe"] if bt else 0
        stable = "stable" if (sh > 0 and bt_sh > 0) or (sh < 0 and bt_sh < 0) else "switch"

        entry = (sym, sh, acc, pnl, pattern, stable,
                 bt_sh if bt else float("nan"))

        if sh >= 1.0:
            tiers["A"].append(entry)
        elif sh >= 0.0:
            tiers["B"].append(entry)
        else:
            tiers["C"].append(entry)

    for t in tiers:
        tiers[t].sort(key=lambda x: x[1], reverse=True)   # sort by forward Sharpe
    return tiers


def print_exp1(tiers):
    print(f"\n{'═'*W}")
    print("  EXPERIMENT 1 — COIN CATEGORISATION  (from eval_100coins.csv)")
    print(f"{'═'*W}")

    labels = {
        "A": "TIER A — Trade It  (forward Sharpe ≥ 1.0)",
        "B": "TIER B — Monitor   (forward Sharpe 0.0–1.0)",
        "C": "TIER C — Avoid     (forward Sharpe < 0.0)",
    }
    col = f"  {'Symbol':<14} {'FW Sharpe':>10} {'FW Acc':>8} {'FW P&L':>10} {'BT Sharpe':>10} {'Pattern':<11} {'Stable?'}"

    for tier in ("A", "B", "C"):
        entries = tiers[tier]
        print(f"\n  ── {labels[tier]}  ({len(entries)} coins) ──")
        if not entries:
            print("    (none)")
            continue
        print(col)
        print(f"  {'─'*W}")
        for sym, sh, acc, pnl, pat, stbl, bt_sh in entries:
            flag = "✓" if stbl == "stable" else "~"
            print(f"  {sym:<14} {sh:>10.2f} {acc:>7.1%} {pnl:>+9.1%} "
                  f"{bt_sh:>10.2f} {pat:<11} {flag}")

    # Summary
    n_total = sum(len(v) for v in tiers.values())
    whitelist = [e[0] for e in tiers["A"] if e[5] == "stable"]
    watchlist = [e[0] for e in tiers["B"] if e[4] == "TRENDING"]
    print(f"\n  ── Recommended live whitelist (Tier A + stable): {len(whitelist)} coins ──")
    print("  " + "  ".join(whitelist) if whitelist else "  (none)")
    print(f"\n  ── Watchlist (Tier B + TRENDING pattern): {len(watchlist)} coins ──")
    print("  " + "  ".join(watchlist) if watchlist else "  (none)")
    print(f"\n  Tier A: {len(tiers['A'])}/{n_total}  "
          f"Tier B: {len(tiers['B'])}/{n_total}  "
          f"Tier C: {len(tiers['C'])}/{n_total}")


# ════════════════════════════════════════════════════════════════════════════
#  Shared utilities for EXP 2 + 3
# ════════════════════════════════════════════════════════════════════════════

def fetch_klines(symbol):
    from binance.client import Client
    client = Client(tld="com")
    try:
        klines = client.get_historical_klines(
            symbol, Client.KLINE_INTERVAL_4HOUR, TRAIN_START
        )
        return _parse_klines_df(klines) if klines else None
    except Exception as e:
        print(f"    FETCH ERROR {symbol}: {e}")
        return None


def ground_truth(df, fwd=FWD_BARS):
    lr = np.log(df["close"] / df["close"].shift(1))
    fwd_ret = lr.rolling(fwd).sum().shift(-fwd)
    fwd_vol = lr.rolling(fwd).std().shift(-fwd)
    n = len(df)
    labels = np.full(n, CHOP, dtype=int)
    for i in range(n):
        r = fwd_ret.iloc[i]; v = fwd_vol.iloc[i]
        if pd.isna(r) or pd.isna(v):
            labels[i] = -1
        elif r < -0.06 or (r < -0.02 and v > 0.04):
            labels[i] = CRASH
        elif r < -0.02:
            labels[i] = BEAR
        elif r > 0.03 and v < 0.025:
            labels[i] = BULL
    return labels


def train_hmm(X, n_states):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = GaussianHMM(
        n_components=n_states, covariance_type="full",
        n_iter=config.HMM_ITERATIONS, random_state=42, tol=1e-4,
    )
    model.fit(Xs)
    return model, scaler


def state_map_4(model):
    """4-state: BULL > BEAR > CHOP > CRASH by log_return."""
    order = np.argsort(model.means_[:, 0])[::-1]
    return {int(r): c for r, c in zip(order, [BULL, BEAR, CHOP, CRASH])}


def state_map_3(model):
    """3-state: BULL > CHOP > BEAR by log_return (CRASH merged into BEAR)."""
    order = np.argsort(model.means_[:, 0])[::-1]
    return {int(r): c for r, c in zip(order, [BULL, CHOP, BEAR])}


def raw_posterior_confidence(model, scaler, state_map, X):
    """Standard: confidence = max posterior probability."""
    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)
    raw  = np.argmax(prob, axis=1)
    states = np.array([state_map[s] for s in raw])
    confs  = prob[np.arange(len(raw)), raw]
    return states, confs


def margin_confidence(model, scaler, state_map, X):
    """Margin = prob[best] - prob[2nd_best]. Range [0, 1]."""
    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)
    raw  = np.argmax(prob, axis=1)
    states = np.array([state_map[s] for s in raw])
    sorted_p = np.sort(prob, axis=1)[:, ::-1]
    margin   = sorted_p[:, 0] - sorted_p[:, 1]
    return states, margin


def entropy_certainty(model, scaler, state_map, X):
    """Certainty = 1 - H/log(n_states). Range [0, 1]. 1 = fully certain."""
    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)
    raw  = np.argmax(prob, axis=1)
    states = np.array([state_map[s] for s in raw])
    eps = 1e-12
    H = -np.sum(prob * np.log(prob + eps), axis=1)
    n = prob.shape[1]
    certainty = 1.0 - H / np.log(n)
    return states, certainty


def calibration_table(all_states, all_confs, all_true, n_buckets=10):
    """
    Returns list of (bucket_lo, bucket_hi, n, accuracy) for a confidence metric.
    """
    edges = np.linspace(0, 1, n_buckets + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (all_confs >= lo) & (all_confs < hi)
        if lo == edges[-2]:   # include right edge in last bucket
            mask = (all_confs >= lo) & (all_confs <= 1.0)
        known = (all_true != -1) & mask
        if known.sum() < 5:
            continue
        acc = np.mean(all_states[known] == all_true[known])
        rows.append((lo, hi, int(known.sum()), float(acc)))
    return rows


# ════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2 — 4-state vs 3-state accuracy comparison
# ════════════════════════════════════════════════════════════════════════════

def run_exp2_3(symbols):
    """Evaluate both 4-state and 3-state models, with all three confidence metrics."""

    # Accumulators per variant
    variants = {
        "4-state+raw":    {"preds": [], "confs": [], "true": [], "pnl": []},
        "3-state+raw":    {"preds": [], "confs": [], "true": [], "pnl": []},
        "4-state+margin": {"preds": [], "confs": [], "true": [], "pnl": []},
        "3-state+margin": {"preds": [], "confs": [], "true": [], "pnl": []},
        "3-state+entropy":{"preds": [], "confs": [], "true": [], "pnl": []},
    }
    per_coin = []   # for per-coin table

    print(f"\n  Processing {len(symbols)} coins...")
    for i, sym in enumerate(symbols):
        print(f"  [{i+1:>2}/{len(symbols)}] {sym:<14}", end="", flush=True)
        df_raw = fetch_klines(sym)
        if df_raw is None or len(df_raw) < MIN_TRAIN_BARS + MIN_TEST_BARS:
            print("  SKIP")
            time.sleep(SLEEP_COINS)
            continue

        df = compute_hmm_features(df_raw)
        gt = ground_truth(df)
        lr = np.log(df["close"] / df["close"].shift(1)).fillna(0).values
        ts = df["timestamp"]

        train_mask   = ts <  TRAIN_CUTOFF
        fwd_mask     = ts >= BACKTEST_CUTOFF

        train_df = df[HMM_FEATURES][train_mask].dropna()
        fwd_df   = df[HMM_FEATURES][fwd_mask].dropna()

        if len(train_df) < MIN_TRAIN_BARS or len(fwd_df) < MIN_TEST_BARS:
            print("  SKIP (data)")
            time.sleep(SLEEP_COINS)
            continue

        try:
            m4, sc4 = train_hmm(train_df.values, 4)
            m3, sc3 = train_hmm(train_df.values, 3)
        except Exception as e:
            print(f"  SKIP (train err: {e})")
            time.sleep(SLEEP_COINS)
            continue

        sm4 = state_map_4(m4)
        sm3 = state_map_3(m3)

        fwd_idx = fwd_df.index.tolist()
        gt_fwd  = gt[fwd_idx]
        next_lr = np.array([lr[j+1] if j+1 < len(lr) else 0.0 for j in fwd_idx])

        X_fwd = fwd_df.values

        # Ground truth for 3-state: merge CRASH → BEAR
        gt3 = gt_fwd.copy()
        gt3[gt3 == CRASH] = BEAR

        coin_row = {"sym": sym}

        for key, n_states, model, scaler, sm, gt_eval, conf_fn in [
            ("4-state+raw",     4, m4, sc4, sm4, gt_fwd, raw_posterior_confidence),
            ("3-state+raw",     3, m3, sc3, sm3, gt3,    raw_posterior_confidence),
            ("4-state+margin",  4, m4, sc4, sm4, gt_fwd, margin_confidence),
            ("3-state+margin",  3, m3, sc3, sm3, gt3,    margin_confidence),
            ("3-state+entropy", 3, m3, sc3, sm3, gt3,    entropy_certainty),
        ]:
            states, confs = conf_fn(model, scaler, sm, X_fwd)
            known = gt_eval != -1
            if known.sum() < MIN_TEST_BARS:
                continue

            p = states[known]; c = confs[known]; t = gt_eval[known]
            lr_k = next_lr[known]

            acc = np.mean(p == t)
            strat = np.where(p == BULL, lr_k,
                    np.where((p == BEAR) | (p == CRASH), -lr_k, 0.0))
            sh = float(strat.mean() / strat.std() * np.sqrt(BARS_PER_YEAR)
                       if strat.std() > 1e-10 else 0.0)

            variants[key]["preds"].extend(p.tolist())
            variants[key]["confs"].extend(c.tolist())
            variants[key]["true"].extend(t.tolist())
            variants[key]["pnl"].extend(strat.tolist())

            coin_row[key + "_sh"] = sh
            coin_row[key + "_acc"] = acc

        per_coin.append(coin_row)
        c4 = coin_row.get("4-state+raw_acc", float("nan"))
        c3 = coin_row.get("3-state+raw_acc", float("nan"))
        print(f"  4st={c4:.1%}  3st={c3:.1%}")
        time.sleep(SLEEP_COINS)

    return variants, per_coin


def print_exp2(variants, per_coin):
    print(f"\n{'═'*W}")
    print("  EXPERIMENT 2 — 4-STATE vs 3-STATE ACCURACY  (forward test)")
    print(f"{'═'*W}")

    print(f"\n  {'Variant':<22} {'Acc':>8} {'Sharpe':>8} {'P&L':>10} {'n':>8}")
    print(f"  {'─'*W}")
    for key, d in variants.items():
        if not d["preds"]:
            continue
        p = np.array(d["preds"]); t = np.array(d["true"])
        pnl = np.array(d["pnl"])
        known = t != -1
        acc = np.mean(p[known] == t[known])
        sh  = (pnl.mean() / pnl.std() * np.sqrt(BARS_PER_YEAR)
               if pnl.std() > 1e-10 else 0.0)
        pnl_total = pnl.sum()
        print(f"  {key:<22} {acc:>7.1%} {sh:>8.2f} {pnl_total:>+9.1%} {int(known.sum()):>8,}")

    # Per-regime breakdown for 4-state and 3-state
    print(f"\n  Per-regime accuracy (4-state vs 3-state, all coins pooled):")
    print(f"  {'Regime':<10} {'4-state':>10} {'3-state':>10}")
    print(f"  {'─'*32}")
    d4 = variants["4-state+raw"]
    d3 = variants["3-state+raw"]
    p4 = np.array(d4["preds"]); t4 = np.array(d4["true"])
    p3 = np.array(d3["preds"]); t3 = np.array(d3["true"])

    for regime, name in [(BULL,"BULL"),(BEAR,"BEAR"),(CHOP,"CHOP"),(CRASH,"CRASH (4-st only)")]:
        m4 = t4 == regime
        m3 = t3 == regime   # CRASH won't exist in 3-state gt
        a4 = np.mean(p4[m4] == t4[m4]) if m4.sum() > 5 else float("nan")
        a3 = np.mean(p3[m3] == t3[m3]) if m3.sum() > 5 else float("nan")
        a4s = f"{a4:.1%}" if not np.isnan(a4) else "N/A"
        a3s = f"{a3:.1%}" if not np.isnan(a3) else "N/A (merged)"
        print(f"  {name:<10} {a4s:>10} {a3s:>10}")


def print_exp3(variants):
    print(f"\n{'═'*W}")
    print("  EXPERIMENT 3 — CONFIDENCE CALIBRATION COMPARISON  (forward test)")
    print(f"{'═'*W}")
    print("  Does higher confidence actually mean higher accuracy?")
    print("  A good metric should be monotonically increasing.\n")

    bucket_labels = [f"{int(lo*100)}-{int(hi*100)}%"
                     for lo, hi in zip(np.linspace(0,1,11)[:-1], np.linspace(0,1,11)[1:])]

    headers = {
        "4-state+raw":     "4st raw posterior",
        "3-state+raw":     "3st raw posterior",
        "3-state+margin":  "3st margin",
        "3-state+entropy": "3st certainty",
    }
    # Print calibration for each metric
    for key, label in headers.items():
        d = variants[key]
        if not d["preds"]:
            continue
        p = np.array(d["preds"]); c = np.array(d["confs"]); t = np.array(d["true"])
        rows = calibration_table(p, c, t, n_buckets=10)
        if not rows:
            continue

        print(f"  ── {label} ──")
        print(f"  {'Conf bucket':<14} {'N':>7} {'Actual acc':>12} {'Bar'}")
        print(f"  {'─'*60}")

        accs = [r[3] for r in rows]
        # Check monotonicity
        mono_violations = sum(1 for a, b in zip(accs[:-1], accs[1:]) if b < a - 0.02)

        for lo, hi, n, acc in rows:
            bar_len = int(acc * 30)
            bar = "█" * bar_len
            print(f"  {lo:.0%}–{hi:.0%}        {n:>7,}      {acc:>8.1%}    {bar}")

        # Summary stats
        rng = max(accs) - min(accs) if accs else 0
        print(f"  Monotonicity violations: {mono_violations}  "
              f"| Accuracy range: {rng:.1%}  "
              f"| Peak bucket: {max(rows, key=lambda x: x[3])[0]:.0%}–{max(rows, key=lambda x: x[3])[1]:.0%}\n")

    # Which metric best separates high-confidence from low-confidence predictions?
    print("  ── Summary: high-confidence (top 20%) vs low-confidence (bottom 20%) accuracy gap ──")
    print(f"  {'Metric':<22} {'Low-conf acc':>14} {'High-conf acc':>14} {'Gap':>8} {'Better?'}")
    print(f"  {'─'*W}")
    for key, label in headers.items():
        d = variants[key]
        if not d["preds"]:
            continue
        p = np.array(d["preds"]); c = np.array(d["confs"]); t = np.array(d["true"])
        known = t != -1
        p=p[known]; c=c[known]; t=t[known]
        lo20 = np.percentile(c, 20); hi80 = np.percentile(c, 80)
        lo_acc = np.mean(p[c < lo20] == t[c < lo20]) if (c < lo20).sum() > 5 else float("nan")
        hi_acc = np.mean(p[c > hi80] == t[c > hi80]) if (c > hi80).sum() > 5 else float("nan")
        gap    = hi_acc - lo_acc if not (np.isnan(lo_acc) or np.isnan(hi_acc)) else float("nan")
        better = "YES ✓" if not np.isnan(gap) and gap > 0.02 else ("no" if not np.isnan(gap) else "N/A")
        la = f"{lo_acc:.1%}" if not np.isnan(lo_acc) else "N/A"
        ha = f"{hi_acc:.1%}" if not np.isnan(hi_acc) else "N/A"
        gs = f"{gap:+.1%}" if not np.isnan(gap) else "N/A"
        print(f"  {label:<22} {la:>14} {ha:>14} {gs:>8} {better}")


def print_recommendations(variants, tiers):
    print(f"\n{'═'*W}")
    print("  PRODUCTION RECOMMENDATIONS")
    print(f"{'═'*W}")

    # Best confidence metric (largest high-low gap)
    best_metric = None
    best_gap = -1.0
    for key in ("3-state+margin", "3-state+entropy", "4-state+margin"):
        d = variants.get(key)
        if not d or not d["preds"]:
            continue
        p = np.array(d["preds"]); c = np.array(d["confs"]); t = np.array(d["true"])
        known = t != -1
        p=p[known]; c=c[known]; t=t[known]
        lo20 = np.percentile(c, 20); hi80 = np.percentile(c, 80)
        if (c < lo20).sum() < 5 or (c > hi80).sum() < 5:
            continue
        lo_acc = np.mean(p[c < lo20] == t[c < lo20])
        hi_acc = np.mean(p[c > hi80] == t[c > hi80])
        gap = hi_acc - lo_acc
        if gap > best_gap:
            best_gap = gap
            best_metric = key

    # Compare 3-state vs 4-state summary
    d4 = variants.get("4-state+raw")
    d3 = variants.get("3-state+raw")
    acc4 = np.mean(np.array(d4["preds"]) == np.array(d4["true"])) if d4 and d4["preds"] else float("nan")
    acc3 = np.mean(np.array(d3["preds"]) == np.array(d3["true"])) if d3 and d3["preds"] else float("nan")

    print(f"""
  1. HMM STATE COUNT
     Current: 4 states  (accuracy {acc4:.1%})
     3-state:            accuracy {acc3:.1%}  (CRASH merged into BEAR)
     Recommendation: {"Switch to 3-state — it avoids the near-random CRASH label." if acc3 > acc4 else "Keep 4-state — no clear gain from 3-state."}
     → Change config.HMM_N_STATES = 3

  2. CONFIDENCE METRIC
     Current: raw HMM posterior (99%+ always, 29% actual accuracy = USELESS)
     Best alternative found: {best_metric or "none"} (high-low accuracy gap = {best_gap:+.1%})
     Recommendation: Replace max(predict_proba) with MARGIN (prob[best] - prob[2nd])
     → In hmm_brain.predict(): return margin instead of max_posterior
     → Recalibrate HMM_CONF_TIER_HIGH/MED etc. based on margin percentiles:
         margin > 0.60 → Full weight  (HMM_CONF_TIER_HIGH)
         margin > 0.40 → 85% weight   (HMM_CONF_TIER_MED_HIGH)
         margin > 0.25 → 65% weight   (HMM_CONF_TIER_MED)
         margin > 0.10 → 40% weight   (HMM_CONF_TIER_LOW)
         margin < 0.10 → 0 pts (uncertain — skip trade)

  3. COIN WHITELIST (Tier A — stable forward Sharpe ≥ 1.0)""")

    whitelist = [e[0] for e in tiers.get("A", []) if e[5] == "stable"]
    watchlist = [e[0] for e in tiers.get("B", []) if e[4] == "TRENDING"]
    avoidlist = [e[0] for e in tiers.get("C", [])]
    print(f"     Trade:   {', '.join(whitelist) if whitelist else '(none yet)'}")
    print(f"     Watch:   {', '.join(watchlist[:10]) if watchlist else '(none)'}")
    print(f"     Avoid:   {', '.join(avoidlist[:10]) if avoidlist else '(none)'}")
    print(f"""
  4. CONVICTION SCORE CHANGE
     Current CONVICTION_WEIGHT_HMM = 22 pts (flat, regardless of signal quality)
     After margin fix: same 22 pts max, but actually SCALED by margin tiers.
     Net effect: low-confidence HMM calls stop inflating leverage.

  5. SAVE COIN TIERS to data/coin_tiers.csv for live coin_scanner filtering
""")
    # Save coin tiers
    tiers_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "coin_tiers.csv"
    )
    try:
        os.makedirs(os.path.dirname(tiers_csv), exist_ok=True)
        with open(tiers_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "tier", "fwd_sharpe", "fwd_accuracy",
                        "fwd_pnl", "bt_sharpe", "pattern", "stable"])
            for tier, entries in tiers.items():
                for sym, sh, acc, pnl, pat, stbl, bt_sh in entries:
                    w.writerow([sym, tier, round(sh,3), round(acc,4),
                                round(pnl,4), round(bt_sh,3), pat, stbl])
        print(f"  Coin tiers saved → {tiers_csv}")
    except Exception as e:
        print(f"  Coin tiers save failed: {e}")

    print(f"\n{'═'*W}\n")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def get_top_symbols(n):
    from binance.client import Client
    client = Client(tld="com")
    EXCLUDE = {"EURUSDT","WBTCUSDT","USDCUSDT","TUSDUSDT","BUSDUSDT",
               "USTUSDT","DAIUSDT","FDUSDUSDT","CVCUSDT","USD1USDT"}
    LEV = ("UP","DOWN","BULL","BEAR")
    tickers = client.get_ticker()
    usdt = [t for t in tickers
            if t["symbol"].endswith("USDT")
            and t["symbol"] not in EXCLUDE
            and not any(k in t["symbol"].replace("USDT","") for k in LEV)]
    usdt.sort(key=lambda t: float(t.get("quoteVolume",0)), reverse=True)
    return [t["symbol"] for t in usdt[:n]]


def run(skip_exp23=False):
    print(f"\n{'═'*W}")
    print("  3-STATE CALIBRATION EXPERIMENT + COIN CATEGORISATION")
    print(f"{'═'*W}")

    # ── EXP 1 ──────────────────────────────────────────────────────────────
    print("\n  Loading eval_100coins.csv for coin categorisation...")
    coins_data = load_csv(CSV_PATH)
    tiers = {}
    if coins_data:
        tiers = categorise_coins(coins_data)
        print_exp1(tiers)
    else:
        print(f"  CSV not found at {CSV_PATH} — skipping EXP 1.")
        print("  Run tools/backtest_fronttest_100coins.py first.")

    if skip_exp23:
        return

    # ── EXP 2 + 3 ──────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  EXPERIMENTS 2 + 3 — Fetching top {EXP23_TOP_N} coins...")
    print(f"{'═'*W}")
    symbols = get_top_symbols(EXP23_TOP_N)
    variants, per_coin = run_exp2_3(symbols)

    print_exp2(variants, per_coin)
    print_exp3(variants)
    print_recommendations(variants, tiers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip EXP 2+3 (coin categorisation only, no API calls)")
    args = parser.parse_args()
    run(skip_exp23=args.no_fetch)
