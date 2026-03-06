"""
tools/backtest_historical.py
Walk-forward backtest comparing BEFORE vs AFTER HMM feature fix using real Binance data.

BEFORE: HMM_FEATURES = ["volatility", "volume_change", "rsi_norm"]   (3 features — bug)
  → _build_state_map sorts by means[:,0] = VOLATILITY → BULL/CRASH labels are wrong

AFTER:  HMM_FEATURES = ["log_return", "volatility", "volume_change", "rsi_norm"]  (4 features — fix)
  → _build_state_map sorts by means[:,0] = LOG_RETURN → labels are correct

Usage: python tools/backtest_historical.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import config
from data_pipeline import _parse_klines_df
from feature_engine import compute_hmm_features

# ── Backtest Settings ─────────────────────────────────────────────────────────
SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL   = "4h"        # 4h candles: 6/day, 1000 candles ≈ 166 days
START_DATE = "1 Jan, 2024"
TRAIN_BARS = 180         # 30 days of 4h candles per training window
TEST_BARS  = 42          # 7 days per test window
STEP_BARS  = 42          # slide by 7 days
FWD_BARS   = 12          # forward horizon for ground truth labels (12×4h = 48h)
N_STATES   = config.HMM_N_STATES

# Old 3-feature config (the bug)
BEFORE_FEATURES = ["volatility", "volume_change", "rsi_norm"]
# New 4-feature config (the fix)
AFTER_FEATURES  = ["log_return", "volatility", "volume_change", "rsi_norm"]


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_historical(symbol, interval, start_date):
    """
    Fetch klines from production Binance public API (no auth required for OHLCV).
    Uses get_historical_klines which handles pagination internally.
    """
    from binance.client import Client
    # Use production endpoint (not testnet) — public kline data, no auth needed
    client = Client(tld="com")
    interval_map = {
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
    }
    binance_interval = interval_map.get(interval, interval)
    print(f"  Fetching {symbol} {interval} from {start_date} (production Binance)...")
    try:
        klines = client.get_historical_klines(symbol, binance_interval, start_date)
        if not klines:
            print(f"  No data returned for {symbol}")
            return None
        df = _parse_klines_df(klines)
        print(f"  Got {len(df):,} candles  "
              f"({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ── Ground Truth Labels ───────────────────────────────────────────────────────

def compute_ground_truth(df, fwd_bars=FWD_BARS):
    """
    Label each candle's regime based on what actually happens over the next
    fwd_bars candles (forward-looking, post-hoc — used only for evaluation).

    Rules (for 4h candles, fwd_bars=12 → 48h horizon):
      CRASH : fwd_return < -6%  OR  (fwd_return < -2% AND fwd_vol > 4%)
      BEAR  : fwd_return < -2%
      BULL  : fwd_return > +3%  AND  fwd_vol < 2.5%
      CHOP  : everything else
    """
    lr = np.log(df["close"] / df["close"].shift(1))
    # Cumulative forward return and volatility over fwd_bars candles ahead
    fwd_ret = lr.rolling(fwd_bars).sum().shift(-fwd_bars)
    fwd_vol = lr.rolling(fwd_bars).std().shift(-fwd_bars)

    n = len(df)
    labels = np.full(n, config.REGIME_CHOP, dtype=int)
    for i in range(n):
        r = fwd_ret.iloc[i]
        v = fwd_vol.iloc[i]
        if pd.isna(r) or pd.isna(v):
            labels[i] = -1   # unknown (end of series)
            continue
        if r < -0.06 or (r < -0.02 and v > 0.04):
            labels[i] = config.REGIME_CRASH
        elif r < -0.02:
            labels[i] = config.REGIME_BEAR
        elif r > 0.03 and v < 0.025:
            labels[i] = config.REGIME_BULL
        # else CHOP (already set)
    return labels


# ── HMM Training (standalone, avoids module-level state) ─────────────────────

def train_hmm_model(X_train):
    """Fit a GaussianHMM on X_train (n_samples × n_features). Returns model, scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=config.HMM_ITERATIONS,
        random_state=42,
        tol=1e-4,
    )
    model.fit(X_scaled)
    return model, scaler


def build_state_map(model, col0_is_log_return):
    """
    Map raw HMM states → canonical regimes by sorting means[:, 0] descending.

    col0_is_log_return=True  → AFTER fix: highest log_return = BULL (correct)
    col0_is_log_return=False → BEFORE bug: highest volatility = BULL (wrong)
    """
    order = np.argsort(model.means_[:, 0])[::-1]   # descending by col 0
    # Re-label: rank 0 → BULL, 1 → BEAR, 2 → CHOP, 3 → CRASH
    canonical = [config.REGIME_BULL, config.REGIME_BEAR,
                 config.REGIME_CHOP, config.REGIME_CRASH]
    return {int(raw): canonical[i] for i, raw in enumerate(order)}


def predict_regimes(model, scaler, state_map, X_test):
    """Return (canonical_states, confidences) arrays for X_test."""
    X_scaled = scaler.transform(X_test)
    raw_states = model.predict(X_scaled)
    proba      = model.predict_proba(X_scaled)
    canonical  = np.array([state_map[s] for s in raw_states], dtype=int)
    confidence = np.array([proba[i, s] for i, s in enumerate(raw_states)])
    return canonical, confidence


# ── Walk-Forward Loop ─────────────────────────────────────────────────────────

def walk_forward(df_feat, feat_cols, col0_is_log_return,
                 gt_labels, log_returns,
                 train_bars, test_bars, step_bars):
    """
    Slide train/test window across the full series.

    Returns parallel arrays: preds, confs, true_labels, strategy_pnl, bh_pnl
    """
    n = len(df_feat)
    all_preds, all_confs, all_true, all_pnl, all_bh = [], [], [], [], []

    for start in range(0, n - train_bars - test_bars, step_bars):
        train_end = start + train_bars
        test_end  = min(train_end + test_bars, n - 1)

        # Extract training & test feature matrices (drop NaN rows)
        train_df = df_feat[feat_cols].iloc[start:train_end].dropna()
        test_df  = df_feat[feat_cols].iloc[train_end:test_end].dropna()

        if len(train_df) < 100 or len(test_df) < 5:
            continue

        try:
            model, scaler = train_hmm_model(train_df.values)
        except Exception:
            continue

        state_map = build_state_map(model, col0_is_log_return)
        preds, confs = predict_regimes(model, scaler, state_map, test_df.values)

        # Align ground truth and next-bar returns to the test indices
        test_pos = test_df.index.tolist()   # integer positions in df_feat
        true = gt_labels[test_pos]

        # P&L: enter at candle i close, exit at candle i+1 close
        strategy_pnl = []
        bh_pnl_list  = []
        for j, pos in enumerate(test_pos):
            next_pos = pos + 1
            if next_pos >= n:
                strategy_pnl.append(0.0)
                bh_pnl_list.append(0.0)
                continue
            next_ret = log_returns[next_pos]
            if preds[j] == config.REGIME_BULL:
                strategy_pnl.append(next_ret)        # long
            elif preds[j] in (config.REGIME_BEAR, config.REGIME_CRASH):
                strategy_pnl.append(-next_ret)       # short
            else:
                strategy_pnl.append(0.0)             # flat in CHOP
            bh_pnl_list.append(next_ret)

        # Skip windows where ground truth is mostly unknown (-1)
        known_mask = true != -1
        if known_mask.sum() < 5:
            continue

        all_preds.extend(preds[known_mask].tolist())
        all_confs.extend(confs[known_mask].tolist())
        all_true.extend(true[known_mask].tolist())
        all_pnl.extend(np.array(strategy_pnl)[known_mask].tolist())
        all_bh.extend(np.array(bh_pnl_list)[known_mask].tolist())

    return (np.array(all_preds), np.array(all_confs),
            np.array(all_true), np.array(all_pnl), np.array(all_bh))


# ── Metrics ───────────────────────────────────────────────────────────────────

REGIME_NAMES = {
    config.REGIME_BULL:  "BULL",
    config.REGIME_BEAR:  "BEAR",
    config.REGIME_CHOP:  "CHOP",
    config.REGIME_CRASH: "CRASH",
}

def compute_metrics(preds, confs, true, pnl, bh_pnl):
    """Return dict of evaluation metrics."""
    acc      = np.mean(preds == true) * 100
    mean_conf = np.mean(confs) * 100
    total_ret = np.sum(pnl) * 100       # log-return → approx % (small returns)
    bh_ret    = np.sum(bh_pnl) * 100

    # Annualised Sharpe (4h bars → 2190/year)
    bars_per_year = 2190
    sharpe    = (pnl.mean() / pnl.std() * np.sqrt(bars_per_year)
                 if pnl.std() > 1e-10 else 0.0)
    sharpe_bh = (bh_pnl.mean() / bh_pnl.std() * np.sqrt(bars_per_year)
                 if bh_pnl.std() > 1e-10 else 0.0)

    # Per-regime accuracy
    per_acc = {}
    for r, name in REGIME_NAMES.items():
        mask = true == r
        per_acc[name] = np.mean(preds[mask] == r) * 100 if mask.sum() > 0 else float("nan")

    # Regime distribution in predictions
    pred_dist = {name: np.mean(preds == r) * 100 for r, name in REGIME_NAMES.items()}

    # Avg return in BULL-predicted bars vs CRASH-predicted bars
    bull_bars  = pnl[preds == config.REGIME_BULL]
    crash_bars = pnl[preds == config.REGIME_CRASH]
    bull_avg   = bull_bars.mean() * 100 if len(bull_bars) > 0 else float("nan")
    crash_avg  = crash_bars.mean() * 100 if len(crash_bars) > 0 else float("nan")

    return {
        "accuracy": acc, "mean_conf": mean_conf,
        "total_ret": total_ret, "bh_ret": bh_ret,
        "sharpe": sharpe, "sharpe_bh": sharpe_bh,
        "per_acc": per_acc, "pred_dist": pred_dist,
        "bull_avg_ret": bull_avg, "crash_avg_ret": crash_avg,
        "n_samples": len(preds),
    }


def print_comparison(symbol, bm, am):
    """Pretty-print side-by-side comparison for one symbol."""
    W = 70
    print(f"\n  Symbol: {symbol}  ({bm['n_samples']} test samples)")
    print(f"  {'─'*W}")
    print(f"  {'Metric':<30} {'BEFORE (3-feat)':>16} {'AFTER (4-feat)':>16} {'Δ':>8}")
    print(f"  {'─'*W}")

    rows = [
        ("Label Accuracy",      "accuracy",  "%",  ".1f"),
        ("Mean Confidence",     "mean_conf", "%",  ".1f"),
        ("Strategy P&L",        "total_ret", "%",  ".1f"),
        ("Buy-and-Hold P&L",    "bh_ret",    "%",  ".1f"),
        ("Sharpe (strategy)",   "sharpe",    "",   ".2f"),
        ("Sharpe (B&H)",        "sharpe_bh", "",   ".2f"),
        ("Avg ret in BULL bars","bull_avg_ret","%",".2f"),
        ("Avg ret in CRASH bars","crash_avg_ret","%",".2f"),
    ]
    for label, key, sfx, fmt in rows:
        bv = bm[key]
        av = am[key]
        if isinstance(bv, float) and np.isnan(bv):
            bv_s, av_s, delta_s = "  N/A", "  N/A", "  N/A"
        else:
            delta = av - bv
            bv_s    = f"{bv:{fmt}}{sfx}"
            av_s    = f"{av:{fmt}}{sfx}"
            delta_s = f"{delta:+.1f}{sfx}"
        print(f"  {label:<30} {bv_s:>16} {av_s:>16} {delta_s:>8}")

    print(f"\n  Per-regime label accuracy:")
    print(f"  {'Regime':<30} {'BEFORE':>16} {'AFTER':>16} {'Δ':>8}")
    print(f"  {'─'*W}")
    for name in ["BULL", "BEAR", "CHOP", "CRASH"]:
        bv = bm["per_acc"].get(name, float("nan"))
        av = am["per_acc"].get(name, float("nan"))
        if np.isnan(bv) or np.isnan(av):
            bv_s, av_s, delta_s = "  N/A", "  N/A", "  N/A"
        else:
            bv_s    = f"{bv:.1f}%"
            av_s    = f"{av:.1f}%"
            delta_s = f"{av-bv:+.1f}pp"
        print(f"  {name:<30} {bv_s:>16} {av_s:>16} {delta_s:>8}")

    print(f"\n  Predicted regime distribution:")
    for name in ["BULL", "BEAR", "CHOP", "CRASH"]:
        print(f"    BEFORE {name}: {bm['pred_dist'].get(name,0):.1f}%   "
              f"AFTER {name}: {am['pred_dist'].get(name,0):.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*72)
    print("  HISTORICAL BACKTEST: HMM BEFORE vs AFTER FEATURE FIX")
    print(f"  Symbols : {', '.join(SYMBOLS)}")
    print(f"  Interval: {INTERVAL}   Start: {START_DATE}")
    print(f"  Train: {TRAIN_BARS} bars | Test: {TEST_BARS} bars | Step: {STEP_BARS} bars")
    print("="*72)

    summary_before = {k: [] for k in ("accuracy","mean_conf","total_ret","sharpe")}
    summary_after  = {k: [] for k in ("accuracy","mean_conf","total_ret","sharpe")}

    for symbol in SYMBOLS:
        print(f"\n{'─'*72}")
        df_raw = fetch_historical(symbol, INTERVAL, START_DATE)
        if df_raw is None or len(df_raw) < TRAIN_BARS + TEST_BARS * 3:
            print("  Skipping — insufficient data.")
            continue

        df_feat = compute_hmm_features(df_raw)
        gt_labels  = compute_ground_truth(df_feat)
        log_returns = np.log(df_feat["close"] / df_feat["close"].shift(1)).fillna(0).values

        print(f"  Ground truth distribution: "
              f"BULL={np.mean(gt_labels==0)*100:.1f}%  "
              f"BEAR={np.mean(gt_labels==1)*100:.1f}%  "
              f"CHOP={np.mean(gt_labels==2)*100:.1f}%  "
              f"CRASH={np.mean(gt_labels==3)*100:.1f}%  "
              f"Unknown={np.mean(gt_labels==-1)*100:.1f}%")

        print(f"  Walk-forward BEFORE (3-feature)...")
        b_preds, b_confs, b_true, b_pnl, b_bh = walk_forward(
            df_feat, BEFORE_FEATURES, col0_is_log_return=False,
            gt_labels=gt_labels, log_returns=log_returns,
            train_bars=TRAIN_BARS, test_bars=TEST_BARS, step_bars=STEP_BARS,
        )
        print(f"  Walk-forward AFTER  (4-feature)...")
        a_preds, a_confs, a_true, a_pnl, a_bh = walk_forward(
            df_feat, AFTER_FEATURES, col0_is_log_return=True,
            gt_labels=gt_labels, log_returns=log_returns,
            train_bars=TRAIN_BARS, test_bars=TEST_BARS, step_bars=STEP_BARS,
        )

        if len(b_preds) == 0 or len(a_preds) == 0:
            print("  No usable walk-forward windows — skipping.")
            continue

        bm = compute_metrics(b_preds, b_confs, b_true, b_pnl, b_bh)
        am = compute_metrics(a_preds, a_confs, a_true, a_pnl, a_bh)

        print_comparison(symbol, bm, am)

        for k in summary_before:
            summary_before[k].append(bm[k])
            summary_after[k].append(am[k])

    # ── Cross-Symbol Summary ───────────────────────────────────────────────────
    if any(len(v) > 0 for v in summary_before.values()):
        print(f"\n{'='*72}")
        print("  CROSS-SYMBOL AVERAGE")
        print(f"  {'Metric':<30} {'BEFORE':>16} {'AFTER':>16} {'Δ':>8}")
        print(f"  {'─'*72}")
        labels_map = [
            ("Accuracy",       "accuracy",  "%",  ".1f"),
            ("Confidence",     "mean_conf", "%",  ".1f"),
            ("Strategy P&L",   "total_ret", "%",  ".1f"),
            ("Sharpe",         "sharpe",    "",   ".2f"),
        ]
        for label, key, sfx, fmt in labels_map:
            bv = np.mean(summary_before[key]) if summary_before[key] else float("nan")
            av = np.mean(summary_after[key])  if summary_after[key]  else float("nan")
            print(f"  {label:<30} {bv:{fmt}}{sfx:>15} {av:{fmt}}{sfx:>15} "
                  f"{av-bv:+.1f}{sfx:>8}")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    run()
